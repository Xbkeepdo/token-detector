"""Torch MLP probes and metrics for the unified yes/no QA feature records."""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.dgst_t import COST_VARIANT_RISK_KEYS


QA_POSITIONS = ("prompt_last_token", "question_object_pre_token")
QA_LABEL_PROTOCOLS = (
    "answer_correctness_all",
    "object_hallucination_yes_only",
)
QA_DGST_METHODS = (
    "hpre_raw_logit_gauss",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "hpre_softmax_prob_direct",
    "raw_attention",
)
QA_DGST_COMPONENTS = (
    "risk",
    "target_cosine",
    "ev_target_dist_mass_x_cosine",
)


class QAProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


def default_feature_sets(
    dataset: str,
    positions: Sequence[str] | None = None,
) -> list[str]:
    """Return the current QA comparison matrix for each prediction position."""

    del dataset  # Both prepared QA datasets use the same detector feature families.
    selected_positions = normalize_positions(positions)
    names: list[str] = []
    for position in selected_positions:
        names.extend(
            f"{block}@{position}" for block in ("ads", "cgc", "ads+cgc")
        )
        for method in QA_DGST_METHODS:
            risk = f"{method}_risk"
            cosine = f"{method}_target_cosine"
            ev = f"{method}_ev_target_dist_mass_x_cosine"
            names.append(
                f"{risk}+{cosine}+{ev}@{position}"
            )
    return names


def normalize_positions(positions: Sequence[str] | None) -> tuple[str, ...]:
    values = (
        ("prompt_last_token",)
        if positions is None
        else tuple(str(value) for value in positions)
    )
    unknown = sorted(set(values) - set(QA_POSITIONS))
    if unknown:
        raise ValueError(f"Unknown QA prediction positions: {unknown}")
    return tuple(dict.fromkeys(values))


def legacy_feature_sets(dataset: str) -> list[str]:
    names = ["ads", "cgc", "ads+cgc", "token_uncertainty", "svar", "legacy_all"]
    targets = ("answer", "object") if dataset == "pope" else ("answer",)
    for target in targets:
        names.append(f"hprecosine@{target}")
        for risk in COST_VARIANT_RISK_KEYS:
            names.append(f"{risk}@{target}")
            names.append(f"{risk}+hprecosine@{target}")
    return names


def feature_vector(row: dict, feature_set: str) -> np.ndarray:
    position = feature_set_position(feature_set)
    if position in QA_POSITIONS:
        block = feature_set.rsplit("@", 1)[0]
        return _position_feature_vector(row, position, block)
    if feature_set == "ads":
        return _concat(row["ads_score"], row["ads_per_layer"])
    if feature_set == "cgc":
        return _concat(row["answer_cgc_score"], row["answer_cgc_per_layer"])
    if feature_set == "ads+cgc":
        return np.concatenate((feature_vector(row, "ads"), feature_vector(row, "cgc")))
    if feature_set == "token_uncertainty":
        return _concat(row["token_log_probability"], row["token_entropy"], row["token_nll"])
    if feature_set == "svar":
        return _concat(row["svar_score"])
    if feature_set == "legacy_all":
        return np.concatenate((
            feature_vector(row, "ads+cgc"),
            feature_vector(row, "token_uncertainty"),
            feature_vector(row, "svar"),
            _concat(row["attention_per_head_mid"]),
        ))
    if feature_set.startswith("best_dgst_legacy:"):
        selected = feature_set.split(":", 1)[1]
        return np.concatenate((feature_vector(row, selected), feature_vector(row, "legacy_all")))

    block, target = _parse_target_feature(feature_set)
    target_data = row["targets"][target]
    if block == "hprecosine":
        return _concat(target_data["hprecosine"])
    if block.endswith("+hprecosine"):
        risk = block[: -len("+hprecosine")]
        return np.concatenate((_concat(target_data[risk]), _concat(target_data["hprecosine"])))
    return _concat(target_data[block])


def feature_set_position(feature_set: str) -> str:
    """Return the explicit position namespace without aliasing legacy @object."""

    if "@" not in feature_set:
        return "shared"
    suffix = feature_set.rsplit("@", 1)[1]
    if suffix in QA_POSITIONS:
        return suffix
    if suffix in ("answer", "object"):
        return f"legacy_{suffix}_target"
    return "shared"


def label_for_protocol(row: Mapping[str, object], label_protocol: str) -> int | None:
    if label_protocol not in QA_LABEL_PROTOCOLS:
        raise ValueError(f"Unknown QA label protocol: {label_protocol}")
    key = (
        "label"
        if label_protocol == "answer_correctness_all"
        else "object_hallucination_yes_only_label"
    )
    if key not in row:
        raise KeyError(
            f"QA row {row.get('key')!r} is missing label field {key!r}"
        )
    value = row[key]
    if value is None:
        if label_protocol == "object_hallucination_yes_only":
            return None
        raise ValueError(f"{key} cannot be None for {label_protocol}")
    if isinstance(value, (bool, np.bool_)) or int(value) not in (0, 1):
        raise ValueError(f"{key} must be 0, 1, or None; got {value!r}")
    if (
        label_protocol == "object_hallucination_yes_only"
        and row.get("prediction") is not None
        and str(row.get("prediction")).lower() != "yes"
    ):
        raise ValueError(
            "object_hallucination_yes_only_label is set for a non-Yes prediction"
        )
    return int(value)


def _position_feature_vector(row: dict, position: str, block: str) -> np.ndarray:
    positions = row.get("positions")
    if not isinstance(positions, Mapping) or position not in positions:
        raise KeyError(f"Missing QA position {position!r}")
    position_data = positions[position]
    if not isinstance(position_data, Mapping):
        raise TypeError(f"positions.{position} must be a mapping")
    vectors = [
        _position_single_vector(position_data, part)
        for part in block.split("+")
    ]
    return np.concatenate(vectors) if vectors else np.empty(0, np.float32)


def _position_single_vector(position_data: Mapping[str, object], block: str) -> np.ndarray:
    if block == "ads":
        return _concat(position_data["ads_score"], position_data["ads_per_layer"])
    if block == "cgc":
        return _concat(position_data["cgc_score"], position_data["cgc_per_layer"])
    dgst = position_data.get("dgst")
    if not isinstance(dgst, Mapping):
        raise KeyError("Missing position DGST payload")
    method, component = _parse_current_dgst_block(block)
    if block in dgst:
        return _concat(dgst[block])
    method_payload = dgst.get(method)
    if isinstance(method_payload, Mapping):
        candidates = [component]
        if component == "ev_target_dist_mass_x_cosine":
            candidates.append("ev")
        for candidate in candidates:
            if candidate in method_payload:
                return _concat(method_payload[candidate])
    state = "hmid" if method.startswith("hmid_") else "hpre"
    raw_candidate = (
        f"dgst_t_{method}_risk_sqrt_{state}_per_layer"
        if component == "risk"
        else (
            f"dgst_t_{method}_target_cosine_topk32_{state}_per_layer"
            if component == "target_cosine"
            else f"dgst_t_{method}_ev_target_dist_mass_x_cosine_topk32_{state}_per_layer"
        )
    )
    if raw_candidate in dgst:
        return _concat(dgst[raw_candidate])
    raise KeyError(f"Missing DGST component {method}.{component}")


def _parse_current_dgst_block(block: str) -> tuple[str, str]:
    for method in sorted(QA_DGST_METHODS, key=len, reverse=True):
        prefix = f"{method}_"
        if not block.startswith(prefix):
            continue
        component = block[len(prefix):]
        if component in QA_DGST_COMPONENTS:
            return method, component
    raise ValueError(
        f"Unknown current QA DGST feature block {block!r}; expected one of "
        f"{list(QA_DGST_METHODS)} x {list(QA_DGST_COMPONENTS)}"
    )


def build_matrix(
    rows: Sequence[dict],
    feature_set: str,
    label_protocol: str = "answer_correctness_all",
):
    vectors, labels, kept = [], [], []
    for row in rows:
        label = label_for_protocol(row, label_protocol)
        if label is None:
            continue
        try:
            vector = feature_vector(row, feature_set)
        except (KeyError, TypeError):
            continue
        if vector.size == 0 or not np.isfinite(vector).all():
            continue
        vectors.append(vector.astype(np.float32))
        labels.append(label)
        kept.append(row)
    if not vectors:
        return np.empty((0, 0), np.float32), np.empty(0, np.int64), []
    widths = {vector.shape[0] for vector in vectors}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent feature widths for {feature_set}: {sorted(widths)}")
    return np.stack(vectors), np.asarray(labels, np.int64), kept


def validate_image_level_splits(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Reject rows whose physical image appears in multiple probe splits.

    POPE strategies share one COCO image namespace, so ``source_split`` is
    excluded from their identity. CLEVR train/val indices are separate image
    namespaces, so their official source split remains part of the identity.
    Legacy synthetic rows without image metadata remain supported.
    """

    valid_splits = ("train", "val", "test")
    image_owner: dict[str, str] = {}
    key_owner: dict[str, str] = {}
    split_images = {split: set() for split in valid_splits}
    split_counts = {split: 0 for split in valid_splits}
    for row in rows:
        split = str(row.get("probe_split") or "").strip()
        if split not in split_counts:
            raise ValueError(
                f"QA row {row.get('key')!r} has invalid probe_split {split!r}"
            )
        split_counts[split] += 1
        key = str(row.get("key") or "").strip()
        if key:
            if key in key_owner:
                raise ValueError(f"Duplicate QA feature row key: {key}")
            key_owner[key] = split
        identity = _qa_image_identity(row)
        if identity is None:
            continue
        previous_split = image_owner.get(identity)
        if previous_split is not None and previous_split != split:
            raise ValueError(
                "QA image leakage across probe splits: "
                f"{identity} occurs in {previous_split} and {split}"
            )
        image_owner[identity] = split
        split_images[split].add(identity)
    if split_counts["val"] != 0:
        raise ValueError("Strict QA 8:2 requires an empty validation split")
    if split_counts["train"] <= 0 or split_counts["test"] <= 0:
        raise ValueError("Strict QA 8:2 requires non-empty train and test splits")
    train_images = len(split_images["train"])
    test_images = len(split_images["test"])
    total_images = train_images + test_images
    if total_images:
        expected_train = int(total_images * 0.8)
        actual_train = train_images
        unit = "physical images"
    else:
        total_rows = split_counts["train"] + split_counts["test"]
        expected_train = int(total_rows * 0.8)
        actual_train = split_counts["train"]
        unit = "question rows"
    if actual_train != expected_train:
        raise ValueError(
            f"QA {unit} are not strict 80/20: train={actual_train}, "
            f"expected_train={expected_train}"
        )
    return {
        split: len(split_images[split]) if split_images[split] else split_counts[split]
        for split in valid_splits
    }


def _qa_image_identity(row: Mapping[str, object]) -> str | None:
    explicit = str(row.get("qa_image_identity") or "").strip()
    if explicit:
        return explicit
    dataset = str(row.get("dataset") or "").strip()
    image_id = row.get("image_id")
    if not dataset or image_id is None:
        return None
    image_text = str(image_id).strip()
    if not image_text:
        return None
    if dataset in {"pope", "amber_discriminative"}:
        return f"{dataset}::{image_text}"
    source_split = str(row.get("source_split") or "").strip()
    if not source_split:
        raise ValueError(
            f"QA row {row.get('key')!r} is missing source_split for {dataset}"
        )
    return f"{dataset}::{source_split}::{image_text}"


def train_one_seed(
    rows: Sequence[dict],
    feature_set: str,
    seed: int,
    output_dir: str,
    cfg: dict,
    device: str | None = None,
    label_protocol: str = "answer_correctness_all",
) -> dict:
    image_counts = validate_image_level_splits(rows)
    split_rows = {split: [row for row in rows if row["probe_split"] == split] for split in ("train", "val", "test")}
    X_train, y_train, train_rows = build_matrix(
        split_rows["train"], feature_set, label_protocol
    )
    X_val, y_val, val_rows = build_matrix([], feature_set, label_protocol)
    X_test, y_test, test_rows = build_matrix(
        split_rows["test"], feature_set, label_protocol
    )
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        if len(X) == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                f"{feature_set} protocol={label_protocol} seed={seed}: {split} "
                "must contain both classes after protocol filtering, got "
                f"{np.bincount(y, minlength=2).tolist()}"
            )

    _seed_everything(seed)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QAProbe(
        X_train.shape[1],
        tuple(cfg.get("hidden_sizes", [128, 64, 32])),
        float(cfg.get("dropout", 0.3)),
    ).to(chosen_device)
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32, device=chosen_device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 4)),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
    )
    history = []
    maximum_epochs = int(cfg.get("num_epochs", cfg.get("epochs", 100)))
    for epoch in range(maximum_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(chosen_device), batch_y.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses))
        scheduler.step(train_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "monitor": "train_loss",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    best_epoch = maximum_epochs
    train_probability = _predict(model, X_train, chosen_device)
    threshold, train_f1 = choose_real_f1_threshold(
        y_train,
        train_probability,
    )
    test_probability = _predict(model, X_test, chosen_device)

    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    _atomic_torch_save(checkpoint_path, {
        "state_dict": best_state,
        "feature_set": feature_set,
        "seed": seed,
        "label_protocol": label_protocol,
        "position": feature_set_position(feature_set),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "input_dim": int(X_train.shape[1]),
    })
    result = {
        "feature_set": feature_set,
        "seed": seed,
        "label_protocol": label_protocol,
        "label_field": (
            "label"
            if label_protocol == "answer_correctness_all"
            else "object_hallucination_yes_only_label"
        ),
        "position": feature_set_position(feature_set),
        "positive_class": "real",
        "input_counts": {split: len(split_rows[split]) for split in split_rows},
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "image_counts": image_counts,
        "class_counts": {
            "train": np.bincount(y_train, minlength=2).tolist(),
            "val": np.bincount(y_val, minlength=2).tolist(),
            "test": np.bincount(y_test, minlength=2).tolist(),
        },
        "input_dim": int(X_train.shape[1]),
        "pos_weight": float(pos_weight.item()),
        "best_val_loss": None,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "threshold": threshold,
        "val_metrics": None,
        "train_threshold_f1": train_f1,
        "train_metrics": classification_metrics(
            y_train,
            train_probability,
            threshold,
        ),
        "test_metrics": classification_metrics(y_test, test_probability, threshold),
        "test_groups": grouped_metrics(test_rows, y_test, test_probability, threshold),
        "history": history,
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "last_epoch",
        "threshold_selection": "train_f1",
    }
    _atomic_json(checkpoint_dir / "result.json", result)
    return result


def choose_macro_f1_threshold(y_true, probability) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    best = (-1.0, 0.5)
    for threshold in candidates:
        score = f1_score(y_true, probability >= threshold, average="macro", zero_division=0)
        if score > best[0] or (score == best[0] and abs(threshold - 0.5) < abs(best[1] - 0.5)):
            best = (float(score), float(threshold))
    return best[1], best[0]


def choose_real_f1_threshold(y_true, probability) -> tuple[float, float]:
    """Choose the Real-positive F1 threshold using training predictions."""

    targets = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    if targets.size == 0 or targets.size != scores.size:
        raise ValueError("Threshold selection requires equal non-empty arrays")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_key = (-np.inf, -np.inf, -np.inf)
    best_threshold = 0.5
    for threshold in candidates:
        prediction = (scores >= threshold).astype(np.int64)
        key = (
            float(f1_score(targets, prediction, pos_label=1, zero_division=0)),
            float(accuracy_score(targets, prediction)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, float(best_key[0])


def classification_metrics(y_true, probability, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "auroc": _safe_binary_metric(roc_auc_score, y_true, probability),
        "real_aupr": _safe_binary_metric(average_precision_score, y_true, probability),
        "hallucination_aupr": _safe_binary_metric(average_precision_score, 1 - y_true, 1 - probability),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predicted))
            if len(np.unique(y_true)) == 2 else None
        ),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "hallucination": _class_metrics(precision, recall, f1, support, 0),
        "real": _class_metrics(precision, recall, f1, support, 1),
    }


def grouped_metrics(rows, y_true, probability, threshold) -> dict:
    if not rows:
        return {}
    definitions = {
        "source_split": lambda row: row.get("source_split"),
        "question_family_index": lambda row: row.get("question_family_index"),
        "gt_answer": lambda row: row.get("report_gt_answer"),
        "error_type": lambda row: row.get("error_type"),
    }
    result = {}
    for name, getter in definitions.items():
        groups = {}
        values = sorted({str(getter(row)) for row in rows if getter(row) is not None})
        for value in values:
            indices = [index for index, row in enumerate(rows) if str(getter(row)) == value]
            groups[value] = classification_metrics(
                np.asarray(y_true)[indices], np.asarray(probability)[indices], threshold
            )
        if groups:
            result[name] = groups
    return result


def aggregate_seed_results(results: Sequence[dict]) -> dict:
    if not results:
        raise ValueError("Cannot aggregate an empty QA seed result list")
    paths = (
        "auroc", "real_aupr", "hallucination_aupr", "accuracy",
        "balanced_accuracy", "macro_f1",
        "hallucination.precision", "hallucination.recall", "hallucination.f1",
        "real.precision", "real.recall", "real.f1",
    )
    metadata_fields = (
        "feature_set",
        "label_protocol",
        "position",
        "positive_class",
        "split_protocol",
        "checkpoint_selection",
        "threshold_selection",
    )
    for field in metadata_fields:
        values = {str(result.get(field)) for result in results}
        if len(values) != 1:
            raise ValueError(f"Cannot aggregate mixed {field} values: {sorted(values)}")
    summary = {
        **{field: results[0].get(field) for field in metadata_fields},
        "seeds": [int(result["seed"]) for result in results],
        "counts": results[0].get("counts"),
        "class_counts": results[0].get("class_counts"),
        "image_counts": results[0].get("image_counts"),
    }
    for path in paths:
        values = [_nested(result["test_metrics"], path) for result in results]
        summary[path] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    return summary


def _parse_target_feature(feature_set: str) -> tuple[str, str]:
    if "@" not in feature_set:
        raise ValueError(f"Target feature must end in @answer or @object: {feature_set}")
    block, target = feature_set.rsplit("@", 1)
    if target not in ("answer", "object"):
        raise ValueError(target)
    return block, target


def _concat(*values) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    return np.concatenate(arrays) if arrays else np.empty(0, np.float32)


def _predict(model, features, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _safe_binary_metric(function, y_true, score):
    return float(function(y_true, score)) if len(np.unique(y_true)) == 2 else None


def _class_metrics(precision, recall, f1, support, index):
    return {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "support": int(support[index]),
    }


def _nested(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return float(value)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "__", value)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
