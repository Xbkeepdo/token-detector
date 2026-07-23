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

from features.baseline.svar import svar_training_vector
from features.dgst_t import COST_VARIANT_RISK_KEYS


QA_POSITIONS = ("prompt_last_token", "question_object_pre_token")
QA_LABEL_PROTOCOLS = (
    "answer_correctness_all",
    "object_hallucination_yes_only",
)
QA_DGST_BASE_METHODS = (
    "hpre_raw_logit_gauss",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "hpre_softmax_prob_direct",
    "raw_attention",
)
QA_DGST_METHODS = (
    *QA_DGST_BASE_METHODS,
    *(f"vp_{method}" for method in QA_DGST_BASE_METHODS),
)
QA_DGST_COMPONENTS = (
    "risk",
    "risk_sqrt_matched_state",
    "risk_cosine_matched_state",
    "risk_geo_stateupd_lu1",
    "target_cosine",
    "ev_target_dist_mass_x_cosine",
)


class QAProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_sizes:
            layers.extend((
                nn.Linear(previous, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)


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


def feature_vector(
    row: dict,
    feature_set: str,
    *,
    svar_layer_start: int = 5,
    svar_layer_end: int = 19,
) -> np.ndarray:
    if feature_set.startswith("baseline:"):
        return baseline_probe_vector(
            row,
            feature_set.split(":", 1)[1],
            svar_layer_start=svar_layer_start,
            svar_layer_end=svar_layer_end,
        )
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


def baseline_probe_vector(
    row: Mapping[str, object],
    method: str,
    *,
    svar_layer_start: int = 5,
    svar_layer_end: int = 19,
) -> np.ndarray:
    """Return one baseline's dense input for the shared QA Torch MLP.

    MetaToken and SVAR already serialize their canonical paper feature vector.
    ProjectAway is originally training-free, so its shared-MLP adaptation uses
    the per-layer internal-confidence curve plus its global maximum.  The
    complementary hallucination score is deliberately omitted because it is
    exactly ``1 - internal_confidence`` and adds no information.
    """

    normalized = str(method).strip().lower()
    baselines = row.get("baselines")
    if not isinstance(baselines, Mapping):
        raise KeyError("Missing baseline payload mapping")
    payload = baselines.get(normalized)
    if not isinstance(payload, Mapping):
        raise KeyError(f"Missing baseline payload {normalized!r}")
    if normalized == "svar":
        return svar_training_vector(
            payload,
            layer_start=svar_layer_start,
            layer_end=svar_layer_end,
        )
    if normalized == "metatoken":
        vector = payload.get("vector")
    elif normalized == "projectaway":
        per_layer = payload.get("per_layer_internal_confidence")
        confidence = payload.get("internal_confidence")
        if per_layer is None or confidence is None:
            raise KeyError(
                "ProjectAway shared MLP requires internal_confidence and "
                "per_layer_internal_confidence"
            )
        vector = _concat(confidence, per_layer)
    else:
        raise ValueError(
            "The shared QA MLP currently supports dense MetaToken, SVAR, and "
            f"ProjectAway features, got {method!r}"
        )
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(
            f"Baseline {normalized!r} has an empty or non-finite MLP vector"
        )
    return result


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
    # QA baseline artifacts are protocol-specific directories.  Their generic
    # ``label`` field is therefore authoritative only when the saved protocol
    # explicitly matches the requested one.
    if str(row.get("qa_label_protocol") or "") == label_protocol:
        key = "label"
    else:
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
    if block in dgst:
        return _concat(dgst[block])
    method, component = _parse_current_dgst_block(block)
    method_payload = dgst.get(method)
    if isinstance(method_payload, Mapping):
        candidates = [component]
        if component == "ev_target_dist_mass_x_cosine":
            candidates.append("ev")
        for candidate in candidates:
            if candidate in method_payload:
                return _concat(method_payload[candidate])
    state = "hmid" if method.startswith("hmid_") else "hpre"
    metadata = dgst.get("metadata")
    target_region_top_k = (
        metadata.get("dgst_t_target_region_top_k", 32)
        if isinstance(metadata, Mapping)
        else dgst.get("dgst_t_target_region_top_k", 32)
    )
    topk_slug = f"topk{int(target_region_top_k)}"
    raw_candidate = (
        f"dgst_t_{method}_risk_sqrt_{state}_per_layer"
        if component == "risk"
        else (
            f"dgst_t_{method}_target_cosine_{topk_slug}_{state}_per_layer"
            if component == "target_cosine"
            else f"dgst_t_{method}_ev_target_dist_mass_x_cosine_{topk_slug}_{state}_per_layer"
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
        if component in QA_DGST_COMPONENTS or re.fullmatch(
            r"risk_sqrt_stateupd_alpha0[1-9]", component
        ):
            return method, component
    raise ValueError(
        f"Unknown current QA DGST feature block {block!r}; expected one of "
        f"{list(QA_DGST_METHODS)} x {list(QA_DGST_COMPONENTS)}"
    )


def build_matrix(
    rows: Sequence[dict],
    feature_set: str,
    label_protocol: str = "answer_correctness_all",
    *,
    svar_layer_start: int = 5,
    svar_layer_end: int = 19,
):
    vectors, labels, kept = [], [], []
    for row in rows:
        label = label_for_protocol(row, label_protocol)
        if label is None:
            continue
        try:
            vector = feature_vector(
                row,
                feature_set,
                svar_layer_start=svar_layer_start,
                svar_layer_end=svar_layer_end,
            )
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


def _validate_default_probe_config(cfg: Mapping[str, object]) -> None:
    """Reject silent drift from the configured default MLP protocol."""

    expected = {
        "structure": "Linear-BatchNorm-ReLU-Dropout",
        "activation": "relu",
        "batch_norm": True,
        "initialization": "kaiming_uniform_relu",
        "output_dim": 1,
        "optimizer": "adam",
        "loss": "bce_with_logits",
        "scheduler_monitor": "train_loss",
        "early_stopping_monitor": "train_loss",
        "checkpoint_selection": "minimum_train_loss",
        "feature_normalization": "none",
    }
    mismatches = {
        key: {"expected": value, "found": cfg.get(key)}
        for key, value in expected.items()
        if key in cfg and cfg.get(key) != value
    }
    reporting = tuple(
        str(value)
        for value in cfg.get(
            "threshold_reporting", ("fixed_0.5", "train_f1")
        )
    )
    if reporting != ("fixed_0.5", "train_f1"):
        mismatches["threshold_reporting"] = {
            "expected": ["fixed_0.5", "train_f1"],
            "found": list(reporting),
        }
    if mismatches:
        raise ValueError(f"Unsupported default Torch probe config: {mismatches}")


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
    svar_layer_start = int(cfg.get("svar_layer_start", 5))
    svar_layer_end = int(cfg.get("svar_layer_end", 19))
    X_train, y_train, train_rows = build_matrix(
        split_rows["train"], feature_set, label_protocol,
        svar_layer_start=svar_layer_start,
        svar_layer_end=svar_layer_end,
    )
    X_val, y_val, val_rows = build_matrix(
        [], feature_set, label_protocol,
        svar_layer_start=svar_layer_start,
        svar_layer_end=svar_layer_end,
    )
    X_test, y_test, test_rows = build_matrix(
        split_rows["test"], feature_set, label_protocol,
        svar_layer_start=svar_layer_start,
        svar_layer_end=svar_layer_end,
    )
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        if len(X) == 0 or len(np.unique(y)) < 2:
            raise ValueError(
                f"{feature_set} protocol={label_protocol} seed={seed}: {split} "
                "must contain both classes after protocol filtering, got "
                f"{np.bincount(y, minlength=2).tolist()}"
            )

    _seed_everything(seed)
    _validate_default_probe_config(cfg)
    # The requested default uses network-internal BatchNorm and no separate
    # z-score preprocessing. Keep identity arrays in the checkpoint so older
    # inference consumers that expect mean/std remain compatible.
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std = np.ones(X_train.shape[1], dtype=np.float32)

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QAProbe(
        X_train.shape[1],
        tuple(cfg.get("hidden_sizes", [128, 64, 32])),
        float(cfg.get("dropout", 0.3)),
    ).to(chosen_device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("lr_factor", 0.5)),
        patience=int(cfg.get("lr_patience", 5)),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
        drop_last=(len(X_train) % int(cfg.get("batch_size", 256)) == 1),
    )
    history = []
    maximum_epochs = int(
        cfg.get("max_epochs", cfg.get("num_epochs", cfg.get("epochs", 100)))
    )
    early_stopping_patience = int(cfg.get("early_stopping_patience", 10))
    best_state = None
    best_train_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(maximum_epochs):
        model.train()
        train_loss_sum = 0.0
        train_row_count = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(chosen_device), batch_y.to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite train loss for {feature_set} seed={seed}"
                )
            loss.backward()
            optimizer.step()
            batch_rows = int(batch_y.shape[0])
            train_loss_sum += float(loss.item()) * batch_rows
            train_row_count += batch_rows
        train_loss = train_loss_sum / max(train_row_count, 1)
        improved = train_loss < best_train_loss
        if improved:
            best_train_loss = float(train_loss)
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        scheduler.step(train_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "monitor": "train_loss",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": bool(improved),
            "stale_epochs": int(stale_epochs),
        })
        if stale_epochs >= early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    train_probability = _predict(model, X_train, chosen_device)
    threshold, train_f1 = choose_real_f1_threshold(
        y_train,
        train_probability,
    )
    test_probability = _predict(model, X_test, chosen_device)
    fixed_threshold = float(cfg.get("fixed_threshold", 0.5))
    train_metrics_by_threshold = {
        "fixed_0.5": classification_metrics(
            y_train, train_probability, fixed_threshold
        ),
        "train_f1": classification_metrics(y_train, train_probability, threshold),
    }
    test_metrics_by_threshold = {
        "fixed_0.5": classification_metrics(
            y_test, test_probability, fixed_threshold
        ),
        "train_f1": classification_metrics(y_test, test_probability, threshold),
    }

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
        "thresholds": {"fixed_0.5": fixed_threshold, "train_f1": threshold},
        "input_dim": int(X_train.shape[1]),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "feature_normalization": "none",
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
        "pos_weight": None,
        "best_val_loss": None,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_train_loss": float(best_train_loss),
        "threshold": threshold,
        "thresholds": {"fixed_0.5": fixed_threshold, "train_f1": threshold},
        "val_metrics": None,
        "train_threshold_f1": train_f1,
        "train_metrics": train_metrics_by_threshold["train_f1"],
        "test_metrics": test_metrics_by_threshold["train_f1"],
        "test_groups": grouped_metrics(test_rows, y_test, test_probability, threshold),
        "threshold_reports": {
            mode: {
                "threshold": (
                    fixed_threshold if mode == "fixed_0.5" else threshold
                ),
                "train_metrics": train_metrics_by_threshold[mode],
                "test_metrics": test_metrics_by_threshold[mode],
                "test_groups": grouped_metrics(
                    test_rows,
                    y_test,
                    test_probability,
                    fixed_threshold if mode == "fixed_0.5" else threshold,
                ),
            }
            for mode in ("fixed_0.5", "train_f1")
        },
        "history": history,
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "minimum_train_loss",
        "threshold_selection": "train_f1",
        "threshold_reporting": ["fixed_0.5", "train_f1"],
        "feature_normalization": "none",
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
    if not np.isfinite(scores).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("Threshold selection requires finite scores and binary labels")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    # For threshold t, every sorted score at index >= searchsorted(t) is
    # predicted Real. Prefix class counts therefore evaluate all unique
    # thresholds exactly in O(N log N), instead of invoking sklearn over the
    # full training set once per candidate (O(N^2)).
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    positive_prefix = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.cumsum(sorted_targets, dtype=np.int64),
    ))
    starts = np.searchsorted(sorted_scores, candidates, side="left")
    total_positive = int(positive_prefix[-1])
    true_positive = total_positive - positive_prefix[starts]
    predicted_positive = targets.size - starts
    false_positive = predicted_positive - true_positive
    false_negative = total_positive - true_positive
    true_negative = starts - positive_prefix[starts]
    denominators = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(
        2.0 * true_positive,
        denominators,
        out=np.zeros_like(candidates, dtype=np.float64),
        where=denominators != 0,
    )
    accuracy_values = (true_positive + true_negative) / float(targets.size)
    best_key = (-np.inf, -np.inf, -np.inf)
    best_threshold = 0.5
    for index, threshold in enumerate(candidates):
        key = (
            float(f1_values[index]),
            float(accuracy_values[index]),
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
        "threshold_reporting": results[0].get("threshold_reporting"),
    }
    for path in paths:
        values = [_nested(result["test_metrics"], path) for result in results]
        summary[path] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    reporting = tuple(results[0].get("threshold_reporting") or ())
    if reporting:
        for result in results:
            if tuple(result.get("threshold_reporting") or ()) != reporting:
                raise ValueError("Cannot aggregate mixed threshold reporting modes")
        summary["threshold_reports"] = {}
        for mode in reporting:
            mode_summary = {
                "threshold": _metric_statistics([
                    float(result["threshold_reports"][mode]["threshold"])
                    for result in results
                ])
            }
            for path in paths:
                mode_summary[path] = _metric_statistics([
                    _nested(
                        result["threshold_reports"][mode]["test_metrics"],
                        path,
                    )
                    for result in results
                ])
            summary["threshold_reports"][mode] = mode_summary
    return summary


def _metric_statistics(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(value) for value in array],
    }


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
