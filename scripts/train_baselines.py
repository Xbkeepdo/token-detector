#!/usr/bin/env python3
"""Train/evaluate paper baselines from ``OUTPUT/baseline/features.pkl``."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.baselines import (
    DHCPMLP,
    SVARMLP,
    build_dense_baseline_matrix,
    build_metatoken_classifier,
    evaluate_hallucination_scores,
    raw_labels_to_hallucination_targets,
    select_hallucination_threshold,
    sklearn_hallucination_scores,
    split_records_by_image,
    torch_hallucination_scores,
    train_torch_detector,
)
from features.baseline import (
    DHCPShardReader,
    HalLocObjectDetector,
    baseline_config,
    get_baseline_payload,
    halloc_optimizer_config,
    normalize_baseline_methods,
    validate_baseline_record,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, load_pkl, save_json, save_pkl
from utils.split_utils import validate_strict_811_split


DEFAULT_METHODS = ("metatoken", "svar", "dhcp", "projectaway", "halloc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override feature_extraction.baseline.seed for this training run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override the configured baseline methods (for example, omit halloc).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Store results/checkpoints in isolated subdirectories such as seed42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baseline_cfg = baseline_config(config)
    if args.seed is not None:
        baseline_cfg["seed"] = int(args.seed)
    baseline_dir = Path(args.output_dir) / str(
        baseline_cfg.get("output_subdir", "baseline")
    )
    feature_path = baseline_dir / "features.pkl"
    split_path = Path(args.output_dir) / "image_splits.json"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    records = load_pkl(str(feature_path))
    for record in records:
        validate_baseline_record(record)
    image_splits = load_json(str(split_path))
    image_split_counts = validate_strict_811_split(image_splits)
    configured_count = int((config.get("dataset") or {}).get("num_images", 0))
    if configured_count and sum(image_split_counts.values()) != configured_count:
        raise ValueError(
            "Strict split size differs from dataset.num_images: "
            f"{sum(image_split_counts.values())} != {configured_count}"
        )
    split_records = split_records_by_image(records, image_splits)
    _require_strict_splits(split_records)
    methods = normalize_baseline_methods(
        args.methods
        if args.methods is not None
        else baseline_cfg.get("methods", DEFAULT_METHODS)
    )
    if not methods:
        raise ValueError("No baseline methods selected for training")

    device = _resolve_device(args.device)
    result_dir = baseline_dir / "results"
    checkpoint_dir = baseline_dir / "checkpoints"
    if args.run_name is not None:
        run_name = str(args.run_name).strip()
        if (
            not run_name
            or Path(run_name).name != run_name
            or run_name in {".", ".."}
        ):
            raise ValueError("--run-name must be one safe path component")
        result_dir = result_dir / run_name
        checkpoint_dir = checkpoint_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{args.model}_baselines.json"
    output: dict[str, Any] = {
        "model": args.model,
        "seed": int(baseline_cfg.get("seed", 42)),
        "configured_methods": list(methods),
        "feature_path": str(feature_path),
        "split_path": str(split_path),
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "headline_positive_class": "hallucination",
        "label_protocol": "local_chair_object_spans_without_gpt_semantic_review",
        "counts": {name: len(rows) for name, rows in split_records.items()},
        "image_split_counts": image_split_counts,
        "methods": {},
    }

    for method in methods:
        print(f"[BaselineTrain] method={method} device={device}")
        if method == "metatoken":
            result = _train_metatoken(split_records, checkpoint_dir, baseline_cfg)
        elif method == "svar":
            result = _train_svar(split_records, checkpoint_dir, baseline_cfg, device)
        elif method == "dhcp":
            result = _train_dhcp(
                split_records, baseline_dir, checkpoint_dir, baseline_cfg, device
            )
        elif method == "projectaway":
            result = _evaluate_projectaway(split_records)
        else:
            result = _train_halloc(
                split_records, baseline_dir, checkpoint_dir, baseline_cfg, device
            )
        output["methods"][method] = result
        save_json(output, str(result_path))

    print(f"[BaselineTrain] saved {result_path}")


def _train_metatoken(split_records, checkpoint_dir, cfg) -> dict[str, Any]:
    matrices = {}
    for split in ("train", "val", "test"):
        matrices[split] = build_dense_baseline_matrix(
            split_records[split], "metatoken"
        )[:2]
    result = {}
    metatoken_cfg = dict(cfg.get("metatoken") or {})
    configured_classifiers = metatoken_cfg.get("classifiers", ("lr", "gb"))
    if isinstance(configured_classifiers, str):
        configured_classifiers = [configured_classifiers]
    classifiers = [
        str(value).strip().lower()
        for value in configured_classifiers
    ]
    unknown = set(classifiers) - {"lr", "gb"}
    if unknown or not classifiers:
        raise ValueError(
            "MetaToken classifiers must be a non-empty subset of ['lr', 'gb']; "
            f"got {classifiers}"
        )
    for kind in dict.fromkeys(classifiers):
        classifier = build_metatoken_classifier(kind, seed=int(cfg.get("seed", 42)))
        classifier.fit(
            matrices["train"][0],
            raw_labels_to_hallucination_targets(matrices["train"][1]),
        )
        val_scores = sklearn_hallucination_scores(classifier, matrices["val"][0])
        test_scores = sklearn_hallucination_scores(classifier, matrices["test"][0])
        threshold = select_hallucination_threshold(matrices["val"][1], val_scores)
        path = checkpoint_dir / f"metatoken_{kind}.pkl"
        save_pkl(classifier, str(path))
        result[kind] = {
            "paper_config": {
                "classifier": "LogisticRegression(lbfgs)"
                if kind == "lr"
                else "GradientBoostingClassifier(n_estimators=100)",
                "standardize": True,
            },
            "input_dim": int(matrices["train"][0].shape[1]),
            "threshold": threshold,
            "val_metrics": evaluate_hallucination_scores(
                matrices["val"][1], val_scores, threshold
            ),
            "test_metrics": evaluate_hallucination_scores(
                matrices["test"][1], test_scores, threshold
            ),
            "checkpoint": str(path),
        }
    return result


def _train_svar(split_records, checkpoint_dir, cfg, device) -> dict[str, Any]:
    matrices = {
        split: build_dense_baseline_matrix(split_records[split], "svar")[:2]
        for split in ("train", "val", "test")
    }
    svar_cfg = dict(cfg.get("svar") or {})
    seed = int(cfg.get("seed", 42))
    # Seed before module construction so the requested run seed controls both
    # SVAR's initial weights and the subsequent minibatch order.
    _seed_everything(seed)
    model = SVARMLP(
        matrices["train"][0].shape[1],
        hidden_dim=int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
    )
    trained = train_torch_detector(
        model=model,
        X_train=matrices["train"][0],
        raw_y_train=matrices["train"][1],
        X_val=matrices["val"][0],
        raw_y_val=matrices["val"][1],
        X_test=matrices["test"][0],
        raw_y_test=matrices["test"][1],
        epochs=int(_setting(svar_cfg, "epochs", "max_epochs", default=50)),
        learning_rate=float(svar_cfg.get("learning_rate", 1e-3)),
        batch_size=int(svar_cfg.get("batch_size", 32)),
        device=device,
        weighted_sampler=bool(svar_cfg.get("weighted_sampler", False)),
        standardize=bool(svar_cfg.get("standardize", False)),
        early_stopping_patience=int(
            svar_cfg.get("early_stopping_patience", 5)
        ),
        seed=seed,
    )
    path = checkpoint_dir / "svar.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": trained.state_dict,
            "input_dim": int(matrices["train"][0].shape[1]),
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "threshold": trained.threshold,
        },
    )
    return {
        "paper_config": {
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "learning_rate": float(svar_cfg.get("learning_rate", 1e-3)),
            "epochs": int(_setting(svar_cfg, "epochs", "max_epochs", default=50)),
            "early_stopping_patience": int(
                svar_cfg.get("early_stopping_patience", 5)
            ),
        },
        "input_dim": int(matrices["train"][0].shape[1]),
        "threshold": trained.threshold,
        "val_metrics": trained.val_metrics,
        "test_metrics": trained.test_metrics,
        "history": trained.history,
        "checkpoint": str(path),
    }


def _train_dhcp(split_records, baseline_dir, checkpoint_dir, cfg, device):
    datasets = {
        split: DHCPRecordDataset(
            split_records[split], baseline_dir / "dhcp" / "shards"
        )
        for split in ("train", "val", "test")
    }
    shape = datasets["train"].item_shape
    for split in ("val", "test"):
        if datasets[split].item_shape != shape:
            raise ValueError("DHCP tensor shape differs across train/val/test")
    input_dim = int(np.prod(shape))
    dhcp_cfg = dict(cfg.get("dhcp") or {})
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)
    dhcp_hidden = int(_setting(dhcp_cfg, "hidden_dim", "hidden_size", default=128))
    model = DHCPMLP(input_dim, hidden_dim=dhcp_hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(dhcp_cfg.get("learning_rate", 1e-3))
    )
    criterion = nn.CrossEntropyLoss()
    train_targets = np.asarray(datasets["train"].targets, dtype=np.int64)
    counts = np.bincount(train_targets, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights[train_targets], dtype=torch.double),
        num_samples=len(train_targets),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = int(dhcp_cfg.get("batch_size", 1024))
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, num_workers=0)
    best_loss = math.inf
    best_state = None
    history = []
    dhcp_epochs = int(_setting(dhcp_cfg, "epochs", "max_epochs", default=30))
    patience = int(dhcp_cfg.get("early_stopping_patience", 5))
    stale_epochs = 0
    for epoch in range(dhcp_epochs):
        model.train()
        losses = []
        for values, targets in train_loader:
            values, targets = values.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = _stream_scores(model, val_loader, device, criterion)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, patience):
                break
    if best_state is None:
        raise RuntimeError("No DHCP checkpoint selected")
    model.load_state_dict(best_state)
    _, val_scores = _stream_scores(model, val_loader, device, criterion)
    _, test_scores = _stream_scores(
        model,
        DataLoader(datasets["test"], batch_size=batch_size, num_workers=0),
        device,
        criterion,
    )
    val_raw = np.asarray([record["label"] for record in datasets["val"].records])
    test_raw = np.asarray([record["label"] for record in datasets["test"].records])
    threshold = select_hallucination_threshold(val_raw, val_scores)
    path = checkpoint_dir / "dhcp.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "input_shape": shape,
            "hidden_dim": dhcp_hidden,
            "threshold": threshold,
        },
    )
    return {
        "paper_config": {
            "protocol": "object_prediction_fixed_grid_mass_preserving",
            "target_grid": [12, 12],
            "hidden_dim": dhcp_hidden,
            "learning_rate": float(dhcp_cfg.get("learning_rate", 1e-3)),
            "batch_size": batch_size,
            "epochs": dhcp_epochs,
            "early_stopping_patience": patience,
            "weighted_sampler": True,
        },
        "input_shape": list(shape),
        "input_dim": input_dim,
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(val_raw, val_scores, threshold),
        "test_metrics": evaluate_hallucination_scores(test_raw, test_scores, threshold),
        "history": history,
        "checkpoint": str(path),
    }


def _evaluate_projectaway(split_records) -> dict[str, Any]:
    scores, labels = {}, {}
    for split in ("val", "test"):
        payloads = [
            get_baseline_payload(record, "projectaway")
            for record in split_records[split]
        ]
        scores[split] = np.asarray(
            [float(payload["hallucination_score"]) for payload in payloads]
        )
        labels[split] = np.asarray(
            [int(record["label"]) for record in split_records[split]]
        )
    threshold = select_hallucination_threshold(labels["val"], scores["val"])
    return {
        "paper_config": {"training_free": True, "threshold_selected_on": "val"},
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(
            labels["val"], scores["val"], threshold
        ),
        "test_metrics": evaluate_hallucination_scores(
            labels["test"], scores["test"], threshold
        ),
    }


def _train_halloc(split_records, baseline_dir, checkpoint_dir, cfg, device):
    datasets = {
        split: HalLocCachedDataset(split_records[split], baseline_dir)
        for split in ("train", "val", "test")
    }
    sample = datasets["train"][0]
    halloc_cfg = dict(cfg.get("halloc") or {})
    if not bool(halloc_cfg.get("freeze_clip", True)):
        raise ValueError(
            "The cache-based HalLoc pipeline requires freeze_clip=true. "
            "Unfreezing CLIP would require storing images and recomputing it in training."
        )
    paper = halloc_optimizer_config()
    model = HalLocObjectDetector(
        lvlm_hidden_size=int(sample[0].shape[-1]),
        clip_model_name=str(halloc_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
        visualbert_model_name=str(
            halloc_cfg.get("visualbert_model", "uclanlp/visualbert-vqa-coco-pre")
        ),
        freeze_clip=True,
        load_clip=False,
        clip_hidden_size=int(sample[1].shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(halloc_cfg.get("learning_rate", paper["learning_rate"])),
        betas=tuple(halloc_cfg.get("betas", paper["betas"])),
        weight_decay=float(halloc_cfg.get("weight_decay", paper["weight_decay"])),
    )
    criterion = nn.CrossEntropyLoss()
    batch_size = int(halloc_cfg.get("batch_size", paper["batch_size"]))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=_collate_halloc,
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    best_loss, best_state, history = math.inf, None, []
    halloc_epochs = int(_setting(halloc_cfg, "epochs", "max_epochs", default=25))
    scheduler_name = str(halloc_cfg.get("scheduler", "cosine")).strip().lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, halloc_epochs),
        )
    elif scheduler_name in {"none", "off", "disabled"}:
        scheduler = None
    else:
        raise ValueError("HalLoc scheduler must be 'cosine' or 'none'")
    patience = int(halloc_cfg.get("early_stopping_patience", 3))
    stale_epochs = 0
    for epoch in range(halloc_epochs):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                lvlm_embeddings=batch["lvlm_embeddings"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                clip_visual_features=batch["clip_visual_features"].to(device),
                visual_attention_mask=batch["visual_attention_mask"].to(device),
                object_indices=batch["object_indices"].to(device),
            )
            loss = criterion(logits, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = _halloc_scores(model, loaders["val"], device, criterion)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("No HalLoc checkpoint selected")
    model.load_state_dict(best_state)
    _, val_scores = _halloc_scores(model, loaders["val"], device, criterion)
    _, test_scores = _halloc_scores(model, loaders["test"], device, criterion)
    val_raw = np.asarray([record["label"] for record in datasets["val"].records])
    test_raw = np.asarray([record["label"] for record in datasets["test"].records])
    threshold = select_hallucination_threshold(val_raw, val_scores)
    path = checkpoint_dir / "halloc.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "threshold": threshold,
            "paper_metadata": model.paper_metadata(),
            "lvlm_hidden_size": int(sample[0].shape[-1]),
            "clip_hidden_size": int(sample[1].shape[-1]),
            "clip_model": str(
                halloc_cfg.get("clip_model", "openai/clip-vit-base-patch32")
            ),
            "visualbert_model": str(
                halloc_cfg.get(
                    "visualbert_model", "uclanlp/visualbert-vqa-coco-pre"
                )
            ),
        },
    )
    return {
        "paper_config": {
            **paper,
            "max_epochs": halloc_epochs,
            "scheduler": scheduler_name,
            "early_stopping_patience": patience,
        },
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(val_raw, val_scores, threshold),
        "test_metrics": evaluate_hallucination_scores(test_raw, test_scores, threshold),
        "history": history,
        "checkpoint": str(path),
    }


class DHCPRecordDataset(Dataset):
    def __init__(self, records, shard_root: Path):
        self.records = [
            record for record in records if "dhcp" in record.get("baselines", {})
        ]
        if not self.records:
            raise ValueError("No DHCP records found in split")
        self.reader = DHCPShardReader(shard_root)
        first_ref = _dhcp_reference(get_baseline_payload(self.records[0], "dhcp"))
        self.item_shape = tuple(int(value) for value in first_ref["shape"])
        self.targets = raw_labels_to_hallucination_targets(
            [record["label"] for record in self.records]
        ).tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        payload = get_baseline_payload(self.records[index], "dhcp")
        value = self.reader.load(_dhcp_reference(payload), as_tensor=True).float()
        return value, int(self.targets[index])


class HalLocCachedDataset(Dataset):
    def __init__(self, records, root: Path):
        self.records = [
            record for record in records if "halloc" in record.get("baselines", {})
        ]
        if not self.records:
            raise ValueError("No HalLoc records found in split")
        self.root = root.resolve()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        payload = get_baseline_payload(record, "halloc")
        if "cache_file" in payload:
            path = (self.root / str(payload["cache_file"])).resolve()
            if self.root != path and self.root not in path.parents:
                raise ValueError("HalLoc cache path escapes baseline directory")
            with np.load(path, allow_pickle=False) as cached:
                text = np.asarray(cached["lvlm_embeddings"], dtype=np.float32)
                visual = np.asarray(cached["clip_visual_features"], dtype=np.float32)
        else:
            text = np.asarray(payload["lvlm_embeddings"], dtype=np.float32)
            visual = np.asarray(payload["clip_visual_features"], dtype=np.float32)
        object_index = int(payload.get("object_index", record["response_token_idx"]))
        target = int(raw_labels_to_hallucination_targets([record["label"]])[0])
        return torch.from_numpy(text), torch.from_numpy(visual), object_index, target


def _collate_halloc(samples):
    max_text = max(item[0].shape[0] for item in samples)
    max_visual = max(item[1].shape[0] for item in samples)
    text_dim, visual_dim = samples[0][0].shape[1], samples[0][1].shape[1]
    text = torch.zeros(len(samples), max_text, text_dim)
    visual = torch.zeros(len(samples), max_visual, visual_dim)
    text_mask = torch.zeros(len(samples), max_text, dtype=torch.long)
    visual_mask = torch.zeros(len(samples), max_visual, dtype=torch.long)
    indices, targets = [], []
    for row, (text_value, visual_value, index, target) in enumerate(samples):
        text[row, : len(text_value)] = text_value
        visual[row, : len(visual_value)] = visual_value
        text_mask[row, : len(text_value)] = 1
        visual_mask[row, : len(visual_value)] = 1
        indices.append(index)
        targets.append(target)
    return {
        "lvlm_embeddings": text,
        "clip_visual_features": visual,
        "attention_mask": text_mask,
        "visual_attention_mask": visual_mask,
        "object_indices": torch.tensor(indices, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def _stream_scores(model, loader, device, criterion):
    model.eval()
    losses, scores = [], []
    with torch.no_grad():
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            logits = model(values)
            losses.append(float(criterion(logits, targets).item()))
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return float(np.mean(losses)), torch.cat(scores).numpy()


def _halloc_scores(model, loader, device, criterion):
    model.eval()
    losses, scores = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                lvlm_embeddings=batch["lvlm_embeddings"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                clip_visual_features=batch["clip_visual_features"].to(device),
                visual_attention_mask=batch["visual_attention_mask"].to(device),
                object_indices=batch["object_indices"].to(device),
            )
            targets = batch["targets"].to(device)
            losses.append(float(criterion(logits, targets).item()))
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return float(np.mean(losses)), torch.cat(scores).numpy()


def _dhcp_reference(payload):
    return payload.get("shard_reference", payload)


def _setting(
    config: Mapping[str, Any],
    primary: str,
    alias: str,
    *,
    default: Any,
) -> Any:
    if primary in config:
        return config[primary]
    return config.get(alias, default)


def _require_strict_splits(split_records) -> None:
    for name in ("train", "val", "test"):
        rows = split_records[name]
        labels = {int(row["label"]) for row in rows}
        if not rows or labels != {0, 1}:
            raise ValueError(
                f"Strict 8:1:1 requires both classes in {name}; got labels={labels}"
            )


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
