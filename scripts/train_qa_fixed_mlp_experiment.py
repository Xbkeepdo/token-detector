#!/usr/bin/env python3
"""Run an isolated QA MLP experiment with an explicitly fixed protocol.

This entry point intentionally does not reuse the active QA probe trainer.  It
combines MetaToken/SVAR records from the protocol-specific baseline artifact
with one DGST feature set from the root QA artifact, while keeping the requested
network, optimizer, early-stopping, checkpoint, and threshold behavior isolated
from formal pipeline results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import load_jsonl, sha256_file  # noqa: E402
from detection.qa_probe import (  # noqa: E402
    baseline_probe_vector,
    classification_metrics,
    feature_vector,
    label_for_protocol,
    safe_name,
    validate_image_level_splits,
)
from scripts.extract_qa_baselines import MANIFEST_NAME  # noqa: E402
from scripts.train_qa_probes import _validate_training_artifacts  # noqa: E402
from utils.config_utils import load_config  # noqa: E402
from utils.qa_paths import resolve_qa_output_name, resolve_qa_paths  # noqa: E402


EXPERIMENT_NAME = "mlp_bn_relu_trainloss_es_fixed05"
DGST_FEATURE_SET = (
    "hpre_raw_logit_gauss_risk+"
    "hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine@prompt_last_token"
)


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    loss: str = "BCEWithLogitsLoss"
    scheduler_monitor: str = "train_loss"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    early_stopping_monitor: str = "train_loss"
    early_stopping_patience: int = 10
    checkpoint_selection: str = "minimum_train_loss"
    threshold: float = 0.5
    threshold_selection: str = "fixed"
    initialization: str = "kaiming_uniform_relu"
    feature_normalization: str = "none"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    display_name: str
    source: str
    selector: str


FEATURE_SPECS = (
    FeatureSpec("metatoken", "MetaToken", "baseline", "metatoken"),
    FeatureSpec("svar", "SVAR", "baseline", "svar"),
    FeatureSpec(
        "hpre_raw_logit_gauss_risk_plus_ev",
        "hpre raw-logit Gaussian risk + EV",
        "root",
        DGST_FEATURE_SET,
    ),
)
SEEDS = (42, 43, 44)
METRIC_PATHS = (
    "auroc",
    "real_aupr",
    "hallucination_aupr",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "real.precision",
    "real.recall",
    "real.f1",
    "hallucination.precision",
    "hallucination.recall",
    "hallucination.f1",
)


class FixedMLP(nn.Module):
    """Linear-BatchNorm-ReLU-Dropout blocks followed by one binary logit."""

    def __init__(self, input_dim: int, config: ExperimentConfig):
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in config.hidden_sizes:
            layers.extend((
                nn.Linear(previous, int(hidden)),
                nn.BatchNorm1d(int(hidden)),
                nn.ReLU(),
                nn.Dropout(float(config.dropout)),
            ))
            previous = int(hidden)
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the isolated fixed BN-ReLU MLP QA experiment."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--output", dest="output_name")
    parser.add_argument(
        "--experiment", dest="output_name", help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--label-protocol",
        choices=("answer_correctness_all", "object_hallucination_yes_only"),
        default="answer_correctness_all",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    yaml_config = load_config(args.config)
    qa_cfg = yaml_config.get("qa_benchmarks") or {}
    output_root = args.output_root or qa_cfg.get("output_root")
    if not output_root:
        raise ValueError("QA output root is required by CLI or YAML")
    output_name = resolve_qa_output_name(args.output_name, qa_cfg)
    run_root = resolve_qa_paths(
        output_root, args.model, output_name, args.dataset
    ).benchmark_dir
    baseline_root = run_root / "baseline" / args.label_protocol
    paths = {
        "root_features": run_root / "features.pkl",
        "baseline_features": baseline_root / "features.pkl",
        "labels": run_root / "labels.jsonl",
        "splits": run_root / "image_splits.json",
        "baseline_manifest": baseline_root / MANIFEST_NAME,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing experiment inputs: {missing}")
    _validate_baseline_manifest(
        paths=paths,
        model=args.model,
        dataset=args.dataset,
        label_protocol=args.label_protocol,
    )
    fingerprint, provenance = _training_fingerprint(
        paths=paths,
        model=args.model,
        dataset=args.dataset,
        label_protocol=args.label_protocol,
        config=config,
    )

    print("[FixedMLP] Loading and validating root QA features...")
    with paths["root_features"].open("rb") as handle:
        root_rows = pickle.load(handle)
    if not isinstance(root_rows, list) or not root_rows:
        raise RuntimeError("Root QA features.pkl must contain a non-empty list")
    labels_list = load_jsonl(paths["labels"])
    labels = {str(row["key"]): row for row in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("labels.jsonl contains duplicate keys")
    with paths["splits"].open(encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    _validate_training_artifacts(root_rows, labels, split_manifest)
    image_counts = validate_image_level_splits(root_rows)

    print("[FixedMLP] Loading and aligning protocol-specific baseline features...")
    with paths["baseline_features"].open("rb") as handle:
        baseline_rows = pickle.load(handle)
    if not isinstance(baseline_rows, list) or not baseline_rows:
        raise RuntimeError("Baseline QA features.pkl must contain a non-empty list")
    baseline_by_key = {str(row.get("key")): row for row in baseline_rows}
    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Baseline QA features contain duplicate keys")
    root_keys = {str(row.get("key")) for row in root_rows}
    if root_keys != set(baseline_by_key):
        raise ValueError(
            "Root and baseline QA feature cohorts differ: "
            f"root_only={len(root_keys - set(baseline_by_key))}, "
            f"baseline_only={len(set(baseline_by_key) - root_keys)}"
        )
    _validate_aligned_rows(root_rows, baseline_by_key, args.label_protocol)
    matrices = _build_matrices(
        root_rows,
        baseline_by_key,
        label_protocol=args.label_protocol,
    )
    del root_rows, baseline_rows, baseline_by_key

    result_root = (
        run_root / "experiments" / EXPERIMENT_NAME / args.label_protocol
    )
    device = _resolve_device(args.device)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for spec in FEATURE_SPECS:
        split_data = matrices[spec.name]
        all_results[spec.name] = []
        for seed in SEEDS:
            seed_dir = result_root / safe_name(spec.name) / f"seed_{seed}"
            result_path = seed_dir / "result.json"
            if result_path.is_file() and not args.force:
                result = _load_json(result_path)
                _validate_reusable_result(
                    result,
                    feature_name=spec.name,
                    seed=seed,
                    fingerprint=fingerprint,
                )
                print(f"[FixedMLP] Reusing {spec.name} seed={seed}")
            else:
                print(
                    f"[FixedMLP] Training {spec.name} seed={seed} "
                    f"input_dim={split_data['train'][0].shape[1]} device={device}"
                )
                result = train_one(
                    feature_spec=spec,
                    split_data=split_data,
                    seed=seed,
                    config=config,
                    device=device,
                    output_dir=seed_dir,
                    label_protocol=args.label_protocol,
                    image_counts=image_counts,
                    fingerprint=fingerprint,
                    provenance=provenance,
                )
            all_results[spec.name].append(result)
            metrics = result["test_metrics"]
            print(
                f"[FixedMLP] {spec.name} seed={seed} "
                f"epoch={result['best_epoch']}/{result['epochs_ran']} "
                f"AUC={metrics['auroc']:.4f} "
                f"Real-F1={metrics['real']['f1']:.4f} "
                f"Hall-F1={metrics['hallucination']['f1']:.4f}"
            )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "model": args.model,
        "dataset": args.dataset,
        "label_protocol": args.label_protocol,
        "seeds": list(SEEDS),
        "network": {
            "hidden_sizes": list(config.hidden_sizes),
            "structure": "Linear-BatchNorm-ReLU-Dropout",
            "output_dim": 1,
            "initialization": config.initialization,
        },
        "training_config": asdict(config),
        "training_input_fingerprint": fingerprint,
        "training_provenance": provenance,
        "image_counts": image_counts,
        "features": {
            spec.name: _aggregate_feature_results(all_results[spec.name], spec)
            for spec in FEATURE_SPECS
        },
    }
    json_path = result_root / "summary_3seed.json"
    markdown_path = result_root / "summary_3seed.md"
    _atomic_json(json_path, summary)
    _atomic_text(markdown_path, _markdown(summary, all_results))
    print(f"[FixedMLP] Summary JSON: {json_path}")
    print(f"[FixedMLP] Summary Markdown: {markdown_path}")


def _validate_baseline_manifest(
    *, paths: Mapping[str, Path], model: str, dataset: str, label_protocol: str
) -> None:
    manifest = _load_json(paths["baseline_manifest"])
    expected = {
        "status": "complete",
        "model": model,
        "dataset": dataset,
        "label_protocol": label_protocol,
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Baseline manifest mismatch: {mismatches}")
    if sha256_file(paths["baseline_features"]) != manifest.get("features_sha256"):
        raise RuntimeError("Baseline features hash differs from its manifest")
    if sha256_file(paths["labels"]) != manifest.get("labels_sha256"):
        raise RuntimeError("QA labels hash differs from the baseline manifest")


def _training_fingerprint(
    *,
    paths: Mapping[str, Path],
    model: str,
    dataset: str,
    label_protocol: str,
    config: ExperimentConfig,
) -> tuple[str, dict[str, Any]]:
    provenance = {
        "input_sha256": {
            name: sha256_file(path)
            for name, path in paths.items()
            if name != "baseline_manifest"
        },
        "trainer_sha256": sha256_file(Path(__file__)),
        "model": model,
        "dataset": dataset,
        "label_protocol": label_protocol,
        "experiment_config": asdict(config),
        "feature_specs": [asdict(spec) for spec in FEATURE_SPECS],
        "seeds": list(SEEDS),
    }
    canonical = json.dumps(
        provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), provenance


def _validate_aligned_rows(
    root_rows: Sequence[Mapping[str, object]],
    baseline_by_key: Mapping[str, Mapping[str, object]],
    label_protocol: str,
) -> None:
    fields = ("image_id", "source_split", "probe_split")
    for root_row in root_rows:
        key = str(root_row.get("key"))
        baseline_row = baseline_by_key[key]
        mismatches = {
            field: {"root": root_row.get(field), "baseline": baseline_row.get(field)}
            for field in fields
            if root_row.get(field) != baseline_row.get(field)
        }
        root_label = label_for_protocol(root_row, label_protocol)
        baseline_label = label_for_protocol(baseline_row, label_protocol)
        if root_label != baseline_label:
            mismatches["label"] = {"root": root_label, "baseline": baseline_label}
        if mismatches:
            raise ValueError(f"Root/baseline mismatch for {key!r}: {mismatches}")


def _build_matrices(
    root_rows: Sequence[Mapping[str, object]],
    baseline_by_key: Mapping[str, Mapping[str, object]],
    *,
    label_protocol: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    storage: dict[str, dict[str, dict[str, list[Any]]]] = {
        spec.name: {
            split: {"vectors": [], "labels": []}
            for split in ("train", "test")
        }
        for spec in FEATURE_SPECS
    }
    for root_row in root_rows:
        split = str(root_row.get("probe_split"))
        if split == "val":
            raise ValueError("This experiment requires validation=none")
        if split not in ("train", "test"):
            raise ValueError(f"Unexpected probe split {split!r}")
        label = label_for_protocol(root_row, label_protocol)
        if label is None:
            continue
        baseline_row = baseline_by_key[str(root_row.get("key"))]
        for spec in FEATURE_SPECS:
            vector = (
                baseline_probe_vector(baseline_row, spec.selector)
                if spec.source == "baseline"
                else feature_vector(dict(root_row), spec.selector)
            )
            vector = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(
                    f"Invalid {spec.name} vector for {root_row.get('key')!r}"
                )
            storage[spec.name][split]["vectors"].append(vector)
            storage[spec.name][split]["labels"].append(int(label))

    matrices: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for spec in FEATURE_SPECS:
        matrices[spec.name] = {}
        expected_width: int | None = None
        for split in ("train", "test"):
            vectors = storage[spec.name][split]["vectors"]
            labels = storage[spec.name][split]["labels"]
            widths = {int(vector.shape[0]) for vector in vectors}
            if len(widths) != 1:
                raise ValueError(
                    f"Inconsistent {spec.name} widths in {split}: {sorted(widths)}"
                )
            width = next(iter(widths))
            if expected_width is not None and width != expected_width:
                raise ValueError(f"Train/test width mismatch for {spec.name}")
            expected_width = width
            X = np.stack(vectors).astype(np.float32, copy=False)
            y = np.asarray(labels, dtype=np.int64)
            if len(np.unique(y)) != 2:
                raise ValueError(f"{spec.name} {split} does not contain both labels")
            matrices[spec.name][split] = (X, y)
        train_X, train_y = matrices[spec.name]["train"]
        test_X, test_y = matrices[spec.name]["test"]
        print(
            f"[FixedMLP] {spec.name}: dim={train_X.shape[1]}, "
            f"train={len(train_y)} {np.bincount(train_y, minlength=2).tolist()}, "
            f"test={len(test_y)} {np.bincount(test_y, minlength=2).tolist()}"
        )
    return matrices


def train_one(
    *,
    feature_spec: FeatureSpec,
    split_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    label_protocol: str,
    image_counts: Mapping[str, int],
    fingerprint: str,
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    _seed_everything(seed)
    X_train, y_train = split_data["train"]
    X_test, y_test = split_data["test"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=0,
    )
    model = FixedMLP(X_train.shape[1], config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_train_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite train loss for {feature_spec.name} seed={seed}"
                )
            loss.backward()
            optimizer.step()
            batch_rows = int(batch_y.shape[0])
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
        train_loss = total_loss / max(total_rows, 1)
        improved = train_loss < best_train_loss
        if improved:
            best_train_loss = float(train_loss)
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        scheduler.step(train_loss)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": bool(improved),
            "stale_epochs": int(stale_epochs),
        })
        if stale_epochs >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    train_loss, train_probability = _predict(
        model, X_train, y_train, config.batch_size, criterion, device
    )
    test_loss, test_probability = _predict(
        model, X_test, y_test, config.batch_size, criterion, device
    )
    threshold = float(config.threshold)
    class_counts = {
        "train": np.bincount(y_train, minlength=2).tolist(),
        "test": np.bincount(y_test, minlength=2).tolist(),
        "val": [0, 0],
    }
    counts = {"train": len(y_train), "test": len(y_test), "val": 0}
    checkpoint = {
        "state_dict": best_state,
        "feature_set": feature_spec.name,
        "selector": feature_spec.selector,
        "seed": int(seed),
        "label_protocol": label_protocol,
        "input_dim": int(X_train.shape[1]),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "threshold": threshold,
        "config": asdict(config),
        "training_input_fingerprint": fingerprint,
    }
    result = {
        "experiment": EXPERIMENT_NAME,
        "feature_set": feature_spec.name,
        "display_name": feature_spec.display_name,
        "feature_source": feature_spec.source,
        "feature_selector": feature_spec.selector,
        "seed": int(seed),
        "label_protocol": label_protocol,
        "position": (
            "prompt_last_token" if feature_spec.source == "root" else "shared"
        ),
        "positive_class": "real",
        "input_dim": int(X_train.shape[1]),
        "counts": counts,
        "class_counts": class_counts,
        "image_counts": dict(image_counts),
        "network": {
            "hidden_sizes": list(config.hidden_sizes),
            "structure": "Linear-BatchNorm-ReLU-Dropout",
            "output_dim": 1,
        },
        "training_config": asdict(config),
        "epochs_ran": len(history),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "restored_train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "threshold": threshold,
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": config.checkpoint_selection,
        "threshold_selection": "fixed_0.5",
        "train_metrics": classification_metrics(
            y_train, train_probability, threshold
        ),
        "test_metrics": classification_metrics(y_test, test_probability, threshold),
        "training_input_fingerprint": fingerprint,
        "training_provenance": dict(provenance),
        "artifacts": {
            "checkpoint": str(output_dir / "checkpoint.pt"),
            "history": str(output_dir / "history.json"),
        },
    }
    _atomic_torch_save(output_dir / "checkpoint.pt", checkpoint)
    _atomic_json(output_dir / "history.json", history)
    _atomic_json(output_dir / "result.json", result)
    return result


def _predict(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    total_loss = 0.0
    total_rows = 0
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            batch_rows = int(batch_y.shape[0])
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return total_loss / max(total_rows, 1), np.concatenate(probabilities)


def _aggregate_feature_results(
    results: Sequence[Mapping[str, Any]], spec: FeatureSpec
) -> dict[str, Any]:
    first = results[0]
    return {
        "display_name": spec.display_name,
        "source": spec.source,
        "selector": spec.selector,
        "input_dim": first["input_dim"],
        "counts": first["counts"],
        "class_counts": first["class_counts"],
        "test_metrics": _aggregate_metrics(results, "test_metrics"),
        "train_metrics": _aggregate_metrics(results, "train_metrics"),
        "epochs_ran": _aggregate_values(results, "epochs_ran"),
        "best_epoch": _aggregate_values(results, "best_epoch"),
        "best_train_loss": _aggregate_values(results, "best_train_loss"),
    }


def _aggregate_metrics(
    results: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for path in METRIC_PATHS:
        values = [float(_nested(result[field], path)) for result in results]
        target = aggregated
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _stats(values)
    return aggregated


def _aggregate_values(
    results: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    return _stats([float(result[field]) for result in results])


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(value) for value in values],
    }


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        current = current[part]
    return current


def _markdown(
    summary: Mapping[str, Any],
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    cfg = summary["training_config"]
    lines = [
        f"# {summary['model']} {summary['dataset']} 固定 MLP 实验",
        "",
        "## 实验协议",
        "",
        f"- 标签协议：`{summary['label_protocol']}`；0=hallucination，1=real。",
        f"- seeds：`{', '.join(str(value) for value in summary['seeds'])}`。",
        f"- image split：train/val/test = {summary['image_counts']['train']}/0/{summary['image_counts']['test']}。",
        "- 网络：`input → 128 → 64 → 32 → 1`；每个隐藏层为 Linear-BatchNorm-ReLU-Dropout(0.3)。",
        "- 初始化：所有 Linear 使用 Kaiming uniform（ReLU），bias=0；BatchNorm weight=1、bias=0。",
        f"- Adam：lr={cfg['learning_rate']}，weight_decay={cfg['weight_decay']}，batch={cfg['batch_size']}；plain BCEWithLogitsLoss。",
        f"- ReduceLROnPlateau：监控 train loss，factor={cfg['scheduler_factor']}，patience={cfg['scheduler_patience']}。",
        f"- Early stopping：监控 train loss，patience={cfg['early_stopping_patience']}，恢复 minimum-train-loss checkpoint，最多 {cfg['max_epochs']} epochs。",
        "- 阈值固定为 0.5；test 不参与训练、调度、early stopping、checkpoint 或阈值选择。",
        "- 输入不做额外 z-score；首个隐藏层内的 BatchNorm 属于网络本身。",
        "",
        "## Test 结果（3 seeds 总体均值 ± 总体标准差）",
        "",
        "| 特征 | 维数 | AUROC | Real AUPR | Real F1 | Hall. F1 | Accuracy | Balanced Acc. | Macro-F1 | Best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, feature in summary["features"].items():
        test = feature["test_metrics"]
        lines.append(
            "| "
            + " | ".join((
                str(feature["display_name"]),
                str(feature["input_dim"]),
                _fmt(test["auroc"]),
                _fmt(test["real_aupr"]),
                _fmt(test["real"]["f1"]),
                _fmt(test["hallucination"]["f1"]),
                _fmt(test["accuracy"]),
                _fmt(test["balanced_accuracy"]),
                _fmt(test["macro_f1"]),
                _fmt(feature["best_epoch"], digits=1),
            ))
            + " |"
        )
    lines.extend((
        "",
        "## Train 诊断",
        "",
        "| 特征 | Train AUROC | Train Real F1 | Train Hall. F1 | Minimum train loss | Epochs ran |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for feature in summary["features"].values():
        train = feature["train_metrics"]
        lines.append(
            "| "
            + " | ".join((
                str(feature["display_name"]),
                _fmt(train["auroc"]),
                _fmt(train["real"]["f1"]),
                _fmt(train["hallucination"]["f1"]),
                _fmt(feature["best_train_loss"]),
                _fmt(feature["epochs_ran"], digits=1),
            ))
            + " |"
        )
    lines.extend((
        "",
        "## 各 seed Test 结果",
        "",
        "| 特征 | Seed | Best/ran epoch | AUROC | Real F1 | Hall. F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for spec in FEATURE_SPECS:
        for result in all_results[spec.name]:
            test = result["test_metrics"]
            lines.append(
                f"| {spec.display_name} | {result['seed']} | "
                f"{result['best_epoch']}/{result['epochs_ran']} | "
                f"{test['auroc']:.4f} | {test['real']['f1']:.4f} | "
                f"{test['hallucination']['f1']:.4f} | {test['accuracy']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _fmt(stat: Mapping[str, Any], *, digits: int = 4) -> str:
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def _validate_reusable_result(
    result: Mapping[str, Any],
    *,
    feature_name: str,
    seed: int,
    fingerprint: str,
) -> None:
    expected = {
        "experiment": EXPERIMENT_NAME,
        "feature_set": feature_name,
        "seed": seed,
        "training_input_fingerprint": fingerprint,
        "checkpoint_selection": "minimum_train_loss",
        "threshold_selection": "fixed_0.5",
        "threshold": 0.5,
    }
    mismatches = {
        key: {"expected": value, "found": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse incompatible fixed-MLP output; pass --force. "
            f"Mismatches: {mismatches}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
