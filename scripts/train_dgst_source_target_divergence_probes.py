#!/usr/bin/env python3
"""Train standalone 36-D probes for each source/target KL or JS curve."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_dgst_source_target_divergence_by_label import (
    BRANCHES,
    METRICS,
    _matrix,
    divergences,
    target_distribution,
)
from train_torch_probe_feature_sets import (
    TorchProbeConfig,
    _json_ready,
    _resolve_device,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, load_pkl, save_json
from utils.split_utils import validate_strict_82_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="source_target_divergence_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    split_protocol = str((config.get("training") or {}).get("split_protocol"))
    threshold_selection = str(
        (config.get("training") or {}).get("threshold_selection")
    )
    if (split_protocol, threshold_selection) != (
        "strict_82_no_validation",
        "train_f1",
    ):
        raise ValueError(
            "Divergence probes require strict_82_no_validation + train_f1."
        )

    torch_cfg = (config.get("training") or {}).get("torch_probe") or {}
    probe_config = TorchProbeConfig(
        hidden_sizes=tuple(int(value) for value in torch_cfg.get("hidden_sizes", [128, 64, 32])),
        dropout=float(torch_cfg.get("dropout", 0.3)),
        batch_size=int(torch_cfg.get("batch_size", 256)),
        num_epochs=int(torch_cfg.get("max_epochs", torch_cfg.get("num_epochs", 100))),
        learning_rate=float(torch_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(torch_cfg.get("weight_decay", 1.0e-5)),
        lr_factor=float(torch_cfg.get("lr_factor", 0.5)),
        lr_patience=int(torch_cfg.get("lr_patience", 5)),
        early_stopping_patience=int(torch_cfg.get("early_stopping_patience", 10)),
        fixed_threshold=float(torch_cfg.get("fixed_threshold", 0.5)),
        seed=int(args.seed),
        positive_class="real",
        split_protocol=split_protocol,
        threshold_selection=threshold_selection,
    )

    splits = load_json(str(output_dir / "image_splits.json"))
    validate_strict_82_split(splits)
    train_ids = {int(value) for value in splits["train"]}
    test_ids = {int(value) for value in splits["test"]}
    rows = load_pkl(str(output_dir / "features.pkl"))
    matrices, labels, image_ids = build_divergence_matrices(rows)
    train_mask = np.asarray([value in train_ids for value in image_ids], dtype=bool)
    test_mask = np.asarray([value in test_ids for value in image_ids], dtype=bool)
    if np.any(train_mask & test_mask):
        raise ValueError("Train/test token masks overlap.")
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Train/test token masks must both be non-empty.")
    for name, mask in (("train", train_mask), ("test", test_mask)):
        if set(np.unique(labels[mask]).tolist()) != {0, 1}:
            raise ValueError(f"{name} token split is not binary.")

    result_dir = output_dir / "results" / args.run_name / f"seed{args.seed}"
    artifact_root = result_dir / "torch_probe"
    result_path = result_dir / f"{args.model}_selected_feature_sets.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    device = _resolve_device(args.device)
    print(
        f"[DivergenceProbe] seed={args.seed} device={device} "
        f"rows={len(rows)} train={int(train_mask.sum())} test={int(test_mask.sum())} "
        f"config={asdict(probe_config)}"
    )

    for feature_name, matrix in matrices.items():
        artifact_dir = artifact_root / feature_name
        metrics = train_and_evaluate_probe(
            X_train=matrix[train_mask],
            y_train=labels[train_mask],
            X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
            y_val=np.empty((0,), dtype=np.int32),
            X_test=matrix[test_mask],
            y_test=labels[test_mask],
            config=probe_config,
            device=device,
            output_dir=str(artifact_dir),
        )
        metrics["num_features"] = int(matrix.shape[1])
        metrics["best_params"] = asdict(probe_config)
        metrics["artifacts"] = {
            "model": str(artifact_dir / "model.pt"),
            "history": str(artifact_dir / "history.json"),
            "config": str(artifact_dir / "config.json"),
        }
        result[feature_name] = {"torch_probe": _json_ready(metrics)}
        save_json(result, str(result_path))
        print(
            f"[DivergenceProbe] seed={args.seed} {feature_name} "
            f"AUC={metrics['auc']:.4f} real-F1={metrics['real_positive']['f1']:.4f} "
            f"hall-F1={metrics['hallucination_positive']['f1']:.4f}"
        )
    print(f"[DivergenceProbe] saved {result_path}")


def build_divergence_matrices(
    rows: list[dict],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    curves = {
        f"{branch}_{metric}": []
        for branch, _ in BRANCHES
        for metric, _ in METRICS
    }
    labels = []
    image_ids = []
    for row in rows:
        label = row.get("label")
        if label not in (0, 1):
            continue
        source = _matrix(row, "dgst_t_source_dist_per_layer")
        attention = _matrix(row, "dgst_t_attention_support_per_layer")
        for branch, _ in BRANCHES:
            target = target_distribution(row, branch, attention)
            values = divergences(source, target)
            for metric, _ in METRICS:
                curves[f"{branch}_{metric}"].append(
                    np.asarray(values[metric], dtype=np.float32)
                )
        labels.append(int(label))
        image_ids.append(int(row["image_id"]))
    matrices = {
        name: np.stack(values).astype(np.float32, copy=False)
        for name, values in curves.items()
    }
    if any(not np.all(np.isfinite(value)) for value in matrices.values()):
        raise ValueError("Derived divergence matrices contain non-finite values.")
    return (
        matrices,
        np.asarray(labels, dtype=np.int32),
        np.asarray(image_ids, dtype=np.int64),
    )


if __name__ == "__main__":
    main()
