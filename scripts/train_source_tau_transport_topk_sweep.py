#!/usr/bin/env python3
"""Train the YAML source-tau x transport-top-K four-gate sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.train_feature_sets import feature_block
from scripts.train_torch_probe_feature_sets import (
    TorchProbeConfig,
    _json_ready,
    _resolve_device,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, load_pkl
from utils.split_utils import validate_strict_82_split


SCOPE_BLOCKS = {
    "vv": {
        "ev": "hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine",
    },
    "vpend": {
        "ev": "vpend_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine",
    },
}
SWEEP_RISK_KEY = (
    "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_per_layer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--layer-start", type=int, default=20)
    parser.add_argument("--layer-end", type=int, default=29)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--run-name",
        default="source_tau_x_transport_topk_hpre_risk_plus_ev",
    )
    return parser.parse_args()


def probe_config_from_yaml(config_path: str, seed: int) -> TorchProbeConfig:
    root = load_config(config_path)
    training = root.get("training") or {}
    cfg = training.get("torch_probe") or {}
    return TorchProbeConfig(
        hidden_sizes=tuple(int(x) for x in cfg.get("hidden_sizes", [128, 64, 32])),
        dropout=float(cfg.get("dropout", 0.3)),
        drop_last=bool(cfg.get("drop_last", False)),
        batch_size=int(cfg.get("batch_size", 256)),
        num_epochs=int(cfg.get("max_epochs", cfg.get("num_epochs", 100))),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        lr_factor=float(cfg.get("lr_factor", 0.5)),
        lr_patience=int(cfg.get("lr_patience", 5)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 10)),
        seed=int(seed),
        positive_class="real",
        split_protocol=str(training.get("split_protocol", "strict_82_no_validation")),
        threshold_selection=str(training.get("threshold_selection", "train_f1")),
        fixed_threshold=float(cfg.get("fixed_threshold", 0.5)),
        threshold_reporting=tuple(cfg.get("threshold_reporting", ["fixed_0.5", "train_f1"])),
        checkpoint_selection=str(cfg.get("checkpoint_selection", "minimum_train_loss")),
    )


def build_matrices(
    rows: list[dict],
    expected_variants: dict[str, tuple[float, int]],
    layer_start: int,
    layer_end: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    values = {
        f"{scope}_{variant_slug}_hpre_risk_plus_ev": []
        for scope in SCOPE_BLOCKS
        for variant_slug in expected_variants
    }
    labels = []
    image_ids = []
    risk_dims = set()
    ev_dims = set()
    for index, row in enumerate(rows):
        label = row.get("label")
        if label not in (0, 1):
            continue
        row_sweep = row.get("dgst_t_hparam_sweep")
        if not isinstance(row_sweep, dict):
            raise KeyError(f"Missing dgst_t_hparam_sweep at row {index}")
        if set(row_sweep) != set(expected_variants):
            raise ValueError(
                f"Sweep variants differ at row {index}: "
                f"{sorted(row_sweep)} != {sorted(expected_variants)}"
            )
        for scope, blocks in SCOPE_BLOCKS.items():
            ev = feature_block(row, blocks["ev"]).astype(np.float32, copy=False)
            if ev.ndim != 1 or not np.all(np.isfinite(ev)):
                raise ValueError(f"Invalid {scope} EV curve at row {index}")
            for variant_slug, expected in expected_variants.items():
                variant = row_sweep[variant_slug]
                actual = (
                    float(variant["source_tau"]),
                    int(variant["transport_top_k"]),
                )
                if actual != expected:
                    raise ValueError(
                        f"Variant metadata mismatch at row {index}: "
                        f"{variant_slug}={actual}, expected {expected}"
                    )
                try:
                    risk = np.asarray(
                        variant[scope][SWEEP_RISK_KEY], dtype=np.float32
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Missing {scope}/{variant_slug}/{SWEEP_RISK_KEY} "
                        f"at row {index}"
                    ) from exc
                if risk.ndim != 1 or not np.all(np.isfinite(risk)):
                    raise ValueError(
                        f"Invalid {scope}/{variant_slug} risk at row {index}"
                    )
                if risk.shape != ev.shape:
                    raise ValueError(
                        f"{scope}/{variant_slug} risk/EV shape mismatch: "
                        f"{risk.shape} != {ev.shape}"
                    )
                if (
                    layer_start < 0
                    or layer_end > risk.size
                    or layer_end <= layer_start
                ):
                    raise ValueError(
                        f"Requested risk slice [{layer_start},{layer_end}) "
                        f"outside curve length {risk.size}"
                    )
                risk_dims.add(int(risk.size))
                ev_dims.add(int(ev.size))
                values[
                    f"{scope}_{variant_slug}_hpre_risk_plus_ev"
                ].append(
                    np.concatenate([risk[layer_start:layer_end], ev]).astype(
                        np.float32
                    )
                )
        labels.append(int(label))
        image_ids.append(int(row["image_id"]))
        if len(labels) % 2000 == 0:
            print(f"[derive] processed {len(labels)} rows", flush=True)
    matrices = {
        name: np.stack(items).astype(np.float32, copy=False)
        for name, items in values.items()
    }
    return (
        matrices,
        np.asarray(labels, dtype=np.int32),
        np.asarray(image_ids, dtype=np.int64),
        {
            "rows_total": len(rows),
            "rows_binary": len(labels),
            "risk_curve_dimensions": sorted(risk_dims),
            "ev_curve_dimensions": sorted(ev_dims),
            "risk_layer_slice": [int(layer_start), int(layer_end)],
            "risk_layers_inclusive": [int(layer_start), int(layer_end - 1)],
            "selected_risk_dimension": int(layer_end - layer_start),
            "variants": {
                slug: {"source_tau": tau, "transport_top_k": top_k}
                for slug, (tau, top_k) in expected_variants.items()
            },
        },
    )


def extract_report(metrics: dict, report: str) -> dict:
    if report == "train_f1":
        item = metrics
        threshold = metrics["decision_threshold"]
    else:
        item = metrics["threshold_reports"]["fixed_0.5"]["test_metrics"]
        threshold = 0.5
    return {
        "threshold": float(threshold),
        "auc": float(item["real_positive"]["auc"]),
        "real_aupr": float(item["real_positive"]["aupr"]),
        "hall_aupr": float(item["hallucination_positive"]["aupr"]),
        "real_f1": float(item["real_positive"]["f1"]),
        "hall_f1": float(item["hallucination_positive"]["f1"]),
        "accuracy": float(item["accuracy"]),
    }


def train_all(
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    image_ids: np.ndarray,
    splits: dict,
    seeds: list[int],
    config_path: str,
    device_name: str,
    result_dir: Path,
) -> tuple[list[dict], dict]:
    train_ids = {int(x) for x in splits["train"]}
    test_ids = {int(x) for x in splits["test"]}
    train_mask = np.asarray([int(x) in train_ids for x in image_ids], dtype=bool)
    test_mask = np.asarray([int(x) in test_ids for x in image_ids], dtype=bool)
    if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
        raise ValueError("Invalid strict train/test masks")
    if set(np.unique(labels[train_mask]).tolist()) != {0, 1}:
        raise ValueError("Train token split is not binary")
    if set(np.unique(labels[test_mask]).tolist()) != {0, 1}:
        raise ValueError("Test token split is not binary")
    device = _resolve_device(device_name)
    base_config = probe_config_from_yaml(config_path, seeds[0])
    print(
        f"[train] device={device} train_tokens={int(train_mask.sum())} "
        f"test_tokens={int(test_mask.sum())} config={asdict(base_config)}",
        flush=True,
    )
    records = []
    raw_metrics = {}
    for seed in seeds:
        cfg = replace(base_config, seed=int(seed))
        raw_metrics[str(seed)] = {}
        for feature, matrix in matrices.items():
            artifact_dir = result_dir / "torch_probe" / f"seed{seed}" / feature
            print(f"[train] seed={seed} feature={feature} dims={matrix.shape[1]}", flush=True)
            metrics = train_and_evaluate_probe(
                X_train=matrix[train_mask],
                y_train=labels[train_mask],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=matrix[test_mask],
                y_test=labels[test_mask],
                config=cfg,
                device=device,
                output_dir=str(artifact_dir),
            )
            metrics["num_features"] = int(matrix.shape[1])
            metrics["best_params"] = asdict(cfg)
            raw_metrics[str(seed)][feature] = _json_ready(metrics)
            for report in ("train_f1", "fixed_0.5"):
                records.append({
                    "seed": int(seed),
                    "feature": feature,
                    "num_features": int(matrix.shape[1]),
                    "threshold_report": report,
                    **extract_report(metrics, report),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_ran": int(metrics["epochs_ran"]),
                })
            (result_dir / "raw_metrics.json").write_text(
                json.dumps(raw_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[train] done seed={seed} feature={feature} "
                f"AUC={metrics['auc']:.4f} realF1={metrics['real_positive']['f1']:.4f} "
                f"hallF1={metrics['hallucination_positive']['f1']:.4f}",
                flush=True,
            )
    audit = {
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "train_token_rows": int(train_mask.sum()),
        "test_token_rows": int(test_mask.sum()),
        "probe_config": asdict(base_config),
    }
    return records, audit


def aggregate(records: list[dict]) -> list[dict]:
    result = []
    metric_names = (
        "auc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
        "accuracy", "threshold", "best_epoch", "epochs_ran",
    )
    for feature in sorted({x["feature"] for x in records}):
        for report in ("train_f1", "fixed_0.5"):
            selected = [x for x in records if x["feature"] == feature and x["threshold_report"] == report]
            row = {
                "feature": feature,
                "num_features": int(selected[0]["num_features"]),
                "threshold_report": report,
                "seeds": [int(x["seed"]) for x in selected],
            }
            for metric in metric_names:
                values = np.asarray([x[metric] for x in selected], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result.append(row)
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    normalized = []
    for source in rows:
        row = dict(source)
        if isinstance(row.get("seeds"), list):
            row["seeds"] = " ".join(str(x) for x in row["seeds"])
        normalized.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def fmt(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"


def write_markdown(path: Path, summary: list[dict], feature_audit: dict, train_audit: dict) -> None:
    primary = [x for x in summary if x["threshold_report"] == "train_f1"]
    fixed = [x for x in summary if x["threshold_report"] == "fixed_0.5"]
    risk_start, risk_end = feature_audit["risk_layers_inclusive"]
    selected_risk_dim = int(feature_audit["selected_risk_dimension"])
    ev_dim = int(feature_audit["ev_curve_dimensions"][0])
    lines = [
        "# Source tau × transport Top-K sweep (VV / VPEND)",
        "",
        "## Protocol",
        "",
        "- Feature: hpre raw-logit Gaussian sqrt-matched-state risk "
        f"layers {risk_start}-{risk_end} + corresponding full-layer EV.",
        f"- Dimension: {selected_risk_dim} risk layers + {ev_dim} EV layers "
        f"= {selected_risk_dim + ev_dim}.",
        "- Grid: YAML `source_tau_values` x `transport_top_k_values`.",
        "- VV and VPEND are trained independently with identical image splits, seeds, and MLP settings.",
        f"- Feature audit: `{json.dumps(feature_audit, ensure_ascii=False)}`",
        f"- Training audit: `{json.dumps(train_audit, ensure_ascii=False)}`",
        "",
        "## Train-F1 threshold selected on training rows only",
        "",
        "| Feature | Dim | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda x: x["auc_mean"], reverse=True):
        lines.append(
            f"| {row['feature']} | {row['num_features']} | {fmt(row, 'auc')} | "
            f"{fmt(row, 'real_aupr')} | {fmt(row, 'hall_aupr')} | "
            f"{fmt(row, 'real_f1')} | {fmt(row, 'hall_f1')} | {fmt(row, 'accuracy')} |"
        )
    lines += [
        "", "## Fixed threshold 0.5", "",
        "| Feature | Dim | AUROC | Real F1 | Hall F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(fixed, key=lambda x: x["auc_mean"], reverse=True):
        lines.append(
            f"| {row['feature']} | {row['num_features']} | {fmt(row, 'auc')} | "
            f"{fmt(row, 'real_f1')} | {fmt(row, 'hall_f1')} | {fmt(row, 'accuracy')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.layer_start < 0 or args.layer_end <= args.layer_start:
        raise ValueError("Expected 0 <= layer_start < layer_end")
    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "results" / args.run_name
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    config_root = load_config(args.config)
    torch_cfg = ((config_root.get("training") or {}).get("torch_probe") or {})
    dgst_cfg = ((config_root.get("feature_extraction") or {}).get("dgst_t") or {})
    tau_values = [float(x) for x in dgst_cfg.get("source_tau_values", [])]
    top_k_values = [int(x) for x in dgst_cfg.get("transport_top_k_values", [])]
    if not tau_values or not top_k_values:
        raise ValueError(
            "YAML must define feature_extraction.dgst_t.source_tau_values "
            "and transport_top_k_values"
        )

    def tau_slug(value: float) -> str:
        return format(value, ".12g").replace("-", "m").replace(".", "p")
    expected_variants = {
        f"tau{tau_slug(tau)}_topk{top_k}": (tau, top_k)
        for tau in tau_values
        for top_k in top_k_values
    }
    seeds = args.seeds or [int(x) for x in torch_cfg.get("seeds", [43, 44, 45])]
    splits = load_json(str(output_dir / "image_splits.json"))
    validate_strict_82_split(splits)
    print(f"[load] {output_dir / 'features.pkl'}", flush=True)
    rows = load_pkl(str(output_dir / "features.pkl"))
    matrices, labels, image_ids, feature_audit = build_matrices(
        rows, expected_variants, args.layer_start, args.layer_end
    )
    del rows
    records, train_audit = train_all(
        matrices, labels, image_ids, splits, seeds, args.config,
        args.device, result_dir,
    )
    summary = aggregate(records)
    write_csv(records, result_dir / "per_seed.csv")
    write_csv(summary, result_dir / "summary.csv")
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(result_dir / "summary.md", summary, feature_audit, train_audit)
    print(f"[done] {result_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
