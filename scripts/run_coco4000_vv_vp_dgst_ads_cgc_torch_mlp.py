#!/usr/bin/env python3
"""COCO4000 VV/VP risk-cosine vs DGST ADS/CGC torch MLP table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_torch_probe_feature_sets import (  # noqa: E402
    TorchProbeConfig,
    _json_ready,
    _resolve_device,
    train_and_evaluate_probe,
)
from utils.io_utils import load_json, load_pkl, save_json  # noqa: E402


MODEL_DISPLAY = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
}

TOKEN_FEATURES = [
    (
        "V(risk)",
        "V",
        "risk",
        ["dgst_t_transport_risk_relative_vll_cost_geo_per_layer"],
    ),
    (
        "V(cos)",
        "V",
        "cos",
        ["dgst_t_target_visual_hidden_cosine_relative_vll_per_layer"],
    ),
    (
        "V(risk+cos)",
        "V",
        "risk+cos",
        [
            "dgst_t_transport_risk_relative_vll_cost_geo_per_layer",
            "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
        ],
    ),
    (
        "V(risk_upd)",
        "V",
        "risk_upd",
        ["dgst_t_transport_risk_relative_vll_cost_geo_stateupd_lu1_per_layer"],
    ),
    (
        "V(risk_upd+cos)",
        "V",
        "risk_upd+cos",
        [
            "dgst_t_transport_risk_relative_vll_cost_geo_stateupd_lu1_per_layer",
            "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
        ],
    ),
    (
        "VP(risk)",
        "VP",
        "risk",
        ["dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_per_layer"],
    ),
    (
        "VP(cos)",
        "VP",
        "cos",
        ["dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"],
    ),
    (
        "VP(risk+cos)",
        "VP",
        "risk+cos",
        [
            "dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_per_layer",
            "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer",
        ],
    ),
    (
        "VP(risk_upd)",
        "VP",
        "risk_upd",
        ["dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1_per_layer"],
    ),
    (
        "VP(risk_upd+cos)",
        "VP",
        "risk_upd+cos",
        [
            "dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1_per_layer",
            "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer",
        ],
    ),
]

DGST_FEATURES = [
    ("ADS", "DGST", "ADS", ["ads_per_layer"]),
    ("CGC", "DGST", "CGC", ["cgc_per_layer"]),
    ("ADS+CGC", "DGST", "ADS+CGC", ["ads_per_layer", "cgc_per_layer"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/coco4000_vv_vp_vs_dgst_ads_cgc_torch_mlp",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["qwen2_5_vl_7b", "internvl_2_5_8b", "llava_1_5_7b"],
        default=["qwen2_5_vl_7b", "internvl_2_5_8b", "llava_1_5_7b"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = TorchProbeConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        num_epochs=int(args.num_epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        positive_class="hallucination",
    )
    device = _resolve_device(args.device)
    rows = []

    print(f"[COCO4000] out_dir={out_dir}")
    print(f"[COCO4000] device={device}, config={config}")

    for model_key in args.models:
        model_rows = run_model(
            repo_root=repo_root,
            out_dir=out_dir,
            model_key=model_key,
            config=config,
            device=device,
            seed=int(args.seed),
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
        )
        rows.extend(model_rows)
        write_outputs(out_dir, rows)

    write_outputs(out_dir, rows)
    print(f"[COCO4000] wrote {len(rows)} rows to {out_dir}")


def run_model(
    *,
    repo_root: Path,
    out_dir: Path,
    model_key: str,
    config: TorchProbeConfig,
    device,
    seed: int,
    resume: bool,
    dry_run: bool,
) -> list[dict]:
    token_dirs = token_feature_dirs(repo_root, model_key)
    dgst_dir = dgst_feature_dir(model_key)
    split_ids = load_split_ids(token_dirs["V"])
    split = make_811_split(split_ids, seed=seed)
    split_summary = {name: len(ids) for name, ids in split.items()}
    print(f"\n[COCO4000] {model_key}: split={split_summary}")

    model_rows = []
    token_cache: dict[str, list[dict]] = {}
    for display_name, family, feature_kind, keys in TOKEN_FEATURES:
        source_dir = token_dirs[family]
        cache_key = str(source_dir)
        if cache_key not in token_cache:
            token_cache[cache_key] = load_pkl(source_dir / "features.pkl")
        features = token_cache[cache_key]
        row = train_row(
            out_dir=out_dir,
            model_key=model_key,
            source="token-detector",
            source_dir=source_dir,
            display_name=display_name,
            family=family,
            feature_kind=feature_kind,
            keys=keys,
            features=features,
            split=split,
            config=config,
            device=device,
            resume=resume,
            dry_run=dry_run,
        )
        model_rows.append(row)

    dgst_features = load_pkl(dgst_dir / "features.pkl")
    for display_name, family, feature_kind, keys in DGST_FEATURES:
        row = train_row(
            out_dir=out_dir,
            model_key=model_key,
            source="dgst",
            source_dir=dgst_dir,
            display_name=display_name,
            family=family,
            feature_kind=feature_kind,
            keys=keys,
            features=dgst_features,
            split=split,
            config=config,
            device=device,
            resume=resume,
            dry_run=dry_run,
        )
        model_rows.append(row)

    return model_rows


def train_row(
    *,
    out_dir: Path,
    model_key: str,
    source: str,
    source_dir: Path,
    display_name: str,
    family: str,
    feature_kind: str,
    keys: Sequence[str],
    features: Sequence[dict],
    split: dict[str, set[int]],
    config: TorchProbeConfig,
    device,
    resume: bool,
    dry_run: bool,
) -> dict:
    row_slug = slug(display_name)
    artifacts_dir = out_dir / "artifacts" / model_key / row_slug
    metrics_path = artifacts_dir / "metrics.json"
    if resume and metrics_path.exists() and not dry_run:
        metrics = load_json(metrics_path)
        print(f"[COCO4000] skip {model_key} {display_name} (resume)")
    else:
        train_feats = split_features(features, split["train"])
        val_feats = split_features(features, split["val"])
        test_feats = split_features(features, split["test"])
        X_train, y_train = build_matrix(train_feats, keys)
        X_val, y_val = build_matrix(val_feats, keys)
        X_test, y_test = build_matrix(test_feats, keys)
        print(
            f"[COCO4000] {model_key:16s} {display_name:18s} "
            f"dims={X_train.shape[1]:3d} rows train/val/test="
            f"{X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}"
        )
        if dry_run:
            metrics = dry_metrics()
        else:
            metrics = train_and_evaluate_probe(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                config=config,
                device=device,
                output_dir=str(artifacts_dir),
            )
            metrics["num_train_rows"] = int(X_train.shape[0])
            metrics["num_val_rows"] = int(X_val.shape[0])
            metrics["num_test_rows"] = int(X_test.shape[0])
            metrics["feature_keys"] = list(keys)
            metrics["source_dir"] = str(source_dir)
            save_json(_json_ready(metrics), metrics_path)

    return result_row(
        model_key=model_key,
        source=source,
        source_dir=source_dir,
        display_name=display_name,
        family=family,
        feature_kind=feature_kind,
        keys=keys,
        metrics=metrics,
    )


def result_row(
    *,
    model_key: str,
    source: str,
    source_dir: Path,
    display_name: str,
    family: str,
    feature_kind: str,
    keys: Sequence[str],
    metrics: dict,
) -> dict:
    return {
        "model": model_key,
        "model_display": MODEL_DISPLAY[model_key],
        "feature": display_name,
        "family": family,
        "feature_kind": feature_kind,
        "source": source,
        "auc": as_float(metrics.get("auc")),
        "aupr": as_float(metrics.get("aupr")),
        "f1": as_float(metrics.get("f1")),
        "precision": as_float(metrics.get("precision")),
        "recall": as_float(metrics.get("recall")),
        "accuracy": as_float(metrics.get("accuracy")),
        "n": as_int(metrics.get("num_test_rows")),
        "num_features": as_int(metrics.get("num_features")),
        "best_epoch": as_int(metrics.get("best_epoch")),
        "source_dir": str(source_dir),
        "feature_keys": ";".join(keys),
    }


def token_feature_dirs(repo_root: Path, model_key: str) -> dict[str, Path]:
    if model_key == "qwen2_5_vl_7b":
        return {
            "V": repo_root / "outputs/qwen2_5_vl_7b/COCO4000-vv-relativevll-cost-geo-updlu1",
            "VP": repo_root / "outputs/qwen2_5_vl_7b/COCO4000-vpvp-relativevll-cost-geo-updlu1",
        }
    base = repo_root / "outputs" / model_key / "COCO4000-dualscope-relativevll-cost-geo-updlu1"
    return {"V": base, "VP": base}


def dgst_feature_dir(model_key: str) -> Path:
    base = Path("/home/apulis-dev/userdata/DGST/token-grounding-detector/outputs")
    if model_key == "llava_1_5_7b":
        return base / model_key / "4000COCO"
    return base / model_key / "COCO4000 TGD"


def load_split_ids(source_dir: Path) -> list[int]:
    splits = load_json(source_dir / "image_splits.json")
    ids = set()
    for name in ("train", "val", "test"):
        ids.update(int(item) for item in splits.get(name, []))
    if not ids:
        raise ValueError(f"No image ids in {source_dir / 'image_splits.json'}")
    return sorted(ids)


def make_811_split(image_ids: Sequence[int], *, seed: int) -> dict[str, set[int]]:
    ids = list(dict.fromkeys(int(item) for item in image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_train = int(round(n_total * 0.8))
    n_val = int(round(n_total * 0.1))
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def split_features(features: Sequence[dict], image_ids: set[int]) -> list[dict]:
    return [feat for feat in features if int(feat.get("image_id", -1)) in image_ids]


def build_matrix(features: Sequence[dict], keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for feat in features:
        if feat.get("label") not in (0, 1):
            continue
        vectors = []
        missing = False
        for key in keys:
            value = feat.get(key)
            if value is None:
                missing = True
                break
            vectors.append(np.asarray(value, dtype=np.float32).reshape(-1))
        if missing:
            continue
        rows.append(np.concatenate(vectors).astype(np.float32))
        labels.append(int(feat["label"]))
    if not rows:
        raise ValueError(f"No rows for keys={keys}")
    return np.stack(rows, axis=0), np.asarray(labels, dtype=np.int32)


def write_outputs(out_dir: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    csv_path = out_dir / "coco4000_vv_vp_dgst_ads_cgc_all.csv"
    json_path = out_dir / "coco4000_vv_vp_dgst_ads_cgc_all.json"
    md_path = out_dir / "coco4000_vv_vp_dgst_ads_cgc_table.md"
    tex_path = out_dir / "coco4000_vv_vp_dgst_ads_cgc_table.tex"

    fieldnames = [
        "model",
        "model_display",
        "feature",
        "family",
        "feature_kind",
        "source",
        "auc",
        "aupr",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "n",
        "num_features",
        "best_epoch",
        "source_dir",
        "feature_keys",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(_json_ready(list(rows)), json_path)
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    tex_path.write_text(latex_table(rows), encoding="utf-8")


def markdown_table(rows: Sequence[dict]) -> str:
    lines = [
        "# COCO4000 VV/VP vs DGST ADS/CGC Torch MLP",
        "",
        "Split: image-level `train:val:test = 8:1:1`, seed `42`. "
        "Validation selects checkpoint; only test metrics are reported.",
        "",
        "| Model | Feature | AUC | AUPR | F1 | Precision | Recall | Acc | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_display"]),
                    f"`{row['feature']}`",
                    fmt(row["auc"]),
                    fmt(row["aupr"]),
                    fmt(row["f1"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["accuracy"]),
                    str(row["n"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def latex_table(rows: Sequence[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Feature & AUC & AUPR & F1 & Prec. & Rec. & Acc. & N \\",
        r"\midrule",
    ]
    previous_model = None
    for row in rows:
        if previous_model is not None and row["model_display"] != previous_model:
            lines.append(r"\midrule")
        previous_model = row["model_display"]
        lines.append(
            " & ".join(
                [
                    latex_escape(str(row["model_display"])),
                    latex_escape(str(row["feature"])),
                    fmt(row["auc"]),
                    fmt(row["aupr"]),
                    fmt(row["f1"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["accuracy"]),
                    str(row["n"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def dry_metrics() -> dict:
    return {
        "auc": float("nan"),
        "aupr": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "accuracy": float("nan"),
        "num_test_rows": 0,
        "num_features": 0,
        "best_epoch": -1,
    }


def slug(value: str) -> str:
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(keep).split("_") if part)


def as_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def as_int(value) -> int:
    if value is None:
        return 0
    return int(value)


def fmt(value) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.3f}"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


if __name__ == "__main__":
    main()
