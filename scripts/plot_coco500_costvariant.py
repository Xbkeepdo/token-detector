#!/usr/bin/env python3
"""Plot and export VV cost-variant layer curves."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


FEATURES = {
    "risk-geo": "dgst_t_risk_geo_per_layer",
    "risk-cosine-hpre": "dgst_t_risk_cosine_hpre_per_layer",
    "risk-sqrt-hmid": "dgst_t_risk_sqrt_hmid_per_layer",
    "risk-sqrt-hpre": "dgst_t_risk_sqrt_hpre_per_layer",
    "risk-rawAttention-hmid": "dgst_t_risk_raw_attention_hmid_per_layer",
    "risk-rawAttention-hpre": "dgst_t_risk_raw_attention_hpre_per_layer",
    "gauss-risk-geo": "dgst_t_gauss_risk_geo_per_layer",
    "gauss-risk-cosine-hpre": "dgst_t_gauss_risk_cosine_hpre_per_layer",
    "gauss-risk-sqrt-hmid": "dgst_t_gauss_risk_sqrt_hmid_per_layer",
    "gauss-risk-sqrt-hpre": "dgst_t_gauss_risk_sqrt_hpre_per_layer",
    "hprecosine": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label", default="COCO500")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    features = load_pkl(output_dir / "features.pkl")
    if not features:
        raise ValueError(f"No feature rows in {output_dir / 'features.pkl'}")

    rows = []
    curves = {}
    for display_name, key in FEATURES.items():
        by_label = {}
        for label in (0, 1):
            label_rows = [row[key] for row in features if int(row["label"]) == label]
            if not label_rows:
                continue
            values = np.asarray(label_rows, dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"Invalid values for {display_name}, label={label}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values for {display_name}, label={label}")
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            by_label[label] = (mean, std)
            for layer, (mean_value, std_value) in enumerate(zip(mean, std), start=1):
                rows.append(
                    {
                        "model": args.model,
                        "feature": display_name,
                        "layer": layer,
                        "label": "hallucination" if label == 0 else "non_hallucination",
                        "count": int(values.shape[0]),
                        "mean": float(mean_value),
                        "std": float(std_value),
                    }
                )
        if not by_label:
            raise ValueError(f"No labeled rows for {display_name}")
        curves[display_name] = by_label

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(args.dataset_label)
    ).strip("_")
    stem = results_dir / f"{args.model}_{dataset_slug}_costvariant_layerwise_by_label"
    _write_csv(stem.with_suffix(".csv"), rows)
    _plot(stem, args.model, str(args.dataset_label), curves)
    print(f"[CostVariantPlot] Wrote {stem}.{{csv,png,pdf}}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(stem: Path, model: str, dataset_label: str, curves: dict) -> None:
    columns = 3
    rows = math.ceil(len(curves) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.5 * rows), squeeze=False)
    for axis, (name, by_label) in zip(axes.flat, curves.items()):
        for label, (mean, std) in by_label.items():
            color = "#c44e52" if label == 0 else "#4c72b0"
            layers = np.arange(1, len(mean) + 1)
            label_name = "hallucination" if label == 0 else "non-hallucination"
            axis.plot(layers, mean, color=color, linewidth=1.8, label=label_name)
            axis.fill_between(layers, mean - std, mean + std, color=color, alpha=0.12)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(curves) :]:
        axis.axis("off")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{model}: {dataset_label} VV cost variants", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
