#!/usr/bin/env python3
"""Plot the three-model COCO100 target-gate risk comparison in one figure."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_coco100_gate_comparison import (  # noqa: E402
    LABEL_HALLUCINATION,
    LABEL_NON_HALLUCINATION,
    METHOD_KEYS,
    _stats_for_key,
)
from utils.io_utils import load_pkl  # noqa: E402


DEFAULT_MODELS = (
    "llava_1_5_7b",
    "internvl_2_5_8b",
    "qwen2_5_vl_7b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--outputs-root", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "coco100-gate-comparison-summary"),
    )
    parser.add_argument("--dataset-label", default="COCO100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    stats: dict[str, dict[str, dict[int, dict[str, object]]]] = {}
    for model in args.models:
        feature_path = outputs_root / model / "COCO100-gate-comparison" / "features.pkl"
        features = load_pkl(feature_path)
        if not features:
            raise ValueError(f"No feature rows in {feature_path}")
        stats[model] = {
            method: _stats_for_key(features, key)
            for method, key in METHOD_KEYS.items()
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "three_model_coco100_gate_risk_by_label"
    _write_csv(stem.with_suffix(".csv"), stats, str(args.dataset_label))
    _plot(stem, stats, str(args.dataset_label))
    print(f"[GateComparisonCombined] Wrote {stem}.{{csv,png,pdf}}")


def _plot(stem: Path, stats: dict, dataset_label: str) -> None:
    models = list(stats)
    methods = list(METHOD_KEYS)
    figure, axes = plt.subplots(
        len(models),
        len(methods),
        figsize=(5.7 * len(methods), 4.0 * len(models)),
        squeeze=False,
    )
    colors = {
        LABEL_HALLUCINATION: "#d55e00",
        LABEL_NON_HALLUCINATION: "#0072b2",
    }
    label_names = {
        LABEL_HALLUCINATION: "Hallucination",
        LABEL_NON_HALLUCINATION: "Non-hallucination",
    }
    legend_handles = None

    for row_index, model in enumerate(models):
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            by_label = stats[model][method]
            for label in (LABEL_HALLUCINATION, LABEL_NON_HALLUCINATION):
                item = by_label[label]
                mean = np.asarray(item["mean"], dtype=np.float64)
                sem = np.asarray(item["sem"], dtype=np.float64)
                layers = np.arange(1, mean.size + 1)
                axis.plot(
                    layers,
                    mean,
                    color=colors[label],
                    linewidth=1.8,
                    label=label_names[label],
                )
                axis.fill_between(
                    layers,
                    mean - sem,
                    mean + sem,
                    color=colors[label],
                    alpha=0.16,
                )
            if legend_handles is None:
                legend_handles = axis.get_legend_handles_labels()
            if row_index == 0:
                axis.set_title(method, fontsize=12)
            if column_index == 0:
                axis.set_ylabel(f"{model}\nMean OT risk", fontsize=10)
            axis.set_xlabel("Layer")
            axis.grid(True, alpha=0.22)
            hall_n = by_label[LABEL_HALLUCINATION]["n"]
            non_n = by_label[LABEL_NON_HALLUCINATION]["n"]
            axis.text(
                0.02,
                0.04,
                f"n_H={hall_n}, n_N={non_n}",
                transform=axis.transAxes,
                fontsize=8,
                alpha=0.8,
            )

    if legend_handles is not None:
        handles, labels = legend_handles
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
    figure.suptitle(
        f"{dataset_label}: target-gate risk | sqrt-cosine h_pre OT; "
        "softmax over vocabulary; relative gates use 1.4826 x MAD; "
        "legacy_prob has no MAD",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, stats: dict, dataset_label: str) -> None:
    fieldnames = [
        "model",
        "dataset",
        "method",
        "feature_key",
        "layer",
        "n_hallucination",
        "n_non_hallucination",
        "hallucination_mean",
        "hallucination_sem",
        "non_hallucination_mean",
        "non_hallucination_sem",
        "diff_h_minus_non",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model, by_method in stats.items():
            for method, by_label in by_method.items():
                hall = by_label[LABEL_HALLUCINATION]
                non = by_label[LABEL_NON_HALLUCINATION]
                hall_mean = np.asarray(hall["mean"], dtype=np.float64)
                hall_sem = np.asarray(hall["sem"], dtype=np.float64)
                non_mean = np.asarray(non["mean"], dtype=np.float64)
                non_sem = np.asarray(non["sem"], dtype=np.float64)
                for layer_index in range(hall_mean.size):
                    writer.writerow(
                        {
                            "model": model,
                            "dataset": dataset_label,
                            "method": method,
                            "feature_key": METHOD_KEYS[method],
                            "layer": layer_index + 1,
                            "n_hallucination": hall["n"],
                            "n_non_hallucination": non["n"],
                            "hallucination_mean": f"{hall_mean[layer_index]:.8f}",
                            "hallucination_sem": f"{hall_sem[layer_index]:.8f}",
                            "non_hallucination_mean": f"{non_mean[layer_index]:.8f}",
                            "non_hallucination_sem": f"{non_sem[layer_index]:.8f}",
                            "diff_h_minus_non": (
                                f"{hall_mean[layer_index] - non_mean[layer_index]:.8f}"
                            ),
                        }
                    )


if __name__ == "__main__":
    main()
