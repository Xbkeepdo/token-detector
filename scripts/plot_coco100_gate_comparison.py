#!/usr/bin/env python3
"""Plot matched target-gate risk curves by hallucination label."""

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


METHOD_KEYS = {
    "relative-vll": "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer",
    "softmax-relative-vll": (
        "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer"
    ),
    "legacy_prob": "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer",
}
LABEL_HALLUCINATION = 0
LABEL_NON_HALLUCINATION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot relative-vll, softmax-relative-vll, and legacy_prob risk "
            "curves for hallucinated and non-hallucinated object tokens."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label", default="COCO100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    feature_path = output_dir / "features.pkl"
    features = load_pkl(feature_path)
    if not features:
        raise ValueError(f"No feature rows in {feature_path}")

    stats_by_method = {
        method: _stats_for_key(features, key)
        for method, key in METHOD_KEYS.items()
    }

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = _slug(args.dataset_label)
    stem = results_dir / f"{args.model}_{dataset_slug}_gate_risk_by_label"

    _write_csv(
        stem.with_suffix(".csv"),
        model=args.model,
        dataset_label=str(args.dataset_label),
        stats_by_method=stats_by_method,
    )
    _plot(
        stem=stem,
        model=args.model,
        dataset_label=str(args.dataset_label),
        stats_by_method=stats_by_method,
    )

    print("png", stem.with_suffix(".png").resolve())
    print("pdf", stem.with_suffix(".pdf").resolve())
    print("csv", stem.with_suffix(".csv").resolve())
    for method, by_label in stats_by_method.items():
        hall = by_label[LABEL_HALLUCINATION]
        non = by_label[LABEL_NON_HALLUCINATION]
        gap = hall["mean"] - non["mean"]
        peak = int(np.argmax(np.abs(gap)))
        print(method)
        print("  n_hallucination", hall["n"])
        print("  n_non_hallucination", non["n"])
        print("  diff_avg", float(gap.mean()))
        print("  peak_abs_diff_layer", peak + 1)
        print("  peak_abs_diff", float(gap[peak]))


def _stats_for_key(features: list[dict], key: str) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[np.ndarray]] = {
        LABEL_HALLUCINATION: [],
        LABEL_NON_HALLUCINATION: [],
    }
    for row in features:
        label = row.get("label")
        if label not in grouped or key not in row:
            continue
        values = np.asarray(row[key], dtype=np.float64).reshape(-1)
        if values.size == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {key}, label={label}")
        grouped[int(label)].append(values)

    missing = [label for label, rows in grouped.items() if not rows]
    if missing:
        label_names = {
            LABEL_HALLUCINATION: "hallucination",
            LABEL_NON_HALLUCINATION: "non-hallucination",
        }
        missing_names = ", ".join(label_names[label] for label in missing)
        raise ValueError(f"Feature {key!r} has no rows for: {missing_names}")

    stats: dict[int, dict[str, object]] = {}
    for label, rows in grouped.items():
        lengths = {int(values.size) for values in rows}
        if len(lengths) != 1:
            raise ValueError(
                f"Inconsistent layer counts for {key}, label={label}: {sorted(lengths)}"
            )
        values = np.stack(rows, axis=0)
        sem = (
            values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros(values.shape[1], dtype=np.float64)
        )
        stats[label] = {
            "n": int(values.shape[0]),
            "mean": values.mean(axis=0),
            "sem": sem,
        }

    hall_layers = np.asarray(stats[LABEL_HALLUCINATION]["mean"]).size
    non_layers = np.asarray(stats[LABEL_NON_HALLUCINATION]["mean"]).size
    if hall_layers != non_layers:
        raise ValueError(
            f"Layer count differs by label for {key}: {hall_layers} vs {non_layers}"
        )
    return stats


def _plot(
    *,
    stem: Path,
    model: str,
    dataset_label: str,
    stats_by_method: dict[str, dict[int, dict[str, object]]],
) -> None:
    method_count = len(stats_by_method)
    figure, axes = plt.subplots(
        1,
        method_count,
        figsize=(6.0 * method_count, 4.8),
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

    for axis, (method, by_label) in zip(axes.flat, stats_by_method.items()):
        for label in (LABEL_HALLUCINATION, LABEL_NON_HALLUCINATION):
            item = by_label[label]
            mean = np.asarray(item["mean"], dtype=np.float64)
            sem = np.asarray(item["sem"], dtype=np.float64)
            layers = np.arange(1, mean.size + 1)
            display = f"{label_names[label]} (n={item['n']})"
            axis.plot(
                layers,
                mean,
                color=colors[label],
                linewidth=2.0,
                label=display,
            )
            axis.fill_between(
                layers,
                mean - sem,
                mean + sem,
                color=colors[label],
                alpha=0.18,
            )
        axis.set_title(method)
        axis.set_xlabel("Layer")
        axis.set_ylabel("Mean OT risk (sqrt-cosine h_pre)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        f"{model}: {dataset_label} target-gate risk by label | "
        "softmax over vocabulary; MAD gates use 1.4826 x MAD; legacy_prob has no MAD",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    path: Path,
    *,
    model: str,
    dataset_label: str,
    stats_by_method: dict[str, dict[int, dict[str, object]]],
) -> None:
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
        for method, by_label in stats_by_method.items():
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


def _slug(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    )
    return "_".join(part for part in normalized.split("_") if part) or "dataset"


if __name__ == "__main__":
    main()
