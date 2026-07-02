#!/usr/bin/env python3
"""Plot two per-layer feature curves by hallucination label."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


FEATURE_KEYS = {
    "risk": "dgst_t_transport_risk_per_layer",
    "risk_capped_topmass_085": "dgst_t_transport_risk_capped_topmass_085_per_layer",
    "risk_relative_vll": "dgst_t_transport_risk_relative_vll_per_layer",
    "risk_relative_vll_capped_topmass_085": "dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer",
    "risk_visual_prompt_relative_vll": "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer",
    "risk_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_hidden_cosine": "dgst_t_target_visual_hidden_cosine_per_layer",
    "target_visual_prompt_hidden_cosine": "dgst_t_target_visual_prompt_hidden_cosine_per_layer",
    "target_visual_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer",
    "target_visual_hidden_cosine_relative_vll": "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer",
    "context_confidence": "dgst_t_context_confidence_per_layer",
    "context_confidence_max_prompt": "dgst_t_context_confidence_max_prompt_per_layer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--features",
        nargs=2,
        default=["target_visual_hidden_cosine", "target_visual_prompt_hidden_cosine"],
        help="Two feature aliases or raw feature keys to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["visual-only top-k", "visual+prompt top-k"],
        help="Display labels for the two feature curves.",
    )
    parser.add_argument("--name", default=None, help="Output filename stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = os.path.join(args.output_dir, "features.pkl")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    features = load_pkl(feature_path)
    keys = [_resolve_key(item) for item in args.features]
    _ensure_keys(features, keys)

    grouped = [_group_by_label(features, key) for key in keys]
    stats = [_stats_by_label(group) for group in grouped]

    stem = args.name or f"{args.model}_{args.features[0]}_vs_{args.features[1]}_by_label"
    png_path = os.path.join(results_dir, f"{stem}.png")
    pdf_path = os.path.join(results_dir, f"{stem}.pdf")
    csv_path = os.path.join(results_dir, f"{stem}.csv")

    _plot(stats, args.labels, png_path, pdf_path)
    _write_csv(stats, args.labels, csv_path)

    print("png", os.path.abspath(png_path))
    print("pdf", os.path.abspath(pdf_path))
    print("csv", os.path.abspath(csv_path))
    for label, item in zip(args.labels, stats):
        diff = item[1]["mean"] - item[0]["mean"]
        peak = int(np.argmax(np.abs(diff)))
        print(label)
        print("  n_hallucination", item[1]["n"])
        print("  n_non_hallucination", item[0]["n"])
        print("  hallucination_mean_avg", float(item[1]["mean"].mean()))
        print("  non_hallucination_mean_avg", float(item[0]["mean"].mean()))
        print("  diff_avg", float(diff.mean()))
        print("  peak_abs_diff_layer", peak)
        print("  peak_abs_diff", float(diff[peak]))


def _resolve_key(value: str) -> str:
    return FEATURE_KEYS.get(value, value)


def _ensure_keys(features: list[dict], keys: list[str]) -> None:
    missing = [key for key in keys if not any(key in feat for feat in features)]
    if missing:
        raise KeyError(
            "Missing feature key(s): "
            + ", ".join(missing)
            + ". Re-run feature extraction if these are newly added features."
        )


def _group_by_label(features: list[dict], key: str) -> dict[int, np.ndarray]:
    curves = {0: [], 1: []}
    for feat in features:
        label = feat.get("label")
        if label not in (0, 1) or key not in feat:
            continue
        values = np.asarray(feat[key], dtype=np.float32).reshape(-1)
        if values.size:
            curves[int(label)].append(values)
    if not curves[0] or not curves[1]:
        raise ValueError(f"Feature {key!r} needs both label=0 and label=1 rows.")
    return {label: np.stack(items, axis=0) for label, items in curves.items()}


def _stats_by_label(grouped: dict[int, np.ndarray]) -> dict[int, dict]:
    result = {}
    for label, values in grouped.items():
        result[label] = {
            "n": int(values.shape[0]),
            "mean": values.mean(axis=0),
            "sem": values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]),
        }
    return result


def _plot(stats: list[dict[int, dict]], labels: list[str], png_path: str, pdf_path: str) -> None:
    layers = np.arange(stats[0][0]["mean"].shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), sharex=True)
    colors = {1: "#d55e00", 0: "#0072b2"}
    class_labels = {1: "Hallucination", 0: "Non-hallucination"}

    for col, (item, feature_label) in enumerate(zip(stats, labels)):
        ax = axes[0][col]
        for class_id in (1, 0):
            mean = item[class_id]["mean"]
            sem = item[class_id]["sem"]
            display = f"{class_labels[class_id]} (n={item[class_id]['n']})"
            ax.plot(layers, mean, color=colors[class_id], linewidth=2.2, label=display)
            ax.fill_between(layers, mean - sem, mean + sem, color=colors[class_id], alpha=0.18)
        ax.set_title(feature_label)
        ax.set_ylabel("Mean feature value")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

        diff = item[1]["mean"] - item[0]["mean"]
        peak = int(np.argmax(np.abs(diff)))
        ax_diff = axes[1][col]
        ax_diff.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)
        ax_diff.plot(layers, diff, color="#333333", linewidth=2.0)
        ax_diff.scatter([peak], [diff[peak]], color="#cc79a7", s=42, zorder=3)
        ax_diff.set_xlabel("Layer")
        ax_diff.set_ylabel("Mean diff\n(label1-label0)")
        ax_diff.grid(True, alpha=0.25)
        ax_diff.set_xticks(layers)

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(stats: list[dict[int, dict]], labels: list[str], csv_path: str) -> None:
    layer_count = int(stats[0][0]["mean"].shape[0])
    header = ["layer"]
    for label in labels:
        stem = _slug(label)
        header.extend(
            [
                f"{stem}_hallucination_mean",
                f"{stem}_hallucination_sem",
                f"{stem}_non_hallucination_mean",
                f"{stem}_non_hallucination_sem",
                f"{stem}_diff_h_minus_non",
            ]
        )

    lines = [",".join(header)]
    for layer in range(layer_count):
        row = [str(layer)]
        for item in stats:
            diff = item[1]["mean"] - item[0]["mean"]
            row.extend(
                [
                    f"{item[1]['mean'][layer]:.8f}",
                    f"{item[1]['sem'][layer]:.8f}",
                    f"{item[0]['mean'][layer]:.8f}",
                    f"{item[0]['sem'][layer]:.8f}",
                    f"{diff[layer]:.8f}",
                ]
            )
        lines.append(",".join(row))
    Path(csv_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(keep).split("_") if part)


if __name__ == "__main__":
    main()
