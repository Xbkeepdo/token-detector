#!/usr/bin/env python3
"""Plot LLaVA FFN injection diagnostics by hallucination label."""

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


FEATURES = [
    (
        "ffn_attn_dominance",
        "dgst_t_ffn_attn_dominance_per_layer",
        "FFN/Attention Dominance",
        "llava_ffn_attn_dominance_by_label",
        "Positive values mean the FFN update norm is larger than the attention update norm.",
    ),
    (
        "ffn_evidence_orthogonal_dose",
        "dgst_t_ffn_evidence_orthogonal_dose_per_layer",
        "Evidence-Orthogonal FFN Dose",
        "llava_ffn_evidence_orthogonal_dose_by_label",
        "Higher values mean more FFN update magnitude lies outside the visual evidence subspace.",
    ),
    (
        "ffn_logit_lift",
        "dgst_t_ffn_logit_lift_per_layer",
        "FFN Logit Lift",
        "llava_ffn_logit_lift_by_label",
        "Positive values mean the FFN update linearly increases the target object-token logit.",
    ),
]

LABEL_NAMES = {
    0: "non_hallucination",
    1: "hallucination",
}

LABEL_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-pkl",
        default="outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/features.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_ffn_injection_diagnostic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_pkl(args.features_pkl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries = []
    for feature_name, feature_key, title, stem, explanation in FEATURES:
        rows = summarize_feature(features, feature_name, feature_key)
        if not rows:
            raise KeyError(f"No rows found for feature key {feature_key!r}.")
        all_rows.extend(rows)
        write_csv(output_dir / f"{stem}.csv", rows)
        plot_feature(rows, title, output_dir / stem)
        summaries.append(summary_for_feature(rows, feature_name, title, explanation))

    write_csv(output_dir / "llava_ffn_injection_3features_by_label.csv", all_rows)
    plot_combined(all_rows, output_dir / "llava_ffn_injection_3features_by_label")
    write_summary(output_dir / "llava_ffn_injection_summary.md", summaries)


def summarize_feature(features: list[dict], feature_name: str, feature_key: str) -> list[dict[str, object]]:
    max_layers = 0
    values_by_label: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for feat in features:
        label = int(feat.get("label", -1))
        if label not in values_by_label:
            continue
        values = feat.get(feature_key)
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        values_by_label[label].append(arr)
        max_layers = max(max_layers, int(arr.size))

    rows: list[dict[str, object]] = []
    for layer_idx in range(max_layers):
        for label, arrays in values_by_label.items():
            layer_values = np.asarray(
                [arr[layer_idx] for arr in arrays if int(arr.size) > layer_idx],
                dtype=np.float64,
            )
            if layer_values.size == 0:
                continue
            rows.append(
                {
                    "layer": layer_idx + 1,
                    "label": LABEL_NAMES[label],
                    "mean": float(layer_values.mean()),
                    "std": float(layer_values.std(ddof=1)) if layer_values.size > 1 else 0.0,
                    "count": int(layer_values.size),
                    "feature": feature_name,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["layer", "label", "mean", "std", "count", "feature"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature(rows: list[dict[str, object]], title: str, output_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    draw_feature_axis(ax, rows, title)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_combined(rows: list[dict[str, object]], output_stem: Path) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(8.5, 10.5), sharex=True)
    for ax, (feature_name, _feature_key, title, _stem, _explanation) in zip(axes, FEATURES):
        feature_rows = [row for row in rows if row["feature"] == feature_name]
        draw_feature_axis(ax, feature_rows, title)
    axes[-1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def draw_feature_axis(ax, rows: list[dict[str, object]], title: str) -> None:
    for label_id, label_name in LABEL_NAMES.items():
        label_rows = sorted(
            [row for row in rows if row["label"] == label_name],
            key=lambda item: int(item["layer"]),
        )
        if not label_rows:
            continue
        layers = np.asarray([int(row["layer"]) for row in label_rows], dtype=np.int32)
        means = np.asarray([float(row["mean"]) for row in label_rows], dtype=np.float64)
        stds = np.asarray([float(row["std"]) for row in label_rows], dtype=np.float64)
        color = LABEL_COLORS[label_id]
        ax.plot(layers, means, label=label_name, color=color, linewidth=2.0)
        ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Mean")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def summary_for_feature(
    rows: list[dict[str, object]],
    feature_name: str,
    title: str,
    explanation: str,
) -> dict[str, object]:
    hall = series_for_label(rows, "hallucination")
    non = series_for_label(rows, "non_hallucination")
    common_layers = sorted(set(hall) & set(non))
    if not common_layers:
        raise ValueError(f"No common label layers for {feature_name}.")
    hall_values = np.asarray([hall[layer] for layer in common_layers], dtype=np.float64)
    non_values = np.asarray([non[layer] for layer in common_layers], dtype=np.float64)
    diffs = hall_values - non_values
    peak_idx = int(np.argmax(np.abs(diffs)))
    return {
        "feature": feature_name,
        "title": title,
        "hall_avg": float(hall_values.mean()),
        "non_avg": float(non_values.mean()),
        "diff_avg": float(diffs.mean()),
        "peak_abs_layer": int(common_layers[peak_idx]),
        "peak_h_minus_n_gap": float(diffs[peak_idx]),
        "explanation": explanation,
    }


def series_for_label(rows: list[dict[str, object]], label: str) -> dict[int, float]:
    return {
        int(row["layer"]): float(row["mean"])
        for row in rows
        if row["label"] == label
    }


def write_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# LLaVA FFN Injection Diagnostic Summary",
        "",
        "| Feature | Hall avg | Non avg | H-N diff avg | Peak abs layer | Peak H-N gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {feature} | {hall_avg:.6g} | {non_avg:.6g} | {diff_avg:.6g} | "
            "{peak_abs_layer} | {peak_h_minus_n_gap:.6g} |".format(**item)
        )
    lines.extend(["", "## Direction Notes", ""])
    for item in summaries:
        lines.append(f"- `{item['feature']}`: {item['explanation']}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
