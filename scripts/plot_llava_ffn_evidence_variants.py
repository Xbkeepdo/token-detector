#!/usr/bin/env python3
"""Plot LLaVA FFN evidence-subspace variants by hallucination label."""

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
        "ffn_eif_fraction_svd",
        "dgst_t_ffn_eif_fraction_svd_per_layer",
        "Raw SVD EIF Fraction",
        "llava_ffn_eif_fraction_svd_by_label",
        "Fraction of FFN update energy outside the raw-SVD visual evidence subspace.",
    ),
    (
        "ffn_eif_dose_svd",
        "dgst_t_ffn_eif_dose_svd_per_layer",
        "Raw SVD EIF Dose",
        "llava_ffn_eif_dose_svd_by_label",
        "FFN evidence-orthogonal update norm under raw-SVD subspace, normalized by h_mid norm.",
    ),
    (
        "ffn_eif_fraction_pca",
        "dgst_t_ffn_eif_fraction_pca_per_layer",
        "Centered PCA EIF Fraction",
        "llava_ffn_eif_fraction_pca_by_label",
        "Fraction of FFN update energy outside the centered-PCA visual evidence subspace.",
    ),
    (
        "ffn_eif_dose_pca",
        "dgst_t_ffn_eif_dose_pca_per_layer",
        "Centered PCA EIF Dose",
        "llava_ffn_eif_dose_pca_by_label",
        "FFN evidence-orthogonal update norm under centered-PCA subspace, normalized by h_mid norm.",
    ),
]

LABEL_NAMES = {0: "non_hallucination", 1: "hallucination"}
LABEL_COLORS = {0: "#1f77b4", 1: "#d62728"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-pkl",
        default="outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly/features.pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_ffn_injection_diagnostic/evidence_variants",
    )
    parser.add_argument(
        "--error-band",
        choices=["std", "sem"],
        default="std",
        help="Error band to draw around the mean curves.",
    )
    parser.add_argument(
        "--stem-suffix",
        default="",
        help="Optional suffix inserted before '_by_label' in output file stems.",
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
        output_stem = output_dir / apply_stem_suffix(stem, args.stem_suffix)
        write_csv(output_stem.with_suffix(".csv"), rows)
        plot_feature(rows, title, output_stem, error_band=args.error_band)
        summaries.append(summary_for_feature(rows, feature_name, title, explanation))

    combined_stem = output_dir / apply_stem_suffix(
        "llava_ffn_evidence_variants_4features_by_label",
        args.stem_suffix,
    )
    write_csv(combined_stem.with_suffix(".csv"), all_rows)
    plot_combined(all_rows, combined_stem, error_band=args.error_band)
    summary_name = (
        "llava_ffn_evidence_variants_summary.md"
        if not args.stem_suffix
        else f"llava_ffn_evidence_variants{args.stem_suffix}_summary.md"
    )
    write_summary(
        output_dir / summary_name,
        summaries,
        error_band=args.error_band,
    )


def apply_stem_suffix(stem: str, suffix: str) -> str:
    if not suffix:
        return stem
    if stem.endswith("_by_label"):
        return f"{stem[:-len('_by_label')]}{suffix}_by_label"
    return f"{stem}{suffix}"


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
                    "sem": float(layer_values.std(ddof=1) / np.sqrt(layer_values.size))
                    if layer_values.size > 1
                    else 0.0,
                    "count": int(layer_values.size),
                    "feature": feature_name,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["layer", "label", "mean", "std", "sem", "count", "feature"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature(
    rows: list[dict[str, object]],
    title: str,
    output_stem: Path,
    *,
    error_band: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    draw_feature_axis(ax, rows, title, error_band=error_band)
    ax.set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_combined(
    rows: list[dict[str, object]],
    output_stem: Path,
    *,
    error_band: str,
) -> None:
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(8.5, 13.5), sharex=True)
    for ax, (feature_name, _feature_key, title, _stem, _explanation) in zip(axes, FEATURES):
        feature_rows = [row for row in rows if row["feature"] == feature_name]
        draw_feature_axis(ax, feature_rows, title, error_band=error_band)
    axes[-1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def draw_feature_axis(
    ax,
    rows: list[dict[str, object]],
    title: str,
    *,
    error_band: str,
) -> None:
    for label_id, label_name in LABEL_NAMES.items():
        label_rows = sorted(
            [row for row in rows if row["label"] == label_name],
            key=lambda item: int(item["layer"]),
        )
        if not label_rows:
            continue
        layers = np.asarray([int(row["layer"]) for row in label_rows], dtype=np.int32)
        means = np.asarray([float(row["mean"]) for row in label_rows], dtype=np.float64)
        errors = np.asarray([float(row[error_band]) for row in label_rows], dtype=np.float64)
        color = LABEL_COLORS[label_id]
        ax.plot(layers, means, label=label_name, color=color, linewidth=2.0)
        ax.fill_between(layers, means - errors, means + errors, color=color, alpha=0.14, linewidth=0)
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
    return {int(row["layer"]): float(row["mean"]) for row in rows if row["label"] == label}


def write_summary(
    path: Path,
    summaries: list[dict[str, object]],
    *,
    error_band: str,
) -> None:
    lines = [
        "# LLaVA FFN Evidence Variants Diagnostic Summary",
        "",
        f"Error band: mean +/- {error_band.upper()}",
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
