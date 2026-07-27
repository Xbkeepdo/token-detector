#!/usr/bin/env python3
"""Plot VV/VP hpre-risk and FFAD-weighted variants by raw label."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_feature_sets import feature_block
from utils.io_utils import load_pkl


LABELS = {0: "Hallucination", 1: "Non-hallucination"}
COLORS = {0: "#D55E00", 1: "#0072B2"}
FEATURE_SPECS = (
    (
        "vv_hpre_risk",
        "hpre_raw_logit_gauss_risk_sqrt_matched_state",
        "VV hpre-risk",
    ),
    (
        "vp_hpre_risk",
        "vp_hpre_raw_logit_gauss_risk_sqrt_matched_state",
        "VP hpre-risk",
    ),
    (
        "ffad_x_vv_hpre_risk",
        "__product__:ffn_fad:hpre_raw_logit_gauss_risk_sqrt_matched_state",
        "FFAD × VV hpre-risk",
    ),
    (
        "ffad_x_vp_hpre_risk",
        "__product__:ffn_fad:vp_hpre_raw_logit_gauss_risk_sqrt_matched_state",
        "FFAD × VP hpre-risk",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stem",
        default="vv_vp_hpre_risk_ffad_by_label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = load_pkl(args.features_pkl)
    if not isinstance(features, list) or not features:
        raise ValueError(f"Expected a non-empty feature list: {args.features_pkl}")

    stats = {
        slug: summarize(features, expression)
        for slug, expression, _title in FEATURE_SPECS
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / str(args.stem)
    plot_combined(stats, stem)
    write_csv(stats, stem.with_suffix(".csv"))
    summary = build_summary(
        stats,
        features_path=Path(args.features_pkl),
    )
    stem.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, stem.with_name(f"{stem.name}_summary.md"))

    for suffix in (".png", ".pdf", ".csv", ".json"):
        print(suffix[1:], stem.with_suffix(suffix).resolve())
    print(
        "summary",
        stem.with_name(f"{stem.name}_summary.md").resolve(),
    )
    for slug, _expression, title in FEATURE_SPECS:
        item = summary["features"][slug]
        print(title)
        print("  hall_mean", item["hallucination_layer_mean"])
        print("  real_mean", item["non_hallucination_layer_mean"])
        print("  hall_minus_real", item["hall_minus_real_layer_mean"])
        print("  peak_abs_diff_layer", item["peak_abs_diff_layer"])
        print("  peak_abs_diff", item["peak_abs_diff"])


def summarize(
    features: Sequence[Mapping[str, Any]],
    expression: str,
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for row in features:
        label = row.get("label")
        if label not in (0, 1):
            continue
        vector = np.asarray(
            feature_block(dict(row), expression), dtype=np.float64
        ).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(
                f"{expression} has an empty or non-finite vector for "
                f"image_id={row.get('image_id')}"
            )
        grouped[int(label)].append(vector)
    if not grouped[0] or not grouped[1]:
        raise ValueError(f"{expression} requires both raw labels 0 and 1")

    result: dict[int, dict[str, Any]] = {}
    for label, values in grouped.items():
        widths = {int(value.size) for value in values}
        if len(widths) != 1:
            raise ValueError(
                f"{expression} has inconsistent layer widths: {sorted(widths)}"
            )
        matrix = np.stack(values)
        std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
        sem = std / np.sqrt(matrix.shape[0])
        result[label] = {
            "n": int(matrix.shape[0]),
            "mean": matrix.mean(axis=0),
            "std": std,
            "sem": sem,
            "ci95": 1.96 * sem,
        }
    return result


def plot_combined(
    stats: Mapping[str, Mapping[int, Mapping[str, Any]]],
    stem: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5), sharex=True)
    for ax, (slug, _expression, title) in zip(axes.flat, FEATURE_SPECS):
        item = stats[slug]
        layer_count = int(np.asarray(item[0]["mean"]).size)
        layers = np.arange(1, layer_count + 1)
        for label in (0, 1):
            mean = np.asarray(item[label]["mean"])
            ci95 = np.asarray(item[label]["ci95"])
            ax.plot(
                layers,
                mean,
                color=COLORS[label],
                linewidth=2.1,
                label=f"{LABELS[label]} (n={item[label]['n']})",
            )
            ax.fill_between(
                layers,
                mean - ci95,
                mean + ci95,
                color=COLORS[label],
                alpha=0.16,
                linewidth=0,
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.45)
        ax.set_title(title)
        ax.set_ylabel("Mean feature value")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.24)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Decoder layer (1-based)")
    fig.suptitle(
        "Qwen2.5-VL-7B: VV/VP hpre-risk and FFAD-weighted curves",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(
    stats: Mapping[str, Mapping[int, Mapping[str, Any]]],
    path: Path,
) -> None:
    fields = [
        "feature",
        "layer",
        "hallucination_n",
        "hallucination_mean",
        "hallucination_std",
        "hallucination_sem",
        "hallucination_ci95",
        "non_hallucination_n",
        "non_hallucination_mean",
        "non_hallucination_std",
        "non_hallucination_sem",
        "non_hallucination_ci95",
        "hall_minus_real",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for slug, _expression, _title in FEATURE_SPECS:
            item = stats[slug]
            layer_count = int(np.asarray(item[0]["mean"]).size)
            for index in range(layer_count):
                writer.writerow(
                    {
                        "feature": slug,
                        "layer": index + 1,
                        "hallucination_n": item[0]["n"],
                        "hallucination_mean": float(item[0]["mean"][index]),
                        "hallucination_std": float(item[0]["std"][index]),
                        "hallucination_sem": float(item[0]["sem"][index]),
                        "hallucination_ci95": float(item[0]["ci95"][index]),
                        "non_hallucination_n": item[1]["n"],
                        "non_hallucination_mean": float(item[1]["mean"][index]),
                        "non_hallucination_std": float(item[1]["std"][index]),
                        "non_hallucination_sem": float(item[1]["sem"][index]),
                        "non_hallucination_ci95": float(item[1]["ci95"][index]),
                        "hall_minus_real": float(
                            item[0]["mean"][index] - item[1]["mean"][index]
                        ),
                    }
                )


def build_summary(
    stats: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    features_path: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "features_path": str(features_path.resolve()),
        "raw_label_semantics": {"0": "hallucination", "1": "non-hallucination"},
        "error_band": "95% CI of the per-class token-level mean (1.96 * SEM)",
        "layer_indexing": "1-based decoder layers",
        "features": {},
    }
    for slug, expression, title in FEATURE_SPECS:
        item = stats[slug]
        hall = np.asarray(item[0]["mean"])
        real = np.asarray(item[1]["mean"])
        diff = hall - real
        peak = int(np.argmax(np.abs(diff)))
        output["features"][slug] = {
            "title": title,
            "training_expression": expression,
            "layer_count": int(hall.size),
            "n_hallucination": int(item[0]["n"]),
            "n_non_hallucination": int(item[1]["n"]),
            "hallucination_layer_mean": float(hall.mean()),
            "non_hallucination_layer_mean": float(real.mean()),
            "hall_minus_real_layer_mean": float(diff.mean()),
            "peak_abs_diff_layer": peak + 1,
            "peak_abs_diff": float(diff[peak]),
        }
    return output


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# VV/VP hpre-risk 与 FFAD 乘积逐层曲线",
        "",
        "原始标签：`0=幻觉`，`1=非幻觉/real`；阴影为 class 内 token-level 均值的 95% CI。",
        "",
        "| 特征 | 幻觉 n | 非幻觉 n | 幻觉逐层均值 | 非幻觉逐层均值 | Hall-Real | 最大绝对差层 | 该层 Hall-Real |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, _expression, _title in FEATURE_SPECS:
        item = summary["features"][slug]
        lines.append(
            "| {title} | {n_hallucination} | {n_non_hallucination} | "
            "{hallucination_layer_mean:.6f} | "
            "{non_hallucination_layer_mean:.6f} | "
            "{hall_minus_real_layer_mean:.6f} | {peak_abs_diff_layer} | "
            "{peak_abs_diff:.6f} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
