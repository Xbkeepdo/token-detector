#!/usr/bin/env python3
"""Plot Gaussian/Relative-VLL VV/VP mass-times-cosine curves by label."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = (
    (
        "gaussian",
        "vv",
        "Raw-logit Gaussian — VV",
        "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "gaussian",
        "vp",
        "Raw-logit Gaussian — VP",
        "dgst_t_vp_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "relative_vll",
        "vv",
        "Relative-VLL — VV",
        "dgst_t_hpre_raw_logit_relative_vll_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "relative_vll",
        "vp",
        "Relative-VLL — VP",
        "dgst_t_vp_hpre_raw_logit_relative_vll_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
)
CLASSES = (
    (0, "hallucination", "Hallucination", "#d62728"),
    (1, "real", "Real", "#1f77b4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="LLaVA-1.5-7B")
    parser.add_argument(
        "--stem",
        default="llava_1_5_7b_relative_vll_gaussian_vv_vp_mass_x_cosine_by_label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.features).open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty feature list")

    stats, layer_count, image_count = collect_stats(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / args.stem
    plot_curves(stats, layer_count, stem, args.model_label)
    write_csv(stats, layer_count, stem.with_suffix(".csv"))
    summary = build_summary(
        stats,
        layer_count,
        rows=len(rows),
        images=image_count,
        features_path=Path(args.features),
    )
    stem.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, output_dir / "summary.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for suffix in (".png", ".pdf", ".csv", ".json"):
        print(f"Wrote {stem.with_suffix(suffix)}")
    print(f"Wrote {output_dir / 'summary.md'}")


def collect_stats(
    rows: list[dict],
) -> tuple[dict[tuple[str, str], dict[str, dict[str, object]]], int, int]:
    accumulators: dict[tuple[tuple[str, str], int], dict[str, object]] = {}
    layer_count: int | None = None
    image_ids: set[int] = set()
    for row in rows:
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError(f"Unexpected label: {label}")
        image_ids.add(int(row["image_id"]))
        for target, scope, _title, key in FEATURES:
            if key not in row:
                raise KeyError(f"Missing required feature: {key}")
            values = np.asarray(row[key], dtype=np.float64).reshape(-1)
            if layer_count is None:
                layer_count = int(values.size)
            if values.size != layer_count:
                raise ValueError(
                    f"Inconsistent layer count for {key}: {values.size} vs {layer_count}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {key}")
            state = accumulators.setdefault(
                ((target, scope), label),
                {
                    "sum": np.zeros(layer_count, dtype=np.float64),
                    "sum_sq": np.zeros(layer_count, dtype=np.float64),
                    "count": 0,
                },
            )
            state["sum"] += values
            state["sum_sq"] += values * values
            state["count"] += 1

    if layer_count is None:
        raise ValueError("No feature rows were collected")
    stats: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for target, scope, _title, _key in FEATURES:
        spec = (target, scope)
        stats[spec] = {}
        for label, class_slug, _class_title, _color in CLASSES:
            state = accumulators[(spec, label)]
            count = int(state["count"])
            mean = state["sum"] / count
            variance = np.maximum(state["sum_sq"] / count - mean * mean, 0.0)
            sem = np.sqrt(variance) / math.sqrt(count)
            stats[spec][class_slug] = {
                "count": count,
                "mean": mean,
                "sem": sem,
            }
    return stats, layer_count, len(image_ids)


def plot_curves(
    stats: dict[tuple[str, str], dict[str, dict[str, object]]],
    layer_count: int,
    stem: Path,
    model_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True)
    layers = np.arange(1, layer_count + 1)
    for axis, (target, scope, title, _key) in zip(axes.flat, FEATURES):
        item = stats[(target, scope)]
        for _label, class_slug, class_title, color in CLASSES:
            mean = np.asarray(item[class_slug]["mean"])
            sem = np.asarray(item[class_slug]["sem"])
            ci95 = 1.96 * sem
            axis.plot(
                layers,
                mean,
                color=color,
                linewidth=2.1,
                label=f"{class_title} (n={item[class_slug]['count']})",
            )
            axis.fill_between(
                layers,
                mean - ci95,
                mean + ci95,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_ylabel("Mean mass × cosine")
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel("Decoder layer (1-based)")
    fig.suptitle(
        f"{model_label}: target top-k mass × cosine by target, scope, and label",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(
    stats: dict[tuple[str, str], dict[str, dict[str, object]]],
    layer_count: int,
    path: Path,
) -> None:
    fields = (
        "target",
        "scope",
        "layer",
        "hall_n",
        "hall_mean",
        "hall_sem",
        "real_n",
        "real_mean",
        "real_sem",
        "hall_minus_real",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target, scope, _title, _key in FEATURES:
            item = stats[(target, scope)]
            hall = item["hallucination"]
            real = item["real"]
            for index in range(layer_count):
                writer.writerow({
                    "target": target,
                    "scope": scope,
                    "layer": index + 1,
                    "hall_n": hall["count"],
                    "hall_mean": float(hall["mean"][index]),
                    "hall_sem": float(hall["sem"][index]),
                    "real_n": real["count"],
                    "real_mean": float(real["mean"][index]),
                    "real_sem": float(real["sem"][index]),
                    "hall_minus_real": float(
                        hall["mean"][index] - real["mean"][index]
                    ),
                })


def build_summary(
    stats: dict[tuple[str, str], dict[str, dict[str, object]]],
    layer_count: int,
    *,
    rows: int,
    images: int,
    features_path: Path,
) -> dict[str, object]:
    summaries = []
    for target, scope, title, key in FEATURES:
        item = stats[(target, scope)]
        hall = np.asarray(item["hallucination"]["mean"])
        real = np.asarray(item["real"]["mean"])
        gap = hall - real
        peak = int(np.argmax(np.abs(gap)))
        summaries.append({
            "target": target,
            "scope": scope,
            "title": title,
            "feature_key": key,
            "hall_count": int(item["hallucination"]["count"]),
            "real_count": int(item["real"]["count"]),
            "hall_all_layer_mean": float(hall.mean()),
            "real_all_layer_mean": float(real.mean()),
            "hall_minus_real_all_layer_mean": float(gap.mean()),
            "peak_abs_gap_layer": peak + 1,
            "peak_abs_gap": float(gap[peak]),
        })
    return {
        "features": str(features_path.resolve()),
        "definition": "target-distribution top-k mass * matched hpre target cosine",
        "top_k": 32,
        "rows": rows,
        "images": images,
        "layers": layer_count,
        "error_band": "95% CI of token-level class mean (1.96 * SEM)",
        "summaries": summaries,
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Gaussian / Relative-VLL × VV / VP mass × cosine curves",
        "",
        f"- Rows: {summary['rows']:,}; images: {summary['images']:,}; layers: {summary['layers']}",
        "- Definition: target-distribution top-32 mass × matched hpre target cosine.",
        "- Labels: 0=hallucination; 1=real.",
        "- Curves: token-level class mean; band: 95% CI of the mean.",
        "",
        "| Target | Scope | Hall mean | Real mean | Hall-Real | Peak layer | Peak gap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["summaries"]:
        lines.append(
            "| {target} | {scope} | {hall_all_layer_mean:.6f} | "
            "{real_all_layer_mean:.6f} | {hall_minus_real_all_layer_mean:+.6f} | "
            "{peak_abs_gap_layer} | {peak_abs_gap:+.6f} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
