#!/usr/bin/env python3
"""Plot VP hpre softmax-prob Gaussian risk and EV curves by CHAIR label."""

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
        "risk",
        "VP hpre softmax-prob Gaussian risk",
        "Mean transport risk",
        "dgst_t_vp_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer",
    ),
    (
        "ev_mass_cosine",
        "VP hpre softmax-prob Gaussian EV mass × cosine",
        "Mean target Top-32 mass × cosine",
        "dgst_t_vp_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
)
CLASSES = (
    (0, "hallucination", "Hallucination", "#d55e00"),
    (1, "real", "Non-hallucination", "#0072b2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="LLaVA-1.5-7B")
    parser.add_argument(
        "--stem",
        default="llava_1_5_7b_vp_hpre_softmax_prob_gauss_risk_ev_by_label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    with features_path.open("rb") as handle:
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
        features_path=features_path,
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
) -> tuple[dict[str, dict[str, dict[str, object]]], int, int]:
    accumulators: dict[tuple[str, int], dict[str, object]] = {}
    layer_count: int | None = None
    image_ids: set[int] = set()

    for row in rows:
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError(f"Unexpected label: {label}")
        image_ids.add(int(row["image_id"]))
        for slug, _title, _ylabel, key in FEATURES:
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
                (slug, label),
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

    stats: dict[str, dict[str, dict[str, object]]] = {}
    for slug, _title, _ylabel, _key in FEATURES:
        stats[slug] = {}
        for label, class_slug, _class_title, _color in CLASSES:
            state = accumulators[(slug, label)]
            count = int(state["count"])
            mean = state["sum"] / count
            variance = np.maximum(state["sum_sq"] / count - mean * mean, 0.0)
            sem = np.sqrt(variance) / math.sqrt(count)
            stats[slug][class_slug] = {
                "count": count,
                "mean": mean,
                "sem": sem,
            }
    return stats, layer_count, len(image_ids)


def plot_curves(
    stats: dict[str, dict[str, dict[str, object]]],
    layer_count: int,
    stem: Path,
    model_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    layers = np.arange(1, layer_count + 1)

    for axis, (slug, title, ylabel, _key) in zip(axes, FEATURES):
        item = stats[slug]
        for _label, class_slug, class_title, color in CLASSES:
            mean = np.asarray(item[class_slug]["mean"])
            ci95 = 1.96 * np.asarray(item[class_slug]["sem"])
            axis.plot(
                layers,
                mean,
                color=color,
                linewidth=2.2,
                label=f"{class_title} (n={item[class_slug]['count']})",
            )
            axis.fill_between(
                layers,
                mean - ci95,
                mean + ci95,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        axis.set_title(title)
        axis.set_xlabel("Decoder layer (1-based)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"{model_label}: VP softmax-prob Gaussian features by CHAIR label",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(
    stats: dict[str, dict[str, dict[str, object]]],
    layer_count: int,
    path: Path,
) -> None:
    fields = (
        "feature",
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
        for slug, _title, _ylabel, _key in FEATURES:
            hall = stats[slug]["hallucination"]
            real = stats[slug]["real"]
            for index in range(layer_count):
                writer.writerow(
                    {
                        "feature": slug,
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
                    }
                )


def build_summary(
    stats: dict[str, dict[str, dict[str, object]]],
    layer_count: int,
    *,
    rows: int,
    images: int,
    features_path: Path,
) -> dict[str, object]:
    summaries = []
    for slug, title, _ylabel, key in FEATURES:
        hall = np.asarray(stats[slug]["hallucination"]["mean"])
        real = np.asarray(stats[slug]["real"]["mean"])
        gap = hall - real
        peak = int(np.argmax(np.abs(gap)))
        summaries.append(
            {
                "feature": slug,
                "title": title,
                "feature_key": key,
                "hall_count": int(stats[slug]["hallucination"]["count"]),
                "real_count": int(stats[slug]["real"]["count"]),
                "hall_all_layer_mean": float(hall.mean()),
                "real_all_layer_mean": float(real.mean()),
                "hall_minus_real_all_layer_mean": float(gap.mean()),
                "peak_abs_gap_layer": peak + 1,
                "peak_abs_gap": float(gap[peak]),
            }
        )
    return {
        "features": str(features_path.resolve()),
        "rows": rows,
        "images": images,
        "layers": layer_count,
        "labels": "0=hallucination; 1=real/non-hallucination",
        "error_band": "95% CI of token-level class mean (1.96 * SEM)",
        "summaries": summaries,
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# VP hpre softmax-prob Gaussian curves",
        "",
        f"- Rows: {summary['rows']:,}; images: {summary['images']:,}; layers: {summary['layers']}",
        "- Labels: 0=hallucination; 1=real/non-hallucination.",
        "- Curves: token-level class mean; band: 95% CI of the mean.",
        "",
        "| Feature | Hall mean | Real mean | Hall-Real | Peak layer | Peak gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["summaries"]:
        lines.append(
            "| {feature} | {hall_all_layer_mean:.6f} | "
            "{real_all_layer_mean:.6f} | {hall_minus_real_all_layer_mean:+.6f} | "
            "{peak_abs_gap_layer} | {peak_abs_gap:+.6f} |".format(**item)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
