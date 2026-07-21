#!/usr/bin/env python3
"""Plot VV/VP transport-risk curves for two targets and three costs."""

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


TARGETS = (
    ("gaussian", "hpre_raw_logit_gauss", "Raw-logit Gaussian"),
    ("relative_vll", "hpre_raw_logit_relative_vll", "Relative-VLL"),
)
SCOPES = (
    ("vv", "", "VV", "-"),
    ("vp", "vp_", "VP", "--"),
)
COSTS = (
    ("sqrt_matched", "risk_sqrt_hpre_per_layer", r"$\sqrt{(1-\cos(h^{pre}))/2}$"),
    ("cosine_matched", "risk_cosine_hpre_per_layer", r"$1-\cos(h^{pre})$"),
    (
        "sqrt_stateupd_alpha05",
        "risk_sqrt_stateupd_alpha05_per_layer",
        r"$0.5d_{state}+0.5d_{FFN}$",
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
        "--multiply-fad",
        action="store_true",
        help=(
            "Multiply every risk curve layerwise by FFAD "
            "(dgst_t_ffn_attn_dominance_per_layer)."
        ),
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Optional output filename stem without an extension.",
    )
    return parser.parse_args()


def feature_key(method: str, scope_prefix: str, cost_suffix: str) -> str:
    return f"dgst_t_{scope_prefix}{method}_{cost_suffix}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.features, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError("Feature artifact is empty.")

    keys = {
        (target_slug, scope_slug, cost_slug): feature_key(
            method, scope_prefix, cost_suffix
        )
        for target_slug, method, _target_label in TARGETS
        for scope_slug, scope_prefix, _scope_label, _linestyle in SCOPES
        for cost_slug, cost_suffix, _cost_label in COSTS
    }
    layer_count = None
    accumulators = {}
    class_counts = {0: 0, 1: 0}
    image_ids = set()
    fad_key = "dgst_t_ffn_attn_dominance_per_layer"
    for row in rows:
        label = int(row["label"])
        if label not in class_counts:
            raise ValueError(f"Unexpected label: {label}")
        class_counts[label] += 1
        image_ids.add(int(row["image_id"]))
        fad_values = None
        if args.multiply_fad:
            if fad_key not in row:
                raise KeyError(f"Missing required FFAD feature: {fad_key}")
            fad_values = np.asarray(row[fad_key], dtype=np.float32).reshape(-1)
            if fad_values.size == 0 or not np.isfinite(fad_values).all():
                raise ValueError(f"Empty or non-finite values in {fad_key}")
        for spec, key in keys.items():
            if key not in row:
                raise KeyError(f"Missing required feature: {key}")
            values = np.asarray(
                row[key],
                dtype=np.float32 if args.multiply_fad else np.float64,
            ).reshape(-1)
            if layer_count is None:
                layer_count = int(values.size)
            if values.size != layer_count:
                raise ValueError(
                    f"Inconsistent layer count for {key}: {values.size} vs {layer_count}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values in {key}")
            if fad_values is not None:
                if fad_values.size != values.size:
                    raise ValueError(
                        f"FFAD and risk must have the same layer count: "
                        f"{fad_values.size} vs {values.size}"
                    )
                values = (fad_values * values).astype(np.float32)
            values = values.astype(np.float64)
            state = accumulators.setdefault(
                (spec, label),
                {
                    "sum": np.zeros(layer_count, dtype=np.float64),
                    "sum_sq": np.zeros(layer_count, dtype=np.float64),
                    "count": 0,
                },
            )
            state["sum"] += values
            state["sum_sq"] += values * values
            state["count"] += 1

    stats = {}
    for spec in keys:
        stats[spec] = {}
        for label, class_slug, _class_label, _color in CLASSES:
            state = accumulators[(spec, label)]
            count = int(state["count"])
            mean = state["sum"] / count
            variance = np.maximum(state["sum_sq"] / count - mean * mean, 0.0)
            sem = np.sqrt(variance) / math.sqrt(count)
            stats[spec][class_slug] = {"mean": mean, "sem": sem, "count": count}

    layers = np.arange(1, layer_count + 1)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.5), sharex=True)
    for row_index, (target_slug, _method, target_label) in enumerate(TARGETS):
        for column_index, (cost_slug, _cost_suffix, cost_label) in enumerate(COSTS):
            axis = axes[row_index, column_index]
            for scope_slug, _scope_prefix, scope_label, linestyle in SCOPES:
                spec = (target_slug, scope_slug, cost_slug)
                for _label, class_slug, class_label, color in CLASSES:
                    mean = stats[spec][class_slug]["mean"]
                    sem = stats[spec][class_slug]["sem"]
                    line_label = f"{scope_label} {class_label}"
                    axis.plot(
                        layers,
                        mean,
                        color=color,
                        linestyle=linestyle,
                        linewidth=2.0,
                        label=line_label,
                    )
                    axis.fill_between(
                        layers,
                        mean - 1.96 * sem,
                        mean + 1.96 * sem,
                        color=color,
                        alpha=0.06,
                        linewidth=0,
                    )
            axis.set_title(f"{target_label}\nCost: {cost_label}", fontsize=12)
            axis.grid(True, alpha=0.25)
            axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
            if column_index == 0:
                axis.set_ylabel(
                    "Mean FFAD × transport risk"
                    if args.multiply_fad
                    else "Mean transport risk"
                )
            if row_index == 1:
                axis.set_xlabel("Decoder layer (1-based)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        (
            f"{args.model_label}: VV/VP FFAD × risk by target construction, "
            "cost, and label"
            if args.multiply_fad
            else (
                f"{args.model_label}: VV/VP risk by target construction, "
                "cost, and label"
            )
        ),
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.925))

    stem = args.stem or (
        "llava_1_5_7b_relative_vll_gaussian_vv_vp_three_cost_"
        + ("fad_x_risk" if args.multiply_fad else "risk")
        + "_by_label"
    )
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "target",
                "scope",
                "cost",
                "layer",
                "hall_mean",
                "hall_sem",
                "real_mean",
                "real_sem",
                "hall_minus_real",
            ),
        )
        writer.writeheader()
        for target_slug, _method, _target_label in TARGETS:
            for scope_slug, _scope_prefix, _scope_label, _linestyle in SCOPES:
                for cost_slug, _cost_suffix, _cost_label in COSTS:
                    spec = (target_slug, scope_slug, cost_slug)
                    hall = stats[spec]["hallucination"]
                    real = stats[spec]["real"]
                    for layer_index in range(layer_count):
                        writer.writerow(
                            {
                                "target": target_slug,
                                "scope": scope_slug,
                                "cost": cost_slug,
                                "layer": layer_index + 1,
                                "hall_mean": float(hall["mean"][layer_index]),
                                "hall_sem": float(hall["sem"][layer_index]),
                                "real_mean": float(real["mean"][layer_index]),
                                "real_sem": float(real["sem"][layer_index]),
                                "hall_minus_real": float(
                                    hall["mean"][layer_index] - real["mean"][layer_index]
                                ),
                            }
                        )

    summaries = []
    for target_slug, _method, _target_label in TARGETS:
        for scope_slug, _scope_prefix, _scope_label, _linestyle in SCOPES:
            for cost_slug, _cost_suffix, _cost_label in COSTS:
                spec = (target_slug, scope_slug, cost_slug)
                hall_mean = stats[spec]["hallucination"]["mean"]
                real_mean = stats[spec]["real"]["mean"]
                gap = hall_mean - real_mean
                peak_index = int(np.argmax(np.abs(gap)))
                summaries.append(
                    {
                        "target": target_slug,
                        "scope": scope_slug,
                        "cost": cost_slug,
                        "hall_all_layer_mean": float(hall_mean.mean()),
                        "real_all_layer_mean": float(real_mean.mean()),
                        "hall_minus_real_all_layer_mean": float(gap.mean()),
                        "peak_abs_gap_layer": peak_index + 1,
                        "peak_abs_gap": float(gap[peak_index]),
                    }
                )

    json_path = output_dir / f"{stem}.json"
    payload = {
        "features": str(Path(args.features).resolve()),
        "transform": (
            "ffn_fad * risk" if args.multiply_fad else "risk"
        ),
        "ffn_fad_feature": fad_key if args.multiply_fad else None,
        "rows": len(rows),
        "images": len(image_ids),
        "layers": layer_count,
        "class_counts": {"hallucination": class_counts[0], "real": class_counts[1]},
        "summaries": summaries,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_dir / "summary.md"
    lines = [
        (
            "# Relative-VLL / Gaussian × VV / VP × three-cost FFAD × risk curves"
            if args.multiply_fad
            else "# Relative-VLL / Gaussian × VV / VP × three-cost risk curves"
        ),
        "",
        f"- Rows: {len(rows):,}; images: {len(image_ids):,}; layers: {layer_count}",
        f"- Hallucination: {class_counts[0]:,}; Real: {class_counts[1]:,}",
        (
            f"- Transform: layerwise `{fad_key} * risk` (float32 product)"
            if args.multiply_fad
            else "- Transform: raw risk"
        ),
        "",
        "| Target | Scope | Cost | Hall mean | Real mean | Hall-Real | Peak layer | Peak gap |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {target} | {scope} | {cost} | {hall_all_layer_mean:.6f} | "
            "{real_all_layer_mean:.6f} | {hall_minus_real_all_layer_mean:+.6f} | "
            "{peak_abs_gap_layer} | {peak_abs_gap:+.6f} |".format(**item)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
