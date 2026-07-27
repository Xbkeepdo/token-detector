#!/usr/bin/env python3
"""Summarize LLaVA full-prompt VP vs user-prompt-only VP."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava_1_5_7b"
BASELINE_DIR = ROOT / "outputs" / MODEL / "COCO500-visualprompt-relativevll-cost-3way"
USERPROMPT_DIR = ROOT / "outputs" / MODEL / "COCO500-visualprompt-userprompt-relativevll-cost-geo"
OUT_DIR = ROOT / "outputs" / "llava_userprompt_vp_comparison"
FIELD = "dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_per_layer"
FEATURE_SETS = [
    "risk_visual_prompt_relative_vll_cost_geo",
    "risk_visual_prompt_relative_vll_cost_geo+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll",
]
VARIANTS = [
    ("full_prompt", "Full prompt VP", BASELINE_DIR),
    ("user_prompt_only", "User prompt only VP", USERPROMPT_DIR),
]
LABEL_HALLUCINATED = 0
LABEL_REAL = 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    curve_rows = []
    stats_by_variant = {}
    for slug, label, directory in VARIANTS:
        features = load_pkl(directory / "features.pkl")
        stats = _layer_stats(features, FIELD)
        stats_by_variant[slug] = stats
        for layer, row in enumerate(stats):
            layer_rows.append(
                {
                    "variant": slug,
                    "variant_label": label,
                    "layer": layer,
                    "n_hallucination": row[LABEL_HALLUCINATED]["n"],
                    "n_non_hallucination": row[LABEL_REAL]["n"],
                    "hallucination_mean": row[LABEL_HALLUCINATED]["mean"],
                    "hallucination_sem": row[LABEL_HALLUCINATED]["sem"],
                    "non_hallucination_mean": row[LABEL_REAL]["mean"],
                    "non_hallucination_sem": row[LABEL_REAL]["sem"],
                    "gap_h_minus_non": row[LABEL_HALLUCINATED]["mean"] - row[LABEL_REAL]["mean"],
                }
            )
        gaps = np.array(
            [
                row[LABEL_HALLUCINATED]["mean"] - row[LABEL_REAL]["mean"]
                for row in stats
            ],
            dtype=np.float64,
        )
        curve_rows.append(
            {
                "variant": slug,
                "variant_label": label,
                "n_hallucination": stats[0][LABEL_HALLUCINATED]["n"],
                "n_non_hallucination": stats[0][LABEL_REAL]["n"],
                "mean_hallucination": float(
                    np.mean([row[LABEL_HALLUCINATED]["mean"] for row in stats])
                ),
                "mean_non_hallucination": float(
                    np.mean([row[LABEL_REAL]["mean"] for row in stats])
                ),
                "mean_gap_h_minus_non": float(gaps.mean()),
                "peak_abs_gap_layer": int(np.argmax(np.abs(gaps))),
                "peak_abs_gap_h_minus_non": float(gaps[np.argmax(np.abs(gaps))]),
                "final_layer_gap_h_minus_non": float(gaps[-1]),
            }
        )

    torch_rows = _load_torch_rows()
    _write_csv(OUT_DIR / "llava_userprompt_vp_layerwise.csv", layer_rows)
    _write_csv(OUT_DIR / "llava_userprompt_vp_curve_summary.csv", curve_rows)
    _write_csv(OUT_DIR / "llava_userprompt_vp_torch_mlp.csv", torch_rows)
    _plot_by_label(stats_by_variant)
    _plot_gap(stats_by_variant)
    _write_target_distribution_summary()
    _write_summary(curve_rows, torch_rows)
    print(f"Wrote {OUT_DIR}")


def _layer_stats(features: list[dict], field: str) -> list[dict[int, dict[str, float]]]:
    arrays_by_label: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for feat in features:
        if field not in feat:
            continue
        label = int(feat.get("label", 0))
        arrays_by_label.setdefault(label, []).append(np.asarray(feat[field], dtype=np.float64))
    layer_count = int(len(next(iter(arrays_by_label[0] or arrays_by_label[1]))))
    stats = []
    for layer in range(layer_count):
        item = {}
        for label in (0, 1):
            values = np.array([arr[layer] for arr in arrays_by_label.get(label, [])], dtype=np.float64)
            sem = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
            item[label] = {"n": int(values.size), "mean": float(values.mean()), "sem": sem}
        stats.append(item)
    return stats


def _load_torch_rows() -> list[dict]:
    rows = []
    for slug, label, directory in VARIANTS:
        path = directory / "results" / f"{MODEL}_selected_feature_sets.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for feature_set in FEATURE_SETS:
            metrics = data.get(feature_set, {}).get("torch_probe")
            if not metrics:
                continue
            rows.append(
                {
                    "variant": slug,
                    "variant_label": label,
                    "feature_set": feature_set,
                    "auc": metrics.get("auc"),
                    "aupr": metrics.get("aupr"),
                    "f1": metrics.get("f1"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "accuracy": metrics.get("accuracy"),
                    "best_epoch": metrics.get("best_epoch"),
                    "num_features": metrics.get("num_features"),
                }
            )
    return rows


def _plot_by_label(stats_by_variant: dict[str, list[dict[int, dict[str, float]]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    labels = {LABEL_HALLUCINATED: "Hallucination", LABEL_REAL: "Non-hallucination"}
    for ax, (slug, display, _directory) in zip(axes, VARIANTS):
        stats = stats_by_variant[slug]
        layers = np.arange(len(stats))
        for label, color in ((LABEL_HALLUCINATED, "#d55e00"), (LABEL_REAL, "#0072b2")):
            means = np.array([row[label]["mean"] for row in stats])
            sems = np.array([row[label]["sem"] for row in stats])
            ax.plot(layers, means, color=color, label=labels[label])
            ax.fill_between(layers, means - sems, means + sems, color=color, alpha=0.18)
        ax.set_title(display)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Risk")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "llava_userprompt_vp_risk_by_label.png", dpi=180)
    fig.savefig(OUT_DIR / "llava_userprompt_vp_risk_by_label.pdf")
    plt.close(fig)


def _plot_gap(stats_by_variant: dict[str, list[dict[int, dict[str, float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for slug, display, _directory in VARIANTS:
        stats = stats_by_variant[slug]
        layers = np.arange(len(stats))
        gaps = np.array([
            row[LABEL_HALLUCINATED]["mean"] - row[LABEL_REAL]["mean"]
            for row in stats
        ])
        ax.plot(layers, gaps, label=display)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Hallucination - non-hallucination risk")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "llava_userprompt_vp_gap.png", dpi=180)
    fig.savefig(OUT_DIR / "llava_userprompt_vp_gap.pdf")
    plt.close(fig)


def _write_target_distribution_summary() -> None:
    diag_csv = OUT_DIR / "target_diagnostic" / "llava_10image_target_distribution_layerwise.csv"
    selected_csv = OUT_DIR / "target_diagnostic" / "llava_10image_selected_cases.csv"
    out_path = OUT_DIR / "llava_userprompt_vp_target_distribution_summary.md"
    if not diag_csv.exists():
        out_path.write_text("Target distribution diagnostic has not been generated yet.\n")
        return
    rows = list(csv.DictReader(diag_csv.open()))
    max_layer = max(int(row["layer"]) for row in rows)
    final_rows = [row for row in rows if int(row["layer"]) == max_layer]
    lines = [
        "# LLaVA User-Prompt-Only Target Distribution",
        "",
        "| Label | n | target prompt mass | attention prompt mass | source prompt mass | top16 target prompt share | top16 quality prompt share |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (0, 1):
        label_rows = [row for row in final_rows if int(row["label"]) == label]
        if not label_rows:
            continue
        lines.append(
            "| {label} | {n} | {t:.4f} | {a:.4f} | {s:.4f} | {tt:.4f} | {qt:.4f} |".format(
                label=label,
                n=len(label_rows),
                t=float(np.mean([float(row["vp_target_prompt_mass"]) for row in label_rows])),
                a=float(np.mean([float(row["attention_prompt_mass"]) for row in label_rows])),
                s=float(np.mean([float(row["source_prompt_mass"]) for row in label_rows])),
                tt=float(np.mean([float(row["vp_top16_target_prompt_count"]) / 16.0 for row in label_rows])),
                qt=float(np.mean([float(row["vp_top16_gate_prompt_count"]) / 16.0 for row in label_rows])),
            )
        )
    if selected_csv.exists():
        selected_rows = list(csv.DictReader(selected_csv.open()))
        if selected_rows:
            first = selected_rows[0]
            lines.extend(
                [
                    "",
                    "## Prompt Support",
                    "",
                    f"- mode: `{first.get('prompt_support_mode', 'full')}`",
                    f"- text: `{first.get('prompt_support_text', '')}`",
                    f"- selected token count: `{first.get('prompt_support_token_count', '')}`",
                    f"- selected tokens: `{first.get('prompt_support_tokens', '')}`",
                ]
            )
    out_path.write_text("\n".join(lines) + "\n")


def _write_summary(curve_rows: list[dict], torch_rows: list[dict]) -> None:
    lines = [
        "# LLaVA User-Prompt-Only VP Comparison",
        "",
        "Compares full-prompt VP/VP against user-prompt-only VP/VP on raw top-k relative-VLL geo risk.",
        "",
        "## Curve Summary",
        "",
        "| Variant | mean hall | mean non | mean gap H-N | peak layer | peak gap | final gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve_rows:
        lines.append(
            "| {variant_label} | {mean_hallucination:.6f} | {mean_non_hallucination:.6f} | "
            "{mean_gap_h_minus_non:.6f} | {peak_abs_gap_layer} | "
            "{peak_abs_gap_h_minus_non:.6f} | {final_layer_gap_h_minus_non:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Torch MLP",
            "",
            "| Variant | Feature set | AUC | AUPR | F1 | Precision | Recall | Accuracy |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in torch_rows:
        lines.append(
            "| {variant_label} | `{feature_set}` | {auc:.6f} | {aupr:.6f} | {f1:.6f} | "
            "{precision:.6f} | {recall:.6f} | {accuracy:.6f} |".format(
                **{key: _to_float(value) if key in {"auc", "aupr", "f1", "precision", "recall", "accuracy"} else value for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `llava_userprompt_vp_layerwise.csv`",
            "- `llava_userprompt_vp_curve_summary.csv`",
            "- `llava_userprompt_vp_torch_mlp.csv`",
            "- `llava_userprompt_vp_target_distribution_summary.md`",
            "- `llava_userprompt_vp_risk_by_label.{png,pdf}`",
            "- `llava_userprompt_vp_gap.{png,pdf}`",
        ]
    )
    (OUT_DIR / "llava_userprompt_vp_summary.md").write_text("\n".join(lines) + "\n")


def _to_float(value) -> float:
    return float("nan") if value is None else float(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
