#!/usr/bin/env python3
"""Summarize raw geo cost hidden-state variant experiments."""

from __future__ import annotations

import csv
import json
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


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "relative_cost_state_variants_comparison"

MODELS = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
    "llava_1_5_7b": "LLaVA-1.5-7B",
}

VARIANTS = [
    ("mid", "mid"),
    ("out", "out"),
    ("avg", "avg"),
    ("stateupd_lu005", "state+upd lu=0.05"),
    ("stateupd_lu01", "state+upd lu=0.10"),
    ("stateupd_lu02", "state+upd lu=0.20"),
    ("stateupd_lu1", "state+upd lu=1.0"),
]

MODES = {
    "vv": {
        "label": "V/V",
        "directory": "COCO500-visualonly-relativevll-cost-state-variants",
        "risk_prefix": "risk_relative_vll",
        "field_prefix": "dgst_t_transport_risk_relative_vll_cost_geo_",
        "cosine": "target_visual_hidden_cosine_relative_vll",
    },
    "vpvp": {
        "label": "VP/VP",
        "directory": "COCO500-visualprompt-relativevll-cost-state-variants",
        "risk_prefix": "risk_visual_prompt_relative_vll",
        "field_prefix": "dgst_t_transport_risk_visual_prompt_relative_vll_cost_geo_",
        "cosine": "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll",
    },
}

COLORS = {
    "mid": "#0072b2",
    "out": "#d55e00",
    "avg": "#009e73",
    "stateupd_lu005": "#cc79a7",
    "stateupd_lu01": "#f0e442",
    "stateupd_lu02": "#56b4e9",
    "stateupd_lu1": "#000000",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_rows = []
    curve_summaries = []

    for model, model_label in MODELS.items():
        for mode, mode_info in MODES.items():
            features = _load_features(model, mode_info["directory"])
            stats_by_variant = {}
            for variant, display in VARIANTS:
                key = f"{mode_info['field_prefix']}{variant}_per_layer"
                grouped = _group_by_label(features, key)
                stats = _stats_by_label(grouped)
                stats_by_variant[variant] = stats
                diff = stats[1]["mean"] - stats[0]["mean"]
                peak = int(np.argmax(np.abs(diff)))
                curve_summaries.append(
                    {
                        "model": model,
                        "model_label": model_label,
                        "mode": mode,
                        "mode_label": mode_info["label"],
                        "variant": variant,
                        "variant_label": display,
                        "n_hallucination": stats[1]["n"],
                        "n_non_hallucination": stats[0]["n"],
                        "mean_hallucination": float(stats[1]["mean"].mean()),
                        "mean_non_hallucination": float(stats[0]["mean"].mean()),
                        "mean_gap_h_minus_non": float(diff.mean()),
                        "peak_abs_gap_layer": peak,
                        "peak_abs_gap_h_minus_non": float(diff[peak]),
                    }
                )
                for layer in range(stats[0]["mean"].shape[0]):
                    layer_rows.append(
                        {
                            "model": model,
                            "model_label": model_label,
                            "mode": mode,
                            "mode_label": mode_info["label"],
                            "variant": variant,
                            "variant_label": display,
                            "layer": layer,
                            "n_hallucination": stats[1]["n"],
                            "n_non_hallucination": stats[0]["n"],
                            "hallucination_mean": float(stats[1]["mean"][layer]),
                            "hallucination_sem": float(stats[1]["sem"][layer]),
                            "non_hallucination_mean": float(stats[0]["mean"][layer]),
                            "non_hallucination_sem": float(stats[0]["sem"][layer]),
                            "gap_h_minus_non": float(diff[layer]),
                        }
                    )
            _plot_curves(model, model_label, mode, mode_info["label"], stats_by_variant)
            _plot_gaps(model, model_label, mode, mode_info["label"], stats_by_variant)

    mlp_rows = _load_torch_probe_rows()
    best_rows = _best_by_family(mlp_rows)

    _write_csv(OUTPUT_DIR / "relative_cost_state_variants_layerwise.csv", layer_rows)
    _write_csv(OUTPUT_DIR / "relative_cost_state_variants_curve_summary.csv", curve_summaries)
    _write_csv(OUTPUT_DIR / "relative_cost_state_variants_torch_mlp.csv", mlp_rows)
    _write_csv(OUTPUT_DIR / "relative_cost_state_variants_best_by_family.csv", best_rows)
    _write_markdown(curve_summaries, mlp_rows, best_rows)

    print(f"Wrote {OUTPUT_DIR}")


def _load_features(model: str, directory: str) -> list[dict]:
    path = ROOT / "outputs" / model / directory / "features.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    features = load_pkl(str(path))
    if not isinstance(features, list) or not features:
        raise ValueError(f"No features loaded from {path}")
    return features


def _group_by_label(features: list[dict], key: str) -> dict[int, np.ndarray]:
    grouped = {0: [], 1: []}
    for feat in features:
        label = feat.get("label")
        if label not in (0, 1) or key not in feat:
            continue
        values = np.asarray(feat[key], dtype=np.float32).reshape(-1)
        if values.size and np.isfinite(values).all():
            grouped[int(label)].append(values)
    if not grouped[0] or not grouped[1]:
        raise ValueError(f"Feature {key!r} needs both label=0 and label=1 rows.")
    return {label: np.stack(values, axis=0) for label, values in grouped.items()}


def _stats_by_label(grouped: dict[int, np.ndarray]) -> dict[int, dict]:
    result = {}
    for label, values in grouped.items():
        sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        result[label] = {
            "n": int(values.shape[0]),
            "mean": values.mean(axis=0),
            "sem": sem,
        }
    return result


def _plot_curves(
    model: str,
    model_label: str,
    mode: str,
    mode_label: str,
    stats_by_variant: dict[str, dict[int, dict]],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=False)
    axes_flat = axes.reshape(-1)
    class_colors = {1: "#d55e00", 0: "#0072b2"}
    class_labels = {1: "Hallucination", 0: "Non-hallucination"}
    for index, (variant, display) in enumerate(VARIANTS):
        ax = axes_flat[index]
        stats = stats_by_variant[variant]
        layers = np.arange(stats[0]["mean"].shape[0])
        for label in (1, 0):
            mean = stats[label]["mean"]
            sem = stats[label]["sem"]
            ax.plot(
                layers,
                mean,
                color=class_colors[label],
                linewidth=2.0,
                label=f"{class_labels[label]} (n={stats[label]['n']})",
            )
            ax.fill_between(layers, mean - sem, mean + sem, color=class_colors[label], alpha=0.16)
        ax.set_title(display)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Raw risk")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes_flat[-1].axis("off")
    fig.suptitle(f"{model_label} {mode_label} raw risk by label", fontsize=15)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    stem = f"{model}_{mode}_raw_cost_state_variants_by_label"
    _save_fig(fig, stem)


def _plot_gaps(
    model: str,
    model_label: str,
    mode: str,
    mode_label: str,
    stats_by_variant: dict[str, dict[int, dict]],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)
    for variant, display in VARIANTS:
        stats = stats_by_variant[variant]
        layers = np.arange(stats[0]["mean"].shape[0])
        gap = stats[1]["mean"] - stats[0]["mean"]
        ax.plot(layers, gap, color=COLORS[variant], linewidth=2.1, label=display)
    ax.set_title(f"{model_label} {mode_label} raw risk gap")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Hallucination - non-hallucination")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    stem = f"{model}_{mode}_raw_cost_state_variants_gap"
    _save_fig(fig, stem)


def _save_fig(fig, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_torch_probe_rows() -> list[dict]:
    rows = []
    for model, model_label in MODELS.items():
        for mode, mode_info in MODES.items():
            path = (
                ROOT
                / "outputs"
                / model
                / mode_info["directory"]
                / "results"
                / f"{model}_selected_feature_sets.json"
            )
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for variant, display in VARIANTS:
                for family, feature_set in _feature_sets_for_variant(mode_info, variant):
                    metrics = data.get(feature_set, {}).get("torch_probe")
                    if metrics is None:
                        continue
                    row = {
                        "model": model,
                        "model_label": model_label,
                        "mode": mode,
                        "mode_label": mode_info["label"],
                        "variant": variant,
                        "variant_label": display,
                        "family": family,
                        "feature_set": feature_set,
                    }
                    for key in (
                        "auc",
                        "aupr",
                        "f1",
                        "precision",
                        "recall",
                        "accuracy",
                        "best_epoch",
                        "best_val_loss",
                        "val_score",
                        "num_features",
                    ):
                        row[key] = metrics.get(key)
                    rows.append(row)
    return rows


def _feature_sets_for_variant(mode_info: dict, variant: str) -> list[tuple[str, str]]:
    risk = f"{mode_info['risk_prefix']}_cost_geo_{variant}"
    cosine = mode_info["cosine"]
    return [
        ("risk_only", risk),
        ("risk_plus_cosine", f"{risk}+{cosine}"),
    ]


def _best_by_family(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["mode"], row["family"]), []).append(row)
    best = []
    for (model, mode, family), items in sorted(groups.items()):
        item = max(items, key=lambda row: _metric_value(row.get("auc")))
        best.append(dict(item))
    return best


def _metric_value(value) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(curve_summaries: list[dict], mlp_rows: list[dict], best_rows: list[dict]) -> None:
    lines = [
        "# Relative Cost State Variants",
        "",
        "Scope: raw top-k only, geo cost, three models, V/V and VP/VP.",
        "",
        "Variants: mid, out, avg, stateupd_lu005, stateupd_lu01, stateupd_lu02, stateupd_lu1.",
        "",
        "## Curve Gap Summary",
        "",
        "| Model | Mode | Best abs mean gap variant | Mean gap | Peak layer | Peak gap |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for mode in MODES:
            items = [
                row
                for row in curve_summaries
                if row["model"] == model and row["mode"] == mode
            ]
            best = max(items, key=lambda row: abs(row["mean_gap_h_minus_non"]))
            lines.append(
                "| {model_label} | {mode_label} | {variant_label} | {mean_gap:.6f} | {peak_layer} | {peak_gap:.6f} |".format(
                    model_label=best["model_label"],
                    mode_label=best["mode_label"],
                    variant_label=best["variant_label"],
                    mean_gap=best["mean_gap_h_minus_non"],
                    peak_layer=best["peak_abs_gap_layer"],
                    peak_gap=best["peak_abs_gap_h_minus_non"],
                )
            )

    lines.extend(
        [
            "",
            "## Torch MLP Best By Family",
            "",
            "| Model | Mode | Family | Variant | AUC | AUPR | F1 | Precision | Recall | Accuracy | Best epoch |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows:
        lines.append(
            "| {model_label} | {mode_label} | {family} | {variant_label} | {auc:.6f} | {aupr:.6f} | {f1:.6f} | {precision:.6f} | {recall:.6f} | {accuracy:.6f} | {best_epoch} |".format(
                model_label=row["model_label"],
                mode_label=row["mode_label"],
                family=row["family"],
                variant_label=row["variant_label"],
                auc=_safe_float(row.get("auc")),
                aupr=_safe_float(row.get("aupr")),
                f1=_safe_float(row.get("f1")),
                precision=_safe_float(row.get("precision")),
                recall=_safe_float(row.get("recall")),
                accuracy=_safe_float(row.get("accuracy")),
                best_epoch=row.get("best_epoch", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Mid Baseline Versus Best AUC",
            "",
            "| Model | Mode | Family | mid AUC | Best variant | Best AUC | Delta |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for model in MODELS:
        for mode in MODES:
            for family in ("risk_only", "risk_plus_cosine"):
                items = [
                    row
                    for row in mlp_rows
                    if row["model"] == model and row["mode"] == mode and row["family"] == family
                ]
                if not items:
                    continue
                mid = next((row for row in items if row["variant"] == "mid"), None)
                best = max(items, key=lambda row: _metric_value(row.get("auc")))
                mid_auc = _safe_float(mid.get("auc")) if mid else float("nan")
                best_auc = _safe_float(best.get("auc"))
                lines.append(
                    "| {model_label} | {mode_label} | {family} | {mid_auc:.6f} | {variant_label} | {best_auc:.6f} | {delta:.6f} |".format(
                        model_label=best["model_label"],
                        mode_label=best["mode_label"],
                        family=family,
                        mid_auc=mid_auc,
                        variant_label=best["variant_label"],
                        best_auc=best_auc,
                        delta=best_auc - mid_auc,
                    )
                )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `relative_cost_state_variants_layerwise.csv`",
            "- `relative_cost_state_variants_curve_summary.csv`",
            "- `relative_cost_state_variants_torch_mlp.csv`",
            "- `relative_cost_state_variants_best_by_family.csv`",
            "- Per-model `*_by_label.{png,pdf}` and `*_gap.{png,pdf}` plots.",
            "",
        ]
    )
    (OUTPUT_DIR / "relative_cost_state_variants_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


if __name__ == "__main__":
    main()
