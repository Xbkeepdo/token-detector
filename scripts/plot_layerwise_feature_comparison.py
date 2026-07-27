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
    "risk_geo_raw": "dgst_t_transport_risk_relative_vll_cost_geo_per_layer",
    "risk_hmid_proj": "dgst_t_transport_risk_relative_vll_source_hmid_proj_per_layer",
    "hmid_proj": "dgst_t_transport_risk_relative_vll_source_hmid_proj_per_layer",
    "risk_hprev_cos": "dgst_t_transport_risk_relative_vll_source_hprev_cos_per_layer",
    "hprev_cos": "dgst_t_transport_risk_relative_vll_source_hprev_cos_per_layer",
    "hpre_cos": "dgst_t_transport_risk_relative_vll_source_hprev_cos_per_layer",
    "risk_hprev_proj": "dgst_t_transport_risk_relative_vll_source_hprev_proj_per_layer",
    "hprev_proj": "dgst_t_transport_risk_relative_vll_source_hprev_proj_per_layer",
    "hpre_proj": "dgst_t_transport_risk_relative_vll_source_hprev_proj_per_layer",
    "vp_risk_hmid_proj": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hmid_proj_per_layer",
    "vp_hmid_proj": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hmid_proj_per_layer",
    "vp_risk_hprev_cos": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_cos_per_layer",
    "vp_hprev_cos": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_cos_per_layer",
    "vp_hpre_cos": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_cos_per_layer",
    "vp_risk_hprev_proj": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_proj_per_layer",
    "vp_hprev_proj": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_proj_per_layer",
    "vp_hpre_proj": "dgst_t_transport_risk_visual_prompt_relative_vll_source_hprev_proj_per_layer",
    "c_vp": "dgst_t_c_vp_relative_vll_cost_geo_per_layer",
    "cvp": "dgst_t_c_vp_relative_vll_cost_geo_per_layer",
    "m_p": "dgst_t_m_p_per_layer",
    "mp": "dgst_t_m_p_per_layer",
    "r_es": "dgst_t_r_es_relative_vll_cost_geo_per_layer",
    "res": "dgst_t_r_es_relative_vll_cost_geo_per_layer",
    "es": "dgst_t_relative_vll_evidence_strength_per_layer",
    "evidence_strength": "dgst_t_relative_vll_evidence_strength_per_layer",
    "vp_es": "dgst_t_visual_prompt_relative_vll_evidence_strength_per_layer",
    "vp_evidence_strength": "dgst_t_visual_prompt_relative_vll_evidence_strength_per_layer",
    "t_v": "dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer",
    "tv": "dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer",
    "evidence_visual_mass": "dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer",
    "t_p": "dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer",
    "tp": "dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer",
    "evidence_prompt_mass": "dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer",
    "b_v": "dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer",
    "bv": "dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer",
    "source_visual_mass": "dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer",
    "b_p": "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer",
    "bp": "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer",
    "source_prompt_mass": "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer",
    "vv_source_entropy": "dgst_t_vv_source_entropy_per_layer",
    "vv_h_source": "dgst_t_vv_source_entropy_per_layer",
    "vv_target_entropy": "dgst_t_vv_target_entropy_per_layer",
    "vv_h_target": "dgst_t_vv_target_entropy_per_layer",
    "vv_evidence_entropy": "dgst_t_vv_evidence_entropy_per_layer",
    "vv_h_evidence": "dgst_t_vv_evidence_entropy_per_layer",
    "vv_source_topk_entropy": "dgst_t_vv_source_topk_entropy_per_layer",
    "vv_h_source_topk": "dgst_t_vv_source_topk_entropy_per_layer",
    "vp_source_entropy": "dgst_t_vp_source_entropy_per_layer",
    "vp_h_source": "dgst_t_vp_source_entropy_per_layer",
    "vp_target_entropy": "dgst_t_vp_target_entropy_per_layer",
    "vp_h_target": "dgst_t_vp_target_entropy_per_layer",
    "vp_evidence_entropy": "dgst_t_vp_evidence_entropy_per_layer",
    "vp_h_evidence": "dgst_t_vp_evidence_entropy_per_layer",
    "vp_source_topk_entropy": "dgst_t_vp_source_topk_entropy_per_layer",
    "vp_h_source_topk": "dgst_t_vp_source_topk_entropy_per_layer",
    "js_relative_vll": "dgst_t_js_relative_vll_per_layer",
    "js_vv": "dgst_t_js_relative_vll_per_layer",
    "vv_js": "dgst_t_js_relative_vll_per_layer",
    "kl_target_source_relative_vll": "dgst_t_kl_target_source_relative_vll_per_layer",
    "kl_target_source_vv": "dgst_t_kl_target_source_relative_vll_per_layer",
    "vv_kl_target_source": "dgst_t_kl_target_source_relative_vll_per_layer",
    "kl_source_target_relative_vll": "dgst_t_kl_source_target_relative_vll_per_layer",
    "kl_source_target_vv": "dgst_t_kl_source_target_relative_vll_per_layer",
    "vv_kl_source_target": "dgst_t_kl_source_target_relative_vll_per_layer",
    "js_visual_prompt_relative_vll": "dgst_t_js_visual_prompt_relative_vll_per_layer",
    "js_vp": "dgst_t_js_visual_prompt_relative_vll_per_layer",
    "vp_js": "dgst_t_js_visual_prompt_relative_vll_per_layer",
    "kl_target_source_visual_prompt_relative_vll": (
        "dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer"
    ),
    "kl_target_source_vp": "dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer",
    "vp_kl_target_source": "dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer",
    "kl_source_target_visual_prompt_relative_vll": (
        "dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer"
    ),
    "kl_source_target_vp": "dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer",
    "vp_kl_source_target": "dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer",
    "target_visual_hidden_cosine": "dgst_t_target_visual_hidden_cosine_per_layer",
    "target_visual_prompt_hidden_cosine": "dgst_t_target_visual_prompt_hidden_cosine_per_layer",
    "target_visual_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer",
    "target_visual_hidden_cosine_relative_vll": "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
    "target_cosine": "dgst_t_target_visual_hidden_cosine_relative_vll_per_layer",
    "cosine16": "dgst_t_target_visual_hidden_cosine16_relative_vll_per_layer",
    "target_cosine16": "dgst_t_target_visual_hidden_cosine16_relative_vll_per_layer",
    "hprecosine": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
    "hpre_cosine": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
    "hprecosine_raw": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
    "target_visual_hpre_cosine": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
    "target_visual_hpre_cosine_relative_vll": "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
    "hprecosine16": "dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer",
    "hpre_cosine16": "dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer",
    "target_visual_hpre_cosine16": "dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer",
    "hprecosine_cap085": "dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer",
    "hpre_cosine_cap085": "dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"
    ),
    "vp_target_cosine": "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer",
    "vp_cosine16": "dgst_t_target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer",
    "vp_hprecosine": "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer",
    "vp_hpre_cosine": "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer",
    "target_visual_prompt_hpre_cosine": "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer",
    "vp_hprecosine16": "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer",
    "vp_hpre_cosine16": "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer",
    "target_visual_prompt_hpre_cosine16": "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer",
    "vp_hprecosine_cap085": (
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "vp_hpre_cosine_cap085": (
        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer",
    "context_confidence": "dgst_t_context_confidence_per_layer",
    "context_confidence_max_prompt": "dgst_t_context_confidence_max_prompt_per_layer",
    "ffn_fad": "dgst_t_ffn_attn_dominance_per_layer",
    "fad": "dgst_t_ffn_attn_dominance_per_layer",
    "ffn_eifdose": "dgst_t_ffn_evidence_orthogonal_dose_per_layer",
    "eifdose": "dgst_t_ffn_evidence_orthogonal_dose_per_layer",
    "ffn_logitlift": "dgst_t_ffn_logit_lift_per_layer",
    "logitlift": "dgst_t_ffn_logit_lift_per_layer",
    "ffn_eiffrac_svd": "dgst_t_ffn_eif_fraction_svd_per_layer",
    "eiffrac_svd": "dgst_t_ffn_eif_fraction_svd_per_layer",
    "ffn_eifdose_svd": "dgst_t_ffn_eif_dose_svd_per_layer",
    "eifdose_svd": "dgst_t_ffn_eif_dose_svd_per_layer",
    "ffn_eiffrac_pca": "dgst_t_ffn_eif_fraction_pca_per_layer",
    "eiffrac_pca": "dgst_t_ffn_eif_fraction_pca_per_layer",
    "ffn_eifdose_pca": "dgst_t_ffn_eif_dose_pca_per_layer",
    "eifdose_pca": "dgst_t_ffn_eif_dose_pca_per_layer",
}
LABEL_HALLUCINATED = 0
LABEL_REAL = 1


def _register_delta_source_aliases() -> None:
    for target_slug in ("rvll", "vp_rvll"):
        for gamma_slug in ("g0", "g05", "g1"):
            block = f"{target_slug}_delta_{gamma_slug}"
            key_stem = f"{target_slug}_delta_src_{gamma_slug}"
            FEATURE_KEYS.update(
                {
                    block: f"dgst_t_risk_{key_stem}_per_layer",
                    f"risk_{block}": f"dgst_t_risk_{key_stem}_per_layer",
                    f"{block}_cap085": f"dgst_t_risk_{key_stem}_cap085_per_layer",
                    f"risk_{block}_cap085": f"dgst_t_risk_{key_stem}_cap085_per_layer",
                    f"{block}_cos": f"dgst_t_cos_{key_stem}_per_layer",
                    f"cos_{block}": f"dgst_t_cos_{key_stem}_per_layer",
                    f"{block}_cos_cap085": f"dgst_t_cos_{key_stem}_cap085_per_layer",
                    f"{block}_cap085_cos": f"dgst_t_cos_{key_stem}_cap085_per_layer",
                    f"cos_{block}_cap085": f"dgst_t_cos_{key_stem}_cap085_per_layer",
                }
            )


def _register_relative_cost_aliases() -> None:
    for slug in ("geo", "tbar", "sbar", "tadd", "sadd", "tsadd", "qmatch"):
        for prefix in ("risk_relative_vll", "risk_visual_prompt_relative_vll"):
            block = f"{prefix}_cost_{slug}"
            key = f"dgst_t_transport_{prefix}_cost_{slug}_per_layer"
            cap_block = f"{block}_capped_topmass_085"
            cap_key = f"dgst_t_transport_{prefix}_cost_{slug}_capped_topmass_085_per_layer"
            FEATURE_KEYS.update(
                {
                    block: key,
                    f"{block}_cap085": cap_key,
                    cap_block: cap_key,
                    f"{prefix}_{slug}": key,
                    f"{prefix}_{slug}_cap085": cap_key,
                    f"{prefix}_{slug}_capped_topmass_085": cap_key,
                }
            )
    for state_slug in (
        "mid",
        "out",
        "avg",
        "stateupd_lu005",
        "stateupd_lu01",
        "stateupd_lu02",
        "stateupd_lu1",
    ):
        for prefix in ("risk_relative_vll", "risk_visual_prompt_relative_vll"):
            block = f"{prefix}_cost_geo_{state_slug}"
            FEATURE_KEYS[block] = f"dgst_t_transport_{prefix}_cost_geo_{state_slug}_per_layer"


def _register_topk_region_aliases() -> None:
    source_specs = (
        ("", ("",)),
        ("hmid_proj", ("hmid_proj", "hmid")),
        ("hprev_cos", ("hprev_cos", "hpre_cos")),
        ("hprev_proj", ("hprev_proj", "hpre_proj")),
    )
    stat_names = ("skm", "tkm", "cov_st", "es")
    selectors = ("union", "rec", "target")
    variants = ("lk", "", "la", "lp")

    for scope in ("vv", "vp"):
        for source_key, source_aliases in source_specs:
            stem = f"dgst_t_{scope}"
            alias_middle = ""
            if source_key:
                stem = f"{stem}_source_{source_key}"
                alias_middle = f"_{source_key}"
            for stat in stat_names:
                block = f"{scope}{alias_middle}_topk_{stat}"
                key = f"{stem}_topk_{stat}_per_layer"
                FEATURE_KEYS[block] = key
                if scope == "vv" and not source_key:
                    FEATURE_KEYS[f"topk_{stat}"] = key
                for source_alias in source_aliases:
                    if source_alias:
                        FEATURE_KEYS[f"{scope}_{source_alias}_topk_{stat}"] = key
            for selector in selectors:
                for variant in variants:
                    suffix = f"_{variant}" if variant else ""
                    block = f"{scope}{alias_middle}_r_{selector}{suffix}"
                    key = f"{stem}_r_{selector}{suffix}_per_layer"
                    FEATURE_KEYS[block] = key
                    if scope == "vv" and not source_key:
                        FEATURE_KEYS[f"r_{selector}{suffix}"] = key
                    for source_alias in source_aliases:
                        if source_alias:
                            FEATURE_KEYS[f"{scope}_{source_alias}_r_{selector}{suffix}"] = key


_register_delta_source_aliases()
_register_relative_cost_aliases()
_register_topk_region_aliases()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["target_visual_hidden_cosine", "target_visual_prompt_hidden_cosine"],
        help="Feature aliases or raw feature keys to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["visual-only top-k", "visual+prompt top-k"],
        help="Display labels for the feature curves.",
    )
    parser.add_argument("--name", default=None, help="Output filename stem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.labels) != len(args.features):
        raise ValueError("--labels must have the same length as --features.")
    feature_path = os.path.join(args.output_dir, "features.pkl")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    features = load_pkl(feature_path)
    keys = [_resolve_key(item) for item in args.features]
    _ensure_keys(features, keys)

    grouped = [_group_by_label(features, key) for key in keys]
    stats = [_stats_by_label(group) for group in grouped]

    stem = args.name or f"{args.model}_{'_vs_'.join(args.features)}_by_label"
    png_path = os.path.join(results_dir, f"{stem}.png")
    pdf_path = os.path.join(results_dir, f"{stem}.pdf")
    csv_path = os.path.join(results_dir, f"{stem}.csv")

    _plot(stats, args.labels, png_path, pdf_path)
    _write_csv(stats, args.labels, csv_path)

    print("png", os.path.abspath(png_path))
    print("pdf", os.path.abspath(pdf_path))
    print("csv", os.path.abspath(csv_path))
    for label, item in zip(args.labels, stats):
        diff = item[LABEL_HALLUCINATED]["mean"] - item[LABEL_REAL]["mean"]
        peak = int(np.argmax(np.abs(diff)))
        print(label)
        print("  n_hallucination", item[LABEL_HALLUCINATED]["n"])
        print("  n_non_hallucination", item[LABEL_REAL]["n"])
        print("  hallucination_mean_avg", float(item[LABEL_HALLUCINATED]["mean"].mean()))
        print("  non_hallucination_mean_avg", float(item[LABEL_REAL]["mean"].mean()))
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
    layers = np.arange(stats[0][LABEL_HALLUCINATED]["mean"].shape[0])
    col_count = len(stats)
    fig, axes = plt.subplots(
        2,
        col_count,
        figsize=(max(6.6 * col_count, 8.0), 7.2),
        sharex=True,
    )
    if col_count == 1:
        axes = np.asarray(axes).reshape(2, 1)
    colors = {LABEL_HALLUCINATED: "#d55e00", LABEL_REAL: "#0072b2"}
    class_labels = {LABEL_HALLUCINATED: "Hallucination", LABEL_REAL: "Non-hallucination"}

    for col, (item, feature_label) in enumerate(zip(stats, labels)):
        ax = axes[0][col]
        for class_id in (LABEL_HALLUCINATED, LABEL_REAL):
            mean = item[class_id]["mean"]
            sem = item[class_id]["sem"]
            display = f"{class_labels[class_id]} (n={item[class_id]['n']})"
            ax.plot(layers, mean, color=colors[class_id], linewidth=2.2, label=display)
            ax.fill_between(layers, mean - sem, mean + sem, color=colors[class_id], alpha=0.18)
        ax.set_title(feature_label)
        ax.set_ylabel("Mean feature value")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

        diff = item[LABEL_HALLUCINATED]["mean"] - item[LABEL_REAL]["mean"]
        peak = int(np.argmax(np.abs(diff)))
        ax_diff = axes[1][col]
        ax_diff.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.8)
        ax_diff.plot(layers, diff, color="#333333", linewidth=2.0)
        ax_diff.scatter([peak], [diff[peak]], color="#cc79a7", s=42, zorder=3)
        ax_diff.set_xlabel("Layer")
        ax_diff.set_ylabel("Mean diff\nhall-non")
        ax_diff.grid(True, alpha=0.25)
        ax_diff.set_xticks(layers)

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(stats: list[dict[int, dict]], labels: list[str], csv_path: str) -> None:
    layer_count = int(stats[0][LABEL_HALLUCINATED]["mean"].shape[0])
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
            diff = item[LABEL_HALLUCINATED]["mean"] - item[LABEL_REAL]["mean"]
            row.extend(
                [
                    f"{item[LABEL_HALLUCINATED]['mean'][layer]:.8f}",
                    f"{item[LABEL_HALLUCINATED]['sem'][layer]:.8f}",
                    f"{item[LABEL_REAL]['mean'][layer]:.8f}",
                    f"{item[LABEL_REAL]['sem'][layer]:.8f}",
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
