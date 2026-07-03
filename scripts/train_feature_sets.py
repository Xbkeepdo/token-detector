#!/usr/bin/env python3
"""Train classifiers on selected DGST-T feature blocks."""

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io_utils import load_json, load_pkl, save_json

from summarize_feature_set_results import write_summary_tables


FEATURE_ALIASES = {
    "risk": "risk",
    "transport_risk": "risk",
    "risk_topmass_085": "risk_topmass_085",
    "topmass_085": "risk_topmass_085",
    "risk_capped_topmass_085": "risk_capped_topmass_085",
    "capped_topmass_085": "risk_capped_topmass_085",
    "risk_relative_vll": "risk_relative_vll",
    "relative_vll_risk": "risk_relative_vll",
    "risk_relative_vll_capped_topmass_085": "risk_relative_vll_capped_topmass_085",
    "relative_vll_capped_topmass_085": "risk_relative_vll_capped_topmass_085",
    "risk_visual_prompt_relative_vll": "risk_visual_prompt_relative_vll",
    "visual_prompt_relative_vll_risk": "risk_visual_prompt_relative_vll",
    "risk_visual_prompt_relative_vll_capped_topmass_085": "risk_visual_prompt_relative_vll_capped_topmass_085",
    "visual_prompt_relative_vll_capped_topmass_085": "risk_visual_prompt_relative_vll_capped_topmass_085",
    "prompt_confidence": "prompt_confidence_top3",
    "prompt_confidence_top3": "prompt_confidence_top3",
    "prompt_confidence_max": "prompt_confidence_max",
    "context_confidence": "context_confidence",
    "contextconfidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
    "visual_cosine": "target_visual_hidden_cosine",
    "target_visual_hidden_cosine": "target_visual_hidden_cosine",
    "visual_prompt_cosine": "target_visual_prompt_hidden_cosine",
    "target_visual_prompt_hidden_cosine": "target_visual_prompt_hidden_cosine",
    "visual_cosine_capped_topmass_085": "target_visual_hidden_cosine_capped_topmass_085",
    "target_visual_hidden_cosine_capped_topmass_085": "target_visual_hidden_cosine_capped_topmass_085",
    "visual_cosine_relative_vll": "target_visual_hidden_cosine_relative_vll",
    "target_visual_hidden_cosine_relative_vll": "target_visual_hidden_cosine_relative_vll",
    "visual_cosine_relative_vll_capped_topmass_085": "target_visual_hidden_cosine_relative_vll_capped_topmass_085",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "target_visual_hidden_cosine_relative_vll_capped_topmass_085"
    ),
    "visual_prompt_cosine_visual_prompt_relative_vll": "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll",
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll"
    ),
    "visual_prompt_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
    ),
    "visual_prompt_cosine_capped_topmass_085": "target_visual_prompt_hidden_cosine_capped_topmass_085",
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "target_visual_prompt_hidden_cosine_capped_topmass_085",
    "prompt_last_cosine": "prompt_last_cosine",
    "prompt_mean_cosine": "prompt_mean_cosine",
    "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085_times_inverse_target_visual_hidden_cosine": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085*(1-target_visual_hidden_cosine)": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
    "risk_capped_topmass_085*(1_target_visual_hidden_cosine)": "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine",
}

FEATURE_KEYS = {
    "risk": "dgst_t_transport_risk_per_layer",
    "risk_topmass_085": "dgst_t_transport_risk_topmass_085_per_layer",
    "risk_capped_topmass_085": "dgst_t_transport_risk_capped_topmass_085_per_layer",
    "risk_relative_vll": "dgst_t_transport_risk_relative_vll_per_layer",
    "risk_relative_vll_capped_topmass_085": "dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer",
    "risk_visual_prompt_relative_vll": "dgst_t_transport_risk_visual_prompt_relative_vll_per_layer",
    "risk_visual_prompt_relative_vll_capped_topmass_085": (
        "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer"
    ),
    "prompt_confidence_top3": "dgst_t_prompt_confidence_top3_per_layer",
    "prompt_confidence_max": "dgst_t_prompt_confidence_max_per_layer",
    "context_confidence": "dgst_t_context_confidence_per_layer",
    "context_confidence_max_prompt": "dgst_t_context_confidence_max_prompt_per_layer",
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
    "prompt_last_cosine": "dgst_t_prompt_last_cosine_per_layer",
    "prompt_mean_cosine": "dgst_t_prompt_mean_cosine_per_layer",
}

LAYER_STAT_KEYS = {
    "prompt_confidence_top3": "prompt_logit_lens_top3_confidence",
    "prompt_confidence_max": "prompt_logit_lens_max_confidence",
    "target_visual_hidden_cosine": "target_hidden_top32_visual_cosine",
    "target_visual_prompt_hidden_cosine": "target_hidden_top32_visual_prompt_cosine",
    "target_visual_hidden_cosine_capped_topmass_085": "target_hidden_capped_topmass_085_visual_cosine",
    "target_visual_hidden_cosine_relative_vll": "target_hidden_top32_visual_cosine_relative_vll",
    "target_visual_hidden_cosine_relative_vll_capped_topmass_085": (
        "target_hidden_capped_topmass_085_visual_cosine_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll": (
        "target_hidden_top32_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085": (
        "target_hidden_capped_topmass_085_visual_prompt_cosine_visual_prompt_relative_vll"
    ),
    "target_visual_prompt_hidden_cosine_capped_topmass_085": "target_hidden_capped_topmass_085_visual_prompt_cosine",
    "context_confidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
}


def _register_delta_source_aliases() -> None:
    for target_slug in ("rvll", "vp_rvll"):
        for gamma_slug in ("g0", "g05", "g1"):
            block = f"{target_slug}_delta_{gamma_slug}"
            key_stem = f"{target_slug}_delta_src_{gamma_slug}"
            risk_key = f"dgst_t_risk_{key_stem}_per_layer"
            risk_cap_key = f"dgst_t_risk_{key_stem}_cap085_per_layer"
            cos_key = f"dgst_t_cos_{key_stem}_per_layer"
            cos_cap_key = f"dgst_t_cos_{key_stem}_cap085_per_layer"

            FEATURE_ALIASES.update(
                {
                    block: block,
                    f"risk_{block}": block,
                    f"{block}_cap085": f"{block}_cap085",
                    f"risk_{block}_cap085": f"{block}_cap085",
                    f"{block}_cos": f"{block}_cos",
                    f"cos_{block}": f"{block}_cos",
                    f"{block}_cos_cap085": f"{block}_cos_cap085",
                    f"{block}_cap085_cos": f"{block}_cos_cap085",
                    f"cos_{block}_cap085": f"{block}_cos_cap085",
                }
            )
            FEATURE_KEYS.update(
                {
                    block: risk_key,
                    f"{block}_cap085": risk_cap_key,
                    f"{block}_cos": cos_key,
                    f"{block}_cos_cap085": cos_cap_key,
                }
            )


def _register_relative_cost_aliases() -> None:
    for slug in ("geo", "tbar", "sbar"):
        for prefix in ("risk_relative_vll", "risk_visual_prompt_relative_vll"):
            block = f"{prefix}_cost_{slug}"
            cap_block = f"{block}_capped_topmass_085"
            FEATURE_ALIASES.update(
                {
                    block: block,
                    cap_block: cap_block,
                    f"{block}_cap085": cap_block,
                    f"{prefix}_{slug}": block,
                    f"{prefix}_{slug}_capped_topmass_085": cap_block,
                    f"{prefix}_{slug}_cap085": cap_block,
                }
            )
            FEATURE_KEYS.update(
                {
                    block: f"dgst_t_transport_{prefix}_cost_{slug}_per_layer",
                    cap_block: (
                        f"dgst_t_transport_{prefix}_cost_{slug}_capped_topmass_085_per_layer"
                    ),
                }
            )


_register_delta_source_aliases()
_register_relative_cost_aliases()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=[
            "risk",
            "risk_topmass_085",
            "risk_capped_topmass_085",
            "context_confidence",
            "context_confidence_max_prompt",
            "target_visual_hidden_cosine",
            "risk+context_confidence",
            "risk+context_confidence_max_prompt",
            "risk+target_visual_hidden_cosine",
        ],
        help="Feature blocks to concatenate, e.g. risk+context_confidence.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["xgb", "rf"],
        choices=["xgb", "rf", "mlp"],
    )
    parser.add_argument("--scoring", default="f1", choices=["f1", "accuracy", "auc"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from detection.train import evaluate_classifier, grid_search, split_by_image_id
    from utils.config_utils import get_classifier_cfgs, load_config

    config = load_config(args.config)
    clf_cfgs = get_classifier_cfgs(config)

    feature_path = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(feature_path):
        raise FileNotFoundError(feature_path)
    if not os.path.exists(splits_path):
        raise FileNotFoundError(splits_path)

    all_features = load_pkl(feature_path)
    splits = load_json(splits_path)
    train_feats, val_feats, test_feats = split_by_image_id(
        all_features,
        train_image_ids={int(x) for x in splits["train"]},
        val_image_ids={int(x) for x in splits["val"]},
        test_image_ids={int(x) for x in splits["test"]},
    )

    out_path = os.path.join(results_dir, f"{args.model}_selected_feature_sets.json")
    results = load_json(out_path) if os.path.exists(out_path) else {}

    print(
        f"[FeatureSets] Loaded {len(all_features)} token features: "
        f"train={len(train_feats)}, val={len(val_feats)}, test={len(test_feats)}"
    )

    for feature_set in args.feature_sets:
        blocks = parse_feature_set(feature_set)
        X_train, y_train = build_selected_matrix(train_feats, blocks)
        X_val, y_val = build_selected_matrix(val_feats, blocks)
        X_test, y_test = build_selected_matrix(test_feats, blocks)
        if X_train.shape[0] == 0:
            raise ValueError(f"No training rows for feature set {feature_set!r}.")
        if X_val.shape[0] == 0 or len(np.unique(y_val)) < 2:
            print(f"[FeatureSets] WARNING: val split unusable for {feature_set}; using train as val.")
            X_val, y_val = X_train.copy(), y_train.copy()
        if X_test.shape[0] == 0 or len(np.unique(y_test)) < 2:
            print(f"[FeatureSets] WARNING: test split unusable for {feature_set}; using val as test.")
            X_test, y_test = X_val.copy(), y_val.copy()

        print(f"\n[FeatureSets] {feature_set}: X={X_train.shape[1]} dims")
        set_results = results.setdefault(feature_set, {})
        for clf_name in args.classifiers:
            grid = deepcopy(clf_cfgs.get(clf_name, {}))
            _sanitise_grid(grid)
            best_clf, best_params, val_score = grid_search(
                clf_name,
                grid,
                X_train,
                y_train,
                X_val,
                y_val,
                scoring=args.scoring,
            )
            metrics = evaluate_classifier(best_clf, X_test, y_test)
            metrics["best_params"] = best_params
            metrics["val_score"] = float(val_score)
            metrics["num_features"] = int(X_train.shape[1])
            set_results[clf_name] = _json_ready(metrics)
            print(
                f"  {clf_name.upper():<4} "
                f"F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} "
                f"PR={metrics['precision']:.3f} RC={metrics['recall']:.3f}"
            )
        save_json(results, out_path)

    print(f"\n[FeatureSets] Saved results to {out_path}")
    _write_summary_table(out_path)


def parse_feature_set(value: str) -> list[str]:
    blocks = []
    for raw in value.split("+"):
        key = raw.strip().lower().replace("（", "(").replace("）", ")").replace(" ", "")
        if key not in FEATURE_ALIASES:
            key = key.replace("-", "_")
        if not key:
            continue
        if key not in FEATURE_ALIASES:
            raise ValueError(f"Unknown feature block {raw!r}. Choices: {sorted(FEATURE_ALIASES)}")
        blocks.append(FEATURE_ALIASES[key])
    if not blocks:
        raise ValueError(f"Empty feature set {value!r}.")
    return blocks


def build_selected_matrix(features: Sequence[dict], blocks: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for feat in features:
        if feat.get("label") not in (0, 1):
            continue
        vectors = [feature_block(feat, block) for block in blocks]
        if any(vec.size == 0 for vec in vectors):
            continue
        rows.append(np.concatenate(vectors).astype(np.float32))
        labels.append(int(feat["label"]))
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(rows, axis=0), np.array(labels, dtype=np.int32)


def feature_block(feat: dict, block: str) -> np.ndarray:
    if block == "risk_capped_topmass_085_x_1_minus_target_visual_hidden_cosine":
        risk = feature_block(feat, "risk_capped_topmass_085")
        visual = feature_block(feat, "target_visual_hidden_cosine")
        if risk.shape != visual.shape:
            raise ValueError(
                "risk_capped_topmass_085 and target_visual_hidden_cosine must have "
                f"the same shape, got {risk.shape} and {visual.shape}."
            )
        return (risk * (1.0 - visual)).astype(np.float32)

    key = FEATURE_KEYS[block]
    values = feat.get(key)
    if values is None and block == "risk":
        values = feat.get("dgst_t_per_layer")
    if values is None and block == "target_visual_hidden_cosine":
        values = feat.get("dgst_t_atarget_visual_cosine_per_layer")
    if values is None and block in LAYER_STAT_KEYS:
        values = [item.get(LAYER_STAT_KEYS[block], 0.0) for item in feat.get("dgst_t_layer_stats", [])]
    if values is None:
        raise KeyError(f"Feature block {block!r} requires missing key {key!r}.")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _sanitise_grid(grid: dict) -> None:
    for key, value in list(grid.items()):
        if not isinstance(value, list):
            grid[key] = [value]


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_summary_table(results_path: str) -> None:
    try:
        write_summary_tables(results_path, formats=("md",), print_table=True)
    except Exception as exc:
        print(f"[FeatureSets] WARNING: failed to write summary table: {exc}")


if __name__ == "__main__":
    main()
