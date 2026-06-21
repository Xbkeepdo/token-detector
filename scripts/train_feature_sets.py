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

from utils.io_utils import load_json, load_pkl, save_json


FEATURE_ALIASES = {
    "risk": "risk",
    "transport_risk": "risk",
    "prompt_confidence": "prompt_confidence_top3",
    "prompt_confidence_top3": "prompt_confidence_top3",
    "prompt_confidence_max": "prompt_confidence_max",
    "context_confidence": "context_confidence",
    "contextconfidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
    "visual_cosine": "target_visual_hidden_cosine",
    "target_visual_hidden_cosine": "target_visual_hidden_cosine",
    "prompt_last_cosine": "prompt_last_cosine",
    "prompt_mean_cosine": "prompt_mean_cosine",
}

FEATURE_KEYS = {
    "risk": "dgst_t_transport_risk_per_layer",
    "prompt_confidence_top3": "dgst_t_prompt_confidence_top3_per_layer",
    "prompt_confidence_max": "dgst_t_prompt_confidence_max_per_layer",
    "context_confidence": "dgst_t_context_confidence_per_layer",
    "context_confidence_max_prompt": "dgst_t_context_confidence_max_prompt_per_layer",
    "target_visual_hidden_cosine": "dgst_t_target_visual_hidden_cosine_per_layer",
    "prompt_last_cosine": "dgst_t_prompt_last_cosine_per_layer",
    "prompt_mean_cosine": "dgst_t_prompt_mean_cosine_per_layer",
}

LAYER_STAT_KEYS = {
    "prompt_confidence_top3": "prompt_logit_lens_top3_confidence",
    "prompt_confidence_max": "prompt_logit_lens_max_confidence",
    "target_visual_hidden_cosine": "target_hidden_top32_visual_cosine",
    "context_confidence": "context_confidence",
    "context_confidence_max_prompt": "context_confidence_max_prompt",
}


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
    parser.add_argument("--scoring", default="f1", choices=["f1", "accuracy"])
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


def parse_feature_set(value: str) -> list[str]:
    blocks = []
    for raw in value.split("+"):
        key = raw.strip().lower().replace("-", "_")
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


if __name__ == "__main__":
    main()
