#!/usr/bin/env python3
"""Evaluate TGD hallucination detectors without ablations."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.evaluate import evaluate_ads_threshold, evaluate_cgc_threshold
from detection.train import train_and_evaluate
from utils.config_utils import load_config, get_classifier_cfgs
from utils.io_utils import load_json, load_pkl, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--classifiers", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    feat_path = os.path.join(args.output_dir, "features.pkl")
    splits_path = os.path.join(args.output_dir, "image_splits.json")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    features = load_pkl(feat_path)
    splits = load_json(splits_path)
    test_ids = set(splits["test"])
    test_feats = [f for f in features if f["image_id"] in test_ids]
    labeled_test = [f for f in test_feats if f.get("label") in (0, 1)]

    summary = {
        "model": args.model,
        "feature_path": feat_path,
        "test_images": len(test_ids),
        "test_object_tokens": len(labeled_test),
        "test_hallucinated_tokens": sum(1 for f in labeled_test if f["label"] == 0),
        "label_semantics": "0=hallucinated, 1=real",
    }

    print("\nDGST-T hallucination detection results")
    print("=" * 60)
    print(
        f"Test: {summary['test_images']} images, "
        f"{summary['test_object_tokens']} object tokens, "
        f"{summary['test_hallucinated_tokens']} hallucinated"
    )

    print("\nDGST-T score detector")
    summary["dgst_t"] = evaluate_ads_threshold(labeled_test)
    _print_metrics(summary["dgst_t"])

    if args.classifiers:
        config = load_config(args.config)
        summary["classifiers"] = train_and_evaluate(
            feature_path=feat_path,
            train_image_ids=set(splits["train"]),
            val_image_ids=set(splits["val"]),
            test_image_ids=test_ids,
            clf_configs=get_classifier_cfgs(config),
            output_dir=results_dir,
            model_key=args.model,
        )

    out_path = os.path.join(results_dir, f"{args.model}_detection_only.json")
    save_json(summary, out_path)
    print(f"\nSaved detection summary to {out_path}")


def _print_metrics(m: dict) -> None:
    print(
        f"  PR={m.get('precision', 0):.3f}  "
        f"RC={m.get('recall', 0):.3f}  "
        f"F1={m.get('f1', 0):.3f}  "
        f"ACC={m.get('accuracy', 0):.3f}  "
        f"AUC={m.get('auc', 0):.3f}"
    )


if __name__ == "__main__":
    main()
