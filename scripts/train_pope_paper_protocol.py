#!/usr/bin/env python3
"""Run the paper-compatible POPE Yes-only ADS/CGC five-fold baseline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.pope_paper_protocol import FEATURE_SETS, run_five_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Completed strict POPE model output directory")
    parser.add_argument("--classifiers", nargs="+", default=["mlp", "rf", "xgb"], choices=("mlp", "rf", "xgb"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.run_dir)
    with open(root / "features.pkl", "rb") as handle:
        rows = pickle.load(handle)
    results = {}
    for feature_set in FEATURE_SETS:
        results[feature_set] = {}
        for classifier in args.classifiers:
            result = run_five_fold(rows, feature_set, classifier, args.seed)
            results[feature_set][classifier] = result
            print(
                f"{feature_set}/{classifier}: n={result['num_samples']} "
                f"hall={result['num_hallucination']} "
                f"F1={result['mean_std']['hallucination_f1']['mean']:.4f} "
                f"AUC={result['mean_std']['auc']['mean']:.4f}"
            )
    output = root / "results" / "pope_ads_cgc_paper_protocol_5fold.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, output)
    print(output)


if __name__ == "__main__":
    main()
