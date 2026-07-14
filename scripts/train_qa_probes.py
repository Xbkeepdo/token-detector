#!/usr/bin/env python3
"""Train all fixed QA feature-set probes for seeds 42/43/44."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.qa_probe import (
    aggregate_seed_results,
    default_feature_sets,
    safe_name,
    train_one_seed,
)
from data.qa_benchmark import load_jsonl
from utils.config_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("pope", "clevr_exist_5k"), required=True)
    parser.add_argument("--config", default="configs/qa_benchmarks_server_fj01.yaml")
    parser.add_argument("--output-root", default="/root/rivermind-data/project/token-detector/outputs/qa_benchmarks")
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    cfg = config["qa_probe"]
    seeds = args.seeds or cfg.get("seeds", [42, 43, 44])
    feature_sets = args.feature_sets or default_feature_sets(args.dataset)
    run_root = Path(args.output_root) / args.model / args.dataset
    with open(run_root / "features.pkl", "rb") as handle:
        rows = pickle.load(handle)
    labels = {row["key"]: row for row in load_jsonl(run_root / "labels.jsonl")}
    # Reporting-only metadata is joined in memory and never stored in features.pkl
    # or consumed by feature_vector, preventing GT feature leakage.
    rows = [
        {**row, "report_gt_answer": labels.get(row["key"], {}).get("gt_answer")}
        for row in rows
    ]
    results_root = run_root / "results"
    all_results = {}

    for feature_set in feature_sets:
        seed_results = []
        for seed in seeds:
            seed_dir = results_root / safe_name(feature_set) / f"seed_{seed}"
            result_path = seed_dir / "result.json"
            if result_path.exists() and not args.force:
                with open(result_path, encoding="utf-8") as handle:
                    result = json.load(handle)
            else:
                result = train_one_seed(rows, feature_set, seed, str(seed_dir), cfg, args.device)
            seed_results.append(result)
            print(f"[{feature_set}] seed={seed} test macro-F1={result['test_metrics']['macro_f1']:.4f}")
        all_results[feature_set] = aggregate_seed_results(seed_results)

    # Select DGST strictly on validation macro-F1, independently for each seed.
    best_results = []
    for seed in seeds:
        candidates = []
        for feature_set in feature_sets:
            if "+hprecosine@" not in feature_set:
                continue
            path = results_root / safe_name(feature_set) / f"seed_{seed}" / "result.json"
            if path.exists():
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
                candidates.append((result["val_metrics"]["macro_f1"], feature_set))
        if not candidates:
            continue
        _, selected = max(candidates, key=lambda item: (item[0], item[1]))
        combined = f"best_dgst_legacy:{selected}"
        seed_dir = results_root / "best_dgst_plus_legacy_all" / f"seed_{seed}"
        result_path = seed_dir / "result.json"
        if result_path.exists() and not args.force:
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = train_one_seed(rows, combined, seed, str(seed_dir), cfg, args.device)
        result["selected_dgst_feature_set"] = selected
        best_results.append(result)
    if best_results:
        all_results["best_dgst_plus_legacy_all"] = aggregate_seed_results(best_results)
        all_results["best_dgst_plus_legacy_all"]["selected_by_seed"] = {
            str(result["seed"]): result["selected_dgst_feature_set"] for result in best_results
        }

    summary_path = results_root / "summary_mean_std.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, summary_path)
    print(f"[train_qa_probes] Summary: {summary_path}")


if __name__ == "__main__":
    main()
