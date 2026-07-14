#!/usr/bin/env python3
"""Paired exact-EMD vs Sinkhorn risk comparison on a fixed QA subset."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.dgst_t import COST_VARIANT_RISK_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = args.run_dir
    with open(os.path.join(root, "exact_features.pkl"), "rb") as handle:
        exact = {row["key"]: row for row in pickle.load(handle)}
    with open(os.path.join(root, "features.pkl"), "rb") as handle:
        sinkhorn = {row["key"]: row for row in pickle.load(handle)}
    exact_count = len(exact)
    keys = sorted(exact.keys() & sinkhorn.keys())
    if not keys:
        raise ValueError("Exact/Sinkhorn key intersection is empty")
    labels = np.asarray([1 - int(exact[key]["label"]) for key in keys], dtype=np.int64)
    result = {
        "num_samples": len(keys),
        "exact_num_samples": exact_count,
        "sinkhorn_missing_samples": exact_count - len(keys),
        "positive_class": "hallucination",
        "sinkhorn_reg": sinkhorn[keys[0]].get("sinkhorn_reg"),
        "targets": {},
    }
    for target in exact[keys[0]]["targets"]:
        target_result = {}
        for risk in COST_VARIANT_RISK_KEYS:
            exact_matrix = np.asarray(
                [exact[key]["targets"][target][risk] for key in keys], dtype=np.float64
            )
            sinkhorn_matrix = np.asarray(
                [sinkhorn[key]["targets"][target][risk] for key in keys], dtype=np.float64
            )
            delta = sinkhorn_matrix - exact_matrix
            exact_flat, sinkhorn_flat = exact_matrix.ravel(), sinkhorn_matrix.ravel()
            target_result[risk] = {
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta ** 2))),
                "max_abs": float(np.max(np.abs(delta))),
                "mean_bias": float(np.mean(delta)),
                "mean_relative_abs": float(
                    np.mean(np.abs(delta) / np.maximum(np.abs(exact_matrix), 1e-8))
                ),
                "pearson": float(np.corrcoef(exact_flat, sinkhorn_flat)[0, 1]),
                "spearman": float(spearmanr(exact_flat, sinkhorn_flat).statistic),
                "exact_cv": cross_validated_linear_probe(exact_matrix, labels),
                "sinkhorn_cv": cross_validated_linear_probe(sinkhorn_matrix, labels),
            }
        result["targets"][target] = target_result

    hpre_delta = []
    for key in keys:
        for target in exact[key]["targets"]:
            hpre_delta.extend(
                np.asarray(sinkhorn[key]["targets"][target]["hprecosine"])
                - np.asarray(exact[key]["targets"][target]["hprecosine"])
            )
    result["hprecosine_max_abs_delta"] = float(np.max(np.abs(hpre_delta)))
    output = args.output or os.path.join(root, "exact_vs_sinkhorn.json")
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, output)
    print(json.dumps(summarize(result), ensure_ascii=False, indent=2))
    print(output)


def cross_validated_linear_probe(features, labels):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    predictions = np.zeros(len(labels), dtype=np.int64)
    for train, test in splitter.split(features, labels):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        )
        classifier.fit(features[train], labels[train])
        probabilities[test] = classifier.predict_proba(features[test])[:, 1]
        predictions[test] = probabilities[test] >= 0.5
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def summarize(result):
    summary = {
        "num_samples": result["num_samples"],
        "sinkhorn_missing_samples": result["sinkhorn_missing_samples"],
        "sinkhorn_reg": result["sinkhorn_reg"],
        "hprecosine_max_abs_delta": result["hprecosine_max_abs_delta"],
        "targets": {},
    }
    for target, risks in result["targets"].items():
        summary["targets"][target] = {
            "mean_mae": float(np.mean([row["mae"] for row in risks.values()])),
            "mean_pearson": float(np.mean([row["pearson"] for row in risks.values()])),
            "mean_spearman": float(np.mean([row["spearman"] for row in risks.values()])),
            "mean_exact_auc": float(np.mean([row["exact_cv"]["auroc"] for row in risks.values()])),
            "mean_sinkhorn_auc": float(np.mean([row["sinkhorn_cv"]["auroc"] for row in risks.values()])),
            "mean_exact_f1": float(np.mean([row["exact_cv"]["f1"] for row in risks.values()])),
            "mean_sinkhorn_f1": float(np.mean([row["sinkhorn_cv"]["f1"] for row in risks.values()])),
        }
    return summary


if __name__ == "__main__":
    main()
