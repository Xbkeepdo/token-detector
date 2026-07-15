#!/usr/bin/env python3
"""Validate QA artifacts for deduplication, leakage, completeness, and finite layers."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import load_jsonl
from features.dgst_t import COST_VARIANT_RISK_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--require-object-cgc", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_dir)
    generations = load_jsonl(root / "generations.jsonl")
    labels = load_jsonl(root / "labels.jsonl")
    with open(root / "features.pkl", "rb") as handle:
        features = pickle.load(handle)
    for name, rows in (("generations", generations), ("labels", labels), ("features", features)):
        keys = [row["key"] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{name} contains duplicate keys")
        if args.expected is not None and len(rows) != args.expected:
            raise ValueError(f"{name}: expected {args.expected}, got {len(rows)}")
    if {row["key"] for row in features} != {row["key"] for row in labels}:
        raise ValueError("features/labels key sets differ")
    layer_shapes = Counter()
    for row in features:
        if any(str(key).lower().startswith("gt") for key in row):
            raise ValueError(f"GT field leaked into features: {row['key']}")
        _finite(row, row["key"])
        for target, values in row["targets"].items():
            lengths = {len(values[name]) for name in (*COST_VARIANT_RISK_KEYS, "hprecosine")}
            if len(lengths) != 1:
                raise ValueError(f"Layer length mismatch: {row['key']} {target} {lengths}")
            layer_shapes[(target, next(iter(lengths)))] += 1
        if args.require_object_cgc and row.get("object_cgc_per_layer") is None:
            raise ValueError(f"Missing object-position CGC: {row['key']}")
    print({
        "generations": len(generations),
        "labels": len(labels),
        "features": len(features),
        "labels_by_class": dict(Counter(row["label"] for row in labels)),
        "errors": dict(Counter(row["error_type"] for row in labels)),
        "target_layer_shapes": {str(key): value for key, value in layer_shapes.items()},
    })


def _finite(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value: {path}={value}")


if __name__ == "__main__":
    main()
