#!/usr/bin/env python3
"""Report original-model yes/no metrics from labels.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import load_jsonl


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    y_true = [1 if row["gt_answer"] == "yes" else 0 for row in rows]
    y_pred = [1 if row.get("prediction") == "yes" else 0 for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "count": len(rows),
        "accuracy": float(accuracy_score([row["label"] for row in rows], [1] * len(rows))),
        "precision_yes": float(precision),
        "recall_yes": float(recall),
        "f1_yes": float(f1),
        "yes_ratio": sum(row.get("prediction") == "yes" for row in rows) / len(rows),
        "invalid_ratio": sum(row.get("prediction") is None for row in rows) / len(rows),
    }


def grouped(rows: list[dict], key: str) -> dict:
    buckets = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            buckets[str(value)].append(row)
    return {value: metrics(items) for value, items in sorted(buckets.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    rows = load_jsonl(run_dir / "labels.jsonl")
    enriched = rows
    result = {
        "overall": metrics(enriched),
        "source_split": grouped(enriched, "source_split"),
        "gt_answer": grouped(enriched, "gt_answer"),
        "error_type": grouped(enriched, "error_type"),
        "question_family_index": grouped(enriched, "question_family_index"),
    }
    path = run_dir / "raw_model_metrics.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    print(path)


if __name__ == "__main__":
    main()
