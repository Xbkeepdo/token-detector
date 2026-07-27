#!/usr/bin/env python3
"""Build a fixed balanced subset from completed exact-EMD QA feature shards."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import atomic_write_json, atomic_write_jsonl, load_jsonl
from utils.qa_paths import generations_path_for_benchmark_dir, locate_qa_generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-run-dir", required=True)
    parser.add_argument("--exact-generations")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--destination-run-dir", required=True)
    parser.add_argument("--destination-generations")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exact_root = Path(args.exact_run_dir)
    features = {}
    for path in sorted(glob.glob(str(exact_root / "features.parts" / "part-*.pkl"))):
        with open(path, "rb") as handle:
            for row in pickle.load(handle):
                features[row["key"]] = row
    by_label = {
        label: sorted(
            (row for row in features.values() if int(row["label"]) == label),
            key=lambda row: row["key"],
        )
        for label in (0, 1)
    }
    rng = random.Random(args.seed)
    selected = []
    for label in (0, 1):
        if len(by_label[label]) < args.per_class:
            raise ValueError(f"label={label}: need {args.per_class}, have {len(by_label[label])}")
        selected.extend(rng.sample(by_label[label], args.per_class))
    selected.sort(key=lambda row: row["key"])
    keys = {row["key"] for row in selected}

    questions = [row for row in load_jsonl(args.questions) if row["key"] in keys]
    exact_generations = locate_qa_generations(exact_root, args.exact_generations)
    generations = [
        row for row in load_jsonl(exact_generations) if row["key"] in keys
    ]
    labels = [row for row in load_jsonl(exact_root / "labels.jsonl") if row["key"] in keys]
    expected = args.per_class * 2
    for name, rows in (("questions", questions), ("generations", generations), ("labels", labels)):
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(rows)}")

    prepared = Path(args.prepared_root) / "pope"
    destination = Path(args.destination_run_dir)
    prepared.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(prepared / "questions.jsonl", questions)
    with open(exact_root / "image_splits.json", encoding="utf-8") as handle:
        import json
        atomic_write_json(prepared / "image_splits.json", json.load(handle))
    destination_generations = Path(
        args.destination_generations
        or generations_path_for_benchmark_dir(destination)
    )
    atomic_write_jsonl(destination_generations, generations)
    atomic_write_jsonl(destination / "labels.jsonl", labels)
    with open(destination / "exact_features.pkl.tmp", "wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(destination / "exact_features.pkl.tmp", destination / "exact_features.pkl")
    atomic_write_json(
        destination / "sample_manifest.json",
        {"seed": args.seed, "per_class": args.per_class, "keys": sorted(keys)},
    )
    print({"selected": expected, "real": args.per_class, "hallucination": args.per_class})


if __name__ == "__main__":
    main()
