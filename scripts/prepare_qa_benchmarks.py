#!/usr/bin/env python3
"""Create the immutable, model-shared POPE and CLEVR-Exist question splits."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import (
    prepare_amber_discriminative,
    prepare_clevr_exist,
    prepare_pope,
)
from utils.config_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("pope", "clevr", "amber", "all"), default="all"
    )
    parser.add_argument("--config")
    parser.add_argument("--pope-dir")
    parser.add_argument("--coco-image-dir")
    parser.add_argument("--clevr-root")
    parser.add_argument("--amber-root")
    parser.add_argument("--output-root")
    parser.add_argument("--clevr-dataset-name")
    parser.add_argument("--clevr-train-questions", type=int)
    parser.add_argument("--clevr-val-questions", type=int)
    parser.add_argument("--clevr-test-questions", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config).get("qa_benchmarks", {}) if args.config else {}
    split_protocol = str(cfg.get("split", "strict_image_level_82"))
    if split_protocol != "strict_image_level_82":
        raise ValueError(
            "QA preparation requires qa_benchmarks.split="
            "strict_image_level_82"
        )
    pope_dir = args.pope_dir or cfg.get("pope_dir")
    coco_image_dir = args.coco_image_dir or cfg.get("coco_image_dir")
    clevr_root = args.clevr_root or cfg.get("clevr_root")
    amber_root = args.amber_root or cfg.get("amber_root")
    output_root = args.output_root or cfg.get("prepared_root")
    seed = int(args.seed if args.seed is not None else cfg.get("split_seed", 42))
    if not output_root:
        raise ValueError("prepared output root is required by --output-root or YAML")
    if args.dataset in ("pope", "all"):
        if not pope_dir or not coco_image_dir:
            raise ValueError("POPE and COCO image paths are required by CLI or YAML")
        rows = prepare_pope(
            pope_dir,
            coco_image_dir,
            os.path.join(output_root, "pope"),
            seed,
        )
        print(f"[prepare] POPE: {len(rows)} questions")
    if args.dataset in ("clevr", "all"):
        if not clevr_root:
            raise ValueError("CLEVR root is required by CLI or YAML")
        dataset_name = str(
            args.clevr_dataset_name
            or cfg.get("clevr_dataset_name", "clevr_exist_9k")
        )
        train_count = int(
            args.clevr_train_questions
            if args.clevr_train_questions is not None
            else cfg.get("clevr_train_questions", 7200)
        )
        val_count = int(
            args.clevr_val_questions
            if args.clevr_val_questions is not None
            else cfg.get("clevr_val_questions", 0)
        )
        test_count = int(
            args.clevr_test_questions
            if args.clevr_test_questions is not None
            else cfg.get("clevr_test_questions", 1800)
        )
        rows = prepare_clevr_exist(
            clevr_root,
            os.path.join(output_root, dataset_name),
            seed,
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
            dataset_name=dataset_name,
        )
        print(f"[prepare] CLEVR {dataset_name}: {len(rows)} questions")
    if args.dataset in ("amber", "all"):
        if not amber_root:
            raise ValueError("AMBER root is required by CLI or YAML")
        rows = prepare_amber_discriminative(
            amber_root,
            os.path.join(output_root, "amber_discriminative"),
            seed,
        )
        print(f"[prepare] AMBER discriminative: {len(rows)} questions")


if __name__ == "__main__":
    main()
