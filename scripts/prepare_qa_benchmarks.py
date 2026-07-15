#!/usr/bin/env python3
"""Create the immutable, model-shared POPE and CLEVR-Exist question splits."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import prepare_clevr_exist, prepare_pope


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("pope", "clevr", "all"), default="all")
    parser.add_argument("--pope-dir", default="/root/rivermind-data/dataset/pope")
    parser.add_argument("--coco-image-dir", default="/root/rivermind-data/dataset/coco/val2014")
    parser.add_argument("--clevr-root", default="/root/rivermind-data/dataset/CLEVR_v1.0")
    parser.add_argument("--output-root", default="/root/rivermind-data/dataset/qa_benchmarks")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset in ("pope", "all"):
        rows = prepare_pope(
            args.pope_dir,
            args.coco_image_dir,
            os.path.join(args.output_root, "pope"),
            args.seed,
        )
        print(f"[prepare] POPE: {len(rows)} questions")
    if args.dataset in ("clevr", "all"):
        rows = prepare_clevr_exist(
            args.clevr_root,
            os.path.join(args.output_root, "clevr_exist_5k"),
            args.seed,
        )
        print(f"[prepare] CLEVR official exist 5K subset: {len(rows)} questions")


if __name__ == "__main__":
    main()
