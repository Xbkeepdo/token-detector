#!/usr/bin/env python3
"""Run generation, labeling, and second-forward extraction for QA benchmarks."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import atomic_write_json, load_jsonl
from features.qa_extractor import extract_questions, generate_questions, label_generations
from models import build_model
from utils.config_utils import get_ads_cfg, get_cgc_cfg, get_model_cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("pope", "clevr_exist_5k"), required=True)
    parser.add_argument("--config", default="configs/qa_benchmarks_server_fj01.yaml")
    parser.add_argument("--prepared-root", default="/root/rivermind-data/dataset/qa_benchmarks")
    parser.add_argument("--output-root", default="/root/rivermind-data/project/token-detector/outputs/qa_benchmarks")
    parser.add_argument("--stage", choices=("generate", "label", "extract", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--feature-shard-size", type=int, default=25)
    parser.add_argument("--skip-object-cgc", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    questions_path = os.path.join(args.prepared_root, args.dataset, "questions.jsonl")
    questions = load_jsonl(questions_path)
    if args.limit is not None:
        questions = questions[: args.limit]
    output_dir = os.path.join(args.output_root, args.model, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    splits_path = os.path.join(args.prepared_root, args.dataset, "image_splits.json")
    with open(splits_path, encoding="utf-8") as handle:
        import json
        atomic_write_json(os.path.join(output_dir, "image_splits.json"), json.load(handle))

    needs_model = args.stage in ("generate", "extract", "all")
    wrapper = None
    if needs_model:
        wrapper = build_model(args.model, get_model_cfg(config, args.model), device=args.device)
    if args.stage in ("generate", "all"):
        generate_questions(wrapper, args.model, questions, output_dir, args.checkpoint_every)
    if args.stage in ("label", "all"):
        label_generations(questions, output_dir, args.checkpoint_every)
    if args.stage in ("extract", "all"):
        extract_questions(
            wrapper,
            args.model,
            questions,
            output_dir,
            config["feature_extraction"]["dgst_t"],
            get_ads_cfg(config),
            get_cgc_cfg(config),
            args.feature_shard_size,
            include_object_cgc=not args.skip_object_cgc,
        )
    print(f"[qa_pipeline] Complete: {args.model}/{args.dataset}/{args.stage}")


if __name__ == "__main__":
    main()
