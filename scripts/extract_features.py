#!/usr/bin/env python3
"""Extract DGST-T features for all labeled object tokens."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coco_loader import load_coco_samples
from features.extractor import extract_features_for_dataset
from models import build_model
from utils.config_utils import (
    load_config, get_model_cfg, get_dataset_cfg, get_dgst_t_cfg
)
from utils.io_utils import load_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--config",     default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    model_cfg   = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    dgst_t_cfg  = get_dgst_t_cfg(config)

    images_dir = os.path.join(dataset_cfg["coco_root"], "val2014")
    label_path = os.path.join(args.output_dir, "labeling.json")
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Labeling file not found: {label_path}\n"
            "Run generate_and_label.py first."
        )
    raw_labels = load_json(label_path)
    labeling_results = {int(k): v for k, v in raw_labels.items()}
    print(f"[Extract] Loaded labeling for {len(labeling_results)} images.")

    samples = load_coco_samples(
        images_dir=images_dir,
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=None,
        seed=args.seed if args.seed is not None else dataset_cfg["seed"],
    )
    labeled_ids = set(labeling_results)
    samples = [sample for sample in samples if int(sample["image_id"]) in labeled_ids]
    if args.num_images is not None and args.num_images < len(samples):
        samples = samples[:args.num_images]
    print(f"[Extract] Using {len(samples)} labeled COCO samples.")

    print(f"[Extract] Loading model '{args.model}' …")
    wrapper = build_model(args.model, model_cfg, device=args.device)

    output_path = os.path.join(args.output_dir, "features.pkl")
    extract_features_for_dataset(
        model_wrapper=wrapper,
        coco_samples=samples,
        labeling_results=labeling_results,
        cfg_dgst_t=dgst_t_cfg,
        output_path=output_path,
        resume=args.resume,
    )
    print(f"[Extract] Features saved to {output_path}")


if __name__ == "__main__":
    main()
