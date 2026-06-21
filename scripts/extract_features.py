#!/usr/bin/env python3
"""Extract DGST-T features for all labeled object tokens."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import get_context

from data.coco_loader import load_coco_samples
from utils.io_utils import load_json, load_pkl, save_pkl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--config",     default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--feature-devices", nargs="+", default=None,
                   help="Run feature extraction in parallel, one worker per device, e.g. cuda:0 cuda:1")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--seed",       type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    from features.extractor import extract_features_for_dataset
    from models import build_model
    from utils.config_utils import (
        load_config, get_model_cfg, get_dataset_cfg, get_dgst_t_cfg
    )

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

    output_path = os.path.join(args.output_dir, "features.pkl")
    devices = args.feature_devices or [args.device]
    if len(devices) > 1:
        print(
            f"[Extract] Running parallel feature extraction on "
            f"{', '.join(devices)} for {len(samples)} images."
        )
        _parallel_extract(
            model_key=args.model,
            model_cfg=model_cfg,
            samples=samples,
            labeling_results=labeling_results,
            cfg_dgst_t=dgst_t_cfg,
            output_dir=args.output_dir,
            output_path=output_path,
            devices=devices,
            resume=args.resume,
        )
    else:
        print(f"[Extract] Loading model '{args.model}' on {devices[0]} …")
        wrapper = build_model(args.model, model_cfg, device=devices[0])
        extract_features_for_dataset(
            model_wrapper=wrapper,
            coco_samples=samples,
            labeling_results=labeling_results,
            cfg_dgst_t=dgst_t_cfg,
            output_path=output_path,
            resume=args.resume,
        )
    print(f"[Extract] Features saved to {output_path}")


def _parallel_extract(
    *,
    model_key: str,
    model_cfg: dict,
    samples: list[dict],
    labeling_results: dict[int, dict],
    cfg_dgst_t: dict,
    output_dir: str,
    output_path: str,
    devices: list[str],
    resume: bool,
) -> None:
    part_paths = [
        os.path.join(output_dir, f"features.part{worker_id}.pkl")
        for worker_id in range(len(devices))
    ]
    done_image_ids = _done_image_ids(output_path, part_paths) if resume else set()
    pending_samples = [
        sample for sample in samples
        if int(sample["image_id"]) not in done_image_ids
    ]

    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending_samples):
        chunks[index % len(devices)].append(sample)

    ctx = get_context("spawn")
    jobs = []
    with ctx.Pool(processes=len(devices)) as pool:
        for worker_id, (device, chunk, part_path) in enumerate(zip(devices, chunks, part_paths)):
            if chunk:
                jobs.append(
                    pool.apply_async(
                        _extract_worker,
                        (
                            worker_id,
                            model_key,
                            model_cfg,
                            device,
                            chunk,
                            labeling_results,
                            cfg_dgst_t,
                            part_path,
                            resume,
                        ),
                    )
                )
            elif not os.path.exists(part_path):
                save_pkl([], part_path)

        for job in jobs:
            job.get()

    _merge_feature_parts(
        output_path=output_path,
        part_paths=part_paths,
        resume=resume,
    )


def _extract_worker(
    worker_id: int,
    model_key: str,
    model_cfg: dict,
    device: str,
    samples: list[dict],
    labeling_results: dict[int, dict],
    cfg_dgst_t: dict,
    part_path: str,
    resume: bool,
) -> str:
    from features.extractor import extract_features_for_dataset
    from models import build_model

    print(
        f"[Extract worker {worker_id}] Loading model '{model_key}' on {device} "
        f"for {len(samples)} images."
    )
    wrapper = build_model(model_key, model_cfg, device=device)
    extract_features_for_dataset(
        model_wrapper=wrapper,
        coco_samples=samples,
        labeling_results=labeling_results,
        cfg_dgst_t=cfg_dgst_t,
        output_path=part_path,
        resume=resume,
    )
    return part_path


def _merge_feature_parts(
    *,
    output_path: str,
    part_paths: list[str],
    resume: bool,
) -> None:
    merged = []
    seen = set()
    if resume and os.path.exists(output_path):
        for feat in load_pkl(output_path):
            key = _feature_key(feat)
            if key not in seen:
                merged.append(feat)
                seen.add(key)

    for part_path in part_paths:
        if not os.path.exists(part_path):
            continue
        for feat in load_pkl(part_path):
            key = _feature_key(feat)
            if key not in seen:
                merged.append(feat)
                seen.add(key)

    save_pkl(merged, output_path)


def _done_image_ids(output_path: str, part_paths: list[str]) -> set[int]:
    done = set()
    for path in [output_path, *part_paths]:
        if not os.path.exists(path):
            continue
        for feat in load_pkl(path):
            if "image_id" in feat:
                done.add(int(feat["image_id"]))
    return done


def _feature_key(feat: dict) -> tuple:
    return (
        int(feat.get("image_id", -1)),
        int(feat.get("response_token_idx", -1)),
        str(feat.get("token_str", "")),
    )


if __name__ == "__main__":
    main()
