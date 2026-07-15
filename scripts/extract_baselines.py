#!/usr/bin/env python3
"""Extract paper baselines into ``OUTPUT/baseline`` independently of DGST."""

from __future__ import annotations

import argparse
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from typing import Sequence

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coco_loader import load_coco_samples
from features.baseline import (
    BaselineRuntime,
    baseline_config,
    normalize_baseline_methods,
)
from utils.config_utils import get_dataset_cfg, get_model_cfg, load_config
from utils.io_utils import append_pkl, load_json, load_pkl, save_pkl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-devices", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Caption instruction. Overrides the model prompt when provided.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional per-image processor pixel cap; unset keeps native preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    if args.max_pixels is not None:
        if args.max_pixels <= 0:
            raise ValueError("--max-pixels must be a positive integer")
        model_cfg["max_pixels"] = int(args.max_pixels)
    dataset_cfg = get_dataset_cfg(config)
    baseline_cfg = baseline_config(config)
    prompt = str(
        args.prompt
        or model_cfg.get("prompt")
        or "Describe this image."
    )
    methods = normalize_baseline_methods(baseline_cfg.get("methods", "all"))
    if not methods:
        raise ValueError("feature_extraction.baseline.methods cannot be empty")

    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / str(baseline_cfg.get("output_subdir", "baseline"))
    baseline_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "labeling.json"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    labeling = {int(key): value for key, value in load_json(str(label_path)).items()}
    samples = load_coco_samples(
        images_dir=os.path.join(dataset_cfg["coco_root"], "val2014"),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=None,
        seed=dataset_cfg["seed"],
    )
    samples = [sample for sample in samples if int(sample["image_id"]) in labeling]
    if args.num_images is not None:
        samples = samples[: int(args.num_images)]
    generations = _load_generations(output_dir)
    devices = args.feature_devices or [args.device]
    output_path = baseline_dir / "features.pkl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Baseline feature file already exists: {output_path}. "
            "Use --resume to reuse it."
        )
    if len(devices) == 1:
        _extract_worker(
            worker_id=0,
            model_key=args.model,
            model_cfg=model_cfg,
            device=devices[0],
            samples=samples,
            labeling=labeling,
            generations=generations,
            baseline_cfg=baseline_cfg,
            baseline_dir=str(baseline_dir),
            part_path=str(output_path),
            methods=methods,
            prompt=prompt,
            resume=args.resume,
            parallel=False,
        )
    else:
        _parallel_extract(
            model_key=args.model,
            model_cfg=model_cfg,
            devices=devices,
            samples=samples,
            labeling=labeling,
            generations=generations,
            baseline_cfg=baseline_cfg,
            baseline_dir=baseline_dir,
            output_path=output_path,
            methods=methods,
            prompt=prompt,
            resume=args.resume,
        )
    print(f"[BaselineExtract] saved {output_path}")


def _parallel_extract(
    *,
    model_key,
    model_cfg,
    devices,
    samples,
    labeling,
    generations,
    baseline_cfg,
    baseline_dir,
    output_path,
    methods,
    prompt,
    resume,
):
    parts_dir = baseline_dir / "feature_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [parts_dir / f"worker_{index}.pkl" for index in range(len(devices))]
    if not resume:
        existing_parts = [str(path) for path in part_paths if path.exists()]
        if existing_parts:
            raise FileExistsError(
                "Baseline worker parts already exist; use --resume: "
                + ", ".join(existing_parts)
            )
    done = _done_image_ids([output_path, *part_paths]) if resume else set()
    pending = [sample for sample in samples if int(sample["image_id"]) not in done]
    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending):
        chunks[index % len(devices)].append(sample)
    jobs = []
    with get_context("spawn").Pool(len(devices)) as pool:
        for worker_id, (device, chunk, part_path) in enumerate(
            zip(devices, chunks, part_paths)
        ):
            if not chunk:
                if not part_path.exists():
                    save_pkl([], str(part_path))
                continue
            jobs.append(
                pool.apply_async(
                    _extract_worker,
                    kwds={
                        "worker_id": worker_id,
                        "model_key": model_key,
                        "model_cfg": model_cfg,
                        "device": device,
                        "samples": chunk,
                        "labeling": labeling,
                        "generations": generations,
                        "baseline_cfg": baseline_cfg,
                        "baseline_dir": str(baseline_dir),
                        "part_path": str(part_path),
                        "methods": methods,
                        "prompt": prompt,
                        "resume": resume,
                        "parallel": True,
                    },
                )
            )
        for job in jobs:
            job.get()
    _merge_parts(output_path, part_paths, resume=resume)


def _extract_worker(
    *,
    worker_id: int,
    model_key: str,
    model_cfg: dict,
    device: str,
    samples: list[dict],
    labeling: dict[int, dict],
    generations: dict[int, dict],
    baseline_cfg: dict,
    baseline_dir: str,
    part_path: str,
    methods: Sequence[str],
    prompt: str,
    resume: bool,
    parallel: bool,
) -> None:
    from models import build_model

    baseline_root = Path(baseline_dir)
    existing = load_pkl(part_path) if resume and os.path.exists(part_path) else []
    done = {int(record["image_id"]) for record in existing}
    wrapper = build_model(model_key, model_cfg, device=device)
    runtime = BaselineRuntime(
        wrapper=wrapper,
        methods=methods,
        baseline_dir=baseline_root,
        config=baseline_cfg,
        device=device,
        worker_id=worker_id,
        parallel=parallel,
        resume=resume,
    )
    requirements = runtime.requirements

    try:
        for sample in samples:
            image_id = int(sample["image_id"])
            if image_id in done:
                continue
            label_info = labeling.get(image_id)
            if not label_info:
                continue
            spans = [
                span
                for span in label_info.get("object_token_spans", [])
                if span.get("token_indices")
            ]
            if not spans:
                continue
            response_ids = _response_ids(
                wrapper, image_id, label_info, generations
            )
            valid_spans = [
                span
                for span in spans
                if all(
                    0 <= int(index) < len(response_ids)
                    for index in span["token_indices"]
                )
            ]
            if not valid_spans:
                continue
            indices = [int(span["token_indices"][0]) for span in valid_spans]
            targets = [response_ids[index] for index in indices]
            with Image.open(sample["image_path"]) as source_image:
                image = source_image.convert("RGB")
            outputs = wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=response_ids,
                response_token_indices=indices,
                target_token_ids=targets,
                cfg_dgst_t=None,
                prompt=prompt,
                requirements=requirements,
            )
            if len(outputs) != len(valid_spans):
                raise RuntimeError(
                    f"Image {image_id}: wrapper returned {len(outputs)} outputs "
                    f"for {len(valid_spans)} object spans"
                )

            image_records = runtime.build_image_records(
                image=image,
                image_id=image_id,
                response_token_ids=response_ids,
                spans=valid_spans,
                model_outputs=outputs,
            )
            append_pkl(image_records, part_path)
            done.add(image_id)
    finally:
        runtime.close()


def _response_ids(wrapper, image_id, label_info, generations) -> list[int]:
    generation = generations.get(image_id, {})
    token_ids = generation.get("response_token_ids") or []
    if token_ids:
        return [int(value) for value in token_ids]
    generated_text = str(label_info.get("generated_text", ""))
    return [
        int(value)
        for value in wrapper.tokenizer.encode(generated_text, add_special_tokens=False)
    ]


def _load_generations(output_dir: Path) -> dict[int, dict]:
    path = output_dir / "generations.json"
    if not path.exists():
        return {}
    return {int(key): value for key, value in load_json(str(path)).items()}


def _done_image_ids(paths) -> set[int]:
    done = set()
    for path in paths:
        if os.path.exists(path):
            done.update(int(row["image_id"]) for row in load_pkl(str(path)))
    return done


def _merge_parts(output_path: Path, part_paths: Sequence[Path], *, resume: bool) -> None:
    rows, seen = [], set()
    sources = ([output_path] if resume and output_path.exists() else []) + list(part_paths)
    for path in sources:
        if not path.exists():
            continue
        for row in load_pkl(str(path)):
            key = (
                int(row["image_id"]),
                int(row["response_token_idx"]),
                str(row["token_str"]),
            )
            if key not in seen:
                rows.append(row)
                seen.add(key)
    save_pkl(rows, str(output_path))


if __name__ == "__main__":
    main()
