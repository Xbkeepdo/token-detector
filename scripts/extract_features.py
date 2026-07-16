#!/usr/bin/env python3
"""Extract DGST-T features for all labeled object tokens."""

import argparse
import copy
import glob
import os
import sys
from typing import Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import get_context

from data.coco_loader import load_coco_samples
from utils.io_utils import load_json, load_pkl, save_pkl


FOUR_GATE_METHODS = (
    "hpre_raw_logit_gauss",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "raw_attention",
)


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
    p.add_argument(
        "--prompt",
        default=None,
        help="Caption instruction. Overrides the model prompt when provided.",
    )
    p.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional per-image processor pixel cap; unset keeps native preprocessing.",
    )
    p.add_argument(
        "--extraction-mode",
        choices=("all", "method_only", "ads_cgc_only", "baseline_only"),
        default=None,
        help=(
            "Override run.extraction_mode from YAML. baseline_only delegates to "
            "scripts/extract_baselines.py so root features.pkl is untouched."
        ),
    )
    p.add_argument(
        "--dgst-branches",
        nargs="+",
        choices=FOUR_GATE_METHODS,
        default=None,
        help="Extract only the selected DGST target-comparison branches.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from features.extractor import extract_features_for_dataset
    from models import build_model
    from utils.config_utils import (
        extraction_mode_flags,
        get_dataset_cfg,
        get_dgst_t_cfg,
        get_model_cfg,
        load_config,
    )

    config = load_config(args.config)
    extraction_mode = _resolve_extraction_mode(config, args.extraction_mode)
    if extraction_mode == "baseline_only":
        baseline_config = (config.get("feature_extraction") or {}).get(
            "baseline", {}
        )
        if isinstance(baseline_config, dict) and not bool(
            baseline_config.get("enabled", True)
        ):
            raise ValueError(
                "run.extraction_mode=baseline_only but baseline.enabled=false"
            )
        _run_baseline_only(args)
        return

    model_cfg   = get_model_cfg(config, args.model)
    if args.max_pixels is not None:
        if args.max_pixels <= 0:
            raise ValueError("--max-pixels must be a positive integer")
        model_cfg["max_pixels"] = int(args.max_pixels)
    dataset_cfg = get_dataset_cfg(config)
    feature_cfg = copy.deepcopy(config.get("feature_extraction") or {})
    if extraction_mode is not None:
        flags = extraction_mode_flags(extraction_mode)
        for family in ("method", "ads_cgc", "baseline"):
            section = feature_cfg.get(family)
            if isinstance(section, dict):
                configured = bool(section.get("enabled", True))
            else:
                configured = True if section is None else bool(section)
                section = {}
                feature_cfg[family] = section
            section["enabled"] = bool(flags[family] and configured)
    # Workers receive this local resolved copy; the source YAML is never
    # rewritten and old commands without --extraction-mode retain its switches.
    config["feature_extraction"] = feature_cfg
    dgst_t_cfg = copy.deepcopy(get_dgst_t_cfg(config))
    if args.dgst_branches is not None:
        selected_branches = list(dict.fromkeys(args.dgst_branches))
        selected_set = set(selected_branches)
        dgst_t_cfg["four_gate_methods"] = selected_branches
        dgst_t_cfg["branches"] = {
            method: method in selected_set for method in FOUR_GATE_METHODS
        }
    baseline_section = dict(feature_cfg.get("baseline") or {})
    baseline_enabled = bool(baseline_section.get("enabled", False))
    baseline_subdir = str(baseline_section.get("output_subdir", "baseline"))
    prompt = str(
        args.prompt
        or model_cfg.get("prompt")
        or "Describe this image."
    )

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
    generations_path = os.path.join(args.output_dir, "generations.json")
    raw_generations = (
        load_json(generations_path) if os.path.exists(generations_path) else {}
    )
    generation_results = {int(key): value for key, value in raw_generations.items()}
    labeled_sample_count = len(samples)
    samples = [
        sample
        for sample in samples
        if _has_extractable_object_spans(
            labeling_results[int(sample["image_id"])],
            generation_results.get(int(sample["image_id"])),
        )
    ]
    skipped = labeled_sample_count - len(samples)
    print(
        f"[Extract] Using {len(samples)} extractable labeled COCO samples"
        + (f" ({skipped} have no valid object-token span)." if skipped else ".")
    )

    output_path = os.path.join(args.output_dir, "features.pkl")
    devices = args.feature_devices or [args.device]
    if args.resume:
        root_part_paths = sorted(
            glob.glob(os.path.join(args.output_dir, "features.part*.pkl"))
        )
        baseline_output_path = None
        baseline_part_paths = []
        if baseline_enabled:
            baseline_dir = os.path.join(args.output_dir, baseline_subdir)
            baseline_output_path = os.path.join(baseline_dir, "features.pkl")
            baseline_part_paths = sorted(
                glob.glob(os.path.join(baseline_dir, "features.part*.pkl"))
            )
        pending_samples, complete_count = _pending_samples_for_resume(
            samples=samples,
            root_output_path=output_path,
            root_part_paths=root_part_paths,
            baseline_output_path=baseline_output_path,
            baseline_part_paths=baseline_part_paths,
        )
        if not pending_samples:
            print(
                "[Extract] Resume — root"
                + (" and baseline" if baseline_enabled else "")
                + f" features already cover all {len(samples)} extractable "
                "images; skipping model loading."
            )
            return
        print(
            f"[Extract] Resume — {complete_count} images complete, "
            f"{len(pending_samples)} pending."
        )
        samples = pending_samples

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
            cfg_feature_extraction=feature_cfg,
            prompt=prompt,
            full_config=config,
            baseline_enabled=baseline_enabled,
            baseline_subdir=baseline_subdir,
            output_dir=args.output_dir,
            output_path=output_path,
            devices=devices,
            resume=args.resume,
        )
    else:
        print(f"[Extract] Loading model '{args.model}' on {devices[0]} …")
        wrapper = build_model(args.model, model_cfg, device=devices[0])
        if baseline_enabled:
            from features.baseline import BaselineRuntime, baseline_config

            baseline_dir = os.path.join(args.output_dir, baseline_subdir)
            baseline_output_path = os.path.join(baseline_dir, "features.pkl")
            with BaselineRuntime(
                wrapper=wrapper,
                methods=baseline_section.get("methods", "all"),
                baseline_dir=baseline_dir,
                config=baseline_config(config),
                device=devices[0],
                resume=args.resume,
            ) as baseline_runtime:
                extract_features_for_dataset(
                    model_wrapper=wrapper,
                    coco_samples=samples,
                    labeling_results=labeling_results,
                    cfg_dgst_t=dgst_t_cfg,
                    output_path=output_path,
                    resume=args.resume,
                    prompt=prompt,
                    cfg_feature_extraction=feature_cfg,
                    baseline_runtime=baseline_runtime,
                    baseline_output_path=baseline_output_path,
                )
        else:
            extract_features_for_dataset(
                model_wrapper=wrapper,
                coco_samples=samples,
                labeling_results=labeling_results,
                cfg_dgst_t=dgst_t_cfg,
                output_path=output_path,
                resume=args.resume,
                prompt=prompt,
                cfg_feature_extraction=feature_cfg,
            )
    print(f"[Extract] Features saved to {output_path}")


def _resolve_extraction_mode(
    config: dict,
    cli_value: Optional[str],
) -> Optional[str]:
    if cli_value is not None:
        return str(cli_value).strip().lower()
    configured = (config.get("run") or {}).get("extraction_mode")
    if configured in (None, ""):
        return None
    return str(configured).strip().lower()


def _run_baseline_only(args) -> None:
    """Keep the three-stage shell while preserving baseline output isolation."""
    import subprocess

    command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "extract_baselines.py"),
        "--model",
        args.model,
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
    ]
    if args.feature_devices:
        command.extend(["--feature-devices", *args.feature_devices])
    if args.prompt is not None:
        command.extend(["--prompt", args.prompt])
    if args.max_pixels is not None:
        command.extend(["--max-pixels", str(args.max_pixels)])
    if args.num_images is not None:
        command.extend(["--num-images", str(args.num_images)])
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)


def _parallel_extract(
    *,
    model_key: str,
    model_cfg: dict,
    samples: list[dict],
    labeling_results: dict[int, dict],
    cfg_dgst_t: dict,
    cfg_feature_extraction: dict,
    prompt: str,
    full_config: dict,
    baseline_enabled: bool,
    baseline_subdir: str,
    output_dir: str,
    output_path: str,
    devices: list[str],
    resume: bool,
) -> None:
    part_paths = [
        os.path.join(output_dir, f"features.part{worker_id}.pkl")
        for worker_id in range(len(devices))
    ]
    baseline_dir = os.path.join(output_dir, baseline_subdir)
    baseline_output_path = os.path.join(baseline_dir, "features.pkl")
    baseline_part_paths = [
        os.path.join(baseline_dir, f"features.part{worker_id}.pkl")
        for worker_id in range(len(devices))
    ]
    pending_samples = samples
    if not pending_samples:
        return

    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending_samples):
        chunks[index % len(devices)].append(sample)

    ctx = get_context("spawn")
    jobs = []
    with ctx.Pool(processes=len(devices)) as pool:
        for worker_id, (device, chunk, part_path, baseline_part_path) in enumerate(
            zip(devices, chunks, part_paths, baseline_part_paths)
        ):
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
                            cfg_feature_extraction,
                            prompt,
                            full_config,
                            baseline_enabled,
                            baseline_dir,
                            baseline_part_path,
                            len(devices) > 1,
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
    if baseline_enabled:
        _merge_feature_parts(
            output_path=baseline_output_path,
            part_paths=baseline_part_paths,
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
    cfg_feature_extraction: dict,
    prompt: str,
    full_config: dict,
    baseline_enabled: bool,
    baseline_dir: str,
    baseline_part_path: str,
    parallel: bool,
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
    if baseline_enabled:
        from features.baseline import BaselineRuntime, baseline_config

        baseline_section = dict(cfg_feature_extraction.get("baseline") or {})
        with BaselineRuntime(
            wrapper=wrapper,
            methods=baseline_section.get("methods", "all"),
            baseline_dir=baseline_dir,
            config=baseline_config(full_config),
            device=device,
            worker_id=worker_id,
            parallel=parallel,
            resume=resume,
        ) as baseline_runtime:
            extract_features_for_dataset(
                model_wrapper=wrapper,
                coco_samples=samples,
                labeling_results=labeling_results,
                cfg_dgst_t=cfg_dgst_t,
                output_path=part_path,
                resume=resume,
                prompt=prompt,
                cfg_feature_extraction=cfg_feature_extraction,
                baseline_runtime=baseline_runtime,
                baseline_output_path=baseline_part_path,
            )
    else:
        extract_features_for_dataset(
            model_wrapper=wrapper,
            coco_samples=samples,
            labeling_results=labeling_results,
            cfg_dgst_t=cfg_dgst_t,
            output_path=part_path,
            resume=resume,
            prompt=prompt,
            cfg_feature_extraction=cfg_feature_extraction,
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


def _pending_samples_for_resume(
    *,
    samples: list[dict],
    root_output_path: str,
    root_part_paths: list[str],
    baseline_output_path: Optional[str] = None,
    baseline_part_paths: Optional[list[str]] = None,
) -> tuple[list[dict], int]:
    root_done = _done_image_ids(root_output_path, root_part_paths)
    if baseline_output_path is not None:
        baseline_done = _done_image_ids(
            baseline_output_path,
            baseline_part_paths or [],
        )
        done_image_ids = root_done & baseline_done
    else:
        done_image_ids = root_done
    pending = [
        sample
        for sample in samples
        if int(sample["image_id"]) not in done_image_ids
    ]
    return pending, len(samples) - len(pending)


def _has_extractable_object_spans(
    label_info: dict,
    generation: Optional[dict] = None,
) -> bool:
    """Ignore images that can never produce an object-token feature record."""
    if not isinstance(label_info, dict) or not label_info.get("generated_text"):
        return False
    response_ids = []
    if isinstance(generation, dict):
        response_ids = generation.get("response_token_ids") or []
    response_length = len(response_ids)
    for span in label_info.get("object_token_spans") or []:
        if not isinstance(span, dict):
            continue
        token_indices = span.get("token_indices") or []
        if not token_indices:
            continue
        if response_length and not all(
            0 <= int(index) < response_length for index in token_indices
        ):
            continue
        return True
    return False


def _feature_key(feat: dict) -> tuple:
    return (
        int(feat.get("image_id", -1)),
        int(feat.get("response_token_idx", -1)),
        str(feat.get("token_str", "")),
    )


if __name__ == "__main__":
    main()
