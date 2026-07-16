#!/usr/bin/env python3
"""Extract DGST-T features for all labeled object tokens."""

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import get_context

from data.coco_loader import load_coco_samples
from utils.generation_provenance import validate_generation_manifest
from utils.io_utils import load_json, load_pkl, save_pkl


FOUR_GATE_METHODS = (
    "hpre_raw_logit_gauss",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "raw_attention",
)

_FEATURE_FAMILIES_NEEDED = "_feature_families_needed"


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
        _prepare_baseline_only_provenance(args, config)
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
    effective_baseline_config = _combined_baseline_config(config)
    baseline_enabled = bool(baseline_section.get("enabled", False))
    controlled_baseline_enabled = bool(
        baseline_enabled
        and _controlled_baseline_enabled(effective_baseline_config)
    )
    baseline_subdir = str(
        effective_baseline_config.get("output_subdir", "baseline")
    )
    prompt = _resolve_prompt(
        cli_prompt=args.prompt,
        config=config,
        model_cfg=model_cfg,
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
    root_enabled = bool(
        _feature_section_enabled(feature_cfg.get("method"))
        or _feature_section_enabled(feature_cfg.get("ads_cgc"))
    )
    official_svar_enabled = bool(
        baseline_enabled and _official_svar_enabled(effective_baseline_config)
    )
    samples = [
        sample
        for sample in samples
        if _sample_has_any_extractable_protocol(
            label_info=labeling_results[int(sample["image_id"])],
            generation=generation_results.get(int(sample["image_id"])),
            controlled_enabled=bool(
                root_enabled or controlled_baseline_enabled
            ),
            official_enabled=official_svar_enabled,
        )
    ]
    skipped = labeled_sample_count - len(samples)
    print(
        f"[Extract] Using {len(samples)} extractable labeled COCO samples"
        + (
            f" ({skipped} have no valid span for any enabled protocol)."
            if skipped
            else "."
        )
    )

    output_path = os.path.join(args.output_dir, "features.pkl")
    devices = args.feature_devices or [args.device]
    root_manifest_path = os.path.join(args.output_dir, "features_manifest.json")
    root_provenance = _feature_provenance(
        artifact_family="root",
        model_key=args.model,
        model_cfg=model_cfg,
        prompt=prompt,
        feature_config={
            "method": feature_cfg.get("method"),
            "ads_cgc": feature_cfg.get("ads_cgc"),
            "dgst_t": dgst_t_cfg,
            "ads": feature_cfg.get("ads"),
            "cgc": feature_cfg.get("cgc"),
        },
        output_dir=args.output_dir,
        config=config,
    )
    root_artifacts = [
        output_path,
        *glob.glob(os.path.join(args.output_dir, "features.part*.pkl")),
    ]
    _validate_or_write_feature_manifest(
        root_manifest_path,
        root_provenance,
        artifact_paths=root_artifacts,
        resume=args.resume,
        adopt_legacy=bool((config.get("run") or {}).get("adopt_legacy_artifacts")),
    )

    baseline_output_path = None
    baseline_official_output_path = None
    if baseline_enabled:
        baseline_dir = os.path.join(args.output_dir, baseline_subdir)
        if controlled_baseline_enabled:
            baseline_output_path = os.path.join(baseline_dir, "features.pkl")
            baseline_provenance = _feature_provenance(
                artifact_family="baseline_controlled",
                model_key=args.model,
                model_cfg=model_cfg,
                prompt=prompt,
                feature_config=_controlled_baseline_feature_config(
                    effective_baseline_config
                ),
                output_dir=args.output_dir,
                config=config,
            )
            _validate_or_write_feature_manifest(
                os.path.join(baseline_dir, "features_manifest.json"),
                baseline_provenance,
                artifact_paths=[
                    baseline_output_path,
                    *glob.glob(os.path.join(baseline_dir, "features.part*.pkl")),
                    *glob.glob(
                        os.path.join(baseline_dir, "feature_parts", "worker_*.pkl")
                    ),
                ],
                resume=args.resume,
                adopt_legacy=bool(
                    (config.get("run") or {}).get("adopt_legacy_artifacts")
                ),
            )
        if _official_svar_enabled(effective_baseline_config):
            official_dir = os.path.join(baseline_dir, "svar_official")
            baseline_official_output_path = os.path.join(
                official_dir, "features.pkl"
            )
            official_provenance = _feature_provenance(
                artifact_family="baseline_svar_official",
                model_key=args.model,
                model_cfg=model_cfg,
                prompt=prompt,
                feature_config=_official_svar_feature_config(
                    effective_baseline_config
                ),
                output_dir=args.output_dir,
                config=config,
            )
            _validate_or_write_feature_manifest(
                os.path.join(official_dir, "features_manifest.json"),
                official_provenance,
                artifact_paths=[
                    baseline_official_output_path,
                    *glob.glob(
                        os.path.join(official_dir, "features.part*.pkl")
                    ),
                    *glob.glob(
                        os.path.join(official_dir, "feature_parts", "worker_*.pkl")
                    ),
                ],
                resume=args.resume,
                adopt_legacy=bool(
                    (config.get("run") or {}).get("adopt_legacy_artifacts")
                ),
            )
    if args.resume:
        root_part_paths = sorted(
            glob.glob(os.path.join(args.output_dir, "features.part*.pkl"))
        )
        baseline_part_paths = []
        baseline_official_part_paths = []
        if baseline_output_path is not None:
            baseline_dir = os.path.join(args.output_dir, baseline_subdir)
            baseline_part_paths = sorted(
                {
                    *glob.glob(os.path.join(baseline_dir, "features.part*.pkl")),
                    *glob.glob(
                        os.path.join(baseline_dir, "feature_parts", "worker_*.pkl")
                    ),
                }
            )
        if baseline_official_output_path is not None:
            baseline_dir = os.path.join(args.output_dir, baseline_subdir)
            baseline_official_part_paths = sorted(
                {
                    *glob.glob(
                        os.path.join(
                            baseline_dir,
                            "svar_official",
                            "features.part*.pkl",
                        )
                    ),
                    *glob.glob(
                        os.path.join(
                            baseline_dir,
                            "svar_official",
                            "feature_parts",
                            "worker_*.pkl",
                        )
                    ),
                }
            )
        # Consolidate every historical worker part before deciding which
        # images are complete. This makes resume safe when GPU count or
        # extraction mode changes after an interrupted run.
        if root_part_paths:
            _merge_feature_parts(
                output_path=output_path,
                part_paths=root_part_paths,
                resume=True,
            )
        if baseline_output_path is not None and baseline_part_paths:
            _merge_feature_parts(
                output_path=baseline_output_path,
                part_paths=baseline_part_paths,
                resume=True,
            )
        if (
            baseline_official_output_path is not None
            and baseline_official_part_paths
        ):
            _merge_feature_parts(
                output_path=baseline_official_output_path,
                part_paths=baseline_official_part_paths,
                resume=True,
            )
        official_required_ids = (
            {
                int(sample["image_id"])
                for sample in samples
                if _has_extractable_official_svar_spans(
                    labeling_results[int(sample["image_id"])],
                    generation_results.get(int(sample["image_id"])),
                )
            }
            if baseline_official_output_path is not None
            else None
        )
        pending_samples, complete_count = _pending_samples_for_resume(
            samples=samples,
            root_output_path=output_path,
            root_part_paths=root_part_paths,
            baseline_output_path=baseline_output_path,
            baseline_part_paths=baseline_part_paths,
            baseline_official_output_path=baseline_official_output_path,
            baseline_official_part_paths=baseline_official_part_paths,
            baseline_official_required_image_ids=official_required_ids,
        )
        if not pending_samples:
            if (
                baseline_official_output_path is not None
                and not os.path.exists(baseline_official_output_path)
            ):
                save_pkl([], baseline_official_output_path)
            print(
                "[Extract] Resume — root"
                + (" and baseline" if baseline_output_path is not None else "")
                + (
                    " and official SVAR"
                    if baseline_official_output_path is not None
                    else ""
                )
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
            with BaselineRuntime(
                wrapper=wrapper,
                methods=effective_baseline_config.get("methods", "all"),
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
                    baseline_official_output_path=baseline_official_output_path,
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


def _resolve_prompt(
    *,
    cli_prompt: Optional[str],
    config: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> str:
    """Resolve one prompt consistently for direct and coordinated runs."""

    run_cfg = config.get("run") or {}
    yaml_prompt = run_cfg.get("prompt") if isinstance(run_cfg, Mapping) else None
    return str(
        cli_prompt
        or yaml_prompt
        or model_cfg.get("prompt")
        or "Describe this image."
    )


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
    effective_baseline_config = _combined_baseline_config(full_config)
    controlled_baseline_enabled = _controlled_baseline_enabled(
        effective_baseline_config
    )
    baseline_part_paths: list[Optional[str]] = [
        (
            os.path.join(baseline_dir, f"features.part{worker_id}.pkl")
            if controlled_baseline_enabled
            else None
        )
        for worker_id in range(len(devices))
    ]
    baseline_official_dir = os.path.join(baseline_dir, "svar_official")
    baseline_official_output_path = os.path.join(
        baseline_official_dir, "features.pkl"
    )
    baseline_official_part_paths = [
        os.path.join(
            baseline_official_dir,
            f"features.part{worker_id}.pkl",
        )
        for worker_id in range(len(devices))
    ]
    official_svar_enabled = _official_svar_enabled(
        effective_baseline_config
    )
    pending_samples = samples
    if not pending_samples:
        return

    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending_samples):
        chunks[index % len(devices)].append(sample)

    ctx = get_context("spawn")
    jobs = []
    with ctx.Pool(processes=len(devices)) as pool:
        for worker_id, (
            device,
            chunk,
            part_path,
            baseline_part_path,
            baseline_official_part_path,
        ) in enumerate(
            zip(
                devices,
                chunks,
                part_paths,
                baseline_part_paths,
                baseline_official_part_paths,
            )
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
                            (
                                baseline_official_part_path
                                if official_svar_enabled
                                else None
                            ),
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
    if baseline_enabled and controlled_baseline_enabled:
        _merge_feature_parts(
            output_path=baseline_output_path,
            part_paths=[
                path for path in baseline_part_paths if path is not None
            ],
            resume=resume,
        )
    if baseline_enabled and official_svar_enabled:
        _merge_feature_parts(
            output_path=baseline_official_output_path,
            part_paths=baseline_official_part_paths,
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
    baseline_part_path: Optional[str],
    baseline_official_part_path: Optional[str],
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

        effective_baseline_config = _combined_baseline_config(full_config)
        with BaselineRuntime(
            wrapper=wrapper,
            methods=effective_baseline_config.get("methods", "all"),
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
                baseline_official_output_path=baseline_official_part_path,
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
    baseline_official_output_path: Optional[str] = None,
    baseline_official_part_paths: Optional[list[str]] = None,
    baseline_official_required_image_ids: Optional[set[int]] = None,
) -> tuple[list[dict], int]:
    """Annotate each pending image with exactly the artifact families it lacks.

    The annotations survive multi-GPU sharding.  A worker therefore never
    mistakes an absent worker-local part file for an absent final artifact and
    does not recompute DGST or controlled baselines when only official SVAR is
    missing.
    """

    root_done = _done_image_ids(root_output_path, root_part_paths)
    controlled_done = (
        _done_image_ids(baseline_output_path, baseline_part_paths or [])
        if baseline_output_path is not None
        else set()
    )
    official_done = (
        _done_image_ids(
            baseline_official_output_path,
            baseline_official_part_paths or [],
        )
        if baseline_official_output_path is not None
        else set()
    )
    official_required = baseline_official_required_image_ids

    pending: list[dict] = []
    complete_count = 0
    for sample in samples:
        image_id = int(sample["image_id"])
        official_is_required = bool(
            baseline_official_output_path is not None
            and (
                official_required is None
                or image_id in official_required
            )
        )
        needed = {
            "root": image_id not in root_done,
            "controlled": bool(
                baseline_output_path is not None
                and image_id not in controlled_done
            ),
            "official": bool(
                official_is_required and image_id not in official_done
            ),
        }
        if any(needed.values()):
            pending_sample = dict(sample)
            pending_sample[_FEATURE_FAMILIES_NEEDED] = needed
            pending.append(pending_sample)
        else:
            complete_count += 1
    return pending, complete_count


def _has_extractable_object_spans(
    label_info: dict,
    generation: Optional[dict] = None,
) -> bool:
    """Ignore images that can never produce an object-token feature record."""
    if not isinstance(label_info, dict) or not label_info.get("generated_text"):
        return False
    response_ids = _validated_generation_response_ids(label_info, generation)
    response_length = len(response_ids)
    for span in label_info.get("object_token_spans") or []:
        if not isinstance(span, dict):
            continue
        token_indices = span.get("token_indices") or []
        if not token_indices:
            continue
        try:
            parsed = [int(index) for index in token_indices]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-integer object token indices: {token_indices}") from exc
        if response_length and not all(0 <= index < response_length for index in parsed):
            raise ValueError(
                f"Object token indices {parsed} are outside response length "
                f"{response_length}"
            )
        return True
    return False


def _feature_section_enabled(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("enabled", False))
    return bool(value)


def _feature_model_config_payload(
    model_cfg: Mapping[str, Any],
    artifact_family: str,
) -> dict[str, Any]:
    """Remove run scheduling fields that do not change extracted tensors."""

    excluded = {"extraction_mode", "prompt", "generation_prompt"}
    if str(artifact_family).startswith("baseline_"):
        # DGST support scope changes root method tensors, never a baseline.
        excluded.add("dgst_t_support_scope")
    return {
        str(key): value
        for key, value in model_cfg.items()
        if str(key) not in excluded
    }


def _sample_has_any_extractable_protocol(
    *,
    label_info: dict,
    generation: Optional[dict],
    controlled_enabled: bool,
    official_enabled: bool,
) -> bool:
    """Keep images needed by either the shared or official sample protocol."""

    return bool(
        (
            controlled_enabled
            and _has_extractable_object_spans(label_info, generation)
        )
        or (
            official_enabled
            and _has_extractable_official_svar_spans(label_info, generation)
        )
    )


def _has_extractable_official_svar_spans(
    label_info: dict,
    generation: Optional[dict] = None,
) -> bool:
    if not isinstance(label_info, dict):
        return False
    response_ids = _validated_generation_response_ids(label_info, generation)
    from features.baseline.svar import prepare_official_svar_spans

    return bool(
        prepare_official_svar_spans(
            label_info.get("official_svar_samples") or [],
            response_ids,
        )
    )


def _validated_generation_response_ids(
    label_info: Mapping[str, Any],
    generation: Optional[Mapping[str, Any]],
) -> list[int]:
    image_id = label_info.get("image_id", "?")
    if not isinstance(generation, Mapping):
        raise RuntimeError(
            f"Image {image_id}: generations.json has no matching row. "
            "Schema-v2 extraction requires the actual generated response IDs."
        )
    generation_text = str(generation.get("generated_text", ""))
    labeling_text = str(label_info.get("generated_text", ""))
    if generation_text != labeling_text:
        raise RuntimeError(
            f"Image {image_id}: generated_text differs between "
            "generations.json and labeling.json."
        )
    raw_ids = generation.get("response_token_ids") or []
    if not raw_ids:
        raise RuntimeError(
            f"Image {image_id}: generations.json has no response_token_ids. "
            "Re-encoding generated_text is intentionally forbidden."
        )
    try:
        return [int(value) for value in raw_ids]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Image {image_id}: generations.json contains invalid "
            "response_token_ids."
        ) from exc


def _feature_key(feat: dict) -> tuple:
    return (
        int(feat.get("image_id", -1)),
        int(feat.get("response_token_idx", -1)),
        str(feat.get("token_str", "")),
    )


def _prepare_baseline_only_provenance(args, config: dict) -> None:
    """Protect the delegated standalone baseline extractor with provenance."""

    from utils.config_utils import get_model_cfg

    model_cfg = get_model_cfg(config, args.model)
    if args.max_pixels is not None:
        model_cfg["max_pixels"] = int(args.max_pixels)
    baseline_cfg = _combined_baseline_config(config)
    baseline_subdir = str(baseline_cfg.get("output_subdir", "baseline"))
    baseline_dir = os.path.join(args.output_dir, baseline_subdir)
    output_path = os.path.join(baseline_dir, "features.pkl")
    prompt = _resolve_prompt(
        cli_prompt=args.prompt,
        config=config,
        model_cfg=model_cfg,
    )
    common = {
        "model_key": args.model,
        "model_cfg": model_cfg,
        "prompt": prompt,
        "output_dir": args.output_dir,
        "config": config,
    }
    if _controlled_baseline_enabled(baseline_cfg):
        _validate_or_write_feature_manifest(
            os.path.join(baseline_dir, "features_manifest.json"),
            _feature_provenance(
                artifact_family="baseline_controlled",
                feature_config=_controlled_baseline_feature_config(
                    baseline_cfg
                ),
                **common,
            ),
            artifact_paths=[
                output_path,
                *glob.glob(os.path.join(baseline_dir, "features.part*.pkl")),
                *glob.glob(
                    os.path.join(baseline_dir, "feature_parts", "worker_*.pkl")
                ),
            ],
            resume=args.resume,
            adopt_legacy=bool(
                (config.get("run") or {}).get("adopt_legacy_artifacts")
            ),
        )
    if _official_svar_enabled(baseline_cfg):
        official_dir = os.path.join(baseline_dir, "svar_official")
        _validate_or_write_feature_manifest(
            os.path.join(official_dir, "features_manifest.json"),
            _feature_provenance(
                artifact_family="baseline_svar_official",
                feature_config=_official_svar_feature_config(baseline_cfg),
                **common,
            ),
            artifact_paths=[
                os.path.join(official_dir, "features.pkl"),
                *glob.glob(os.path.join(official_dir, "features.part*.pkl")),
                *glob.glob(
                    os.path.join(
                        official_dir,
                        "feature_parts",
                        "worker_*.pkl",
                    )
                ),
            ],
            resume=args.resume,
            adopt_legacy=bool(
                (config.get("run") or {}).get("adopt_legacy_artifacts")
            ),
        )


def _feature_provenance(
    *,
    artifact_family: str,
    model_key: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    feature_config: object,
    output_dir: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    labeling_path = Path(output_dir) / "labeling.json"
    if not labeling_path.exists():
        raise FileNotFoundError(labeling_path)
    labeling_manifest_path = Path(output_dir) / "labeling_manifest.json"
    if not labeling_manifest_path.exists():
        raise RuntimeError(
            "Refusing feature extraction because labeling_manifest.json is "
            "missing. Existing labeling may use the old broken token "
            "alignment; run the schema-v2 labeling stage first."
        )
    loaded = load_json(str(labeling_manifest_path))
    if not isinstance(loaded, Mapping):
        raise RuntimeError(
            f"Invalid labeling provenance manifest: {labeling_manifest_path}"
        )
    labeling_manifest: Mapping[str, Any] = loaded
    labeling_payload = load_json(str(labeling_path))
    labeling_sha256 = _stable_sha256(labeling_payload)
    manifest_labeling_sha256 = labeling_manifest.get("labeling_sha256")
    if manifest_labeling_sha256 != labeling_sha256:
        raise RuntimeError(
            "Refusing feature extraction because labeling.json does not match "
            "labeling_manifest.json. Re-run the schema-v2 labeling stage."
        )
    labeling_cfg = config.get("labeling") or {}
    if not isinstance(labeling_cfg, Mapping):
        labeling_cfg = {}
    schema = labeling_manifest.get("label_schema_version")
    locator = labeling_manifest.get("primary_locator")
    sample_unit = labeling_manifest.get("sample_unit")
    if str(schema) != "2":
        raise RuntimeError(
            "Refusing feature extraction from non-v2 labeling "
            f"(label_schema_version={schema!r}); run labeling again."
        )
    if str(locator) != "exact_response_offsets":
        raise RuntimeError(
            "Refusing feature extraction from a non-exact token locator "
            f"({locator!r}); expected 'exact_response_offsets'."
        )
    if str(sample_unit) != "first_canonical_mention":
        raise RuntimeError(
            "Refusing feature extraction from an incompatible sample unit "
            f"({sample_unit!r}); expected 'first_canonical_mention'."
        )
    if not isinstance(labeling_payload, Mapping) or any(
        not isinstance(row, Mapping) or int(row.get("schema_version", -1)) != 2
        for row in labeling_payload.values()
    ):
        raise RuntimeError(
            "Refusing feature extraction because one or more labeling rows "
            "are not schema version 2."
        )
    generation_path = Path(output_dir) / "generations.json"
    if not generation_path.exists():
        raise RuntimeError(
            "Refusing feature extraction because generations.json is missing. "
            "Schema-v2 token positions are defined over its actual response IDs."
        )
    generation_payload = load_json(str(generation_path))
    generation_manifest_payload = _validated_generation_manifest_payload(
        labeling_payload=labeling_payload,
        generation_payload=generation_payload,
    )
    expected_generation_sha256 = labeling_manifest.get("generation_sha256")
    actual_generation_sha256 = _stable_sha256(generation_manifest_payload)
    if expected_generation_sha256 != actual_generation_sha256:
        raise RuntimeError(
            "Refusing feature extraction because generations.json does not "
            "match labeling_manifest.json. Rebuild schema-v2 labeling."
        )
    generation_manifest_path = Path(output_dir) / "generation_manifest.json"
    if not generation_manifest_path.exists():
        raise RuntimeError(
            "Refusing schema-v2 feature extraction because "
            "generation_manifest.json is missing. Run the labeling/generation "
            "stage once to migrate generation provenance."
        )
    loaded_generation_manifest = load_json(str(generation_manifest_path))
    try:
        validated_generation_manifest = validate_generation_manifest(
            loaded_generation_manifest,
            model=str(model_key),
            model_cfg=model_cfg,
            prompt=str(prompt),
            generations=generation_payload,
            expected_image_ids={int(value) for value in labeling_payload},
        )
    except ValueError as exc:
        raise RuntimeError(
            "Refusing feature extraction because generation_manifest.json "
            "does not match the requested model, prompt, model configuration, "
            "or generation content."
        ) from exc
    if validated_generation_manifest["generation_sha256"] != actual_generation_sha256:
        raise RuntimeError(
            "generation_manifest.json and labeling_manifest.json disagree on "
            "the canonical generation content hash."
        )
    return {
        "manifest_version": 1,
        "artifact_family": str(artifact_family),
        "model": str(model_key),
        "model_config_sha256": _stable_sha256(
            _feature_model_config_payload(model_cfg, artifact_family)
        ),
        "prompt": str(prompt),
        "feature_config_sha256": _stable_sha256(feature_config),
        "labeling_config_sha256": _stable_sha256(labeling_cfg),
        "labeling_sha256": labeling_sha256,
        "labeling_file_sha256": _file_sha256(labeling_path),
        "labeling_manifest_sha256": _file_sha256(labeling_manifest_path),
        "labeling_schema_version": str(schema),
        "labeling_primary_locator": str(locator),
        "labeling_sample_unit": str(sample_unit),
        "generation_sha256": actual_generation_sha256,
        "generation_file_sha256": _file_sha256(generation_path),
        "generation_manifest_sha256": _file_sha256(generation_manifest_path),
    }


def _validated_generation_manifest_payload(
    *,
    labeling_payload: Mapping[str, Any],
    generation_payload: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(generation_payload, Mapping):
        raise RuntimeError("generations.json must contain a JSON object.")
    label_ids = {str(int(value)) for value in labeling_payload}
    generation_ids = {str(int(value)) for value in generation_payload}
    if generation_ids != label_ids:
        missing = sorted(label_ids - generation_ids)[:10]
        extra = sorted(generation_ids - label_ids)[:10]
        raise RuntimeError(
            "generations.json and labeling.json contain different image IDs; "
            f"missing={missing}, extra={extra}."
        )
    result: dict[str, dict[str, object]] = {}
    for image_id in sorted(label_ids, key=int):
        generation = generation_payload.get(image_id)
        label = labeling_payload.get(image_id)
        if not isinstance(generation, Mapping) or not isinstance(label, Mapping):
            raise RuntimeError(
                f"Invalid generation/label row for image {image_id}."
            )
        generated_text = str(generation.get("generated_text", ""))
        if generated_text != str(label.get("generated_text", "")):
            raise RuntimeError(
                f"Image {image_id}: generated_text differs between "
                "generations.json and labeling.json."
            )
        raw_ids = generation.get("response_token_ids") or []
        if not raw_ids:
            raise RuntimeError(
                f"Image {image_id}: generations.json has no response_token_ids."
            )
        try:
            token_ids = [int(value) for value in raw_ids]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Image {image_id}: invalid response_token_ids."
            ) from exc
        result[image_id] = {
            "generated_text": generated_text,
            "response_token_ids": token_ids,
        }
    return result


def _validate_or_write_feature_manifest(
    manifest_path: str,
    current: Mapping[str, Any],
    *,
    artifact_paths: list[str],
    resume: bool,
    adopt_legacy: bool,
) -> None:
    path = Path(manifest_path)
    retained = any(Path(value).exists() for value in artifact_paths)
    previous: Mapping[str, Any] = {}
    if path.exists():
        loaded = load_json(str(path))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Invalid feature manifest: {path}")
        previous = loaded
    elif retained:
        raise ValueError(
            f"Existing feature artifacts have no provenance manifest: {path}. "
            "They cannot be proven to use schema-v2 exact token alignment. "
            "Use a new output directory or remove and re-extract those "
            "features; legacy feature adoption is intentionally disabled."
        )

    if retained and not resume:
        raise FileExistsError(
            f"Feature artifacts already exist for {current['artifact_family']}; "
            "use --resume after provenance validation or start from a new "
            "output directory."
        )
    if retained and previous:
        mismatches = {
            key: (previous.get(key), value)
            for key, value in current.items()
            if previous.get(key) != value
        }
        if mismatches:
            action = "resume" if resume else "reuse"
            raise ValueError(
                f"Refusing to {action} incompatible {current['artifact_family']} "
                f"features: {mismatches}"
            )
    _atomic_write_json(path, dict(current))


def _official_svar_enabled(baseline_cfg: Mapping[str, Any]) -> bool:
    methods = baseline_cfg.get("methods", "all")
    method_values = [methods] if isinstance(methods, str) else list(methods or [])
    normalized_methods = {
        str(value).strip().lower().replace("-", "_") for value in method_values
    }
    if not ({"all", "svar"} & normalized_methods):
        return False
    svar_cfg = baseline_cfg.get("svar") or {}
    if not isinstance(svar_cfg, Mapping):
        return False
    protocols = svar_cfg.get("protocols", ["controlled"])
    protocol_values = (
        [protocols] if isinstance(protocols, str) else list(protocols or [])
    )
    return any(
        str(value).strip().lower().replace("-", "_")
        in {"official", "paper", "svar_official"}
        for value in protocol_values
    )


def _controlled_baseline_enabled(baseline_cfg: Mapping[str, Any]) -> bool:
    methods = baseline_cfg.get("methods", "all")
    method_values = [methods] if isinstance(methods, str) else list(methods or [])
    normalized = {
        str(value).strip().lower().replace("-", "_") for value in method_values
    }
    if "all" in normalized:
        return True
    non_svar = normalized - {"svar"}
    if non_svar:
        return True
    if "svar" not in normalized:
        return False
    svar_cfg = baseline_cfg.get("svar") or {}
    protocols = (
        svar_cfg.get("protocols", ["controlled"])
        if isinstance(svar_cfg, Mapping)
        else ["controlled"]
    )
    values = [protocols] if isinstance(protocols, str) else list(protocols or [])
    return any(
        str(value).strip().lower().replace("-", "_")
        in {"controlled", "fair", "shared"}
        for value in values
    )


def _controlled_baseline_feature_config(
    baseline_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Fingerprint only settings that affect controlled features/caches."""

    methods = _baseline_methods_for_protocol(baseline_cfg, "controlled")
    result: dict[str, Any] = {
        "fingerprint_schema_version": 1,
        "protocol": "controlled",
        "methods": methods,
    }
    if "metatoken" in methods:
        cfg = dict(baseline_cfg.get("metatoken") or {})
        result["metatoken"] = {
            "length_penalty": float(cfg.get("length_penalty", 1.0)),
            "attention_layer": cfg.get("attention_layer", -1),
        }
    if "svar" in methods:
        result["svar"] = _svar_extraction_config(baseline_cfg)
    if "dhcp" in methods:
        cfg = dict(baseline_cfg.get("dhcp") or {})
        grid = cfg.get("target_grid", cfg.get("spatial_size", (12, 12)))
        result["dhcp"] = {
            "target_grid": [int(value) for value in grid],
            "shard_size": int(cfg.get("shard_size", 256)),
        }
    if "projectaway" in methods:
        cfg = dict(baseline_cfg.get("projectaway") or {})
        result["projectaway"] = {
            "vocab_chunk_size": int(cfg.get("vocab_chunk_size", 8192)),
            "row_chunk_size": int(cfg.get("row_chunk_size", 256)),
        }
    if "halloc" in methods:
        cfg = dict(baseline_cfg.get("halloc") or {})
        result["halloc"] = {
            "clip_model": str(
                cfg.get("clip_model", "openai/clip-vit-base-patch32")
            ),
        }
    return result


def _official_svar_feature_config(
    baseline_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one mode-independent extraction fingerprint for official SVAR."""

    return {
        "fingerprint_schema_version": 1,
        "protocol": "official",
        "methods": _baseline_methods_for_protocol(baseline_cfg, "official"),
        "svar": _svar_extraction_config(baseline_cfg),
    }


def _baseline_methods_for_protocol(
    baseline_cfg: Mapping[str, Any],
    protocol: str,
) -> list[str]:
    methods = baseline_cfg.get("methods", "all")
    values = [methods] if isinstance(methods, str) else list(methods or [])
    normalized = [
        str(value).strip().lower().replace("-", "_") for value in values
    ]
    aliases = {"project_away": "projectaway", "meta_token": "metatoken"}
    normalized = [aliases.get(value, value) for value in normalized]
    if "all" in normalized:
        normalized = [
            "dhcp",
            "halloc",
            "metatoken",
            "projectaway",
            "svar",
        ]
    normalized = sorted(dict.fromkeys(normalized))
    svar_cfg = baseline_cfg.get("svar") or {}
    raw_protocols = (
        svar_cfg.get("protocols", ["controlled"])
        if isinstance(svar_cfg, Mapping)
        else ["controlled"]
    )
    protocol_values = (
        [raw_protocols]
        if isinstance(raw_protocols, str)
        else list(raw_protocols or [])
    )
    protocols = {
        str(value).strip().lower().replace("-", "_")
        for value in protocol_values
    }
    aliases_by_protocol = {
        "controlled": {"controlled", "fair", "shared"},
        "official": {"official", "paper", "svar_official"},
    }
    svar_enabled = bool(protocols & aliases_by_protocol[protocol])
    if protocol == "official":
        return ["svar"] if "svar" in normalized and svar_enabled else []
    return [
        method
        for method in normalized
        if method != "svar" or svar_enabled
    ]


def _svar_extraction_config(
    baseline_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = dict(baseline_cfg.get("svar") or {})
    return {
        "layer_start": cfg.get("layer_start", 5),
        "layer_end": cfg.get("layer_end", 19),
        "start_fraction": float(cfg.get("start_fraction", 0.15)),
        "end_fraction": float(cfg.get("end_fraction", 0.55)),
    }


def _combined_baseline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config.get("baselines") or {}))
    nested = (config.get("feature_extraction") or {}).get("baseline") or {}
    return _deep_merge(result, nested)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
