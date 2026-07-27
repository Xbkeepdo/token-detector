#!/usr/bin/env python3
"""Generate/reuse COCO captions and build CHAIR-style token labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tqdm import tqdm

from data.coco_loader import load_coco_samples
from utils.generation_provenance import (
    GENERATION_MANIFEST_NAME,
    build_generation_manifest,
    build_generation_run_manifest,
    stable_sha256 as stable_generation_sha256,
    validate_generation_identity,
    validate_generation_manifest,
)
from utils.io_utils import load_json, save_json
from utils.split_utils import ensure_strict_82_split
from utils.token_alignment import (
    TokenAlignmentError,
    build_response_token_offsets,
    locate_first_token_id,
    token_indices_for_char_span,
    validate_token_surface,
)

_CHAIR_PATH = Path(__file__).with_name("coco_chair.py")
_CHAIR_MODULE_NAME = "coco_chair"
_SPEC = importlib.util.spec_from_file_location(_CHAIR_MODULE_NAME, _CHAIR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot import {_CHAIR_PATH}")
_CHAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_CHAIR_MODULE_NAME] = _CHAIR
try:
    _SPEC.loader.exec_module(_CHAIR)
except Exception:
    if sys.modules.get(_CHAIR_MODULE_NAME) is _CHAIR:
        del sys.modules[_CHAIR_MODULE_NAME]
    raise
CocoChairEvaluator = _CHAIR.CocoChairEvaluator
chair_summary = _CHAIR.chair_summary
iter_ground_truth_entries = _CHAIR.iter_ground_truth_entries
LABEL_HALLUCINATED = _CHAIR.LABEL_HALLUCINATED
LABEL_REAL = _CHAIR.LABEL_REAL
LABELING_SCHEMA_VERSION = 2
LABELING_MANIFEST_NAME = "labeling_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caption-file", default=None, help="Optional JSONL rows with image_id and caption.")
    parser.add_argument("--chair-cache", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-devices", nargs="+", default=None)
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
    parser.add_argument(
        "--reuse-generations-from",
        default=None,
        help=(
            "Copy generations.json (and image_splits.json when absent) from an "
            "older output into a new output. Existing target files are never "
            "overwritten and symlinks are never created."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.config_utils import (
        get_dataset_cfg,
        get_labeling_cfg,
        get_model_cfg,
        load_config,
        manifest_validation_enabled,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    labeling_cfg = get_labeling_cfg(config)
    validate_manifests = manifest_validation_enabled(config)
    adopt_legacy_generation = _legacy_generation_adoption_enabled(config)
    if args.max_pixels is not None:
        if args.max_pixels <= 0:
            raise ValueError("--max-pixels must be a positive integer")
        model_cfg["max_pixels"] = int(args.max_pixels)
    dataset_cfg = get_dataset_cfg(config)
    run_cfg = config.get("run") or {}
    prompt = str(
        args.prompt
        or (run_cfg.get("prompt") if isinstance(run_cfg, Mapping) else None)
        or model_cfg.get("prompt")
        or "Describe this image."
    )
    seed = args.seed if args.seed is not None else int(dataset_cfg["seed"])
    num_images = args.num_images or int(dataset_cfg["num_images"])

    samples = load_coco_samples(
        images_dir=os.path.join(dataset_cfg["coco_root"], "val2014"),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=num_images if args.caption_file is None else None,
        seed=seed,
    )
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    if args.reuse_generations_from:
        if args.caption_file:
            raise ValueError(
                "--reuse-generations-from cannot be combined with --caption-file"
            )
        _reuse_generation_artifacts(
            source_output=Path(args.reuse_generations_from),
            target_output=Path(args.output_dir),
            expected_image_ids=set(sample_by_id),
            model_key=args.model,
            model_cfg=model_cfg,
            prompt=prompt,
            resume=bool(args.resume),
            adopt_legacy=adopt_legacy_generation,
            validate_manifest=validate_manifests,
        )

    if args.caption_file:
        caption_rows = _load_caption_rows(Path(args.caption_file), sample_by_id)
        if num_images is not None and num_images < len(caption_rows):
            random.seed(seed)
            caption_rows = random.sample(caption_rows, num_images)
        selected_ids = {int(row["image_id"]) for row in caption_rows}
        selected_samples = [sample for sample in samples if int(sample["image_id"]) in selected_ids]
    else:
        caption_rows = []
        selected_samples = samples

    splits = _load_or_create_splits(
        output_dir=args.output_dir,
        samples=selected_samples,
        seed=seed,
        shared_splits_path=dataset_cfg.get("shared_split_path"),
    )

    generations_path = os.path.join(args.output_dir, "generations.json")
    labeling_path = os.path.join(args.output_dir, "labeling.json")
    labeling_manifest_path = os.path.join(args.output_dir, LABELING_MANIFEST_NAME)
    ground_truth_path = os.path.join(args.output_dir, "coco_ground_truth.jsonl")
    generation_shard_dir = os.path.join(args.output_dir, "generation_shards")
    expected_generation_ids = {
        int(sample["image_id"]) for sample in selected_samples
    }
    load_existing_generations = bool(args.resume or args.reuse_generations_from)
    if not load_existing_generations:
        # A fresh direct invocation must not append to shards from an older run.
        shutil.rmtree(generation_shard_dir, ignore_errors=True)
    generations = (
        load_json(generations_path)
        if load_existing_generations and os.path.exists(generations_path)
        else {}
    )
    existing_labeling = (
        load_json(labeling_path)
        if args.resume and os.path.exists(labeling_path)
        else {}
    )
    can_resume_labels = args.resume and args.caption_file is None
    generation_manifest = _validate_or_initialize_generation_run_manifest(
        output_dir=Path(args.output_dir),
        model_key=args.model,
        model_cfg=model_cfg,
        prompt=prompt,
        generations=generations,
        generation_shard_dir=Path(generation_shard_dir),
        expected_image_ids=expected_generation_ids,
        fresh_start=not load_existing_generations,
        adopt_legacy=adopt_legacy_generation,
        validate_manifest=validate_manifests,
    )
    if args.resume:
        _merge_generation_shards(generations, generation_shard_dir)
    if _all_generations_available(selected_samples, generations):
        generation_manifest = _validate_or_adopt_generation_manifest(
            output_dir=Path(args.output_dir),
            model_key=args.model,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=generations,
            expected_image_ids=expected_generation_ids,
            adopt_legacy=adopt_legacy_generation,
            validate_manifest=validate_manifests,
        )
    expected_labeling_manifest = _expected_labeling_manifest(
        samples=selected_samples,
        generations=generations,
        model_cfg=model_cfg,
        labeling_cfg=labeling_cfg,
        generation_manifest=generation_manifest,
    )
    if validate_manifests and can_resume_labels and existing_labeling:
        _validate_existing_labeling_for_resume(
            labeling=existing_labeling,
            manifest_path=labeling_manifest_path,
            expected_manifest=expected_labeling_manifest,
            samples=selected_samples,
            generations=generations,
            adopt_legacy=adopt_legacy_generation,
        )
    if can_resume_labels and _reuse_complete_labeling(
        samples=selected_samples,
        generations=generations,
        labeling=existing_labeling,
        generations_path=generations_path,
        summary_path=os.path.join(args.output_dir, "chair_summary.json"),
        manifest_path=labeling_manifest_path,
        expected_manifest=(
            expected_labeling_manifest if validate_manifests else None
        ),
        expected_sample_unit=str(labeling_cfg["sample_unit"]),
        validate_manifest=validate_manifests,
        ground_truth_path=ground_truth_path,
        adopt_legacy_ground_truth=adopt_legacy_generation,
        persist_generations=True,
    ):
        return

    evaluator = CocoChairEvaluator.from_cache(
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg.get("captions_file"),
        cache_path=args.chair_cache or os.path.join(args.output_dir, "chair.pkl"),
    )

    if args.caption_file:
        print(f"[COCO-CHAIR] Loading tokenizer for {model_cfg['hf_name']}")
        tokenizer = _load_tokenizer(model_cfg["hf_name"], model_key=args.model)
    else:
        caption_rows, tokenizer = _caption_rows_from_generation(
            args=args,
            model_cfg=model_cfg,
            samples=selected_samples,
            generations=generations,
            generations_path=generations_path,
            generation_shard_dir=generation_shard_dir,
            prompt=prompt,
        )

    labeling: dict[str, dict] = {}
    for row in tqdm(caption_rows, desc="COCO-CHAIR labeling"):
        image_id = int(row["image_id"])
        caption = str(row["caption"])
        token_ids = _generation_token_ids(
            generations,
            image_id,
            caption,
            tokenizer,
            provided_token_ids=row.get("response_token_ids"),
        )
        chair_info = evaluator.compute_chair_token(image_id, caption)
        spans = _chair_token_spans(
            evaluator=evaluator,
            tokenizer=tokenizer,
            image_id=image_id,
            caption=caption,
            token_ids=token_ids,
            chair_info=chair_info,
            alignment_failure_policy=str(
                labeling_cfg.get("alignment_failure_policy", "error")
            ),
        )
        official_svar_samples = (
            _build_official_svar_samples(
                tokenizer=tokenizer,
                token_ids=token_ids,
                chair_info=chair_info,
            )
            if bool(labeling_cfg.get("save_svar_official_samples", True))
            else []
        )
        generations[str(image_id)] = {
            "generated_text": caption,
            "response_token_ids": [int(token_id) for token_id in token_ids],
        }
        labeling[str(image_id)] = _compact_label_entry(
            image_id=image_id,
            caption=caption,
            spans=spans,
            chair_info=chair_info,
            official_svar_samples=official_svar_samples,
            sample_unit=str(labeling_cfg["sample_unit"]),
        )

    save_json(generations, generations_path)
    generation_manifest = build_generation_manifest(
        model=args.model,
        model_cfg=model_cfg,
        prompt=prompt,
        generations=generations,
        expected_image_ids={
            int(sample["image_id"]) for sample in selected_samples
        },
    )
    if validate_manifests:
        save_json(
            generation_manifest,
            os.path.join(args.output_dir, GENERATION_MANIFEST_NAME),
        )
    save_json(labeling, labeling_path)
    save_json(chair_summary(labeling), os.path.join(args.output_dir, "chair_summary.json"))
    completed_manifest = _expected_labeling_manifest(
        samples=selected_samples,
        generations=generations,
        model_cfg=model_cfg,
        labeling_cfg=labeling_cfg,
        generation_manifest=generation_manifest,
    )
    completed_manifest["labeling_sha256"] = _json_sha256(labeling)
    completed_manifest.update(_labeling_sample_counts(labeling))
    _save_ground_truth(evaluator, splits, ground_truth_path)
    completed_manifest["ground_truth_sha256"] = _validate_ground_truth_artifact(
        ground_truth_path,
        expected_image_ids=expected_generation_ids,
    )
    # This manifest is the completion marker and must land only after every
    # labeling-side artifact has been written and validated.
    if validate_manifests:
        save_json(completed_manifest, labeling_manifest_path)
    print(f"[COCO-CHAIR] Saved {len(labeling)} labels to {labeling_path}")


def _load_or_create_splits(
    *,
    output_dir: str,
    samples: list[dict],
    seed: int,
    shared_splits_path: str | os.PathLike[str] | None = None,
) -> dict:
    # The active protocol is always the leak-free outer 8:2 image split. This
    # protocol has no validation partition, and
    # ``ensure_strict_82_split`` backs up any incompatible historical split.
    shared_path = None
    if shared_splits_path:
        shared_path = Path(shared_splits_path).expanduser()
        if not shared_path.is_absolute():
            shared_path = Path(__file__).resolve().parents[1] / shared_path
    splits, backup_path = ensure_strict_82_split(
        Path(output_dir) / "image_splits.json",
        [int(sample["image_id"]) for sample in samples],
        seed=int(seed),
        shared_splits_path=shared_path,
    )
    if backup_path is not None:
        print(f"[COCO-CHAIR] Backed up previous split to {backup_path}")
    print(
        "[COCO-CHAIR] Strict image split (no validation): "
        f"{len(splits['train'])} train, {len(splits['test'])} test "
        "(minimum-train-loss checkpoint, fixed-0.5 + train-F1 thresholds)."
    )
    return splits


def _caption_rows_from_generation(
    *,
    args,
    model_cfg: dict,
    samples: list[dict],
    generations: dict,
    generations_path: str,
    generation_shard_dir: str,
    prompt: str,
):
    from models import build_model

    if _all_generations_available(samples, generations):
        print("[COCO-CHAIR] Reusing existing generations and loading tokenizer only.")
        tokenizer = _load_tokenizer(model_cfg["hf_name"], model_key=args.model)
        return _rows_from_generations(samples, generations), tokenizer

    pending = [
        sample for sample in samples
        if not _generation_entry_is_complete(
            generations.get(str(int(sample["image_id"])), {})
        )
    ]
    devices = args.generation_devices or [args.device]
    if len(devices) > 1 and pending:
        print(
            f"[COCO-CHAIR] Parallel generation on {', '.join(devices)} "
            f"for {len(pending)} images."
        )
        for image_id, caption, token_ids in _parallel_generate(
            model_key=args.model,
            model_cfg=model_cfg,
            samples=pending,
            devices=devices,
            shard_dir=generation_shard_dir,
            prompt=prompt,
        ):
            generations[str(image_id)] = {
                "generated_text": caption,
                "response_token_ids": [int(token_id) for token_id in token_ids],
            }
            save_json(generations, generations_path)
        save_json(generations, generations_path)
        tokenizer = _load_tokenizer(model_cfg["hf_name"], model_key=args.model)
    else:
        print(f"[COCO-CHAIR] Loading model '{args.model}' on {devices[0]}.")
        wrapper = build_model(args.model, model_cfg, device=devices[0])
        tokenizer = wrapper.tokenizer
        for sample in tqdm(pending, desc="Generating"):
            image_id = int(sample["image_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image, prompt=prompt)
            generations[str(image_id)] = {
                "generated_text": gen_out.generated_text,
                "response_token_ids": [int(token_id) for token_id in gen_out.response_token_ids],
            }
            save_json(generations, generations_path)

    return _rows_from_generations(samples, generations), tokenizer


def _parallel_generate(
    *,
    model_key: str,
    model_cfg: dict,
    samples: list[dict],
    devices: list[str],
    shard_dir: str,
    prompt: str,
) -> list[tuple[int, str, list[int]]]:
    os.makedirs(shard_dir, exist_ok=True)
    chunks = [[] for _ in devices]
    for index, sample in enumerate(samples):
        chunks[index % len(devices)].append(sample)
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(processes=len(devices)) as pool:
        jobs = [
            pool.apply_async(
                _generate_worker,
                (
                    worker_id,
                    model_key,
                    model_cfg,
                    device,
                    chunk,
                    os.path.join(shard_dir, f"worker_{worker_id}.jsonl"),
                    prompt,
                ),
            )
            for worker_id, (device, chunk) in enumerate(zip(devices, chunks))
            if chunk
        ]
        for job in jobs:
            results.extend(job.get())
    return results


def _generate_worker(
    worker_id: int,
    model_key: str,
    model_cfg: dict,
    device: str,
    samples: list[dict],
    shard_path: str,
    prompt: str,
) -> list[tuple[int, str, list[int]]]:
    from models import build_model

    print(f"[COCO-CHAIR worker {worker_id}] Loading model '{model_key}' on {device}.")
    wrapper = build_model(model_key, model_cfg, device=device)
    rows = []
    with open(shard_path, "a", encoding="utf-8", buffering=1) as shard_handle:
        for sample in tqdm(samples, desc=f"Generating worker {worker_id}"):
            image_id = int(sample["image_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image, prompt=prompt)
            token_ids = [int(token_id) for token_id in gen_out.response_token_ids]
            row = (image_id, gen_out.generated_text, token_ids)
            rows.append(row)
            shard_handle.write(
                json.dumps(
                    {
                        "image_id": image_id,
                        "generated_text": gen_out.generated_text,
                        "response_token_ids": token_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            shard_handle.flush()
    return rows


def _merge_generation_shards(generations: dict, shard_dir: str) -> None:
    path = Path(shard_dir)
    if not path.exists():
        return
    merged = 0
    for shard_path in sorted(path.glob("*.jsonl")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_id = int(row["image_id"])
                caption = str(row.get("generated_text") or row.get("caption") or "")
                if not caption:
                    continue
                generations[str(image_id)] = {
                    "generated_text": caption,
                    "response_token_ids": [
                        int(token_id) for token_id in row.get("response_token_ids", [])
                    ],
                }
                merged += 1
    if merged:
        print(f"[COCO-CHAIR] Recovered {merged} generated rows from generation shards.")



def _legacy_generation_adoption_enabled(config: Mapping[str, Any]) -> bool:
    raw = os.environ.get("ADOPT_LEGACY_ARTIFACTS")
    if raw in (None, ""):
        run_cfg = config.get("run") or {}
        raw = (
            run_cfg.get("adopt_legacy_artifacts", False)
            if isinstance(run_cfg, Mapping)
            else False
        )
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"ADOPT_LEGACY_ARTIFACTS must be a boolean, got {raw!r}"
    )


def _validate_or_initialize_generation_run_manifest(
    *,
    output_dir: Path,
    model_key: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    generations: Mapping[str, Any],
    generation_shard_dir: Path,
    expected_image_ids: set[int],
    fresh_start: bool,
    adopt_legacy: bool,
    validate_manifest: bool = True,
) -> dict[str, object]:
    """Bind partial generations/shards to one model, prompt, and cohort."""

    path = output_dir / GENERATION_MANIFEST_NAME
    if validate_manifest and path.is_symlink():
        raise ValueError(f"Refusing a symlinked generation manifest: {path}")
    expected = build_generation_run_manifest(
        model=model_key,
        model_cfg=model_cfg,
        prompt=prompt,
        expected_image_ids=expected_image_ids,
    )
    if not validate_manifest:
        current = (
            build_generation_manifest(
                model=model_key,
                model_cfg=model_cfg,
                prompt=prompt,
                generations=generations,
                expected_image_ids=expected_image_ids,
            )
            if _generation_mapping_complete(generations, expected_image_ids)
            else expected
        )
        return current
    if fresh_start:
        save_json(expected, str(path))
        return expected

    artifact_files_exist = bool(generations) or any(
        candidate.is_file() for candidate in generation_shard_dir.glob("*.jsonl")
    )
    if path.exists():
        manifest = load_json(str(path))
        validate_generation_identity(
            manifest,
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            expected_image_ids=expected_image_ids,
        )
        if manifest.get("status") == "complete" and not _generation_mapping_complete(
            generations, expected_image_ids
        ):
            raise RuntimeError(
                "A complete generation manifest is paired with incomplete "
                "generations.json. Refusing partial resume because retained "
                "content was removed or changed; use a fresh output directory."
            )
        return dict(manifest)

    if _generation_mapping_complete(generations, expected_image_ids):
        completed = build_generation_manifest(
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=generations,
            expected_image_ids=expected_image_ids,
        )
        save_json(completed, str(path))
        print(
            "[COCO-CHAIR] Found a complete generations.json in the output "
            f"directory; wrote {GENERATION_MANIFEST_NAME} automatically."
        )
        return completed

    if artifact_files_exist and not adopt_legacy:
        raise RuntimeError(
            f"Partial generation artifacts exist but {GENERATION_MANIFEST_NAME} "
            "is missing, so their model/prompt/cohort cannot be verified. Set "
            "ADOPT_LEGACY_ARTIFACTS=true only after checking them, or start "
            "from a fresh output directory."
        )
    if artifact_files_exist:
        expected["adopted_legacy_partial_output"] = str(output_dir.resolve())
        print(
            "[COCO-CHAIR] WARNING: adopting legacy partial generation "
            "artifacts after explicit user opt-in."
        )
    save_json(expected, str(path))
    return expected


def _generation_mapping_complete(
    generations: Mapping[str, Any],
    expected_image_ids: set[int],
) -> bool:
    return bool(expected_image_ids) and all(
        _generation_entry_is_complete(generations.get(str(int(image_id)), {}))
        for image_id in expected_image_ids
    )


def _validate_or_adopt_generation_manifest(
    *,
    output_dir: Path,
    model_key: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    generations: Mapping[str, Any],
    expected_image_ids: set[int],
    adopt_legacy: bool,
    validate_manifest: bool = True,
) -> dict[str, object]:
    path = output_dir / GENERATION_MANIFEST_NAME
    if validate_manifest and path.is_symlink():
        raise ValueError(f"Refusing a symlinked generation manifest: {path}")
    if not validate_manifest:
        manifest = build_generation_manifest(
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=generations,
            expected_image_ids=expected_image_ids,
        )
        return manifest
    if path.exists():
        manifest = load_json(str(path))
        validate_generation_identity(
            manifest,
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            expected_image_ids=expected_image_ids,
        )
        if manifest.get("status") == "complete":
            validate_generation_manifest(
                manifest,
                model=model_key,
                model_cfg=model_cfg,
                prompt=prompt,
                generations=generations,
                expected_image_ids=expected_image_ids,
            )
            return dict(manifest)
        completed = build_generation_manifest(
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=generations,
            expected_image_ids=expected_image_ids,
        )
        for key, value in manifest.items():
            if str(key).startswith("adopted_legacy"):
                completed[key] = value
        save_json(completed, str(path))
        return completed
    manifest = build_generation_manifest(
        model=model_key,
        model_cfg=model_cfg,
        prompt=prompt,
        generations=generations,
        expected_image_ids=expected_image_ids,
    )
    save_json(manifest, str(path))
    print(
        "[COCO-CHAIR] Found a complete generations.json; wrote "
        f"{GENERATION_MANIFEST_NAME} automatically."
    )
    return manifest


def _reuse_generation_artifacts(
    *,
    source_output: Path,
    target_output: Path,
    expected_image_ids: set[int],
    model_key: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    resume: bool = False,
    adopt_legacy: bool = False,
    validate_manifest: bool = True,
) -> None:
    """Safely copy immutable generation inputs into a new experiment output."""

    source = source_output.expanduser().resolve()
    target = target_output.expanduser().resolve()
    if source == target:
        raise ValueError(
            "--reuse-generations-from must point to a different output directory"
        )
    if not resume:
        _refuse_reuse_over_downstream_artifacts(target)

    source_generations = source / "generations.json"
    if not source_generations.is_file():
        raise FileNotFoundError(
            f"Reusable generations not found: {source_generations}"
        )
    source_payload = load_json(str(source_generations))
    _validate_generation_cohort(
        source_payload, expected_image_ids, source_generations
    )
    source_manifest_path = source / GENERATION_MANIFEST_NAME
    if source_manifest_path.is_file() and validate_manifest:
        source_manifest = load_json(str(source_manifest_path))
        validate_generation_manifest(
            source_manifest,
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_payload,
            expected_image_ids=expected_image_ids,
        )
    else:
        source_manifest = build_generation_manifest(
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_payload,
            expected_image_ids=expected_image_ids,
        )
        if not source_manifest_path.is_file():
            print(
                "[COCO-CHAIR] Reusable generations have no manifest; their "
                "content and image cohort were validated and a manifest will be "
                "created automatically in the new output."
            )

    target_generations = target / "generations.json"
    if target_generations.is_symlink():
        raise ValueError(
            f"Refusing a symlinked target generation file: {target_generations}"
        )
    if target_generations.exists():
        target_payload = load_json(str(target_generations))
        _validate_generation_cohort(
            target_payload, expected_image_ids, target_generations
        )
        if _json_sha256(target_payload) != _json_sha256(source_payload):
            raise ValueError(
                "Refusing --reuse-generations-from because target "
                f"generations differ from source: {target_generations}"
            )
        print(
            "[COCO-CHAIR] Target generations.json already matches source; "
            "nothing was overwritten."
        )
    else:
        _atomic_copy_file(source_generations, target_generations)
        print(
            f"[COCO-CHAIR] Copied reusable generations from {source_generations}"
        )

    target_manifest_path = target / GENERATION_MANIFEST_NAME
    if validate_manifest and target_manifest_path.is_symlink():
        raise ValueError(
            f"Refusing a symlinked generation manifest: {target_manifest_path}"
        )
    if target_manifest_path.exists() and validate_manifest:
        target_manifest = load_json(str(target_manifest_path))
        validate_generation_manifest(
            target_manifest,
            model=model_key,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_payload,
            expected_image_ids=expected_image_ids,
        )
    elif validate_manifest:
        save_json(dict(source_manifest), str(target_manifest_path))

    source_splits = source / "image_splits.json"
    target_splits = target / "image_splits.json"
    if target_splits.is_symlink():
        raise ValueError(f"Refusing a symlinked target split file: {target_splits}")
    if source_splits.is_file():
        source_split_payload = load_json(str(source_splits))
        _validate_split_cohort(
            source_split_payload, expected_image_ids, source_splits
        )
        if target_splits.exists():
            target_split_payload = load_json(str(target_splits))
            _validate_split_cohort(
                target_split_payload, expected_image_ids, target_splits
            )
            if _json_sha256(target_split_payload) != _json_sha256(
                source_split_payload
            ):
                raise ValueError(
                    "Refusing --reuse-generations-from because target image "
                    f"split differs from source: {target_splits}"
                )
        else:
            _atomic_copy_file(source_splits, target_splits)
            print(f"[COCO-CHAIR] Copied reusable image split from {source_splits}")


def _refuse_reuse_over_downstream_artifacts(target: Path) -> None:
    if not target.exists():
        return
    protected = [
        target / "labeling.json",
        target / LABELING_MANIFEST_NAME,
        target / "chair_summary.json",
        target / "coco_ground_truth.jsonl",
        target / "features.pkl",
        target / "features_manifest.json",
        target / "baseline",
        target / "results",
        target / "pipeline_manifest.json",
        target / "resolved_pipeline_config.yaml",
        target / "generation_shards",
        *target.glob("features.part*.pkl"),
    ]
    existing = sorted({str(path) for path in protected if path.exists()})
    if existing:
        raise RuntimeError(
            "Refusing --reuse-generations-from without --resume because the "
            "target already contains downstream artifacts: "
            + ", ".join(existing)
        )

def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_generation_cohort(
    generations: dict,
    expected_image_ids: set[int],
    path: Path,
) -> None:
    if not isinstance(generations, dict):
        raise ValueError(f"{path} must contain a JSON object")
    actual_ids = {int(value) for value in generations}
    if actual_ids != {int(value) for value in expected_image_ids}:
        missing = sorted(expected_image_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_image_ids)[:10]
        raise ValueError(
            f"{path} does not match the selected COCO cohort; "
            f"missing={missing}, extra={extra}"
        )
    for image_id in sorted(expected_image_ids):
        entry = generations.get(str(int(image_id)))
        if not isinstance(entry, dict):
            raise ValueError(f"{path} has no valid row for image {image_id}")
        caption = entry.get("generated_text")
        token_ids = entry.get("response_token_ids")
        if not isinstance(caption, str) or not caption:
            raise ValueError(
                f"{path} has an empty generated_text for image {image_id}"
            )
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(
                f"{path} has no response_token_ids for image {image_id}"
            )
        try:
            [int(value) for value in token_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} has invalid response_token_ids for image {image_id}"
            ) from exc


def _validate_split_cohort(
    splits: dict,
    expected_image_ids: set[int],
    path: Path,
) -> None:
    if not isinstance(splits, dict) or not {"train", "val", "test"}.issubset(splits):
        raise ValueError(f"{path} is not a train/val/test split")
    values = {
        name: [int(value) for value in splits[name]]
        for name in ("train", "val", "test")
    }
    flattened = values["train"] + values["val"] + values["test"]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{path} contains overlapping image IDs")
    if set(flattened) != {int(value) for value in expected_image_ids}:
        raise ValueError(f"{path} does not match the selected COCO cohort")


def _all_generations_available(samples: list[dict], generations: dict) -> bool:
    return bool(generations) and all(
        _generation_entry_is_complete(
            generations.get(str(int(sample["image_id"])), {})
        )
        for sample in samples
    )


def _generation_entry_is_complete(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and isinstance(entry.get("generated_text"), str)
        and bool(entry.get("generated_text"))
        and isinstance(entry.get("response_token_ids"), list)
        and bool(entry.get("response_token_ids"))
    )


def _all_labeling_available(
    samples: list[dict],
    generations: dict,
    labeling: dict,
    *,
    expected_manifest: dict | None = None,
    expected_sample_unit: str | None = None,
) -> bool:
    """Return whether resume can safely skip the entire labeling stage."""
    expected_ids = {str(int(sample["image_id"])) for sample in samples}
    if not expected_ids or set(labeling) != expected_ids:
        return False
    generations_ready = (
        _all_generations_available(samples, generations)
        if expected_manifest is not None
        else bool(generations) and all(
            generations.get(str(int(sample["image_id"])), {}).get("generated_text")
            for sample in samples
        )
    )
    if not generations_ready:
        return False

    required_keys = {
        "image_id",
        "generated_text",
        "hallucinated_words",
        "real_words",
        "object_token_spans",
        "chair_s",
        "chair_i",
    }
    if expected_manifest is not None:
        required_keys.update(
            {
                "schema_version",
                "all_object_token_spans",
                "official_svar_samples",
            }
        )
    for image_id in expected_ids:
        generation = generations.get(image_id)
        label = labeling.get(image_id)
        if not isinstance(generation, dict) or not isinstance(label, dict):
            return False
        if not required_keys.issubset(label):
            return False
        if int(label.get("image_id", -1)) != int(image_id):
            return False
        if str(label.get("generated_text", "")) != str(
            generation.get("generated_text", "")
        ):
            return False
        if not isinstance(label.get("object_token_spans"), list):
            return False
        if not isinstance(label.get("hallucinated_words"), list):
            return False
        if not isinstance(label.get("real_words"), list):
            return False
        if expected_sample_unit is not None:
            protocol = label.get("labeling_protocol") or {}
            if not isinstance(protocol, Mapping) or str(
                protocol.get("sample_unit", "")
            ) != str(expected_sample_unit):
                return False
        if expected_manifest is not None:
            if int(label.get("schema_version", -1)) != int(
                expected_manifest["label_schema_version"]
            ):
                return False
            if not isinstance(label.get("all_object_token_spans"), list):
                return False
            if not isinstance(label.get("official_svar_samples"), list):
                return False
    return True


def _reuse_complete_labeling(
    *,
    samples: list[dict],
    generations: dict,
    labeling: dict,
    generations_path: str,
    summary_path: str,
    manifest_path: str | None = None,
    expected_manifest: dict | None = None,
    ground_truth_path: str | None = None,
    adopt_legacy_ground_truth: bool = False,
    persist_generations: bool = False,
    expected_sample_unit: str | None = None,
    validate_manifest: bool = True,
) -> bool:
    if not _all_labeling_available(
        samples,
        generations,
        labeling,
        expected_manifest=expected_manifest,
        expected_sample_unit=expected_sample_unit,
    ):
        return False
    actual_manifest = None
    if validate_manifest and expected_manifest is not None:
        if not manifest_path or not os.path.exists(manifest_path):
            return False
        actual_manifest = load_json(manifest_path)
        if not _manifest_matches(
            actual_manifest,
            expected_manifest,
            labeling=labeling,
        ):
            return False
    if ground_truth_path is not None:
        try:
            ground_truth_sha256 = _validate_ground_truth_artifact(
                ground_truth_path,
                expected_image_ids={int(sample["image_id"]) for sample in samples},
            )
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        saved_ground_truth_sha256 = (
            actual_manifest.get("ground_truth_sha256")
            if isinstance(actual_manifest, Mapping)
            else None
        )
        if saved_ground_truth_sha256 is None and validate_manifest:
            if not adopt_legacy_ground_truth or actual_manifest is None or manifest_path is None:
                return False
            actual_manifest = dict(actual_manifest)
            actual_manifest["ground_truth_sha256"] = ground_truth_sha256
            save_json(actual_manifest, manifest_path)
            print(
                "[COCO-CHAIR] WARNING: registered a complete legacy "
                "coco_ground_truth.jsonl after explicit user opt-in."
            )
        elif saved_ground_truth_sha256 != ground_truth_sha256:
            return False
    if persist_generations:
        # A previous process may have exited after writing worker shards but
        # before merging them into generations.json.
        save_json(generations, generations_path)
    if actual_manifest is not None and manifest_path is not None:
        enriched_manifest = dict(actual_manifest)
        enriched_manifest.update(_labeling_sample_counts(labeling))
        if enriched_manifest != actual_manifest:
            save_json(enriched_manifest, manifest_path)
    expected_summary = chair_summary(labeling)
    summary_matches = False
    if os.path.exists(summary_path):
        try:
            summary_matches = load_json(summary_path) == expected_summary
        except (OSError, ValueError, json.JSONDecodeError):
            summary_matches = False
    if not summary_matches:
        save_json(expected_summary, summary_path)
    print(
        "[COCO-CHAIR] Resume — all "
        f"{len(samples)} generations and labels already exist; "
        "skipping model/tokenizer loading and CHAIR labeling."
    )
    return True



def _labeling_sample_counts(labeling: Mapping[str, Any]) -> dict[str, int]:
    full_mentions = 0
    controlled_samples = 0
    official_total = 0
    official_found = 0
    official_not_found = 0
    for row in labeling.values():
        if not isinstance(row, Mapping):
            continue
        full_mentions += len(row.get("all_object_token_spans") or [])
        controlled_samples += len(row.get("object_token_spans") or [])
        for sample in row.get("official_svar_samples") or []:
            if not isinstance(sample, Mapping):
                continue
            official_total += 1
            status = str(sample.get("status", "")).strip().lower()
            if status == "found":
                official_found += 1
            elif status == "not_found":
                official_not_found += 1
    return {
        "full_object_mention_count": int(full_mentions),
        "controlled_sample_count": int(controlled_samples),
        "official_svar_query_count": int(official_total),
        "official_svar_found_count": int(official_found),
        "official_svar_not_found_count": int(official_not_found),
    }


def _expected_labeling_manifest(
    *,
    samples: list[dict],
    generations: dict,
    model_cfg: dict,
    labeling_cfg: dict,
    generation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "label_schema_version": int(
            labeling_cfg.get("schema_version", LABELING_SCHEMA_VERSION)
        ),
        "sample_unit": str(
            labeling_cfg.get("sample_unit", "first_canonical_mention")
        ),
        "primary_locator": str(
            labeling_cfg.get("primary_locator", "exact_response_offsets")
        ),
        "save_all_mentions": bool(
            labeling_cfg.get("save_all_mentions", True)
        ),
        "save_svar_official_samples": bool(
            labeling_cfg.get("save_svar_official_samples", True)
        ),
        "alignment_failure_policy": str(
            labeling_cfg.get("alignment_failure_policy", "error")
        ),
        "tokenizer_source": str(model_cfg.get("hf_name", "")),
        "generation_sha256": _generation_sha256(samples, generations),
        "generation_provenance_sha256": (
            stable_generation_sha256(dict(generation_manifest))
            if generation_manifest is not None
            else None
        ),
    }


def _generation_sha256(samples: list[dict], generations: dict) -> str:
    payload = {}
    for sample in sorted(samples, key=lambda item: int(item["image_id"])):
        image_id = str(int(sample["image_id"]))
        entry = generations.get(image_id) or {}
        payload[image_id] = {
            "generated_text": str(entry.get("generated_text", "")),
            "response_token_ids": [
                int(value) for value in entry.get("response_token_ids", [])
            ],
        }
    return _json_sha256(payload)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_matches(
    actual: dict,
    expected: dict,
    *,
    labeling: dict,
) -> bool:
    if not isinstance(actual, dict):
        return False
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            return False
    return actual.get("labeling_sha256") == _json_sha256(labeling)


def _validate_existing_labeling_for_resume(
    *,
    labeling: dict,
    manifest_path: str,
    expected_manifest: dict,
    samples: list[dict],
    generations: dict,
    adopt_legacy: bool = False,
) -> None:
    hint = (
        "Use a new `*-svar-aligned` output directory, or run explicitly "
        "without --resume if overwriting this output is intended."
    )
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            "Refusing to overwrite existing legacy labeling.json during "
            f"--resume because {LABELING_MANIFEST_NAME} is missing. {hint}"
        )
    actual_manifest = load_json(manifest_path)
    provenance_key = "generation_provenance_sha256"
    if (
        actual_manifest.get(provenance_key)
        != expected_manifest.get(provenance_key)
        and adopt_legacy
    ):
        legacy_expected = {
            key: value
            for key, value in expected_manifest.items()
            if key != provenance_key
        }
        if _manifest_matches(
            actual_manifest,
            legacy_expected,
            labeling=labeling,
        ):
            actual_manifest[provenance_key] = expected_manifest.get(provenance_key)
            save_json(actual_manifest, manifest_path)
            print(
                "[COCO-CHAIR] WARNING: registered generation provenance in "
                "an otherwise matching legacy schema-v2 labeling manifest."
            )
    if not _manifest_matches(
        actual_manifest,
        expected_manifest,
        labeling=labeling,
    ):
        raise RuntimeError(
            "Refusing to overwrite existing labeling.json during --resume: "
            "its v2 manifest, generation hash, tokenizer, protocol, or label "
            f"hash does not match the requested run. {hint}"
        )
    if not _all_labeling_available(
        samples,
        generations,
        labeling,
        expected_manifest=expected_manifest,
    ):
        raise RuntimeError(
            "Refusing to overwrite incomplete or incompatible v2 labeling.json "
            f"during --resume. {hint}"
        )


def _rows_from_generations(samples: list[dict], generations: dict) -> list[dict]:
    rows = []
    for sample in samples:
        image_id = int(sample["image_id"])
        entry = generations.get(str(image_id), {})
        if not entry.get("generated_text"):
            continue
        rows.append(
            {
                "image_id": image_id,
                "image_path": sample["image_path"],
                "caption": entry["generated_text"],
            }
        )
    return rows


def _load_caption_rows(path: Path, sample_by_id: dict[int, dict]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            caption = str(row.get("caption") or row.get("generated_text") or "")
            if image_id not in sample_by_id or not caption:
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": sample_by_id[image_id]["image_path"],
                    "caption": caption,
                    "response_token_ids": [
                        int(value)
                        for value in (row.get("response_token_ids") or [])
                    ],
                }
            )
    if not rows:
        raise ValueError(f"No usable captions found in {path}")
    return rows


def _load_tokenizer(hf_name: str, *, model_key: str | None = None):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        if "llava_onevision" in str(model_key or "").lower():
            try:
                return AutoProcessor.from_pretrained(
                    hf_name,
                    trust_remote_code=True,
                    fix_mistral_regex=True,
                    use_fast=True,
                ).tokenizer
            except TypeError:
                return AutoProcessor.from_pretrained(
                    hf_name,
                    trust_remote_code=True,
                    use_fast=True,
                ).tokenizer
        return AutoProcessor.from_pretrained(
            hf_name,
            trust_remote_code=True,
        ).tokenizer
    except Exception:
        return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)


def _generation_token_ids(
    generations: dict,
    image_id: int,
    caption: str,
    tokenizer,
    *,
    provided_token_ids: list[int] | None = None,
) -> list[int]:
    del caption, tokenizer
    if provided_token_ids:
        return [int(token_id) for token_id in provided_token_ids]
    entry = generations.get(str(int(image_id)), {})
    token_ids = entry.get("response_token_ids")
    if token_ids:
        return [int(token_id) for token_id in token_ids]
    raise ValueError(
        "Schema-v2 labeling requires the actual response_token_ids emitted by "
        f"generation for image_id={image_id}. Re-encoding generated_text is "
        "not a safe substitute. Include response_token_ids in --caption-file "
        "rows or reuse a generations.json that contains them."
    )


def _chair_token_spans(
    *,
    evaluator,
    tokenizer,
    image_id: int,
    caption: str,
    token_ids: list[int],
    chair_info: dict | None = None,
    alignment_failure_policy: str = "error",
) -> list[dict]:
    if chair_info is None:
        chair_info = evaluator.compute_chair_token(image_id, caption)
    mentions = chair_info["object_mentions"]
    failure_policy = str(alignment_failure_policy).strip().lower()
    if failure_policy not in {"error", "skip"}:
        raise ValueError(
            "alignment_failure_policy must be 'error' or 'skip', got "
            f"{alignment_failure_policy!r}"
        )
    try:
        response_offsets = build_response_token_offsets(
            tokenizer,
            token_ids,
            caption,
        )
        global_alignment_error = None
    except TokenAlignmentError as exc:
        if failure_policy == "error":
            raise TokenAlignmentError(
                f"image_id={image_id}: {exc}"
            ) from exc
        response_offsets = [None] * len(token_ids)
        global_alignment_error = str(exc)

    canonical_counts: dict[str, int] = {}
    for mention in mentions:
        canonical = str(mention["canonical_object"])
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
    canonical_seen: dict[str, int] = {}

    spans = []
    for mention in mentions:
        char_start = int(mention["char_start"])
        char_end = int(mention["char_end"])
        canonical = str(mention["canonical_object"])
        canonical_seen[canonical] = canonical_seen.get(canonical, 0) + 1
        surface = caption[char_start:char_end]
        normalized_word = str(
            mention.get("normalized_word")
            or mention.get("word")
            or surface
        )

        exact_error = global_alignment_error
        exact_indices: list[int] = []
        if exact_error is None:
            try:
                exact_indices = token_indices_for_char_span(
                    response_offsets,
                    char_start,
                    char_end,
                )
                validate_token_surface(
                    tokenizer=tokenizer,
                    response_token_ids=token_ids,
                    token_indices=exact_indices,
                    offsets=response_offsets,
                    caption=caption,
                    char_start=char_start,
                    char_end=char_end,
                )
            except TokenAlignmentError as exc:
                exact_error = str(exc)
                exact_indices = []
                if failure_policy == "error":
                    raise TokenAlignmentError(
                        f"image_id={image_id}, surface={surface!r}, "
                        f"chars=[{char_start}, {char_end}): {exc}"
                    ) from exc

        exact_location = {
            "status": "found" if exact_indices else "not_found",
            "query": surface,
            "query_token_id": (
                int(token_ids[exact_indices[0]]) if exact_indices else None
            ),
            "token_indices": [int(value) for value in exact_indices],
            "char_start": char_start,
            "char_end": char_end,
            "used_plural_fallback": False,
        }
        if exact_error:
            exact_location["error"] = exact_error

        surface_location = locate_first_token_id(
            tokenizer=tokenizer,
            response_token_ids=token_ids,
            query=normalized_word,
            pluralize=_official_plural,
        )
        canonical_location = locate_first_token_id(
            tokenizer=tokenizer,
            response_token_ids=token_ids,
            query=canonical,
            pluralize=_official_plural,
        )
        spans.append(
            {
                "word": canonical,
                "canonical_object": canonical,
                "normalized_word": normalized_word,
                "surface": surface,
                "surface_word": surface,
                "word_idx": int(mention["word_idx"]),
                "char_start": char_start,
                "char_end": char_end,
                "token_indices": [int(value) for value in exact_indices],
                "label": int(mention["label"]),
                "occurrence_count": int(canonical_counts[canonical]),
                "occurrence_index": int(canonical_seen[canonical]),
                "token_locations": {
                    "exact_response_offsets": exact_location,
                    "svar_surface_first_token_id": surface_location,
                    "svar_canonical_first_token_id": canonical_location,
                },
            }
        )
    spans.sort(
        key=lambda item: (
            int(item["char_start"]),
            int(item["word_idx"]),
        )
    )
    return spans


_INFLECT_ENGINE = None


def _official_plural(text: str) -> str:
    """Use the exact pluralizer used by the official SVAR implementation."""

    global _INFLECT_ENGINE
    if _INFLECT_ENGINE is None:
        try:
            import inflect
        except ImportError as exc:
            raise RuntimeError(
                "Official SVAR plural fallback requires the `inflect` package. "
                "Install project requirements (including inflect>=7,<8)."
            ) from exc
        _INFLECT_ENGINE = inflect.engine()
    plural = _INFLECT_ENGINE.plural(str(text))
    return str(plural or text)


def _build_official_svar_samples(
    *,
    tokenizer,
    token_ids: list[int],
    chair_info: dict,
) -> list[dict]:
    """Build the official SVAR object queries, including its known quirks."""

    generated = {
        str(value) for value in chair_info.get("mscoco_generated_words", [])
    }
    ground_truth = {
        str(value) for value in chair_info.get("mscoco_gt_words", [])
    }
    samples = []

    for canonical in sorted(generated & ground_truth):
        location = locate_first_token_id(
            tokenizer=tokenizer,
            response_token_ids=token_ids,
            query=canonical,
            pluralize=_official_plural,
        )
        samples.append(
            _official_svar_sample_entry(
                query=canonical,
                canonical=canonical,
                label=LABEL_REAL,
                search_source="canonical",
                location=location,
            )
        )

    hallucinated_queries: dict[str, dict[str, set[str]]] = {}
    for pair in chair_info.get("mscoco_hallucinated_words", []):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        normalized, canonical = str(pair[0]), str(pair[1])
        for query, source in (
            (normalized, "normalized"),
            (canonical, "canonical"),
        ):
            metadata = hallucinated_queries.setdefault(
                query,
                {"sources": set(), "canonicals": set()},
            )
            metadata["sources"].add(source)
            metadata["canonicals"].add(canonical)

    for query in sorted(hallucinated_queries):
        metadata = hallucinated_queries[query]
        canonical = sorted(metadata["canonicals"])[0]
        search_source = "+".join(sorted(metadata["sources"]))
        location = locate_first_token_id(
            tokenizer=tokenizer,
            response_token_ids=token_ids,
            query=query,
            pluralize=_official_plural,
        )
        samples.append(
            _official_svar_sample_entry(
                query=query,
                canonical=canonical,
                label=LABEL_HALLUCINATED,
                search_source=search_source,
                location=location,
            )
        )
    return samples


def _official_svar_sample_entry(
    *,
    query: str,
    canonical: str,
    label: int,
    search_source: str,
    location: dict,
) -> dict:
    indices = [int(value) for value in location.get("token_indices", [])]
    return {
        "word": str(query),
        "query": str(query),
        "search_term": str(query),
        "search_source": str(search_source),
        "surface": str(query),
        "canonical": str(canonical),
        "canonical_object": str(canonical),
        "label": int(label),
        "token_indices": indices,
        "response_token_idx": indices[0] if indices else None,
        "status": str(location.get("status", "not_found")),
        "token_location": dict(location),
    }


def _compact_label_entry(
    *,
    image_id: int,
    caption: str,
    spans: list[dict],
    chair_info: dict,
    official_svar_samples: list[dict] | None = None,
    sample_unit: str = "first_canonical_mention",
) -> dict:
    all_object_spans = [
        {
            "word": str(span["word"]),
            "canonical_object": str(
                span.get("canonical_object", span["word"])
            ),
            "normalized_word": str(
                span.get("normalized_word", span.get("surface", span["word"]))
            ),
            "surface": str(span.get("surface", span["word"])),
            "surface_word": str(span.get("surface", span["word"])),
            "token_indices": [int(idx) for idx in span.get("token_indices", [])],
            "word_idx": int(span["word_idx"]),
            "char_start": int(span["char_start"]),
            "char_end": int(span["char_end"]),
            "label": int(span["label"]),
            "occurrence_count": int(span.get("occurrence_count", 1)),
            "occurrence_index": int(span.get("occurrence_index", 1)),
            "token_locations": dict(span.get("token_locations") or {}),
        }
        for span in spans
        if int(span.get("label", -100)) in (LABEL_HALLUCINATED, LABEL_REAL)
    ]
    if sample_unit == "first_canonical_mention":
        object_spans = _first_canonical_mentions(all_object_spans)
    elif sample_unit == "all_mentions":
        object_spans = list(all_object_spans)
    else:
        raise ValueError(f"Unsupported labeling sample_unit: {sample_unit!r}")
    return {
        "schema_version": LABELING_SCHEMA_VERSION,
        "labeling_protocol": {
            "sample_unit": sample_unit,
            "primary_locator": "exact_response_offsets",
        },
        "image_id": int(image_id),
        "generated_text": caption,
        "hallucinated_words": [
            span["surface"] for span in all_object_spans
            if int(span["label"]) == LABEL_HALLUCINATED
        ],
        "real_words": [
            span["surface"] for span in all_object_spans
            if int(span["label"]) == LABEL_REAL
        ],
        "all_object_token_spans": all_object_spans,
        "object_token_spans": object_spans,
        "official_svar_samples": [
            dict(value) for value in (official_svar_samples or [])
        ],
        "chair_s": int(chair_info["metrics"]["CHAIRs"]),
        "chair_i": float(chair_info["metrics"]["CHAIRi"]),
    }


def _first_canonical_mentions(spans: list[dict]) -> list[dict]:
    seen = set()
    selected = []
    for span in sorted(
        spans,
        key=lambda item: (int(item["char_start"]), int(item["word_idx"])),
    ):
        canonical = str(span.get("canonical_object") or span.get("word") or "")
        if not canonical or canonical in seen:
            continue
        # The first actual mention owns the canonical sample even when exact
        # alignment failed. Mark it seen before filtering so a later mention
        # is never silently substituted for the failed first mention.
        seen.add(canonical)
        exact = (span.get("token_locations") or {}).get(
            "exact_response_offsets"
        ) or {}
        if (
            str(exact.get("status", "")).strip().lower() != "found"
            or not span.get("token_indices")
        ):
            continue
        selected.append(dict(span))
    return selected


def _save_ground_truth(evaluator, splits: dict, path: str) -> None:
    """Atomically persist one ground-truth row for every split image."""

    image_ids = sorted(
        {int(image_id) for values in splits.values() for image_id in values}
    )
    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"Refusing a symlinked ground-truth target: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for entry in iter_ground_truth_entries(evaluator, image_ids):
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_ground_truth_artifact(
    path: str | os.PathLike[str],
    *,
    expected_image_ids: set[int],
) -> str:
    target = Path(path)
    if target.is_symlink():
        raise ValueError(f"Refusing a symlinked ground-truth artifact: {target}")
    if not target.is_file():
        raise FileNotFoundError(target)
    actual_ids: list[int] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or "image_id" not in row:
                raise ValueError(
                    f"Invalid ground-truth row at line {line_number}: {target}"
                )
            actual_ids.append(int(row["image_id"]))
    expected = {int(value) for value in expected_image_ids}
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected:
        raise ValueError(
            f"{target} does not exactly cover the selected image cohort"
        )
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
