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
    normalize_svar_protocols,
    prepare_official_svar_spans,
)
from utils.config_utils import get_dataset_cfg, get_model_cfg, load_config
from utils.io_utils import append_pkl, load_json, load_pkl, save_pkl
from scripts.extract_features import _resolve_prompt


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
    prompt = _resolve_prompt(
        cli_prompt=args.prompt,
        config=config,
        model_cfg=model_cfg,
    )
    methods = normalize_baseline_methods(baseline_cfg.get("methods", "all"))
    if not methods:
        raise ValueError("feature_extraction.baseline.methods cannot be empty")
    svar_protocols = (
        normalize_svar_protocols(
            dict(baseline_cfg.get("svar") or {}).get("protocols")
        )
        if "svar" in methods
        else ()
    )
    controlled_methods = tuple(
        method
        for method in methods
        if method != "svar" or "controlled" in svar_protocols
    )
    official_svar_enabled = "official" in svar_protocols

    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / str(baseline_cfg.get("output_subdir", "baseline"))
    baseline_dir.mkdir(parents=True, exist_ok=True)
    official_dir = baseline_dir / "svar_official"
    if official_svar_enabled:
        official_dir.mkdir(parents=True, exist_ok=True)
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
    labeled_sample_count = len(samples)
    samples = [
        sample
        for sample in samples
        if _sample_needs_any_protocol(
            label_info=labeling[int(sample["image_id"])],
            generation=generations.get(int(sample["image_id"])),
            controlled_enabled=bool(controlled_methods),
            official_enabled=official_svar_enabled,
        )
    ]
    skipped = labeled_sample_count - len(samples)
    print(
        f"[BaselineExtract] Using {len(samples)} extractable labeled images"
        + (f" ({skipped} have no valid object-token span)." if skipped else ".")
    )
    devices = args.feature_devices or [args.device]
    output_path = baseline_dir / "features.pkl"
    official_output_path = official_dir / "features.pkl"
    if not samples:
        if controlled_methods and not output_path.exists():
            save_pkl([], str(output_path))
        if official_svar_enabled and not official_output_path.exists():
            save_pkl([], str(official_output_path))
        print("[BaselineExtract] No extractable samples; wrote empty artifacts.")
        return
    if controlled_methods and output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Baseline feature file already exists: {output_path}. "
            "Use --resume to reuse it."
        )
    if official_svar_enabled and official_output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Official SVAR feature file already exists: {official_output_path}. "
            "Use --resume to reuse it."
        )
    if args.resume:
        parts_dir = baseline_dir / "feature_parts"
        part_paths = (
            sorted(
                {
                    *parts_dir.glob("worker_*.pkl"),
                    *baseline_dir.glob("features.part*.pkl"),
                }
            )
            if controlled_methods
            else []
        )
        official_parts_dir = official_dir / "feature_parts"
        official_part_paths = (
            sorted(
                {
                    *official_parts_dir.glob("worker_*.pkl"),
                    *official_dir.glob("features.part*.pkl"),
                }
            )
            if official_svar_enabled
            else []
        )
        if controlled_methods and part_paths:
            _merge_parts(output_path, part_paths, resume=True)
        if official_svar_enabled and official_part_paths:
            _merge_parts(
                official_output_path,
                official_part_paths,
                resume=True,
            )
        controlled_done = (
            _done_image_ids([output_path]) if controlled_methods else set()
        )
        official_done = (
            _done_image_ids([official_output_path])
            if official_svar_enabled
            else set()
        )
        pending = _pending_samples_for_protocols(
            samples=samples,
            labeling=labeling,
            generations=generations,
            controlled_enabled=bool(controlled_methods),
            official_enabled=official_svar_enabled,
            controlled_done=controlled_done,
            official_done=official_done,
        )
        if not pending:
            if controlled_methods and not output_path.exists():
                save_pkl([], str(output_path))
            if official_svar_enabled and not official_output_path.exists():
                save_pkl([], str(official_output_path))
            print(
                "[BaselineExtract] Resume — requested baseline protocols cover "
                f"all {len(samples)} extractable images; skipping model loading."
            )
            return
        print(
            f"[BaselineExtract] Resume — {len(samples) - len(pending)} images "
            f"complete, {len(pending)} pending."
        )
        samples = pending
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
            part_path=str(output_path) if controlled_methods else None,
            official_part_path=(
                str(official_output_path) if official_svar_enabled else None
            ),
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
            official_output_path=official_output_path,
            methods=methods,
            prompt=prompt,
            resume=args.resume,
        )
    if controlled_methods:
        print(f"[BaselineExtract] saved {output_path}")
    if official_svar_enabled:
        print(f"[BaselineExtract] saved {official_output_path}")


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
    official_output_path,
    methods,
    prompt,
    resume,
):
    svar_protocols = (
        normalize_svar_protocols(
            dict(baseline_cfg.get("svar") or {}).get("protocols")
        )
        if "svar" in methods
        else ()
    )
    controlled_enabled = any(
        method != "svar" or "controlled" in svar_protocols
        for method in methods
    )
    official_enabled = "official" in svar_protocols
    parts_dir = baseline_dir / "feature_parts"
    if controlled_enabled:
        parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = (
        [parts_dir / f"worker_{index}.pkl" for index in range(len(devices))]
        if controlled_enabled
        else [None for _ in devices]
    )
    official_parts_dir = baseline_dir / "svar_official" / "feature_parts"
    if official_enabled:
        official_parts_dir.mkdir(parents=True, exist_ok=True)
    official_part_paths = (
        [
            official_parts_dir / f"worker_{index}.pkl"
            for index in range(len(devices))
        ]
        if official_enabled
        else [None for _ in devices]
    )
    if not resume and controlled_enabled:
        existing_parts = [str(path) for path in part_paths if path.exists()]
        if existing_parts:
            raise FileExistsError(
                "Baseline worker parts already exist; use --resume: "
                + ", ".join(existing_parts)
            )
    if not resume and official_enabled:
        existing_official_parts = [
            str(path) for path in official_part_paths if path.exists()
        ]
        if existing_official_parts:
            raise FileExistsError(
                "Official SVAR worker parts already exist; use --resume: "
                + ", ".join(existing_official_parts)
            )
    pending = samples
    if not pending:
        return
    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending):
        chunks[index % len(devices)].append(sample)
    jobs = []
    with get_context("spawn").Pool(len(devices)) as pool:
        for worker_id, (device, chunk, part_path, official_part_path) in enumerate(
            zip(devices, chunks, part_paths, official_part_paths)
        ):
            if not chunk:
                if part_path is not None and not part_path.exists():
                    save_pkl([], str(part_path))
                if (
                    official_part_path is not None
                    and not official_part_path.exists()
                ):
                    save_pkl([], str(official_part_path))
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
                        "part_path": (
                            str(part_path) if part_path is not None else None
                        ),
                        "official_part_path": (
                            str(official_part_path)
                            if official_part_path is not None
                            else None
                        ),
                        "methods": methods,
                        "prompt": prompt,
                        "resume": resume,
                        "parallel": True,
                    },
                )
            )
        for job in jobs:
            job.get()
    if controlled_enabled:
        _merge_parts(output_path, part_paths, resume=resume)
    if official_enabled:
        _merge_parts(
            official_output_path,
            official_part_paths,
            resume=resume,
        )


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
    part_path: str | None,
    official_part_path: str | None,
    methods: Sequence[str],
    prompt: str,
    resume: bool,
    parallel: bool,
) -> None:
    from models import build_model

    baseline_root = Path(baseline_dir)
    controlled_existing = (
        load_pkl(part_path)
        if resume and part_path is not None and os.path.exists(part_path)
        else []
    )
    controlled_done = {
        int(record["image_id"]) for record in controlled_existing
    }
    official_existing = (
        load_pkl(official_part_path)
        if (
            resume
            and official_part_path is not None
            and os.path.exists(official_part_path)
        )
        else []
    )
    official_done = {int(record["image_id"]) for record in official_existing}
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
    try:
        for sample in samples:
            image_id = int(sample["image_id"])
            protocol_needs = sample.get("_baseline_protocols_needed") or {}
            needs_controlled = bool(protocol_needs.get("controlled", True))
            needs_official = bool(protocol_needs.get("official", True))
            label_info = labeling.get(image_id)
            if not label_info:
                continue
            response_ids = _response_ids(
                wrapper, image_id, label_info, generations
            )
            controlled_spans = [
                span
                for span in label_info.get("object_token_spans", [])
                if (
                    runtime.methods
                    and needs_controlled
                    and image_id not in controlled_done
                    and span.get("token_indices")
                    and all(
                        0 <= int(index) < len(response_ids)
                        for index in span["token_indices"]
                    )
                )
            ]
            official_spans = (
                runtime.prepare_official_svar_spans(label_info, response_ids)
                if needs_official and image_id not in official_done
                else []
            )
            if not controlled_spans and not official_spans:
                continue
            requirements = runtime.requirements_for(
                controlled=bool(controlled_spans),
                official=bool(official_spans),
            )
            # The two protocols may target the same response position.  Run
            # one prefix forward per unique index and share its attention.
            indices = list(
                dict.fromkeys(
                    [
                        int(span["token_indices"][0])
                        for span in (*controlled_spans, *official_spans)
                    ]
                )
            )
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
            if len(outputs) != len(indices):
                raise RuntimeError(
                    f"Image {image_id}: wrapper returned {len(outputs)} outputs "
                    f"for {len(indices)} unique baseline token positions"
                )
            for requested_index, output in zip(indices, outputs):
                returned_index = getattr(output, "response_token_idx", None)
                if returned_index is not None and int(returned_index) != int(
                    requested_index
                ):
                    raise AssertionError(
                        f"Image {image_id}: wrapper returned "
                        f"response_token_idx={returned_index} for requested "
                        f"causal position {requested_index}"
                    )
            output_by_index = dict(zip(indices, outputs))

            if controlled_spans:
                image_records = runtime.build_image_records(
                    image=image,
                    image_id=image_id,
                    response_token_ids=response_ids,
                    spans=controlled_spans,
                    model_outputs=[
                        output_by_index[int(span["token_indices"][0])]
                        for span in controlled_spans
                    ],
                )
                if image_records and part_path is not None:
                    append_pkl(image_records, part_path)
                controlled_done.add(image_id)
            if official_spans:
                official_records = runtime.build_official_svar_records(
                    image_id=image_id,
                    response_token_ids=response_ids,
                    spans=official_spans,
                    model_outputs=[
                        output_by_index[int(span["token_indices"][0])]
                        for span in official_spans
                    ],
                )
                if official_records and official_part_path is not None:
                    append_pkl(official_records, official_part_path)
                official_done.add(image_id)
    finally:
        runtime.close()


def _response_ids(wrapper, image_id, label_info, generations) -> list[int]:
    del wrapper
    generation = generations.get(image_id)
    if not isinstance(generation, dict):
        raise RuntimeError(
            f"Image {image_id}: generations.json has no matching row. "
            "Schema-v2 baseline extraction requires actual response IDs."
        )
    if str(generation.get("generated_text", "")) != str(
        label_info.get("generated_text", "")
    ):
        raise RuntimeError(
            f"Image {image_id}: generated_text differs between "
            "generations.json and labeling.json."
        )
    token_ids = generation.get("response_token_ids") or []
    if token_ids:
        return [int(value) for value in token_ids]
    raise RuntimeError(
        f"Image {image_id}: generations.json has no response_token_ids. "
        "Re-encoding generated_text is intentionally forbidden."
    )


def _load_generations(output_dir: Path) -> dict[int, dict]:
    path = output_dir / "generations.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return {int(key): value for key, value in load_json(str(path)).items()}


def _done_image_ids(paths) -> set[int]:
    done = set()
    for path in paths:
        if os.path.exists(path):
            done.update(int(row["image_id"]) for row in load_pkl(str(path)))
    return done


def _has_extractable_object_spans(label_info: dict, generation=None) -> bool:
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
        if not all(0 <= int(index) < response_length for index in token_indices):
            raise ValueError(
                f"Object token indices {token_indices} are outside response "
                f"length {response_length}"
            )
        return True
    return False


def _has_extractable_official_svar_samples(
    label_info: dict,
    generation=None,
) -> bool:
    if not isinstance(label_info, dict) or not label_info.get("generated_text"):
        return False
    response_ids = _validated_generation_response_ids(label_info, generation)
    return bool(
        prepare_official_svar_spans(
            label_info.get("official_svar_samples") or [],
            response_ids,
        )
    )


def _validated_generation_response_ids(label_info, generation) -> list[int]:
    image_id = label_info.get("image_id", "?")
    if not isinstance(generation, dict):
        raise RuntimeError(
            f"Image {image_id}: generations.json has no matching row."
        )
    if str(generation.get("generated_text", "")) != str(
        label_info.get("generated_text", "")
    ):
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
            f"Image {image_id}: invalid response_token_ids."
        ) from exc


def _sample_needs_any_protocol(
    *,
    label_info: dict,
    generation,
    controlled_enabled: bool,
    official_enabled: bool,
) -> bool:
    return bool(
        (
            controlled_enabled
            and _has_extractable_object_spans(label_info, generation)
        )
        or (
            official_enabled
            and _has_extractable_official_svar_samples(label_info, generation)
        )
    )


def _pending_samples_for_protocols(
    *,
    samples,
    labeling,
    generations,
    controlled_enabled: bool,
    official_enabled: bool,
    controlled_done: set[int],
    official_done: set[int],
):
    pending = []
    for sample in samples:
        image_id = int(sample["image_id"])
        label_info = labeling[image_id]
        generation = generations.get(image_id)
        needs_controlled = bool(
            controlled_enabled
            and _has_extractable_object_spans(label_info, generation)
        )
        needs_official = bool(
            official_enabled
            and _has_extractable_official_svar_samples(label_info, generation)
        )
        if (
            (needs_controlled and image_id not in controlled_done)
            or (needs_official and image_id not in official_done)
        ):
            pending_sample = dict(sample)
            pending_sample["_baseline_protocols_needed"] = {
                "controlled": bool(
                    needs_controlled and image_id not in controlled_done
                ),
                "official": bool(
                    needs_official and image_id not in official_done
                ),
            }
            pending.append(pending_sample)
    return pending


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
