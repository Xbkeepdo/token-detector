#!/usr/bin/env python3
"""Run generation, labeling, and one-forward joint extraction for QA benchmarks."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
import pickle
import sys
import tempfile
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import (
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    sha256_file,
)
from features.baseline import (
    BaselineRuntime,
    baseline_config,
    normalize_baseline_methods,
    validate_baseline_record,
)
from features.qa_baseline import (
    DEFAULT_QA_BASELINES,
    QA_LABEL_PROTOCOLS,
    QABaselineAdapter,
    QABaselineFeatureStore,
    build_qa_probe_split_manifest,
    normalize_qa_label_protocol,
    qa_label_for_protocol,
    validate_halloc_cache_uniqueness,
    validate_qa_cache_ids,
)
from features.qa_extractor import (
    QA_FEATURE_SCHEMA_VERSION,
    extract_questions,
    generate_questions,
    label_generations,
    qa_generation_fingerprint,
    qa_generation_record_is_complete,
    qa_label_fingerprint,
    qa_prompt,
    qa_question_input_fingerprint,
)
from models import build_model
from scripts.extract_qa_baselines import (
    MANIFEST_NAME as QA_BASELINE_MANIFEST_NAME,
    SPLIT_MANIFEST_NAME as QA_BASELINE_SPLIT_MANIFEST_NAME,
    _controlled_qa_baseline_config,
    _expected_manifest as _expected_baseline_manifest,
    _validate_halloc_cache_files,
    _validate_or_initialize_manifest,
    _validate_source_rows,
)
from utils.config_utils import (
    get_ads_cfg,
    get_cgc_cfg,
    get_model_cfg,
    load_config,
    qa_extraction_family_flags,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=("pope", "clevr_exist_5k", "amber_discriminative"),
        required=True,
    )
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-root")
    parser.add_argument("--stage", choices=("generate", "label", "extract", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--generation-devices",
        nargs="+",
        default=None,
        help="Generate disjoint question shards in parallel, e.g. cuda:0 cuda:1.",
    )
    parser.add_argument(
        "--feature-devices",
        nargs="+",
        default=None,
        help="Extract disjoint question shards in parallel, e.g. cuda:0 cuda:1.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--feature-shard-size", type=int, default=25)
    parser.add_argument(
        "--baseline-label-protocols",
        nargs="+",
        choices=QA_LABEL_PROTOCOLS,
        default=None,
        help="Baseline label cohorts extracted in the same LVLM forward.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def _normalize_devices(values, fallback: str) -> tuple[str, ...]:
    devices = tuple(dict.fromkeys(str(value).strip() for value in (values or (fallback,))))
    if not devices or any(not value for value in devices):
        raise ValueError("QA device lists cannot be empty")
    return devices


def _question_partitions(
    questions: list[dict], num_workers: int
) -> list[list[dict]]:
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    partitions = [[] for _ in range(int(num_workers))]
    for index, question in enumerate(questions):
        partitions[index % int(num_workers)].append(question)
    return partitions


def _generation_worker(
    *,
    model_key: str,
    model_cfg: dict,
    questions: list[dict],
    worker_dir: str,
    device: str,
    checkpoint_every: int,
) -> None:
    wrapper = build_model(model_key, model_cfg, device=device)
    generate_questions(
        wrapper,
        model_key,
        questions,
        worker_dir,
        checkpoint_every,
    )


def _feature_worker(
    *,
    model_key: str,
    model_cfg: dict,
    questions: list[dict],
    worker_dir: str,
    device: str,
    dgst_cfg: dict,
    ads_cfg: dict,
    cgc_cfg: dict,
    shard_size: int,
    position_protocols: tuple[str, ...],
    extraction_fingerprint: str,
    method_enabled: bool,
    ads_cgc_enabled: bool,
    baseline_specs: list[dict],
    worker_id: int,
) -> None:
    wrapper = build_model(model_key, model_cfg, device=device)
    with ExitStack() as stack:
        consumers = _open_baseline_consumers(
            stack=stack,
            wrapper=wrapper,
            baseline_specs=baseline_specs,
            device=device,
            shard_size=shard_size,
            worker_id=worker_id,
            parallel=True,
        )
        extract_questions(
            wrapper,
            model_key,
            questions,
            worker_dir,
            dgst_cfg,
            ads_cfg,
            cgc_cfg,
            shard_size,
            position_protocols=position_protocols,
            extraction_fingerprint=extraction_fingerprint,
            method_enabled=method_enabled,
            ads_cgc_enabled=ads_cgc_enabled,
            baseline_consumers=consumers,
        )


def _open_baseline_consumers(
    *,
    stack: ExitStack,
    wrapper,
    baseline_specs: list[dict],
    device: str,
    shard_size: int,
    worker_id: int,
    parallel: bool,
) -> dict[str, dict]:
    """Open protocol-specific baseline writers around one shared wrapper."""

    consumers: dict[str, dict] = {}
    for spec in baseline_specs:
        protocol = str(spec["label_protocol"])
        store = QABaselineFeatureStore(
            spec["baseline_dir"],
            shard_size=shard_size,
            resume=True,
            part_prefix=(
                f"part-worker{int(worker_id):03d}" if parallel else "part"
            ),
        )
        runtime = stack.enter_context(
            BaselineRuntime(
                wrapper=wrapper,
                methods=spec["methods"],
                baseline_dir=spec["baseline_dir"],
                config=spec["baseline_cfg"],
                device=device,
                worker_id=worker_id,
                parallel=parallel,
                resume=True,
            )
        )
        consumers[protocol] = {
            "adapter": QABaselineAdapter(runtime, label_protocol=protocol),
            "store": store,
            "cache_ids": spec["cache_ids"],
        }
    return consumers


def _run_spawn_workers(target, jobs: list[dict], description: str) -> None:
    context = get_context("spawn")
    processes = []
    for worker_id, job in enumerate(jobs):
        if not job.get("questions"):
            continue
        process = context.Process(
            target=target,
            kwargs=job,
            name=f"qa-{description}-worker-{worker_id}",
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append((process.name, process.exitcode))
    if failures:
        raise RuntimeError(f"QA {description} workers failed: {failures}")


def _parallel_worker_root(
    output_dir: str, stage: str, num_workers: int
) -> Path:
    root = (
        Path(output_dir)
        / ".qa_parallel"
        / str(stage)
        / f"workers-{int(num_workers)}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_parallel_generation(
    *,
    model_key: str,
    model_cfg: dict,
    questions: list[dict],
    output_dir: str,
    devices: tuple[str, ...],
    checkpoint_every: int,
) -> None:
    partitions = _question_partitions(questions, len(devices))
    root = _parallel_worker_root(output_dir, "generation", len(devices))
    expected = {str(question["key"]): question for question in questions}
    main_rows = {
        str(row.get("key")): row
        for row in load_jsonl(Path(output_dir) / "generations.jsonl")
    }
    jobs = []
    for worker_id, (device, partition) in enumerate(zip(devices, partitions)):
        worker_dir = root / f"worker-{worker_id:03d}"
        assigned = {str(question["key"]): question for question in partition}
        seeded: dict[str, dict] = {}
        for row in load_jsonl(worker_dir / "generations.jsonl"):
            key = str(row.get("key"))
            question = assigned.get(key)
            if question is not None and qa_generation_record_is_complete(
                row, qa_prompt(model_key, question["question"])
            ):
                seeded[key] = row
        # The consolidated file is authoritative and lets a first multi-GPU
        # resume reuse rows produced by an earlier single-GPU run.
        for key, question in assigned.items():
            row = main_rows.get(key)
            if row is not None and qa_generation_record_is_complete(
                row, qa_prompt(model_key, question["question"])
            ):
                seeded[key] = row
        atomic_write_jsonl(
            worker_dir / "generations.jsonl",
            [seeded[key] for key in sorted(seeded)],
        )
        jobs.append(
            {
                "model_key": model_key,
                "model_cfg": model_cfg,
                "questions": partition,
                "worker_dir": str(worker_dir),
                "device": device,
                "checkpoint_every": checkpoint_every,
            }
        )
    print(
        f"[qa_pipeline] Parallel generation: {len(devices)} workers on "
        f"{list(devices)}."
    )
    _run_spawn_workers(_generation_worker, jobs, "generation")

    merged: dict[str, dict] = {}
    main_path = Path(output_dir) / "generations.jsonl"
    for row in load_jsonl(main_path):
        key = str(row.get("key"))
        question = expected.get(key)
        if question is not None and qa_generation_record_is_complete(
            row, qa_prompt(model_key, question["question"])
        ):
            merged[key] = row
    failure_rows: dict[str, dict] = {
        str(row.get("key")): row
        for row in load_jsonl(Path(output_dir) / "generation_failures.jsonl")
    }
    for worker_id, partition in enumerate(partitions):
        worker_dir = root / f"worker-{worker_id:03d}"
        assigned = {str(question["key"]) for question in partition}
        for row in load_jsonl(worker_dir / "generations.jsonl"):
            key = str(row.get("key"))
            if key not in assigned:
                continue
            question = expected[key]
            if qa_generation_record_is_complete(
                row, qa_prompt(model_key, question["question"])
            ):
                merged[key] = row
                failure_rows.pop(key, None)
        for row in load_jsonl(worker_dir / "generation_failures.jsonl"):
            key = str(row.get("key"))
            if key in assigned and key not in merged:
                failure_rows[key] = row
    atomic_write_jsonl(
        main_path,
        [merged[str(question["key"])] for question in questions if str(question["key"]) in merged],
    )
    atomic_write_jsonl(
        Path(output_dir) / "generation_failures.jsonl",
        [failure_rows[key] for key in sorted(failure_rows)],
    )


def _run_parallel_extraction(
    *,
    model_key: str,
    model_cfg: dict,
    questions: list[dict],
    output_dir: str,
    devices: tuple[str, ...],
    dgst_cfg: dict,
    ads_cfg: dict,
    cgc_cfg: dict,
    shard_size: int,
    position_protocols: tuple[str, ...],
    extraction_fingerprint: str,
    method_enabled: bool,
    ads_cgc_enabled: bool,
    baseline_specs: list[dict],
) -> list[dict]:
    root_enabled = bool(method_enabled or ads_cgc_enabled)
    partitions = _question_partitions(questions, len(devices))
    root = _parallel_worker_root(output_dir, "features", len(devices))
    generations = {
        str(row["key"]): row
        for row in load_jsonl(Path(output_dir) / "generations.jsonl")
    }
    labels = {
        str(row["key"]): row
        for row in load_jsonl(Path(output_dir) / "labels.jsonl")
    }
    jobs = []
    for worker_id, (device, partition) in enumerate(zip(devices, partitions)):
        worker_dir = root / f"worker-{worker_id:03d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(
            worker_dir / "generations.jsonl",
            [generations[str(question["key"])] for question in partition],
        )
        atomic_write_jsonl(
            worker_dir / "labels.jsonl",
            [labels[str(question["key"])] for question in partition],
        )
        jobs.append(
            {
                "model_key": model_key,
                "model_cfg": model_cfg,
                "questions": partition,
                "worker_dir": str(worker_dir),
                "device": device,
                "dgst_cfg": dgst_cfg,
                "ads_cfg": ads_cfg,
                "cgc_cfg": cgc_cfg,
                "shard_size": shard_size,
                "position_protocols": position_protocols,
                "extraction_fingerprint": extraction_fingerprint,
                "method_enabled": method_enabled,
                "ads_cgc_enabled": ads_cgc_enabled,
                "baseline_specs": baseline_specs,
                "worker_id": worker_id,
            }
        )
    print(
        f"[qa_pipeline] Parallel feature extraction: {len(devices)} workers on "
        f"{list(devices)}."
    )
    _run_spawn_workers(_feature_worker, jobs, "feature extraction")

    rows: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    for worker_id, partition in enumerate(partitions):
        worker_dir = root / f"worker-{worker_id:03d}"
        assigned = {str(question["key"]) for question in partition}
        feature_path = worker_dir / "features.pkl"
        if root_enabled and feature_path.exists():
            with feature_path.open("rb") as handle:
                worker_rows = pickle.load(handle)
            if not isinstance(worker_rows, list):
                raise ValueError(f"Invalid worker feature file: {feature_path}")
            for row in worker_rows:
                key = str(row.get("key"))
                if key not in assigned:
                    continue
                if key in rows and rows[key] != row:
                    raise RuntimeError(f"Conflicting QA feature rows for {key}")
                rows[key] = row
        for failure in load_jsonl(worker_dir / "extraction_failures.jsonl"):
            key = str(failure.get("key"))
            if key in assigned and key not in rows:
                failures[key] = failure
    ordered = [
        rows[str(question["key"])]
        for question in questions
        if str(question["key"]) in rows
    ]
    if root_enabled:
        _atomic_pickle(Path(output_dir) / "features.pkl", ordered)
    atomic_write_jsonl(
        Path(output_dir) / "extraction_failures.jsonl",
        [failures[key] for key in sorted(failures)],
    )
    return ordered


def _atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _prepare_baseline_specs(
    *,
    config: dict,
    model: str,
    dataset: str,
    model_cfg: dict,
    questions: list[dict],
    questions_path: Path,
    generations_path: Path,
    labels_path: Path,
    run_dir: Path,
    label_protocols: tuple[str, ...],
    resume: bool,
) -> list[dict]:
    """Validate provenance and describe every baseline output transaction."""

    generations = {
        str(row["key"]): row for row in load_jsonl(generations_path)
    }
    labels = {str(row["key"]): row for row in load_jsonl(labels_path)}
    all_questions = {str(row["key"]): row for row in questions}
    baseline_cfg = _controlled_qa_baseline_config(baseline_config(config))
    methods = normalize_baseline_methods(
        baseline_cfg.get("methods") or DEFAULT_QA_BASELINES
    )
    if not methods:
        raise ValueError("QA baseline extraction requires at least one method")
    specs: list[dict] = []
    for raw_protocol in label_protocols:
        protocol = normalize_qa_label_protocol(raw_protocol)
        _validate_source_rows(
            question_map=all_questions,
            generations=generations,
            labels=labels,
            dataset=dataset,
            label_protocol=protocol,
        )
        selected_questions = [
            question
            for question in questions
            if qa_label_for_protocol(labels[str(question["key"])], protocol)
            is not None
        ]
        if not selected_questions:
            raise RuntimeError(
                f"No QA questions remain under baseline protocol {protocol!r}"
            )
        selected_keys = tuple(str(row["key"]) for row in selected_questions)
        baseline_dir = run_dir / "baseline" / protocol
        expected_manifest = _expected_baseline_manifest(
            model=model,
            dataset=dataset,
            label_protocol=protocol,
            model_cfg=model_cfg,
            baseline_cfg=baseline_cfg,
            methods=methods,
            questions_path=questions_path,
            generations_path=generations_path,
            labels_path=labels_path,
            selected_keys=selected_keys,
        )
        baseline_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "label_protocol": protocol,
            "baseline_dir": str(baseline_dir),
            "baseline_cfg": baseline_cfg,
            "methods": tuple(methods),
            "selected_keys": selected_keys,
            "cache_ids": validate_qa_cache_ids(selected_questions),
            "expected_manifest": expected_manifest,
        }
        if not _baseline_spec_complete(spec):
            _validate_or_initialize_manifest(
                baseline_dir / QA_BASELINE_MANIFEST_NAME,
                expected_manifest,
                baseline_dir=baseline_dir,
                resume=resume,
            )
        if (
            resume
            and (baseline_dir / "features.pkl").exists()
            and not any((baseline_dir / "feature_parts").glob("part-*.pkl"))
        ):
            raise RuntimeError(
                "Cannot safely resume QA baseline extraction from features.pkl "
                f"alone: {baseline_dir / 'feature_parts'} is missing"
            )
        specs.append(spec)
    return specs


def _baseline_spec_complete(spec: dict) -> bool:
    baseline_dir = Path(spec["baseline_dir"])
    manifest_path = baseline_dir / QA_BASELINE_MANIFEST_NAME
    feature_path = baseline_dir / "features.pkl"
    split_path = baseline_dir / QA_BASELINE_SPLIT_MANIFEST_NAME
    if not manifest_path.is_file() or not feature_path.is_file() or not split_path.is_file():
        return False
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") != "complete":
            return False
        for key, value in spec["expected_manifest"].items():
            if key != "status" and manifest.get(key) != value:
                return False
        return (
            manifest.get("features_sha256") == sha256_file(feature_path)
            and manifest.get("probe_splits_sha256") == sha256_file(split_path)
            and int(manifest.get("num_records", -1))
            == len(spec["selected_keys"])
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _finalize_baseline_spec(spec: dict, shard_size: int) -> None:
    baseline_dir = Path(spec["baseline_dir"])
    store = QABaselineFeatureStore(
        baseline_dir,
        shard_size=shard_size,
        resume=True,
    )
    feature_path = store.consolidate()
    records = [store.rows[key] for key in sorted(store.rows)]
    expected_keys = set(spec["selected_keys"])
    actual_keys = {str(record.get("key")) for record in records}
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"QA baseline {spec['label_protocol']} is incomplete: "
            f"missing={len(expected_keys - actual_keys)}, "
            f"extra={len(actual_keys - expected_keys)}"
        )
    for record in records:
        validate_baseline_record(record, required=spec["methods"])
    validate_halloc_cache_uniqueness(records)
    _validate_halloc_cache_files(records, baseline_dir)
    split_manifest = build_qa_probe_split_manifest(
        records,
        label_protocol=spec["label_protocol"],
    )
    split_path = baseline_dir / QA_BASELINE_SPLIT_MANIFEST_NAME
    atomic_write_json(split_path, split_manifest)
    completed = {
        **spec["expected_manifest"],
        "status": "complete",
        "num_records": len(records),
        "features_sha256": sha256_file(feature_path),
        "probe_splits_sha256": sha256_file(split_path),
        "question_counts": split_manifest["question_counts"],
        "image_counts": split_manifest["image_counts"],
        "shared_forward": True,
    }
    atomic_write_json(
        baseline_dir / QA_BASELINE_MANIFEST_NAME,
        completed,
    )


def _qa_extraction_config_fingerprint(
    *,
    config: dict,
    model_key: str,
    model_cfg: dict,
    dgst_cfg: dict,
    ads_cfg: dict,
    cgc_cfg: dict,
    position_protocols: tuple[str, ...],
    method_enabled: bool = True,
    ads_cgc_enabled: bool = True,
) -> str:
    """Hash every input that can change root QA feature values or labels."""

    repo_root = Path(__file__).resolve().parents[1]
    wrapper_by_model = {
        "llava_1_5_7b": "models/llava_wrapper.py",
        "llava_next_8b": "models/llava_next_wrapper.py",
        "llava_next_llama3_8b": "models/llava_next_wrapper.py",
        "internvl_2_5_8b": "models/internvl_wrapper.py",
        "qwen2_5_vl_7b": "models/qwen_wrapper.py",
        "qwen3_vl_8b": "models/qwen3_vl_wrapper.py",
        "llava_onevision_1_5_8b": "models/llava_onevision_wrapper.py",
    }
    source_files = [
        "features/qa_extractor.py",
        "features/dgst_t.py",
        "features/ads.py",
        "features/cgc.py",
        "features/extractor.py",
        "models/base_wrapper.py",
        "models/dgst_capture.py",
        "models/prompt_support.py",
        "models/prompt_target.py",
        wrapper_by_model.get(model_key, ""),
    ]
    source_hashes = {
        relative: _sha256_file(repo_root / relative)
        for relative in source_files
        if relative and (repo_root / relative).is_file()
    }
    qa_cfg = config.get("qa_benchmarks") or {}
    payload = {
        "schema_version": "qa-extraction-provenance-v1",
        "feature_schema_version": QA_FEATURE_SCHEMA_VERSION,
        "model_key": str(model_key),
        "model_cfg": model_cfg,
        "model_artifacts": _model_artifact_metadata(model_cfg.get("hf_name")),
        "dgst_cfg": dgst_cfg,
        "ads_cfg": ads_cfg,
        "cgc_cfg": cgc_cfg,
        "position_protocols": list(position_protocols),
        "feature_families": {
            "method": bool(method_enabled),
            "ads_cgc": bool(ads_cgc_enabled),
        },
        "label_protocols": list(qa_cfg.get("label_protocols") or ()),
        "dgst_runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "DGST_OT_SOLVER_OVERRIDE",
                "DGST_SINKHORN_REG",
                "DGST_SINKHORN_MAX_ITER",
                "DGST_SINKHORN_TOL",
                "DGST_SINKHORN_MAX_MARGINAL_ERROR",
                "DGST_SINKHORN_BATCH_SIZE",
            )
        },
        "source_hashes": source_hashes,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_artifact_metadata(hf_name) -> dict:
    if not hf_name:
        return {"path": None}
    path = Path(str(hf_name)).expanduser()
    result = {"path": str(path.resolve()) if path.exists() else str(path)}
    if not path.is_dir():
        if path.is_file():
            stat = path.stat()
            result["file"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return result
    metadata_names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    result["metadata_sha256"] = {
        name: _sha256_file(path / name)
        for name in metadata_names
        if (path / name).is_file()
    }
    weight_patterns = ("*.safetensors", "pytorch_model*.bin")
    weight_paths = sorted(
        {candidate for pattern in weight_patterns for candidate in path.glob(pattern)}
    )
    result["weight_files"] = [
        {
            "name": candidate.name,
            "size": candidate.stat().st_size,
            "mtime_ns": candidate.stat().st_mtime_ns,
        }
        for candidate in weight_paths
    ]
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qa_model_cfg(
    config: dict,
    model_key: str,
    dataset: str | None = None,
) -> dict:
    """Resolve one model config plus QA generation/dataset preprocessing overrides."""

    model_cfg = deepcopy(get_model_cfg(config, model_key))
    qa_cfg = config.get("qa_benchmarks") or {}
    generation_cfg = qa_cfg.get("generation") or {}
    if not isinstance(generation_cfg, dict):
        raise ValueError("qa_benchmarks.generation must be a YAML mapping")
    allowed = {"max_new_tokens", "temperature", "top_p"}
    unknown = sorted(set(generation_cfg) - allowed)
    if unknown:
        raise ValueError(f"Unknown QA generation overrides: {unknown}")
    model_cfg.update(generation_cfg)

    dataset_overrides = qa_cfg.get("dataset_model_overrides") or {}
    if not isinstance(dataset_overrides, dict):
        raise ValueError("qa_benchmarks.dataset_model_overrides must be a YAML mapping")
    if dataset is not None:
        overrides = dataset_overrides.get(str(dataset)) or {}
        if not isinstance(overrides, dict):
            raise ValueError(
                "qa_benchmarks.dataset_model_overrides."
                f"{dataset} must be a YAML mapping"
            )
        allowed_dataset_keys = {"max_pixels", "min_pixels"}
        unknown_dataset_keys = sorted(set(overrides) - allowed_dataset_keys)
        if unknown_dataset_keys:
            raise ValueError(
                f"Unknown QA model overrides for {dataset}: {unknown_dataset_keys}"
            )
        for key, value in overrides.items():
            numeric = int(value)
            if numeric <= 0:
                raise ValueError(
                    f"qa_benchmarks.dataset_model_overrides.{dataset}.{key} "
                    "must be a positive integer"
                )
            model_cfg[key] = numeric
    return model_cfg


def main():
    args = parse_args()
    generation_devices = _normalize_devices(args.generation_devices, args.device)
    feature_devices = _normalize_devices(args.feature_devices, args.device)
    config = load_config(args.config)
    qa_cfg = config.get("qa_benchmarks") or {}
    family_flags = qa_extraction_family_flags(config)
    extraction_mode = str(family_flags["mode"])
    method_enabled = bool(family_flags["method"])
    ads_cgc_enabled = bool(family_flags["ads_cgc"])
    baseline_enabled = bool(family_flags["baseline"])
    root_enabled = method_enabled or ads_cgc_enabled
    if not root_enabled and not baseline_enabled:
        raise ValueError(
            f"QA extraction_mode={extraction_mode!r} has no enabled family"
        )
    baseline_label_protocols = tuple(dict.fromkeys(
        normalize_qa_label_protocol(value)
        for value in (
            args.baseline_label_protocols
            or qa_cfg.get(
                "baseline_label_protocols",
                ["object_hallucination_yes_only"],
            )
        )
    ))
    position_protocols = tuple(dict.fromkeys(
        str(value)
        for value in qa_cfg.get("position_protocols", ["prompt_last_token"])
    ))
    allowed_positions = {"prompt_last_token", "question_object_pre_token"}
    unknown_positions = sorted(set(position_protocols) - allowed_positions)
    if unknown_positions:
        raise ValueError(f"Unknown QA position protocols: {unknown_positions}")
    if "prompt_last_token" not in position_protocols:
        raise ValueError("QA pipeline currently requires prompt_last_token")
    model_cfg = _qa_model_cfg(config, args.model, args.dataset)
    dgst_cfg = config["feature_extraction"]["dgst_t"]
    ads_cfg = get_ads_cfg(config)
    cgc_cfg = get_cgc_cfg(config)
    extraction_fingerprint = _qa_extraction_config_fingerprint(
        config=config,
        model_key=args.model,
        model_cfg=model_cfg,
        dgst_cfg=dgst_cfg,
        ads_cfg=ads_cfg,
        cgc_cfg=cgc_cfg,
        position_protocols=position_protocols,
        method_enabled=method_enabled,
        ads_cgc_enabled=ads_cgc_enabled,
    )
    prepared_root = args.prepared_root or qa_cfg.get("prepared_root")
    output_root = args.output_root or qa_cfg.get("output_root")
    if not prepared_root or not output_root:
        raise ValueError(
            "prepared/output roots must be provided by CLI or qa_benchmarks YAML"
        )
    questions_path = os.path.join(prepared_root, args.dataset, "questions.jsonl")
    questions = load_jsonl(questions_path)
    if args.limit is not None:
        questions = questions[: args.limit]
    output_dir = os.path.join(output_root, args.model, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    splits_path = os.path.join(prepared_root, args.dataset, "image_splits.json")
    with open(splits_path, encoding="utf-8") as handle:
        atomic_write_json(os.path.join(output_dir, "image_splits.json"), json.load(handle))

    if not args.resume:
        stage_files = {
            "generate": (
                "generations.jsonl",
                "generation_failures.jsonl",
                ".qa_parallel/generation",
            ),
            "label": ("labels.jsonl",),
            "extract": (
                "features.pkl",
                "features.parts",
                "extraction_failures.jsonl",
                "qa_feature_summary.json",
                ".qa_parallel/features",
                *( ("baseline",) if baseline_enabled else () ),
            ),
            "all": (
                "generations.jsonl",
                "generation_failures.jsonl",
                "labels.jsonl",
                "features.pkl",
                "features.parts",
                "extraction_failures.jsonl",
                "qa_feature_summary.json",
                ".qa_parallel/generation",
                ".qa_parallel/features",
                *( ("baseline",) if baseline_enabled else () ),
            ),
        }[args.stage]
        existing = [name for name in stage_files if os.path.exists(os.path.join(output_dir, name))]
        if existing:
            raise FileExistsError(
                f"--no-resume refuses to overwrite {existing}; use a new output root"
            )

    expected_keys = {str(row["key"]) for row in questions}
    generation_rows = {
        str(row.get("key")): row
        for row in load_jsonl(os.path.join(output_dir, "generations.jsonl"))
    }
    label_rows = {
        str(row.get("key")): row
        for row in load_jsonl(os.path.join(output_dir, "labels.jsonl"))
    }
    generation_complete = _generations_complete(
        generation_rows, questions, args.model
    )
    feature_complete = (
        not root_enabled
        or _feature_keys_complete(
            os.path.join(output_dir, "features.pkl"),
            expected_keys,
            generation_rows,
            label_rows,
            questions,
            args.model,
            position_protocols,
            extraction_fingerprint,
            method_enabled=method_enabled,
            ads_cgc_enabled=ads_cgc_enabled,
        )
    )
    wrapper = None
    wrapper_device = None

    def get_single_wrapper(device: str):
        nonlocal wrapper, wrapper_device
        if wrapper is not None and wrapper_device != device:
            del wrapper
            wrapper = None
            wrapper_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if wrapper is None:
            wrapper = build_model(args.model, model_cfg, device=device)
            wrapper_device = device
        return wrapper

    def release_single_wrapper() -> None:
        nonlocal wrapper, wrapper_device
        if wrapper is not None:
            del wrapper
            wrapper = None
            wrapper_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.stage in ("generate", "all"):
        if generation_complete:
            print("[qa_pipeline] Resume: generations are complete; model generation skipped.")
        elif len(generation_devices) > 1:
            release_single_wrapper()
            _run_parallel_generation(
                model_key=args.model,
                model_cfg=model_cfg,
                questions=questions,
                output_dir=output_dir,
                devices=generation_devices,
                checkpoint_every=args.checkpoint_every,
            )
        else:
            generate_questions(
                get_single_wrapper(generation_devices[0]),
                args.model,
                questions,
                output_dir,
                args.checkpoint_every,
            )
    generation_rows = {
        str(row.get("key")): row
        for row in load_jsonl(os.path.join(output_dir, "generations.jsonl"))
    }
    if not _generations_complete(generation_rows, questions, args.model):
        invalid = [
            str(question["key"])
            for question in questions
            if not qa_generation_record_is_complete(
                generation_rows.get(str(question["key"]), {}),
                qa_prompt(args.model, question["question"]),
            )
        ]
        raise RuntimeError(
            "QA generation is incomplete or lacks a valid saved yes/no answer "
            f"token for {len(invalid)} questions (examples: {invalid[:5]}). "
            "Resume generation before labeling or extraction."
        )
    if args.stage in ("label", "all"):
        label_generations(questions, output_dir, args.checkpoint_every)
    if args.stage in ("extract", "all"):
        # Generation and labeling may have changed earlier in this same run.
        # Re-evaluate feature provenance after both stages before deciding to
        # resume, otherwise stale embedded labels could be silently retained.
        generation_rows = {
            str(row.get("key")): row
            for row in load_jsonl(os.path.join(output_dir, "generations.jsonl"))
        }
        label_rows = {
            str(row.get("key")): row
            for row in load_jsonl(os.path.join(output_dir, "labels.jsonl"))
        }
        feature_complete = (
            not root_enabled
            or _feature_keys_complete(
                os.path.join(output_dir, "features.pkl"),
                expected_keys,
                generation_rows,
                label_rows,
                questions,
                args.model,
                position_protocols,
                extraction_fingerprint,
                method_enabled=method_enabled,
                ads_cgc_enabled=ads_cgc_enabled,
            )
        )
        baseline_specs = (
            _prepare_baseline_specs(
                config=config,
                model=args.model,
                dataset=args.dataset,
                model_cfg=model_cfg,
                questions=questions,
                questions_path=Path(questions_path),
                generations_path=Path(output_dir) / "generations.jsonl",
                labels_path=Path(output_dir) / "labels.jsonl",
                run_dir=Path(output_dir),
                label_protocols=baseline_label_protocols,
                resume=bool(args.resume),
            )
            if baseline_enabled
            else []
        )
        pending_baseline_specs = [
            spec for spec in baseline_specs if not _baseline_spec_complete(spec)
        ]
        if feature_complete and not pending_baseline_specs:
            print(
                "[qa_pipeline] Resume: all selected QA feature families are "
                "complete; extraction skipped."
            )
            features = []
            if root_enabled:
                with open(os.path.join(output_dir, "features.pkl"), "rb") as handle:
                    features = pickle.load(handle)
        elif len(feature_devices) > 1:
            release_single_wrapper()
            features = _run_parallel_extraction(
                model_key=args.model,
                model_cfg=model_cfg,
                questions=questions,
                output_dir=output_dir,
                devices=feature_devices,
                dgst_cfg=dgst_cfg,
                ads_cfg=ads_cfg,
                cgc_cfg=cgc_cfg,
                shard_size=args.feature_shard_size,
                position_protocols=position_protocols,
                extraction_fingerprint=extraction_fingerprint,
                method_enabled=bool(method_enabled and not feature_complete),
                ads_cgc_enabled=bool(ads_cgc_enabled and not feature_complete),
                baseline_specs=pending_baseline_specs,
            )
            if feature_complete and root_enabled:
                with open(os.path.join(output_dir, "features.pkl"), "rb") as handle:
                    features = pickle.load(handle)
        else:
            with ExitStack() as stack:
                consumers = _open_baseline_consumers(
                    stack=stack,
                    wrapper=get_single_wrapper(feature_devices[0]),
                    baseline_specs=pending_baseline_specs,
                    device=feature_devices[0],
                    shard_size=args.feature_shard_size,
                    worker_id=0,
                    parallel=False,
                )
                features = extract_questions(
                    get_single_wrapper(feature_devices[0]),
                    args.model,
                    questions,
                    output_dir,
                    dgst_cfg,
                    ads_cfg,
                    cgc_cfg,
                    args.feature_shard_size,
                    position_protocols=position_protocols,
                    extraction_fingerprint=extraction_fingerprint,
                    method_enabled=bool(method_enabled and not feature_complete),
                    ads_cgc_enabled=bool(ads_cgc_enabled and not feature_complete),
                    baseline_consumers=consumers,
                )
            if feature_complete and root_enabled:
                with open(os.path.join(output_dir, "features.pkl"), "rb") as handle:
                    features = pickle.load(handle)
        for spec in pending_baseline_specs:
            _finalize_baseline_spec(spec, args.feature_shard_size)
        if root_enabled and not _feature_keys_complete(
            os.path.join(output_dir, "features.pkl"),
            expected_keys,
            generation_rows,
            label_rows,
            questions,
            args.model,
            position_protocols,
            extraction_fingerprint,
            method_enabled=method_enabled,
            ads_cgc_enabled=ads_cgc_enabled,
        ):
            raise RuntimeError(
                "QA root feature extraction is incomplete; inspect "
                "extraction_failures.jsonl and resume."
            )
        incomplete_baselines = [
            spec["label_protocol"]
            for spec in baseline_specs
            if not _baseline_spec_complete(spec)
        ]
        if incomplete_baselines:
            raise RuntimeError(
                f"QA baseline extraction is incomplete: {incomplete_baselines}"
            )
        statuses = {}
        for row in features:
            status = str((row.get("question_object_position") or {}).get("status", "missing"))
            statuses[status] = statuses.get(status, 0) + 1
        atomic_write_json(
            os.path.join(output_dir, "qa_feature_summary.json"),
            {
                "feature_schema_version": QA_FEATURE_SCHEMA_VERSION,
                "num_questions": len(features),
                "position_protocols": list(position_protocols),
                "extraction_fingerprint": extraction_fingerprint,
                "extraction_mode": extraction_mode,
                "feature_families": {
                    "method": method_enabled,
                    "ads_cgc": ads_cgc_enabled,
                    "baseline": baseline_enabled,
                },
                "baseline_label_protocols": (
                    list(baseline_label_protocols) if baseline_enabled else []
                ),
                "baseline_record_counts": {
                    spec["label_protocol"]: len(spec["selected_keys"])
                    for spec in baseline_specs
                },
                "prompt_last_token": sum(
                    "prompt_last_token" in row.get("positions", {}) for row in features
                ),
                "question_object_pre_token": sum(
                    "question_object_pre_token" in row.get("positions", {})
                    for row in features
                ),
                "question_object_status_counts": statuses,
            },
        )
    release_single_wrapper()
    print(f"[qa_pipeline] Complete: {args.model}/{args.dataset}/{args.stage}")


def _generations_complete(
    generations: dict[str, dict],
    questions: list[dict],
    model_key: str,
) -> bool:
    for question in questions:
        row = generations.get(str(question["key"]))
        prompt = qa_prompt(model_key, question["question"])
        if row is None or not qa_generation_record_is_complete(row, prompt):
            return False
    return True


def _feature_keys_complete(
    path: str,
    expected_keys: set[str],
    generations: dict[str, dict],
    labels: dict[str, dict],
    questions: list[dict],
    model_key: str,
    position_protocols: tuple[str, ...],
    extraction_fingerprint: str,
    *,
    method_enabled: bool = True,
    ads_cgc_enabled: bool = True,
) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False
    if not isinstance(rows, list):
        return False
    selected = {str(row.get("key")): row for row in rows if isinstance(row, dict)}
    if (
        len(selected) != len(rows)
        or set(selected) != set(expected_keys)
    ):
        return False
    question_map = {str(row["key"]): row for row in questions}
    for key in expected_keys:
        generation = generations.get(key)
        label_row = labels.get(key)
        question = question_map[key]
        if generation is None or label_row is None:
            return False
        expected_fingerprint = qa_generation_fingerprint(
            generation, qa_prompt(model_key, question["question"])
        )
        expected_label_fingerprint = qa_label_fingerprint(label_row)
        expected_question_fingerprint = qa_question_input_fingerprint(
            question, qa_prompt(model_key, question["question"])
        )
        selected_row = selected[key]
        selected_positions = selected_row.get("positions")
        if selected_row.get("feature_families") != {
            "method": bool(method_enabled),
            "ads_cgc": bool(ads_cgc_enabled),
        }:
            return False
        object_status = str(
            (selected_row.get("question_object_position") or {}).get("status")
            or "missing"
        )
        required_positions = ["prompt_last_token"]
        if "question_object_pre_token" in position_protocols:
            if object_status == "extracted":
                required_positions.append("question_object_pre_token")
            elif (
                str(question.get("object_span_status") or "unavailable")
                == "found"
            ):
                return False
        if (
            selected_row.get("feature_schema_version")
            != QA_FEATURE_SCHEMA_VERSION
            or selected_row.get("generation_fingerprint")
            != expected_fingerprint
            or selected_row.get("label_fingerprint")
            != expected_label_fingerprint
            or selected_row.get("question_input_fingerprint")
            != expected_question_fingerprint
            or selected_row.get("extraction_fingerprint")
            != extraction_fingerprint
            or tuple(selected_row.get("position_protocols") or ())
            != tuple(position_protocols)
            or not isinstance(selected_positions, dict)
            or any(position not in selected_positions for position in required_positions)
        ):
            return False
        for position in required_positions:
            value = selected_positions[position]
            if not isinstance(value, dict):
                return False
            if method_enabled and "dgst" not in value:
                return False
            if ads_cgc_enabled and any(
                field not in value
                for field in (
                    "ads_score",
                    "ads_per_layer",
                    "cgc_score",
                    "cgc_per_layer",
                )
            ):
                return False
    return True


if __name__ == "__main__":
    main()
