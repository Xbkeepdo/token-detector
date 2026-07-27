#!/usr/bin/env python3
"""Extract MetaToken/SVAR/DHCP/ProjectAway/HalLoc for QA answers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from multiprocessing import get_context
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import (  # noqa: E402
    atomic_write_json,
    load_jsonl,
    sha256_file,
)
from features.baseline import (  # noqa: E402
    BaselineRuntime,
    baseline_config,
    normalize_baseline_methods,
    validate_baseline_record,
)
from features.qa_baseline import (  # noqa: E402
    DEFAULT_QA_BASELINES,
    DEFAULT_QA_LABEL_PROTOCOL,
    QA_BASELINE_PROTOCOL,
    QA_CACHE_ID_SCHEME,
    QA_LABEL_PROTOCOLS,
    QABaselineAdapter,
    QABaselineFeatureStore,
    build_qa_probe_split_manifest,
    normalize_qa_label_protocol,
    qa_baseline_feature_config,
    qa_label_for_protocol,
    resolve_qa_answer_index,
    validate_halloc_cache_uniqueness,
    validate_qa_cache_ids,
)
from features.qa_extractor import qa_prompt  # noqa: E402
from models import build_model  # noqa: E402
from utils.config_utils import (  # noqa: E402
    get_model_cfg,
    load_config,
    manifest_validation_enabled,
)
from utils.generation_provenance import stable_sha256  # noqa: E402
from utils.qa_paths import resolve_qa_output_name, resolve_qa_paths  # noqa: E402


MANIFEST_NAME = "qa_baseline_manifest.json"
SPLIT_MANIFEST_NAME = "qa_probe_splits.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract prompt-last-token QA adaptations of the paper baselines."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=(
            "pope",
            "clevr_exist_9k",
            "clevr_exist_5k",
            "amber_discriminative",
        ),
        required=True,
    )
    parser.add_argument(
        "--config", default="configs/model_configs_unified.yaml"
    )
    parser.add_argument("--prepared-root")
    parser.add_argument("--output-root")
    parser.add_argument("--output", dest="output_name")
    parser.add_argument(
        "--experiment", dest="output_name", help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature-devices",
        nargs="+",
        default=None,
        help="Extract disjoint QA baseline shards on multiple GPUs.",
    )
    parser.add_argument(
        "--label-protocol",
        choices=QA_LABEL_PROTOCOLS,
        default=DEFAULT_QA_LABEL_PROTOCOL,
        help=(
            "answer_correctness_all uses labels.jsonl[label]; "
            "object_hallucination_yes_only uses "
            "object_hallucination_yes_only_label and excludes null rows"
        ),
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--feature-shard-size", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def _normalize_devices(values, fallback: str) -> tuple[str, ...]:
    devices = tuple(dict.fromkeys(str(value).strip() for value in (values or (fallback,))))
    if not devices or any(not value for value in devices):
        raise ValueError("QA baseline device lists cannot be empty")
    return devices


def _question_partitions(
    questions: Sequence[Mapping[str, Any]], num_workers: int
) -> list[list[dict]]:
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    partitions: list[list[dict]] = [[] for _ in range(int(num_workers))]
    for index, question in enumerate(questions):
        partitions[index % int(num_workers)].append(dict(question))
    return partitions


def _extract_partition(
    *,
    model: str,
    model_cfg: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
    methods: Sequence[str],
    baseline_dir: str,
    questions: Sequence[Mapping[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    cache_ids: Mapping[str, int],
    label_protocol: str,
    device: str,
    feature_shard_size: int,
    resume: bool,
    worker_id: int,
    parallel: bool,
) -> None:
    if not questions:
        return
    wrapper = build_model(model, dict(model_cfg), device=device)
    store = QABaselineFeatureStore(
        baseline_dir,
        shard_size=feature_shard_size,
        resume=resume,
        part_prefix=(f"part-worker{int(worker_id):03d}" if parallel else "part"),
    )
    with BaselineRuntime(
        wrapper=wrapper,
        methods=methods,
        baseline_dir=baseline_dir,
        config=baseline_cfg,
        device=device,
        resume=resume,
        worker_id=worker_id,
        parallel=parallel,
    ) as runtime:
        adapter = QABaselineAdapter(
            runtime,
            label_protocol=label_protocol,
        )
        for question in tqdm(
            questions,
            desc=(
                f"Extract QA baselines [{label_protocol}]"
                + (f" worker={worker_id}" if parallel else "")
            ),
        ):
            key = str(question["key"])
            if key in store.rows:
                continue
            generation = generations[key]
            label_row = labels[key]
            response_ids, _answer_index = resolve_qa_answer_index(
                generation, wrapper.tokenizer
            )
            target_index = 0
            prompt = qa_prompt(model, str(question["question"]))
            with Image.open(question["image_path"]) as raw_image:
                image = raw_image.convert("RGB")
            try:
                outputs = wrapper.extract_token_features_batch(
                    image=image,
                    response_token_ids=response_ids,
                    response_token_indices=[target_index],
                    target_token_ids=[response_ids[target_index]],
                    cfg_dgst_t=None,
                    prompt=prompt,
                    requirements=adapter.requirements,
                )
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"Expected one prompt-last output for {key}, got "
                        f"{len(outputs)}"
                    )
                record = adapter.build_record(
                    image=image,
                    question=question,
                    generation=generation,
                    label_row=label_row,
                    response_token_ids=response_ids,
                    target_index=target_index,
                    model_output=outputs[0],
                    cache_id=cache_ids[key],
                )
                store.add(record)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    store.flush()


def _run_parallel_partitions(jobs: list[dict]) -> None:
    context = get_context("spawn")
    processes = []
    for worker_id, job in enumerate(jobs):
        if not job.get("questions"):
            continue
        process = context.Process(
            target=_extract_partition,
            kwargs=job,
            name=f"qa-baseline-worker-{worker_id}",
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append((process.name, process.exitcode))
    if failures:
        raise RuntimeError(f"QA baseline workers failed: {failures}")


def main() -> None:
    args = parse_args()
    feature_devices = _normalize_devices(args.feature_devices, args.device)
    config = load_config(args.config)
    validate_manifests = manifest_validation_enabled(config)
    model_cfg = get_model_cfg(config, args.model)
    label_protocol = normalize_qa_label_protocol(args.label_protocol)
    qa_cfg = config.get("qa_benchmarks") or {}
    prepared_root = args.prepared_root or qa_cfg.get("prepared_root")
    output_root = args.output_root or qa_cfg.get("output_root")
    if not prepared_root or not output_root:
        raise ValueError("QA prepared/output roots are required by CLI or YAML")
    prepared_dir = Path(prepared_root) / args.dataset
    output_name = resolve_qa_output_name(args.output_name, qa_cfg)
    qa_paths = resolve_qa_paths(output_root, args.model, output_name, args.dataset)
    run_dir = qa_paths.benchmark_dir
    baseline_dir = run_dir / "baseline" / label_protocol
    questions_path = prepared_dir / "questions.jsonl"
    generations_path = qa_paths.generations_path
    labels_path = run_dir / "labels.jsonl"
    questions = load_jsonl(questions_path)
    if args.limit is not None:
        if int(args.limit) <= 0:
            raise ValueError("--limit must be positive")
        questions = questions[: int(args.limit)]
    generations = _rows_by_key(load_jsonl(generations_path), "generations")
    labels = _rows_by_key(load_jsonl(labels_path), "labels")
    question_map = _rows_by_key(questions, "questions")
    _validate_source_rows(
        question_map=question_map,
        generations=generations,
        labels=labels,
        dataset=args.dataset,
        label_protocol=label_protocol,
    )
    questions = [
        question
        for question in questions
        if qa_label_for_protocol(labels[str(question["key"])], label_protocol)
        is not None
    ]
    if not questions:
        raise RuntimeError(
            f"No QA questions remain under label protocol {label_protocol!r}"
        )
    question_map = _rows_by_key(questions, "selected questions")
    cache_ids = validate_qa_cache_ids(questions)

    baseline_cfg = _controlled_qa_baseline_config(baseline_config(config))
    methods = normalize_baseline_methods(
        args.methods or baseline_cfg.get("methods") or DEFAULT_QA_BASELINES
    )
    if not methods:
        raise ValueError("At least one QA baseline method is required")
    if "svar" in methods:
        print(
            "[extract_qa_baselines] SVAR uses the controlled prompt-last-token "
            "adaptation. SVAR-official caption canonical/surface lookup has "
            "no reliable yes/no-QA analogue and is intentionally unavailable."
        )
    expected_manifest = _expected_manifest(
        model=args.model,
        dataset=args.dataset,
        label_protocol=label_protocol,
        model_cfg=model_cfg,
        baseline_cfg=baseline_cfg,
        methods=methods,
        questions_path=questions_path,
        generations_path=generations_path,
        labels_path=labels_path,
        selected_keys=tuple(question_map),
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = baseline_dir / MANIFEST_NAME
    _validate_or_initialize_manifest(
        manifest_path,
        expected_manifest,
        baseline_dir=baseline_dir,
        resume=bool(args.resume),
        enabled=validate_manifests,
    )

    if (
        bool(args.resume)
        and (baseline_dir / "features.pkl").exists()
        and not any((baseline_dir / "feature_parts").glob("part-*.pkl"))
    ):
        raise RuntimeError(
            "Cannot safely resume QA baseline extraction from features.pkl alone; "
            "feature_parts are the transaction log. Use a new baseline directory."
        )

    if len(feature_devices) > 1:
        partitions = _question_partitions(questions, len(feature_devices))
        print(
            f"[extract_qa_baselines] Parallel extraction: "
            f"{len(feature_devices)} workers on {list(feature_devices)}."
        )
        jobs = []
        for worker_id, (device, partition) in enumerate(
            zip(feature_devices, partitions)
        ):
            jobs.append(
                {
                    "model": args.model,
                    "model_cfg": model_cfg,
                    "baseline_cfg": baseline_cfg,
                    "methods": methods,
                    "baseline_dir": str(baseline_dir),
                    "questions": partition,
                    "generations": generations,
                    "labels": labels,
                    "cache_ids": cache_ids,
                    "label_protocol": label_protocol,
                    "device": device,
                    "feature_shard_size": args.feature_shard_size,
                    # The parent already enforced --no-resume before workers
                    # start. Workers must tolerate shards atomically committed
                    # by their peers during this same fresh parallel run.
                    "resume": True,
                    "worker_id": worker_id,
                    "parallel": True,
                }
            )
        _run_parallel_partitions(jobs)
    else:
        _extract_partition(
            model=args.model,
            model_cfg=model_cfg,
            baseline_cfg=baseline_cfg,
            methods=methods,
            baseline_dir=str(baseline_dir),
            questions=questions,
            generations=generations,
            labels=labels,
            cache_ids=cache_ids,
            label_protocol=label_protocol,
            device=feature_devices[0],
            feature_shard_size=args.feature_shard_size,
            resume=bool(args.resume),
            worker_id=0,
            parallel=False,
        )

    store = QABaselineFeatureStore(
        baseline_dir,
        shard_size=args.feature_shard_size,
        resume=True,
    )

    feature_path = store.consolidate()
    records = [store.rows[key] for key in sorted(store.rows)]
    expected_keys = set(question_map)
    actual_keys = {str(record["key"]) for record in records}
    if actual_keys != expected_keys:
        raise RuntimeError(
            "QA baseline feature cohort is incomplete: "
            f"missing={len(expected_keys - actual_keys)}, "
            f"extra={len(actual_keys - expected_keys)}"
        )
    for record in records:
        validate_baseline_record(record, required=methods)
    validate_halloc_cache_uniqueness(records)
    _validate_halloc_cache_files(records, baseline_dir)
    split_manifest = build_qa_probe_split_manifest(
        records,
        label_protocol=label_protocol,
    )
    split_path = baseline_dir / SPLIT_MANIFEST_NAME
    atomic_write_json(split_path, split_manifest)
    completed = {
        **expected_manifest,
        "status": "complete",
        "num_records": len(records),
        "features_sha256": sha256_file(feature_path),
        "probe_splits_sha256": sha256_file(split_path),
        "question_counts": split_manifest["question_counts"],
        "image_counts": split_manifest["image_counts"],
    }
    if validate_manifests:
        atomic_write_json(manifest_path, completed)
    print(
        f"[extract_qa_baselines] Complete: {args.model}/{args.dataset}/"
        f"{label_protocol}; records={len(records)}, methods={list(methods)}, "
        f"output={baseline_dir}"
    )


def _rows_by_key(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for raw_row in rows:
        row = dict(raw_row)
        key = str(row.get("key") or "").strip()
        if not key:
            raise ValueError(f"{name} row is missing key")
        if key in result:
            raise ValueError(f"{name} contains duplicate key {key}")
        result[key] = row
    return result


def _validate_source_rows(
    *,
    question_map: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    dataset: str,
    label_protocol: str,
) -> None:
    expected = set(question_map)
    for name, values in (("generations", generations), ("labels", labels)):
        selected = expected & set(values)
        if selected != expected:
            raise RuntimeError(
                f"QA {name} are incomplete for the selected cohort: "
                f"missing={sorted(expected - selected)[:10]}"
            )
    for key, question in question_map.items():
        if str(question.get("dataset")) != str(dataset):
            raise ValueError(f"Question {key} belongs to another dataset")
        generation = generations[key]
        label = labels[key]
        for name, row in (("generation", generation), ("label", label)):
            if str(row.get("dataset")) != str(dataset):
                raise ValueError(f"{name} {key} belongs to another dataset")
            if str(row.get("probe_split")) != str(question.get("probe_split")):
                raise ValueError(f"{name} {key} has a different probe_split")
        qa_label_for_protocol(label, label_protocol)


def _controlled_qa_baseline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    svar = dict(result.get("svar") or {})
    svar["protocols"] = ["controlled"]
    result["svar"] = svar
    return result


def _expected_manifest(
    *,
    model: str,
    dataset: str,
    label_protocol: str,
    model_cfg: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
    methods: Sequence[str],
    questions_path: Path,
    generations_path: Path,
    labels_path: Path,
    selected_keys: Sequence[str],
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "artifact_family": "qa_baseline_prompt_last_token",
        "status": "in_progress",
        "model": str(model),
        "dataset": str(dataset),
        "protocol": QA_BASELINE_PROTOCOL,
        "label_protocol": normalize_qa_label_protocol(label_protocol),
        "label_field": (
            "label"
            if label_protocol == "answer_correctness_all"
            else "object_hallucination_yes_only_label"
        ),
        "cache_id_scheme": QA_CACHE_ID_SCHEME,
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "methods": list(methods),
        "model_config_sha256": stable_sha256(dict(model_cfg)),
        "feature_config_sha256": stable_sha256(
            qa_baseline_feature_config(baseline_cfg, methods)
        ),
        "questions_sha256": sha256_file(questions_path),
        "generations_sha256": sha256_file(generations_path),
        "labels_sha256": sha256_file(labels_path),
        "selected_question_keys_sha256": stable_sha256(sorted(selected_keys)),
        "num_selected_questions": len(selected_keys),
        "prompt_protocol": "features.qa_extractor.qa_prompt+answer_only_yes_or_no",
        "adaptation_notes": {
            "metatoken": "prompt-last causal-state adaptation",
            "svar": (
                "controlled prompt-last causal-state adaptation only; "
                "SVAR-official caption object lookup is unavailable"
            ),
            "dhcp": "prompt-last causal-state adaptation",
            "projectaway": "prompt-last causal-state adaptation",
            "halloc": "first response token is the adapted object-head position",
        },
    }


def _validate_or_initialize_manifest(
    path: Path,
    expected: Mapping[str, Any],
    *,
    baseline_dir: Path,
    resume: bool,
    enabled: bool = True,
) -> None:
    artifacts_exist = any(
        (
            (baseline_dir / "features.pkl").exists(),
            any((baseline_dir / "feature_parts").glob("part-*.pkl")),
            (baseline_dir / "dhcp").exists(),
            (baseline_dir / "halloc").exists(),
        )
    )
    if enabled and path.exists():
        import json

        with path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        mismatches = {
            key: (previous.get(key), value)
            for key, value in expected.items()
            if key != "status" and previous.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"QA baseline resume provenance mismatch: {mismatches}"
            )
    elif enabled and artifacts_exist:
        raise RuntimeError(
            f"QA baseline artifacts exist without {MANIFEST_NAME}; refusing "
            "unverifiable resume. Use a new baseline directory."
        )
    if artifacts_exist and not resume:
        raise FileExistsError(
            f"QA baseline artifacts already exist: {baseline_dir}"
        )
    if enabled:
        atomic_write_json(path, dict(expected))


def _validate_halloc_cache_files(
    records: Sequence[Mapping[str, Any]], baseline_dir: Path
) -> None:
    root = baseline_dir.resolve()
    for record in records:
        payload = (record.get("baselines") or {}).get("halloc")
        if not isinstance(payload, Mapping) or "cache_file" not in payload:
            continue
        path = (root / str(payload["cache_file"])).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"HalLoc cache path escapes baseline directory: {path}")
        if not path.is_file():
            raise FileNotFoundError(
                f"HalLoc cache referenced by {record.get('key')} is missing: {path}"
            )


if __name__ == "__main__":
    main()
