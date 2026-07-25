#!/usr/bin/env python3
"""Train and compare COCO-aligned baseline heads on QA question splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import sha256_file  # noqa: E402
from detection.qa_probe import train_one_seed  # noqa: E402
from features.baseline import (  # noqa: E402
    baseline_config,
    normalize_baseline_methods,
    validate_baseline_record,
)
from features.qa_baseline import (  # noqa: E402
    DEFAULT_QA_LABEL_PROTOCOL,
    QA_BASELINE_PROTOCOL,
    QA_LABEL_PROTOCOLS,
    QA_SPLITS,
    build_qa_probe_split_manifest,
    normalize_qa_label_protocol,
    split_qa_baseline_records,
    validate_halloc_cache_uniqueness,
)
from scripts.extract_qa_baselines import (  # noqa: E402
    MANIFEST_NAME,
    SPLIT_MANIFEST_NAME,
)
from scripts.train_baselines import (  # noqa: E402
    _configured_baseline_trainers,
    _resolve_device,
    _run_shared_mlp_protocol,
    _run_training_protocol,
    _write_baseline_markdown,
    _write_trainer_comparison,
    aggregate_baseline_outputs,
)
from utils.config_utils import (  # noqa: E402
    load_config,
    qa_extraction_family_flags,
)
from utils.qa_paths import resolve_qa_output_name, resolve_qa_paths  # noqa: E402


DEFAULT_SEEDS = (43, 44, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QA-adapted baselines for seeds 43/44/45 by default."
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
    parser.add_argument("--output-root")
    parser.add_argument("--output", dest="output_name")
    parser.add_argument(
        "--experiment", dest="output_name", help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--label-protocol",
        choices=QA_LABEL_PROTOCOLS,
        default=DEFAULT_QA_LABEL_PROTOCOL,
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    trainer_group = parser.add_mutually_exclusive_group()
    trainer_group.add_argument(
        "--trainer",
        choices=("native_paper", "shared_torch_mlp"),
        default=None,
        help="Run one baseline trainer instead of training.baseline.trainers.",
    )
    trainer_group.add_argument(
        "--trainers",
        nargs="+",
        choices=("native_paper", "shared_torch_mlp"),
        default=None,
        help="Run one or both baseline trainers, overriding YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    family_flags = qa_extraction_family_flags(config)
    if not family_flags["baseline"]:
        print(
            "[train_qa_baselines] Skipped: QA extraction_mode="
            f"{family_flags['mode']} does not enable baselines."
        )
        return
    if str((config.get("training") or {}).get("split_protocol", "strict_82_no_validation")) != "strict_82_no_validation":
        raise ValueError(
            "QA baseline training requires training.split_protocol="
            "strict_82_no_validation"
        )
    if str((config.get("training") or {}).get("threshold_selection", "train_f1")) != "train_f1":
        raise ValueError(
            "QA baseline training requires training.threshold_selection=train_f1"
        )
    label_protocol = normalize_qa_label_protocol(args.label_protocol)
    qa_cfg = config.get("qa_benchmarks") or {}
    output_root = args.output_root or qa_cfg.get("output_root")
    if not output_root:
        raise ValueError("QA output root is required by CLI or YAML")
    output_name = resolve_qa_output_name(args.output_name, qa_cfg)
    run_dir = resolve_qa_paths(
        output_root, args.model, output_name, args.dataset
    ).benchmark_dir
    baseline_dir = run_dir / "baseline" / label_protocol
    feature_path = baseline_dir / "features.pkl"
    manifest_path = baseline_dir / MANIFEST_NAME
    split_path = baseline_dir / SPLIT_MANIFEST_NAME
    manifest = _load_json_object(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"QA baseline extraction is not complete according to {manifest_path}"
        )
    if str(manifest.get("model")) != args.model:
        raise RuntimeError("QA baseline manifest belongs to another model")
    if str(manifest.get("dataset")) != args.dataset:
        raise RuntimeError("QA baseline manifest belongs to another dataset")
    if str(manifest.get("protocol")) != QA_BASELINE_PROTOCOL:
        raise RuntimeError("Unsupported QA baseline extraction protocol")
    if str(manifest.get("label_protocol")) != label_protocol:
        raise RuntimeError(
            "QA baseline manifest belongs to another label protocol"
        )
    if sha256_file(feature_path) != manifest.get("features_sha256"):
        raise RuntimeError("QA baseline features.pkl hash differs from its manifest")
    labels_path = run_dir / "labels.jsonl"
    if sha256_file(labels_path) != manifest.get("labels_sha256"):
        raise RuntimeError(
            "QA labels changed after baseline extraction; re-extract baselines"
        )

    with feature_path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No QA baseline records found in {feature_path}")
    if len(records) != int(manifest.get("num_records", -1)):
        raise RuntimeError("QA baseline record count differs from its manifest")
    extracted_methods = normalize_baseline_methods(manifest.get("methods") or ())
    training_cfg = config.get("training") or {}
    baseline_training_cfg = training_cfg.get("baseline") or {}
    if not isinstance(baseline_training_cfg, Mapping):
        raise ValueError("training.baseline must be a YAML mapping")
    methods = normalize_baseline_methods(
        args.methods
        or baseline_training_cfg.get("methods")
        or extracted_methods
    )
    unavailable = set(methods) - set(extracted_methods)
    if unavailable:
        raise ValueError(
            f"Requested QA baselines were not extracted: {sorted(unavailable)}"
        )
    for record in records:
        validate_baseline_record(record, required=methods)
    validate_halloc_cache_uniqueness(records)

    split_records = split_qa_baseline_records(
        records,
        label_protocol=label_protocol,
    )
    _require_both_labels(split_records)
    expected_split_manifest = build_qa_probe_split_manifest(
        records,
        label_protocol=label_protocol,
    )
    stored_split_manifest = _load_json_object(split_path)
    if stored_split_manifest != expected_split_manifest:
        raise RuntimeError(
            "QA baseline probe split manifest differs from feature records"
        )
    seeds = list(dict.fromkeys(
        int(value)
        for value in (
            args.seeds
            or baseline_training_cfg.get("seeds")
            or DEFAULT_SEEDS
        )
    ))
    if not seeds:
        raise ValueError("At least one QA baseline seed is required")
    trainers = _configured_baseline_trainers(args, baseline_training_cfg)
    baseline_cfg = baseline_config(config)
    probe_cfg = training_cfg.get("torch_probe") or {}
    if "shared_torch_mlp" in trainers and (
        not isinstance(probe_cfg, Mapping) or not probe_cfg
    ):
        raise ValueError(
            "shared_torch_mlp requires training.torch_probe"
        )
    common_kwargs = {
        "model": args.model,
        "dataset": args.dataset,
        "label_protocol": label_protocol,
        "seeds": seeds,
        "methods": methods,
        "split_records": split_records,
        "image_split_counts": expected_split_manifest["image_counts"],
        "feature_path": feature_path,
        "split_path": split_path,
        "baseline_dir": baseline_dir,
        "baseline_cfg": baseline_cfg,
        "device": _resolve_device(args.device),
    }
    trainer_runs: dict[str, tuple[dict[str, Any], list[Path]]] = {}
    for trainer in trainers:
        if trainer == "shared_torch_mlp":
            trainer_runs[trainer] = run_qa_shared_mlp_training(
                **common_kwargs,
                probe_cfg=dict(probe_cfg),
            )
        else:
            trainer_runs[trainer] = run_qa_baseline_training(
                **common_kwargs,
                trainer_namespace=(
                    "native_paper" if len(trainers) > 1 else None
                ),
            )
    if len(trainer_runs) > 1:
        _write_trainer_comparison(
            result_root=baseline_dir,
            result_stem=(
                f"{args.model}_{args.dataset}_{label_protocol}_qa_baselines"
            ),
            trainer_runs=trainer_runs,
            probe_cfg=dict(probe_cfg),
        )
    print(
        f"[train_qa_baselines] Complete: {args.model}/{args.dataset}/"
        f"{label_protocol}; trainers={list(trainers)}, seeds={seeds}, "
        f"methods={list(methods)}, "
        f"output={baseline_dir / 'results'}"
    )


def _normalize_qa_baseline_trainer(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "native": "native_paper",
        "paper": "native_paper",
        "torch_mlp": "shared_torch_mlp",
        "shared_mlp": "shared_torch_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"native_paper", "shared_torch_mlp"}:
        raise ValueError(
            "qa_benchmarks.baseline_trainer must be native_paper or "
            f"shared_torch_mlp, got {value!r}"
        )
    return normalized


def run_qa_shared_mlp_training(
    *,
    model: str,
    dataset: str,
    label_protocol: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    image_split_counts: Mapping[str, int],
    feature_path: Path,
    split_path: Path,
    baseline_dir: Path,
    probe_cfg: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, Any], list[Path]]:
    """Run the exact shared-MLP protocol used by COCO baselines."""

    protocol = normalize_qa_label_protocol(label_protocol)
    return _run_shared_mlp_protocol(
        model=model,
        seeds=seeds,
        methods=methods,
        split_records=split_records,
        image_split_counts=image_split_counts,
        feature_path=feature_path,
        split_path=split_path,
        result_root=baseline_dir,
        probe_cfg=probe_cfg,
        baseline_cfg=baseline_cfg,
        device=device,
        run_name=None,
        result_stem=f"{model}_{dataset}_{protocol}_qa_baselines",
        label_protocol=(
            f"qa_{protocol}_0hall_1real_question_probe_split_"
            "physical_image_disjoint"
        ),
        write_summary=True,
        positive_class="real",
    )


def _qa_result_as_baseline_result(
    result: Mapping[str, Any],
    *,
    checkpoint: Path,
    probe_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    converted = {
        "paper_config": None,
        "adaptation": "shared_torch_mlp_for_controlled_trainer_comparison",
        "trainer_config": dict(probe_cfg),
        "input_dim": int(result["input_dim"]),
        "threshold": float(result["threshold"]),
        "threshold_score_class": "real",
        "train_metrics": _qa_metrics_as_baseline_metrics(
            result["train_metrics"]
        ),
        "test_metrics": _qa_metrics_as_baseline_metrics(
            result["test_metrics"]
        ),
        "checkpoint": str(checkpoint),
        "epochs_completed": int(result["epochs_completed"]),
        "best_epoch": int(result["best_epoch"]),
        "best_train_loss": float(result["best_train_loss"]),
        "selection_protocol": (
            "minimum_train_loss_dual_fixed_0.5_and_train_real_f1_threshold"
        ),
    }
    reports = result.get("threshold_reports") or {}
    if reports:
        converted["threshold_reports"] = {
            str(mode): {
                "threshold": float(report["threshold"]),
                "train_metrics": _qa_metrics_as_baseline_metrics(
                    report["train_metrics"]
                ),
                "test_metrics": _qa_metrics_as_baseline_metrics(
                    report["test_metrics"]
                ),
            }
            for mode, report in reports.items()
        }
    return converted


def _qa_metrics_as_baseline_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    real = metrics["real"]
    hall = metrics["hallucination"]
    auroc = float(metrics["auroc"])
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "real_positive": {
            "precision": float(real["precision"]),
            "recall": float(real["recall"]),
            "f1": float(real["f1"]),
            "auc": auroc,
            "aupr": float(metrics["real_aupr"]),
        },
        "hallucination_positive": {
            "precision": float(hall["precision"]),
            "recall": float(hall["recall"]),
            "f1": float(hall["f1"]),
            "auc": auroc,
            "aupr": float(metrics["hallucination_aupr"]),
        },
    }


def _record_split_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        split: sum(str(record.get("probe_split")) == split for record in records)
        for split in QA_SPLITS
    }


def _shared_mlp_fingerprint(
    *,
    feature_path: Path,
    split_path: Path,
    probe_cfg: Mapping[str, Any],
    methods: Sequence[str],
    label_protocol: str,
    svar_layer_start: int,
    svar_layer_end: int,
) -> tuple[str, dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    provenance = {
        "schema_version": "qa-baseline-shared-torch-mlp-v2",
        "features_sha256": sha256_file(feature_path),
        "splits_sha256": sha256_file(split_path),
        "probe_cfg": dict(probe_cfg),
        "methods": list(methods),
        "label_protocol": str(label_protocol),
        "svar_training_layers": {
            "start": int(svar_layer_start),
            "end_exclusive": int(svar_layer_end),
        },
        "code_sha256": {
            "detection/qa_probe.py": sha256_file(
                repo_root / "detection" / "qa_probe.py"
            ),
            "scripts/train_qa_baselines.py": sha256_file(Path(__file__)),
        },
    }
    encoded = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), provenance


def _validate_shared_mlp_seed_output(
    output: Mapping[str, Any],
    *,
    seed: int,
    methods: Sequence[str],
    fingerprint: str,
) -> None:
    expected = {
        "seed": int(seed),
        "trainer": "shared_torch_mlp",
        "configured_methods": list(methods),
        "training_input_fingerprint": fingerprint,
    }
    mismatches = {
        key: {"expected": value, "found": output.get(key)}
        for key, value in expected.items()
        if output.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse incompatible shared baseline MLP results; "
            f"pass --force to retrain. Mismatches: {mismatches}"
        )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_qa_baseline_training(
    *,
    model: str,
    dataset: str,
    label_protocol: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    image_split_counts: Mapping[str, int],
    feature_path: Path,
    split_path: Path,
    baseline_dir: Path,
    baseline_cfg: Mapping[str, Any],
    device: str,
    trainer_namespace: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Run native heads and write isolated per-seed plus aggregate reports."""

    protocol = normalize_qa_label_protocol(label_protocol)
    return _run_training_protocol(
        model=model,
        seeds=seeds,
        methods=methods,
        split_records=split_records,
        image_split_counts=image_split_counts,
        feature_path=feature_path,
        split_path=split_path,
        result_root=baseline_dir,
        baseline_cfg=baseline_cfg,
        device=device,
        run_name=None,
        result_stem=f"{model}_{dataset}_{protocol}_qa_baselines",
        label_protocol=(
            f"qa_{protocol}_0hall_1real_question_probe_split_"
            "physical_image_disjoint"
        ),
        write_summary=True,
        positive_class="real",
        trainer_namespace=trainer_namespace,
    )


def _load_json_object(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_both_labels(split_records) -> None:
    if split_records["val"]:
        raise ValueError("Strict QA 8:2 requires no validation records")
    for split in ("train", "test"):
        labels = {int(record["label"]) for record in split_records[split]}
        if labels != {0, 1}:
            raise ValueError(
                f"QA baseline {split} split must contain labels 0 and 1, got "
                f"{sorted(labels)}"
            )


if __name__ == "__main__":
    main()
