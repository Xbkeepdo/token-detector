#!/usr/bin/env python3
"""Train/evaluate paper baselines from ``OUTPUT/baseline/features.pkl``."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.baselines import (
    DHCPMLP,
    SVARMLP,
    baseline_vector,
    build_dense_baseline_matrix,
    build_metatoken_classifier,
    evaluate_detection_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
    sklearn_hallucination_scores,
    split_records_by_image,
    torch_hallucination_scores,
    train_torch_detector,
)
from features.baseline import (
    DHCPShardReader,
    HalLocObjectDetector,
    baseline_config,
    get_baseline_payload,
    halloc_optimizer_config,
    normalize_baseline_methods,
    normalize_svar_protocols,
    svar_training_vector,
    validate_baseline_record,
)
from scripts.training_provenance import load_validated_training_features
from scripts.train_torch_probe_feature_sets import (
    TorchProbeConfig,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, save_json, save_pkl
from utils.split_utils import validate_strict_82_split


DEFAULT_METHODS = ("metatoken", "svar", "dhcp", "projectaway", "halloc")
SUMMARY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "aupr",
    "other_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one seed, overriding training.baseline.seeds.",
    )
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run multiple seeds, overriding training.baseline.seeds.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override the configured baseline methods (for example, omit halloc).",
    )
    parser.add_argument(
        "--trainer",
        choices=("native_paper", "shared_torch_mlp"),
        default=None,
        help=(
            "Override training.baseline.trainer. shared_torch_mlp gives every "
            "dense baseline the same configured three-hidden-layer probe."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Store results/checkpoints in isolated subdirectories such as seed42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if str((config.get("training") or {}).get("split_protocol", "strict_82_no_validation")) != "strict_82_no_validation":
        raise ValueError(
            "Training requires training.split_protocol=strict_82_no_validation"
        )
    if str((config.get("training") or {}).get("threshold_selection", "train_f1")) != "train_f1":
        raise ValueError(
            "Strict 8:2 training requires training.threshold_selection=train_f1"
        )
    baseline_cfg = baseline_config(config)
    training_cfg = _baseline_training_config(config)
    baseline_dir = Path(args.output_dir) / str(
        baseline_cfg.get("output_subdir", "baseline")
    )
    split_path = Path(args.output_dir) / "image_splits.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    image_splits = load_json(str(split_path))
    image_split_counts = validate_strict_82_split(image_splits)
    configured_count = int((config.get("dataset") or {}).get("num_images", 0))
    if configured_count and sum(image_split_counts.values()) != configured_count:
        raise ValueError(
            "Strict split size differs from dataset.num_images: "
            f"{sum(image_split_counts.values())} != {configured_count}"
        )
    methods = normalize_baseline_methods(
        args.methods
        if args.methods is not None
        else training_cfg.get(
            "methods", baseline_cfg.get("methods", DEFAULT_METHODS)
        )
    )
    if not methods:
        raise ValueError("No baseline methods selected for training")
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
    official_svar_enabled = "svar" in methods and "official" in svar_protocols

    seeds = _configured_training_seeds(args, training_cfg, baseline_cfg)
    if args.run_name is not None and len(seeds) != 1:
        raise ValueError("--run-name can only be used with one effective seed")
    device = _resolve_device(args.device)
    write_summary = bool(training_cfg.get("write_summary", True))
    positive_class = _normalize_reporting_positive_class(
        training_cfg.get("positive_class", "real")
    )
    trainer = _normalize_baseline_trainer(
        args.trainer or training_cfg.get("trainer", "native_paper")
    )
    probe_cfg = (config.get("training") or {}).get("torch_probe") or {}
    if trainer == "shared_torch_mlp" and not isinstance(probe_cfg, Mapping):
        raise ValueError(
            "shared_torch_mlp requires training.torch_probe to be a mapping"
        )

    if controlled_methods:
        feature_path = baseline_dir / "features.pkl"
        split_records = _load_and_split_baseline_records(
            feature_path=feature_path,
            image_splits=image_splits,
            model_key=args.model,
            config=config,
            output_dir=Path(args.output_dir),
            expected_artifact_family="baseline_controlled",
            expected_svar_protocol=(
                "controlled" if "svar" in controlled_methods else None
            ),
        )
        training_kwargs = {
            "model": args.model,
            "seeds": seeds,
            "methods": controlled_methods,
            "split_records": split_records,
            "image_split_counts": image_split_counts,
            "feature_path": feature_path,
            "split_path": split_path,
            "result_root": baseline_dir,
            "device": device,
            "run_name": args.run_name,
            "result_stem": f"{args.model}_baselines",
            "label_protocol": (
                "shared_first_canonical_mention_exact_response_offsets"
            ),
            "write_summary": write_summary,
            "positive_class": positive_class,
        }
        if trainer == "shared_torch_mlp":
            _run_shared_mlp_protocol(
                **training_kwargs,
                probe_cfg=dict(probe_cfg),
                baseline_cfg=baseline_cfg,
            )
        else:
            _run_training_protocol(
                **training_kwargs,
                baseline_cfg=baseline_cfg,
            )

    if official_svar_enabled:
        official_dir = baseline_dir / "svar_official"
        official_feature_path = official_dir / "features.pkl"
        official_split_records = _load_and_split_baseline_records(
            feature_path=official_feature_path,
            image_splits=image_splits,
            model_key=args.model,
            config=config,
            output_dir=Path(args.output_dir),
            required=("svar",),
            expected_artifact_family="baseline_svar_official",
            expected_svar_protocol="official",
        )
        official_sample_audit = _official_svar_sample_audit(
            Path(args.output_dir) / "labeling.json",
            image_splits,
        )
        extracted_found = sum(
            len(records) for records in official_split_records.values()
        )
        expected_found = int(official_sample_audit["overall"]["found"])
        if expected_found != extracted_found:
            raise RuntimeError(
                "SVAR-official found sample count differs between labeling "
                f"and features: {expected_found} != {extracted_found}."
            )
        training_kwargs = {
            "model": args.model,
            "seeds": seeds,
            "methods": ("svar",),
            "split_records": official_split_records,
            "image_split_counts": image_split_counts,
            "feature_path": official_feature_path,
            "split_path": split_path,
            "result_root": official_dir,
            "device": device,
            "run_name": args.run_name,
            "result_stem": f"{args.model}_svar_official",
            "label_protocol": (
                "official_svar_set_first_token_id_first_occurrence"
            ),
            "sample_audit": official_sample_audit,
            "write_summary": write_summary,
            "positive_class": positive_class,
        }
        if trainer == "shared_torch_mlp":
            _run_shared_mlp_protocol(
                **training_kwargs,
                probe_cfg=dict(probe_cfg),
                baseline_cfg=baseline_cfg,
            )
        else:
            _run_training_protocol(
                **training_kwargs,
                baseline_cfg=baseline_cfg,
            )



def _official_svar_sample_audit(
    labeling_path: Path,
    image_splits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    if not labeling_path.exists():
        raise FileNotFoundError(labeling_path)
    labeling = load_json(str(labeling_path))
    if not isinstance(labeling, Mapping):
        raise RuntimeError(f"Invalid labeling file: {labeling_path}")
    image_to_split: dict[int, str] = {}
    for split in ("train", "val", "test"):
        for raw_image_id in image_splits[split]:
            image_id = int(raw_image_id)
            if image_id in image_to_split:
                raise RuntimeError(
                    f"Image {image_id} occurs in multiple data splits."
                )
            image_to_split[image_id] = split

    def empty_counts() -> dict[str, int]:
        return {
            "total": 0,
            "found": 0,
            "not_found": 0,
            "hallucination_total": 0,
            "real_total": 0,
            "hallucination_found": 0,
            "real_found": 0,
        }

    overall = empty_counts()
    by_split = {split: empty_counts() for split in ("train", "val", "test")}
    for raw_image_id, raw_row in labeling.items():
        image_id = int(raw_image_id)
        split = image_to_split.get(image_id)
        if split is None:
            raise RuntimeError(
                f"Labeling image {image_id} is outside image_splits.json."
            )
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(f"Invalid labeling row for image {image_id}.")
        for sample in raw_row.get("official_svar_samples") or []:
            if not isinstance(sample, Mapping):
                raise RuntimeError(
                    f"Invalid official SVAR sample for image {image_id}."
                )
            status = str(sample.get("status", "")).strip().lower()
            if status not in {"found", "not_found"}:
                raise RuntimeError(
                    f"Official SVAR sample for image {image_id} has status "
                    f"{status!r}."
                )
            label = int(sample.get("label", -1))
            if label not in (0, 1):
                raise RuntimeError(
                    f"Official SVAR sample for image {image_id} has invalid "
                    f"label {label}."
                )
            for bucket in (overall, by_split[split]):
                bucket["total"] += 1
                bucket[status] += 1
                label_name = "hallucination" if label == 0 else "real"
                bucket[f"{label_name}_total"] += 1
                if status == "found":
                    bucket[f"{label_name}_found"] += 1
    return {"overall": overall, "by_split": by_split}


def _load_and_split_baseline_records(
    *,
    feature_path: Path,
    image_splits: Mapping[str, Sequence[int]],
    model_key: str,
    config: Mapping[str, Any],
    output_dir: Path,
    required: Sequence[str] = (),
    expected_artifact_family: Optional[str] = None,
    expected_svar_protocol: Optional[str] = None,
) -> dict[str, list[Mapping[str, Any]]]:
    if expected_artifact_family is None:
        raise ValueError("expected_artifact_family is required")
    records = load_validated_training_features(
        feature_path=feature_path,
        artifact_family=expected_artifact_family,
        model_key=model_key,
        config=config,
        output_dir=output_dir,
        image_splits=image_splits,
    )
    if not records:
        raise RuntimeError(
            f"No extractable samples are available in {feature_path}."
        )
    split_image_ids = {
        int(image_id)
        for split_name in ("train", "val", "test")
        for image_id in image_splits[split_name]
    }
    feature_image_ids = {int(record.get("image_id", -1)) for record in records}
    unexpected_ids = sorted(feature_image_ids - split_image_ids)
    if unexpected_ids:
        raise RuntimeError(
            "Baseline features contain image IDs outside image_splits.json: "
            f"{unexpected_ids[:10]}"
        )
    for record in records:
        validate_baseline_record(record, required=required)
        if expected_svar_protocol is not None:
            actual_protocol = str(
                (record.get("metadata") or {}).get("svar_protocol") or ""
            )
            if actual_protocol != expected_svar_protocol:
                raise RuntimeError(
                    f"SVAR record protocol {actual_protocol!r} does not match "
                    f"expected {expected_svar_protocol!r}."
                )
            if expected_svar_protocol == "official" and set(
                record.get("baselines") or {}
            ) != {"svar"}:
                raise RuntimeError(
                    "SVAR-official feature records must contain only the SVAR "
                    "baseline payload."
                )
    split_records = split_records_by_image(records, image_splits)
    _require_strict_splits(split_records)
    return split_records


def _run_shared_mlp_protocol(
    *,
    model: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    image_split_counts: Mapping[str, int],
    feature_path: Path,
    split_path: Path,
    result_root: Path,
    probe_cfg: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
    device: str,
    run_name: Optional[str],
    result_stem: str,
    label_protocol: str,
    write_summary: bool,
    positive_class: str,
    sample_audit: Optional[Mapping[str, Any]] = None,
) -> None:
    """Train every dense baseline with the same configured Torch probe."""

    supported = {"metatoken", "svar", "projectaway"}
    unsupported = sorted(set(methods) - supported)
    if unsupported:
        raise ValueError(
            "shared_torch_mlp supports MetaToken, SVAR, and ProjectAway; "
            f"unsupported methods: {unsupported}"
        )
    result_dir = result_root / "results" / "shared_torch_mlp"
    checkpoint_dir = result_root / "checkpoints" / "shared_torch_mlp"
    run_outputs: list[dict[str, Any]] = []
    run_paths: list[Path] = []
    svar_layer_start, svar_layer_end = _svar_training_layer_range(baseline_cfg)
    for seed in seeds:
        seed_name = run_name or f"seed{int(seed)}"
        seed_name = _safe_run_name(seed_name)
        output: dict[str, Any] = {
            "model": model,
            "seed": int(seed),
            "trainer": "shared_torch_mlp",
            "configured_methods": list(methods),
            "feature_path": str(feature_path),
            "split_path": str(split_path),
            "stored_label_semantics": {"0": "hallucination", "1": "real"},
            "detector_target_semantics": {"0": "hallucination", "1": "real"},
            "headline_positive_class": str(positive_class),
            "label_protocol": str(label_protocol),
            "counts": {name: len(rows) for name, rows in split_records.items()},
            "image_split_counts": dict(image_split_counts),
            "methods": {},
            "split_protocol": "strict_82_no_validation",
            "checkpoint_selection": "minimum_train_loss",
            "threshold_selection": "train_f1",
            "threshold_reporting": ["fixed_0.5", "train_f1"],
            "svar_training_layers": {
                "start": svar_layer_start,
                "end_exclusive": svar_layer_end,
            },
        }
        if sample_audit is not None:
            output["sample_audit"] = dict(sample_audit)
        for method in methods:
            matrices = {
                split: _shared_mlp_baseline_matrix(
                    split_records[split],
                    method,
                    svar_layer_start=svar_layer_start,
                    svar_layer_end=svar_layer_end,
                )
                for split in ("train", "test")
            }
            X_train, y_train = matrices["train"]
            X_test, y_test = matrices["test"]
            _require_shared_mlp_binary_matrix(method, "train", X_train, y_train)
            _require_shared_mlp_binary_matrix(method, "test", X_test, y_test)
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(
                    f"{method} train/test widths differ: "
                    f"{X_train.shape[1]} != {X_test.shape[1]}"
                )
            config = _shared_probe_config(
                probe_cfg,
                seed=int(seed),
                positive_class=positive_class,
            )
            artifacts_dir = checkpoint_dir / seed_name / method
            print(
                f"[BaselineSharedMLP] seed={seed} method={method} "
                f"X={X_train.shape[1]} device={device}"
            )
            metrics = train_and_evaluate_probe(
                X_train=X_train,
                y_train=y_train,
                X_val=np.empty((0, X_train.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int64),
                X_test=X_test,
                y_test=y_test,
                config=config,
                device=torch.device(device),
                output_dir=str(artifacts_dir),
            )
            converted = _shared_probe_as_baseline_result(
                metrics,
                config=config,
                input_dim=int(X_train.shape[1]),
                artifacts_dir=artifacts_dir,
            )
            if method == "metatoken":
                output["methods"][method] = {"shared_mlp": converted}
            else:
                output["methods"][method] = converted

        seed_path = result_dir / seed_name / (
            f"{result_stem}_shared_torch_mlp.json"
        )
        save_json(output, str(seed_path))
        run_outputs.append(output)
        run_paths.append(seed_path)
        if write_summary:
            seed_summary = aggregate_baseline_outputs(
                [output], positive_class=positive_class
            )
            _write_baseline_markdown(
                seed_path.with_name(
                    f"{result_stem}_shared_torch_mlp_summary.md"
                ),
                seed_summary,
                source_paths=[seed_path],
            )
        print(f"[BaselineSharedMLP] saved {seed_path}")

    if write_summary:
        summary = aggregate_baseline_outputs(
            run_outputs, positive_class=positive_class
        )
        summary["trainer"] = "shared_torch_mlp"
        summary["seed_result_paths"] = [str(path) for path in run_paths]
        stem = f"{result_stem}_shared_torch_mlp_{len(seeds)}seed"
        json_path = result_dir / f"{stem}.json"
        markdown_path = result_dir / f"{stem}_summary.md"
        save_json(summary, str(json_path))
        _write_baseline_markdown(
            markdown_path,
            summary,
            source_paths=run_paths,
        )
        print(f"[BaselineSharedMLP] saved {json_path}")
        print(f"[BaselineSharedMLP] saved {markdown_path}")


def _shared_mlp_baseline_matrix(
    records: Sequence[Mapping[str, Any]],
    method: str,
    *,
    svar_layer_start: int = 5,
    svar_layer_end: int = 19,
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    normalized = str(method).strip().lower()
    for record in records:
        label = record.get("label")
        if label not in (0, 1):
            raise ValueError(f"Invalid shared-MLP baseline label: {label!r}")
        if normalized in {"metatoken", "svar"}:
            if normalized == "svar":
                vector = svar_training_vector(
                    get_baseline_payload(record, normalized),
                    layer_start=svar_layer_start,
                    layer_end=svar_layer_end,
                )
            else:
                vector = baseline_vector(record, normalized)
        elif normalized == "projectaway":
            payload = get_baseline_payload(record, normalized)
            confidence = payload.get("internal_confidence")
            per_layer = payload.get("per_layer_internal_confidence")
            if confidence is None or per_layer is None:
                raise KeyError(
                    "ProjectAway shared MLP requires internal_confidence and "
                    "per_layer_internal_confidence"
                )
            vector = np.concatenate((
                np.asarray([confidence], dtype=np.float32),
                np.asarray(per_layer, dtype=np.float32).reshape(-1),
            ))
        else:
            raise ValueError(f"Unsupported shared-MLP baseline: {method!r}")
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(
                f"Baseline {normalized!r} has an empty or non-finite MLP vector"
            )
        vectors.append(vector)
        labels.append(int(label))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    widths = {int(vector.size) for vector in vectors}
    if len(widths) != 1:
        raise ValueError(
            f"Inconsistent {normalized} shared-MLP widths: {sorted(widths)}"
        )
    return (
        np.stack(vectors).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
    )


def _require_shared_mlp_binary_matrix(
    method: str,
    split: str,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> None:
    if matrix.shape[0] == 0 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(
            f"{method} shared-MLP {split} split must contain both classes"
        )


def _shared_probe_config(
    probe_cfg: Mapping[str, Any],
    *,
    seed: int,
    positive_class: str,
) -> TorchProbeConfig:
    hidden_sizes = tuple(int(value) for value in probe_cfg.get(
        "hidden_sizes", (128, 64, 32)
    ))
    if len(hidden_sizes) != 3:
        raise ValueError(
            "shared_torch_mlp requires exactly three configured hidden layers"
        )
    return TorchProbeConfig(
        hidden_sizes=hidden_sizes,
        dropout=float(probe_cfg.get("dropout", 0.3)),
        batch_size=int(probe_cfg.get("batch_size", 256)),
        num_epochs=int(probe_cfg.get(
            "max_epochs", probe_cfg.get("num_epochs", 100)
        )),
        learning_rate=float(probe_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(probe_cfg.get("weight_decay", 1e-5)),
        lr_factor=float(probe_cfg.get("lr_factor", 0.5)),
        lr_patience=int(probe_cfg.get("lr_patience", 5)),
        early_stopping_patience=int(probe_cfg.get(
            "early_stopping_patience", 10
        )),
        fixed_threshold=float(probe_cfg.get("fixed_threshold", 0.5)),
        seed=int(seed),
        positive_class=str(positive_class),
    )


def _shared_probe_as_baseline_result(
    metrics: Mapping[str, Any],
    *,
    config: TorchProbeConfig,
    input_dim: int,
    artifacts_dir: Path,
) -> dict[str, Any]:
    test_metrics = {
        key: metrics[key]
        for key in (
            "accuracy",
            "reported_positive_class",
            "real_positive",
            "hallucination_positive",
        )
    }
    return {
        "paper_config": None,
        "adaptation": "shared_three_hidden_layer_torch_mlp",
        "trainer_config": asdict(config),
        "input_dim": int(input_dim),
        "threshold": float(metrics["decision_threshold"]),
        "threshold_score_class": str(config.positive_class),
        "train_metrics": metrics["train_metrics"],
        "test_metrics": test_metrics,
        "threshold_reports": metrics["threshold_reports"],
        "checkpoint": str(artifacts_dir / "model.pt"),
        "history": str(artifacts_dir / "history.json"),
        "best_epoch": int(metrics["best_epoch"]),
        "epochs_completed": int(metrics["epochs_ran"]),
        "best_train_loss": float(metrics["best_train_loss"]),
        "selection_protocol": (
            "minimum_train_loss_dual_fixed_0.5_and_train_real_f1_threshold"
        ),
    }


def _run_training_protocol(
    *,
    model: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    image_split_counts: Mapping[str, int],
    feature_path: Path,
    split_path: Path,
    result_root: Path,
    baseline_cfg: Mapping[str, Any],
    device: str,
    run_name: Optional[str],
    result_stem: str,
    label_protocol: str,
    write_summary: bool,
    positive_class: str,
    sample_audit: Optional[Mapping[str, Any]] = None,
) -> None:
    run_outputs: list[dict[str, Any]] = []
    run_paths: list[Path] = []
    for seed in seeds:
        seed_cfg = dict(baseline_cfg)
        seed_cfg["seed"] = int(seed)
        seed_cfg["_strict_82_no_validation"] = True
        effective_run_name = run_name
        if effective_run_name is None and len(seeds) > 1:
            effective_run_name = f"seed{seed}"
        output, result_path = _train_one_seed(
            model=model,
            seed=int(seed),
            methods=methods,
            split_records=split_records,
            image_split_counts=image_split_counts,
            feature_path=feature_path,
            split_path=split_path,
            baseline_dir=result_root,
            baseline_cfg=seed_cfg,
            device=device,
            run_name=effective_run_name,
            result_stem=result_stem,
            label_protocol=label_protocol,
            sample_audit=sample_audit,
            positive_class=positive_class,
        )
        run_outputs.append(output)
        run_paths.append(result_path)

    if write_summary:
        _write_training_summaries(
            model=model,
            baseline_dir=result_root,
            outputs=run_outputs,
            result_paths=run_paths,
            result_stem=result_stem,
            positive_class=positive_class,
        )


def _train_one_seed(
    *,
    model: str,
    seed: int,
    methods: Sequence[str],
    split_records: Mapping[str, Sequence[Mapping[str, Any]]],
    image_split_counts: Mapping[str, int],
    feature_path: Path,
    split_path: Path,
    baseline_dir: Path,
    baseline_cfg: Mapping[str, Any],
    device: str,
    run_name: Optional[str],
    result_stem: Optional[str] = None,
    label_protocol: str = (
        "shared_first_canonical_mention_exact_response_offsets"
    ),
    sample_audit: Optional[Mapping[str, Any]] = None,
    positive_class: str = "real",
) -> tuple[dict[str, Any], Path]:
    result_dir = baseline_dir / "results"
    checkpoint_dir = baseline_dir / "checkpoints"
    if run_name is not None:
        run_name = _safe_run_name(run_name)
        result_dir = result_dir / run_name
        checkpoint_dir = checkpoint_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_stem = str(result_stem or f"{model}_baselines")
    result_path = result_dir / f"{result_stem}.json"
    output: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "configured_methods": list(methods),
        "feature_path": str(feature_path),
        "split_path": str(split_path),
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "headline_positive_class": str(positive_class),
        "label_protocol": str(label_protocol),
        "counts": {name: len(rows) for name, rows in split_records.items()},
        "image_split_counts": image_split_counts,
        "methods": {},
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "last_epoch",
        "threshold_selection": "train_f1",
    }
    if sample_audit is not None:
        output["sample_audit"] = dict(sample_audit)

    training_records = dict(split_records)
    # The baseline implementations retain a ``val_metrics`` compatibility
    # field. Under pure 8:2 it reports train diagnostics only; it is never used
    # for checkpoint or threshold selection.
    training_records["val"] = list(split_records["train"])

    for method in methods:
        print(f"[BaselineTrain] seed={seed} method={method} device={device}")
        if method == "metatoken":
            result = _train_metatoken(
                training_records,
                checkpoint_dir,
                baseline_cfg,
                positive_class,
            )
        elif method == "svar":
            result = _train_svar(
                training_records,
                checkpoint_dir,
                baseline_cfg,
                device,
                positive_class,
            )
        elif method == "dhcp":
            result = _train_dhcp(
                training_records,
                baseline_dir,
                checkpoint_dir,
                baseline_cfg,
                device,
                positive_class,
            )
        elif method == "projectaway":
            result = _evaluate_projectaway(
                training_records,
                positive_class,
                strict_82_no_validation=True,
            )
        else:
            result = _train_halloc(
                training_records,
                baseline_dir,
                checkpoint_dir,
                baseline_cfg,
                device,
                positive_class,
            )
        _rename_train_diagnostics(result)
        output["methods"][method] = result
        save_json(output, str(result_path))

    print(f"[BaselineTrain] saved {result_path}")
    return output, result_path


def _rename_train_diagnostics(result: dict[str, Any]) -> None:
    """Expose compatibility ``val`` evaluations honestly as train diagnostics."""

    variants = result.values() if all(
        isinstance(value, Mapping) for value in result.values()
    ) and set(result).issubset({"lr", "gb"}) else (result,)
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        if "val_metrics" in variant:
            variant["train_metrics"] = variant.pop("val_metrics")


def _baseline_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training") or {}
    if not isinstance(training, Mapping):
        raise ValueError("training must be a YAML mapping")
    baseline = training.get("baseline") or {}
    if not isinstance(baseline, Mapping):
        raise ValueError("training.baseline must be a YAML mapping")
    allowed = {
        "methods",
        "seeds",
        "write_summary",
        "positive_class",
        "trainer",
    }
    unknown = sorted(set(baseline) - allowed)
    if unknown:
        raise ValueError(f"Unknown training.baseline options: {unknown}")
    return dict(baseline)


def _normalize_baseline_trainer(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "native": "native_paper",
        "paper": "native_paper",
        "torch_mlp": "shared_torch_mlp",
        "shared_mlp": "shared_torch_mlp",
        "three_layer_mlp": "shared_torch_mlp",
        "3layer_mlp": "shared_torch_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"native_paper", "shared_torch_mlp"}:
        raise ValueError(
            "training.baseline.trainer must be native_paper or "
            f"shared_torch_mlp, got {value!r}"
        )
    return normalized


def _configured_training_seeds(
    args: argparse.Namespace,
    training_cfg: Mapping[str, Any],
    baseline_cfg: Mapping[str, Any],
) -> list[int]:
    if args.seed is not None:
        values = [args.seed]
    elif args.seeds is not None:
        values = args.seeds
    else:
        values = training_cfg.get(
            "seeds", [int(baseline_cfg.get("seed", 42))]
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("training.baseline.seeds must be a non-empty list")
    seeds = list(dict.fromkeys(int(value) for value in values))
    if not seeds:
        raise ValueError("At least one baseline training seed is required")
    return seeds


def _safe_run_name(value: str) -> str:
    run_name = str(value).strip()
    if (
        not run_name
        or Path(run_name).name != run_name
        or run_name in {".", ".."}
    ):
        raise ValueError("--run-name must be one safe path component")
    return run_name


def _normalize_reporting_positive_class(value: object) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"real", "non_hallucination", "nonhallucination"}:
        return "real"
    if normalized in {"hall", "hallucinated", "hallucination"}:
        return "hallucination"
    raise ValueError(
        "training.baseline.positive_class must be 'real' or 'hallucination', "
        f"got {value!r}"
    )


def _write_training_summaries(
    *,
    model: str,
    baseline_dir: Path,
    outputs: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
    result_stem: Optional[str] = None,
    positive_class: str = "real",
) -> None:
    if len(outputs) != len(result_paths):
        raise ValueError("Baseline outputs and result paths must have equal length")
    base_stem = str(result_stem or f"{model}_baselines")
    for output, result_path in zip(outputs, result_paths):
        summary = aggregate_baseline_outputs(
            [output],
            positive_class=positive_class,
        )
        markdown_path = result_path.with_name(f"{base_stem}_summary.md")
        _write_baseline_markdown(
            markdown_path,
            summary,
            source_paths=[result_path],
        )
        print(f"[BaselineTrain] saved {markdown_path}")

    if len(outputs) > 1:
        summary = aggregate_baseline_outputs(
            outputs,
            positive_class=positive_class,
        )
        result_dir = baseline_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{base_stem}_{len(outputs)}seed"
        json_path = result_dir / f"{stem}.json"
        markdown_path = result_dir / f"{stem}_summary.md"
        summary["seed_result_paths"] = [str(path) for path in result_paths]
        save_json(summary, str(json_path))
        _write_baseline_markdown(
            markdown_path,
            summary,
            source_paths=result_paths,
        )
        print(f"[BaselineTrain] saved {json_path}")
        print(f"[BaselineTrain] saved {markdown_path}")


def aggregate_baseline_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    positive_class: str = "real",
) -> dict[str, Any]:
    if not outputs:
        raise ValueError("At least one baseline output is required")
    model = str(outputs[0]["model"])
    positive_class = _normalize_reporting_positive_class(positive_class)
    seeds = [int(output["seed"]) for output in outputs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Baseline seeds must be unique, got {seeds}")
    first_variants = _baseline_method_variants(outputs[0])
    expected = list(first_variants)
    for output in outputs:
        if str(output["model"]) != model:
            raise ValueError("Cannot aggregate baseline outputs from different models")
        if output.get("sample_audit") != outputs[0].get("sample_audit"):
            raise ValueError(
                "All seeds must use the same baseline sample audit"
            )
        actual = list(_baseline_method_variants(output))
        if actual != expected:
            raise ValueError(
                "All seeds must contain the same ordered baseline methods: "
                f"expected {expected}, got {actual}"
            )

    rows: dict[str, Any] = {}
    for key in expected:
        display_name = first_variants[key][0]
        row: dict[str, Any] = {"display_name": display_name}
        for split in ("train", "test"):
            seed_metrics = [
                _headline_metrics(
                    _baseline_method_variants(output)[key][1],
                    split,
                    positive_class=positive_class,
                )
                for output in outputs
            ]
            row[f"{split}_metrics"] = {
                metric: _metric_statistics(
                    [float(values[metric]) for values in seed_metrics]
                )
                for metric in SUMMARY_METRICS
            }
        first_result = first_variants[key][1]
        first_reports = first_result.get("threshold_reports") or {}
        if first_reports:
            reporting = tuple(str(mode) for mode in first_reports)
            for output in outputs:
                result = _baseline_method_variants(output)[key][1]
                if tuple(str(mode) for mode in (result.get("threshold_reports") or {})) != reporting:
                    raise ValueError(
                        f"All seeds must share threshold reports for {key}"
                    )
            row["threshold_reports"] = {}
            for mode in reporting:
                mode_row = {
                    "threshold": _metric_statistics([
                        float(
                            _baseline_method_variants(output)[key][1]
                            ["threshold_reports"][mode]["threshold"]
                        )
                        for output in outputs
                    ])
                }
                for split in ("train", "test"):
                    seed_metrics = [
                        _headline_metrics(
                            _baseline_method_variants(output)[key][1],
                            split,
                            positive_class=positive_class,
                            threshold_mode=mode,
                        )
                        for output in outputs
                    ]
                    mode_row[f"{split}_metrics"] = {
                        metric: _metric_statistics([
                            float(values[metric]) for values in seed_metrics
                        ])
                        for metric in SUMMARY_METRICS
                    }
                row["threshold_reports"][mode] = mode_row
        rows[key] = row

    return {
        "model": model,
        "trainer": outputs[0].get("trainer", "native_paper"),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "std_definition": "population",
        "headline_positive_class": positive_class,
        "source_headline_positive_classes": sorted(
            {
                str(output.get("headline_positive_class") or "unknown")
                for output in outputs
            }
        ),
        "stored_label_semantics": outputs[0].get("stored_label_semantics"),
        "detector_target_semantics": outputs[0].get("detector_target_semantics"),
        "label_protocol": outputs[0].get("label_protocol"),
        "counts": outputs[0].get("counts"),
        "image_split_counts": outputs[0].get("image_split_counts"),
        "configured_methods": outputs[0].get("configured_methods"),
        "sample_audit": outputs[0].get("sample_audit"),
        "split_protocol": outputs[0].get("split_protocol"),
        "checkpoint_selection": outputs[0].get("checkpoint_selection"),
        "threshold_selection": outputs[0].get("threshold_selection"),
        "threshold_reporting": outputs[0].get("threshold_reporting"),
        "methods": rows,
    }


def _baseline_method_variants(
    output: Mapping[str, Any],
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    variants: dict[str, tuple[str, Mapping[str, Any]]] = {}
    methods = output.get("methods") or {}
    if not isinstance(methods, Mapping):
        raise ValueError("Baseline output methods must be a mapping")
    for method, result in methods.items():
        normalized = str(method).lower()
        if normalized == "metatoken":
            if not isinstance(result, Mapping):
                raise ValueError("MetaToken result must be a mapping")
            for classifier, classifier_result in result.items():
                key = f"metatoken_{str(classifier).lower()}"
                classifier_label = {
                    "shared_mlp": "MLP",
                }.get(str(classifier).lower(), str(classifier).upper())
                variants[key] = (
                    f"MetaToken-{classifier_label}",
                    classifier_result,
                )
        else:
            labels = {
                "svar": "SVAR",
                "dhcp": "DHCP",
                "projectaway": "ProjectAway",
                "halloc": "HalLoc",
            }
            variants[normalized] = (labels.get(normalized, str(method)), result)
    return variants


def _headline_metrics(
    result: Mapping[str, Any],
    split: str,
    *,
    positive_class: str,
    threshold_mode: str | None = None,
) -> dict[str, float]:
    if threshold_mode is None:
        metrics = result.get(f"{split}_metrics") or {}
    else:
        reports = result.get("threshold_reports") or {}
        report = reports.get(threshold_mode) or {}
        metrics = report.get(f"{split}_metrics") or {}
    hallucination = metrics.get("hallucination_positive") or {}
    real = metrics.get("real_positive") or {}
    positive = real if positive_class == "real" else hallucination
    other = hallucination if positive_class == "real" else real
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(positive["precision"]),
        "recall": float(positive["recall"]),
        "f1": float(positive["f1"]),
        "auc": float(positive["auc"]),
        "aupr": float(positive["aupr"]),
        "other_f1": float(other["f1"]),
    }


def _metric_statistics(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(value) for value in array],
    }


def _write_baseline_markdown(
    path: Path,
    summary: Mapping[str, Any],
    *,
    source_paths: Sequence[Path],
) -> None:
    seeds = [int(seed) for seed in summary["seeds"]]
    multi_seed = len(seeds) > 1
    counts = summary.get("counts") or {}
    image_counts = summary.get("image_split_counts") or {}
    methods = summary.get("methods") or {}
    dual_threshold = bool(methods) and all(
        isinstance(row.get("threshold_reports"), Mapping)
        and {"fixed_0.5", "train_f1"}.issubset(row["threshold_reports"])
        for row in methods.values()
    )
    label_protocol = str(summary.get("label_protocol") or "")
    positive_class = _normalize_reporting_positive_class(
        summary.get("headline_positive_class", "real")
    )
    positive_label = "Real" if positive_class == "real" else "Hall."
    other_label = "Hall." if positive_class == "real" else "Real"
    experiment_name = (
        "SVAR Official"
        if "official_svar" in label_protocol
        else (
            "Baseline（三层共享 MLP）"
            if summary.get("trainer") == "shared_torch_mlp"
            else "Baseline"
        )
    )
    lines = [
        f"# {summary['model']} {experiment_name} 结果汇总",
        "",
        "## 实验协议",
        "",
        f"- 随机种子：`{', '.join(str(seed) for seed in seeds)}`。",
        f"- headline 正类：{positive_class}。",
        f"- 标签协议：`{label_protocol or 'unknown'}`。",
        (
            "- 严格 8:2 无验证集：train-loss early stopping，恢复 minimum-train-loss checkpoint；同一权重同时报告固定 0.5 与 train 正类 F1 搜索阈值。"
            if dual_threshold
            else "- 严格 8:2 无验证集：按该 baseline 的既定 checkpoint 与阈值协议训练。"
        ),
        "- test set 只用于最终评估，不用于调参、早停或选阈值。",
        f"- image split：train/val/test = "
        f"{image_counts.get('train', '?')}/{image_counts.get('val', '?')}/"
        f"{image_counts.get('test', '?')}。",
        f"- token 样本：train/val/test = "
        f"{counts.get('train', '?')}/{counts.get('val', '?')}/"
        f"{counts.get('test', '?')}。",
    ]
    sample_audit = summary.get("sample_audit") or {}
    overall_audit = sample_audit.get("overall") or {}
    if overall_audit:
        lines.extend(
            [
                "- SVAR official 查询：total={}，found={}，not_found={}。".format(
                    overall_audit.get("total", "?"),
                    overall_audit.get("found", "?"),
                    overall_audit.get("not_found", "?"),
                ),
                "- found 标签构成：hallucination={}，real={}。".format(
                    overall_audit.get("hallucination_found", "?"),
                    overall_audit.get("real_found", "?"),
                ),
            ]
        )
        split_audit = sample_audit.get("by_split") or {}
        lines.append(
            "- found split："
            + "，".join(
                "{}={}".format(
                    split,
                    (split_audit.get(split) or {}).get("found", "?"),
                )
                for split in ("train", "val", "test")
            )
            + "。"
        )
    if multi_seed:
        lines.append(
            f"- 表中数值为 {len(seeds)} 个随机种子的总体均值 ± 总体标准差。"
        )
    else:
        lines.append("- 表中数值来自单次随机种子运行，不是多 seed 平均。")
    lines.extend(
        [
            "- 原始结果："
            + "、".join(f"`{source}`" for source in source_paths)
            + "。",
            "",
            (
                "## Test 结果（Train Real-F1 搜索阈值）"
                if dual_threshold else "## Test 结果"
            ),
            "",
            f"| 方法 | Accuracy | {positive_label} Precision | "
            f"{positive_label} Recall | {positive_label} F1 | AUROC | AUPR | "
            f"{other_label} F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in methods.values():
        metrics = row["test_metrics"]
        lines.append(
            f"| {row['display_name']} | "
            + " | ".join(
                _format_summary_value(metrics[metric], multi_seed)
                for metric in SUMMARY_METRICS
            )
            + " |"
        )
    if dual_threshold:
        lines.extend(
            [
                "",
                "## Test 结果（固定阈值 0.5）",
                "",
                f"| 方法 | Accuracy | {positive_label} Precision | "
                f"{positive_label} Recall | {positive_label} F1 | AUROC | AUPR | "
                f"{other_label} F1 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in methods.values():
            metrics = row["threshold_reports"]["fixed_0.5"]["test_metrics"]
            lines.append(
                f"| {row['display_name']} | "
                + " | ".join(
                    _format_summary_value(metrics[metric], multi_seed)
                    for metric in SUMMARY_METRICS
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Train 结果（train-F1 模式在此选择阈值）",
            "",
            f"| 方法 | Accuracy | {positive_label} Precision | "
            f"{positive_label} Recall | {positive_label} F1 | AUROC | AUPR | "
            f"{other_label} F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in methods.values():
        metrics = row["train_metrics"]
        lines.append(
            f"| {row['display_name']} | "
            + " | ".join(
                _format_summary_value(metrics[metric], multi_seed)
                for metric in SUMMARY_METRICS
            )
            + " |"
        )
    if multi_seed:
        lines.extend(
            [
                "",
                f"## 各随机种子的 Test {positive_label} F1",
                "",
                "| 方法 | " + " | ".join(f"seed {seed}" for seed in seeds) + " |",
                "|---|" + "---:|" * len(seeds),
            ]
        )
        for row in methods.values():
            values = row["test_metrics"]["f1"]["values"]
            lines.append(
                f"| {row['display_name']} | "
                + " | ".join(f"{float(value):.4f}" for value in values)
                + " |"
            )
    if methods:
        best = max(
            methods.values(),
            key=lambda row: float(row["test_metrics"]["f1"]["mean"]),
        )
        lines.extend(
            [
                "",
                "## 简要结论",
                "",
                f"- Test {positive_label} F1 最高的方法是 **{best['display_name']}**："
                f"{_format_summary_value(best['test_metrics']['f1'], multi_seed)}。",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary_value(statistics: Mapping[str, Any], multi_seed: bool) -> str:
    mean = float(statistics["mean"])
    if not multi_seed:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {float(statistics['std']):.4f}"


def _train_metatoken(
    split_records,
    checkpoint_dir,
    cfg,
    positive_class,
) -> dict[str, Any]:
    matrices = {}
    for split in ("train", "val", "test"):
        matrices[split] = build_dense_baseline_matrix(
            split_records[split], "metatoken"
        )[:2]
    result = {}
    metatoken_cfg = dict(cfg.get("metatoken") or {})
    configured_classifiers = metatoken_cfg.get("classifiers", ("lr", "gb"))
    if isinstance(configured_classifiers, str):
        configured_classifiers = [configured_classifiers]
    classifiers = [
        str(value).strip().lower()
        for value in configured_classifiers
    ]
    unknown = set(classifiers) - {"lr", "gb"}
    if unknown or not classifiers:
        raise ValueError(
            "MetaToken classifiers must be a non-empty subset of ['lr', 'gb']; "
            f"got {classifiers}"
        )
    for kind in dict.fromkeys(classifiers):
        classifier = build_metatoken_classifier(kind, seed=int(cfg.get("seed", 42)))
        classifier.fit(
            matrices["train"][0],
            raw_labels_to_hallucination_targets(matrices["train"][1]),
        )
        val_scores = sklearn_hallucination_scores(classifier, matrices["val"][0])
        test_scores = sklearn_hallucination_scores(classifier, matrices["test"][0])
        strict_82 = bool(cfg.get("_strict_82_no_validation", False))
        threshold = select_detection_threshold(
            matrices["val"][1],
            val_scores,
            positive_class=positive_class,
        )
        path = checkpoint_dir / f"metatoken_{kind}.pkl"
        save_pkl(classifier, str(path))
        result[kind] = {
            "paper_config": {
                "classifier": "LogisticRegression(lbfgs)"
                if kind == "lr"
                else "GradientBoostingClassifier(n_estimators=100)",
                "standardize": True,
            },
            "input_dim": int(matrices["train"][0].shape[1]),
            "threshold": threshold,
            "threshold_score_class": str(positive_class),
            "val_metrics": evaluate_detection_scores(
                matrices["val"][1],
                val_scores,
                threshold,
                positive_class=positive_class,
            ),
            "test_metrics": evaluate_detection_scores(
                matrices["test"][1],
                test_scores,
                threshold,
                positive_class=positive_class,
            ),
            "checkpoint": str(path),
            "selection_protocol": (
                "fixed_last_fit_train_f1_threshold"
                if strict_82
                else "validation_selected"
            ),
        }
    return result


def _train_svar(
    split_records,
    checkpoint_dir,
    cfg,
    device,
    positive_class,
) -> dict[str, Any]:
    svar_cfg = dict(cfg.get("svar") or {})
    layer_start, layer_end = _svar_training_layer_range(cfg)
    matrices = {
        split: build_dense_baseline_matrix(
            split_records[split],
            "svar",
            svar_layer_start=layer_start,
            svar_layer_end=layer_end,
        )[:2]
        for split in ("train", "val", "test")
    }
    seed = int(cfg.get("seed", 42))
    strict_82 = bool(cfg.get("_strict_82_no_validation", False))
    # Seed before module construction so the requested run seed controls both
    # SVAR's initial weights and the subsequent minibatch order.
    _seed_everything(seed)
    model = SVARMLP(
        matrices["train"][0].shape[1],
        hidden_dim=int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
    )
    trained = train_torch_detector(
        model=model,
        X_train=matrices["train"][0],
        raw_y_train=matrices["train"][1],
        X_val=matrices["val"][0],
        raw_y_val=matrices["val"][1],
        X_test=matrices["test"][0],
        raw_y_test=matrices["test"][1],
        epochs=int(_setting(svar_cfg, "epochs", "max_epochs", default=50)),
        learning_rate=float(svar_cfg.get("learning_rate", 1e-3)),
        batch_size=int(svar_cfg.get("batch_size", 32)),
        device=device,
        weighted_sampler=bool(svar_cfg.get("weighted_sampler", False)),
        standardize=bool(svar_cfg.get("standardize", False)),
        early_stopping_patience=int(
            svar_cfg.get("early_stopping_patience", 5)
        ),
        seed=seed,
        positive_class=positive_class,
        strict_82_no_validation=bool(
            cfg.get("_strict_82_no_validation", False)
        ),
    )
    path = checkpoint_dir / "svar.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": trained.state_dict,
            "input_dim": int(matrices["train"][0].shape[1]),
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "threshold": trained.threshold,
            "threshold_score_class": str(positive_class),
            "layer_start": layer_start,
            "layer_end_exclusive": layer_end,
        },
    )
    return {
        "paper_config": {
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "learning_rate": float(svar_cfg.get("learning_rate", 1e-3)),
            "epochs": int(_setting(svar_cfg, "epochs", "max_epochs", default=50)),
            "early_stopping_patience": (
                None
                if strict_82
                else int(svar_cfg.get("early_stopping_patience", 5))
            ),
            "positive_class": str(positive_class),
            "selection_protocol": (
                "fixed_last_epoch_train_f1_threshold"
                if strict_82
                else "validation_selected"
            ),
            "layer_start": layer_start,
            "layer_end_exclusive": layer_end,
        },
        "input_dim": int(matrices["train"][0].shape[1]),
        "threshold": trained.threshold,
        "threshold_score_class": str(positive_class),
        "val_metrics": trained.val_metrics,
        "test_metrics": trained.test_metrics,
        "history": trained.history,
        "checkpoint": str(path),
    }


def _svar_training_layer_range(
    baseline_cfg: Mapping[str, Any],
) -> tuple[int, int]:
    svar_cfg = dict(baseline_cfg.get("svar") or {})
    start = int(svar_cfg.get("layer_start", 5))
    end = int(svar_cfg.get("layer_end", 19))
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid SVAR training layer range [{start},{end}); expected "
            "0 <= layer_start < layer_end"
        )
    return start, end


def _train_dhcp(
    split_records,
    baseline_dir,
    checkpoint_dir,
    cfg,
    device,
    positive_class,
):
    datasets = {
        split: DHCPRecordDataset(
            split_records[split], baseline_dir / "dhcp" / "shards"
        )
        for split in ("train", "val", "test")
    }
    shape = datasets["train"].item_shape
    for split in ("val", "test"):
        if datasets[split].item_shape != shape:
            raise ValueError("DHCP tensor shape differs across train/val/test")
    input_dim = int(np.prod(shape))
    dhcp_cfg = dict(cfg.get("dhcp") or {})
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)
    dhcp_hidden = int(_setting(dhcp_cfg, "hidden_dim", "hidden_size", default=128))
    model = DHCPMLP(input_dim, hidden_dim=dhcp_hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(dhcp_cfg.get("learning_rate", 1e-3))
    )
    criterion = nn.CrossEntropyLoss()
    train_targets = np.asarray(datasets["train"].targets, dtype=np.int64)
    counts = np.bincount(train_targets, minlength=2)
    weights = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights[train_targets], dtype=torch.double),
        num_samples=len(train_targets),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = int(dhcp_cfg.get("batch_size", 1024))
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, num_workers=0)
    strict_82 = bool(cfg.get("_strict_82_no_validation", False))
    best_loss = math.inf
    best_state = None
    history = []
    dhcp_epochs = int(_setting(dhcp_cfg, "epochs", "max_epochs", default=30))
    patience = int(dhcp_cfg.get("early_stopping_patience", 5))
    stale_epochs = 0
    for epoch in range(dhcp_epochs):
        model.train()
        losses = []
        for values, targets in train_loader:
            values, targets = values.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = _stream_scores(model, val_loader, device, criterion)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
        }
        epoch_row["train_monitor_loss" if strict_82 else "val_loss"] = val_loss
        history.append(epoch_row)
        if strict_82:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, patience):
                break
    if best_state is None:
        raise RuntimeError("No DHCP checkpoint selected")
    model.load_state_dict(best_state)
    _, val_scores = _stream_scores(model, val_loader, device, criterion)
    _, test_scores = _stream_scores(
        model,
        DataLoader(datasets["test"], batch_size=batch_size, num_workers=0),
        device,
        criterion,
    )
    val_raw = np.asarray([record["label"] for record in datasets["val"].records])
    test_raw = np.asarray([record["label"] for record in datasets["test"].records])
    threshold = select_detection_threshold(
        val_raw,
        val_scores,
        positive_class=positive_class,
    )
    path = checkpoint_dir / "dhcp.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "input_shape": shape,
            "hidden_dim": dhcp_hidden,
            "threshold": threshold,
            "threshold_score_class": str(positive_class),
        },
    )
    return {
        "paper_config": {
            "protocol": "object_prediction_fixed_grid_mass_preserving",
            "target_grid": [12, 12],
            "hidden_dim": dhcp_hidden,
            "learning_rate": float(dhcp_cfg.get("learning_rate", 1e-3)),
            "batch_size": batch_size,
            "epochs": dhcp_epochs,
            "early_stopping_patience": None if strict_82 else patience,
            "weighted_sampler": True,
            "positive_class": str(positive_class),
        },
        "input_shape": list(shape),
        "input_dim": input_dim,
        "threshold": threshold,
        "threshold_score_class": str(positive_class),
        "val_metrics": evaluate_detection_scores(
            val_raw,
            val_scores,
            threshold,
            positive_class=positive_class,
        ),
        "test_metrics": evaluate_detection_scores(
            test_raw,
            test_scores,
            threshold,
            positive_class=positive_class,
        ),
        "history": history,
        "checkpoint": str(path),
        "selection_protocol": (
            "fixed_last_epoch_train_f1_threshold"
            if strict_82
            else "validation_selected"
        ),
    }


def _evaluate_projectaway(
    split_records,
    positive_class,
    *,
    strict_82_no_validation: bool = False,
) -> dict[str, Any]:
    scores, labels = {}, {}
    for split in ("val", "test"):
        payloads = [
            get_baseline_payload(record, "projectaway")
            for record in split_records[split]
        ]
        scores[split] = np.asarray(
            [float(payload["hallucination_score"]) for payload in payloads]
        )
        labels[split] = np.asarray(
            [int(record["label"]) for record in split_records[split]]
        )
    threshold = select_detection_threshold(
        labels["val"],
        scores["val"],
        positive_class=positive_class,
    )
    return {
        "paper_config": {
            "training_free": True,
            "threshold_selected_on": (
                "train_f1" if strict_82_no_validation else "val"
            ),
            "positive_class": str(positive_class),
        },
        "threshold": threshold,
        "threshold_score_class": str(positive_class),
        "val_metrics": evaluate_detection_scores(
            labels["val"],
            scores["val"],
            threshold,
            positive_class=positive_class,
        ),
        "test_metrics": evaluate_detection_scores(
            labels["test"],
            scores["test"],
            threshold,
            positive_class=positive_class,
        ),
    }


def _train_halloc(
    split_records,
    baseline_dir,
    checkpoint_dir,
    cfg,
    device,
    positive_class,
):
    seed = int(cfg.get("seed", 42))
    strict_82 = bool(cfg.get("_strict_82_no_validation", False))
    _seed_everything(seed)
    datasets = {
        split: HalLocCachedDataset(split_records[split], baseline_dir)
        for split in ("train", "val", "test")
    }
    sample = datasets["train"][0]
    halloc_cfg = dict(cfg.get("halloc") or {})
    if not bool(halloc_cfg.get("freeze_clip", True)):
        raise ValueError(
            "The cache-based HalLoc pipeline requires freeze_clip=true. "
            "Unfreezing CLIP would require storing images and recomputing it in training."
        )
    paper = halloc_optimizer_config()
    model = HalLocObjectDetector(
        lvlm_hidden_size=int(sample[0].shape[-1]),
        clip_model_name=str(halloc_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
        visualbert_model_name=str(
            halloc_cfg.get("visualbert_model", "uclanlp/visualbert-vqa-coco-pre")
        ),
        freeze_clip=True,
        load_clip=False,
        clip_hidden_size=int(sample[1].shape[-1]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(halloc_cfg.get("learning_rate", paper["learning_rate"])),
        betas=tuple(halloc_cfg.get("betas", paper["betas"])),
        weight_decay=float(halloc_cfg.get("weight_decay", paper["weight_decay"])),
    )
    criterion = nn.CrossEntropyLoss()
    batch_size = int(halloc_cfg.get("batch_size", paper["batch_size"]))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            collate_fn=_collate_halloc,
            num_workers=0,
            generator=(
                torch.Generator().manual_seed(seed) if split == "train" else None
            ),
        )
        for split, dataset in datasets.items()
    }
    best_loss, best_state, history = math.inf, None, []
    halloc_epochs = int(_setting(halloc_cfg, "epochs", "max_epochs", default=25))
    scheduler_name = str(halloc_cfg.get("scheduler", "cosine")).strip().lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, halloc_epochs),
        )
    elif scheduler_name in {"none", "off", "disabled"}:
        scheduler = None
    else:
        raise ValueError("HalLoc scheduler must be 'cosine' or 'none'")
    patience = int(halloc_cfg.get("early_stopping_patience", 3))
    stale_epochs = 0
    for epoch in range(halloc_epochs):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                lvlm_embeddings=batch["lvlm_embeddings"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                clip_visual_features=batch["clip_visual_features"].to(device),
                visual_attention_mask=batch["visual_attention_mask"].to(device),
                object_indices=batch["object_indices"].to(device),
            )
            loss = criterion(logits, batch["targets"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_loss, _ = _halloc_scores(model, loaders["val"], device, criterion)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "learning_rate": learning_rate,
        }
        epoch_row["train_monitor_loss" if strict_82 else "val_loss"] = val_loss
        history.append(epoch_row)
        if scheduler is not None:
            scheduler.step()
        if strict_82:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("No HalLoc checkpoint selected")
    model.load_state_dict(best_state)
    _, val_scores = _halloc_scores(model, loaders["val"], device, criterion)
    _, test_scores = _halloc_scores(model, loaders["test"], device, criterion)
    val_raw = np.asarray([record["label"] for record in datasets["val"].records])
    test_raw = np.asarray([record["label"] for record in datasets["test"].records])
    threshold = select_detection_threshold(
        val_raw,
        val_scores,
        positive_class=positive_class,
    )
    path = checkpoint_dir / "halloc.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "threshold": threshold,
            "threshold_score_class": str(positive_class),
            "paper_metadata": model.paper_metadata(),
            "lvlm_hidden_size": int(sample[0].shape[-1]),
            "clip_hidden_size": int(sample[1].shape[-1]),
            "clip_model": str(
                halloc_cfg.get("clip_model", "openai/clip-vit-base-patch32")
            ),
            "visualbert_model": str(
                halloc_cfg.get(
                    "visualbert_model", "uclanlp/visualbert-vqa-coco-pre"
                )
            ),
        },
    )
    return {
        "paper_config": {
            **paper,
            "max_epochs": halloc_epochs,
            "scheduler": scheduler_name,
            "early_stopping_patience": None if strict_82 else patience,
            "positive_class": str(positive_class),
        },
        "threshold": threshold,
        "threshold_score_class": str(positive_class),
        "val_metrics": evaluate_detection_scores(
            val_raw,
            val_scores,
            threshold,
            positive_class=positive_class,
        ),
        "test_metrics": evaluate_detection_scores(
            test_raw,
            test_scores,
            threshold,
            positive_class=positive_class,
        ),
        "history": history,
        "checkpoint": str(path),
        "selection_protocol": (
            "fixed_last_epoch_train_f1_threshold"
            if strict_82
            else "validation_selected"
        ),
    }


class DHCPRecordDataset(Dataset):
    def __init__(self, records, shard_root: Path):
        self.records = [
            record for record in records if "dhcp" in record.get("baselines", {})
        ]
        if not self.records:
            raise ValueError("No DHCP records found in split")
        self.reader = DHCPShardReader(shard_root)
        first_ref = _dhcp_reference(get_baseline_payload(self.records[0], "dhcp"))
        self.item_shape = tuple(int(value) for value in first_ref["shape"])
        self.targets = raw_labels_to_hallucination_targets(
            [record["label"] for record in self.records]
        ).tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        payload = get_baseline_payload(self.records[index], "dhcp")
        value = self.reader.load(_dhcp_reference(payload), as_tensor=True).float()
        return value, int(self.targets[index])


class HalLocCachedDataset(Dataset):
    def __init__(self, records, root: Path):
        self.records = [
            record for record in records if "halloc" in record.get("baselines", {})
        ]
        if not self.records:
            raise ValueError("No HalLoc records found in split")
        self.root = root.resolve()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        payload = get_baseline_payload(record, "halloc")
        if "cache_file" in payload:
            path = (self.root / str(payload["cache_file"])).resolve()
            if self.root != path and self.root not in path.parents:
                raise ValueError("HalLoc cache path escapes baseline directory")
            with np.load(path, allow_pickle=False) as cached:
                text = np.asarray(cached["lvlm_embeddings"], dtype=np.float32)
                visual = np.asarray(cached["clip_visual_features"], dtype=np.float32)
        else:
            text = np.asarray(payload["lvlm_embeddings"], dtype=np.float32)
            visual = np.asarray(payload["clip_visual_features"], dtype=np.float32)
        object_index = int(payload.get("object_index", record["response_token_idx"]))
        target = int(raw_labels_to_hallucination_targets([record["label"]])[0])
        return torch.from_numpy(text), torch.from_numpy(visual), object_index, target


def _collate_halloc(samples):
    max_text = max(item[0].shape[0] for item in samples)
    max_visual = max(item[1].shape[0] for item in samples)
    text_dim, visual_dim = samples[0][0].shape[1], samples[0][1].shape[1]
    text = torch.zeros(len(samples), max_text, text_dim)
    visual = torch.zeros(len(samples), max_visual, visual_dim)
    text_mask = torch.zeros(len(samples), max_text, dtype=torch.long)
    visual_mask = torch.zeros(len(samples), max_visual, dtype=torch.long)
    indices, targets = [], []
    for row, (text_value, visual_value, index, target) in enumerate(samples):
        text[row, : len(text_value)] = text_value
        visual[row, : len(visual_value)] = visual_value
        text_mask[row, : len(text_value)] = 1
        visual_mask[row, : len(visual_value)] = 1
        indices.append(index)
        targets.append(target)
    return {
        "lvlm_embeddings": text,
        "clip_visual_features": visual,
        "attention_mask": text_mask,
        "visual_attention_mask": visual_mask,
        "object_indices": torch.tensor(indices, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def _stream_scores(model, loader, device, criterion):
    model.eval()
    losses, scores = [], []
    with torch.no_grad():
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            logits = model(values)
            losses.append(float(criterion(logits, targets).item()))
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return float(np.mean(losses)), torch.cat(scores).numpy()


def _halloc_scores(model, loader, device, criterion):
    model.eval()
    losses, scores = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                lvlm_embeddings=batch["lvlm_embeddings"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                clip_visual_features=batch["clip_visual_features"].to(device),
                visual_attention_mask=batch["visual_attention_mask"].to(device),
                object_indices=batch["object_indices"].to(device),
            )
            targets = batch["targets"].to(device)
            losses.append(float(criterion(logits, targets).item()))
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return float(np.mean(losses)), torch.cat(scores).numpy()


def _dhcp_reference(payload):
    return payload.get("shard_reference", payload)


def _setting(
    config: Mapping[str, Any],
    primary: str,
    alias: str,
    *,
    default: Any,
) -> Any:
    if primary in config:
        return config[primary]
    return config.get(alias, default)


def _require_strict_splits(split_records) -> None:
    if split_records["val"]:
        raise ValueError("Strict 8:2 requires an empty validation split")
    for name in ("train", "test"):
        rows = split_records[name]
        labels = {int(row["label"]) for row in rows}
        if not rows or labels != {0, 1}:
            raise ValueError(
                "Strict outer-8:2 training requires both classes in the "
                f"effective {name} partition; got labels={labels}"
            )


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
