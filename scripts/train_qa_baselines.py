#!/usr/bin/env python3
"""Train native paper-baseline heads on QA question probe splits."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import sha256_file  # noqa: E402
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
    _resolve_device,
    _run_training_protocol,
)
from utils.config_utils import (  # noqa: E402
    load_config,
    qa_extraction_family_flags,
)


DEFAULT_SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train QA-adapted paper baselines for seeds 42/43/44."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=("pope", "clevr_exist_5k", "amber_discriminative"),
        required=True,
    )
    parser.add_argument(
        "--config", default="configs/model_configs_unified.yaml"
    )
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--label-protocol",
        choices=QA_LABEL_PROTOCOLS,
        default=DEFAULT_QA_LABEL_PROTOCOL,
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
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
    output_root = args.output_root or (config.get("qa_benchmarks") or {}).get("output_root")
    if not output_root:
        raise ValueError("QA output root is required by CLI or YAML")
    run_dir = Path(output_root) / args.model / args.dataset
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
    methods = normalize_baseline_methods(args.methods or extracted_methods)
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
    seeds = list(dict.fromkeys(int(value) for value in (args.seeds or DEFAULT_SEEDS)))
    if not seeds:
        raise ValueError("At least one QA baseline seed is required")
    if args.seeds is None and tuple(seeds) != DEFAULT_SEEDS:
        raise AssertionError("Default QA baseline seeds must remain 42/43/44")

    baseline_cfg = baseline_config(config)
    run_qa_baseline_training(
        model=args.model,
        dataset=args.dataset,
        label_protocol=label_protocol,
        seeds=seeds,
        methods=methods,
        split_records=split_records,
        image_split_counts=expected_split_manifest["image_counts"],
        feature_path=feature_path,
        split_path=split_path,
        baseline_dir=baseline_dir,
        baseline_cfg=baseline_cfg,
        device=_resolve_device(args.device),
    )
    print(
        f"[train_qa_baselines] Complete: {args.model}/{args.dataset}/"
        f"{label_protocol}; seeds={seeds}, methods={list(methods)}, "
        f"output={baseline_dir / 'results'}"
    )


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
) -> None:
    """Run native heads and write isolated per-seed plus aggregate reports."""

    protocol = normalize_qa_label_protocol(label_protocol)
    _run_training_protocol(
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
