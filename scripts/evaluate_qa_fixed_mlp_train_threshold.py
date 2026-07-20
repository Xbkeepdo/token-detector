#!/usr/bin/env python3
"""Re-evaluate the isolated fixed MLP checkpoints with train-set thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.qa_benchmark import load_jsonl, sha256_file  # noqa: E402
from detection.qa_probe import (  # noqa: E402
    choose_real_f1_threshold,
    classification_metrics,
    validate_image_level_splits,
)
from scripts.extract_qa_baselines import MANIFEST_NAME  # noqa: E402
from scripts.train_qa_fixed_mlp_experiment import (  # noqa: E402
    EXPERIMENT_NAME,
    FEATURE_SPECS,
    SEEDS,
    ExperimentConfig,
    FixedMLP,
    _aggregate_feature_results,
    _atomic_json,
    _atomic_text,
    _build_matrices,
    _fmt,
    _load_json,
    _predict,
    _resolve_device,
    _training_fingerprint,
    _validate_aligned_rows,
    _validate_baseline_manifest,
    _validate_reusable_result,
)
from scripts.train_qa_probes import _validate_training_artifacts  # noqa: E402
from utils.config_utils import load_config  # noqa: E402


EVALUATION_NAME = "mlp_bn_relu_trainloss_es_train_real_f1_threshold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Real-F1 thresholds on train for fixed-MLP checkpoints."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--label-protocol",
        choices=("answer_correctness_all", "object_hallucination_yes_only"),
        default="answer_correctness_all",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    yaml_config = load_config(args.config)
    qa_cfg = yaml_config.get("qa_benchmarks") or {}
    output_root = args.output_root or qa_cfg.get("output_root")
    if not output_root:
        raise ValueError("QA output root is required by CLI or YAML")
    run_root = Path(output_root) / args.model / args.dataset
    baseline_root = run_root / "baseline" / args.label_protocol
    paths = {
        "root_features": run_root / "features.pkl",
        "baseline_features": baseline_root / "features.pkl",
        "labels": run_root / "labels.jsonl",
        "splits": run_root / "image_splits.json",
        "baseline_manifest": baseline_root / MANIFEST_NAME,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation inputs: {missing}")
    _validate_baseline_manifest(
        paths=paths,
        model=args.model,
        dataset=args.dataset,
        label_protocol=args.label_protocol,
    )
    training_fingerprint, training_provenance = _training_fingerprint(
        paths=paths,
        model=args.model,
        dataset=args.dataset,
        label_protocol=args.label_protocol,
        config=config,
    )
    evaluation_fingerprint = _evaluation_fingerprint(training_fingerprint)

    print("[TrainThreshold] Loading and validating feature cohorts...")
    with paths["root_features"].open("rb") as handle:
        root_rows = pickle.load(handle)
    labels_list = load_jsonl(paths["labels"])
    labels = {str(row["key"]): row for row in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("labels.jsonl contains duplicate keys")
    with paths["splits"].open(encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    _validate_training_artifacts(root_rows, labels, split_manifest)
    image_counts = validate_image_level_splits(root_rows)
    with paths["baseline_features"].open("rb") as handle:
        baseline_rows = pickle.load(handle)
    baseline_by_key = {str(row.get("key")): row for row in baseline_rows}
    if len(baseline_by_key) != len(baseline_rows):
        raise ValueError("Baseline QA features contain duplicate keys")
    if {str(row.get("key")) for row in root_rows} != set(baseline_by_key):
        raise ValueError("Root and baseline QA feature cohorts differ")
    _validate_aligned_rows(root_rows, baseline_by_key, args.label_protocol)
    matrices = _build_matrices(
        root_rows,
        baseline_by_key,
        label_protocol=args.label_protocol,
    )
    del root_rows, baseline_rows, baseline_by_key

    source_root = (
        run_root / "experiments" / EXPERIMENT_NAME / args.label_protocol
    )
    result_root = run_root / "experiments" / EVALUATION_NAME / args.label_protocol
    device = _resolve_device(args.device)
    all_results: dict[str, list[dict[str, Any]]] = {}
    for spec in FEATURE_SPECS:
        all_results[spec.name] = []
        X_train, y_train = matrices[spec.name]["train"]
        X_test, y_test = matrices[spec.name]["test"]
        for seed in SEEDS:
            source_dir = source_root / spec.name / f"seed_{seed}"
            source_result = _load_json(source_dir / "result.json")
            _validate_reusable_result(
                source_result,
                feature_name=spec.name,
                seed=seed,
                fingerprint=training_fingerprint,
            )
            checkpoint = torch.load(
                source_dir / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("training_input_fingerprint") != training_fingerprint:
                raise RuntimeError(
                    f"Checkpoint fingerprint mismatch: {source_dir / 'checkpoint.pt'}"
                )
            model = FixedMLP(int(source_result["input_dim"]), config)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            criterion = nn.BCEWithLogitsLoss()
            train_loss, train_probability = _predict(
                model, X_train, y_train, config.batch_size, criterion, device
            )
            test_loss, test_probability = _predict(
                model, X_test, y_test, config.batch_size, criterion, device
            )
            threshold, train_threshold_f1 = choose_real_f1_threshold(
                y_train, train_probability
            )
            result = {
                **source_result,
                "evaluation_variant": EVALUATION_NAME,
                "threshold": float(threshold),
                "threshold_selection": "train_real_f1",
                "train_threshold_f1": float(train_threshold_f1),
                "restored_train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "train_metrics": classification_metrics(
                    y_train, train_probability, threshold
                ),
                "test_metrics": classification_metrics(
                    y_test, test_probability, threshold
                ),
                "evaluation_fingerprint": evaluation_fingerprint,
                "source_result": str(source_dir / "result.json"),
                "source_checkpoint": str(source_dir / "checkpoint.pt"),
                "artifacts": {
                    "source_checkpoint": str(source_dir / "checkpoint.pt"),
                    "source_history": str(source_dir / "history.json"),
                },
            }
            output_path = result_root / spec.name / f"seed_{seed}" / "result.json"
            _atomic_json(output_path, result)
            all_results[spec.name].append(result)
            test = result["test_metrics"]
            print(
                f"[TrainThreshold] {spec.name} seed={seed} "
                f"threshold={threshold:.6f} AUC={test['auroc']:.4f} "
                f"Real-F1={test['real']['f1']:.4f} "
                f"Hall-F1={test['hallucination']['f1']:.4f}"
            )

    summary = {
        "evaluation": EVALUATION_NAME,
        "source_experiment": EXPERIMENT_NAME,
        "model": args.model,
        "dataset": args.dataset,
        "label_protocol": args.label_protocol,
        "seeds": list(SEEDS),
        "image_counts": image_counts,
        "network": {
            "hidden_sizes": list(config.hidden_sizes),
            "structure": "Linear-BatchNorm-ReLU-Dropout",
            "output_dim": 1,
            "initialization": config.initialization,
        },
        "training_config": asdict(config),
        "threshold_selection": "train_real_f1",
        "training_input_fingerprint": training_fingerprint,
        "training_provenance": training_provenance,
        "evaluation_fingerprint": evaluation_fingerprint,
        "features": {
            spec.name: {
                **_aggregate_feature_results(all_results[spec.name], spec),
                "threshold": _stats(
                    [float(result["threshold"]) for result in all_results[spec.name]]
                ),
            }
            for spec in FEATURE_SPECS
        },
    }
    json_path = result_root / "summary_3seed.json"
    markdown_path = result_root / "summary_3seed.md"
    _atomic_json(json_path, summary)
    _atomic_text(markdown_path, _markdown(summary, all_results))
    print(f"[TrainThreshold] Summary JSON: {json_path}")
    print(f"[TrainThreshold] Summary Markdown: {markdown_path}")


def _evaluation_fingerprint(training_fingerprint: str) -> str:
    payload = {
        "training_input_fingerprint": training_fingerprint,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "qa_probe_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "detection" / "qa_probe.py"
        ),
        "threshold_selection": "train_real_f1",
        "threshold_algorithm": "exact_unique_scores_max_real_f1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _markdown(
    summary: Mapping[str, Any],
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    lines = [
        f"# {summary['model']} {summary['dataset']} 训练集阈值搜索结果",
        "",
        "## 协议",
        "",
        f"- 复用 `{summary['source_experiment']}` 的 9 个 minimum-train-loss checkpoint，不重新训练。",
        "- 每个 feature/seed 只在 train 上搜索使 Real-positive F1 最大的阈值；test 仅用于一次最终评估。",
        "- 标签为 0=hallucination、1=real；网络、优化器和 early stopping 与固定 0.5 实验完全相同。",
        f"- image split：train/val/test = {summary['image_counts']['train']}/0/{summary['image_counts']['test']}。",
        "",
        "## Test 结果（3 seeds 总体均值 ± 总体标准差）",
        "",
        "| 特征 | Train threshold | AUROC | Real F1 | Hall. F1 | Accuracy | Balanced Acc. | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature in summary["features"].values():
        test = feature["test_metrics"]
        lines.append(
            "| "
            + " | ".join((
                str(feature["display_name"]),
                _fmt(feature["threshold"]),
                _fmt(test["auroc"]),
                _fmt(test["real"]["f1"]),
                _fmt(test["hallucination"]["f1"]),
                _fmt(test["accuracy"]),
                _fmt(test["balanced_accuracy"]),
                _fmt(test["macro_f1"]),
            ))
            + " |"
        )
    lines.extend((
        "",
        "## 各 seed",
        "",
        "| 特征 | Seed | Threshold | AUROC | Real F1 | Hall. F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for spec in FEATURE_SPECS:
        for result in all_results[spec.name]:
            test = result["test_metrics"]
            lines.append(
                f"| {spec.display_name} | {result['seed']} | "
                f"{result['threshold']:.6f} | {test['auroc']:.4f} | "
                f"{test['real']['f1']:.4f} | "
                f"{test['hallucination']['f1']:.4f} | "
                f"{test['accuracy']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "values": [float(value) for value in values],
    }


if __name__ == "__main__":
    main()
