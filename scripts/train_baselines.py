#!/usr/bin/env python3
"""Train/evaluate paper baselines from ``OUTPUT/baseline/features.pkl``."""

from __future__ import annotations

import argparse
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
    build_dense_baseline_matrix,
    build_metatoken_classifier,
    evaluate_hallucination_scores,
    raw_labels_to_hallucination_targets,
    select_hallucination_threshold,
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
    validate_baseline_record,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, load_pkl, save_json, save_pkl
from utils.split_utils import validate_strict_811_split


DEFAULT_METHODS = ("metatoken", "svar", "dhcp", "projectaway", "halloc")
SUMMARY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "aupr",
    "real_f1",
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
        "--run-name",
        default=None,
        help="Store results/checkpoints in isolated subdirectories such as seed42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    baseline_cfg = baseline_config(config)
    training_cfg = _baseline_training_config(config)
    baseline_dir = Path(args.output_dir) / str(
        baseline_cfg.get("output_subdir", "baseline")
    )
    feature_path = baseline_dir / "features.pkl"
    split_path = Path(args.output_dir) / "image_splits.json"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    records = load_pkl(str(feature_path))
    for record in records:
        validate_baseline_record(record)
    image_splits = load_json(str(split_path))
    image_split_counts = validate_strict_811_split(image_splits)
    configured_count = int((config.get("dataset") or {}).get("num_images", 0))
    if configured_count and sum(image_split_counts.values()) != configured_count:
        raise ValueError(
            "Strict split size differs from dataset.num_images: "
            f"{sum(image_split_counts.values())} != {configured_count}"
        )
    split_records = split_records_by_image(records, image_splits)
    _require_strict_splits(split_records)
    methods = normalize_baseline_methods(
        args.methods
        if args.methods is not None
        else training_cfg.get(
            "methods", baseline_cfg.get("methods", DEFAULT_METHODS)
        )
    )
    if not methods:
        raise ValueError("No baseline methods selected for training")

    seeds = _configured_training_seeds(args, training_cfg, baseline_cfg)
    if args.run_name is not None and len(seeds) != 1:
        raise ValueError("--run-name can only be used with one effective seed")
    device = _resolve_device(args.device)
    run_outputs: list[dict[str, Any]] = []
    run_paths: list[Path] = []
    for seed in seeds:
        seed_cfg = dict(baseline_cfg)
        seed_cfg["seed"] = int(seed)
        run_name = args.run_name
        if run_name is None and len(seeds) > 1:
            run_name = f"seed{seed}"
        output, result_path = _train_one_seed(
            model=args.model,
            seed=int(seed),
            methods=methods,
            split_records=split_records,
            image_split_counts=image_split_counts,
            feature_path=feature_path,
            split_path=split_path,
            baseline_dir=baseline_dir,
            baseline_cfg=seed_cfg,
            device=device,
            run_name=run_name,
        )
        run_outputs.append(output)
        run_paths.append(result_path)

    if bool(training_cfg.get("write_summary", True)):
        _write_training_summaries(
            model=args.model,
            baseline_dir=baseline_dir,
            outputs=run_outputs,
            result_paths=run_paths,
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
) -> tuple[dict[str, Any], Path]:
    result_dir = baseline_dir / "results"
    checkpoint_dir = baseline_dir / "checkpoints"
    if run_name is not None:
        run_name = _safe_run_name(run_name)
        result_dir = result_dir / run_name
        checkpoint_dir = checkpoint_dir / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{model}_baselines.json"
    output: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "configured_methods": list(methods),
        "feature_path": str(feature_path),
        "split_path": str(split_path),
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "headline_positive_class": "hallucination",
        "label_protocol": "local_chair_object_spans_without_gpt_semantic_review",
        "counts": {name: len(rows) for name, rows in split_records.items()},
        "image_split_counts": image_split_counts,
        "methods": {},
    }

    for method in methods:
        print(f"[BaselineTrain] seed={seed} method={method} device={device}")
        if method == "metatoken":
            result = _train_metatoken(split_records, checkpoint_dir, baseline_cfg)
        elif method == "svar":
            result = _train_svar(split_records, checkpoint_dir, baseline_cfg, device)
        elif method == "dhcp":
            result = _train_dhcp(
                split_records, baseline_dir, checkpoint_dir, baseline_cfg, device
            )
        elif method == "projectaway":
            result = _evaluate_projectaway(split_records)
        else:
            result = _train_halloc(
                split_records, baseline_dir, checkpoint_dir, baseline_cfg, device
            )
        output["methods"][method] = result
        save_json(output, str(result_path))

    print(f"[BaselineTrain] saved {result_path}")
    return output, result_path


def _baseline_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config.get("training") or {}
    if not isinstance(training, Mapping):
        raise ValueError("training must be a YAML mapping")
    baseline = training.get("baseline") or {}
    if not isinstance(baseline, Mapping):
        raise ValueError("training.baseline must be a YAML mapping")
    allowed = {"methods", "seeds", "write_summary"}
    unknown = sorted(set(baseline) - allowed)
    if unknown:
        raise ValueError(f"Unknown training.baseline options: {unknown}")
    return dict(baseline)


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


def _write_training_summaries(
    *,
    model: str,
    baseline_dir: Path,
    outputs: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
) -> None:
    if len(outputs) != len(result_paths):
        raise ValueError("Baseline outputs and result paths must have equal length")
    for output, result_path in zip(outputs, result_paths):
        summary = aggregate_baseline_outputs([output])
        markdown_path = result_path.with_name(f"{model}_baselines_summary.md")
        _write_baseline_markdown(
            markdown_path,
            summary,
            source_paths=[result_path],
        )
        print(f"[BaselineTrain] saved {markdown_path}")

    if len(outputs) > 1:
        summary = aggregate_baseline_outputs(outputs)
        result_dir = baseline_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{model}_baselines_{len(outputs)}seed"
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
) -> dict[str, Any]:
    if not outputs:
        raise ValueError("At least one baseline output is required")
    model = str(outputs[0]["model"])
    seeds = [int(output["seed"]) for output in outputs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Baseline seeds must be unique, got {seeds}")
    first_variants = _baseline_method_variants(outputs[0])
    expected = list(first_variants)
    for output in outputs:
        if str(output["model"]) != model:
            raise ValueError("Cannot aggregate baseline outputs from different models")
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
        for split in ("val", "test"):
            seed_metrics = [
                _headline_metrics(_baseline_method_variants(output)[key][1], split)
                for output in outputs
            ]
            row[f"{split}_metrics"] = {
                metric: _metric_statistics(
                    [float(values[metric]) for values in seed_metrics]
                )
                for metric in SUMMARY_METRICS
            }
        rows[key] = row

    return {
        "model": model,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "std_definition": "population",
        "headline_positive_class": "hallucination",
        "stored_label_semantics": outputs[0].get("stored_label_semantics"),
        "detector_target_semantics": outputs[0].get("detector_target_semantics"),
        "label_protocol": outputs[0].get("label_protocol"),
        "counts": outputs[0].get("counts"),
        "image_split_counts": outputs[0].get("image_split_counts"),
        "configured_methods": outputs[0].get("configured_methods"),
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
                variants[key] = (
                    f"MetaToken-{str(classifier).upper()}",
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


def _headline_metrics(result: Mapping[str, Any], split: str) -> dict[str, float]:
    metrics = result.get(f"{split}_metrics") or {}
    hallucination = metrics.get("hallucination_positive") or {}
    real = metrics.get("real_positive") or {}
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(hallucination["precision"]),
        "recall": float(hallucination["recall"]),
        "f1": float(hallucination["f1"]),
        "auc": float(hallucination["auc"]),
        "aupr": float(hallucination["aupr"]),
        "real_f1": float(real["f1"]),
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
    lines = [
        f"# {summary['model']} Baseline 结果汇总",
        "",
        "## 实验协议",
        "",
        f"- 随机种子：`{', '.join(str(seed) for seed in seeds)}`。",
        "- headline 正类：hallucination。",
        "- 阈值只在 validation set 上选择，test set 只用于最终评估。",
        f"- image split：train/val/test = "
        f"{image_counts.get('train', '?')}/{image_counts.get('val', '?')}/"
        f"{image_counts.get('test', '?')}。",
        f"- token 样本：train/val/test = "
        f"{counts.get('train', '?')}/{counts.get('val', '?')}/"
        f"{counts.get('test', '?')}。",
    ]
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
            "## Test 结果",
            "",
            "| 方法 | Accuracy | Hall. Precision | Hall. Recall | Hall. F1 | AUROC | AUPR | Real F1 |",
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
    lines.extend(
        [
            "",
            "## Validation 结果",
            "",
            "| 方法 | Accuracy | Hall. Precision | Hall. Recall | Hall. F1 | AUROC | AUPR | Real F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in methods.values():
        metrics = row["val_metrics"]
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
                "## 各随机种子的 Test Hallucination F1",
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
                f"- Test Hallucination F1 最高的方法是 **{best['display_name']}**："
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


def _train_metatoken(split_records, checkpoint_dir, cfg) -> dict[str, Any]:
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
        threshold = select_hallucination_threshold(matrices["val"][1], val_scores)
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
            "val_metrics": evaluate_hallucination_scores(
                matrices["val"][1], val_scores, threshold
            ),
            "test_metrics": evaluate_hallucination_scores(
                matrices["test"][1], test_scores, threshold
            ),
            "checkpoint": str(path),
        }
    return result


def _train_svar(split_records, checkpoint_dir, cfg, device) -> dict[str, Any]:
    matrices = {
        split: build_dense_baseline_matrix(split_records[split], "svar")[:2]
        for split in ("train", "val", "test")
    }
    svar_cfg = dict(cfg.get("svar") or {})
    seed = int(cfg.get("seed", 42))
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
    )
    path = checkpoint_dir / "svar.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": trained.state_dict,
            "input_dim": int(matrices["train"][0].shape[1]),
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "threshold": trained.threshold,
        },
    )
    return {
        "paper_config": {
            "hidden_dim": int(_setting(svar_cfg, "hidden_dim", "hidden_size", default=248)),
            "learning_rate": float(svar_cfg.get("learning_rate", 1e-3)),
            "epochs": int(_setting(svar_cfg, "epochs", "max_epochs", default=50)),
            "early_stopping_patience": int(
                svar_cfg.get("early_stopping_patience", 5)
            ),
        },
        "input_dim": int(matrices["train"][0].shape[1]),
        "threshold": trained.threshold,
        "val_metrics": trained.val_metrics,
        "test_metrics": trained.test_metrics,
        "history": trained.history,
        "checkpoint": str(path),
    }


def _train_dhcp(split_records, baseline_dir, checkpoint_dir, cfg, device):
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
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss:
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
    threshold = select_hallucination_threshold(val_raw, val_scores)
    path = checkpoint_dir / "dhcp.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "input_shape": shape,
            "hidden_dim": dhcp_hidden,
            "threshold": threshold,
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
            "early_stopping_patience": patience,
            "weighted_sampler": True,
        },
        "input_shape": list(shape),
        "input_dim": input_dim,
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(val_raw, val_scores, threshold),
        "test_metrics": evaluate_hallucination_scores(test_raw, test_scores, threshold),
        "history": history,
        "checkpoint": str(path),
    }


def _evaluate_projectaway(split_records) -> dict[str, Any]:
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
    threshold = select_hallucination_threshold(labels["val"], scores["val"])
    return {
        "paper_config": {"training_free": True, "threshold_selected_on": "val"},
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(
            labels["val"], scores["val"], threshold
        ),
        "test_metrics": evaluate_hallucination_scores(
            labels["test"], scores["test"], threshold
        ),
    }


def _train_halloc(split_records, baseline_dir, checkpoint_dir, cfg, device):
    seed = int(cfg.get("seed", 42))
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
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        if scheduler is not None:
            scheduler.step()
        if val_loss < best_loss:
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
    threshold = select_hallucination_threshold(val_raw, val_scores)
    path = checkpoint_dir / "halloc.pt"
    _atomic_torch_save(
        path,
        {
            "state_dict": best_state,
            "threshold": threshold,
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
            "early_stopping_patience": patience,
        },
        "threshold": threshold,
        "val_metrics": evaluate_hallucination_scores(val_raw, val_scores, threshold),
        "test_metrics": evaluate_hallucination_scores(test_raw, test_scores, threshold),
        "history": history,
        "checkpoint": str(path),
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
    for name in ("train", "val", "test"):
        rows = split_records[name]
        labels = {int(row["label"]) for row in rows}
        if not rows or labels != {0, 1}:
            raise ValueError(
                f"Strict 8:1:1 requires both classes in {name}; got labels={labels}"
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
