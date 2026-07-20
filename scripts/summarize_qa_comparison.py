#!/usr/bin/env python3
"""Combine existing QA probe and configured-baseline reports without retraining.

This script is deliberately reporting-only. It reads already aggregated test
metrics (and native baseline per-seed metric JSON only when the aggregate omits
one class), preserves each train-F1-selected threshold, and never ranks or selects a
method from test performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMA_VERSION = "qa-comparison-v1"
POSITIONS = ("prompt_last_token", "question_object_pre_token")
METRIC_KEYS = (
    "auroc",
    "real_aupr",
    "real_f1",
    "real_precision",
    "real_recall",
    "hallucination_aupr",
    "hallucination_f1",
    "hallucination_precision",
    "hallucination_recall",
)
DEFAULT_SEEDS = (43, 44, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a read-only three-seed QA comparison of DGST/ADS+CGC and "
            "configured baselines."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=("pope", "clevr_exist_5k", "amber_discriminative"),
        required=True,
    )
    parser.add_argument(
        "--label-protocol",
        choices=("answer_correctness_all", "object_hallucination_yes_only"),
        required=True,
    )
    parser.add_argument(
        "--config", default="configs/model_configs_unified.yaml"
    )
    parser.add_argument("--output-root")
    parser.add_argument("--probe-summary")
    parser.add_argument("--baseline-summary")
    parser.add_argument("--feature-summary")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--allow-unscoped-probe-summary",
        action="store_true",
        help=(
            "Allow a legacy bare feature-set mapping with no model/dataset/"
            "label_protocol provenance. The report marks the scope as CLI-"
            "asserted; leave disabled for final experiments."
        ),
    )
    return parser.parse_args()


def resolve_output_root(
    config: Mapping[str, Any],
    override: str | os.PathLike[str] | None = None,
) -> Path:
    qa_cfg = config.get("qa_benchmarks") or {}
    if not isinstance(qa_cfg, Mapping):
        raise ValueError("qa_benchmarks must be a YAML mapping")
    value = override or qa_cfg.get("output_root")
    if not value:
        raise ValueError(
            "QA output root must be provided by --output-root or "
            "qa_benchmarks.output_root in --config"
        )
    return Path(value)


def main() -> None:
    args = parse_args()
    from utils.config_utils import load_config, qa_extraction_family_flags

    config = load_config(args.config)
    family_flags = qa_extraction_family_flags(config)
    root_enabled = bool(family_flags["method"] or family_flags["ads_cgc"])
    if not root_enabled or not family_flags["baseline"]:
        print(
            "[summarize_qa_comparison] Skipped: a cross-family comparison "
            f"requires QA mode 'all' (current={family_flags['mode']})."
        )
        return
    output_root = resolve_output_root(config, args.output_root)
    run_dir = Path(output_root) / args.model / args.dataset
    probe_path = (
        Path(args.probe_summary)
        if args.probe_summary
        else _discover_probe_summary(run_dir, args.label_protocol)
    )
    baseline_path = (
        Path(args.baseline_summary)
        if args.baseline_summary
        else _discover_baseline_summary(
            run_dir,
            args.model,
            args.dataset,
            args.label_protocol,
            trainer=str(
                (config.get("qa_benchmarks") or {}).get(
                    "baseline_trainer", "native_paper"
                )
            ),
            num_seeds=len(args.seeds),
        )
    )
    feature_path = (
        Path(args.feature_summary)
        if args.feature_summary
        else run_dir / "qa_feature_summary.json"
    )
    feature_summary = (
        _load_json_object(feature_path) if feature_path.is_file() else None
    )
    comparison = build_comparison(
        model=args.model,
        dataset=args.dataset,
        label_protocol=args.label_protocol,
        expected_seeds=args.seeds,
        probe_summary=_load_json_object(probe_path),
        probe_summary_path=probe_path,
        baseline_summary=_load_json_object(baseline_path),
        baseline_summary_path=baseline_path,
        feature_summary=feature_summary,
        feature_summary_path=feature_path if feature_summary is not None else None,
        allow_unscoped_probe_summary=bool(args.allow_unscoped_probe_summary),
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "comparison" / args.label_protocol
    )
    stem = (
        f"{args.model}_{args.dataset}_{args.label_protocol}_"
        "qa_comparison_3seed"
    )
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    write_comparison(comparison, json_path, markdown_path)
    print(f"[summarize_qa_comparison] JSON: {json_path}")
    print(f"[summarize_qa_comparison] Markdown: {markdown_path}")


def build_comparison(
    *,
    model: str,
    dataset: str,
    label_protocol: str,
    expected_seeds: Sequence[int],
    probe_summary: Mapping[str, Any],
    probe_summary_path: Path,
    baseline_summary: Mapping[str, Any],
    baseline_summary_path: Path,
    feature_summary: Mapping[str, Any] | None = None,
    feature_summary_path: Path | None = None,
    allow_unscoped_probe_summary: bool = False,
) -> dict[str, Any]:
    seeds = _normalize_expected_seeds(expected_seeds)
    probe_methods, probe_metadata, scoped = _probe_methods_and_metadata(
        probe_summary,
        model=model,
        dataset=dataset,
        label_protocol=label_protocol,
        allow_unscoped=allow_unscoped_probe_summary,
    )
    baseline_metadata = _validate_baseline_scope(
        baseline_summary,
        model=model,
        dataset=dataset,
        label_protocol=label_protocol,
    )
    _validate_seed_list(
        baseline_summary.get("seeds"), seeds, "native baseline summary"
    )
    top_probe_seeds = probe_metadata.get("seeds")
    if top_probe_seeds is not None:
        _validate_seed_list(top_probe_seeds, seeds, "QA probe summary")

    extraction_coverage = _feature_summary_coverage(feature_summary)
    probe_coverage = probe_metadata.get("coverage")
    rows: list[dict[str, Any]] = []
    probe_cohort_total = _maximum_probe_count(probe_methods)
    for feature_set, aggregate in probe_methods.items():
        if not isinstance(aggregate, Mapping):
            raise ValueError(
                f"Probe aggregate for {feature_set!r} must be a mapping"
            )
        row_seeds = aggregate.get("seeds")
        if row_seeds is not None:
            _validate_seed_list(
                row_seeds, seeds, f"QA probe feature set {feature_set!r}"
            )
        position = _position_from_feature_set(str(feature_set))
        rows.append(
            {
                "family": _probe_family(str(feature_set)),
                "method": str(feature_set),
                "display_name": str(
                    aggregate.get("display_name") or feature_set
                ),
                "position": position,
                "position_definition": _position_definition(position),
                "coverage": _coverage_for_position(
                    aggregate.get("coverage"),
                    probe_coverage,
                    extraction_coverage,
                    position,
                    counts=aggregate.get("counts"),
                    cohort_total=probe_cohort_total,
                ),
                "metrics": _probe_metrics(aggregate),
                "metrics_by_threshold": _probe_threshold_metrics(aggregate),
                "source": str(probe_summary_path),
            }
        )

    baseline_seed_outputs = _load_baseline_seed_outputs(
        baseline_summary, baseline_summary_path, seeds
    )
    baseline_methods = baseline_summary.get("methods")
    if not isinstance(baseline_methods, Mapping) or not baseline_methods:
        raise ValueError("Native baseline summary has no methods mapping")
    baseline_coverage = _baseline_coverage(
        baseline_summary, baseline_metadata
    )
    for method, aggregate in baseline_methods.items():
        if not isinstance(aggregate, Mapping):
            raise ValueError(
                f"Baseline aggregate for {method!r} must be a mapping"
            )
        metrics = _baseline_metrics_from_aggregate(aggregate)
        threshold_metrics = _baseline_threshold_metrics_from_aggregate(
            aggregate
        )
        if baseline_seed_outputs:
            seed_metrics = [
                _baseline_seed_metrics(output, str(method))
                for output in baseline_seed_outputs
            ]
            complete = _aggregate_metric_rows(seed_metrics)
            _validate_common_statistics(
                metrics,
                complete,
                source=f"native baseline {method!r}",
            )
            metrics = complete
            if threshold_metrics:
                threshold_metrics = {
                    mode: _aggregate_metric_rows([
                        _baseline_seed_metrics(
                            output, str(method), threshold_mode=mode
                        )
                        for output in baseline_seed_outputs
                    ])
                    for mode in threshold_metrics
                }
        rows.append(
            {
                "family": "native_baseline",
                "method": str(method),
                "display_name": str(
                    aggregate.get("display_name") or method
                ),
                "position": "prompt_last_token",
                "position_definition": (
                    "QA-adapted baseline at the final prompt token; this "
                    "causal state predicts the first generated token."
                ),
                "coverage": baseline_coverage,
                "metrics": metrics,
                "metrics_by_threshold": threshold_metrics,
                "source": str(baseline_summary_path),
            }
        )

    active_positions = [
        position
        for position in POSITIONS
        if any(row.get("position") == position for row in rows)
    ]
    active_extraction_coverage = {
        position: extraction_coverage[position]
        for position in active_positions
        if position in extraction_coverage
    }
    active_probe_coverage = _filter_position_coverage(
        probe_coverage, active_positions
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "dataset": dataset,
        "label_protocol": label_protocol,
        "stored_label_semantics": {
            "0": "hallucination",
            "1": "real",
        },
        "headline_positive_class": "real",
        "seeds": seeds,
        "num_seeds": len(seeds),
        "std_definition": "population",
        "report_policy": (
            "read existing test metrics only; report fixed-0.5 and "
            "train-F1-selected thresholds from the same minimum-train-loss "
            "checkpoint; no test-set method selection or ranking"
        ),
        "probe_scope_provenance": (
            "summary_metadata" if scoped else "cli_asserted_legacy_summary"
        ),
        "position_metadata": {
            position: _position_definition(position)
            for position in active_positions
        },
        "coverage_metadata": {
            "probe_summary": active_probe_coverage,
            "feature_extraction": active_extraction_coverage,
            "native_baseline": baseline_coverage,
        },
        "sources": {
            "probe_summary": str(probe_summary_path),
            "baseline_summary": str(baseline_summary_path),
            "feature_summary": (
                str(feature_summary_path)
                if feature_summary_path is not None
                else None
            ),
            "baseline_seed_results_used_for_dual_class_columns": [
                str(path)
                for path in _baseline_seed_paths(
                    baseline_summary, baseline_summary_path, seeds
                )
                if path.is_file()
            ],
        },
        "rows": rows,
    }


def write_comparison(
    comparison: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        json_path,
        json.dumps(
            comparison, ensure_ascii=False, indent=2, sort_keys=False
        )
        + "\n",
    )
    _atomic_text(markdown_path, render_markdown(comparison))


def render_markdown(comparison: Mapping[str, Any]) -> str:
    seeds = ", ".join(str(seed) for seed in comparison["seeds"])
    active_positions = _active_report_positions(comparison)
    lines = [
        (
            f"# {comparison['model']} / {comparison['dataset']} / "
            f"{comparison['label_protocol']} QA 三随机种子对比"
        ),
        "",
        "## 协议",
        "",
        f"- 随机种子：{seeds}；总体均值 ± 总体标准差。",
        "- 标签：0=hallucination，1=real；主报告正类为 real。",
        "- 该脚本只汇总已有 test 指标；同一 minimum-train-loss checkpoint 同时报告固定 0.5 和 train Real-F1 搜索阈值，不按 test 选择、排序或挑选方法。",
    ]
    for position in active_positions:
        lines.append(_position_markdown_bullet(position))
    lines.extend(
        [
            (
                "- Native baselines 与主方法共享 prompt 最后一个 token 的 "
                "因果状态，位置统一记为 prompt_last_token。"
            ),
        ]
    )
    for mode, title in (
        ("fixed_0.5", "固定阈值 0.5"),
        ("train_f1", "Train Real-F1 搜索阈值"),
    ):
        lines.extend((
            "",
            f"## Test 对比（{title}）",
            "",
            "| 类型 | 方法 | 位置 | 覆盖率 | AUROC | Real AUPR | Real F1 | Real P | Real R | Hall. AUPR | Hall. F1 | Hall. P | Hall. R |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for row in comparison["rows"]:
            by_threshold = row.get("metrics_by_threshold") or {}
            metrics = by_threshold.get(mode)
            if not isinstance(metrics, Mapping):
                if mode == "train_f1":
                    metrics = row["metrics"]
                else:
                    continue
            values = [_format_stat(metrics.get(key)) for key in METRIC_KEYS]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    row["family"],
                    row["display_name"],
                    row["position"],
                    _format_coverage(row.get("coverage")),
                    " | ".join(values),
                )
            )
    lines.extend(
        [
            "",
            "## 覆盖率说明",
            "",
            (
                "- 覆盖率只描述已有特征/入选 cohort，不参与 test 指标重算，"
                "也不用于选择方法。"
            ),
        ]
    )
    for position in active_positions:
        coverage = (
            (comparison.get("coverage_metadata") or {})
            .get("feature_extraction", {})
            .get(position)
        )
        if coverage is not None:
            lines.append(
                f"- {position}: {_format_coverage(coverage)}。"
            )
    lines.extend(
        [
            "",
            "## 输入结果",
            "",
            f"- Probe：{comparison['sources']['probe_summary']}",
            f"- Baseline：{comparison['sources']['baseline_summary']}",
        ]
    )
    feature_source = comparison["sources"].get("feature_summary")
    if feature_source:
        lines.append(f"- Feature coverage：{feature_source}")
    return "\n".join(lines) + "\n"


def _active_report_positions(
    comparison: Mapping[str, Any],
) -> list[str]:
    metadata = comparison.get("position_metadata")
    if isinstance(metadata, Mapping):
        return [position for position in POSITIONS if position in metadata]
    rows = comparison.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [
        position
        for position in POSITIONS
        if any(
            isinstance(row, Mapping) and row.get("position") == position
            for row in rows
        )
    ]


def _position_markdown_bullet(position: str) -> str:
    if position == "prompt_last_token":
        return (
            "- prompt_last_token：使用完整 prompt 的最后一个因果状态；"
            "该状态预测第一个生成 token。"
        )
    if position == "question_object_pre_token":
        return (
            "- question_object_pre_token：object 是问题文本中的 object "
            "word；定位其真实上下文首 token，并使用该 token 前一因果位置。"
        )
    raise ValueError(f"Unsupported QA position in report: {position!r}")


def _probe_methods_and_metadata(
    summary: Mapping[str, Any],
    *,
    model: str,
    dataset: str,
    label_protocol: str,
    allow_unscoped: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any], bool]:
    protocols = summary.get("label_protocols")
    if isinstance(protocols, Mapping):
        methods = protocols.get(label_protocol)
        if not isinstance(methods, Mapping):
            raise ValueError(
                "QA probe summary has no results for label protocol "
                f"{label_protocol!r}"
            )
        if summary.get("model") is not None:
            _validate_exact_scope(
                summary, "model", model, "QA probe summary"
            )
        if summary.get("dataset") is not None:
            _validate_exact_scope(
                summary, "dataset", dataset, "QA probe summary"
            )
        metadata = {
            key: value
            for key, value in summary.items()
            if key != "label_protocols"
        }
        metadata["label_protocol"] = label_protocol
        return methods, metadata, True

    methods = summary.get("methods")
    if isinstance(methods, Mapping):
        metadata = summary
        _validate_exact_scope(metadata, "model", model, "QA probe summary")
        _validate_exact_scope(
            metadata, "dataset", dataset, "QA probe summary"
        )
        _validate_exact_scope(
            metadata,
            "label_protocol",
            label_protocol,
            "QA probe summary",
        )
        return methods, metadata, True
    if not allow_unscoped:
        raise ValueError(
            "QA probe summary has no scoped methods wrapper. Refusing to infer "
            "model/dataset/label protocol; regenerate it with the unified "
            "trainer or pass --allow-unscoped-probe-summary explicitly."
        )
    reserved = {
        "model",
        "dataset",
        "label_protocol",
        "seeds",
        "coverage",
        "schema_version",
    }
    methods = {
        key: value for key, value in summary.items()
        if key not in reserved
    }
    if not methods:
        raise ValueError("Legacy QA probe summary has no feature sets")
    return methods, summary, False


def _validate_baseline_scope(
    summary: Mapping[str, Any],
    *,
    model: str,
    dataset: str,
    label_protocol: str,
) -> dict[str, Any]:
    _validate_exact_scope(summary, "model", model, "native baseline summary")
    summary_dataset = summary.get("dataset")
    if summary_dataset is not None and str(summary_dataset) != dataset:
        raise ValueError(
            f"native baseline summary dataset={summary_dataset!r}, "
            f"expected {dataset!r}"
        )
    protocol = str(summary.get("label_protocol") or "")
    if label_protocol not in protocol:
        raise ValueError(
            "native baseline summary label protocol does not match: "
            f"{protocol!r} versus {label_protocol!r}"
        )
    positive = str(
        summary.get("headline_positive_class") or "real"
    ).lower()
    if positive != "real":
        raise ValueError(
            "Native baseline aggregate must report real as the headline "
            f"positive class, got {positive!r}"
        )
    return {
        "counts": summary.get("counts"),
        "image_split_counts": summary.get("image_split_counts"),
        "label_protocol": protocol,
    }


def _probe_metrics(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    payload = aggregate.get("test_metrics")
    if not isinstance(payload, Mapping):
        payload = aggregate
    paths = {
        "auroc": ("auroc", "auc"),
        "real_aupr": ("real_aupr", "real.aupr", "real_positive.aupr"),
        "real_f1": ("real.f1", "real_positive.f1"),
        "real_precision": (
            "real.precision", "real_positive.precision"
        ),
        "real_recall": ("real.recall", "real_positive.recall"),
        "hallucination_aupr": (
            "hallucination_aupr",
            "hallucination.aupr",
            "hallucination_positive.aupr",
        ),
        "hallucination_f1": (
            "hallucination.f1", "hallucination_positive.f1"
        ),
        "hallucination_precision": (
            "hallucination.precision",
            "hallucination_positive.precision",
        ),
        "hallucination_recall": (
            "hallucination.recall",
            "hallucination_positive.recall",
        ),
    }
    return {
        name: _first_stat(payload, candidates)
        for name, candidates in paths.items()
    }


def _probe_threshold_metrics(
    aggregate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reports = aggregate.get("threshold_reports") or {}
    return {
        str(mode): _probe_metrics(report)
        for mode, report in reports.items()
        if isinstance(report, Mapping)
    }


def _baseline_metrics_from_aggregate(
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    payload = aggregate.get("test_metrics")
    if not isinstance(payload, Mapping):
        raise ValueError("Baseline method aggregate has no test_metrics")
    return {
        "auroc": _first_stat(payload, ("auc", "auroc")),
        "real_aupr": _first_stat(payload, ("aupr", "real_aupr")),
        "real_f1": _first_stat(payload, ("f1", "real.f1")),
        "real_precision": _first_stat(
            payload, ("precision", "real.precision")
        ),
        "real_recall": _first_stat(
            payload, ("recall", "real.recall")
        ),
        "hallucination_aupr": _first_stat(
            payload, ("hallucination_aupr",)
        ),
        "hallucination_f1": _first_stat(
            payload, ("other_f1", "hallucination.f1")
        ),
        "hallucination_precision": _first_stat(
            payload, ("hallucination.precision",)
        ),
        "hallucination_recall": _first_stat(
            payload, ("hallucination.recall",)
        ),
    }


def _baseline_threshold_metrics_from_aggregate(
    aggregate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reports = aggregate.get("threshold_reports") or {}
    return {
        str(mode): _baseline_metrics_from_aggregate(report)
        for mode, report in reports.items()
        if isinstance(report, Mapping)
    }


def _baseline_seed_metrics(
    output: Mapping[str, Any],
    method: str,
    *,
    threshold_mode: str | None = None,
) -> dict[str, float]:
    methods = output.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Native baseline seed result has no methods")
    if method.startswith("metatoken_"):
        classifier = method[len("metatoken_"):]
        metatoken = methods.get("metatoken")
        if not isinstance(metatoken, Mapping) or classifier not in metatoken:
            raise ValueError(
                f"Seed result has no MetaToken classifier {classifier!r}"
            )
        result = metatoken[classifier]
    else:
        result = methods.get(method)
    if not isinstance(result, Mapping):
        raise ValueError(f"Seed result has no baseline method {method!r}")
    if threshold_mode is not None:
        reports = result.get("threshold_reports") or {}
        result = reports.get(threshold_mode)
        if not isinstance(result, Mapping):
            raise ValueError(
                f"Baseline method {method!r} lacks threshold report "
                f"{threshold_mode!r}"
            )
    metrics = result.get("test_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Baseline method {method!r} has no test metrics")
    real = metrics.get("real_positive") or metrics.get("real")
    hallucination = (
        metrics.get("hallucination_positive")
        or metrics.get("hallucination")
    )
    if not isinstance(real, Mapping) or not isinstance(
        hallucination, Mapping
    ):
        raise ValueError(
            f"Baseline method {method!r} lacks dual-class test metrics"
        )
    return {
        "auroc": _required_float(real, ("auc", "auroc")),
        "real_aupr": _required_float(real, ("aupr",)),
        "real_f1": _required_float(real, ("f1",)),
        "real_precision": _required_float(real, ("precision",)),
        "real_recall": _required_float(real, ("recall",)),
        "hallucination_aupr": _required_float(
            hallucination, ("aupr",)
        ),
        "hallucination_f1": _required_float(
            hallucination, ("f1",)
        ),
        "hallucination_precision": _required_float(
            hallucination, ("precision",)
        ),
        "hallucination_recall": _required_float(
            hallucination, ("recall",)
        ),
    }


def _aggregate_metric_rows(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, Any]]:
    return {
        metric: _statistics([float(row[metric]) for row in rows])
        for metric in METRIC_KEYS
    }


def _load_baseline_seed_outputs(
    summary: Mapping[str, Any],
    summary_path: Path,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    paths = _baseline_seed_paths(summary, summary_path, seeds)
    if not paths or not all(path.is_file() for path in paths):
        return []
    outputs = [_load_json_object(path) for path in paths]
    for seed, output in zip(seeds, outputs):
        if int(output.get("seed", -1)) != int(seed):
            raise ValueError(
                f"Baseline seed result {seed} points to seed "
                f"{output.get('seed')!r}"
            )
    return outputs


def _baseline_seed_paths(
    summary: Mapping[str, Any],
    summary_path: Path,
    seeds: Sequence[int],
) -> list[Path]:
    configured = summary.get("seed_result_paths")
    if isinstance(configured, Sequence) and not isinstance(
        configured, (str, bytes)
    ):
        values = [Path(str(value)) for value in configured]
        if len(values) == len(seeds):
            resolved = [
                path if path.is_absolute() else summary_path.parent / path
                for path in values
            ]
            if all(path.is_file() for path in resolved):
                return resolved
    suffix = "_3seed"
    stem = summary_path.stem
    base_stem = (
        stem[: -len(suffix)] if stem.endswith(suffix) else stem
    )
    return [
        summary_path.parent / f"seed{seed}" / f"{base_stem}.json"
        for seed in seeds
    ]


def _validate_common_statistics(
    aggregate: Mapping[str, Any],
    complete: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for metric in METRIC_KEYS:
        left = aggregate.get(metric)
        right = complete.get(metric)
        if left is None or right is None:
            continue
        left_mean = left.get("mean")
        right_mean = right.get("mean")
        if left_mean is not None and not math.isclose(
            float(left_mean), float(right_mean), rel_tol=1e-7, abs_tol=1e-9
        ):
            raise ValueError(
                f"{source} aggregate {metric} disagrees with seed results"
            )


def _coverage_for_position(
    row_coverage: Any,
    probe_coverage: Any,
    extraction_coverage: Mapping[str, Any],
    position: str,
    *,
    counts: Any = None,
    cohort_total: int | None = None,
) -> Any:
    if row_coverage is not None:
        return row_coverage
    count_total = _split_count_total(counts)
    if count_total is not None:
        split_counts = {
            split: _optional_int(counts.get(split))
            for split in ("train", "val", "test")
        }
        return {
            **_coverage_record(
                count_total, cohort_total, "qa_probe_protocol_cohort"
            ),
            "split_counts": split_counts,
        }
    found = _mapping_position_value(probe_coverage, position)
    if found is not None:
        return found
    return extraction_coverage.get(position)


def _maximum_probe_count(methods: Mapping[str, Any]) -> int | None:
    totals = [
        total
        for value in methods.values()
        if isinstance(value, Mapping)
        for total in (_split_count_total(value.get("counts")),)
        if total is not None
    ]
    return max(totals) if totals else None


def _split_count_total(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    present = [
        _optional_int(value.get(split))
        for split in ("train", "val", "test")
        if value.get(split) is not None
    ]
    return sum(present) if present else None


def _feature_summary_coverage(
    summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {}
    total = _optional_int(
        summary.get("num_questions", summary.get("total"))
    )
    result = {}
    for position in POSITIONS:
        available = _optional_int(summary.get(position))
        if available is None:
            positions = summary.get("positions")
            if isinstance(positions, Mapping):
                value = positions.get(position)
                if isinstance(value, Mapping):
                    available = _optional_int(
                        value.get("available", value.get("count"))
                    )
                else:
                    available = _optional_int(value)
        if available is not None:
            result[position] = _coverage_record(
                available, total, "qa_feature_summary"
            )
    return result


def _filter_position_coverage(
    coverage: Any,
    active_positions: Sequence[str],
) -> Any:
    if not isinstance(coverage, Mapping):
        return coverage
    active = set(active_positions)
    result = {
        key: value
        for key, value in coverage.items()
        if key not in POSITIONS or key in active
    }
    positions = coverage.get("positions")
    if isinstance(positions, Mapping):
        result["positions"] = {
            position: positions[position]
            for position in POSITIONS
            if position in active and position in positions
        }
    return result


def _baseline_coverage(
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        return None
    available = sum(
        _optional_int(counts.get(split)) or 0
        for split in ("train", "val", "test")
    )
    return {
        **_coverage_record(
            available, available, "native_baseline_selected_cohort"
        ),
        "split_counts": {
            split: _optional_int(counts.get(split))
            for split in ("train", "val", "test")
        },
        "image_split_counts": metadata.get("image_split_counts"),
    }


def _coverage_record(
    available: int,
    total: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "available": int(available),
        "total": int(total) if total is not None else None,
        "rate": (
            float(available / total)
            if total not in (None, 0)
            else None
        ),
        "source": source,
    }


def _mapping_position_value(value: Any, position: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    if position in value:
        return value[position]
    positions = value.get("positions")
    if isinstance(positions, Mapping):
        return positions.get(position)
    return None


def _position_from_feature_set(feature_set: str) -> str:
    if "@" not in feature_set:
        return "shared"
    position = feature_set.rsplit("@", 1)[1]
    return position if position in POSITIONS else f"legacy_{position}"


def _position_definition(position: str) -> str:
    if position == "prompt_last_token":
        return (
            "Final causal state of the complete prompt; it predicts the first "
            "generated response token."
        )
    if position == "question_object_pre_token":
        return (
            "Object word at its actual position in the question; contextual "
            "first object subtoken target and the immediately preceding "
            "causal state."
        )
    return "Shared or legacy feature without a current explicit position."


def _probe_family(feature_set: str) -> str:
    block = feature_set.rsplit("@", 1)[0]
    if block in {"ads", "cgc", "ads+cgc"}:
        return "ads_cgc"
    return "dgst_method"


def _first_stat(
    payload: Mapping[str, Any],
    paths: Sequence[str],
) -> dict[str, Any] | None:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return _normalize_stat(value)
    return None


def _normalize_stat(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        if "mean" not in value:
            raise ValueError(f"Metric mapping has no mean: {value!r}")
        mean = _finite_float(value["mean"])
        std = (
            _finite_float(value["std"])
            if value.get("std") is not None
            else None
        )
        result = {"mean": mean, "std": std}
        if value.get("values") is not None:
            result["values"] = [
                _finite_float(item) for item in value["values"]
            ]
        return result
    return {"mean": _finite_float(value), "std": None}


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric sequence")
    clean = [_finite_float(value) for value in values]
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "values": clean,
    }


def _required_float(
    mapping: Mapping[str, Any], keys: Sequence[str]
) -> float:
    for key in keys:
        if key in mapping:
            return _finite_float(mapping[key])
    raise ValueError(f"Missing required metric {list(keys)}")


def _nested(mapping: Mapping[str, Any], path: str) -> Any:
    if path in mapping:
        return mapping[path]
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _validate_exact_scope(
    mapping: Mapping[str, Any],
    key: str,
    expected: str,
    source: str,
) -> None:
    actual = mapping.get(key)
    if actual is None:
        raise ValueError(f"{source} has no {key} provenance")
    if str(actual) != expected:
        raise ValueError(
            f"{source} {key}={actual!r}, expected {expected!r}"
        )


def _normalize_expected_seeds(values: Sequence[int]) -> list[int]:
    seeds = [int(value) for value in values]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(
            f"QA comparison requires exactly three unique seeds, got {seeds}"
        )
    return seeds


def _validate_seed_list(
    actual: Any,
    expected: Sequence[int],
    source: str,
) -> None:
    if not isinstance(actual, Sequence) or isinstance(
        actual, (str, bytes)
    ):
        raise ValueError(f"{source} has no seed list")
    normalized = [int(value) for value in actual]
    if normalized != list(expected):
        raise ValueError(
            f"{source} seeds={normalized}, expected {list(expected)}"
        )


def _format_stat(value: Any) -> str:
    if not isinstance(value, Mapping) or value.get("mean") is None:
        return "—"
    mean = float(value["mean"])
    if value.get("std") is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {float(value['std']):.4f}"


def _format_coverage(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, Mapping):
        available = value.get("available", value.get("count"))
        total = value.get("total")
        rate = value.get("rate")
        if available is not None and total is not None:
            if rate is None and int(total) != 0:
                rate = float(available) / float(total)
            if rate is not None:
                return f"{int(available)}/{int(total)} ({100 * float(rate):.1f}%)"
            return f"{int(available)}/{int(total)}"
        if rate is not None:
            return f"{100 * float(rate):.1f}%"
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric must be finite, got {value!r}")
    return result


def _discover_probe_summary(
    run_dir: Path,
    label_protocol: str,
) -> Path:
    candidates = (
        run_dir / "results" / "summary_mean_std.json",
        run_dir / "results" / f"summary_mean_std_{label_protocol}.json",
        run_dir / "results" / label_protocol / "summary_mean_std.json",
    )
    return _first_existing(candidates, "QA probe summary")


def _discover_baseline_summary(
    run_dir: Path,
    model: str,
    dataset: str,
    label_protocol: str,
    *,
    trainer: str = "native_paper",
    num_seeds: int = 3,
) -> Path:
    normalized = str(trainer).strip().lower().replace("-", "_")
    if normalized in {"torch_mlp", "shared_mlp"}:
        normalized = "shared_torch_mlp"
    if normalized in {"native", "paper"}:
        normalized = "native_paper"
    base = run_dir / "baseline" / label_protocol / "results"
    if normalized == "shared_torch_mlp":
        stem = (
            f"{model}_{dataset}_{label_protocol}_qa_baselines_"
            f"shared_torch_mlp_{int(num_seeds)}seed.json"
        )
        candidates = (base / "shared_torch_mlp" / stem,)
    elif normalized == "native_paper":
        stem = (
            f"{model}_{dataset}_{label_protocol}_qa_baselines_"
            f"{int(num_seeds)}seed.json"
        )
        candidates = (base / stem,)
    else:
        raise ValueError(
            "qa_benchmarks.baseline_trainer must be native_paper or "
            f"shared_torch_mlp, got {trainer!r}"
        )
    return _first_existing(candidates, "configured baseline summary")


def _first_existing(
    paths: Sequence[Path], description: str
) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No {description} found. Checked: "
        + ", ".join(str(path) for path in paths)
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
