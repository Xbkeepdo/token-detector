#!/usr/bin/env python3
"""Summarize shared-MLP baseline tests under both positive-class views."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = (
    ("metatoken", "MetaToken-MLP"),
    ("svar", "SVAR"),
    ("projectaway", "ProjectAway"),
)
THRESHOLD_MODES = ("train_f1", "fixed_0.5")
CLASS_KEYS = (
    ("real_positive", "real"),
    ("hallucination_positive", "hall"),
)
CLASS_METRICS = ("precision", "recall", "f1", "aupr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-results", nargs="+", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--title",
        default="Shared three-layer MLP baseline dual-positive report",
    )
    return parser.parse_args()


def _method_result(payload: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    result = payload["methods"][method]
    if method == "metatoken":
        result = result["shared_mlp"]
    if not isinstance(result, Mapping):
        raise ValueError(f"Invalid result payload for {method}")
    return result


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list")
    numeric = [float(value) for value in values]
    return {
        "mean": statistics.fmean(numeric),
        "std": statistics.pstdev(numeric),
        "values": numeric,
    }


def _format(metric: Mapping[str, Any]) -> str:
    return f"{float(metric['mean']):.4f} ± {float(metric['std']):.4f}"


def main() -> None:
    args = parse_args()
    source_paths = [Path(value) for value in args.seed_results]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    seeds = [int(payload["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds: {seeds}")
    for payload in payloads:
        if payload.get("trainer") != "shared_torch_mlp":
            raise ValueError("All inputs must be shared_torch_mlp results")
        if payload.get("headline_positive_class") != "real":
            raise ValueError("Expected Real-F1 threshold selection in every input")
        if payload.get("split_protocol") != "strict_82_no_validation":
            raise ValueError("Expected strict_82_no_validation in every input")

    rows: list[dict[str, Any]] = []
    for method, display_name in METHODS:
        results = [_method_result(payload, method) for payload in payloads]
        input_dims = {int(result["input_dim"]) for result in results}
        if len(input_dims) != 1:
            raise ValueError(f"Inconsistent {method} input dimensions: {input_dims}")
        for threshold_mode in THRESHOLD_MODES:
            reports = [result["threshold_reports"][threshold_mode] for result in results]
            test_metrics = [report["test_metrics"] for report in reports]
            row: dict[str, Any] = {
                "method": method,
                "display_name": display_name,
                "input_dim": next(iter(input_dims)),
                "threshold_mode": threshold_mode,
                "threshold": _aggregate([report["threshold"] for report in reports]),
                "accuracy": _aggregate([metrics["accuracy"] for metrics in test_metrics]),
                "auroc": _aggregate(
                    [metrics["real_positive"]["auc"] for metrics in test_metrics]
                ),
            }
            for source_key, output_prefix in CLASS_KEYS:
                for metric in CLASS_METRICS:
                    row[f"{output_prefix}_{metric}"] = _aggregate(
                        [metrics[source_key][metric] for metrics in test_metrics]
                    )
            rows.append(row)

    first = payloads[0]
    result = {
        "title": args.title,
        "model": first.get("model"),
        "trainer": "shared_torch_mlp",
        "seeds": seeds,
        "std_definition": "population_std_ddof_0",
        "split_protocol": "strict_82_no_validation",
        "image_split_counts": first.get("image_split_counts"),
        "token_counts": first.get("counts"),
        "checkpoint_selection": first.get("checkpoint_selection"),
        "threshold_selection": "train_real_f1",
        "hall_metrics_note": (
            "Hall-positive metrics use inverted Real scores and complementary "
            "predictions at the same reported threshold."
        ),
        "source_paths": [str(path) for path in source_paths],
        "rows": rows,
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(prefix.with_suffix(".csv"), rows, seeds)
    write_markdown(prefix.with_suffix(".md"), result)
    for suffix in (".md", ".csv", ".json"):
        print(f"Wrote {prefix.with_suffix(suffix)}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> None:
    metric_names = (
        "threshold",
        "accuracy",
        "auroc",
        "real_precision",
        "real_recall",
        "real_f1",
        "real_aupr",
        "hall_precision",
        "hall_recall",
        "hall_f1",
        "hall_aupr",
    )
    fields = ["method", "display_name", "input_dim", "threshold_mode", "seeds"]
    for metric in metric_names:
        fields.extend((f"{metric}_mean", f"{metric}_std"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {
                "method": row["method"],
                "display_name": row["display_name"],
                "input_dim": row["input_dim"],
                "threshold_mode": row["threshold_mode"],
                "seeds": ",".join(str(seed) for seed in seeds),
            }
            for metric in metric_names:
                output[f"{metric}_mean"] = row[metric]["mean"]
                output[f"{metric}_std"] = row[metric]["std"]
            writer.writerow(output)


def write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        f"# {result['title']}",
        "",
        f"- Model: `{result['model']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in result['seeds'])}`",
        "- Network: shared `128→64→32` Torch MLP.",
        "- Split: strict image-level 8:2, no validation; test is evaluation-only.",
        "- Checkpoint: minimum train loss; train-F1 threshold is selected on train only.",
        "- Values: three-seed population mean ± population standard deviation.",
        f"- Hall metric protocol: {result['hall_metrics_note']}",
        "",
    ]
    for threshold_mode, title in (
        ("train_f1", "Test — train Real-F1 threshold"),
        ("fixed_0.5", "Test — fixed threshold 0.5"),
    ):
        lines.extend((
            f"## {title}",
            "",
            "| Method | Dim | Threshold | Accuracy | AUROC | Real P | Real R | Real F1 | Real AUPR | Hall P | Hall R | Hall F1 | Hall AUPR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ))
        for row in result["rows"]:
            if row["threshold_mode"] != threshold_mode:
                continue
            lines.append(
                "| {display_name} | {input_dim} | {threshold} | {accuracy} | "
                "{auroc} | {real_precision} | {real_recall} | {real_f1} | "
                "{real_aupr} | {hall_precision} | {hall_recall} | {hall_f1} | "
                "{hall_aupr} |".format(
                    display_name=row["display_name"],
                    input_dim=row["input_dim"],
                    **{
                        key: _format(row[key])
                        for key in (
                            "threshold",
                            "accuracy",
                            "auroc",
                            "real_precision",
                            "real_recall",
                            "real_f1",
                            "real_aupr",
                            "hall_precision",
                            "hall_recall",
                            "hall_f1",
                            "hall_aupr",
                        )
                    },
                )
            )
        lines.append("")
    lines.extend(("## Source seed results", ""))
    lines.extend(f"- `{source}`" for source in result["source_paths"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
