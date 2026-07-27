#!/usr/bin/env python3
"""Summarize LLaVA FFN injection torch-probe results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_FEATURE_SETS = [
    "ffn_fad",
    "ffn_eifdose",
    "ffn_logitlift",
    "risk_geo_raw",
    "visualcosine_raw",
    "risk_geo_raw+visualcosine_raw",
    "ffn_fad+visualcosine_raw",
    "ffn_eifdose+visualcosine_raw",
    "ffn_logitlift+visualcosine_raw",
    "ffn_eifdose+risk_geo_raw",
    "ffn_eifdose+risk_geo_raw+visualcosine_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        default=(
            "outputs/llava_1_5_7b/COCO500-ffn-injection-visualonly/"
            "results/llava_1_5_7b_selected_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_ffn_injection_diagnostic",
    )
    parser.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with results_path.open() as handle:
        results = json.load(handle)

    rows = []
    for feature_set in args.feature_sets:
        metrics = results.get(feature_set, {}).get("torch_probe")
        if metrics is None:
            continue
        rows.append(
            {
                "feature_set": feature_set,
                "family": family_for_feature_set(feature_set),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "auc": float(metrics.get("auc", 0.0)),
                "aupr": float(metrics.get("aupr", 0.0)),
                "best_epoch": int(metrics.get("best_epoch", -1)),
                "num_features": int(metrics.get("num_features", 0)),
            }
        )

    csv_path = output_dir / "llava_ffn_injection_torch_probe.csv"
    write_csv(csv_path, rows)
    write_md(output_dir / "llava_ffn_injection_torch_probe_summary.md", rows)


def family_for_feature_set(feature_set: str) -> str:
    parts = set(feature_set.split("+"))
    if parts <= {"ffn_fad", "ffn_eifdose", "ffn_logitlift"}:
        return "single_injection"
    if (
        "risk_geo_raw" in parts
        and "visualcosine_raw" in parts
        and any(part.startswith("ffn_") for part in parts)
    ):
        return "injection_plus_risk_cosine"
    if "visualcosine_raw" in parts and any(part.startswith("ffn_") for part in parts):
        return "injection_plus_cosine"
    if "risk_geo_raw" in parts and any(part.startswith("ffn_") for part in parts):
        return "injection_plus_risk"
    if parts <= {"risk_geo_raw", "visualcosine_raw"}:
        return "baseline"
    return "other"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "feature_set",
        "family",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "auc",
        "aupr",
        "best_epoch",
        "num_features",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# LLaVA FFN Injection Torch Probe Summary",
        "",
        "| Feature set | Family | AUC | F1 | AUPR | Precision | Recall | Num features |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (-float(item["auc"]), str(item["feature_set"]))):
        lines.append(
            "| `{feature_set}` | {family} | {auc:.6g} | {f1:.6g} | {aupr:.6g} | "
            "{precision:.6g} | {recall:.6g} | {num_features} |".format(**row)
        )
    lines.extend(["", "## Required Comparisons", ""])
    for family in (
        "single_injection",
        "baseline",
        "injection_plus_cosine",
        "injection_plus_risk",
        "injection_plus_risk_cosine",
    ):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        best = max(family_rows, key=lambda item: float(item["auc"]))
        lines.append(
            "- {family}: best `{feature_set}` with AUC={auc:.6g}, F1={f1:.6g}, AUPR={aupr:.6g}.".format(
                **best
            )
        )
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
