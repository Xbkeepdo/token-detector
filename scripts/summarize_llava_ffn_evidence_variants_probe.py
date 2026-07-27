#!/usr/bin/env python3
"""Summarize LLaVA FFN evidence-variant torch-probe results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


NEW_FEATURES = [
    "ffn_eiffrac_svd",
    "ffn_eifdose_svd",
    "ffn_eiffrac_pca",
    "ffn_eifdose_pca",
]

OLD_FFN_FEATURES = ["ffn_fad", "ffn_eifdose", "ffn_logitlift"]

DEFAULT_FEATURE_SETS = [
    "risk_geo_raw",
    "visualcosine_raw",
    "risk_geo_raw+visualcosine_raw",
    *OLD_FFN_FEATURES,
    *NEW_FEATURES,
    *[f"{feature}+visualcosine_raw" for feature in NEW_FEATURES],
    *[f"{feature}+risk_geo_raw+visualcosine_raw" for feature in NEW_FEATURES],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        default=(
            "outputs/llava_1_5_7b/COCO500-ffn-evidence-variants-visualonly-train80-test20/"
            "results/llava_1_5_7b_selected_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llava_ffn_injection_diagnostic/evidence_variants",
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

    write_csv(output_dir / "llava_ffn_evidence_variants_torch_probe.csv", rows)
    write_md(output_dir / "llava_ffn_evidence_variants_torch_probe_summary.md", rows)


def family_for_feature_set(feature_set: str) -> str:
    parts = set(feature_set.split("+"))
    if parts <= {"risk_geo_raw", "visualcosine_raw"}:
        return "baseline"
    if parts <= set(OLD_FFN_FEATURES):
        return "old_ffn_single"
    if parts <= set(NEW_FEATURES):
        return "new_evidence_single"
    if "visualcosine_raw" in parts and "risk_geo_raw" in parts and any(part in NEW_FEATURES for part in parts):
        return "new_evidence_plus_risk_cosine"
    if "visualcosine_raw" in parts and any(part in NEW_FEATURES for part in parts):
        return "new_evidence_plus_cosine"
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
        "# LLaVA FFN Evidence Variants Torch Probe Summary",
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
        "baseline",
        "old_ffn_single",
        "new_evidence_single",
        "new_evidence_plus_cosine",
        "new_evidence_plus_risk_cosine",
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
