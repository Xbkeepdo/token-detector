#!/usr/bin/env python3
"""Create simple metric tables from selected feature-set results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


METRIC_ORDER = ["precision", "recall", "f1", "accuracy", "auc"]
CLASS_METRIC_ORDER = ["precision", "recall", "f1", "auc", "aupr"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "accuracy": "Accuracy",
    "auc": "AUC",
}
CLASSIFIER_ORDER = ["xgb", "rf", "mlp", "torch_probe"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        required=True,
        help="Path to <model>_selected_feature_sets.json.",
    )
    parser.add_argument(
        "--format",
        choices=["md", "latex", "both"],
        default="md",
        help="Table format to write.",
    )
    parser.add_argument(
        "--output-path",
        help="Output path. Only valid when --format is md or latex.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Write the table without printing it to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = ("md", "latex") if args.format == "both" else (args.format,)
    paths = write_summary_tables(
        args.results,
        formats=formats,
        output_path=args.output_path,
        print_table=not args.no_print,
    )
    for path in paths:
        print(f"[SummaryTable] Saved {path}")


def write_summary_tables(
    results_path: str | Path,
    *,
    formats: Iterable[str] = ("md",),
    output_path: str | Path | None = None,
    print_table: bool = True,
) -> list[Path]:
    results_path = Path(results_path)
    formats = tuple(formats)
    if output_path is not None and len(formats) != 1:
        raise ValueError("--output-path is only valid for a single output format.")

    with results_path.open("r", encoding="utf-8") as fh:
        results = json.load(fh)

    rows = list(_iter_rows(results))
    dual_class = any(row.get("dual_class_metrics", False) for row in rows)
    written = []
    for fmt in formats:
        if fmt == "md":
            table = _to_markdown(rows, dual_class=dual_class)
        elif fmt == "latex":
            table = _to_latex(rows, dual_class=dual_class)
        else:
            raise ValueError(f"Unsupported table format: {fmt}")

        if print_table:
            print(table)

        path = Path(output_path) if output_path is not None else _default_output_path(results_path, fmt)
        path.write_text(table + "\n", encoding="utf-8")
        written.append(path)
    return written


def _iter_rows(results: dict) -> Iterable[dict]:
    for feature_set, classifiers in results.items():
        if not isinstance(classifiers, dict):
            continue
        classifier_names = [name for name in CLASSIFIER_ORDER if name in classifiers]
        classifier_names.extend(name for name in classifiers if name not in classifier_names)
        for classifier in classifier_names:
            metrics = classifiers.get(classifier)
            if not isinstance(metrics, dict):
                continue
            row = {
                "feature_set": feature_set,
                "classifier": classifier,
            }
            for metric in METRIC_ORDER:
                row[metric] = metrics.get(metric)
            real_metrics = metrics.get("real_positive")
            hallucination_metrics = metrics.get("hallucination_positive")
            row["dual_class_metrics"] = isinstance(
                real_metrics, dict
            ) and isinstance(hallucination_metrics, dict)
            for prefix, class_metrics in (
                ("real", real_metrics),
                ("hallucination", hallucination_metrics),
            ):
                for metric in CLASS_METRIC_ORDER:
                    row[f"{prefix}_{metric}"] = (
                        class_metrics.get(metric)
                        if isinstance(class_metrics, dict)
                        else None
                    )
            yield row


def _to_markdown(rows: list[dict], *, dual_class: bool = False) -> str:
    if dual_class:
        headers = [
            "Feature Set",
            "Classifier",
            "Accuracy",
            "Real PR",
            "Real RC",
            "Real F1",
            "Real AUC",
            "Real AUPR",
            "Hall. PR",
            "Hall. RC",
            "Hall. F1",
            "Hall. AUC",
            "Hall. AUPR",
        ]
        metric_keys = [
            "accuracy",
            *[f"real_{metric}" for metric in CLASS_METRIC_ORDER],
            *[f"hallucination_{metric}" for metric in CLASS_METRIC_ORDER],
        ]
        align = ["---", "---"] + ["---:" for _ in metric_keys]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(align) + " |",
        ]
        for row in rows:
            values = [
                str(row["feature_set"]),
                str(row["classifier"]),
                *[_format_metric(row.get(key)) for key in metric_keys],
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    headers = ["Feature Set", "Classifier"] + [METRIC_LABELS[key] for key in METRIC_ORDER]
    align = ["---", "---"] + ["---:" for _ in METRIC_ORDER]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        values = [
            str(row["feature_set"]),
            str(row["classifier"]),
            *[_format_metric(row[key]) for key in METRIC_ORDER],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _to_latex(rows: list[dict], *, dual_class: bool = False) -> str:
    if dual_class:
        headers = [
            "Feature Set",
            "Classifier",
            "Accuracy",
            "Real PR",
            "Real RC",
            "Real F1",
            "Real AUC",
            "Real AUPR",
            "Hall. PR",
            "Hall. RC",
            "Hall. F1",
            "Hall. AUC",
            "Hall. AUPR",
        ]
        metric_keys = [
            "accuracy",
            *[f"real_{metric}" for metric in CLASS_METRIC_ORDER],
            *[f"hallucination_{metric}" for metric in CLASS_METRIC_ORDER],
        ]
        lines = [
            r"\begin{tabular}{llrrrrrrrrrrr}",
            r"\hline",
            " & ".join(_latex_escape(item) for item in headers) + r" \\",
            r"\hline",
        ]
        for row in rows:
            values = [
                _latex_escape(str(row["feature_set"])),
                _latex_escape(str(row["classifier"])),
                *[_format_metric(row.get(key)) for key in metric_keys],
            ]
            lines.append(" & ".join(values) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(lines)

    headers = ["Feature Set", "Classifier"] + [METRIC_LABELS[key] for key in METRIC_ORDER]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        " & ".join(_latex_escape(item) for item in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [
            _latex_escape(str(row["feature_set"])),
            _latex_escape(str(row["classifier"])),
            *[_format_metric(row[key]) for key in METRIC_ORDER],
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def _format_metric(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _default_output_path(results_path: Path, fmt: str) -> Path:
    suffix = "md" if fmt == "md" else "tex"
    return results_path.with_name(f"{results_path.stem}_table.{suffix}")


if __name__ == "__main__":
    main()
