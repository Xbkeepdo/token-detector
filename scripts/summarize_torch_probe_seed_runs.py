#!/usr/bin/env python3
"""Aggregate torch-probe feature-set metrics across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("precision", "recall", "f1", "accuracy", "auc", "aupr")
CLASS_METRICS = ("precision", "recall", "f1", "auc", "aupr")
CLASS_PREFIXES = ("real_positive", "hallucination_positive")
METRIC_LABELS = {
    "precision": "PR",
    "recall": "RC",
    "f1": "F1",
    "accuracy": "Acc",
    "auc": "AUC",
    "aupr": "AUPR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--model-labels", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument(
        "--run-template",
        required=True,
        help=(
            "Result path template containing {model} and {seed}, for example "
            "outputs/{model}/COCO4000-all/experiment/seed{seed}/results/"
            "{model}_selected_feature_sets.json"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output path without extension; .md, .csv and .json are written.",
    )
    parser.add_argument("--title", default="Torch MLP three-seed summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.model_labels or args.models
    if len(labels) != len(args.models):
        raise ValueError("--model-labels must have the same length as --models.")

    rows = []
    raw = {}
    for model, label in zip(args.models, labels):
        seed_results = {}
        for seed in args.seeds:
            path = Path(args.run_template.format(model=model, seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"Missing seed result: {path}")
            with path.open("r", encoding="utf-8") as handle:
                seed_results[seed] = json.load(handle)

        feature_sets = _common_torch_feature_sets(seed_results)
        if not feature_sets:
            raise ValueError(f"No common torch_probe feature sets found for {model}.")

        model_rows = []
        for feature_set in feature_sets:
            seed_metrics = {
                seed: seed_results[seed][feature_set]["torch_probe"]
                for seed in args.seeds
            }
            _validate_seed_metadata(model, feature_set, seed_metrics)
            reporting = tuple(
                seed_metrics[args.seeds[0]].get("threshold_reporting") or ()
            ) or ("train_f1",)
            for threshold_mode in reporting:
                metrics_by_seed = {
                    seed: (
                        seed_metrics[seed]["threshold_reports"][threshold_mode]
                        ["test_metrics"]
                        if seed_metrics[seed].get("threshold_reports")
                        else seed_metrics[seed]
                    )
                    for seed in args.seeds
                }
                threshold_values = [
                    float(
                        seed_metrics[seed]["threshold_reports"][threshold_mode]
                        ["threshold"]
                    )
                    if seed_metrics[seed].get("threshold_reports")
                    else float(seed_metrics[seed]["decision_threshold"])
                    for seed in args.seeds
                ]
                row = {
                    "model": model,
                    "model_label": label,
                    "feature_set": feature_set,
                    "threshold_mode": threshold_mode,
                    "threshold_mean": statistics.fmean(threshold_values),
                    "threshold_std": statistics.pstdev(threshold_values),
                    "threshold_values": threshold_values,
                    "num_seeds": len(args.seeds),
                    "seeds": list(args.seeds),
                }
                for metric in METRICS:
                    values = [
                        float(metrics_by_seed[seed][metric])
                        for seed in args.seeds
                    ]
                    row[f"{metric}_mean"] = statistics.fmean(values)
                    row[f"{metric}_std"] = statistics.pstdev(values)
                    row[f"{metric}_values"] = values
                row["headline_positive_class"] = str(
                    metrics_by_seed[args.seeds[0]]["reported_positive_class"]
                )
                for class_prefix in CLASS_PREFIXES:
                    for metric in CLASS_METRICS:
                        values = [
                            float(metrics_by_seed[seed][class_prefix][metric])
                            for seed in args.seeds
                        ]
                        key = f"{class_prefix}_{metric}"
                        row[f"{key}_mean"] = statistics.fmean(values)
                        row[f"{key}_std"] = statistics.pstdev(values)
                        row[f"{key}_values"] = values
                model_rows.append(row)

        model_rows.sort(
            key=lambda item: (
                ("fixed_0.5", "train_f1").index(item["threshold_mode"]),
                -item["real_positive_auc_mean"],
                -item["real_positive_f1_mean"],
                item["feature_set"],
            )
        )
        for threshold_mode in ("fixed_0.5", "train_f1"):
            mode_rows = [
                row for row in model_rows
                if row["threshold_mode"] == threshold_mode
            ]
            for rank, row in enumerate(mode_rows, start=1):
                row["rank"] = rank
        rows.extend(model_rows)
        raw[model] = model_rows

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(prefix.with_suffix(".md"), args.title, args.seeds, rows)
    _write_csv(prefix.with_suffix(".csv"), rows)
    _write_json(prefix.with_suffix(".json"), args.title, args.seeds, raw)
    print(f"[SeedSummary] Wrote {prefix.with_suffix('.md')}")
    print(f"[SeedSummary] Wrote {prefix.with_suffix('.csv')}")
    print(f"[SeedSummary] Wrote {prefix.with_suffix('.json')}")


def _common_torch_feature_sets(seed_results: dict[int, dict]) -> list[str]:
    common = None
    for result in seed_results.values():
        names = {
            name
            for name, classifiers in result.items()
            if isinstance(classifiers, dict) and isinstance(classifiers.get("torch_probe"), dict)
        }
        common = names if common is None else common & names
    return sorted(common or set())


def _validate_seed_metadata(model: str, feature_set: str, seed_metrics: dict[int, dict]) -> None:
    positive_classes = set()
    for expected_seed, metrics in seed_metrics.items():
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise KeyError(f"{model}/{feature_set}/seed{expected_seed} missing metrics: {missing}")
        for class_prefix in CLASS_PREFIXES:
            class_metrics = metrics.get(class_prefix)
            if not isinstance(class_metrics, dict):
                raise KeyError(
                    f"{model}/{feature_set}/seed{expected_seed} missing "
                    f"{class_prefix} metrics"
                )
            class_missing = [
                metric for metric in CLASS_METRICS if metric not in class_metrics
            ]
            if class_missing:
                raise KeyError(
                    f"{model}/{feature_set}/seed{expected_seed} "
                    f"{class_prefix} missing metrics: {class_missing}"
                )
        positive_classes.add(str(metrics.get("reported_positive_class")))
        selection = (
            metrics.get("split_protocol"),
            metrics.get("checkpoint_selection"),
            metrics.get("threshold_selection"),
        )
        if selection != (
            "strict_82_no_validation",
            "minimum_train_loss",
            "train_f1",
        ):
            raise ValueError(
                f"{model}/{feature_set}/seed{expected_seed}: incompatible "
                f"selection protocol {selection}"
            )
        reporting = tuple(metrics.get("threshold_reporting") or ())
        if reporting != ("fixed_0.5", "train_f1"):
            raise ValueError(
                f"{model}/{feature_set}/seed{expected_seed}: invalid "
                f"threshold reporting {reporting}"
            )
        reports = metrics.get("threshold_reports") or {}
        if tuple(reports) != reporting:
            raise ValueError(
                f"{model}/{feature_set}/seed{expected_seed}: missing dual "
                "threshold reports"
            )
        actual_seed = metrics.get("best_params", {}).get("seed")
        if actual_seed is not None and int(actual_seed) != int(expected_seed):
            raise ValueError(
                f"{model}/{feature_set}: expected seed {expected_seed}, found {actual_seed}."
            )
    if positive_classes != {"real"}:
        raise ValueError(
            f"{model}/{feature_set}: summary headline must be real-positive; "
            f"found {sorted(positive_classes)}."
        )


def _mean_std(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f}+/-{row[f'{metric}_std']:.3f}"


def _write_markdown(path: Path, title: str, seeds: list[int], rows: list[dict]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`. Values are population mean+/-std.",
        "Real is the headline positive class; hallucination-positive metrics are reported alongside it.",
        "Strict 8:2 has no validation set: train-loss early stopping restores the minimum-train-loss checkpoint.",
        "Every checkpoint is reported twice: fixed threshold 0.5 and a Real-F1 threshold selected on train only.",
        "Real/Hall AUC values are equal under score inversion, while AUPR differs; hallucination metrics use the complementary prediction at the same fixed boundary.",
        "",
        "## Best by model",
        "",
        "| Model | Threshold mode | Best feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    model_order = []
    for row in rows:
        if row["model"] not in model_order:
            model_order.append(row["model"])
    for model in model_order:
        for threshold_mode in ("fixed_0.5", "train_f1"):
            best = next(
                row for row in rows
                if row["model"] == model
                and row["threshold_mode"] == threshold_mode
            )
            lines.append(
                f"| {best['model_label']} | {threshold_mode} | "
                f"`{best['feature_set']}` | "
                f"{_mean_std(best, 'threshold')} | "
                f"{_mean_std(best, 'accuracy')} | "
                f"{_mean_std(best, 'real_positive_precision')} | "
                f"{_mean_std(best, 'real_positive_recall')} | "
                f"{_mean_std(best, 'real_positive_f1')} | "
                f"{_mean_std(best, 'real_positive_auc')} | "
                f"{_mean_std(best, 'real_positive_aupr')} | "
                f"{_mean_std(best, 'hallucination_positive_precision')} | "
                f"{_mean_std(best, 'hallucination_positive_recall')} | "
                f"{_mean_std(best, 'hallucination_positive_f1')} | "
                f"{_mean_std(best, 'hallucination_positive_auc')} | "
                f"{_mean_std(best, 'hallucination_positive_aupr')} |"
            )

    for model in model_order:
        model_rows = [row for row in rows if row["model"] == model]
        lines.extend(
            [
                "",
                f"## {model_rows[0]['model_label']}",
                "",
                "| Threshold mode | Rank | Feature set | Threshold | Acc | Real PR | Real RC | Real F1 | Real AUC | Real AUPR | Hall. PR | Hall. RC | Hall. F1 | Hall. AUC | Hall. AUPR |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in model_rows:
            lines.append(
                f"| {row['threshold_mode']} | {row['rank']} | `{row['feature_set']}` | "
                f"{_mean_std(row, 'threshold')} | "
                f"{_mean_std(row, 'accuracy')} | "
                f"{_mean_std(row, 'real_positive_precision')} | "
                f"{_mean_std(row, 'real_positive_recall')} | "
                f"{_mean_std(row, 'real_positive_f1')} | "
                f"{_mean_std(row, 'real_positive_auc')} | "
                f"{_mean_std(row, 'real_positive_aupr')} | "
                f"{_mean_std(row, 'hallucination_positive_precision')} | "
                f"{_mean_std(row, 'hallucination_positive_recall')} | "
                f"{_mean_std(row, 'hallucination_positive_f1')} | "
                f"{_mean_std(row, 'hallucination_positive_auc')} | "
                f"{_mean_std(row, 'hallucination_positive_aupr')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "model_label",
        "rank",
        "feature_set",
        "threshold_mode",
        "threshold_mean",
        "threshold_std",
        "num_seeds",
        "seeds",
        "headline_positive_class",
    ]
    for metric in METRICS:
        fieldnames.extend((f"{metric}_mean", f"{metric}_std"))
    for class_prefix in CLASS_PREFIXES:
        for metric in CLASS_METRICS:
            key = f"{class_prefix}_{metric}"
            fieldnames.extend((f"{key}_mean", f"{key}_std"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fieldnames if key in row}
            output["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            writer.writerow(output)


def _write_json(path: Path, title: str, seeds: list[int], raw: dict) -> None:
    payload = {"title": title, "seeds": seeds, "models": raw}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
