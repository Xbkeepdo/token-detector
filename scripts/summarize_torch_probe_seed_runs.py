#!/usr/bin/env python3
"""Aggregate torch-probe feature-set metrics across random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("precision", "recall", "f1", "accuracy", "auc", "aupr")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
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
            row = {
                "model": model,
                "model_label": label,
                "feature_set": feature_set,
                "num_seeds": len(args.seeds),
                "seeds": list(args.seeds),
            }
            for metric in METRICS:
                values = [float(seed_metrics[seed][metric]) for seed in args.seeds]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.pstdev(values)
                row[f"{metric}_values"] = values
            model_rows.append(row)

        model_rows.sort(key=lambda item: (-item["auc_mean"], -item["f1_mean"], item["feature_set"]))
        for rank, row in enumerate(model_rows, start=1):
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
    for expected_seed, metrics in seed_metrics.items():
        missing = [metric for metric in METRICS if metric not in metrics]
        if missing:
            raise KeyError(f"{model}/{feature_set}/seed{expected_seed} missing metrics: {missing}")
        actual_seed = metrics.get("best_params", {}).get("seed")
        if actual_seed is not None and int(actual_seed) != int(expected_seed):
            raise ValueError(
                f"{model}/{feature_set}: expected seed {expected_seed}, found {actual_seed}."
            )


def _mean_std(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f}+/-{row[f'{metric}_std']:.3f}"


def _write_markdown(path: Path, title: str, seeds: list[int], rows: list[dict]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`. Values are population mean+/-std.",
        "",
        "## Best by model",
        "",
        "| Model | Best feature set | PR | RC | F1 | AUC | AUPR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    model_order = []
    for row in rows:
        if row["model"] not in model_order:
            model_order.append(row["model"])
    for model in model_order:
        best = next(row for row in rows if row["model"] == model)
        lines.append(
            f"| {best['model_label']} | `{best['feature_set']}` | "
            f"{_mean_std(best, 'precision')} | {_mean_std(best, 'recall')} | "
            f"{_mean_std(best, 'f1')} | {_mean_std(best, 'auc')} | "
            f"{_mean_std(best, 'aupr')} |"
        )

    for model in model_order:
        model_rows = [row for row in rows if row["model"] == model]
        lines.extend(
            [
                "",
                f"## {model_rows[0]['model_label']}",
                "",
                "| Rank | Feature set | PR | RC | F1 | Acc | AUC | AUPR |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in model_rows:
            lines.append(
                f"| {row['rank']} | `{row['feature_set']}` | "
                f"{_mean_std(row, 'precision')} | {_mean_std(row, 'recall')} | "
                f"{_mean_std(row, 'f1')} | {_mean_std(row, 'accuracy')} | "
                f"{_mean_std(row, 'auc')} | {_mean_std(row, 'aupr')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["model", "model_label", "rank", "feature_set", "num_seeds", "seeds"]
    for metric in METRICS:
        fieldnames.extend((f"{metric}_mean", f"{metric}_std"))
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
