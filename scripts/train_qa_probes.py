#!/usr/bin/env python3
"""Train all fixed QA feature-set probes for seeds 42/43/44."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.qa_probe import (
    QA_LABEL_PROTOCOLS,
    QA_POSITIONS,
    aggregate_seed_results,
    default_feature_sets,
    feature_set_position,
    normalize_positions,
    safe_name,
    train_one_seed,
    validate_image_level_splits,
)
from data.qa_benchmark import load_jsonl
from features.qa_extractor import qa_label_fingerprint
from utils.config_utils import load_config, qa_extraction_family_flags


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=("pope", "clevr_exist_5k", "amber_discriminative"),
        required=True,
    )
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--feature-sets", nargs="+")
    parser.add_argument("--label-protocols", nargs="+", choices=QA_LABEL_PROTOCOLS)
    parser.add_argument("--positions", nargs="+", choices=QA_POSITIONS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    family_flags = qa_extraction_family_flags(config)
    if not family_flags["method"] and not family_flags["ads_cgc"]:
        print(
            "[train_qa_probes] Skipped: QA extraction_mode="
            f"{family_flags['mode']} has no root probe features."
        )
        return
    qa_benchmark_cfg = config.get("qa_benchmarks") or {}
    training_cfg = config.get("training") or {}
    cfg = dict(training_cfg.get("torch_probe") or {})
    cfg["split_protocol"] = str(
        training_cfg.get("split_protocol", "strict_82_no_validation")
    )
    cfg["threshold_selection"] = str(
        training_cfg.get("threshold_selection", "train_f1")
    )
    probe_overrides = qa_benchmark_cfg.get("probe_overrides") or {}
    if not isinstance(probe_overrides, dict):
        raise ValueError("qa_benchmarks.probe_overrides must be a YAML mapping")
    cfg.update(probe_overrides)
    if cfg["split_protocol"] != "strict_82_no_validation":
        raise ValueError(
            "QA training requires training.split_protocol="
            "strict_82_no_validation"
        )
    if cfg["threshold_selection"] != "train_f1":
        raise ValueError(
            "QA training requires training.threshold_selection=train_f1"
        )
    if not cfg:
        raise ValueError("training.torch_probe is required for QA probe training")
    seeds = tuple(dict.fromkeys(
        int(value) for value in (args.seeds or cfg.get("seeds", [42, 43, 44]))
    ))
    if not seeds:
        raise ValueError("At least one QA probe seed is required")
    positions = normalize_positions(
        args.positions or qa_benchmark_cfg.get("position_protocols")
    )
    label_protocols = tuple(dict.fromkeys(
        str(value) for value in (
            args.label_protocols
            or qa_benchmark_cfg.get("label_protocols", QA_LABEL_PROTOCOLS)
        )
    ))
    unknown_protocols = sorted(set(label_protocols) - set(QA_LABEL_PROTOCOLS))
    if unknown_protocols:
        raise ValueError(f"Unknown QA label protocols: {unknown_protocols}")
    if args.feature_sets:
        feature_sets = list(args.feature_sets)
    else:
        feature_sets = []
        for feature_set in default_feature_sets(args.dataset, positions):
            block = feature_set.rsplit("@", 1)[0]
            is_ads_cgc = block in {"ads", "cgc", "ads+cgc"}
            if is_ads_cgc and family_flags["ads_cgc"]:
                feature_sets.append(feature_set)
            elif not is_ads_cgc and family_flags["method"]:
                feature_sets.append(feature_set)
    if not feature_sets:
        raise ValueError("No QA probe feature sets remain for the selected mode")
    for feature_set in feature_sets:
        block = feature_set.rsplit("@", 1)[0]
        if block in {"ads", "cgc", "ads+cgc"} and not family_flags["ads_cgc"]:
            raise ValueError(
                f"Feature set {feature_set!r} requires QA ADS+CGC extraction"
            )
        if block not in {"ads", "cgc", "ads+cgc"} and not family_flags["method"]:
            raise ValueError(
                f"Feature set {feature_set!r} requires QA method extraction"
            )
    _validate_feature_positions(feature_sets, positions)
    legacy_dataset_cfg = config.get("dataset") or {}
    output_root = (
        args.output_root
        or qa_benchmark_cfg.get("output_root")
        or legacy_dataset_cfg.get("output_root")
    )
    if not output_root:
        raise ValueError("QA output_root is missing from CLI and qa_benchmarks config")
    run_root = Path(output_root) / args.model / args.dataset
    training_input_fingerprint, training_provenance = (
        _qa_training_input_fingerprint(run_root, cfg, positions)
    )
    with open(run_root / "features.pkl", "rb") as handle:
        rows = pickle.load(handle)
    label_rows = load_jsonl(run_root / "labels.jsonl")
    labels = {row["key"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("labels.jsonl contains duplicate QA keys")
    with open(run_root / "image_splits.json", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    _validate_training_artifacts(rows, labels, split_manifest)
    # Reporting-only metadata is joined in memory and never stored in features.pkl
    # or consumed by feature_vector, preventing GT feature leakage.
    rows = [
        {**row, "report_gt_answer": labels.get(row["key"], {}).get("gt_answer")}
        for row in rows
    ]
    image_counts = validate_image_level_splits(rows)
    results_root = run_root / "results"
    protocol_summaries = {}

    for label_protocol in label_protocols:
        protocol_summary = {}
        for feature_set in feature_sets:
            position = feature_set_position(feature_set)
            seed_results = []
            for seed in seeds:
                seed_dir = (
                    results_root
                    / label_protocol
                    / position
                    / safe_name(feature_set)
                    / f"seed_{seed}"
                )
                result_path = seed_dir / "result.json"
                if result_path.exists() and not args.force:
                    with open(result_path, encoding="utf-8") as handle:
                        result = json.load(handle)
                    _validate_reusable_result(
                        result,
                        feature_set,
                        int(seed),
                        label_protocol,
                        position,
                        training_input_fingerprint,
                    )
                else:
                    result = train_one_seed(
                        rows,
                        feature_set,
                        int(seed),
                        str(seed_dir),
                        cfg,
                        args.device,
                        label_protocol=label_protocol,
                    )
                    result["training_input_fingerprint"] = (
                        training_input_fingerprint
                    )
                    result["training_provenance"] = training_provenance
                    _atomic_json(result_path, result)
                seed_results.append(result)
                metrics = result["test_metrics"]
                print(
                    f"[{label_protocol}/{position}/{feature_set}] seed={seed} "
                    f"AUROC={metrics['auroc']:.4f} "
                    f"real-F1={metrics['real']['f1']:.4f} "
                    f"hallucination-F1={metrics['hallucination']['f1']:.4f}"
                )
            protocol_summary[feature_set] = aggregate_seed_results(seed_results)
        protocol_summaries[label_protocol] = protocol_summary
        protocol_root = results_root / label_protocol
        _atomic_json(protocol_root / "summary_mean_std.json", protocol_summary)
        write_markdown_summary(
            protocol_root / "summary_mean_std.md",
            label_protocol,
            protocol_summary,
        )

    overall = {
        "schema_version": 2,
        "positive_class": "real",
        "seeds": list(seeds),
        "positions": list(positions),
        "image_counts": image_counts,
        "training_input_fingerprint": training_input_fingerprint,
        "training_provenance": training_provenance,
        "label_protocols": protocol_summaries,
    }
    summary_path = results_root / "summary_mean_std.json"
    _atomic_json(summary_path, overall)
    write_overall_markdown(results_root / "summary_mean_std.md", protocol_summaries)
    print(f"[train_qa_probes] Summary: {summary_path}")


def _validate_training_artifacts(
    rows: list[dict], labels: dict[str, dict], split_manifest: dict
) -> None:
    """Reject stale embedded labels/splits before fitting any probe."""

    row_keys = [str(row.get("key")) for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("features.pkl contains duplicate QA keys")
    split_ids = {}
    for split in ("train", "val", "test"):
        values = split_manifest.get(split)
        if not isinstance(values, list):
            raise ValueError(f"image_splits.json is missing list {split!r}")
        split_ids[split] = {int(value) for value in values}

    # The label artifact is authoritative for split/identity.  Validate every
    # labeled question against image_splits, including rows that a later
    # label protocol may filter out.
    for key, label in labels.items():
        split = str(label.get("probe_split") or "")
        if split not in split_ids:
            raise ValueError(f"Label {key!r} has invalid probe_split {split!r}")
        try:
            image_id = int(label["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Label {key!r} has invalid image_id") from exc
        if image_id not in split_ids[split]:
            raise ValueError(
                f"Label {key!r} image {image_id} is absent from split {split!r}"
            )

    compared_fields = (
        "label",
        "answer_correctness_all_label",
        "object_hallucination_yes_only_label",
        "prediction",
        "class_name",
        "error_type",
        "probe_split",
        "image_id",
        "source_split",
    )
    for row in rows:
        key = str(row.get("key"))
        label = labels.get(key)
        if label is None:
            raise ValueError(f"Feature {key!r} has no matching labels.jsonl row")
        mismatches = {
            field: {"feature": row.get(field), "label": label.get(field)}
            for field in compared_fields
            if row.get(field) != label.get(field)
        }
        if mismatches:
            raise ValueError(
                f"Feature {key!r} embeds stale label/split metadata: {mismatches}"
            )
        if row.get("label_fingerprint") != qa_label_fingerprint(label):
            raise ValueError(
                f"Feature {key!r} label fingerprint does not match labels.jsonl"
            )


def _qa_training_input_fingerprint(
    run_root: Path,
    cfg: dict,
    positions: tuple[str, ...],
) -> tuple[str, dict]:
    """Fingerprint features, labels, split, probe config, and trainer code."""

    required = {
        "features.pkl": run_root / "features.pkl",
        "labels.jsonl": run_root / "labels.jsonl",
        "image_splits.json": run_root / "image_splits.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"QA training inputs are incomplete under {run_root}: {missing}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    code_paths = {
        "detection/qa_probe.py": repo_root / "detection/qa_probe.py",
        "scripts/train_qa_probes.py": Path(__file__).resolve(),
    }
    provenance = {
        "schema_version": "qa-probe-training-provenance-v1",
        "artifact_sha256": {
            name: _sha256_file(path) for name, path in required.items()
        },
        "qa_probe_cfg": cfg,
        "positions": list(positions),
        "code_sha256": {
            name: _sha256_file(path) for name, path in code_paths.items()
        },
    }
    encoded = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), provenance


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_feature_positions(feature_sets, positions) -> None:
    selected = set(positions)
    for feature_set in feature_sets:
        position = feature_set_position(feature_set)
        if position in QA_POSITIONS and position not in selected:
            raise ValueError(
                f"Feature set {feature_set!r} uses unselected position {position!r}"
            )


def _validate_reusable_result(
    result: dict,
    feature_set: str,
    seed: int,
    label_protocol: str,
    position: str,
    training_input_fingerprint: str,
) -> None:
    expected = {
        "feature_set": feature_set,
        "seed": seed,
        "label_protocol": label_protocol,
        "position": position,
        "positive_class": "real",
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "last_epoch",
        "threshold_selection": "train_f1",
        "training_input_fingerprint": training_input_fingerprint,
    }
    mismatches = {
        key: {"expected": value, "found": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse an incompatible QA probe result; pass --force "
            f"to retrain. Mismatches: {mismatches}"
        )


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _metric(summary: dict, path: str) -> str:
    value = summary[path]
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def _markdown_lines(label_protocol: str, summaries: dict) -> list[str]:
    lines = [
        f"# QA probe results: {label_protocol}",
        "",
        "Labels use 0=hallucination and 1=real; the headline positive class is real.",
        "Strict 8:2 with no validation: fixed epochs, final checkpoint, and a Real-F1 threshold selected on train only; test is final evaluation only.",
        "",
        "| Feature set | Position | Test n | AUROC | Real AUPR | Real P | Real R | Real F1 | Hall. AUPR | Hall. P | Hall. R | Hall. F1 | Accuracy | Balanced acc. | Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature_set, summary in summaries.items():
        escaped = str(feature_set).replace("|", "\\|")
        counts = summary.get("counts") or {}
        lines.append(
            "| "
            + " | ".join((
                escaped,
                str(summary.get("position")),
                str(counts.get("test", "")),
                _metric(summary, "auroc"),
                _metric(summary, "real_aupr"),
                _metric(summary, "real.precision"),
                _metric(summary, "real.recall"),
                _metric(summary, "real.f1"),
                _metric(summary, "hallucination_aupr"),
                _metric(summary, "hallucination.precision"),
                _metric(summary, "hallucination.recall"),
                _metric(summary, "hallucination.f1"),
                _metric(summary, "accuracy"),
                _metric(summary, "balanced_accuracy"),
                _metric(summary, "macro_f1"),
            ))
            + " |"
        )
    lines.extend(("", "All values are mean ± population standard deviation across seeds.", ""))
    return lines


def write_markdown_summary(path: Path, label_protocol: str, summaries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(_markdown_lines(label_protocol, summaries)),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_overall_markdown(path: Path, protocol_summaries: dict) -> None:
    lines = ["# Unified QA probe summary", ""]
    for label_protocol, summaries in protocol_summaries.items():
        lines.extend(_markdown_lines(label_protocol, summaries))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
