#!/usr/bin/env python3
"""Prepare, extract, train, plot, and summarize the COCO100 gate comparison."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io_utils import load_pkl  # noqa: E402

CONFIG = ROOT / "configs" / "model_configs_coco100_gate_comparison.yaml"
MODELS = ("llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b")
STEPS = ("prepare", "extract", "probe_split", "train", "plot", "summarize")
NUM_IMAGES = 100
SEED = 42
PROBE_SPLIT_SEED = 1
PROBE_SEEDS = (42, 43, 44)
PROBE_RUN_DIR = "torchmlp-seed3-probe-split1-811"
PROBE_FEATURE_SETS = (
    "gate-relative-vll-risk",
    "gate-relative-vll-hprecosine",
    "gate-relative-vll-risk+gate-relative-vll-hprecosine",
    "gate-softmax-relative-vll-risk",
    "gate-softmax-relative-vll-hprecosine",
    "gate-softmax-relative-vll-risk+gate-softmax-relative-vll-hprecosine",
    "gate-legacy-prob-risk",
    "gate-legacy-prob-hprecosine",
    "gate-legacy-prob-risk+gate-legacy-prob-hprecosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(args.models)
    if "prepare" in args.steps:
        _prepare(models)

    for model in models:
        output_dir = _output_dir(model)
        if "extract" in args.steps:
            _require_prepared(output_dir)
            command = [
                args.python,
                "scripts/extract_features.py",
                "--model",
                model,
                "--config",
                str(CONFIG),
                "--output-dir",
                str(output_dir),
                "--num-images",
                str(NUM_IMAGES),
                "--seed",
                str(SEED),
                "--feature-devices",
                *args.devices,
            ]
            if args.resume:
                command.append("--resume")
            _run(*command)

    if "probe_split" in args.steps:
        _prepare_probe_split(models)

    for model in models:
        output_dir = _output_dir(model)
        if "train" in args.steps:
            _train_probes(
                python=args.python,
                model=model,
                output_dir=output_dir,
                devices=tuple(args.devices),
            )
        if "plot" in args.steps:
            _run(
                args.python,
                "scripts/plot_coco100_gate_comparison.py",
                "--model",
                model,
                "--output-dir",
                str(output_dir),
            )

    if "summarize" in args.steps:
        _summarize_probes(args.python, models)


def _prepare(models: tuple[str, ...]) -> None:
    source_records = {}
    label_id_order = None
    for model in models:
        source_dir = ROOT / "outputs" / model / "COCO4000"
        generations = _load_json(source_dir / "generations.json")
        labeling = _load_json(source_dir / "labeling.json")
        if len(generations) != 4000 or len(labeling) != 4000:
            raise ValueError(
                f"{model} COCO4000 must contain exactly 4000 generations and labels; "
                f"found {len(generations)} and {len(labeling)}."
            )

        current_label_order = [str(image_id) for image_id in labeling]
        if label_id_order is None:
            label_id_order = current_label_order
        elif current_label_order != label_id_order:
            mismatch = _first_mismatch(label_id_order, current_label_order)
            raise ValueError(
                "The selected models do not share the same COCO4000 labeling ID order"
                f" (first mismatch at index {mismatch})."
            )
        source_records[model] = (generations, labeling)

    if label_id_order is None or len(label_id_order) < NUM_IMAGES:
        raise ValueError(f"At least {NUM_IMAGES} common COCO4000 labels are required.")
    selected_ids = label_id_order[:NUM_IMAGES]
    if len(set(selected_ids)) != NUM_IMAGES:
        raise ValueError("The first 100 canonical COCO4000 label IDs are not unique.")

    split_ids = [int(image_id) for image_id in selected_ids]
    random.Random(SEED).shuffle(split_ids)
    split = {
        "train": split_ids[:80],
        "val": split_ids[80:90],
        "test": split_ids[90:100],
    }
    _validate_split(split, selected_ids)

    for model in models:
        generations, labeling = source_records[model]
        missing_generations = [
            image_id for image_id in selected_ids if image_id not in generations
        ]
        if missing_generations:
            raise KeyError(
                f"{model} generations.json is missing selected IDs: "
                f"{missing_generations[:5]}"
            )
        output_dir = _output_dir(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "generations.json",
            {image_id: generations[image_id] for image_id in selected_ids},
        )
        _write_json(
            output_dir / "labeling.json",
            {image_id: labeling[image_id] for image_id in selected_ids},
        )
        _write_json(output_dir / "image_splits.json", split)

    print(
        "[GateComparison] Prepared the shared canonical first 100 COCO4000 images "
        "with an 80/10/10 split."
    )


def _validate_split(split: dict[str, list[int]], selected_ids: list[str]) -> None:
    expected_sizes = {"train": 80, "val": 10, "test": 10}
    actual_sizes = {name: len(split[name]) for name in expected_sizes}
    if actual_sizes != expected_sizes:
        raise ValueError(f"Expected an 80/10/10 split, found {actual_sizes}.")
    partitions = {name: set(split[name]) for name in expected_sizes}
    if (
        partitions["train"] & partitions["val"]
        or partitions["train"] & partitions["test"]
        or partitions["val"] & partitions["test"]
    ):
        raise ValueError("COCO100 split partitions must be disjoint.")
    if set.union(*partitions.values()) != {int(image_id) for image_id in selected_ids}:
        raise ValueError("COCO100 split union must equal the selected 100 image IDs.")


def _prepare_probe_split(models: tuple[str, ...]) -> None:
    """Create one shared two-class 80/10/10 split for exploratory probes.

    The extraction split uses seed 42, whose Qwen validation partition has no
    hallucination rows.  Seed 1 is the first deterministic shuffle seed for the
    canonical image order that leaves both labels in every model's val/test
    partitions, avoiding the trainer's train-as-validation fallback.
    """
    canonical_ids = None
    rows_by_model = {}
    for model in models:
        output_dir = _output_dir(model)
        labeling = _load_json(output_dir / "labeling.json")
        current_ids = [str(image_id) for image_id in labeling]
        if len(current_ids) != NUM_IMAGES:
            raise ValueError(
                f"{model} must have {NUM_IMAGES} prepared labels, found {len(current_ids)}."
            )
        if canonical_ids is None:
            canonical_ids = current_ids
        elif current_ids != canonical_ids:
            mismatch = _first_mismatch(canonical_ids, current_ids)
            raise ValueError(
                "Probe split requires identical canonical image order across models "
                f"(first mismatch at index {mismatch})."
            )
        feature_path = output_dir / "features.pkl"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        rows_by_model[model] = load_pkl(feature_path)

    if canonical_ids is None:
        raise ValueError("Probe split requires at least one model.")
    shuffled_ids = [int(image_id) for image_id in canonical_ids]
    random.Random(PROBE_SPLIT_SEED).shuffle(shuffled_ids)
    split = {
        "train": shuffled_ids[:80],
        "val": shuffled_ids[80:90],
        "test": shuffled_ids[90:100],
    }
    _validate_split(split, canonical_ids)

    for model, rows in rows_by_model.items():
        summary = []
        for partition in ("train", "val", "test"):
            image_ids = set(split[partition])
            partition_rows = [
                row for row in rows if int(row["image_id"]) in image_ids
            ]
            label_counts = {
                label: sum(int(row["label"]) == label for row in partition_rows)
                for label in (0, 1)
            }
            if not all(label_counts.values()):
                raise ValueError(
                    f"{model} probe {partition} split is single-class: {label_counts}."
                )
            summary.append(
                f"{partition}=H{label_counts[0]}/N{label_counts[1]}"
            )
        _write_json(_output_dir(model) / "probe_image_splits.json", split)
        print(f"[GateComparison] Probe split {model}: {', '.join(summary)}")


def _train_probes(
    *,
    python: str,
    model: str,
    output_dir: Path,
    devices: tuple[str, ...],
) -> None:
    if not devices:
        raise ValueError("At least one probe device is required.")
    probe_split_path = output_dir / "probe_image_splits.json"
    if not probe_split_path.exists():
        raise FileNotFoundError(
            f"{probe_split_path} is missing; run --steps probe_split first."
        )

    jobs = []
    for index, seed in enumerate(PROBE_SEEDS):
        seed_dir = output_dir / PROBE_RUN_DIR / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _link_input(output_dir / "features.pkl", seed_dir / "features.pkl")
        _link_input(probe_split_path, seed_dir / "image_splits.json")
        command = [
            python,
            "scripts/train_torch_probe_feature_sets.py",
            "--model",
            model,
            "--config",
            str(CONFIG),
            "--output-dir",
            str(seed_dir),
            "--feature-sets",
            *PROBE_FEATURE_SETS,
            "--batch-size",
            "256",
            "--num-epochs",
            "100",
            "--hidden-sizes",
            "128",
            "64",
            "32",
            "--positive-class",
            "real",
            "--seed",
            str(seed),
            "--device",
            devices[index % len(devices)],
        ]
        log_path = seed_dir / "train.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(
            "[GateComparison]",
            " ".join(str(part) for part in command),
            f"> {log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        jobs.append((seed, process, log_handle, log_path))

    for seed, process, log_handle, log_path in jobs:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            raise subprocess.CalledProcessError(
                return_code,
                process.args,
                output=f"Seed {seed}; see {log_path}",
            )


def _summarize_probes(python: str, models: tuple[str, ...]) -> None:
    prefix = (
        ROOT
        / "outputs"
        / "coco100-gate-comparison-summary"
        / "three_model_seed3_probe_split1_summary"
    )
    _run(
        python,
        "scripts/summarize_torch_probe_seed_runs.py",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in PROBE_SEEDS),
        "--run-template",
        (
            "outputs/{model}/COCO100-gate-comparison/"
            f"{PROBE_RUN_DIR}/seed{{seed}}/results/"
            "{model}_selected_feature_sets.json"
        ),
        "--output-prefix",
        str(prefix),
        "--title",
        "COCO100 target-gate Torch MLP probes: shared split seed 1",
    )


def _require_prepared(output_dir: Path) -> None:
    missing = [
        path.name
        for path in (
            output_dir / "generations.json",
            output_dir / "labeling.json",
            output_dir / "image_splits.json",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{output_dir} is not prepared; missing {', '.join(missing)}. "
            "Run the prepare step first."
        )


def _link_input(source: Path, target: Path) -> None:
    relative = os.path.relpath(source, start=target.parent)
    if target.is_symlink():
        if os.readlink(target) == relative:
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"Refusing to replace existing input: {target}")
    os.symlink(relative, target)


def _first_mismatch(left: list[str], right: list[str]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def _output_dir(model: str) -> Path:
    return ROOT / "outputs" / model / "COCO100-gate-comparison"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _run(*command: str) -> None:
    print("[GateComparison]", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
