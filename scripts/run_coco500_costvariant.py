#!/usr/bin/env python3
"""Run the three-model COCO500 VV cost-variant experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "model_configs_coco500_costvariant.yaml"
MODELS = ("llava_1_5_7b", "internvl_2_5_8b", "qwen2_5_vl_7b")
SEEDS = (42, 43, 44)
RISK_FEATURES = (
    "risk-geo",
    "risk-cosine-hpre",
    "risk-sqrt-hmid",
    "risk-sqrt-hpre",
    "risk-rawAttention-hmid",
    "risk-rawAttention-hpre",
    "gauss-risk-geo",
    "gauss-risk-cosine-hpre",
    "gauss-risk-sqrt-hmid",
    "gauss-risk-sqrt-hpre",
)
FEATURE_SETS = (
    "hprecosine",
    *RISK_FEATURES,
    *(f"{risk}+hprecosine" for risk in RISK_FEATURES),
)
STEPS = ("prepare", "label", "extract", "validate", "train", "plot", "summarize")


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
        if "label" in args.steps:
            _run(
                args.python,
                "coco-labeling/label_coco.py",
                "--model",
                model,
                "--config",
                str(CONFIG),
                "--output-dir",
                str(output_dir),
                "--chair-cache",
                str(ROOT / "outputs" / "smoke-fj01" / "chair_cache.pkl"),
                "--num-images",
                "500",
                "--seed",
                "42",
                "--device",
                args.devices[0],
                "--resume",
            )
        if "extract" in args.steps:
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
                "500",
                "--seed",
                "42",
                "--feature-devices",
                *args.devices,
            ]
            if args.resume:
                command.append("--resume")
            _run(*command)
        if "validate" in args.steps:
            _validate_features(model, output_dir)
        if "train" in args.steps:
            _train(args.python, model, output_dir, tuple(args.devices))
        if "plot" in args.steps:
            _run(
                args.python,
                "scripts/plot_coco500_costvariant.py",
                "--model",
                model,
                "--output-dir",
                str(output_dir),
            )
    if "summarize" in args.steps:
        _summarize(args.python, models)


def _prepare(models: tuple[str, ...]) -> None:
    id_sets = []
    generations_by_model = {}
    for model in models:
        source_dir = ROOT / "outputs" / model / "COCO500"
        generations = _load_json(source_dir / "generations.json")
        if len(generations) != 500:
            raise ValueError(f"{model} must have exactly 500 generations, found {len(generations)}")
        ids = {int(image_id) for image_id in generations}
        id_sets.append(ids)
        generations_by_model[model] = source_dir / "generations.json"
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("The three models do not share the same 500 COCO image IDs.")

    image_ids = sorted(id_sets[0])
    random.Random(42).shuffle(image_ids)
    split = {
        "train": image_ids[:400],
        "val": image_ids[400:450],
        "test": image_ids[450:500],
    }
    _validate_split(split)
    for model in models:
        output_dir = _output_dir(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / "generations.json"
        if not destination.exists():
            shutil.copy2(generations_by_model[model], destination)
        _write_json(output_dir / "image_splits.json", split)
    print("[CostVariant] Prepared a shared strict 400/50/50 split.")


def _train(python: str, model: str, output_dir: Path, devices: tuple[str, ...]) -> None:
    jobs = []
    for index, seed in enumerate(SEEDS):
        seed_dir = output_dir / "torchmlp-seed3-811" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _link_input(output_dir / "features.pkl", seed_dir / "features.pkl")
        _link_input(output_dir / "image_splits.json", seed_dir / "image_splits.json")
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
            *FEATURE_SETS,
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
        print("[CostVariant]", " ".join(command), f"> {log_path}", flush=True)
        process = subprocess.Popen(
            command,
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
                output=f"See {log_path}",
            )


def _summarize(python: str, models: tuple[str, ...]) -> None:
    prefix = ROOT / "outputs" / "coco500-costvariant-summary" / "three_model_seed3_summary"
    _run(
        python,
        "scripts/summarize_torch_probe_seed_runs.py",
        "--models",
        *models,
        "--seeds",
        *(str(seed) for seed in SEEDS),
        "--run-template",
        "outputs/{model}/coco500-costvariant/torchmlp-seed3-811/seed{seed}/results/"
        "{model}_selected_feature_sets.json",
        "--output-prefix",
        str(prefix),
        "--title",
        "COCO500 VV cost variants: Torch MLP three-seed summary",
    )


def _validate_features(model: str, output_dir: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from utils.io_utils import load_pkl

    split = _load_json(output_dir / "image_splits.json")
    _validate_split(split)
    rows = load_pkl(output_dir / "features.pkl")
    if not rows:
        raise ValueError(f"{model}: no extracted feature rows")
    expected_layers = 28 if model == "qwen2_5_vl_7b" else 32
    allowed = {
        "image_id",
        "token_str",
        "token_id",
        "target_token_id",
        "response_token_idx",
        "label",
        "dgst_t_relative_vll_logit_source",
        "dgst_t_source_distribution_mode",
        "dgst_t_vv_support_positions",
        "dgst_t_vv_support_attention_per_layer",
        "dgst_t_vv_source_dist_per_layer",
        "dgst_t_vv_semantic_gate_per_layer",
        "dgst_t_vv_gauss_semantic_gate_per_layer",
        "dgst_t_cost_variant_mad_scale",
        "dgst_t_cost_variant_transport_top_k",
        "dgst_t_cost_variant_hprecosine_top_k",
        "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer",
        *(f"dgst_t_{risk.replace('-', '_').replace('rawAttention', 'raw_attention')}_per_layer" for risk in RISK_FEATURES),
    }
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(f"{model}: unexpected saved fields: {sorted(extra)}")
        for key in allowed:
            if key not in row:
                raise KeyError(f"{model}: missing {key}")
        for risk in RISK_FEATURES:
            key = f"dgst_t_{risk.replace('-', '_').replace('rawAttention', 'raw_attention')}_per_layer"
            values = row[key]
            if len(values) != expected_layers or not all(math.isfinite(float(x)) for x in values):
                raise ValueError(f"{model}: invalid {key}")
        raw_shapes = [
            tuple(row[key].shape)
            for key in (
                "dgst_t_vv_support_attention_per_layer",
                "dgst_t_vv_source_dist_per_layer",
                "dgst_t_vv_semantic_gate_per_layer",
                "dgst_t_vv_gauss_semantic_gate_per_layer",
            )
        ]
        if len(set(raw_shapes)) != 1 or raw_shapes[0][0] != expected_layers:
            raise ValueError(f"{model}: raw VV tensor shape mismatch: {raw_shapes}")
    print(f"[CostVariant] Validated {model}: {len(rows)} object-token rows.")


def _validate_split(split: dict) -> None:
    sets = {name: {int(value) for value in split[name]} for name in ("train", "val", "test")}
    if {name: len(values) for name, values in sets.items()} != {"train": 400, "val": 50, "test": 50}:
        raise ValueError("Expected a strict 400/50/50 split.")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("COCO500 split partitions must be disjoint.")
    if len(set.union(*sets.values())) != 500:
        raise ValueError("COCO500 split union must contain 500 images.")


def _link_input(source: Path, target: Path) -> None:
    relative = os.path.relpath(source, start=target.parent)
    if target.is_symlink():
        if os.readlink(target) == relative:
            return
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"Refusing to replace existing input: {target}")
    os.symlink(relative, target)


def _output_dir(model: str) -> Path:
    return ROOT / "outputs" / model / "coco500-costvariant"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _run(*command: str) -> None:
    print("[CostVariant]", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
