#!/usr/bin/env python3
"""Run curated risk/mass/entropy torch-probe combinations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2_5_vl_7b", "internvl_2_5_8b", "llava_1_5_7b")
BASE_EXPERIMENT = "COCO500-mass-dist-topk"
SUBDIR = "risk-mass-entropy"

RISK_FEATURES = (
    "c_vp",
    "vp_hprev_cos_r_target_lp",
    "vp_hprev_cos_r_union_la",
    "vv_hprev_cos_r_union",
    "risk_geo_raw",
)
ENTROPY_FEATURES = (
    "vv_source_topk_entropy",
    "vp_target_entropy",
    "vp_evidence_entropy",
    "vp_source_entropy",
)
MASS_FEATURES = (
    "m_p",
    "t_p",
    "t_v",
    "b_p",
)
MASS_GROUPS = (
    "t_v+t_p",
    "b_v+b_p",
    "t_v+t_p+b_v+b_p+m_p",
)


def _feature_sets() -> list[str]:
    feature_sets: list[str] = []
    feature_sets.extend(RISK_FEATURES)
    feature_sets.extend(ENTROPY_FEATURES)
    feature_sets.extend(MASS_FEATURES)
    feature_sets.extend(MASS_GROUPS)

    for risk in RISK_FEATURES:
        for entropy in ENTROPY_FEATURES:
            feature_sets.append(f"{risk}+{entropy}")
        for mass in MASS_FEATURES:
            if mass != risk:
                feature_sets.append(f"{risk}+{mass}")

    curated_triples = (
        ("vv_source_topk_entropy", "m_p"),
        ("vp_target_entropy", "t_p"),
        ("vp_evidence_entropy", "t_p"),
        ("vp_source_entropy", "m_p"),
        ("vp_target_entropy", "t_v+t_p+b_v+b_p+m_p"),
    )
    for risk in RISK_FEATURES:
        for entropy, mass in curated_triples:
            parts = [risk, entropy, *mass.split("+")]
            deduped = []
            for item in parts:
                if item not in deduped:
                    deduped.append(item)
            feature_sets.append("+".join(deduped))

    return list(dict.fromkeys(feature_sets))


FEATURE_SETS = tuple(_feature_sets())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--positive-class", default="real")
    parser.add_argument("--config", default="configs/model_configs_mass_dist_topk.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model in args.models:
        source_dir = REPO_ROOT / "outputs" / model / BASE_EXPERIMENT
        output_dir = source_dir / SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        _link_required_inputs(source_dir, output_dir)
        command = [
            args.python,
            "scripts/train_torch_probe_feature_sets.py",
            "--model",
            model,
            "--config",
            str((REPO_ROOT / args.config).resolve()),
            "--output-dir",
            str(output_dir),
            "--feature-sets",
            *FEATURE_SETS,
            "--device",
            args.device,
            "--num-epochs",
            str(args.num_epochs),
            "--batch-size",
            str(args.batch_size),
            "--positive-class",
            args.positive_class,
        ]
        print(f"[risk-mass-entropy] {model}: {len(FEATURE_SETS)} feature sets")
        _run(command)


def _link_required_inputs(source_dir: Path, output_dir: Path) -> None:
    for name in ("features.pkl", "image_splits.json"):
        source = source_dir / name
        target = output_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            continue
        os.symlink(os.path.relpath(source, start=output_dir), target)


def _run(command: list[str]) -> None:
    print("[risk-mass-entropy]", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _default_python() -> str:
    candidate = Path("/opt/conda/private/envs/vicr/bin/python")
    if candidate.exists():
        return str(candidate)
    return sys.executable


if __name__ == "__main__":
    main()
