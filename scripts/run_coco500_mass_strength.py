#!/usr/bin/env python3
"""Run COCO500 mass/strength extraction, torch probes, and curves."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen2_5_vl_7b", "internvl_2_5_8b", "llava_1_5_7b")
METADATA_FILES = ("generations.json", "labeling.json", "image_splits.json")
METADATA_SOURCES = {
    "qwen2_5_vl_7b": ("COCO500-KL", "COCO500-JS", "COCO500"),
    "internvl_2_5_8b": ("COCO500-KL", "COCO500-JS", "COCO500-vp"),
    "llava_1_5_7b": (
        "COCO500-KL",
        "COCO500-JS",
        "COCO500-visualprompt-relativevll-cost-geo",
    ),
}
FEATURE_SETS = (
    "risk_geo_raw",
    "c_vp",
    "r_es",
    "target_cosine",
    "cosine16",
    "vp_target_cosine",
    "vp_cosine16",
    "risk_geo_raw+target_cosine",
    "risk_geo_raw+cosine16",
    "c_vp+target_cosine",
    "c_vp+cosine16",
    "r_es+target_cosine",
    "r_es+cosine16",
    "risk_geo_raw+vp_target_cosine",
    "risk_geo_raw+vp_cosine16",
    "c_vp+vp_target_cosine",
    "c_vp+vp_cosine16",
    "r_es+vp_target_cosine",
    "r_es+vp_cosine16",
)

PLOT_GROUPS = (
    (
        "mass_strength_tv_tp_bv_bp",
        ("t_v", "t_p", "b_v", "b_p"),
        ("T_V", "T_P", "B_V", "B_P"),
    ),
    (
        "mass_strength_cvp_mp",
        ("c_vp", "m_p"),
        ("C_VP", "M_P"),
    ),
    (
        "mass_strength_es_res",
        ("es", "r_es", "vp_es"),
        ("ES (VV)", "R_ES", "ES (VP)"),
    ),
    (
        "mass_strength_cosines",
        ("target_cosine", "cosine16", "vp_target_cosine", "vp_cosine16"),
        ("target_cosine", "cosine16", "vp_target_cosine", "vp_cosine16"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--config", default="configs/model_configs_mass_strength.yaml")
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--torch-probe-device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    feature_sets = list(FEATURE_SETS)
    for model in args.models:
        output_dir = REPO_ROOT / "outputs" / model / "COCO500-mass-Strength"
        output_dir.mkdir(parents=True, exist_ok=True)
        _copy_metadata(model, output_dir, refresh=args.refresh_metadata)
        if not args.skip_extract:
            _run(
                [
                    args.python,
                    "scripts/extract_features.py",
                    "--model",
                    model,
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    args.device,
                    "--feature-devices",
                    *args.feature_devices,
                    "--resume",
                    *(
                        ["--num-images", str(args.num_images)]
                        if args.num_images is not None
                        else []
                    ),
                ]
            )
        if not args.skip_train:
            _run(
                [
                    args.python,
                    "scripts/train_torch_probe_feature_sets.py",
                    "--model",
                    model,
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--feature-sets",
                    *feature_sets,
                    "--device",
                    args.torch_probe_device,
                    "--positive-class",
                    "real",
                ]
            )
        if not args.skip_plots:
            for stem, features, labels in PLOT_GROUPS:
                _run(
                    [
                        args.python,
                        "scripts/plot_layerwise_feature_comparison.py",
                        "--model",
                        model,
                        "--output-dir",
                        str(output_dir),
                        "--features",
                        *features,
                        "--labels",
                        *labels,
                        "--name",
                        f"{model}_{stem}",
                    ]
                )


def _copy_metadata(model: str, output_dir: Path, *, refresh: bool) -> None:
    source_dir = _metadata_source_dir(model)
    for name in METADATA_FILES:
        source = source_dir / name
        target = output_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        if refresh or not target.exists():
            shutil.copy2(source, target)
            print(f"[mass-strength] copied {source} -> {target}")


def _metadata_source_dir(model: str) -> Path:
    base = REPO_ROOT / "outputs" / model
    for suffix in METADATA_SOURCES[model]:
        candidate = base / suffix
        if all((candidate / name).exists() for name in METADATA_FILES):
            return candidate
    raise FileNotFoundError(f"No complete COCO500 metadata source found for {model}")


def _run(command: list[str]) -> None:
    print("[mass-strength]", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _default_python() -> str:
    return os.environ.get("PYTHON_BIN", sys.executable)


if __name__ == "__main__":
    main()
