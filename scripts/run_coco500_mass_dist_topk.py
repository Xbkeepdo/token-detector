#!/usr/bin/env python3
"""Run COCO500 mass/distribution/top-k extraction, torch probes, and curves."""

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

ENTROPY_FEATURES = (
    "vv_source_entropy",
    "vv_target_entropy",
    "vv_evidence_entropy",
    "vv_source_topk_entropy",
    "vp_source_entropy",
    "vp_target_entropy",
    "vp_evidence_entropy",
    "vp_source_topk_entropy",
)
CORE_FEATURES = ("risk_geo_raw", "c_vp", "r_es")
SOURCE_FEATURES = (
    "risk_hmid_proj",
    "risk_hprev_cos",
    "risk_hprev_proj",
    "vp_risk_hmid_proj",
    "vp_risk_hprev_cos",
    "vp_risk_hprev_proj",
)
COSINE_FEATURES = ("target_cosine", "cosine16")
TOPK_SCOPES = ("vv", "vp")
TOPK_SOURCES = (
    ("", "base"),
    ("hmid_proj", "h_mid projection"),
    ("hprev_cos", "h_prev cosine"),
    ("hprev_proj", "h_prev projection"),
)
TOPK_SELECTORS = ("union", "rec", "target")
TOPK_VARIANTS = ("lk", "", "la", "lp")


def _topk_name(scope: str, source: str, name: str) -> str:
    middle = f"_{source}" if source else ""
    return f"{scope}{middle}_{name}"


TOPK_RISK_FEATURES = tuple(
    _topk_name(scope, source, f"r_{selector}{'_' + variant if variant else ''}")
    for scope in TOPK_SCOPES
    for source, _label in TOPK_SOURCES
    for selector in TOPK_SELECTORS
    for variant in TOPK_VARIANTS
)
FEATURE_SETS = (
    *CORE_FEATURES,
    *SOURCE_FEATURES,
    *COSINE_FEATURES,
    *ENTROPY_FEATURES,
    *TOPK_RISK_FEATURES,
    *(
        f"{base}+{cosine}"
        for base in (*CORE_FEATURES, *SOURCE_FEATURES, *ENTROPY_FEATURES, *TOPK_RISK_FEATURES)
        for cosine in COSINE_FEATURES
    ),
)

BASE_PLOT_GROUPS = (
    (
        "mass_dist_tv_tp_bv_bp",
        ("t_v", "t_p", "b_v", "b_p"),
        ("T_V", "T_P", "B_V", "B_P"),
    ),
    (
        "mass_dist_cvp_mp",
        ("c_vp", "m_p"),
        ("C_VP", "M_P"),
    ),
    (
        "mass_dist_es_res",
        ("es", "r_es", "vp_es"),
        ("ES (VV)", "R_ES", "ES (VP)"),
    ),
    (
        "mass_dist_source_variants_vv",
        ("risk_geo_raw", "risk_hmid_proj", "risk_hprev_cos", "risk_hprev_proj"),
        ("h_mid cosine", "h_mid projection", "h_prev cosine", "h_prev projection"),
    ),
    (
        "mass_dist_source_variants_vp",
        (
            "risk_visual_prompt_relative_vll",
            "vp_risk_hmid_proj",
            "vp_risk_hprev_cos",
            "vp_risk_hprev_proj",
        ),
        ("VP h_mid cosine", "VP h_mid projection", "VP h_prev cosine", "VP h_prev projection"),
    ),
    (
        "mass_dist_cosines",
        ("target_cosine", "cosine16", "vp_target_cosine", "vp_cosine16"),
        ("target_cosine", "cosine16", "vp_target_cosine", "vp_cosine16"),
    ),
    (
        "mass_dist_entropy_vv",
        (
            "vv_source_entropy",
            "vv_target_entropy",
            "vv_evidence_entropy",
            "vv_source_topk_entropy",
        ),
        (
            "VV source entropy",
            "VV target entropy",
            "VV evidence entropy",
            "VV source top-k entropy",
        ),
    ),
    (
        "mass_dist_entropy_vp",
        (
            "vp_source_entropy",
            "vp_target_entropy",
            "vp_evidence_entropy",
            "vp_source_topk_entropy",
        ),
        (
            "VP source entropy",
            "VP target entropy",
            "VP evidence entropy",
            "VP source top-k entropy",
        ),
    ),
)

BASE_RISK_ALIASES = {
    ("vv", ""): "risk_geo_raw",
    ("vv", "hmid_proj"): "risk_hmid_proj",
    ("vv", "hprev_cos"): "risk_hprev_cos",
    ("vv", "hprev_proj"): "risk_hprev_proj",
    ("vp", ""): "risk_visual_prompt_relative_vll",
    ("vp", "hmid_proj"): "vp_risk_hmid_proj",
    ("vp", "hprev_cos"): "vp_risk_hprev_cos",
    ("vp", "hprev_proj"): "vp_risk_hprev_proj",
}


def _topk_plot_groups() -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
    groups = []
    for scope in TOPK_SCOPES:
        for source, source_label in TOPK_SOURCES:
            prefix = _topk_name(scope, source, "topk")
            stem = f"topk_stats_{scope}_{source or 'base'}"
            groups.append(
                (
                    stem,
                    (
                        f"{prefix}_skm",
                        f"{prefix}_tkm",
                        f"{prefix}_cov_st",
                        f"{prefix}_es",
                        BASE_RISK_ALIASES[(scope, source)],
                    ),
                    (
                        f"{scope.upper()} {source_label} SKM",
                        f"{scope.upper()} {source_label} TKM",
                        f"{scope.upper()} {source_label} COV_ST",
                        f"{scope.upper()} {source_label} ES",
                        f"{scope.upper()} {source_label} risk",
                    ),
                )
            )
            for selector in TOPK_SELECTORS:
                risk_prefix = _topk_name(scope, source, f"r_{selector}")
                groups.append(
                    (
                        f"topk_risk_{scope}_{source or 'base'}_{selector}",
                        (
                            f"{risk_prefix}_lk",
                            risk_prefix,
                            f"{risk_prefix}_la",
                            f"{risk_prefix}_lp",
                        ),
                        (
                            f"{scope.upper()} {source_label} {selector}-lk",
                            f"{scope.upper()} {source_label} {selector}",
                            f"{scope.upper()} {source_label} {selector}-la",
                            f"{scope.upper()} {source_label} {selector}-lp",
                        ),
                    )
                )
    return tuple(groups)


PLOT_GROUPS = (*BASE_PLOT_GROUPS, *_topk_plot_groups())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--config", default="configs/model_configs_mass_dist_topk.yaml")
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
    for model in args.models:
        output_dir = REPO_ROOT / "outputs" / model / "COCO500-mass-dist-topk"
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
                    *FEATURE_SETS,
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
            print(f"[mass-dist-topk] copied {source} -> {target}")


def _metadata_source_dir(model: str) -> Path:
    base = REPO_ROOT / "outputs" / model
    for suffix in METADATA_SOURCES[model]:
        candidate = base / suffix
        if all((candidate / name).exists() for name in METADATA_FILES):
            return candidate
    raise FileNotFoundError(f"No complete COCO500 metadata source found for {model}")


def _run(command: list[str]) -> None:
    print("[mass-dist-topk]", " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _default_python() -> str:
    return os.environ.get("PYTHON_BIN", sys.executable)


if __name__ == "__main__":
    main()
