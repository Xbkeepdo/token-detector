#!/usr/bin/env python3
"""Run generation, labeling, extraction, training, and plotting from one YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import (  # noqa: E402
    VALID_DGST_BRANCHES,
    extraction_mode_flags,
    get_model_cfg,
    load_config,
    resolve_run_config,
)
from utils.generation_provenance import (  # noqa: E402
    GENERATION_MANIFEST_NAME,
    build_generation_manifest,
    canonical_generation_payload,
    stable_sha256 as stable_generation_sha256,
    validate_generation_manifest,
)
from utils.split_utils import (  # noqa: E402
    ensure_strict_82_split,
    validate_strict_82_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the token-detector pipeline from one unified YAML config."
    )
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--model", default=None, help="Override run.model.")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    parser.add_argument("--prompt", default=None, help="Override run.prompt.")
    parser.add_argument(
        "--extraction-mode",
        choices=["all", "method_only", "ads_cgc_only", "baseline_only"],
        default=None,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["generation", "labeling", "feature_extraction", "training", "plotting"],
        default=None,
        help="Run only these stages, overriding run.stages.",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--reuse-generations-from",
        default=None,
        help=(
            "Seed a new output directory from another experiment's "
            "generations.json and image_splits.json. Labeling/features are "
            "never copied."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print commands without creating or changing artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_config(str(config_path))
    environ = dict(os.environ)
    _apply_cli_overrides(environ, args)
    run = resolve_run_config(config, environ=environ)
    config["run"] = run
    _apply_runtime_overrides(config, run)
    _apply_effective_feature_switches(config, run["extraction_mode"])

    output_dir = _repo_path(run["output_dir"])
    resolved_config_path = output_dir / "resolved_pipeline_config.yaml"
    if args.dry_run:
        command_config_path = config_path
        if args.reuse_generations_from:
            print(
                "[Pipeline] DRY RUN: would reuse generations from "
                f"{_repo_path(args.reuse_generations_from)}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_generations_from:
            if bool(run["stages"]["generation"]):
                raise ValueError(
                    "--reuse-generations-from requires the generation stage "
                    "to be disabled; the reused captions are the generation "
                    "artifact for this experiment."
                )
            source = _repo_path(args.reuse_generations_from)
            expected_image_ids = _dataset_selected_image_ids(config)
            model_cfg = get_model_cfg(config, str(run["model"]))
            _reuse_generation_artifacts(
                source,
                output_dir,
                model=str(run["model"]),
                model_cfg=model_cfg,
                prompt=str(run["prompt"]),
                expected_image_ids=expected_image_ids,
                resume=bool(run.get("resume")),
                adopt_legacy=bool(run.get("adopt_legacy_artifacts", False)),
            )
            run["reuse_generations_from"] = str(source.resolve())
        if not bool(run.get("resume")):
            _prepare_fresh_artifacts(output_dir, run, config)
        _validate_or_write_manifest(output_dir, run, config)
        _atomic_write_yaml(resolved_config_path, config)
        command_config_path = resolved_config_path

    print(f"[Pipeline] model={run['model']}")
    print(f"[Pipeline] output={output_dir}")
    print(f"[Pipeline] prompt={run['prompt']!r}")
    print(f"[Pipeline] extraction_mode={run['extraction_mode']}")
    print(
        "[Pipeline] stages="
        + ", ".join(name for name, enabled in run["stages"].items() if enabled)
    )

    generation_enabled = bool(run["stages"]["generation"])
    labeling_enabled = bool(run["stages"]["labeling"])
    if generation_enabled or labeling_enabled:
        if labeling_enabled and not generation_enabled:
            _require_complete_generations(
                output_dir,
                config=config,
                run=run,
                expected_image_ids=_dataset_selected_image_ids(config),
            )
        label_command = build_label_command(
            run=run,
            config_path=command_config_path,
            output_dir=output_dir,
            force_resume=(labeling_enabled and not generation_enabled),
        )
        _run(label_command, dry_run=args.dry_run)
        if not args.dry_run:
            _require_complete_generations(
                output_dir,
                config=config,
                run=run,
                expected_image_ids=_dataset_selected_image_ids(config),
            )
            # Labeling is produced by the combined generation/label command.
            # Refresh the artifact-level provenance only after the atomic JSON
            # has landed.
            _validate_or_write_manifest(
                output_dir,
                run,
                config,
                allow_labeling_update=True,
            )
        if generation_enabled and not labeling_enabled:
            print(
                "[Pipeline] NOTE: label_coco.py combines generation and labeling; "
                "label artifacts were refreshed with generation."
            )

    needs_split = any(
        bool(run["stages"][name])
        for name in ("feature_extraction", "training", "plotting")
    ) or generation_enabled or labeling_enabled
    if needs_split and not args.dry_run:
        image_ids = _selected_image_ids(config, output_dir)
        expected_count = int(config["dataset"]["num_images"])
        if len(image_ids) != expected_count:
            raise ValueError(
                f"Strict split requires {expected_count} selected image IDs, "
                f"found {len(image_ids)}"
            )
        shared_path_value = config["dataset"].get("shared_split_path")
        shared_path = _repo_path(shared_path_value) if shared_path_value else None
        splits, backup = ensure_strict_82_split(
            output_dir / "image_splits.json",
            image_ids,
            seed=int(config["dataset"].get("seed", 42)),
            shared_splits_path=shared_path,
        )
        if backup is not None:
            print(f"[Pipeline] Backed up previous split to {backup}")
        print(
            "[Pipeline] Strict image split (no validation): "
            f"train={len(splits['train'])}, test={len(splits['test'])}; "
            "fixed epochs, last checkpoint, train-F1 threshold"
        )

    flags = _effective_feature_flags(config)
    if run["stages"]["feature_extraction"]:
        if flags["method"] or flags["ads_cgc"]:
            # In ``all`` mode the root extractor owns BaselineRuntime and writes
            # OUTPUT/baseline from the same wrapper outputs.  The standalone
            # extractor is reserved for baseline_only.
            _run(
                build_root_extract_command(run, command_config_path, output_dir),
                dry_run=args.dry_run,
            )
        elif flags["baseline"]:
            _run(
                build_baseline_extract_command(run, command_config_path, output_dir),
                dry_run=args.dry_run,
                require_script=not args.dry_run,
            )

    if run["stages"]["training"]:
        if flags["method"] or flags["ads_cgc"]:
            _run(
                build_root_train_command(config, run, command_config_path, output_dir),
                dry_run=args.dry_run,
            )
        if flags["baseline"]:
            _run(
                build_baseline_train_command(run, command_config_path, output_dir),
                dry_run=args.dry_run,
                require_script=not args.dry_run,
            )

    if run["stages"]["plotting"]:
        if not (flags["method"] or flags["ads_cgc"]):
            print("[Pipeline] Plotting skipped: baseline_only has no root feature curves.")
        else:
            _run(
                build_plot_command(config, run, output_dir),
                dry_run=args.dry_run,
            )

    print("[Pipeline] Completed requested stages.")


def build_label_command(
    *,
    run: Mapping[str, object],
    config_path: Path,
    output_dir: Path,
    force_resume: bool = False,
) -> list[str]:
    devices = run["devices"]
    assert isinstance(devices, Mapping)
    command = [
        sys.executable,
        "coco-labeling/label_coco.py",
        "--model",
        str(run["model"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--prompt",
        str(run["prompt"]),
        "--device",
        str(devices["primary"]),
        "--generation-devices",
        *[str(value) for value in devices["generation"]],
    ]
    if run.get("chair_cache"):
        command.extend(["--chair-cache", str(run["chair_cache"])])
    if run.get("max_pixels") is not None:
        command.extend(["--max-pixels", str(run["max_pixels"])])
    if bool(run.get("resume")) or force_resume:
        command.append("--resume")
    return command


def build_root_extract_command(
    run: Mapping[str, object],
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    devices = run["devices"]
    assert isinstance(devices, Mapping)
    command = [
        sys.executable,
        "scripts/extract_features.py",
        "--model",
        str(run["model"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--prompt",
        str(run["prompt"]),
        "--extraction-mode",
        str(run["extraction_mode"]),
        "--device",
        str(devices["primary"]),
        "--feature-devices",
        *[str(value) for value in devices["feature_extraction"]],
    ]
    branches = run.get("dgst_branches")
    if branches:
        command.extend(["--dgst-branches", *[str(value) for value in branches]])
    if run.get("max_pixels") is not None:
        command.extend(["--max-pixels", str(run["max_pixels"])])
    if bool(run.get("resume")):
        command.append("--resume")
    return command


def build_baseline_extract_command(
    run: Mapping[str, object],
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    devices = run["devices"]
    assert isinstance(devices, Mapping)
    command = [
        sys.executable,
        "scripts/extract_features.py",
        "--model",
        str(run["model"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--prompt",
        str(run["prompt"]),
        "--extraction-mode",
        "baseline_only",
        "--device",
        str(devices["primary"]),
        "--feature-devices",
        *[str(value) for value in devices["feature_extraction"]],
    ]
    if run.get("max_pixels") is not None:
        command.extend(["--max-pixels", str(run["max_pixels"])])
    if bool(run.get("resume")):
        command.append("--resume")
    return command


def build_root_train_command(
    config: Mapping[str, object],
    run: Mapping[str, object],
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    feature_sets = _feature_sets(config, run)
    common = [
        "--model",
        str(run["model"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--feature-sets",
        *feature_sets,
    ]
    trainer = str(run.get("trainer", "torch_mlp"))
    if trainer in {"torch_mlp", "torch_probe"}:
        devices = run["devices"]
        assert isinstance(devices, Mapping)
        command = [
            sys.executable,
            "scripts/train_torch_probe_feature_sets.py",
            *common,
            "--device",
            str(devices["training"]),
            "--positive-class",
            str(run.get("positive_class", "real")),
        ]
        command.extend(shlex.split(str(run.get("torch_probe_args", ""))))
        return command
    return [sys.executable, "scripts/train_feature_sets.py", *common]


def build_baseline_train_command(
    run: Mapping[str, object],
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    devices = run["devices"]
    assert isinstance(devices, Mapping)
    return [
        sys.executable,
        "scripts/train_baselines.py",
        "--model",
        str(run["model"]),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--device",
        str(devices["training"]),
    ]


def build_plot_command(
    config: Mapping[str, object],
    run: Mapping[str, object],
    output_dir: Path,
) -> list[str]:
    plotting = config.get("plotting") or {}
    if not isinstance(plotting, Mapping) or not plotting.get("script"):
        raise ValueError("plotting stage is enabled but plotting.script is not configured")
    command = [
        sys.executable,
        str(plotting["script"]),
        "--model",
        str(run["model"]),
        "--output-dir",
        str(output_dir),
    ]
    features = plotting.get("features") or []
    labels = plotting.get("labels") or []
    if features:
        command.extend(["--features", *[str(item) for item in features]])
    if labels:
        command.extend(["--labels", *[str(item) for item in labels]])
    if plotting.get("name"):
        command.extend(["--name", str(plotting["name"])])
    return command


def _feature_sets(
    config: Mapping[str, object],
    run: Mapping[str, object],
) -> list[str]:
    values = run.get("feature_sets")
    if not values:
        training = config.get("training") or {}
        if isinstance(training, Mapping):
            values = training.get("feature_sets")
            if isinstance(values, Mapping):
                mode = str(run.get("extraction_mode", "all"))
                if "method" in values or "ads_cgc" in values:
                    method_values = _enabled_method_feature_sets(
                        config,
                        list(values.get("method") or []),
                    )
                    ads_cgc_values = list(values.get("ads_cgc") or [])
                    if mode == "method_only":
                        values = method_values
                    elif mode == "ads_cgc_only":
                        values = ads_cgc_values
                    elif mode == "all":
                        values = [*method_values, *ads_cgc_values]
                    else:
                        values = values.get("default")
                else:
                    values = values.get(mode) or values.get("default")
    if not values:
        experiment = config.get("experiment") or {}
        if isinstance(experiment, Mapping):
            configured = experiment.get("feature_sets")
            if isinstance(configured, Mapping):
                values = configured.get(str(experiment.get("mode", ""))) or configured.get(
                    "default"
                )
            elif isinstance(configured, list):
                values = configured
    if not values:
        values = ["risk", "target_cosine", "risk+target_cosine"]
    resolved = [str(item) for item in values]
    if str(run.get("extraction_mode", "all")) in {"all", "method_only"}:
        resolved = _enabled_method_feature_sets(config, resolved)
    if not resolved:
        raise ValueError(
            "No trainable root feature sets remain after applying DGST branch switches."
        )
    return resolved


def _enabled_method_feature_sets(
    config: Mapping[str, object],
    feature_sets: Sequence[object],
) -> list[str]:
    """Drop training blocks belonging to a disabled DGST target branch."""
    extraction = config.get("feature_extraction") or {}
    dgst = extraction.get("dgst_t") or {} if isinstance(extraction, Mapping) else {}
    if not isinstance(dgst, Mapping):
        return [str(value) for value in feature_sets]
    known = {
        "hpre_raw_logit_gauss",
        "hpre_softmax_prob_gauss",
        "hmid_raw_logit_gauss",
        "hmid_softmax_prob_gauss",
        "hpre_softmax_prob_direct",
        "raw_attention",
    }
    configured = dgst.get("four_gate_methods")
    active = (
        {str(method) for method in configured}
        if isinstance(configured, (list, tuple))
        else set(known)
    )
    branches = dgst.get("branches") or {}
    if isinstance(branches, Mapping):
        active = {
            method for method in active if bool(branches.get(method, True))
        }
    disabled = known - active
    return [
        str(value)
        for value in feature_sets
        if not any(
            component.strip().startswith(f"{method}_")
            for component in str(value).split("+")
            for method in disabled
        )
    ]


def _apply_runtime_overrides(config: dict, run: Mapping[str, object]) -> None:
    """Apply lightweight shell overrides before fingerprinting/resolution."""
    model = str(run["model"])
    max_pixels = run.get("max_pixels")
    if max_pixels is not None:
        models = config.get("models") or {}
        model_config = models.get(model)
        if not isinstance(model_config, dict):
            raise ValueError(f"Missing model config for runtime override: {model}")
        model_config["max_pixels"] = int(max_pixels)

    selected = run.get("dgst_branches")
    if selected:
        extraction = config.setdefault("feature_extraction", {})
        dgst = extraction.setdefault("dgst_t", {})
        selected_set = {str(value) for value in selected}
        dgst["four_gate_methods"] = [
            method for method in VALID_DGST_BRANCHES if method in selected_set
        ]
        dgst["branches"] = {
            method: method in selected_set for method in VALID_DGST_BRANCHES
        }


def _apply_effective_feature_switches(config: dict, mode: str) -> None:
    flags = extraction_mode_flags(mode)
    extraction = config.setdefault("feature_extraction", {})
    for family, enabled in flags.items():
        section = extraction.setdefault(family, {})
        configured = bool(section.get("enabled", True))
        section["enabled"] = bool(enabled and configured)


def _effective_feature_flags(config: Mapping[str, object]) -> dict[str, bool]:
    extraction = config.get("feature_extraction") or {}
    if not isinstance(extraction, Mapping):
        raise ValueError("feature_extraction must be a mapping")
    result = {}
    for family in ("method", "ads_cgc", "baseline"):
        section = extraction.get(family) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"feature_extraction.{family} must be a mapping")
        result[family] = bool(section.get("enabled", False))
    return result


def _dataset_selected_image_ids(config: Mapping[str, object]) -> list[int]:
    """Load the deterministic COCO cohort independently of retained artifacts."""

    from data.coco_loader import load_coco_samples

    dataset = config.get("dataset") or {}
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be a mapping")
    expected_count = int(dataset["num_images"])
    samples = load_coco_samples(
        images_dir=os.path.join(str(dataset["coco_root"]), "val2014"),
        instances_file=str(dataset["annotation_file"]),
        captions_file=str(dataset["captions_file"]),
        num_images=expected_count,
        seed=int(dataset.get("seed", 42)),
    )
    image_ids = [int(sample["image_id"]) for sample in samples]
    if len(image_ids) != expected_count or len(set(image_ids)) != expected_count:
        raise ValueError(
            "Selected COCO cohort is incomplete or contains duplicate image IDs: "
            f"selected={len(image_ids)}, unique={len(set(image_ids))}, "
            f"expected={expected_count}"
        )
    return sorted(image_ids)


def _selected_image_ids(config: dict, output_dir: Path) -> list[int]:
    """Return the canonical cohort after checking every retained ID universe."""

    expected_ids = _dataset_selected_image_ids(config)
    expected_set = set(expected_ids)
    labeling_path = output_dir / "labeling.json"
    if labeling_path.exists():
        rows = _load_json(labeling_path)
        try:
            actual_ids = [int(value) for value in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid image ID in {labeling_path}") from exc
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_set:
            raise ValueError(
                f"{labeling_path} does not match the selected COCO cohort; "
                f"missing={len(expected_set - set(actual_ids))}, "
                f"extra={len(set(actual_ids) - expected_set)}"
            )

    generations_path = output_dir / "generations.json"
    if generations_path.exists():
        canonical_generation_payload(
            _load_json(generations_path),
            expected_image_ids=expected_ids,
        )
    return expected_ids


def _require_complete_generations(
    output_dir: Path,
    *,
    config: dict,
    run: Mapping[str, object],
    expected_image_ids: Optional[Sequence[int]] = None,
) -> dict[str, object]:
    """Validate complete captions, actual response IDs, and generation identity."""

    path = output_dir / "generations.json"
    if path.is_symlink():
        raise ValueError(f"Refusing a symlinked generation artifact: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"labeling=true with generation=false requires existing {path}"
        )
    image_ids = list(expected_image_ids or _dataset_selected_image_ids(config))
    generations = _load_json(path)
    canonical_generation_payload(
        generations,
        expected_image_ids=image_ids,
    )

    model = str(run["model"])
    prompt = str(run["prompt"])
    model_cfg = get_model_cfg(config, model)
    manifest_path = output_dir / GENERATION_MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError(f"Refusing a symlinked generation manifest: {manifest_path}")
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        validate_generation_manifest(
            manifest,
            model=model,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=generations,
            expected_image_ids=image_ids,
        )
        return dict(manifest)

    manifest = build_generation_manifest(
        model=model,
        model_cfg=model_cfg,
        prompt=prompt,
        generations=generations,
        expected_image_ids=image_ids,
    )
    _atomic_write_json(manifest_path, manifest)
    print(
        f"[Pipeline] Complete generations.json validated; wrote "
        f"{GENERATION_MANIFEST_NAME} automatically."
    )
    return manifest


def _validate_or_write_manifest(
    output_dir: Path,
    run: dict,
    config: dict,
    *,
    allow_labeling_update: bool = False,
) -> None:
    path = output_dir / "pipeline_manifest.json"
    extraction = config.get("feature_extraction") or {}
    baseline = (
        (extraction.get("baseline") or {})
        if isinstance(extraction, Mapping)
        else {}
    )
    baseline_subdir = str(
        baseline.get("output_subdir", "baseline")
        if isinstance(baseline, Mapping)
        else "baseline"
    )
    flags = _effective_feature_flags(config)
    root_features_path = output_dir / "features.pkl"
    baseline_features_path = output_dir / baseline_subdir / "features.pkl"
    generations_path = output_dir / "generations.json"
    labeling_path = output_dir / "labeling.json"
    generation_artifacts_exist = generations_path.exists() or _directory_has_files(
        output_dir / "generation_shards"
    )
    root_artifacts_exist = root_features_path.exists() or any(
        output_dir.glob("features.part*.pkl")
    )
    baseline_dir = output_dir / baseline_subdir
    baseline_artifacts_exist = any(
        (
            baseline_features_path.exists(),
            any(baseline_dir.glob("features.part*.pkl")),
            _directory_has_files(baseline_dir / "feature_parts"),
            _directory_has_files(baseline_dir / "dhcp"),
            _directory_has_files(baseline_dir / "halloc"),
            _directory_has_files(baseline_dir / "svar_official"),
        )
    )
    safe_labeling_update = bool(
        allow_labeling_update
        and not root_artifacts_exist
        and not baseline_artifacts_exist
    )

    previous = _load_json(path) if path.exists() else {}
    root_hash = _root_features_config_sha256(config, run)
    baseline_hash = _baseline_features_config_sha256(config, run)
    labeling_artifact = _labeling_artifact_metadata(output_dir, config)
    # A mode that does not own one feature family must not overwrite that
    # family's provenance.  This is what makes baseline-only genuinely
    # isolated from the root feature file (and vice versa).
    if not (flags["method"] or flags["ads_cgc"]):
        root_hash = previous.get("root_features_config_sha256")
    if not flags["baseline"]:
        baseline_hash = previous.get("baseline_features_config_sha256")

    labeling_provenance_keys = (
        "labeling_sha256",
        "labeling_manifest_sha256",
        "labeling_schema_version",
        "labeling_primary_locator",
        "labeling_sample_unit",
    )
    current = {
        "manifest_version": 4,
        "model": str(run["model"]),
        "prompt": str(run["prompt"]),
        "num_images": int(config["dataset"]["num_images"]),
        "dataset_seed": int(config["dataset"].get("seed", 42)),
        "split_strategy": "strict_82",
        # Informational only.  It is deliberately not a global resume key:
        # method_only and baseline_only can safely populate the same output in
        # separate invocations.
        "last_extraction_mode": str(run["extraction_mode"]),
        "chair_cache": str(run.get("chair_cache") or ""),
        "generation_config_sha256": _generation_config_sha256(config, run),
        "labeling_config_sha256": _labeling_config_sha256(config, run),
        **labeling_artifact,
        "root_features_config_sha256": root_hash,
        "baseline_features_config_sha256": baseline_hash,
    }
    if (
        not labeling_path.exists()
        and (root_artifacts_exist or baseline_artifacts_exist)
        and previous
    ):
        # A fresh generation/labeling stage must not erase the provenance of
        # retained features before the replacement labels can be compared.
        for key in labeling_provenance_keys:
            current[key] = previous.get(key)
    if run.get("reuse_generations_from"):
        current["generation_reuse_source"] = str(run["reuse_generations_from"])
    has_reusable_artifacts = (
        labeling_path.exists()
        or generation_artifacts_exist
        or root_artifacts_exist
        or baseline_artifacts_exist
    )
    generation_only_seed = False
    if (
        generations_path.is_file()
        and not labeling_path.exists()
        and not root_artifacts_exist
        and not baseline_artifacts_exist
    ):
        generation_payload = canonical_generation_payload(
            _load_json(generations_path)
        )
        dataset_cfg = config.get("dataset") or {}
        expected_count = int(
            dataset_cfg.get("num_images", len(generation_payload))
        )
        generation_only_seed = bool(generation_payload) and (
            len(generation_payload) == expected_count
        )
    if not path.exists() and has_reusable_artifacts:
        if generation_only_seed:
            print(
                "[Pipeline] Registering generation-only artifacts in the new "
                "output manifest."
            )
        elif not bool(run.get("adopt_legacy_artifacts", False)):
            raise ValueError(
                "Existing artifacts have no pipeline_manifest.json, so their "
                "model/prompt/feature configuration cannot be verified. Set "
                "ADOPT_LEGACY_ARTIFACTS=true in run.sh (or "
                "run.adopt_legacy_artifacts=true in the coordinator) once to "
                "trust and register them, or use a new output_dir."
            )
        else:
            print(
                "[Pipeline] WARNING: adopting legacy artifacts without a prior "
                "manifest; subsequent resume runs will be fingerprint-checked."
            )

    if path.exists() and has_reusable_artifacts:
        checks: dict[str, str | None] = {}
        if generation_artifacts_exist:
            checks["generation_config_sha256"] = current[
                "generation_config_sha256"
            ]
        if labeling_path.exists():
            checks["labeling_config_sha256"] = current[
                "labeling_config_sha256"
            ]
            for key in labeling_provenance_keys:
                checks[key] = current.get(key)
        if root_artifacts_exist and (flags["method"] or flags["ads_cgc"]):
            checks["root_features_config_sha256"] = current[
                "root_features_config_sha256"
            ]
        if baseline_artifacts_exist and flags["baseline"]:
            checks["baseline_features_config_sha256"] = current[
                "baseline_features_config_sha256"
            ]

        # Upgrade the short-lived v1 manifest without weakening normal checks.
        # Existing artifacts can be adopted only through the same explicit
        # opt-in used for pre-manifest runs.
        missing_family_keys = [
            key
            for key in checks
            if key.endswith("features_config_sha256") and not previous.get(key)
        ]
        if missing_family_keys:
            old_hash_matches = previous.get("artifact_config_sha256") == (
                _artifact_config_sha256(config, run)
            )
            if not old_hash_matches and not bool(
                run.get("adopt_legacy_artifacts", False)
            ):
                raise ValueError(
                    "Existing feature artifacts use a legacy manifest without "
                    "independent root/baseline fingerprints. Set "
                    "ADOPT_LEGACY_ARTIFACTS=true in run.sh (or "
                    "run.adopt_legacy_artifacts=true in the coordinator) once "
                    "to register them, or use a new output_dir."
                )
            for key in missing_family_keys:
                previous[key] = checks[key]
            print(
                "[Pipeline] WARNING: upgraded legacy feature provenance to "
                "independent root/baseline fingerprints."
            )

        if allow_labeling_update and (
            root_artifacts_exist or baseline_artifacts_exist
        ):
            changed_labeling = {
                key: (previous.get(key), current.get(key))
                for key in labeling_provenance_keys
                if previous.get(key) != current.get(key)
            }
            if changed_labeling:
                raise ValueError(
                    "Refusing to accept newly produced labeling while retained "
                    "feature artifacts still depend on the previous labels: "
                    f"{changed_labeling}. Remove/re-extract those features or "
                    "use a new output_dir."
                )

        missing_labeling_keys = [
            key
            for key in (
                "labeling_sha256",
                "labeling_schema_version",
                "labeling_primary_locator",
            )
            if key in checks and previous.get(key) in (None, "")
        ]
        if missing_labeling_keys and labeling_path.exists():
            if safe_labeling_update:
                for key in missing_labeling_keys:
                    previous[key] = checks[key]
            elif not bool(run.get("adopt_legacy_artifacts", False)):
                raise ValueError(
                    "Existing labeling artifacts use a legacy manifest without "
                    "schema/locator/content provenance. Set "
                    "ADOPT_LEGACY_ARTIFACTS=true once to register them, rebuild "
                    "labeling, or use a new output_dir."
                )
            else:
                for key in missing_labeling_keys:
                    previous[key] = checks[key]
                print(
                    "[Pipeline] WARNING: adopted legacy labeling provenance "
                    f"for {missing_labeling_keys}."
                )

        mismatches = {
            key: (previous.get(key), value)
            for key, value in checks.items()
            if previous.get(key) != value
            and not (
                safe_labeling_update
                and key
                in {
                    "labeling_sha256",
                    "labeling_manifest_sha256",
                    "labeling_schema_version",
                    "labeling_primary_locator",
                    "labeling_sample_unit",
                }
            )
        }
        if mismatches:
            action = "resume" if bool(run.get("resume")) else "reuse"
            raise ValueError(
                f"Refusing to {action} incompatible retained artifacts: "
                f"{mismatches}. Re-run the stage that produces those "
                "artifacts, or use a new output_dir."
            )
    _atomic_write_json(path, current)


def _artifact_config_sha256(config: Mapping[str, object], run: Mapping[str, object]) -> str:
    """Legacy v1 aggregate fingerprint (kept only for manifest migration)."""
    models = config.get("models") or {}
    model_key = str(run["model"])
    payload = {
        "model": model_key,
        "model_config": models.get(model_key) if isinstance(models, Mapping) else None,
        "prompt": str(run["prompt"]),
        "chair_cache": str(run.get("chair_cache") or ""),
        "extraction_mode": str(run["extraction_mode"]),
        "feature_extraction": config.get("feature_extraction"),
        "baselines": config.get("baselines"),
        "dataset": config.get("dataset"),
    }
    return _stable_sha256(payload)


def _root_features_config_sha256(
    config: Mapping[str, object],
    run: Mapping[str, object],
) -> str:
    """Fingerprint only inputs that change the root ``features.pkl``."""
    extraction = config.get("feature_extraction") or {}
    if not isinstance(extraction, Mapping):
        extraction = {}
    return _stable_sha256(
        {
            "labeling_config_sha256": _labeling_config_sha256(config, run),
            "experiment": config.get("experiment"),
            "method": extraction.get("method"),
            "ads_cgc": extraction.get("ads_cgc"),
            "dgst_t": extraction.get("dgst_t"),
            "ads": extraction.get("ads"),
            "cgc": extraction.get("cgc"),
        }
    )


def _baseline_features_config_sha256(
    config: Mapping[str, object],
    run: Mapping[str, object],
) -> str:
    """Fingerprint only inputs that change baseline features or caches."""

    # Keep the coordinator hash identical to the per-artifact manifests used by
    # both joint and baseline-only extraction. Training-only settings (MLP
    # widths, optimizers, epochs, classifiers, and early stopping) must not
    # invalidate already-extracted features.
    from scripts.extract_features import (
        _combined_baseline_config,
        _controlled_baseline_enabled,
        _controlled_baseline_feature_config,
        _official_svar_enabled,
        _official_svar_feature_config,
    )

    baseline_cfg = _combined_baseline_config(config)
    return _stable_sha256(
        {
            "labeling_config_sha256": _labeling_config_sha256(config, run),
            "controlled": (
                _controlled_baseline_feature_config(baseline_cfg)
                if _controlled_baseline_enabled(baseline_cfg)
                else None
            ),
            "official": (
                _official_svar_feature_config(baseline_cfg)
                if _official_svar_enabled(baseline_cfg)
                else None
            ),
        }
    )


def _generation_config_sha256(
    config: Mapping[str, object],
    run: Mapping[str, object],
) -> str:
    models = config.get("models") or {}
    model_key = str(run["model"])
    return _stable_sha256(
        {
            "model": model_key,
            "model_config": (
                models.get(model_key) if isinstance(models, Mapping) else None
            ),
            "prompt": str(run["prompt"]),
            "dataset": config.get("dataset"),
        }
    )


def _labeling_config_sha256(
    config: Mapping[str, object],
    run: Mapping[str, object],
) -> str:
    return _stable_sha256(
        {
            "generation_config_sha256": _generation_config_sha256(config, run),
            "chair_cache": str(run.get("chair_cache") or ""),
            "labeling": config.get("labeling"),
        }
    )


def _labeling_artifact_metadata(
    output_dir: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Return artifact-level labeling provenance for pipeline resume checks."""

    labeling_path = output_dir / "labeling.json"
    manifest_path = output_dir / "labeling_manifest.json"
    labeling_cfg = config.get("labeling") or {}
    if not isinstance(labeling_cfg, Mapping):
        labeling_cfg = {}

    manifest: Mapping[str, object] = {}
    if manifest_path.exists():
        loaded = _load_json(manifest_path)
        if isinstance(loaded, Mapping):
            manifest = loaded
    labeling_sha256 = (
        _stable_sha256(_load_json(labeling_path))
        if labeling_path.exists()
        else None
    )
    if (
        labeling_sha256 is not None
        and manifest.get("labeling_sha256") not in (None, labeling_sha256)
    ):
        raise ValueError(
            "labeling.json does not match labeling_manifest.json; rebuild "
            "schema-v2 labeling before reusing downstream artifacts."
        )

    schema = (
        manifest.get("label_schema_version")
        or manifest.get("schema_version")
        or manifest.get("labeling_schema_version")
        or labeling_cfg.get("schema_version")
    )
    locator = (
        manifest.get("primary_locator")
        or manifest.get("labeling_primary_locator")
        or labeling_cfg.get("primary_locator")
    )
    sample_unit = (
        manifest.get("sample_unit")
        or manifest.get("labeling_sample_unit")
        or labeling_cfg.get("sample_unit")
    )
    return {
        "labeling_sha256": (
            labeling_sha256
        ),
        "labeling_manifest_sha256": (
            _file_sha256(manifest_path) if manifest_path.exists() else None
        ),
        "labeling_schema_version": (
            str(schema) if schema not in (None, "") else None
        ),
        "labeling_primary_locator": (
            str(locator) if locator not in (None, "") else None
        ),
        "labeling_sample_unit": (
            str(sample_unit) if sample_unit not in (None, "") else None
        ),
    }


def _stable_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reuse_generation_artifacts(
    source_dir: Path,
    output_dir: Path,
    *,
    model: str,
    model_cfg: Mapping[str, object],
    prompt: str,
    expected_image_ids: Sequence[int],
    resume: bool = False,
    adopt_legacy: bool = False,
) -> None:
    """Copy only generation artifacts after validating their full identity."""

    source = source_dir.resolve()
    destination = output_dir.resolve()
    if source == destination:
        raise ValueError(
            "--reuse-generations-from must point to a different output directory"
        )
    if not resume:
        _refuse_reuse_over_downstream_artifacts(destination)

    image_ids = [int(value) for value in expected_image_ids]
    source_generations_path = source / "generations.json"
    if source_generations_path.is_symlink():
        raise ValueError(
            f"Refusing a symlinked source generation file: {source_generations_path}"
        )
    if not source_generations_path.is_file():
        raise FileNotFoundError(
            f"Reusable generation artifact not found: {source_generations_path}"
        )
    source_generations = _load_json(source_generations_path)
    source_payload = canonical_generation_payload(
        source_generations,
        expected_image_ids=image_ids,
    )

    source_manifest_path = source / GENERATION_MANIFEST_NAME
    if source_manifest_path.is_symlink():
        raise ValueError(
            f"Refusing a symlinked source generation manifest: {source_manifest_path}"
        )
    if source_manifest_path.exists():
        source_manifest = _load_json(source_manifest_path)
        validate_generation_manifest(
            source_manifest,
            model=model,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_generations,
            expected_image_ids=image_ids,
        )
    else:
        source_manifest = build_generation_manifest(
            model=model,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_generations,
            expected_image_ids=image_ids,
        )
        print(
            "[Pipeline] Reusable generations have no manifest; validated "
            "content and cohort and will create one automatically."
        )

    target_generations_path = destination / "generations.json"
    if target_generations_path.is_symlink():
        raise ValueError(
            f"Refusing a symlinked target generation file: {target_generations_path}"
        )
    if target_generations_path.exists():
        target_generations = _load_json(target_generations_path)
        target_payload = canonical_generation_payload(
            target_generations,
            expected_image_ids=image_ids,
        )
        if stable_generation_sha256(target_payload) != stable_generation_sha256(
            source_payload
        ):
            raise ValueError(
                "Refusing --reuse-generations-from because target captions or "
                f"actual response_token_ids differ from source: {target_generations_path}"
            )
    else:
        _copy_reused_artifact(source_generations_path, target_generations_path)

    target_manifest_path = destination / GENERATION_MANIFEST_NAME
    if target_manifest_path.is_symlink():
        raise ValueError(
            f"Refusing a symlinked target generation manifest: {target_manifest_path}"
        )
    if target_manifest_path.exists():
        target_manifest = _load_json(target_manifest_path)
        validate_generation_manifest(
            target_manifest,
            model=model,
            model_cfg=model_cfg,
            prompt=prompt,
            generations=source_generations,
            expected_image_ids=image_ids,
        )
    else:
        target_manifest = dict(source_manifest)
        _atomic_write_json(target_manifest_path, target_manifest)

    source_split_path = source / "image_splits.json"
    copied_split = False
    if source_split_path.exists():
        if source_split_path.is_symlink():
            raise ValueError(f"Refusing a symlinked source split: {source_split_path}")
        source_split = _load_json(source_split_path)
        validate_strict_82_split(
            source_split,
            expected_image_ids=image_ids,
        )
        target_split_path = destination / "image_splits.json"
        if target_split_path.is_symlink():
            raise ValueError(f"Refusing a symlinked target split: {target_split_path}")
        if target_split_path.exists():
            target_split = _load_json(target_split_path)
            validate_strict_82_split(
                target_split,
                expected_image_ids=image_ids,
            )
            if _stable_sha256(target_split) != _stable_sha256(source_split):
                raise ValueError(
                    "Refusing --reuse-generations-from because the retained "
                    f"image split differs from source: {target_split_path}"
                )
        else:
            _copy_reused_artifact(source_split_path, target_split_path)
        copied_split = True

    print(
        "[Pipeline] Reused validated generations"
        + (" and strict image split" if copied_split else "")
        + f" from {source}"
    )


def _refuse_reuse_over_downstream_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    protected = [
        output_dir / "labeling.json",
        output_dir / "labeling_manifest.json",
        output_dir / "chair_summary.json",
        output_dir / "coco_ground_truth.jsonl",
        output_dir / "features.pkl",
        output_dir / "features_manifest.json",
        output_dir / "baseline",
        output_dir / "results",
        output_dir / "pipeline_manifest.json",
        output_dir / "resolved_pipeline_config.yaml",
        output_dir / "generation_shards",
        *output_dir.glob("features.part*.pkl"),
    ]
    existing = sorted({str(path) for path in protected if path.exists()})
    if existing:
        raise RuntimeError(
            "Refusing --reuse-generations-from without --resume because the "
            "target already contains downstream artifacts: "
            + ", ".join(existing)
        )


def _copy_reused_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _file_sha256(source) != _file_sha256(destination):
            raise ValueError(
                f"Refusing to overwrite different retained artifact: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.reuse.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _prepare_fresh_artifacts(
    output_dir: Path,
    run: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Honor --no-resume by removing only artifacts in requested stages."""
    stages = run["stages"]
    assert isinstance(stages, Mapping)
    if bool(stages.get("generation")):
        _remove_paths(
            output_dir / "generations.json",
            output_dir / GENERATION_MANIFEST_NAME,
            output_dir / "labeling.json",
            output_dir / "labeling_manifest.json",
            output_dir / "generation_shards",
        )
    elif bool(stages.get("labeling")):
        _remove_paths(
            output_dir / "labeling.json",
            output_dir / "labeling_manifest.json",
        )

    flags = _effective_feature_flags(config)
    if bool(stages.get("feature_extraction")):
        if flags["method"] or flags["ads_cgc"]:
            _remove_paths(output_dir / "features.pkl")
            for path in output_dir.glob("features.part*.pkl"):
                _remove_paths(path)
        if flags["baseline"]:
            extraction = config.get("feature_extraction") or {}
            baseline = extraction.get("baseline") or {} if isinstance(extraction, Mapping) else {}
            subdir = str(
                baseline.get("output_subdir", "baseline")
                if isinstance(baseline, Mapping)
                else "baseline"
            )
            baseline_dir = output_dir / subdir
            _remove_paths(
                baseline_dir / "features.pkl",
                baseline_dir / "feature_parts",
                baseline_dir / "dhcp",
                baseline_dir / "halloc",
                baseline_dir / "svar_official",
            )
            for path in baseline_dir.glob("features.part*.pkl"):
                _remove_paths(path)

    if bool(stages.get("training")):
        if flags["method"] or flags["ads_cgc"]:
            _remove_paths(output_dir / "results")
        if flags["baseline"]:
            extraction = config.get("feature_extraction") or {}
            baseline = extraction.get("baseline") or {} if isinstance(extraction, Mapping) else {}
            subdir = str(
                baseline.get("output_subdir", "baseline")
                if isinstance(baseline, Mapping)
                else "baseline"
            )
            _remove_paths(output_dir / subdir / "results")
            _remove_paths(output_dir / subdir / "checkpoints")


def _remove_paths(*paths: Path) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _directory_has_files(path: Path) -> bool:
    """Return true for a non-empty shard/cache directory, not an empty shell."""
    return path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))


def _apply_cli_overrides(environ: dict[str, str], args: argparse.Namespace) -> None:
    overrides = {
        "MODEL": args.model,
        "OUTPUT": args.output_dir,
        "PROMPT": args.prompt,
        "EXTRACTION_MODE": args.extraction_mode,
    }
    for key, value in overrides.items():
        if value is not None:
            environ[key] = str(value)
    if args.resume is not None:
        environ["RESUME"] = "true" if args.resume else "false"
    if args.stages is not None:
        environ["STAGES"] = " ".join(args.stages)


def _run(
    command: Sequence[str],
    *,
    dry_run: bool,
    require_script: bool = False,
) -> None:
    if require_script and len(command) > 1:
        script_path = _repo_path(command[1])
        if not script_path.exists():
            raise FileNotFoundError(
                f"Configured pipeline stage is not installed yet: {script_path}"
            )
    print("[Pipeline] $ " + shlex.join([str(item) for item in command]))
    if not dry_run:
        subprocess.run([str(item) for item in command], check=True, cwd=str(REPO_ROOT))


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _atomic_write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()
