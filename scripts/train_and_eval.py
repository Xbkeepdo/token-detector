#!/usr/bin/env python3
"""Train every feature family selected by the unified YAML configuration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_pipeline import _enabled_method_feature_sets
from utils.config_utils import extraction_mode_flags, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs_unified.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        help="Train only these configured-compatible feature sets.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Isolate seed outputs and the aggregate summary under this run name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configured training commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commands = build_training_commands(
        config=config,
        model=args.model,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        feature_sets_override=args.feature_sets,
        run_name=args.run_name,
    )
    if not commands:
        raise ValueError("YAML configuration enables no trainable feature family")
    for command in commands:
        print("[Train] $ " + shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=str(Path(__file__).resolve().parents[1]))
    print("[Train] Completed configured training and evaluation.")


def build_training_commands(
    *,
    config: dict,
    model: str,
    config_path: str,
    output_dir: str,
    device: str,
    feature_sets_override: Sequence[str] | None = None,
    run_name: str | None = None,
) -> list[list[str]]:
    flags = _configured_family_flags(config)
    training = config.get("training") or {}
    if not isinstance(training, Mapping):
        raise ValueError("training must be a YAML mapping")

    commands: list[list[str]] = []
    if flags["method"] or flags["ads_cgc"]:
        feature_sets = (
            _enabled_method_feature_sets(config, feature_sets_override)
            if feature_sets_override is not None
            else _configured_feature_sets(config, flags)
        )
        if not feature_sets:
            raise ValueError("No feature sets remain after applying DGST branch switches")
        trainer = str(
            training.get(
                "trainer",
                (config.get("run") or {}).get("trainer", "torch_mlp"),
            )
        ).strip().lower()
        common = [
            "--model",
            str(model),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--feature-sets",
            *feature_sets,
        ]
        if trainer in {"torch_mlp", "torch_probe"}:
            positive_class = str(
                training.get(
                    "positive_class",
                    (config.get("run") or {}).get("positive_class", "real"),
                )
            )
            probe_config = training.get("torch_probe") or {}
            seeds = _torch_probe_seeds(probe_config)
            probe_options = _torch_probe_cli_args(probe_config)
            for seed in seeds:
                command = [
                    sys.executable,
                    "scripts/train_torch_probe_feature_sets.py",
                    *common,
                    "--device",
                    str(device),
                    "--positive-class",
                    positive_class,
                    *probe_options,
                    "--seed",
                    str(seed),
                ]
                if run_name is not None:
                    command.extend(["--run-name", f"{run_name}_seed{seed}"])
                elif len(seeds) > 1:
                    command.extend(["--run-name", f"seed{seed}"])
                commands.append(command)
            if len(seeds) > 1:
                result_root = Path(output_dir) / "results"
                seed_run_template = (
                    f"{run_name}_seed{{seed}}" if run_name is not None else "seed{seed}"
                )
                summary_stem = (
                    f"{model}_{run_name}_{len(seeds)}seed_summary"
                    if run_name is not None
                    else f"{model}_selected_feature_sets_{len(seeds)}seed_summary"
                )
                commands.append(
                    [
                        sys.executable,
                        "scripts/summarize_torch_probe_seed_runs.py",
                        "--models",
                        str(model),
                        "--seeds",
                        *[str(seed) for seed in seeds],
                        "--run-template",
                        str(
                            result_root
                            / seed_run_template
                            / "{model}_selected_feature_sets.json"
                        ),
                        "--output-prefix",
                        str(
                            result_root
                            / summary_stem
                        ),
                        "--title",
                        f"{model} method + ADS/CGC {len(seeds)}-seed Torch MLP summary",
                    ]
                )
        elif trainer in {"sklearn", "xgb_rf"}:
            if run_name is not None:
                raise ValueError("--run-name currently requires the torch_mlp trainer")
            commands.append(
                [sys.executable, "scripts/train_feature_sets.py", *common]
            )
        else:
            raise ValueError(
                "training.trainer must be torch_mlp, torch_probe, sklearn, or xgb_rf"
            )

    if flags["baseline"]:
        commands.append(
            [
                sys.executable,
                "scripts/train_baselines.py",
                "--model",
                str(model),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--device",
                str(device),
            ]
        )
    return commands


def _configured_family_flags(config: Mapping[str, object]) -> dict[str, bool]:
    run = config.get("run") or {}
    if not isinstance(run, Mapping):
        raise ValueError("run must be a YAML mapping")
    selected = extraction_mode_flags(str(run.get("extraction_mode", "all")))
    extraction = config.get("feature_extraction") or {}
    if not isinstance(extraction, Mapping):
        raise ValueError("feature_extraction must be a YAML mapping")

    fallback = {
        "method": "dgst_t" in extraction,
        "ads_cgc": "ads" in extraction or "cgc" in extraction,
        "baseline": False,
    }
    for family in selected:
        section = extraction.get(family)
        if isinstance(section, Mapping):
            configured = bool(section.get("enabled", True))
        elif section is None:
            configured = fallback[family]
        else:
            configured = bool(section)
        selected[family] = bool(selected[family] and configured)
    return selected


def _configured_feature_sets(
    config: Mapping[str, object],
    flags: Mapping[str, bool],
) -> list[str]:
    training = config.get("training") or {}
    configured = training.get("feature_sets") if isinstance(training, Mapping) else None
    values: list[str] = []
    if isinstance(configured, Mapping):
        if flags["method"]:
            method_values = _as_string_list(configured.get("method") or [])
            values.extend(_enabled_method_feature_sets(config, method_values))
        if flags["ads_cgc"]:
            values.extend(_as_string_list(configured.get("ads_cgc") or []))
    elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        values.extend(_as_string_list(configured))
    if not values:
        raise ValueError(
            "No root training.feature_sets remain for the YAML extraction mode"
        )
    return list(dict.fromkeys(values))


def _as_string_list(values: Sequence[object]) -> list[str]:
    return [str(value) for value in values]


def _torch_probe_cli_args(config: object) -> list[str]:
    if not isinstance(config, Mapping):
        raise ValueError("training.torch_probe must be a YAML mapping")
    scalar_options = {
        "batch_size": "--batch-size",
        "num_epochs": "--num-epochs",
        "max_epochs": "--num-epochs",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "lr_factor": "--lr-factor",
        "lr_patience": "--lr-patience",
        "early_stopping_patience": "--early-stopping-patience",
        "fixed_threshold": "--fixed-threshold",
        "dropout": "--dropout",
    }
    protocol_options = {
        "structure": "Linear-BatchNorm-ReLU-Dropout",
        "activation": "relu",
        "batch_norm": True,
        "initialization": "kaiming_uniform_relu",
        "output_dim": 1,
        "optimizer": "adam",
        "loss": "bce_with_logits",
        "scheduler_monitor": "train_loss",
        "early_stopping_monitor": "train_loss",
        "checkpoint_selection": "minimum_train_loss",
        "feature_normalization": "none",
    }
    allowed = {
        *scalar_options,
        *protocol_options,
        "threshold_reporting",
        "hidden_sizes",
        "paper_config",
        "seed",
        "seeds",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown training.torch_probe options: {unknown}")
    mismatches = {
        key: {"expected": expected, "found": config.get(key)}
        for key, expected in protocol_options.items()
        if key in config and config.get(key) != expected
    }
    reporting = config.get("threshold_reporting")
    if reporting is not None and list(reporting) != ["fixed_0.5", "train_f1"]:
        mismatches["threshold_reporting"] = {
            "expected": ["fixed_0.5", "train_f1"],
            "found": reporting,
        }
    if mismatches:
        raise ValueError(f"Unsupported default Torch probe protocol: {mismatches}")
    result: list[str] = []
    for key, option in scalar_options.items():
        if key in config:
            result.extend([option, str(config[key])])
    if config.get("hidden_sizes") is not None:
        hidden_sizes = config["hidden_sizes"]
        if not isinstance(hidden_sizes, Sequence) or isinstance(hidden_sizes, (str, bytes)):
            raise ValueError("training.torch_probe.hidden_sizes must be a list")
        result.extend(["--hidden-sizes", *[str(value) for value in hidden_sizes]])
    if bool(config.get("paper_config", False)):
        result.append("--paper-config")
    return result


def _torch_probe_seeds(config: object) -> list[int]:
    if not isinstance(config, Mapping):
        raise ValueError("training.torch_probe must be a YAML mapping")
    values = config.get("seeds")
    if values is None:
        values = (
            [config["seed"]]
            if config.get("seed") is not None
            else [43, 44, 45]
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("training.torch_probe.seeds must be a non-empty list")
    seeds = list(dict.fromkeys(int(value) for value in values))
    if not seeds:
        raise ValueError("At least one torch probe seed is required")
    return seeds


if __name__ == "__main__":
    main()
