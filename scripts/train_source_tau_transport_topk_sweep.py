#!/usr/bin/env python3
"""Train the YAML source-tau x transport-top-K four-gate sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.train_feature_sets import feature_block
from scripts.train_torch_probe_feature_sets import (
    TorchProbeConfig,
    _json_ready,
    _resolve_device,
    train_and_evaluate_probe,
)
from utils.config_utils import load_config
from utils.io_utils import load_json, load_pkl
from utils.split_utils import validate_strict_82_split


FEATURE_METHODS = {
    "hpre_raw_logit_gauss": {
        "label": "hpre raw-logit Gaussian",
        # Preserve the historical feature names for existing raw-logit runs.
        "feature_suffix": "hpre_risk_plus_ev",
    },
    "hpre_softmax_prob_gauss": {
        "label": "hpre softmax-prob Gaussian",
        "feature_suffix": "hpre_softmax_prob_gauss_risk_plus_ev",
    },
}
RISK_MODES = (
    "fixed_topk",
    "capped_topmass_085",
    "capped_topmass_alpha_sweep",
)

DEFAULT_RUN_NAMES = {
    "hpre_raw_logit_gauss": (
        "source_tau_x_transport_topk_full_hpre_risk_plus_full_ev"
    ),
    "hpre_softmax_prob_gauss": (
        "source_tau_x_transport_topk_full_hpre_softmax_prob_gauss_"
        "risk_plus_full_ev"
    ),
}


def feature_method_spec(method: str) -> dict:
    base = FEATURE_METHODS[method]
    return {
        **base,
        "risk_key": f"dgst_t_{method}_risk_sqrt_hpre_per_layer",
        "capped_risk_key": (
            f"dgst_t_{method}_risk_sqrt_hpre_"
            "capped_topmass_085_per_layer"
        ),
        "scope_blocks": {
            "vv": {
                "ev": f"{method}_ev_target_dist_mass_x_cosine",
                "capped_ev": (
                    f"{method}_ev_target_dist_mass_x_cosine_"
                    "capped_topmass_085"
                ),
            },
            "vpend": {
                "ev": f"vpend_{method}_ev_target_dist_mass_x_cosine",
                "capped_ev": (
                    f"vpend_{method}_ev_target_dist_mass_x_cosine_"
                    "capped_topmass_085"
                ),
            },
        },
    }


def capped_topmass_alpha_slug(alpha: float) -> str:
    value = float(alpha)
    percent = value * 100.0
    rounded_percent = int(round(percent))
    if np.isclose(percent, rounded_percent, rtol=0.0, atol=1e-9):
        return f"capped_topmass_{rounded_percent:03d}"
    numeric = format(value, ".12g").replace("-", "m").replace(".", "p")
    return f"capped_topmass_a{numeric}"


def configured_capped_topmass_alphas(config_root: dict) -> tuple[float, ...]:
    dgst_cfg = ((config_root.get("feature_extraction") or {}).get("dgst_t") or {})
    raw_values = dgst_cfg.get("capped_topmass_alphas")
    if raw_values is None:
        raw_values = [dgst_cfg.get("capped_topmass_085_alpha", 0.85)]
    elif isinstance(raw_values, (int, float)):
        raw_values = [raw_values]
    elif not isinstance(raw_values, (list, tuple)):
        raise ValueError(
            "feature_extraction.dgst_t.capped_topmass_alphas must be a number or list"
        )
    values = tuple(dict.fromkeys(float(value) for value in raw_values))
    if not values or any(
        not np.isfinite(value) or not (0.0 < value <= 1.0)
        for value in values
    ):
        raise ValueError(
            "capped_topmass_alphas must contain finite values in (0, 1]"
        )
    slugs = [capped_topmass_alpha_slug(value) for value in values]
    if len(slugs) != len(set(slugs)):
        raise ValueError("capped_topmass_alphas contain colliding field-name slugs")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--layer-start",
        type=int,
        default=0,
        help="Risk layer start (inclusive); defaults to the first layer.",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help=(
            "Risk layer end (exclusive); defaults to the full curve length "
            "for the selected model."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature-method",
        choices=sorted(FEATURE_METHODS),
        default=None,
        help=(
            "Swept risk and matching full-layer EV feature family; manual "
            "runs default to hpre_raw_logit_gauss."
        ),
    )
    parser.add_argument(
        "--risk-modes",
        nargs="+",
        choices=RISK_MODES,
        default=None,
        help=(
            "Risk support selections; manual runs default to fixed_topk. "
            "Automatic runs read risk_modes from YAML."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Result directory name; defaults to a full-layer method-specific name.",
    )
    parser.add_argument(
        "--if-enabled",
        action="store_true",
        help=(
            "Obey training.source_tau_transport_topk_sweep.enabled and train "
            "all YAML-configured feature methods. Intended for run.sh."
        ),
    )
    return parser.parse_args()


def configured_training_methods(config_root: dict) -> list[str]:
    training = config_root.get("training") or {}
    sweep_cfg = training.get("source_tau_transport_topk_sweep") or {}
    if not isinstance(sweep_cfg, dict):
        raise ValueError(
            "training.source_tau_transport_topk_sweep must be a mapping"
        )
    enabled = sweep_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            "training.source_tau_transport_topk_sweep.enabled must be a boolean"
        )
    if not enabled:
        return []
    raw_methods = sweep_cfg.get("feature_methods")
    if raw_methods is None:
        raw_methods = ["hpre_raw_logit_gauss"]
    elif isinstance(raw_methods, str):
        raw_methods = [raw_methods]
    elif not isinstance(raw_methods, (list, tuple)):
        raise ValueError(
            "training.source_tau_transport_topk_sweep.feature_methods must "
            "be a string or list"
        )
    methods = list(dict.fromkeys(str(method) for method in raw_methods))
    if not methods:
        raise ValueError(
            "Enabled source-tau/Top-K sweep training needs at least one "
            "feature method"
        )
    unknown = sorted(set(methods) - set(FEATURE_METHODS))
    if unknown:
        raise ValueError(
            f"Unsupported sweep feature methods {unknown}; expected a subset "
            f"of {sorted(FEATURE_METHODS)}"
        )
    return methods


def configured_training_scopes(config_root: dict) -> tuple[str, ...]:
    dgst_cfg = ((config_root.get("feature_extraction") or {}).get("dgst_t") or {})
    raw_scopes = dgst_cfg.get("support_modes", ["vv"])
    if isinstance(raw_scopes, str):
        raw_scopes = [raw_scopes]
    scopes = [str(scope).strip().lower() for scope in raw_scopes]
    selected = [scope for scope in ("vv", "vpend") if scope in scopes]
    if not selected:
        raise ValueError(
            "Automatic sweep training requires support_modes to include vv "
            "and/or vpend"
        )
    return tuple(selected)


def configured_training_risk_modes(config_root: dict) -> tuple[str, ...]:
    training = config_root.get("training") or {}
    sweep_cfg = training.get("source_tau_transport_topk_sweep") or {}
    raw_modes = sweep_cfg.get("risk_modes", ["fixed_topk"])
    if isinstance(raw_modes, str):
        raw_modes = [raw_modes]
    elif not isinstance(raw_modes, (list, tuple)):
        raise ValueError(
            "training.source_tau_transport_topk_sweep.risk_modes must be a "
            "string or list"
        )
    modes = tuple(dict.fromkeys(str(mode) for mode in raw_modes))
    if not modes:
        raise ValueError("Enabled sweep training needs at least one risk mode")
    unknown = sorted(set(modes) - set(RISK_MODES))
    if unknown:
        raise ValueError(
            f"Unsupported sweep risk modes {unknown}; expected a subset of "
            f"{list(RISK_MODES)}"
        )
    if {"capped_topmass_085", "capped_topmass_alpha_sweep"} & set(modes):
        dgst_cfg = (
            (config_root.get("feature_extraction") or {}).get("dgst_t") or {}
        )
        if not bool(dgst_cfg.get("compute_capped_topmass_085", False)):
            raise ValueError(
                "capped top-mass sweep training requires "
                "feature_extraction.dgst_t.compute_capped_topmass_085=true"
            )
    if "capped_topmass_alpha_sweep" in modes:
        configured_capped_topmass_alphas(config_root)
    return modes


def probe_config_from_yaml(config_path: str, seed: int) -> TorchProbeConfig:
    root = load_config(config_path)
    training = root.get("training") or {}
    cfg = training.get("torch_probe") or {}
    return TorchProbeConfig(
        hidden_sizes=tuple(int(x) for x in cfg.get("hidden_sizes", [128, 64, 32])),
        dropout=float(cfg.get("dropout", 0.3)),
        drop_last=bool(cfg.get("drop_last", False)),
        batch_size=int(cfg.get("batch_size", 256)),
        num_epochs=int(cfg.get("max_epochs", cfg.get("num_epochs", 100))),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        lr_factor=float(cfg.get("lr_factor", 0.5)),
        lr_patience=int(cfg.get("lr_patience", 5)),
        early_stopping_patience=int(cfg.get("early_stopping_patience", 10)),
        seed=int(seed),
        positive_class="real",
        split_protocol=str(training.get("split_protocol", "strict_82_no_validation")),
        threshold_selection=str(training.get("threshold_selection", "train_f1")),
        fixed_threshold=float(cfg.get("fixed_threshold", 0.5)),
        threshold_reporting=tuple(cfg.get("threshold_reporting", ["fixed_0.5", "train_f1"])),
        checkpoint_selection=str(cfg.get("checkpoint_selection", "minimum_train_loss")),
    )


def build_matrices(
    rows: list[dict],
    expected_variants: dict[str, tuple[float, int]],
    layer_start: int,
    layer_end: int | None,
    feature_method: str = "hpre_raw_logit_gauss",
    scopes: Sequence[str] = ("vv", "vpend"),
    risk_modes: Sequence[str] = ("fixed_topk",),
    capped_topmass_alphas: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    spec = feature_method_spec(feature_method)
    unknown_scopes = sorted(set(scopes) - set(spec["scope_blocks"]))
    if unknown_scopes:
        raise ValueError(f"Unsupported sweep scopes: {unknown_scopes}")
    scope_blocks = {
        scope: spec["scope_blocks"][scope]
        for scope in dict.fromkeys(scopes)
    }
    selected_risk_modes = tuple(dict.fromkeys(str(mode) for mode in risk_modes))
    unknown_risk_modes = sorted(set(selected_risk_modes) - set(RISK_MODES))
    if not selected_risk_modes or unknown_risk_modes:
        raise ValueError(
            f"Invalid sweep risk modes {list(risk_modes)}; expected a non-empty "
            f"subset of {list(RISK_MODES)}"
        )
    risk_key = spec["risk_key"]
    capped_risk_key = spec["capped_risk_key"]
    feature_suffix = spec["feature_suffix"]
    capped_alpha_values = tuple(
        dict.fromkeys(float(value) for value in (capped_topmass_alphas or ()))
    )
    if any(
        not np.isfinite(value) or not (0.0 < value <= 1.0)
        for value in capped_alpha_values
    ):
        raise ValueError("capped Top-Mass alphas must be finite values in (0, 1]")
    capped_alpha_specs = tuple(
        (capped_topmass_alpha_slug(value), value)
        for value in capped_alpha_values
    )
    if (
        "capped_topmass_alpha_sweep" in selected_risk_modes
        and not capped_alpha_specs
    ):
        raise ValueError(
            "capped_topmass_alpha_sweep requires at least one capped Top-Mass alpha"
        )
    if len({slug for slug, _value in capped_alpha_specs}) != len(capped_alpha_specs):
        raise ValueError("capped Top-Mass alphas contain colliding field-name slugs")

    def dynamic_capped_risk_key(alpha_slug: str) -> str:
        return (
            f"dgst_t_{feature_method}_risk_sqrt_hpre_"
            f"{alpha_slug}_per_layer"
        )

    def dynamic_capped_ev_block(scope: str, alpha_slug: str) -> str:
        prefix = "" if scope == "vv" else f"{scope}_"
        return (
            f"{prefix}{feature_method}_ev_target_dist_mass_x_cosine_"
            f"{alpha_slug}_hpre"
        )

    tau_variant_groups: dict[float, list[str]] = {}
    for variant_slug, (source_tau, _top_k) in expected_variants.items():
        tau_variant_groups.setdefault(float(source_tau), []).append(variant_slug)
    values = {}
    if "fixed_topk" in selected_risk_modes:
        values.update(
            {
                f"{scope}_{variant_slug}_{feature_suffix}": []
                for scope in scope_blocks
                for variant_slug in expected_variants
            }
        )
    if "capped_topmass_085" in selected_risk_modes:
        values.update(
            {
                f"{scope}_tau{format(source_tau, '.12g').replace('-', 'm').replace('.', 'p')}_"
                f"capped_topmass_085_{feature_suffix}": []
                for scope in scope_blocks
                for source_tau in tau_variant_groups
            }
        )
    if "capped_topmass_alpha_sweep" in selected_risk_modes:
        values.update(
            {
                f"{scope}_tau{format(source_tau, '.12g').replace('-', 'm').replace('.', 'p')}_"
                f"{alpha_slug}_{feature_suffix}": []
                for scope in scope_blocks
                for source_tau in tau_variant_groups
                for alpha_slug, _alpha in capped_alpha_specs
            }
        )
    labels = []
    image_ids = []
    risk_dims = set()
    ev_dims = set()
    resolved_layer_end = layer_end
    use_full_risk_curve = layer_end is None
    for index, row in enumerate(rows):
        label = row.get("label")
        if label not in (0, 1):
            continue
        row_sweep = row.get("dgst_t_hparam_sweep")
        if not isinstance(row_sweep, dict):
            raise KeyError(f"Missing dgst_t_hparam_sweep at row {index}")
        if set(row_sweep) != set(expected_variants):
            raise ValueError(
                f"Sweep variants differ at row {index}: "
                f"{sorted(row_sweep)} != {sorted(expected_variants)}"
            )
        for scope, blocks in scope_blocks.items():
            ev_by_mode = {}
            for risk_mode, block_name in (
                ("fixed_topk", "ev"),
                ("capped_topmass_085", "capped_ev"),
            ):
                if risk_mode not in selected_risk_modes:
                    continue
                ev = feature_block(row, blocks[block_name]).astype(
                    np.float32, copy=False
                )
                if ev.ndim != 1 or not np.all(np.isfinite(ev)):
                    raise ValueError(
                        f"Invalid {scope}/{risk_mode} EV curve at row {index}"
                    )
                ev_by_mode[risk_mode] = ev
            capped_alpha_ev = {}
            if "capped_topmass_alpha_sweep" in selected_risk_modes:
                for alpha_slug, _alpha in capped_alpha_specs:
                    block_name = dynamic_capped_ev_block(scope, alpha_slug)
                    ev_key = f"dgst_t_{block_name}_per_layer"
                    try:
                        ev = np.asarray(row[ev_key], dtype=np.float32)
                    except KeyError as exc:
                        raise KeyError(
                            f"Missing {scope}/{alpha_slug} EV feature {ev_key} "
                            f"at row {index}; re-extract with the configured "
                            "capped alphas"
                        ) from exc
                    if ev.ndim != 1 or not np.all(np.isfinite(ev)):
                        raise ValueError(
                            f"Invalid {scope}/{alpha_slug} EV curve at row {index}"
                        )
                    capped_alpha_ev[alpha_slug] = ev
            for variant_slug, expected in expected_variants.items():
                variant = row_sweep[variant_slug]
                actual = (
                    float(variant["source_tau"]),
                    int(variant["transport_top_k"]),
                )
                if actual != expected:
                    raise ValueError(
                        f"Variant metadata mismatch at row {index}: "
                        f"{variant_slug}={actual}, expected {expected}"
                    )
                if "fixed_topk" not in selected_risk_modes:
                    continue
                ev = ev_by_mode["fixed_topk"]
                try:
                    risk = np.asarray(
                        variant[scope][risk_key], dtype=np.float32
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Missing {scope}/{variant_slug}/{risk_key} "
                        f"at row {index}"
                    ) from exc
                if risk.ndim != 1 or not np.all(np.isfinite(risk)):
                    raise ValueError(
                        f"Invalid {scope}/{variant_slug} risk at row {index}"
                    )
                if risk.shape != ev.shape:
                    raise ValueError(
                        f"{scope}/{variant_slug} risk/EV shape mismatch: "
                        f"{risk.shape} != {ev.shape}"
                    )
                if resolved_layer_end is None:
                    resolved_layer_end = int(risk.size)
                elif use_full_risk_curve and risk.size != resolved_layer_end:
                    raise ValueError(
                        "Full-layer sweep training requires one consistent "
                        f"risk dimension, got {resolved_layer_end} and {risk.size}"
                    )
                if (
                    layer_start < 0
                    or resolved_layer_end > risk.size
                    or resolved_layer_end <= layer_start
                ):
                    raise ValueError(
                        f"Requested risk slice [{layer_start},{resolved_layer_end}) "
                        f"outside curve length {risk.size}"
                    )
                risk_dims.add(int(risk.size))
                ev_dims.add(int(ev.size))
                values[
                    f"{scope}_{variant_slug}_{feature_suffix}"
                ].append(
                    np.concatenate([risk[layer_start:resolved_layer_end], ev]).astype(
                        np.float32
                    )
                )
            if "capped_topmass_085" in selected_risk_modes:
                ev = ev_by_mode["capped_topmass_085"]
                for source_tau, variant_slugs in tau_variant_groups.items():
                    representative_slug = variant_slugs[0]
                    try:
                        risk = np.asarray(
                            row_sweep[representative_slug][scope][capped_risk_key],
                            dtype=np.float32,
                        )
                    except KeyError as exc:
                        raise KeyError(
                            f"Missing {scope}/{representative_slug}/"
                            f"{capped_risk_key} at row {index}; re-extract sweep "
                            "features with capped support enabled"
                        ) from exc
                    if risk.ndim != 1 or not np.all(np.isfinite(risk)):
                        raise ValueError(
                            f"Invalid {scope}/tau={source_tau} capped risk at "
                            f"row {index}"
                        )
                    if risk.shape != ev.shape:
                        raise ValueError(
                            f"{scope}/tau={source_tau} capped risk/EV shape "
                            f"mismatch: {risk.shape} != {ev.shape}"
                        )
                    for duplicate_slug in variant_slugs[1:]:
                        duplicate = np.asarray(
                            row_sweep[duplicate_slug][scope][capped_risk_key],
                            dtype=np.float32,
                        )
                        if not np.array_equal(risk, duplicate):
                            raise ValueError(
                                f"Capped sweep risk unexpectedly depends on fixed "
                                f"Top-K for {scope}/tau={source_tau} at row {index}"
                            )
                    if resolved_layer_end is None:
                        resolved_layer_end = int(risk.size)
                    elif use_full_risk_curve and risk.size != resolved_layer_end:
                        raise ValueError(
                            "Full-layer sweep training requires one consistent "
                            f"risk dimension, got {resolved_layer_end} and "
                            f"{risk.size}"
                        )
                    if (
                        layer_start < 0
                        or resolved_layer_end > risk.size
                        or resolved_layer_end <= layer_start
                    ):
                        raise ValueError(
                            f"Requested risk slice [{layer_start},"
                            f"{resolved_layer_end}) outside curve length "
                            f"{risk.size}"
                        )
                    risk_dims.add(int(risk.size))
                    ev_dims.add(int(ev.size))
                    tau_label = format(source_tau, ".12g").replace(
                        "-", "m"
                    ).replace(".", "p")
                    values[
                        f"{scope}_tau{tau_label}_capped_topmass_085_"
                        f"{feature_suffix}"
                    ].append(
                        np.concatenate(
                            [risk[layer_start:resolved_layer_end], ev]
                        ).astype(np.float32)
                    )
            if "capped_topmass_alpha_sweep" in selected_risk_modes:
                for alpha_slug, _alpha in capped_alpha_specs:
                    ev = capped_alpha_ev[alpha_slug]
                    alpha_risk_key = dynamic_capped_risk_key(alpha_slug)
                    for source_tau, variant_slugs in tau_variant_groups.items():
                        representative_slug = variant_slugs[0]
                        try:
                            risk = np.asarray(
                                row_sweep[representative_slug][scope][
                                    alpha_risk_key
                                ],
                                dtype=np.float32,
                            )
                        except KeyError as exc:
                            raise KeyError(
                                f"Missing {scope}/{representative_slug}/"
                                f"{alpha_risk_key} at row {index}; re-extract sweep "
                                "features with the configured capped alphas"
                            ) from exc
                        if risk.ndim != 1 or not np.all(np.isfinite(risk)):
                            raise ValueError(
                                f"Invalid {scope}/tau={source_tau}/{alpha_slug} "
                                f"capped risk at row {index}"
                            )
                        if risk.shape != ev.shape:
                            raise ValueError(
                                f"{scope}/tau={source_tau}/{alpha_slug} capped "
                                f"risk/EV shape mismatch: {risk.shape} != {ev.shape}"
                            )
                        for duplicate_slug in variant_slugs[1:]:
                            duplicate = np.asarray(
                                row_sweep[duplicate_slug][scope][alpha_risk_key],
                                dtype=np.float32,
                            )
                            if not np.array_equal(risk, duplicate):
                                raise ValueError(
                                    "Capped alpha-sweep risk unexpectedly depends "
                                    f"on fixed Top-K for {scope}/tau={source_tau}/"
                                    f"{alpha_slug} at row {index}"
                                )
                        if resolved_layer_end is None:
                            resolved_layer_end = int(risk.size)
                        elif use_full_risk_curve and risk.size != resolved_layer_end:
                            raise ValueError(
                                "Full-layer sweep training requires one consistent "
                                f"risk dimension, got {resolved_layer_end} and "
                                f"{risk.size}"
                            )
                        if (
                            layer_start < 0
                            or resolved_layer_end > risk.size
                            or resolved_layer_end <= layer_start
                        ):
                            raise ValueError(
                                f"Requested risk slice [{layer_start},"
                                f"{resolved_layer_end}) outside curve length "
                                f"{risk.size}"
                            )
                        risk_dims.add(int(risk.size))
                        ev_dims.add(int(ev.size))
                        tau_label = format(source_tau, ".12g").replace(
                            "-", "m"
                        ).replace(".", "p")
                        values[
                            f"{scope}_tau{tau_label}_{alpha_slug}_"
                            f"{feature_suffix}"
                        ].append(
                            np.concatenate(
                                [risk[layer_start:resolved_layer_end], ev]
                            ).astype(np.float32)
                        )
        labels.append(int(label))
        image_ids.append(int(row["image_id"]))
        if len(labels) % 2000 == 0:
            print(f"[derive] processed {len(labels)} rows", flush=True)
    if resolved_layer_end is None:
        raise ValueError("No binary feature rows were available for sweep training")
    matrices = {
        name: np.stack(items).astype(np.float32, copy=False)
        for name, items in values.items()
    }
    return (
        matrices,
        np.asarray(labels, dtype=np.int32),
        np.asarray(image_ids, dtype=np.int64),
        {
            "rows_total": len(rows),
            "rows_binary": len(labels),
            "feature_method": feature_method,
            "scopes": list(scope_blocks),
            "risk_modes": list(selected_risk_modes),
            "risk_feature_keys": {
                "fixed_topk": risk_key,
                "capped_topmass_085": capped_risk_key,
                "capped_topmass_alpha_sweep": {
                    alpha_slug: dynamic_capped_risk_key(alpha_slug)
                    for alpha_slug, _alpha in capped_alpha_specs
                },
            },
            "ev_feature_blocks": {
                scope: {
                    "fixed_topk": blocks["ev"],
                    "capped_topmass_085": blocks["capped_ev"],
                    "capped_topmass_alpha_sweep": {
                        alpha_slug: (
                            f"dgst_t_{dynamic_capped_ev_block(scope, alpha_slug)}"
                            "_per_layer"
                        )
                        for alpha_slug, _alpha in capped_alpha_specs
                    },
                }
                for scope, blocks in scope_blocks.items()
            },
            "capped_topmass_alphas": list(capped_alpha_values),
            "capped_topmass_alpha_by_slug": {
                alpha_slug: alpha
                for alpha_slug, alpha in capped_alpha_specs
            },
            "risk_curve_dimensions": sorted(risk_dims),
            "ev_curve_dimensions": sorted(ev_dims),
            "risk_layer_slice": [int(layer_start), int(resolved_layer_end)],
            "risk_layers_inclusive": [int(layer_start), int(resolved_layer_end - 1)],
            "selected_risk_dimension": int(resolved_layer_end - layer_start),
            "variants": {
                slug: {"source_tau": tau, "transport_top_k": top_k}
                for slug, (tau, top_k) in expected_variants.items()
            },
        },
    )


def extract_report(metrics: dict, report: str) -> dict:
    if report == "train_f1":
        item = metrics
        threshold = metrics["decision_threshold"]
    else:
        item = metrics["threshold_reports"]["fixed_0.5"]["test_metrics"]
        threshold = 0.5
    return {
        "threshold": float(threshold),
        "auc": float(item["real_positive"]["auc"]),
        "real_aupr": float(item["real_positive"]["aupr"]),
        "hall_aupr": float(item["hallucination_positive"]["aupr"]),
        "real_f1": float(item["real_positive"]["f1"]),
        "hall_f1": float(item["hallucination_positive"]["f1"]),
        "accuracy": float(item["accuracy"]),
    }


def train_all(
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    image_ids: np.ndarray,
    splits: dict,
    seeds: list[int],
    config_path: str,
    device_name: str,
    result_dir: Path,
) -> tuple[list[dict], dict]:
    train_ids = {int(x) for x in splits["train"]}
    test_ids = {int(x) for x in splits["test"]}
    train_mask = np.asarray([int(x) in train_ids for x in image_ids], dtype=bool)
    test_mask = np.asarray([int(x) in test_ids for x in image_ids], dtype=bool)
    if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
        raise ValueError("Invalid strict train/test masks")
    if set(np.unique(labels[train_mask]).tolist()) != {0, 1}:
        raise ValueError("Train token split is not binary")
    if set(np.unique(labels[test_mask]).tolist()) != {0, 1}:
        raise ValueError("Test token split is not binary")
    device = _resolve_device(device_name)
    base_config = probe_config_from_yaml(config_path, seeds[0])
    print(
        f"[train] device={device} train_tokens={int(train_mask.sum())} "
        f"test_tokens={int(test_mask.sum())} config={asdict(base_config)}",
        flush=True,
    )
    records = []
    raw_metrics = {}
    for seed in seeds:
        cfg = replace(base_config, seed=int(seed))
        raw_metrics[str(seed)] = {}
        for feature, matrix in matrices.items():
            artifact_dir = result_dir / "torch_probe" / f"seed{seed}" / feature
            print(f"[train] seed={seed} feature={feature} dims={matrix.shape[1]}", flush=True)
            metrics = train_and_evaluate_probe(
                X_train=matrix[train_mask],
                y_train=labels[train_mask],
                X_val=np.empty((0, matrix.shape[1]), dtype=np.float32),
                y_val=np.empty((0,), dtype=np.int32),
                X_test=matrix[test_mask],
                y_test=labels[test_mask],
                config=cfg,
                device=device,
                output_dir=str(artifact_dir),
            )
            metrics["num_features"] = int(matrix.shape[1])
            metrics["best_params"] = asdict(cfg)
            raw_metrics[str(seed)][feature] = _json_ready(metrics)
            for report in ("train_f1", "fixed_0.5"):
                records.append({
                    "seed": int(seed),
                    "feature": feature,
                    "num_features": int(matrix.shape[1]),
                    "threshold_report": report,
                    **extract_report(metrics, report),
                    "best_epoch": int(metrics["best_epoch"]),
                    "epochs_ran": int(metrics["epochs_ran"]),
                })
            (result_dir / "raw_metrics.json").write_text(
                json.dumps(raw_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[train] done seed={seed} feature={feature} "
                f"AUC={metrics['auc']:.4f} realF1={metrics['real_positive']['f1']:.4f} "
                f"hallF1={metrics['hallucination_positive']['f1']:.4f}",
                flush=True,
            )
    audit = {
        "train_images": len(train_ids),
        "test_images": len(test_ids),
        "train_token_rows": int(train_mask.sum()),
        "test_token_rows": int(test_mask.sum()),
        "probe_config": asdict(base_config),
    }
    return records, audit


def aggregate(records: list[dict]) -> list[dict]:
    result = []
    metric_names = (
        "auc", "real_aupr", "hall_aupr", "real_f1", "hall_f1",
        "accuracy", "threshold", "best_epoch", "epochs_ran",
    )
    for feature in sorted({x["feature"] for x in records}):
        for report in ("train_f1", "fixed_0.5"):
            selected = [x for x in records if x["feature"] == feature and x["threshold_report"] == report]
            row = {
                "feature": feature,
                "num_features": int(selected[0]["num_features"]),
                "threshold_report": report,
                "seeds": [int(x["seed"]) for x in selected],
            }
            for metric in metric_names:
                values = np.asarray([x[metric] for x in selected], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result.append(row)
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    normalized = []
    for source in rows:
        row = dict(source)
        if isinstance(row.get("seeds"), list):
            row["seeds"] = " ".join(str(x) for x in row["seeds"])
        normalized.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)


def fmt(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}"


def write_markdown(path: Path, summary: list[dict], feature_audit: dict, train_audit: dict) -> None:
    primary = [x for x in summary if x["threshold_report"] == "train_f1"]
    fixed = [x for x in summary if x["threshold_report"] == "fixed_0.5"]
    risk_start, risk_end = feature_audit["risk_layers_inclusive"]
    selected_risk_dim = int(feature_audit["selected_risk_dimension"])
    ev_dim = int(feature_audit["ev_curve_dimensions"][0])
    method_label = FEATURE_METHODS[feature_audit["feature_method"]]["label"]
    scope_label = " / ".join(scope.upper() for scope in feature_audit["scopes"])
    lines = [
        f"# Source tau × transport Top-K sweep ({scope_label})",
        "",
        "## Protocol",
        "",
        f"- Feature: {method_label} sqrt-matched-state risk "
        f"layers {risk_start}-{risk_end} + corresponding full-layer EV.",
        f"- Dimension: {selected_risk_dim} risk layers + {ev_dim} EV layers "
        f"= {selected_risk_dim + ev_dim}.",
        "- Grid: YAML `source_tau_values` x `transport_top_k_values`.",
        f"- Risk modes: {', '.join(feature_audit['risk_modes'])}. Fixed-TopK "
        "uses the Cartesian grid; capped-topmass replaces fixed Top-K and is "
        "trained once per source tau.",
        f"- Scopes {scope_label} are trained independently with identical image "
        "splits, seeds, and MLP settings.",
        f"- Feature audit: `{json.dumps(feature_audit, ensure_ascii=False)}`",
        f"- Training audit: `{json.dumps(train_audit, ensure_ascii=False)}`",
        "",
        "## Train-F1 threshold selected on training rows only",
        "",
        "| Feature | Dim | AUROC | Real AUPR | Hall AUPR | Real F1 | Hall F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda x: x["auc_mean"], reverse=True):
        lines.append(
            f"| {row['feature']} | {row['num_features']} | {fmt(row, 'auc')} | "
            f"{fmt(row, 'real_aupr')} | {fmt(row, 'hall_aupr')} | "
            f"{fmt(row, 'real_f1')} | {fmt(row, 'hall_f1')} | {fmt(row, 'accuracy')} |"
        )
    lines += [
        "", "## Fixed threshold 0.5", "",
        "| Feature | Dim | AUROC | Real F1 | Hall F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(fixed, key=lambda x: x["auc_mean"], reverse=True):
        lines.append(
            f"| {row['feature']} | {row['num_features']} | {fmt(row, 'auc')} | "
            f"{fmt(row, 'real_f1')} | {fmt(row, 'hall_f1')} | {fmt(row, 'accuracy')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.layer_start < 0:
        raise ValueError("Expected layer_start >= 0")
    if args.layer_end is not None and args.layer_end <= args.layer_start:
        raise ValueError("Expected layer_end > layer_start")
    output_dir = Path(args.output_dir).resolve()
    config_root = load_config(args.config)
    if args.if_enabled:
        methods = configured_training_methods(config_root)
        if not methods:
            print(
                "[sweep] skipped: "
                "training.source_tau_transport_topk_sweep.enabled=false",
                flush=True,
            )
            return
        if args.feature_method is not None:
            raise ValueError(
                "--feature-method cannot be combined with --if-enabled; "
                "configure feature_methods in YAML"
            )
        if args.risk_modes is not None:
            raise ValueError(
                "--risk-modes cannot be combined with --if-enabled; configure "
                "risk_modes in YAML"
            )
        if args.run_name is not None:
            raise ValueError(
                "--run-name cannot be combined with --if-enabled because each "
                "configured method uses its own result directory"
            )
        scopes = configured_training_scopes(config_root)
        risk_modes = configured_training_risk_modes(config_root)
    else:
        methods = [args.feature_method or "hpre_raw_logit_gauss"]
        scopes = ("vv", "vpend")
        risk_modes = tuple(args.risk_modes or ["fixed_topk"])

    result_dirs = {
        method: output_dir / "results" / (
            args.run_name or DEFAULT_RUN_NAMES[method]
        )
        for method in methods
    }
    for result_dir in result_dirs.values():
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty {result_dir}")

    torch_cfg = ((config_root.get("training") or {}).get("torch_probe") or {})
    dgst_cfg = ((config_root.get("feature_extraction") or {}).get("dgst_t") or {})
    capped_topmass_alphas = configured_capped_topmass_alphas(config_root)
    tau_values = [float(x) for x in (dgst_cfg.get("source_tau_values") or [])]
    top_k_values = [
        int(x) for x in (dgst_cfg.get("transport_top_k_values") or [])
    ]
    if not tau_values or not top_k_values:
        raise ValueError(
            "YAML must define feature_extraction.dgst_t.source_tau_values "
            "and transport_top_k_values"
        )

    def tau_slug(value: float) -> str:
        return format(value, ".12g").replace("-", "m").replace(".", "p")
    expected_variants = {
        f"tau{tau_slug(tau)}_topk{top_k}": (tau, top_k)
        for tau in tau_values
        for top_k in top_k_values
    }
    seeds = args.seeds or [int(x) for x in torch_cfg.get("seeds", [43, 44, 45])]
    splits = load_json(str(output_dir / "image_splits.json"))
    validate_strict_82_split(splits)
    print(f"[load] {output_dir / 'features.pkl'}", flush=True)
    rows = load_pkl(str(output_dir / "features.pkl"))
    for method in methods:
        result_dir = result_dirs[method]
        result_dir.mkdir(parents=True, exist_ok=True)
        matrices, labels, image_ids, feature_audit = build_matrices(
            rows,
            expected_variants,
            args.layer_start,
            args.layer_end,
            method,
            scopes,
            risk_modes,
            capped_topmass_alphas=capped_topmass_alphas,
        )
        records, train_audit = train_all(
            matrices, labels, image_ids, splits, seeds, args.config,
            args.device, result_dir,
        )
        summary = aggregate(records)
        write_csv(records, result_dir / "per_seed.csv")
        write_csv(summary, result_dir / "summary.csv")
        (result_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_markdown(
            result_dir / "summary.md", summary, feature_audit, train_audit
        )
        print(f"[done] {result_dir / 'summary.md'}", flush=True)
        del matrices


if __name__ == "__main__":
    main()
