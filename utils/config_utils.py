"""YAML config loading and unified pipeline helpers."""
from __future__ import annotations

import copy
import shlex
from typing import Any, Mapping, Optional

import yaml


VALID_EXTRACTION_MODES = {
    "all",
    "method_only",
    "ads_cgc_only",
    "baseline_only",
}
VALID_DGST_BRANCHES = (
    "hpre_raw_logit_gauss",
    "hpre_raw_logit_relative_vll",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
    "hpre_softmax_prob_direct",
    "raw_attention",
)
VALID_DGST_TARGET_MODES = (
    "relative_vll",
    "raw_logit_gauss",
)
_DGST_TARGET_METHODS = {
    "relative_vll": "hpre_raw_logit_relative_vll",
    "raw_logit_gauss": "hpre_raw_logit_gauss",
}
_DGST_TARGET_MODE_ALIASES = {
    "relative": "relative_vll",
    "relative_vll": "relative_vll",
    "hpre_raw_logit_relative_vll": "relative_vll",
    "gauss": "raw_logit_gauss",
    "gaussian": "raw_logit_gauss",
    "raw_logit_gauss": "raw_logit_gauss",
    "hpre_raw_logit_gauss": "raw_logit_gauss",
}
VALID_LABEL_SAMPLE_UNITS = {"first_canonical_mention", "all_mentions"}
VALID_LABEL_LOCATORS = {"exact_response_offsets"}
VALID_ALIGNMENT_FAILURE_POLICIES = {"error", "skip"}
PIPELINE_STAGES = (
    "generation",
    "labeling",
    "feature_extraction",
    "training",
    "plotting",
)


def resolve_dgst_four_gate_methods(dgst: Mapping[str, Any]) -> list[str]:
    """Resolve branch booleans and the independent hpre target-mode switch.

    ``target_modes`` is authoritative for the two hpre/raw-logit target
    constructors.  Other historical Gaussian/direct branches continue to use
    ``four_gate_methods`` plus ``branches`` exactly as before.
    """

    configured = dgst.get("four_gate_methods")
    branches = dgst.get("branches") or {}
    if isinstance(configured, str):
        methods = [configured]
    elif isinstance(configured, (list, tuple)):
        methods = [str(method) for method in configured]
    elif isinstance(branches, Mapping) and branches:
        methods = [str(method) for method in branches]
    else:
        # Preserve the historical default: the four Gaussian constructions,
        # not the opt-in direct controls or the new Relative-VLL comparison.
        methods = [
            "hpre_raw_logit_gauss",
            "hpre_softmax_prob_gauss",
            "hmid_raw_logit_gauss",
            "hmid_softmax_prob_gauss",
        ]

    if isinstance(branches, Mapping):
        methods = [
            method for method in methods if bool(branches.get(method, True))
        ]

    raw_target_modes = dgst.get("target_modes")
    if raw_target_modes is not None:
        requested = (
            [raw_target_modes]
            if isinstance(raw_target_modes, str)
            else list(raw_target_modes)
        )
        if not requested:
            raise ValueError("dgst_t.target_modes cannot be empty")
        target_modes: list[str] = []
        unknown: list[str] = []
        for raw_mode in requested:
            alias = str(raw_mode).strip().lower().replace("-", "_")
            mode = _DGST_TARGET_MODE_ALIASES.get(alias)
            if mode is None:
                unknown.append(alias)
            elif mode not in target_modes:
                target_modes.append(mode)
        if unknown:
            raise ValueError(
                f"Unknown DGST target modes {unknown}; expected a subset of "
                f"{list(VALID_DGST_TARGET_MODES)}"
            )
        selectable_methods = set(_DGST_TARGET_METHODS.values())
        methods = [method for method in methods if method not in selectable_methods]
        methods.extend(_DGST_TARGET_METHODS[mode] for mode in target_modes)

    methods = list(dict.fromkeys(methods))
    unknown_methods = sorted(set(methods) - set(VALID_DGST_BRANCHES))
    if unknown_methods:
        raise ValueError(
            f"Unknown DGST branches {unknown_methods}; expected a subset of "
            f"{list(VALID_DGST_BRANCHES)}"
        )
    if not methods:
        raise ValueError("All DGST branch/target switches are disabled.")
    return methods

_LABELING_DEFAULTS = {
    "schema_version": 2,
    "sample_unit": "first_canonical_mention",
    "primary_locator": "exact_response_offsets",
    "save_all_mentions": True,
    "save_svar_official_samples": True,
    "alignment_failure_policy": "error",
}

_RUN_DEFAULTS = {
    "prompt": "Describe this image.",
    "resume": True,
    "adopt_legacy_artifacts": False,
    "validate_manifests": True,
    "extraction_mode": "all",
    "stages": {
        "generation": True,
        "labeling": True,
        "feature_extraction": True,
        "training": True,
        "plotting": False,
    },
    "devices": {
        "primary": "cuda:0",
        "generation": ["cuda:0", "cuda:1"],
        "feature_extraction": ["cuda:0", "cuda:1"],
        "training": "cuda:0",
    },
    "trainer": "torch_mlp",
    "positive_class": "real",
    "torch_probe_args": "",
    "dgst_branches": None,
    "max_pixels": None,
}


def load_config(path: str = "configs/model_configs.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping, got {type(config).__name__}")
    return config


def resolve_run_config(
    config: dict,
    environ: Optional[Mapping[str, str]] = None,
) -> dict:
    """Merge built-in/YAML defaults with explicit shell environment overrides."""

    environ = environ or {}
    raw_run = config.get("run") or {}
    if not isinstance(raw_run, dict):
        raise ValueError("run must be a YAML mapping")
    run = _deep_merge(_RUN_DEFAULTS, raw_run)

    scalar_overrides = {
        "MODEL": "model",
        "OUTPUT": "output_dir",
        "PROMPT": "prompt",
        "EXTRACTION_MODE": "extraction_mode",
        "TRAINER": "trainer",
        "TORCH_PROBE_DEVICE": "torch_probe_device",
        "TORCH_PROBE_ARGS": "torch_probe_args",
        "POSITIVE_CLASS": "positive_class",
        "CHAIR_CACHE": "chair_cache",
    }
    for env_name, key in scalar_overrides.items():
        if environ.get(env_name) not in (None, ""):
            run[key] = str(environ[env_name])

    if environ.get("RESUME") not in (None, ""):
        run["resume"] = _parse_bool(environ["RESUME"], name="RESUME")
    if environ.get("ADOPT_LEGACY_ARTIFACTS") not in (None, ""):
        run["adopt_legacy_artifacts"] = _parse_bool(
            environ["ADOPT_LEGACY_ARTIFACTS"],
            name="ADOPT_LEGACY_ARTIFACTS",
        )
    if environ.get("DGST_BRANCHES") not in (None, ""):
        run["dgst_branches"] = shlex.split(
            environ["DGST_BRANCHES"].replace(",", " ")
        )
    if environ.get("MAX_PIXELS") not in (None, ""):
        try:
            run["max_pixels"] = int(environ["MAX_PIXELS"])
        except ValueError as exc:
            raise ValueError("MAX_PIXELS must be a positive integer") from exc

    devices = dict(run.get("devices") or {})
    if environ.get("DEVICE"):
        devices["primary"] = str(environ["DEVICE"])
    if environ.get("GENERATION_DEVICES"):
        devices["generation"] = shlex.split(
            environ["GENERATION_DEVICES"].replace(",", " ")
        )
    if environ.get("FEATURE_DEVICES"):
        devices["feature_extraction"] = shlex.split(
            environ["FEATURE_DEVICES"].replace(",", " ")
        )
    if environ.get("TORCH_PROBE_DEVICE"):
        devices["training"] = str(environ["TORCH_PROBE_DEVICE"])
    run["devices"] = devices

    stages = dict(run.get("stages") or {})
    if environ.get("STAGES"):
        selected = {
            item.strip().lower()
            for item in environ["STAGES"].replace(",", " ").split()
            if item.strip()
        }
        unknown = sorted(selected - set(PIPELINE_STAGES))
        if unknown:
            raise ValueError(f"STAGES contains unknown stages: {unknown}")
        stages = {name: name in selected for name in PIPELINE_STAGES}
    if environ.get("RUN_GENERATION_LABELING") not in (None, ""):
        enabled = _parse_bool(
            environ["RUN_GENERATION_LABELING"],
            name="RUN_GENERATION_LABELING",
        )
        stages["generation"] = enabled
        stages["labeling"] = enabled
    if environ.get("RUN_TRAIN_EVAL") not in (None, ""):
        stages["training"] = _parse_bool(
            environ["RUN_TRAIN_EVAL"],
            name="RUN_TRAIN_EVAL",
        )
    for name in PIPELINE_STAGES:
        env_name = f"RUN_{name.upper()}"
        if environ.get(env_name) not in (None, ""):
            stages[name] = _parse_bool(environ[env_name], name=env_name)
    run["stages"] = stages

    if environ.get("FEATURE_SETS"):
        run["feature_sets"] = shlex.split(environ["FEATURE_SETS"])

    _validate_run_config(run, config)
    return run


def manifest_validation_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether artifact provenance manifests should block reuse.

    When disabled, pipeline stages neither read nor write provenance manifest
    sidecars. Structural artifact and split validation remains active.
    """

    raw_run = config.get("run") or {}
    if not isinstance(raw_run, Mapping):
        raise ValueError("run must be a YAML mapping")
    return _parse_bool(
        raw_run.get("validate_manifests", True),
        name="run.validate_manifests",
    )


def extraction_mode_flags(mode: str) -> dict[str, bool]:
    """Return the feature families enabled by an extraction mode."""

    value = str(mode).strip().lower()
    if value not in VALID_EXTRACTION_MODES:
        raise ValueError(
            f"Unknown extraction_mode {mode!r}; choose one of "
            f"{sorted(VALID_EXTRACTION_MODES)}"
        )
    return {
        "method": value in {"all", "method_only"},
        "ads_cgc": value in {"all", "ads_cgc_only"},
        "baseline": value in {"all", "baseline_only"},
    }


def qa_extraction_family_flags(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the QA extraction mode against the shared family switches."""

    qa_cfg = config.get("qa_benchmarks") or {}
    run_cfg = config.get("run") or {}
    mode = str(
        qa_cfg.get("extraction_mode", run_cfg.get("extraction_mode", "all"))
    ).strip().lower()
    requested = extraction_mode_flags(mode)
    feature_cfg = config.get("feature_extraction") or {}
    return {
        "mode": mode,
        "method": bool(
            requested["method"]
            and (feature_cfg.get("method") or {}).get("enabled", True)
            and (feature_cfg.get("dgst_t") or {}).get("enabled", True)
        ),
        "ads_cgc": bool(
            requested["ads_cgc"]
            and (feature_cfg.get("ads_cgc") or {}).get("enabled", True)
        ),
        "baseline": bool(
            requested["baseline"]
            and (feature_cfg.get("baseline") or {}).get("enabled", True)
        ),
    }


def get_model_cfg(config: dict, model_key: str) -> dict:
    cfg = config["models"].get(model_key)
    if cfg is None:
        raise ValueError(
            f"Model '{model_key}' not found in config. "
            f"Available: {list(config['models'].keys())}"
        )
    cfg = dict(cfg)
    experiment = config.get("experiment") or {}
    scope = experiment.get("dgst_t_support_scope")
    mode = str(experiment.get("mode", "")).strip().lower()
    if scope is None:
        if mode in ("vv", "v", "visual", "visual_only"):
            scope = "visual"
        elif mode in ("vp", "visual_prompt", "visual+prompt", "visual_prompt_only"):
            scope = "visual_prompt"
        elif mode in (
            "vpend",
            "vp_end",
            "visual_prompt_end",
            "post_visual_prompt",
        ):
            scope = "visual_prompt_end"
    if scope is not None:
        cfg["dgst_t_support_scope"] = str(scope)
    run = config.get("run") or {}
    prompt = run.get("prompt")
    if prompt is not None:
        # Wrappers use this as their default when no explicit prompt argument
        # is supplied.  Architecture-specific prompt templates remain intact.
        cfg["prompt"] = str(prompt)
        cfg["generation_prompt"] = str(prompt)
    if run.get("extraction_mode") is not None:
        cfg["extraction_mode"] = str(run["extraction_mode"])
    return cfg


def get_extraction_model_cfg(config: dict, model_key: str) -> dict:
    """Return model config with feature-extraction-only overrides applied.

    Keeping these overrides outside ``models`` lets an existing generation
    manifest remain bound to the exact settings that produced its captions,
    while feature extraction can deliberately use a cheaper visual layout.
    The effective model config is still included in feature provenance.
    """
    cfg = get_model_cfg(config, model_key)
    feature_cfg = config.get("feature_extraction") or {}
    if not isinstance(feature_cfg, Mapping):
        raise ValueError("feature_extraction must be a mapping")
    overrides_by_model = feature_cfg.get("model_overrides") or {}
    if not isinstance(overrides_by_model, Mapping):
        raise ValueError("feature_extraction.model_overrides must be a mapping")
    overrides = overrides_by_model.get(model_key) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError(
            f"feature_extraction.model_overrides.{model_key} must be a mapping"
        )
    return _deep_merge(cfg, dict(overrides))


def get_ads_cfg(config: dict) -> dict:
    return config["feature_extraction"]["ads"]


def get_cgc_cfg(config: dict) -> dict:
    return config["feature_extraction"]["cgc"]


def get_dgst_t_cfg(config: dict) -> dict:
    return config["feature_extraction"].get("dgst_t", {})


def get_dataset_cfg(config: dict) -> dict:
    return config["dataset"]


def get_labeling_cfg(config: Mapping[str, object]) -> dict:
    """Return and validate the schema-v2 COCO labeling protocol."""

    raw = config.get("labeling") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("labeling must be a YAML mapping")
    unknown = sorted(set(raw) - set(_LABELING_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown labeling options: {unknown}")
    result = _deep_merge(_LABELING_DEFAULTS, dict(raw))
    try:
        result["schema_version"] = int(result["schema_version"])
    except (TypeError, ValueError) as exc:
        raise ValueError("labeling.schema_version must be 2") from exc
    if result["schema_version"] != 2:
        raise ValueError("Only labeling.schema_version=2 is supported")

    sample_unit = str(result["sample_unit"]).strip().lower()
    if sample_unit not in VALID_LABEL_SAMPLE_UNITS:
        raise ValueError(
            "labeling.sample_unit must be one of "
            f"{sorted(VALID_LABEL_SAMPLE_UNITS)}"
        )
    result["sample_unit"] = sample_unit

    locator = str(result["primary_locator"]).strip().lower()
    if locator not in VALID_LABEL_LOCATORS:
        raise ValueError(
            "labeling.primary_locator must be one of "
            f"{sorted(VALID_LABEL_LOCATORS)}"
        )
    result["primary_locator"] = locator

    for key in ("save_all_mentions", "save_svar_official_samples"):
        result[key] = _parse_bool(result[key], name=f"labeling.{key}")
    if not result["save_all_mentions"]:
        raise ValueError(
            "labeling.save_all_mentions must remain true in schema v2 so "
            "standard occurrence-level CHAIRs/CHAIRi can be reproduced"
        )
    failure_policy = str(result["alignment_failure_policy"]).strip().lower()
    if failure_policy not in VALID_ALIGNMENT_FAILURE_POLICIES:
        raise ValueError(
            "labeling.alignment_failure_policy must be one of "
            f"{sorted(VALID_ALIGNMENT_FAILURE_POLICIES)}"
        )
    result["alignment_failure_policy"] = failure_policy
    return result


def get_classifier_cfgs(config: dict) -> dict:
    return config["classifiers"]


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _normalize_device_list(value: object, *, name: str) -> list[str]:
    if isinstance(value, str):
        devices = shlex.split(value.replace(",", " "))
    elif isinstance(value, (list, tuple)):
        devices = [str(item) for item in value]
    else:
        raise ValueError(f"run.devices.{name} must be a string or list")
    devices = [item.strip() for item in devices if item.strip()]
    if not devices:
        raise ValueError(f"run.devices.{name} cannot be empty")
    return devices


def _validate_run_config(run: dict, config: dict) -> None:
    model = str(run.get("model", "")).strip()
    if not model:
        raise ValueError("run.model is required")
    models = config.get("models") or {}
    if model not in models:
        raise ValueError(
            f"run.model {model!r} is not defined in models; available={list(models)}"
        )
    run["model"] = model

    output_dir = str(run.get("output_dir", "")).strip()
    if not output_dir:
        raise ValueError("run.output_dir is required")
    run["output_dir"] = output_dir

    prompt = str(run.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("run.prompt cannot be empty")
    run["prompt"] = prompt
    run["resume"] = _parse_bool(run.get("resume", True), name="run.resume")
    run["adopt_legacy_artifacts"] = _parse_bool(
        run.get("adopt_legacy_artifacts", False),
        name="run.adopt_legacy_artifacts",
    )
    run["validate_manifests"] = _parse_bool(
        run.get("validate_manifests", True),
        name="run.validate_manifests",
    )

    positive_class = str(run.get("positive_class", "real")).strip().lower()
    if positive_class not in {"real", "non_hallucination", "hallucination"}:
        raise ValueError(
            "run.positive_class must be real, non_hallucination, or hallucination"
        )
    run["positive_class"] = positive_class

    raw_branches = run.get("dgst_branches")
    if raw_branches:
        if isinstance(raw_branches, str):
            raw_branches = shlex.split(raw_branches.replace(",", " "))
        elif isinstance(raw_branches, (list, tuple)):
            raw_branches = [str(value) for value in raw_branches]
        else:
            raise ValueError("run.dgst_branches must be a string or list")
        branches = list(
            dict.fromkeys(
                value.strip() for value in raw_branches if value.strip()
            )
        )
        unknown_branches = sorted(set(branches) - set(VALID_DGST_BRANCHES))
        if unknown_branches:
            raise ValueError(
                f"Unknown DGST branches {unknown_branches}; expected a subset of "
                f"{list(VALID_DGST_BRANCHES)}"
            )
        if not branches:
            raise ValueError("run.dgst_branches cannot be empty")
        run["dgst_branches"] = branches
    else:
        run["dgst_branches"] = None

    max_pixels = run.get("max_pixels")
    if max_pixels not in (None, ""):
        try:
            max_pixels = int(max_pixels)
        except (TypeError, ValueError) as exc:
            raise ValueError("run.max_pixels must be a positive integer") from exc
        if max_pixels <= 0:
            raise ValueError("run.max_pixels must be a positive integer")
        run["max_pixels"] = max_pixels
    else:
        run["max_pixels"] = None

    mode = str(run.get("extraction_mode", "all")).strip().lower()
    extraction_mode_flags(mode)
    run["extraction_mode"] = mode

    raw_stages = run.get("stages")
    if not isinstance(raw_stages, dict):
        raise ValueError("run.stages must be a mapping")
    unknown_stages = sorted(set(raw_stages) - set(PIPELINE_STAGES))
    if unknown_stages:
        raise ValueError(f"run.stages contains unknown keys: {unknown_stages}")
    run["stages"] = {
        name: _parse_bool(raw_stages.get(name, False), name=f"run.stages.{name}")
        for name in PIPELINE_STAGES
    }

    raw_devices = run.get("devices")
    if not isinstance(raw_devices, dict):
        raise ValueError("run.devices must be a mapping")
    primary = str(raw_devices.get("primary", "")).strip()
    training = str(
        raw_devices.get("training", run.get("torch_probe_device", primary))
    ).strip()
    if not primary or not training:
        raise ValueError("run.devices.primary and run.devices.training are required")
    run["devices"] = {
        "primary": primary,
        "generation": _normalize_device_list(
            raw_devices.get("generation", [primary]), name="generation"
        ),
        "feature_extraction": _normalize_device_list(
            raw_devices.get("feature_extraction", [primary]),
            name="feature_extraction",
        ),
        "training": training,
    }

    trainer = str(run.get("trainer", "torch_mlp")).strip().lower()
    if trainer not in {"torch_mlp", "torch_probe", "sklearn", "xgb_rf"}:
        raise ValueError(
            "run.trainer must be torch_mlp, torch_probe, sklearn, or xgb_rf"
        )
    run["trainer"] = trainer
    if "feature_sets" in run:
        if not isinstance(run["feature_sets"], list) or not run["feature_sets"]:
            raise ValueError("run.feature_sets must be a non-empty list")
        run["feature_sets"] = [str(item) for item in run["feature_sets"]]
