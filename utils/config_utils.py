"""YAML config loading helpers."""
from __future__ import annotations
import yaml


def load_config(path: str = "configs/model_configs.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
    if scope is not None:
        cfg["dgst_t_support_scope"] = str(scope)
    return cfg


def get_ads_cfg(config: dict) -> dict:
    return config["feature_extraction"]["ads"]


def get_cgc_cfg(config: dict) -> dict:
    return config["feature_extraction"]["cgc"]


def get_dgst_t_cfg(config: dict) -> dict:
    return config["feature_extraction"].get("dgst_t", {})


def get_dataset_cfg(config: dict) -> dict:
    return config["dataset"]


def get_classifier_cfgs(config: dict) -> dict:
    return config["classifiers"]
