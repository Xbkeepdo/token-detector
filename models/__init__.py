"""Model wrapper factory."""

_REGISTRY = {
    "llava_1_5_7b": ("models.llava_wrapper", "LLaVAWrapper"),
    "llava_onevision_1_5_8b": (
        "models.llava_onevision_wrapper",
        "LLaVAOneVisionWrapper",
    ),
    "llava_onevision_1_5_8b_instruct": (
        "models.llava_onevision_wrapper",
        "LLaVAOneVisionWrapper",
    ),
    "internvl_2_5_8b": ("models.internvl_wrapper", "InternVLWrapper"),
    "qwen2_5_vl_7b": ("models.qwen_wrapper", "QwenVLWrapper"),
}


def build_model(model_key: str, cfg: dict, device: str = "cuda"):
    if model_key not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Valid: {list(_REGISTRY.keys())}")
    module_name, class_name = _REGISTRY[model_key]
    module = __import__(module_name, fromlist=[class_name])
    wrapper_cls = getattr(module, class_name)
    return wrapper_cls(cfg, device=device)
