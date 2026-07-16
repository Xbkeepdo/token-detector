"""Paper baselines for token-level LVLM hallucination detection."""

from .dhcp import (
    DHCPShardReader,
    DHCPShardReference,
    DHCPShardWriter,
    infer_patch_grid,
    resize_attention_preserve_mass,
)
from .extractor import (
    BaselineExtractionContext,
    baseline_extraction_requirements,
    compute_baseline_record,
)
from .halloc import (
    DEFAULT_CLIP_MODEL,
    DEFAULT_VISUALBERT_MODEL,
    HalLocCLIPFeatureExtractor,
    HalLocObjectDetector,
    halloc_optimizer_config,
)
from .metatoken import (
    MetaTokenFeatures,
    compute_metatoken_features,
    compute_metatoken_features_from_stats,
)
from .projectaway import (
    ProjectAwayConfidence,
    ProjectAwayProbabilityCache,
    compute_projectaway_internal_confidence,
    compute_projectaway_probability_cache,
)
from .runtime import (
    BaselineRuntime,
    baseline_config,
    normalize_baseline_methods,
)
from .schema import (
    BASELINE_SCHEMA_VERSION,
    SUPPORTED_BASELINES,
    attach_baseline,
    baseline_vector,
    get_baseline_payload,
    make_baseline_record,
    validate_baseline_record,
)
from .svar import (
    SUPPORTED_SVAR_PROTOCOLS,
    SVARFeatures,
    compute_svar_features,
    normalize_svar_protocols,
    prepare_official_svar_spans,
)

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "BaselineExtractionContext",
    "BaselineRuntime",
    "SUPPORTED_BASELINES",
    "SUPPORTED_SVAR_PROTOCOLS",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_VISUALBERT_MODEL",
    "DHCPShardReader",
    "DHCPShardReference",
    "DHCPShardWriter",
    "HalLocObjectDetector",
    "HalLocCLIPFeatureExtractor",
    "MetaTokenFeatures",
    "ProjectAwayConfidence",
    "ProjectAwayProbabilityCache",
    "SVARFeatures",
    "attach_baseline",
    "baseline_extraction_requirements",
    "baseline_config",
    "baseline_vector",
    "compute_metatoken_features",
    "compute_metatoken_features_from_stats",
    "compute_baseline_record",
    "compute_projectaway_internal_confidence",
    "compute_projectaway_probability_cache",
    "compute_svar_features",
    "get_baseline_payload",
    "halloc_optimizer_config",
    "infer_patch_grid",
    "make_baseline_record",
    "normalize_baseline_methods",
    "normalize_svar_protocols",
    "prepare_official_svar_spans",
    "resize_attention_preserve_mass",
    "validate_baseline_record",
]
