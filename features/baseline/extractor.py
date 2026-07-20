"""Pure integration facade from one wrapper output to a baseline record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from models.base_wrapper import (
    AttentionRequirement,
    ExtractionRequirements,
    ModelOutput,
)

from .dhcp import DHCPShardWriter, resize_attention_preserve_mass
from .metatoken import (
    compute_metatoken_features,
    compute_metatoken_features_from_stats,
)
from .projectaway import compute_projectaway_internal_confidence
from .projectaway import ProjectAwayProbabilityCache
from .schema import attach_baseline, make_baseline_record
from .svar import compute_svar_features


@dataclass
class BaselineExtractionContext:
    """Image/caption-level values shared by all object-token records."""

    response_token_ids: Sequence[int]
    image_id: int = -1
    response_token_logits: Optional[torch.Tensor] = None
    metatoken_step_stats: Optional[Mapping[str, Any]] = None
    occurrence_count: int = 1
    target_token_ids: Optional[Sequence[int]] = None
    unembedding_weight: Optional[torch.Tensor] = None
    unembedding_bias: Optional[torch.Tensor] = None
    projectaway_normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    projectaway_probability_cache: Optional[ProjectAwayProbabilityCache] = None
    clip_visual_features: Optional[torch.Tensor] = None
    halloc_cache_file: Optional[str] = None
    halloc_object_index: Optional[int] = None


def baseline_extraction_requirements(
    methods: Sequence[str],
) -> ExtractionRequirements:
    """Return the least wrapper capture needed for the selected baselines."""

    selected = {str(method).strip().lower() for method in methods}
    unknown = selected - {"metatoken", "svar", "dhcp", "projectaway", "halloc"}
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    needs_attention = bool(selected & {"metatoken", "svar", "dhcp"})
    return ExtractionRequirements(
        attention=(
            AttentionRequirement.PER_HEAD
            if needs_attention
            else AttentionRequirement.NONE
        ),
        # MetaToken's six compact statistics are produced and compressed in
        # the one full-caption teacher-forced pass requested below.  No
        # baseline consumes per-object full-vocabulary logits.
        logits=False,
        token_hidden_states=False,
        patch_hidden_states="projectaway" in selected,
        # Current wrappers compute compact MetaToken statistics together with
        # the lightweight full-caption hidden pass.  This remains one extra
        # no-attention pass, never one pass per object token.
        response_hidden_states=bool(selected & {"metatoken", "halloc"}),
        visual_layout="dhcp" in selected,
        dgst_capture=False,
    )


def compute_baseline_record(
    *,
    model_out: ModelOutput,
    span: Mapping[str, Any],
    context: BaselineExtractionContext,
    methods: Sequence[str],
    dhcp_writer: Optional[DHCPShardWriter] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Derive all requested baseline payloads without another LVLM forward.

    The caller can invoke this immediately after DGST/ADS/CGC computation, so
    ``all`` mode shares the same wrapper output.  MetaToken's caption-level
    logits and HalLoc's CLIP cache are image-level values supplied via
    ``context`` rather than duplicated in every ``ModelOutput``.
    """

    cfg = dict(config or {})
    selected = [str(method).strip().lower() for method in methods]
    token_indices = [int(value) for value in span.get("token_indices", [])]
    if not token_indices:
        token_indices = [int(model_out.response_token_idx)]
    start, end = min(token_indices), max(token_indices)
    response_ids = [int(value) for value in context.response_token_ids]
    if start < 0 or end >= len(response_ids):
        raise ValueError(
            f"Object span [{start},{end}] is outside response length {len(response_ids)}"
        )
    label = int(span["label"])
    target_ids = (
        [int(value) for value in context.target_token_ids]
        if context.target_token_ids is not None
        else response_ids[start : end + 1]
    )
    record = make_baseline_record(
        image_id=int(context.image_id),
        token_str=str(span.get("word", model_out.token_str)),
        response_token_idx=start,
        target_token_id=target_ids[0],
        span_start=start,
        span_end=end,
        label=label,
        metadata={
            "visual_grid": list(model_out.visual_grid)
            if model_out.visual_grid is not None
            else None,
        },
    )

    if "metatoken" in selected:
        logits = context.response_token_logits
        compact_stats = context.metatoken_step_stats
        if compact_stats is None and model_out.baseline_capture is not None:
            required_stat_names = {
                "response_target_logprobs",
                "response_target_probs",
                "response_logprob_variances",
                "response_normalized_entropies",
                "response_top1_probs",
                "response_top2_probs",
            }
            if required_stat_names <= set(model_out.baseline_capture):
                compact_stats = model_out.baseline_capture
        if logits is None and compact_stats is None:
            raise ValueError(
                "MetaToken requires compact per-step statistics or "
                "context.response_token_logits [T,V]"
            )
        metatoken_cfg = dict(cfg.get("metatoken") or {})
        common = {
            "response_token_ids": response_ids,
            "visual_attention": model_out.text_to_patch_attn,
            "span_start": start,
            "span_end": end,
            "occurrence_count": int(context.occurrence_count),
            "length_penalty": float(metatoken_cfg.get("length_penalty", 1.0)),
            "attention_layer": metatoken_cfg.get("attention_layer", -1),
        }
        if compact_stats is not None:
            features = compute_metatoken_features_from_stats(
                **common,
                **{
                    name: compact_stats[name]
                    for name in (
                        "response_target_logprobs",
                        "response_target_probs",
                        "response_logprob_variances",
                        "response_normalized_entropies",
                        "response_top1_probs",
                        "response_top2_probs",
                    )
                },
            )
        else:
            features = compute_metatoken_features(token_logits=logits, **common)
        attach_baseline(record, "metatoken", features.as_payload())

    if "svar" in selected:
        features = compute_svar_features(
            model_out.text_to_patch_attn,
            all_layers=True,
        )
        attach_baseline(record, "svar", features.as_payload())

    if "dhcp" in selected:
        grid = model_out.visual_grid
        resized = resize_attention_preserve_mass(
            model_out.text_to_patch_attn,
            source_grid=grid,
            target_grid=tuple(
                cfg.get("dhcp", {}).get(
                    "target_grid",
                    cfg.get("dhcp", {}).get("spatial_size", (12, 12)),
                )
            ),
        )
        payload: dict[str, Any] = {
            "protocol": "object_prediction_fixed_grid_mass_preserving",
            "target_grid": list(resized.shape[-2:]),
            "source_grid": list(grid) if grid is not None else None,
        }
        if dhcp_writer is None:
            # Useful for unit/smoke tests; production extraction passes a shard
            # writer so the large tensor is never duplicated inside features.pkl.
            payload["attention"] = resized.cpu().numpy().astype(np.float16)
        else:
            payload["shard_reference"] = dhcp_writer.add(resized).as_payload()
        attach_baseline(record, "dhcp", payload)

    if "projectaway" in selected:
        if context.projectaway_probability_cache is not None:
            features = context.projectaway_probability_cache.confidence(target_ids)
        elif context.unembedding_weight is None:
            raise ValueError("ProjectAway requires context.unembedding_weight [V,D]")
        else:
            features = compute_projectaway_internal_confidence(
                model_out.patch_hidden_states,
                context.unembedding_weight,
                target_ids,
                unembedding_bias=context.unembedding_bias,
                normalizer=context.projectaway_normalizer,
                vocab_chunk_size=int(
                    cfg.get("projectaway", {}).get("vocab_chunk_size", 8192)
                ),
                row_chunk_size=int(
                    cfg.get("projectaway", {}).get("row_chunk_size", 256)
                ),
            )
        attach_baseline(record, "projectaway", features.as_payload())

    if "halloc" in selected:
        object_index = (
            int(context.halloc_object_index)
            if context.halloc_object_index is not None
            else start
        )
        if context.halloc_cache_file is not None:
            payload = {
                "cache_file": str(context.halloc_cache_file),
                "object_index": object_index,
            }
        elif (
            model_out.response_hidden_states is not None
            and context.clip_visual_features is not None
        ):
            payload = {
                "lvlm_embeddings": model_out.response_hidden_states.detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy(),
                "clip_visual_features": context.clip_visual_features.detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy(),
                "object_index": object_index,
            }
        else:
            raise ValueError(
                "HalLoc requires halloc_cache_file, or response_hidden_states "
                "plus precomputed CLIP visual features"
            )
        attach_baseline(record, "halloc", payload)

    return record
