"""DGST-T continuous risk features.

This module keeps only the core DGST-T signals requested for the TGD port:
source_dist from FFN update, target_dist from attention times semantic
probability, Wasserstein/OT transport risk, prompt cosine features, and
context-confidence features.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import math
import multiprocessing as mp
import os
from typing import Any, Sequence
import warnings

import torch
import torch.nn.functional as F

EPS = 1e-12
TOPMASS_ALPHA_085 = 0.85
CAPPED_TOPMASS_MIN_K = 32
CAPPED_TOPMASS_MAX_K = 64
RELATIVE_VLL_MAD_EPSILON = 1e-6
COSINE16_TOP_K = 16
GAUSSIAN_MAD_SCALE = 1.4826
STATEUPD_ALPHA_TENTHS = tuple(range(1, 10))

# The active DGST profile keeps four matched target-gate constructions.  The
# prefix names deliberately encode both the decoder state used for the logit
# lens and whether the gate sees a raw target logit or a vocabulary-softmax
# probability.  This prevents the corresponding target-cosine features from
# being confused at training time.
FOUR_GATE_METHODS = (
    "hpre_raw_logit_gauss",
    "hpre_softmax_prob_gauss",
    "hmid_raw_logit_gauss",
    "hmid_softmax_prob_gauss",
)
HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD = "hpre_raw_logit_relative_vll"
DIRECT_HPRE_SOFTMAX_METHOD = "hpre_softmax_prob_direct"
RAW_ATTENTION_METHOD = "raw_attention"
TARGET_COMPARISON_METHODS = (
    *FOUR_GATE_METHODS,
    HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD,
    DIRECT_HPRE_SOFTMAX_METHOD,
    RAW_ATTENTION_METHOD,
)
FOUR_GATE_CAPTURE_FIELDS = (
    "prediction_hpre",
    "visual_hpre",
    "attention_support",
    "source_dist",
    "hpre_raw_target_logits",
    "hpre_softmax_target_probs",
    "hmid_raw_target_logits",
    "hmid_softmax_target_probs",
)

GATE_COMPARISON_METHODS = (
    "relative_vll",
    "softmax_relative_vll",
    "legacy_prob",
)

COST_VARIANT_RISK_KEYS = (
    "risk_geo",
    "risk_cosine_hpre",
    "risk_sqrt_hmid",
    "risk_sqrt_hpre",
    "risk_raw_attention_hmid",
    "risk_raw_attention_hpre",
    "gauss_risk_geo",
    "gauss_risk_cosine_hpre",
    "gauss_risk_sqrt_hmid",
    "gauss_risk_sqrt_hpre",
)

_EMD_PROCESS_POOL: ProcessPoolExecutor | None = None
_EMD_PROCESS_POOL_WORKERS: int | None = None


def compute_dgst_t(
    dgst_t_raw: dict[str, Any],
    *,
    tau: float = 0.07,
    source_distribution_mode: str = "softmax",
    transport_top_k: int = 64,
    cost_mode: str = "direct",
    lambda_d: float = 1.0,
    lambda_s: float = 1.0,
    lambda_t: float = 1.0,
    lambda_int: float = 1.0,
    baseline_layers: int = 10,
    risk_start_layer: int = 15,
    alpha: float = 2.0,
    ot_solver: str = "linprog",
    atarget_visual_top_k: int = 32,
    topmass_alpha: float = TOPMASS_ALPHA_085,
    capped_topmass_alpha: float = TOPMASS_ALPHA_085,
    capped_topmass_min_k: int = CAPPED_TOPMASS_MIN_K,
    capped_topmass_max_k: int = CAPPED_TOPMASS_MAX_K,
    compute_topmass_085: bool = True,
    compute_capped_topmass_085: bool = True,
    target_gate_mode: str = "legacy_prob",
    relative_vll_mad_epsilon: float = RELATIVE_VLL_MAD_EPSILON,
    relative_cost_mode: str | None = None,
    relative_cost_modes: Sequence[str] | str | None = None,
    relative_cost_state_modes: Sequence[str] | str | None = None,
    relative_cost_update_lambdas: Sequence[float] | float | None = None,
    relative_barrier_lambda: float = 1.0,
    relative_barrier_margin: float = 0.5,
    relative_barrier_max: float = 3.0,
    source_modes: Sequence[str] | None = None,
    target_attention_gammas: Sequence[float] | None = None,
    target_attention_epsilon: float = EPS,
    compute_ffn_injection_features: bool = True,
    ffn_injection_evidence_top_k: int = 32,
    ffn_injection_evidence_rank: int = 8,
    ffn_injection_eps: float = EPS,
    compute_dual_scope: bool = False,
) -> dict[str, Any]:
    """Compute DGST-T layer features from raw wrapper captures."""
    if _normalize_target_gate_mode(target_gate_mode) == "four_gate":
        raise RuntimeError(
            "four_gate must be computed directly from decoder captures via "
            "compute_dgst_t_batch_from_captures; the compact active profile "
            "does not build legacy dgst_t_raw."
        )
    support_states = dgst_t_raw["support_h_mid_states"]
    support_h_prev_states = dgst_t_raw.get("support_h_prev_states", support_states)
    support_output_states = dgst_t_raw.get("support_output_states", support_states)
    prompt_confidence_max = dgst_t_raw.get(
        "prompt_logit_lens_max_confidence",
        dgst_t_raw["prompt_logit_lens_top3_confidence"],
    )

    result = _compute_dgst_t_from_parts(
        source_ffn_states=_layer_tensors(dgst_t_raw["source_ffn_states"]),
        source_attn_states=_optional_layer_tensors(dgst_t_raw.get("source_attn_states")),
        prediction_hidden_states=_layer_tensors(dgst_t_raw["prediction_hidden_states"]),
        support_h_prev_states=_layer_tensors(support_h_prev_states),
        support_h_mid_states=_layer_tensors(support_states),
        support_output_states=_layer_tensors(support_output_states),
        support_attentions=_layer_tensors(dgst_t_raw["support_attentions"]),
        semantic_probs=_layer_tensors(dgst_t_raw["semantic_probs"]),
        relative_vll_logits=_optional_layer_tensors(dgst_t_raw.get("relative_vll_logits")),
        prompt_last_hidden_states=_layer_tensors(dgst_t_raw["prompt_last_hidden_states"]),
        prompt_mean_hidden_states=_layer_tensors(dgst_t_raw["prompt_mean_hidden_states"]),
        prompt_confidence_top3=_layer_tensors(dgst_t_raw["prompt_logit_lens_top3_confidence"]),
        prompt_confidence_max=_layer_tensors(prompt_confidence_max),
        support_positions=[int(pos) for pos in dgst_t_raw["support_positions"]],
        visual_start=int(dgst_t_raw["visual_start"]),
        visual_end=int(dgst_t_raw["visual_end"]),
        tau=tau,
        source_distribution_mode=source_distribution_mode,
        transport_top_k=transport_top_k,
        cost_mode=cost_mode,
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
        baseline_layers=baseline_layers,
        risk_start_layer=risk_start_layer,
        alpha=alpha,
        ot_solver=ot_solver,
        atarget_visual_top_k=atarget_visual_top_k,
        topmass_alpha=topmass_alpha,
        capped_topmass_alpha=capped_topmass_alpha,
        capped_topmass_min_k=capped_topmass_min_k,
        capped_topmass_max_k=capped_topmass_max_k,
        compute_topmass_085=compute_topmass_085,
        compute_capped_topmass_085=compute_capped_topmass_085,
        target_gate_mode=target_gate_mode,
        relative_vll_mad_epsilon=relative_vll_mad_epsilon,
        relative_cost_mode=relative_cost_mode,
        relative_cost_modes=relative_cost_modes,
        relative_cost_state_modes=relative_cost_state_modes,
        relative_cost_update_lambdas=relative_cost_update_lambdas,
        relative_barrier_lambda=relative_barrier_lambda,
        relative_barrier_margin=relative_barrier_margin,
        relative_barrier_max=relative_barrier_max,
        source_modes=source_modes,
        target_attention_gammas=target_attention_gammas,
        target_attention_epsilon=target_attention_epsilon,
        target_unembedding=dgst_t_raw.get("target_unembedding"),
        compute_ffn_injection_features=compute_ffn_injection_features,
        ffn_injection_evidence_top_k=ffn_injection_evidence_top_k,
        ffn_injection_evidence_rank=ffn_injection_evidence_rank,
        ffn_injection_eps=ffn_injection_eps,
        skip_relative_vll_visual_branch=bool(compute_dual_scope),
    )
    if compute_dual_scope:
        visual_result = _compute_visual_scope_result_from_parts(
            source_ffn_states=_layer_tensors(dgst_t_raw["source_ffn_states"]),
            source_attn_states=_optional_layer_tensors(dgst_t_raw.get("source_attn_states")),
            prediction_hidden_states=_layer_tensors(dgst_t_raw["prediction_hidden_states"]),
            support_h_prev_states=_layer_tensors(support_h_prev_states),
            support_h_mid_states=_layer_tensors(support_states),
            support_output_states=_layer_tensors(support_output_states),
            support_attentions=_layer_tensors(dgst_t_raw["support_attentions"]),
            semantic_probs=_layer_tensors(dgst_t_raw["semantic_probs"]),
            relative_vll_logits=_optional_layer_tensors(dgst_t_raw.get("relative_vll_logits")),
            prompt_last_hidden_states=_layer_tensors(dgst_t_raw["prompt_last_hidden_states"]),
            prompt_mean_hidden_states=_layer_tensors(dgst_t_raw["prompt_mean_hidden_states"]),
            prompt_confidence_top3=_layer_tensors(dgst_t_raw["prompt_logit_lens_top3_confidence"]),
            prompt_confidence_max=_layer_tensors(prompt_confidence_max),
            support_positions=[int(pos) for pos in dgst_t_raw["support_positions"]],
            visual_start=int(dgst_t_raw["visual_start"]),
            visual_end=int(dgst_t_raw["visual_end"]),
            tau=tau,
            source_distribution_mode=source_distribution_mode,
            transport_top_k=transport_top_k,
            cost_mode=cost_mode,
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
            ot_solver=ot_solver,
            atarget_visual_top_k=atarget_visual_top_k,
            topmass_alpha=topmass_alpha,
            capped_topmass_alpha=capped_topmass_alpha,
            capped_topmass_min_k=capped_topmass_min_k,
            capped_topmass_max_k=capped_topmass_max_k,
            compute_topmass_085=compute_topmass_085,
            compute_capped_topmass_085=compute_capped_topmass_085,
            target_gate_mode=target_gate_mode,
            relative_vll_mad_epsilon=relative_vll_mad_epsilon,
            relative_cost_mode=relative_cost_mode,
            relative_cost_modes=relative_cost_modes,
            relative_cost_state_modes=relative_cost_state_modes,
            relative_cost_update_lambdas=relative_cost_update_lambdas,
            relative_barrier_lambda=relative_barrier_lambda,
            relative_barrier_margin=relative_barrier_margin,
            relative_barrier_max=relative_barrier_max,
            source_modes=source_modes,
            target_attention_gammas=target_attention_gammas,
            target_attention_epsilon=target_attention_epsilon,
            target_unembedding=dgst_t_raw.get("target_unembedding"),
        )
        _merge_visual_scope_relative_fields(result, visual_result)
        _attach_c_vp_feature(
            result,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
        )
    result["dgst_t_relative_vll_logit_source"] = str(
        dgst_t_raw.get("relative_vll_logit_source", "h_mid")
    )
    result["dgst_t_dual_scope"] = bool(compute_dual_scope)
    return result


def compute_dgst_t_batch_from_captures(
    *,
    model: Any,
    full_input_ids: Sequence[int],
    prompt_tokenized_length: int,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    image_token_id: int,
    target_token_ids: Sequence[int],
    prediction_positions: Sequence[int],
    support_scope: str = "visual_prompt",
    semantic_chunk_size: int = 64,
    prompt_positions_override: Sequence[int] | None = None,
    tau: float = 0.07,
    source_distribution_mode: str = "softmax",
    transport_top_k: int = 64,
    cost_mode: str = "direct",
    lambda_d: float = 1.0,
    lambda_s: float = 1.0,
    lambda_t: float = 1.0,
    lambda_int: float = 1.0,
    baseline_layers: int = 10,
    risk_start_layer: int = 15,
    alpha: float = 2.0,
    ot_solver: str = "linprog",
    atarget_visual_top_k: int = 32,
    topmass_alpha: float = TOPMASS_ALPHA_085,
    capped_topmass_alpha: float = TOPMASS_ALPHA_085,
    capped_topmass_min_k: int = CAPPED_TOPMASS_MIN_K,
    capped_topmass_max_k: int = CAPPED_TOPMASS_MAX_K,
    compute_topmass_085: bool = True,
    compute_capped_topmass_085: bool = True,
    target_gate_mode: str = "legacy_prob",
    relative_vll_mad_epsilon: float = RELATIVE_VLL_MAD_EPSILON,
    relative_vll_logit_source: str = "h_mid",
    relative_cost_mode: str | None = None,
    relative_cost_modes: Sequence[str] | str | None = None,
    relative_cost_state_modes: Sequence[str] | str | None = None,
    relative_cost_update_lambdas: Sequence[float] | float | None = None,
    relative_barrier_lambda: float = 1.0,
    relative_barrier_margin: float = 0.5,
    relative_barrier_max: float = 3.0,
    source_modes: Sequence[str] | None = None,
    target_attention_gammas: Sequence[float] | None = None,
    target_attention_epsilon: float = EPS,
    compute_ffn_injection_features: bool = True,
    ffn_injection_evidence_top_k: int = 32,
    ffn_injection_evidence_rank: int = 8,
    ffn_injection_eps: float = EPS,
    compute_dual_scope: bool = False,
    four_gate_methods: Sequence[str] | None = None,
    four_gate_cost_modes: Sequence[str] | str | None = None,
    four_gate_support_modes: Sequence[str] | str | None = None,
    compute_prompt_cafe: bool = False,
    prompt_cafe_temperature: float = 10.0,
    prompt_cafe_layer: int = 22,
    release_layer_captures: bool = False,
) -> list[dict[str, Any]]:
    """Compute DGST-T for several target tokens directly from shared captures."""
    from models.dgst_capture import (
        attention_row_from_capture,
        resolve_output_embedding_layer,
        resolve_decoder_final_norm,
        resolve_prompt_positions,
        resolve_support_positions,
        apply_decoder_final_norm,
        normalize_relative_vll_logit_source,
        target_logits_multi,
        target_probabilities_multi,
        unembedding_for_token,
    )

    target_ids = [int(token_id) for token_id in target_token_ids]
    pred_positions = [int(position) for position in prediction_positions]
    if len(target_ids) != len(pred_positions):
        raise ValueError("target_token_ids and prediction_positions must have the same length.")
    if not target_ids:
        return []

    if _normalize_target_gate_mode(target_gate_mode) == "four_gate":
        prompt_positions = (
            [int(position) for position in prompt_positions_override]
            if prompt_positions_override is not None
            else resolve_prompt_positions(
                full_input_ids=full_input_ids,
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=int(image_token_id),
                visual_start=int(visual_start),
                visual_end=int(visual_end),
            )
        )
        return compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=captures,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            prompt_positions=prompt_positions,
            target_token_ids=target_ids,
            prediction_positions=pred_positions,
            semantic_chunk_size=int(semantic_chunk_size),
            tau=float(tau),
            transport_top_k=int(transport_top_k),
            target_region_top_k=int(atarget_visual_top_k),
            mad_epsilon=float(relative_vll_mad_epsilon),
            cost_mode=cost_mode,
            cost_modes=four_gate_cost_modes,
            enabled_methods=four_gate_methods,
            compute_dual_scope=bool(compute_dual_scope),
            support_modes=four_gate_support_modes,
            compute_prompt_cafe=bool(compute_prompt_cafe),
            prompt_cafe_temperature=float(prompt_cafe_temperature),
            prompt_cafe_layer=int(prompt_cafe_layer),
            compute_ffn_injection_features=bool(compute_ffn_injection_features),
            ffn_injection_eps=float(ffn_injection_eps),
            release_layer_captures=bool(release_layer_captures),
        )

    prompt_positions = (
        [int(position) for position in prompt_positions_override]
        if prompt_positions_override is not None
        else resolve_prompt_positions(
            full_input_ids=full_input_ids,
            prompt_tokenized_length=prompt_tokenized_length,
            image_token_id=int(image_token_id),
            visual_start=int(visual_start),
            visual_end=int(visual_end),
        )
    )
    support_positions = resolve_support_positions(
        visual_start=int(visual_start),
        visual_end=int(visual_end),
        prompt_positions=prompt_positions,
        support_scope=support_scope,
    )
    if not support_positions:
        raise ValueError("DGST-T support is empty.")
    if not prompt_positions:
        raise ValueError("DGST-T prompt positions are empty.")

    output_layer = resolve_output_embedding_layer(model)
    relative_source = normalize_relative_vll_logit_source(relative_vll_logit_source)
    final_norm_layer = (
        resolve_decoder_final_norm(model)
        if relative_source == "final_norm_h_mid"
        else None
    )
    hidden_size = int(captures[0]["h_mid"].shape[-1])
    target_unembeddings = [
        unembedding_for_token(
            output_layer=output_layer,
            target_token_id=token_id,
            hidden_size=hidden_size,
            device=captures[0]["h_mid"].device,
        )
        for token_id in target_ids
    ]
    parts = [
        {
            "source_ffn_states": [],
            "source_attn_states": [],
            "prediction_hidden_states": [],
            "support_h_prev_states": [],
            "support_h_mid_states": [],
            "support_output_states": [],
            "support_attentions": [],
            "semantic_probs": [],
            "relative_vll_logits": [],
            "prompt_last_hidden_states": [],
            "prompt_mean_hidden_states": [],
            "prompt_logit_lens_top3_confidence": [],
            "prompt_logit_lens_max_confidence": [],
        }
        for _ in target_ids
    ]
    cost_variant_mode = _normalize_target_gate_mode(target_gate_mode) == "cost_variants"

    for capture in captures:
        h_prev = capture["h_prev"][0]
        h_mid = capture["h_mid"][0]
        o_attn = capture["o_attn"][0]
        o_ffn = capture["o_ffn"][0]
        layer_hidden = h_mid + o_ffn
        device = h_mid.device
        support_index = torch.tensor(support_positions, dtype=torch.long, device=device)
        prompt_index = torch.tensor(prompt_positions, dtype=torch.long, device=device)

        support_prev_states = h_prev.index_select(0, support_index)
        support_states = h_mid.index_select(0, support_index)
        if relative_source == "h_prev":
            support_relative_states = support_prev_states
        elif final_norm_layer is not None:
            support_relative_states = apply_decoder_final_norm(final_norm_layer, support_states)
        else:
            support_relative_states = support_states
        support_output_states = layer_hidden.index_select(0, support_index)
        prompt_states = layer_hidden.index_select(0, prompt_index)

        if cost_variant_mode:
            # Cost variants gate with relative logits and do not persist these
            # legacy full-vocabulary semantic probabilities.
            support_semantic_all = torch.ones(
                (int(support_states.shape[0]), len(target_ids)),
                dtype=torch.float32,
                device=support_states.device,
            )
        else:
            support_semantic_all = target_probabilities_multi(
                output_layer=output_layer,
                states=support_states,
                target_token_ids=target_ids,
                chunk_size=semantic_chunk_size,
            )
        support_relative_logits_all = target_logits_multi(
            output_layer=output_layer,
            states=support_relative_states,
            target_token_ids=target_ids,
            chunk_size=semantic_chunk_size,
        )
        if cost_variant_mode:
            prompt_probs_all = torch.ones(
                (int(prompt_states.shape[0]), len(target_ids)),
                dtype=torch.float32,
                device=prompt_states.device,
            )
        else:
            prompt_probs_all = target_probabilities_multi(
                output_layer=output_layer,
                states=prompt_states,
                target_token_ids=target_ids,
                chunk_size=semantic_chunk_size,
            )
        top_k = min(3, int(prompt_probs_all.shape[0]))
        prompt_conf_top3_all = torch.topk(prompt_probs_all.float(), k=top_k, dim=0).values.mean(dim=0)
        prompt_conf_max_all = prompt_probs_all.float().max(dim=0).values
        prompt_last_state = prompt_states[-1]
        prompt_mean_state = prompt_states.mean(dim=0)

        for target_offset, prediction_position in enumerate(pred_positions):
            attention_row = attention_row_from_capture(
                capture,
                int(prediction_position),
            )
            support_attention = attention_row.index_select(
                1,
                support_index.to(attention_row.device),
            ).mean(dim=0)
            support_attention = support_attention.to(device=device, dtype=torch.float32)

            part = parts[target_offset]
            part["source_ffn_states"].append(o_ffn[int(prediction_position), :])
            part["source_attn_states"].append(o_attn[int(prediction_position), :])
            part["prediction_hidden_states"].append(layer_hidden[int(prediction_position), :])
            part["support_h_prev_states"].append(support_prev_states)
            part["support_h_mid_states"].append(support_states)
            part["support_output_states"].append(support_output_states)
            part["support_attentions"].append(support_attention)
            part["semantic_probs"].append(support_semantic_all[:, target_offset])
            part["relative_vll_logits"].append(support_relative_logits_all[:, target_offset])
            part["prompt_last_hidden_states"].append(prompt_last_state)
            part["prompt_mean_hidden_states"].append(prompt_mean_state)
            part["prompt_logit_lens_top3_confidence"].append(prompt_conf_top3_all[target_offset])
            part["prompt_logit_lens_max_confidence"].append(prompt_conf_max_all[target_offset])

    results: list[dict[str, Any]] = []
    for target_offset, part in enumerate(parts):
        result = _compute_dgst_t_from_parts(
            source_ffn_states=part["source_ffn_states"],
            source_attn_states=part["source_attn_states"],
            prediction_hidden_states=part["prediction_hidden_states"],
            support_h_prev_states=part["support_h_prev_states"],
            support_h_mid_states=part["support_h_mid_states"],
            support_output_states=part["support_output_states"],
            support_attentions=part["support_attentions"],
            semantic_probs=part["semantic_probs"],
            relative_vll_logits=part["relative_vll_logits"],
            prompt_last_hidden_states=part["prompt_last_hidden_states"],
            prompt_mean_hidden_states=part["prompt_mean_hidden_states"],
            prompt_confidence_top3=part["prompt_logit_lens_top3_confidence"],
            prompt_confidence_max=part["prompt_logit_lens_max_confidence"],
            support_positions=[int(position) for position in support_positions],
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            tau=tau,
            source_distribution_mode=source_distribution_mode,
            transport_top_k=transport_top_k,
            cost_mode=cost_mode,
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
            ot_solver=ot_solver,
            atarget_visual_top_k=atarget_visual_top_k,
            topmass_alpha=topmass_alpha,
            capped_topmass_alpha=capped_topmass_alpha,
            capped_topmass_min_k=capped_topmass_min_k,
            capped_topmass_max_k=capped_topmass_max_k,
            compute_topmass_085=compute_topmass_085,
            compute_capped_topmass_085=compute_capped_topmass_085,
            target_gate_mode=target_gate_mode,
            relative_vll_mad_epsilon=relative_vll_mad_epsilon,
            relative_cost_mode=relative_cost_mode,
            relative_cost_modes=relative_cost_modes,
            relative_cost_state_modes=relative_cost_state_modes,
            relative_cost_update_lambdas=relative_cost_update_lambdas,
            relative_barrier_lambda=relative_barrier_lambda,
            relative_barrier_margin=relative_barrier_margin,
            relative_barrier_max=relative_barrier_max,
            source_modes=source_modes,
            target_attention_gammas=target_attention_gammas,
            target_attention_epsilon=target_attention_epsilon,
            target_unembedding=target_unembeddings[target_offset],
            compute_ffn_injection_features=compute_ffn_injection_features,
            ffn_injection_evidence_top_k=ffn_injection_evidence_top_k,
            ffn_injection_evidence_rank=ffn_injection_evidence_rank,
            ffn_injection_eps=ffn_injection_eps,
            skip_relative_vll_visual_branch=bool(compute_dual_scope),
        )
        if compute_dual_scope:
            visual_result = _compute_visual_scope_result_from_parts(
                source_ffn_states=part["source_ffn_states"],
                source_attn_states=part["source_attn_states"],
                prediction_hidden_states=part["prediction_hidden_states"],
                support_h_prev_states=part["support_h_prev_states"],
                support_h_mid_states=part["support_h_mid_states"],
                support_output_states=part["support_output_states"],
                support_attentions=part["support_attentions"],
                semantic_probs=part["semantic_probs"],
                relative_vll_logits=part["relative_vll_logits"],
                prompt_last_hidden_states=part["prompt_last_hidden_states"],
                prompt_mean_hidden_states=part["prompt_mean_hidden_states"],
                prompt_confidence_top3=part["prompt_logit_lens_top3_confidence"],
                prompt_confidence_max=part["prompt_logit_lens_max_confidence"],
                support_positions=[int(position) for position in support_positions],
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                tau=tau,
                source_distribution_mode=source_distribution_mode,
                transport_top_k=transport_top_k,
                cost_mode=cost_mode,
                lambda_d=lambda_d,
                lambda_s=lambda_s,
                lambda_t=lambda_t,
                lambda_int=lambda_int,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
                ot_solver=ot_solver,
                atarget_visual_top_k=atarget_visual_top_k,
                topmass_alpha=topmass_alpha,
                capped_topmass_alpha=capped_topmass_alpha,
                capped_topmass_min_k=capped_topmass_min_k,
                capped_topmass_max_k=capped_topmass_max_k,
                compute_topmass_085=compute_topmass_085,
                compute_capped_topmass_085=compute_capped_topmass_085,
                target_gate_mode=target_gate_mode,
                relative_vll_mad_epsilon=relative_vll_mad_epsilon,
                relative_cost_mode=relative_cost_mode,
                relative_cost_modes=relative_cost_modes,
                relative_cost_state_modes=relative_cost_state_modes,
                relative_cost_update_lambdas=relative_cost_update_lambdas,
                relative_barrier_lambda=relative_barrier_lambda,
                relative_barrier_margin=relative_barrier_margin,
                relative_barrier_max=relative_barrier_max,
                source_modes=source_modes,
                target_attention_gammas=target_attention_gammas,
                target_attention_epsilon=target_attention_epsilon,
                target_unembedding=target_unembeddings[target_offset],
            )
            _merge_visual_scope_relative_fields(result, visual_result)
            _attach_c_vp_feature(
                result,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
            )
        result["dgst_t_relative_vll_logit_source"] = relative_source
        result["dgst_t_dual_scope"] = bool(compute_dual_scope)
        results.append(result)
    return results


def _compute_dgst_t_from_parts(
    *,
    source_ffn_states: Sequence[torch.Tensor],
    source_attn_states: Sequence[torch.Tensor] | None = None,
    prediction_hidden_states: Sequence[torch.Tensor],
    support_h_prev_states: Sequence[torch.Tensor],
    support_h_mid_states: Sequence[torch.Tensor],
    support_output_states: Sequence[torch.Tensor],
    support_attentions: Sequence[torch.Tensor],
    semantic_probs: Sequence[torch.Tensor],
    relative_vll_logits: Sequence[torch.Tensor] | None,
    prompt_last_hidden_states: Sequence[torch.Tensor],
    prompt_mean_hidden_states: Sequence[torch.Tensor],
    prompt_confidence_top3: Sequence[torch.Tensor],
    prompt_confidence_max: Sequence[torch.Tensor],
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    tau: float,
    source_distribution_mode: str,
    transport_top_k: int,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    baseline_layers: int,
    risk_start_layer: int,
    alpha: float,
    ot_solver: str,
    atarget_visual_top_k: int,
    topmass_alpha: float,
    capped_topmass_alpha: float,
    capped_topmass_min_k: int,
    capped_topmass_max_k: int,
    compute_topmass_085: bool,
    compute_capped_topmass_085: bool,
    target_gate_mode: str,
    relative_vll_mad_epsilon: float,
    relative_cost_mode: str,
    relative_cost_modes: Sequence[str] | str | None,
    relative_cost_state_modes: Sequence[str] | str | None,
    relative_cost_update_lambdas: Sequence[float] | float | None,
    relative_barrier_lambda: float,
    relative_barrier_margin: float,
    relative_barrier_max: float,
    source_modes: Sequence[str] | None,
    target_attention_gammas: Sequence[float] | None,
    target_attention_epsilon: float,
    target_unembedding: torch.Tensor | None = None,
    compute_ffn_injection_features: bool = True,
    ffn_injection_evidence_top_k: int = 32,
    ffn_injection_evidence_rank: int = 8,
    ffn_injection_eps: float = EPS,
    skip_relative_vll_visual_branch: bool = False,
) -> dict[str, Any]:
    layer_count = len(source_ffn_states)
    if layer_count == 0:
        raise ValueError("DGST-T requires at least one captured layer.")
    gate_mode = _normalize_target_gate_mode(target_gate_mode)
    relative_cost = _normalize_relative_cost_mode(
        relative_cost_mode,
        legacy_cost_mode=cost_mode,
    )
    emit_relative_cost_modes = _normalize_relative_cost_modes(
        relative_cost_modes,
        primary_cost_mode=relative_cost,
        legacy_cost_mode=cost_mode,
    )
    emit_relative_cost_fields = relative_cost_modes is not None
    relative_cost_state_specs = _relative_cost_state_specs(
        relative_cost_state_modes,
        relative_cost_update_lambdas,
    )
    emit_relative_cost_state_fields = relative_cost_state_modes is not None
    enabled_source_modes = _normalize_source_modes(source_modes)
    compute_delta_src = "delta_src" in enabled_source_modes
    gamma_values = _normalize_target_attention_gammas(target_attention_gammas)
    source_distribution = _normalize_source_distribution_mode(source_distribution_mode)
    compute_cost_variants = gate_mode == "cost_variants"
    compute_gate_comparison = gate_mode == "gate_comparison"
    compute_relative_vll = gate_mode in {"relative_vll", "dual", "cost_variants"}
    if (compute_relative_vll or compute_gate_comparison) and relative_vll_logits is None:
        raise ValueError("target_gate_mode requires relative_vll_logits, but they are missing.")
    for name, values in {
        "prediction_hidden_states": prediction_hidden_states,
        "support_h_prev_states": support_h_prev_states,
        "support_h_mid_states": support_h_mid_states,
        "support_output_states": support_output_states,
        "support_attentions": support_attentions,
        "semantic_probs": semantic_probs,
        "prompt_last_hidden_states": prompt_last_hidden_states,
        "prompt_mean_hidden_states": prompt_mean_hidden_states,
        "prompt_confidence_top3": prompt_confidence_top3,
        "prompt_confidence_max": prompt_confidence_max,
    }.items():
        if len(values) != layer_count:
            raise ValueError(f"DGST-T layer count mismatch for {name}.")
    if source_attn_states is not None and len(source_attn_states) != layer_count:
        raise ValueError("DGST-T layer count mismatch for source_attn_states.")
    if relative_vll_logits is not None and len(relative_vll_logits) != layer_count:
        raise ValueError("DGST-T layer count mismatch for relative_vll_logits.")

    if compute_gate_comparison:
        return _compute_gate_comparison_from_parts(
            source_ffn_states=source_ffn_states,
            source_attn_states=source_attn_states,
            prediction_hidden_states=prediction_hidden_states,
            support_h_prev_states=support_h_prev_states,
            support_h_mid_states=support_h_mid_states,
            support_attentions=support_attentions,
            semantic_probs=semantic_probs,
            relative_vll_logits=relative_vll_logits,
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            tau=tau,
            source_distribution_mode=source_distribution,
            transport_top_k=transport_top_k,
            ot_solver=ot_solver,
            atarget_visual_top_k=atarget_visual_top_k,
            relative_vll_mad_epsilon=relative_vll_mad_epsilon,
            relative_barrier_margin=relative_barrier_margin,
            relative_barrier_max=relative_barrier_max,
        )

    layer_stats = []
    risk_per_layer = []
    risk_topmass_085_per_layer = []
    risk_capped_topmass_085_per_layer = []
    risk_relative_vll_per_layer = []
    risk_relative_vll_capped_topmass_085_per_layer = []
    risk_visual_prompt_relative_vll_per_layer = []
    risk_visual_prompt_relative_vll_capped_topmass_085_per_layer = []
    prompt_last_cosine_per_layer = []
    prompt_mean_cosine_per_layer = []
    target_visual_hidden_cosine_per_layer = []
    target_visual_prompt_hidden_cosine_per_layer = []
    target_visual_hidden_cosine_capped_topmass_085_per_layer = []
    target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer = []
    target_visual_hidden_cosine_relative_vll_per_layer = []
    target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer = []
    target_visual_hidden_cosine16_relative_vll_per_layer = []
    target_visual_hpre_cosine_relative_vll_per_layer = []
    target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer = []
    target_visual_hpre_cosine16_relative_vll_per_layer = []
    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer = []
    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer = []
    target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer = []
    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer = []
    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer = []
    target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer = []
    relative_vll_evidence_strength_per_layer = []
    r_es_relative_vll_cost_geo_per_layer = []
    visual_prompt_relative_vll_target_visual_mass_per_layer = []
    visual_prompt_relative_vll_target_prompt_mass_per_layer = []
    visual_prompt_relative_vll_evidence_visual_mass_per_layer = []
    visual_prompt_relative_vll_evidence_prompt_mass_per_layer = []
    visual_prompt_relative_vll_evidence_strength_per_layer = []
    visual_prompt_relative_vll_source_visual_mass_per_layer = []
    visual_prompt_relative_vll_source_prompt_mass_per_layer = []
    vv_attention_dist_per_layer = []
    vv_support_attention_per_layer = []
    vv_source_dist_per_layer = []
    vv_semantic_gate_per_layer = []
    vv_gauss_semantic_gate_per_layer = []
    vv_raw_evidence_strength_per_layer = []
    vv_source_entropy_per_layer = []
    vv_target_entropy_per_layer = []
    vv_evidence_entropy_per_layer = []
    vv_source_topk_entropy_per_layer = []
    vp_attention_dist_per_layer = []
    vp_support_attention_per_layer = []
    vp_source_dist_per_layer = []
    vp_semantic_gate_per_layer = []
    vp_raw_evidence_visual_mass_per_layer = []
    vp_raw_evidence_prompt_mass_per_layer = []
    vp_raw_evidence_strength_per_layer = []
    vp_source_entropy_per_layer = []
    vp_target_entropy_per_layer = []
    vp_evidence_entropy_per_layer = []
    vp_source_topk_entropy_per_layer = []
    js_relative_vll_per_layer = []
    kl_target_source_relative_vll_per_layer = []
    kl_source_target_relative_vll_per_layer = []
    js_visual_prompt_relative_vll_per_layer = []
    kl_target_source_visual_prompt_relative_vll_per_layer = []
    kl_source_target_visual_prompt_relative_vll_per_layer = []
    prompt_confidence_top3_per_layer = []
    prompt_confidence_max_per_layer = []
    context_confidence_per_layer = []
    context_confidence_max_prompt_per_layer = []
    ffn_attn_dominance_per_layer = []
    ffn_evidence_orthogonal_dose_per_layer = []
    ffn_logit_lift_per_layer = []
    ffn_eif_fraction_svd_per_layer = []
    ffn_eif_dose_svd_per_layer = []
    ffn_eif_fraction_pca_per_layer = []
    ffn_eif_dose_pca_per_layer = []
    ffn_gate_ratio_per_layer = []
    ffn_fgr_per_layer = []
    delta_series: dict[str, dict[str, list[float]]] = {}
    source_variant_modes = [
        mode
        for mode in enabled_source_modes
        if mode in {"hmid_proj", "hprev_cos", "hprev_proj"}
    ]
    source_variant_series: dict[tuple[str, str], dict[str, list[float]]] = {}
    topk_region_series: dict[str, dict[str, list[float]]] = {}
    vv_source_variant_dist_per_layer: dict[str, list[torch.Tensor]] = {
        mode: [] for mode in source_variant_modes
    }
    vp_source_variant_dist_per_layer: dict[str, list[torch.Tensor]] = {
        mode: [] for mode in source_variant_modes
    }
    capped_support_series: dict[str, dict[str, list[list[int]]]] = {}
    cost_variant_series: dict[str, list[float]] = {
        key: [] for key in COST_VARIANT_RISK_KEYS
    }
    cost_variant_problem_series: dict[str, list[Any]] = {
        key: [] for key in COST_VARIANT_RISK_KEYS
    }
    relative_cost_series: dict[str, dict[str, list[float]]] = {}
    if compute_relative_vll and emit_relative_cost_fields:
        for mode in emit_relative_cost_modes:
            relative_cost_series[mode] = {
                "risk": [],
                "risk_cap": [],
                "vp_risk": [],
                "vp_risk_cap": [],
            }
    relative_cost_state_series: dict[tuple[str, str], dict[str, list[float]]] = {}
    emit_relative_cost_state_modes = [
        mode for mode in emit_relative_cost_modes if mode == "geo"
    ]
    if compute_relative_vll and emit_relative_cost_state_fields:
        for mode in emit_relative_cost_state_modes:
            for state_mode, update_lambda, state_slug in relative_cost_state_specs:
                relative_cost_state_series[(mode, state_slug)] = {
                    "state_mode": [state_mode],
                    "update_lambda": [] if update_lambda is None else [float(update_lambda)],
                    "risk": [],
                    "vp_risk": [],
                }
    has_prompt_support = _has_prompt_support_tokens(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
    )
    if compute_relative_vll and compute_delta_src:
        target_slugs = ["rvll"]
        if has_prompt_support:
            target_slugs.append("vp_rvll")
        for target_slug in target_slugs:
            for gamma in gamma_values:
                slug = _gamma_slug(gamma)
                key = f"{target_slug}_delta_src_{slug}"
                delta_series[key] = {"risk": [], "risk_cap": [], "cos": [], "cos_cap": []}

    for layer_idx in range(layer_count):
        layer_support_prev_states = support_h_prev_states[layer_idx].float()
        layer_support_states = support_h_mid_states[layer_idx].float()
        layer_support_output_states = support_output_states[layer_idx].float()
        layer_source_ffn = source_ffn_states[layer_idx].float()
        layer_prediction_hidden = prediction_hidden_states[layer_idx].float()
        layer_source_attn = (
            source_attn_states[layer_idx].to(layer_prediction_hidden.device).float()
            if source_attn_states is not None
            else None
        )
        layer_prediction_h_prev = (
            layer_prediction_hidden
            - layer_source_ffn.to(layer_prediction_hidden.device)
            - layer_source_attn
            if layer_source_attn is not None
            else None
        )
        layer_semantic_probs = semantic_probs[layer_idx].to(layer_support_states.device).float()
        layer_support_attentions = support_attentions[layer_idx].to(layer_support_states.device).float()

        source_dist = _source_distribution(
            source_update=layer_source_ffn,
            support_states=layer_support_states,
            tau=tau,
            mode=source_distribution,
        )
        source_variant_dists: dict[str, torch.Tensor] = {}
        if "hmid_proj" in source_variant_modes:
            source_variant_dists["hmid_proj"] = _source_projection_distribution(
                source_update=layer_source_ffn,
                support_states=layer_support_states,
                tau=tau,
                mode=source_distribution,
            )
        if "hprev_cos" in source_variant_modes:
            source_variant_dists["hprev_cos"] = _source_distribution(
                source_update=layer_source_ffn,
                support_states=layer_support_prev_states,
                tau=tau,
                mode=source_distribution,
            )
        if "hprev_proj" in source_variant_modes:
            source_variant_dists["hprev_proj"] = _source_projection_distribution(
                source_update=layer_source_ffn,
                support_states=layer_support_prev_states,
                tau=tau,
                mode=source_distribution,
            )
        source_dist_delta = None
        if compute_relative_vll and compute_delta_src:
            layer_prediction_h_mid = layer_prediction_hidden - layer_source_ffn
            source_dist_delta = _source_delta_distribution(
                prediction_h_mid=layer_prediction_h_mid,
                prediction_h_out=layer_prediction_hidden,
                support_states=layer_support_states,
                tau=tau,
                mode=source_distribution,
            )
        attention_dist = _renormalize(layer_support_attentions)
        target_dist = _renormalize(attention_dist * layer_semantic_probs)
        relative_vll_evidence = None
        semantic_gate_relative_vll = None
        relative_barrier_vll = None
        relative_vll_stats = None
        relative_vll_evidence_strength = None
        gauss_relative_vll_evidence = None
        gauss_semantic_gate_relative_vll = None
        visual_prompt_relative_vll_evidence = None
        semantic_gate_visual_prompt_relative_vll = None
        visual_prompt_relative_barrier_vll = None
        visual_prompt_relative_vll_stats = None
        visual_prompt_relative_vll_evidence_visual_mass = None
        visual_prompt_relative_vll_evidence_prompt_mass = None
        visual_prompt_relative_vll_evidence_strength = None
        visual_prompt_relative_vll_source_visual_mass = None
        visual_prompt_relative_vll_source_prompt_mass = None
        if compute_relative_vll:
            layer_relative_logits = relative_vll_logits[layer_idx].to(layer_support_states.device).float()
            if not skip_relative_vll_visual_branch:
                (
                    relative_vll_evidence,
                    semantic_gate_relative_vll,
                    relative_barrier_vll,
                    relative_vll_stats,
                ) = _relative_vll_evidence_signal(
                    attention_signal=layer_support_attentions,
                    target_logits=layer_relative_logits,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    candidate_scope="visual",
                    stat_prefix="relative_vll",
                    epsilon=relative_vll_mad_epsilon,
                    barrier_margin=relative_barrier_margin,
                    barrier_max=relative_barrier_max,
                )
                relative_vll_evidence_strength = relative_vll_evidence.sum()
                if compute_cost_variants:
                    (
                        gauss_relative_vll_evidence,
                        gauss_semantic_gate_relative_vll,
                        _gauss_relative_barrier,
                        _gauss_relative_stats,
                    ) = _relative_vll_evidence_signal(
                        attention_signal=layer_support_attentions,
                        target_logits=layer_relative_logits,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        candidate_scope="visual",
                        stat_prefix="gauss_relative_vll",
                        epsilon=relative_vll_mad_epsilon,
                        barrier_margin=relative_barrier_margin,
                        barrier_max=relative_barrier_max,
                        mad_scale=GAUSSIAN_MAD_SCALE,
                    )
                if not has_prompt_support:
                    vv_attention_dist_per_layer.append(attention_dist.detach().cpu())
                    vv_support_attention_per_layer.append(
                        layer_support_attentions.detach().cpu()
                    )
                    vv_source_dist_per_layer.append(source_dist.detach().cpu())
                    for mode, variant_dist in source_variant_dists.items():
                        vv_source_variant_dist_per_layer[mode].append(
                            variant_dist.detach().cpu()
                        )
                    vv_semantic_gate_per_layer.append(
                        semantic_gate_relative_vll.detach().cpu()
                    )
                    if gauss_semantic_gate_relative_vll is not None:
                        vv_gauss_semantic_gate_per_layer.append(
                            gauss_semantic_gate_relative_vll.detach().cpu()
                        )
                    vv_raw_evidence_strength_per_layer.append(
                        float(relative_vll_evidence_strength.detach())
                    )
            if has_prompt_support:
                (
                    visual_prompt_relative_vll_evidence,
                    semantic_gate_visual_prompt_relative_vll,
                    visual_prompt_relative_barrier_vll,
                    visual_prompt_relative_vll_stats,
                ) = _relative_vll_evidence_signal(
                    attention_signal=layer_support_attentions,
                    target_logits=layer_relative_logits,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    candidate_scope="visual_prompt",
                    stat_prefix="visual_prompt_relative_vll",
                    epsilon=relative_vll_mad_epsilon,
                    barrier_margin=relative_barrier_margin,
                    barrier_max=relative_barrier_max,
                )
                visual_prompt_relative_vll_evidence_visual_mass = _mass_for_scope(
                    values=visual_prompt_relative_vll_evidence,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    scope="visual",
                )
                visual_prompt_relative_vll_evidence_prompt_mass = _mass_for_scope(
                    values=visual_prompt_relative_vll_evidence,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    scope="prompt",
                )
                visual_prompt_relative_vll_evidence_strength = (
                    visual_prompt_relative_vll_evidence_visual_mass
                    + visual_prompt_relative_vll_evidence_prompt_mass
                )
                visual_prompt_relative_vll_source_visual_mass = _mass_for_scope(
                    values=source_dist,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    scope="visual",
                )
                visual_prompt_relative_vll_source_prompt_mass = _mass_for_scope(
                    values=source_dist,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    scope="prompt",
                )
                vp_attention_dist_per_layer.append(attention_dist.detach().cpu())
                vp_support_attention_per_layer.append(
                    layer_support_attentions.detach().cpu()
                )
                vp_source_dist_per_layer.append(source_dist.detach().cpu())
                for mode, variant_dist in source_variant_dists.items():
                    vp_source_variant_dist_per_layer[mode].append(
                        variant_dist.detach().cpu()
                    )
                vp_semantic_gate_per_layer.append(
                    semantic_gate_visual_prompt_relative_vll.detach().cpu()
                )
                vp_raw_evidence_visual_mass_per_layer.append(
                    float(visual_prompt_relative_vll_evidence_visual_mass.detach())
                )
                vp_raw_evidence_prompt_mass_per_layer.append(
                    float(visual_prompt_relative_vll_evidence_prompt_mass.detach())
                )
                vp_raw_evidence_strength_per_layer.append(
                    float(visual_prompt_relative_vll_evidence_strength.detach())
                )

        if compute_cost_variants:
            if has_prompt_support:
                raise ValueError("cost_variants mode requires visual-only (VV) support.")
            if relative_vll_evidence is None or gauss_relative_vll_evidence is None:
                raise ValueError("cost_variants mode requires both relative-VLL targets.")
            layer_cost_variant_problems = _prepare_cost_variant_problems(
                source_dist=source_dist,
                relative_target=relative_vll_evidence,
                gauss_target=gauss_relative_vll_evidence,
                raw_attention_target=attention_dist,
                hmid_states=layer_support_states,
                hpre_states=layer_support_prev_states,
                transport_top_k=transport_top_k,
                ot_solver=ot_solver,
            )
            for key, problem in layer_cost_variant_problems.items():
                cost_variant_problem_series[key].append(problem)

        ffn_injection = None
        ffn_gate_ratio = None
        if compute_ffn_injection_features:
            layer_prediction_h_mid_for_gate = layer_prediction_hidden - layer_source_ffn
            ffn_gate_ratio = _ffn_gate_ratio(
                source_ffn=layer_source_ffn,
                prediction_h_mid=layer_prediction_h_mid_for_gate,
                eps=ffn_injection_eps,
            )
            ffn_injection = _compute_ffn_injection_features(
                source_ffn=layer_source_ffn,
                source_attn=layer_source_attn,
                prediction_hidden=layer_prediction_hidden,
                support_states=layer_support_states,
                target_dist_visual=relative_vll_evidence,
                support_positions=support_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                target_unembedding=target_unembedding,
                evidence_top_k=ffn_injection_evidence_top_k,
                evidence_rank=ffn_injection_evidence_rank,
                eps=ffn_injection_eps,
            )
            ffn_attn_dominance_per_layer.append(
                float(ffn_injection["ffn_attn_dominance"])
            )
            ffn_evidence_orthogonal_dose_per_layer.append(
                float(ffn_injection["ffn_evidence_orthogonal_dose"])
            )
            ffn_logit_lift_per_layer.append(float(ffn_injection["ffn_logit_lift"]))
            ffn_eif_fraction_svd_per_layer.append(
                float(ffn_injection["ffn_eif_fraction_svd"])
            )
            ffn_eif_dose_svd_per_layer.append(
                float(ffn_injection["ffn_eif_dose_svd"])
            )
            ffn_eif_fraction_pca_per_layer.append(
                float(ffn_injection["ffn_eif_fraction_pca"])
            )
            ffn_eif_dose_pca_per_layer.append(
                float(ffn_injection["ffn_eif_dose_pca"])
            )

        delta_layer_stats = {}
        if compute_relative_vll and compute_delta_src and source_dist_delta is not None:
            delta_targets = [
                (
                    "rvll",
                    "visual",
                    _target_hidden_topk_visual_cosine,
                    _target_hidden_capped_topmass_visual_cosine,
                )
            ]
            if has_prompt_support:
                delta_targets.append(
                    (
                        "vp_rvll",
                        "visual_prompt",
                        _target_hidden_topk_support_cosine,
                        _target_hidden_capped_topmass_support_cosine,
                    )
                )
            for target_slug, candidate_scope, cosine_fn, capped_cosine_fn in delta_targets:
                for gamma in gamma_values:
                    gamma_slug = _gamma_slug(gamma)
                    series_key = f"{target_slug}_delta_src_{gamma_slug}"
                    target_evidence_delta, semantic_gate_delta, barrier_delta, target_stats = (
                        _relative_vll_evidence_signal(
                            attention_signal=layer_support_attentions,
                            target_logits=layer_relative_logits,
                            support_positions=support_positions,
                            visual_start=visual_start,
                            visual_end=visual_end,
                            candidate_scope=candidate_scope,
                            stat_prefix=series_key,
                            epsilon=relative_vll_mad_epsilon,
                            barrier_margin=relative_barrier_margin,
                            barrier_max=relative_barrier_max,
                            attention_gamma=gamma,
                            attention_epsilon=target_attention_epsilon,
                        )
                    )
                    delta_support = _topk_union_indices(
                        source_dist_delta,
                        target_evidence_delta,
                        transport_top_k,
                    )
                    delta_risk = _transport_risk_on_support(
                        source_dist=source_dist_delta,
                        target_dist=target_evidence_delta,
                        support_states=layer_support_states,
                        semantic_probs=semantic_gate_delta,
                        support=delta_support,
                        cost_mode=relative_cost,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=barrier_delta,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    if target_slug == "rvll":
                        delta_cos = cosine_fn(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=target_evidence_delta,
                            support_positions=support_positions,
                            visual_start=visual_start,
                            visual_end=visual_end,
                            top_k=atarget_visual_top_k,
                        )
                    else:
                        delta_cos = cosine_fn(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=target_evidence_delta,
                            top_k=atarget_visual_top_k,
                        )
                    delta_series[series_key]["risk"].append(float(delta_risk))
                    delta_series[series_key]["cos"].append(float(delta_cos))
                    delta_layer_stats[f"risk_{series_key}"] = float(delta_risk)
                    delta_layer_stats[f"cos_{series_key}"] = float(delta_cos)
                    delta_layer_stats[f"selected_support_size_{series_key}"] = int(
                        delta_support.numel()
                    )
                    delta_layer_stats.update(target_stats)

                    if compute_capped_topmass_085:
                        delta_capped_support = _capped_topmass_union_indices(
                            source_dist_delta,
                            target_evidence_delta,
                            capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                        delta_risk_cap = _transport_risk_on_support(
                            source_dist=source_dist_delta,
                            target_dist=target_evidence_delta,
                            support_states=layer_support_states,
                            semantic_probs=semantic_gate_delta,
                            support=delta_capped_support,
                            cost_mode=relative_cost,
                            lambda_d=lambda_d,
                            lambda_s=lambda_s,
                            lambda_t=lambda_t,
                            lambda_int=lambda_int,
                            relative_barrier=barrier_delta,
                            relative_barrier_lambda=relative_barrier_lambda,
                            ot_solver=ot_solver,
                        )
                        if target_slug == "rvll":
                            delta_cos_cap = capped_cosine_fn(
                                target_hidden=layer_prediction_hidden,
                                support_output_states=layer_support_output_states,
                                target_dist=target_evidence_delta,
                                support_positions=support_positions,
                                visual_start=visual_start,
                                visual_end=visual_end,
                                alpha=capped_topmass_alpha,
                                min_k=capped_topmass_min_k,
                                max_k=capped_topmass_max_k,
                            )
                        else:
                            delta_cos_cap = capped_cosine_fn(
                                target_hidden=layer_prediction_hidden,
                                support_output_states=layer_support_output_states,
                                target_dist=target_evidence_delta,
                                alpha=capped_topmass_alpha,
                                min_k=capped_topmass_min_k,
                                max_k=capped_topmass_max_k,
                            )
                        delta_series[series_key]["risk_cap"].append(float(delta_risk_cap))
                        delta_series[series_key]["cos_cap"].append(float(delta_cos_cap))
                        delta_layer_stats[f"risk_{series_key}_cap085"] = float(delta_risk_cap)
                        delta_layer_stats[f"cos_{series_key}_cap085"] = float(delta_cos_cap)
                        delta_layer_stats[f"selected_support_size_{series_key}_cap085"] = int(
                            delta_capped_support.numel()
                        )

        support = _topk_union_indices(source_dist, target_dist, transport_top_k)
        transport_risk = _transport_risk_on_support(
            source_dist=source_dist,
            target_dist=target_dist,
            support_states=layer_support_states,
            semantic_probs=layer_semantic_probs,
            support=support,
            cost_mode=cost_mode,
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
            ot_solver=ot_solver,
        )
        topmass_support = None
        transport_risk_topmass_085 = None
        if compute_topmass_085:
            topmass_support = _topmass_union_indices(source_dist, target_dist, topmass_alpha)
            transport_risk_topmass_085 = _transport_risk_on_support(
                source_dist=source_dist,
                target_dist=target_dist,
                support_states=layer_support_states,
                semantic_probs=layer_semantic_probs,
                support=topmass_support,
                cost_mode=cost_mode,
                lambda_d=lambda_d,
                lambda_s=lambda_s,
                lambda_t=lambda_t,
                lambda_int=lambda_int,
                ot_solver=ot_solver,
            )

        capped_topmass_support = None
        transport_risk_capped_topmass_085 = None
        if compute_capped_topmass_085:
            capped_topmass_support = _capped_topmass_union_indices(
                source_dist,
                target_dist,
                capped_topmass_alpha,
                min_k=capped_topmass_min_k,
                max_k=capped_topmass_max_k,
            )
            _append_capped_support_series(
                series=capped_support_series,
                stem="dgst_t",
                support=capped_topmass_support,
                support_positions=support_positions,
            )
            transport_risk_capped_topmass_085 = _transport_risk_on_support(
                source_dist=source_dist,
                target_dist=target_dist,
                support_states=layer_support_states,
                semantic_probs=layer_semantic_probs,
                support=capped_topmass_support,
                cost_mode=cost_mode,
                lambda_d=lambda_d,
                lambda_s=lambda_s,
                lambda_t=lambda_t,
                lambda_int=lambda_int,
                ot_solver=ot_solver,
            )

        relative_vll_support = None
        transport_risk_relative_vll = None
        transport_risk_relative_vll_cost_geo = None
        relative_vll_capped_support = None
        transport_risk_relative_vll_capped_topmass_085 = None
        visual_prompt_relative_vll_support = None
        transport_risk_visual_prompt_relative_vll = None
        visual_prompt_relative_vll_capped_support = None
        transport_risk_visual_prompt_relative_vll_capped_topmass_085 = None
        visual_prompt_relative_vll_target_visual_mass = None
        visual_prompt_relative_vll_target_prompt_mass = None
        divergence_relative_vll = None
        divergence_visual_prompt_relative_vll = None
        if compute_relative_vll:
            if relative_vll_evidence is not None:
                relative_vll_support = _topk_union_indices(
                    source_dist,
                    relative_vll_evidence,
                    transport_top_k,
                )
                divergence_relative_vll = _source_target_divergences_on_support(
                    source=source_dist,
                    target=relative_vll_evidence,
                    support=relative_vll_support,
                )
                vv_source_entropy_per_layer.append(_normalized_entropy(source_dist))
                vv_target_entropy_per_layer.append(
                    _normalized_entropy(
                        relative_vll_evidence.index_select(0, relative_vll_support),
                        normalizer_count=int(relative_vll_support.numel()),
                    )
                )
                vv_evidence_entropy_per_layer.append(
                    _normalized_entropy(relative_vll_evidence)
                )
                vv_source_topk_entropy_per_layer.append(
                    _normalized_entropy(
                        source_dist.index_select(0, relative_vll_support),
                        normalizer_count=int(relative_vll_support.numel()),
                    )
                )
                relative_vll_risks = _transport_risks_by_cost(
                    source_dist=source_dist,
                    target_dist=relative_vll_evidence,
                    support_states=layer_support_states,
                    semantic_probs=semantic_gate_relative_vll,
                    support=relative_vll_support,
                    cost_modes=emit_relative_cost_modes,
                    lambda_d=lambda_d,
                    lambda_s=lambda_s,
                    lambda_t=lambda_t,
                    lambda_int=lambda_int,
                    relative_barrier=relative_barrier_vll,
                    relative_barrier_lambda=relative_barrier_lambda,
                    ot_solver=ot_solver,
                )
                transport_risk_relative_vll = relative_vll_risks[relative_cost]
                transport_risk_relative_vll_cost_geo = relative_vll_risks.get("geo")
                _append_topk_region_features(
                    series=topk_region_series,
                    stem="dgst_t_vv",
                    source_dist=source_dist,
                    target_dist=relative_vll_evidence,
                    support_states=layer_support_states,
                    semantic_probs=semantic_gate_relative_vll,
                    transport_top_k=transport_top_k,
                    lambda_d=lambda_d,
                    lambda_s=lambda_s,
                    lambda_t=lambda_t,
                    lambda_int=lambda_int,
                    ot_solver=ot_solver,
                    union_risk=transport_risk_relative_vll_cost_geo,
                )
                if emit_relative_cost_fields:
                    for mode, value in relative_vll_risks.items():
                        relative_cost_series[mode]["risk"].append(float(value))
                if emit_relative_cost_state_fields:
                    relative_vll_state_risks = _transport_risks_by_cost_state(
                        source_dist=source_dist,
                        target_dist=relative_vll_evidence,
                        support_states=layer_support_states,
                        support_output_states=layer_support_output_states,
                        semantic_probs=semantic_gate_relative_vll,
                        support=relative_vll_support,
                        cost_modes=emit_relative_cost_state_modes,
                        cost_state_specs=relative_cost_state_specs,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    for key, value in relative_vll_state_risks.items():
                        relative_cost_state_series[key]["risk"].append(float(value))
                if compute_capped_topmass_085:
                    relative_vll_capped_support = _capped_topmass_union_indices(
                        source_dist,
                        relative_vll_evidence,
                        capped_topmass_alpha,
                        min_k=capped_topmass_min_k,
                        max_k=capped_topmass_max_k,
                    )
                    _append_capped_support_series(
                        series=capped_support_series,
                        stem="dgst_t_vv",
                        support=relative_vll_capped_support,
                        support_positions=support_positions,
                    )
                    relative_vll_capped_risks = _transport_risks_by_cost(
                        source_dist=source_dist,
                        target_dist=relative_vll_evidence,
                        support_states=layer_support_states,
                        semantic_probs=semantic_gate_relative_vll,
                        support=relative_vll_capped_support,
                        cost_modes=emit_relative_cost_modes,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    transport_risk_relative_vll_capped_topmass_085 = (
                        relative_vll_capped_risks[relative_cost]
                    )
                    if emit_relative_cost_fields:
                        for mode, value in relative_vll_capped_risks.items():
                            relative_cost_series[mode]["risk_cap"].append(float(value))
                for source_mode, variant_source_dist in source_variant_dists.items():
                    variant_cost_states = (
                        layer_support_states
                        if source_mode.startswith("hmid")
                        else layer_support_prev_states
                    )
                    variant_series = source_variant_series.setdefault(
                        ("relative_vll", source_mode),
                        {"risk": [], "risk_cap": []},
                    )
                    variant_support = _topk_union_indices(
                        variant_source_dist,
                        relative_vll_evidence,
                        transport_top_k,
                    )
                    variant_risk = _transport_risk_on_support(
                        source_dist=variant_source_dist,
                        target_dist=relative_vll_evidence,
                        support_states=variant_cost_states,
                        semantic_probs=semantic_gate_relative_vll,
                        support=variant_support,
                        cost_mode=relative_cost,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    variant_series["risk"].append(float(variant_risk))
                    _append_topk_region_features(
                        series=topk_region_series,
                        stem=f"dgst_t_vv_source_{source_mode}",
                        source_dist=variant_source_dist,
                        target_dist=relative_vll_evidence,
                        support_states=variant_cost_states,
                        semantic_probs=semantic_gate_relative_vll,
                        transport_top_k=transport_top_k,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        ot_solver=ot_solver,
                        union_risk=variant_risk if relative_cost == "geo" else None,
                    )
                    if compute_capped_topmass_085:
                        variant_capped_support = _capped_topmass_union_indices(
                            variant_source_dist,
                            relative_vll_evidence,
                            capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                        _append_capped_support_series(
                            series=capped_support_series,
                            stem=f"dgst_t_vv_source_{source_mode}",
                            support=variant_capped_support,
                            support_positions=support_positions,
                        )
                        variant_risk_cap = _transport_risk_on_support(
                            source_dist=variant_source_dist,
                            target_dist=relative_vll_evidence,
                            support_states=variant_cost_states,
                            semantic_probs=semantic_gate_relative_vll,
                            support=variant_capped_support,
                            cost_mode=relative_cost,
                            lambda_d=lambda_d,
                            lambda_s=lambda_s,
                            lambda_t=lambda_t,
                            lambda_int=lambda_int,
                            relative_barrier=relative_barrier_vll,
                            relative_barrier_lambda=relative_barrier_lambda,
                            ot_solver=ot_solver,
                        )
                        variant_series["risk_cap"].append(float(variant_risk_cap))
            if visual_prompt_relative_vll_evidence is not None:
                visual_prompt_relative_vll_support = _topk_union_indices(
                    source_dist,
                    visual_prompt_relative_vll_evidence,
                    transport_top_k,
                )
                visual_prompt_relative_vll_target_visual_mass = (
                    _local_mass_for_scope(
                        values=visual_prompt_relative_vll_evidence,
                        support=visual_prompt_relative_vll_support,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        scope="visual",
                    )
                )
                visual_prompt_relative_vll_target_prompt_mass = (
                    _local_mass_for_scope(
                        values=visual_prompt_relative_vll_evidence,
                        support=visual_prompt_relative_vll_support,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        scope="prompt",
                    )
                )
                divergence_visual_prompt_relative_vll = _source_target_divergences_on_support(
                    source=source_dist,
                    target=visual_prompt_relative_vll_evidence,
                    support=visual_prompt_relative_vll_support,
                )
                vp_source_entropy_per_layer.append(_normalized_entropy(source_dist))
                vp_target_entropy_per_layer.append(
                    _normalized_entropy(
                        visual_prompt_relative_vll_evidence.index_select(
                            0, visual_prompt_relative_vll_support
                        ),
                        normalizer_count=int(visual_prompt_relative_vll_support.numel()),
                    )
                )
                vp_evidence_entropy_per_layer.append(
                    _normalized_entropy(visual_prompt_relative_vll_evidence)
                )
                vp_source_topk_entropy_per_layer.append(
                    _normalized_entropy(
                        source_dist.index_select(0, visual_prompt_relative_vll_support),
                        normalizer_count=int(visual_prompt_relative_vll_support.numel()),
                    )
                )
                visual_prompt_relative_vll_risks = _transport_risks_by_cost(
                    source_dist=source_dist,
                    target_dist=visual_prompt_relative_vll_evidence,
                    support_states=layer_support_states,
                    semantic_probs=semantic_gate_visual_prompt_relative_vll,
                    support=visual_prompt_relative_vll_support,
                    cost_modes=emit_relative_cost_modes,
                    lambda_d=lambda_d,
                    lambda_s=lambda_s,
                    lambda_t=lambda_t,
                    lambda_int=lambda_int,
                    relative_barrier=visual_prompt_relative_barrier_vll,
                    relative_barrier_lambda=relative_barrier_lambda,
                    ot_solver=ot_solver,
                )
                transport_risk_visual_prompt_relative_vll = (
                    visual_prompt_relative_vll_risks[relative_cost]
                )
                _append_topk_region_features(
                    series=topk_region_series,
                    stem="dgst_t_vp",
                    source_dist=source_dist,
                    target_dist=visual_prompt_relative_vll_evidence,
                    support_states=layer_support_states,
                    semantic_probs=semantic_gate_visual_prompt_relative_vll,
                    transport_top_k=transport_top_k,
                    lambda_d=lambda_d,
                    lambda_s=lambda_s,
                    lambda_t=lambda_t,
                    lambda_int=lambda_int,
                    ot_solver=ot_solver,
                    union_risk=visual_prompt_relative_vll_risks.get("geo"),
                )
                if emit_relative_cost_fields:
                    for mode, value in visual_prompt_relative_vll_risks.items():
                        relative_cost_series[mode]["vp_risk"].append(float(value))
                if emit_relative_cost_state_fields:
                    visual_prompt_relative_vll_state_risks = _transport_risks_by_cost_state(
                        source_dist=source_dist,
                        target_dist=visual_prompt_relative_vll_evidence,
                        support_states=layer_support_states,
                        support_output_states=layer_support_output_states,
                        semantic_probs=semantic_gate_visual_prompt_relative_vll,
                        support=visual_prompt_relative_vll_support,
                        cost_modes=emit_relative_cost_state_modes,
                        cost_state_specs=relative_cost_state_specs,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=visual_prompt_relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    for key, value in visual_prompt_relative_vll_state_risks.items():
                        relative_cost_state_series[key]["vp_risk"].append(float(value))
                if compute_capped_topmass_085:
                    visual_prompt_relative_vll_capped_support = _capped_topmass_union_indices(
                        source_dist,
                        visual_prompt_relative_vll_evidence,
                        capped_topmass_alpha,
                        min_k=capped_topmass_min_k,
                        max_k=capped_topmass_max_k,
                    )
                    _append_capped_support_series(
                        series=capped_support_series,
                        stem="dgst_t_vp",
                        support=visual_prompt_relative_vll_capped_support,
                        support_positions=support_positions,
                    )
                    visual_prompt_relative_vll_capped_risks = _transport_risks_by_cost(
                        source_dist=source_dist,
                        target_dist=visual_prompt_relative_vll_evidence,
                        support_states=layer_support_states,
                        semantic_probs=semantic_gate_visual_prompt_relative_vll,
                        support=visual_prompt_relative_vll_capped_support,
                        cost_modes=emit_relative_cost_modes,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=visual_prompt_relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    transport_risk_visual_prompt_relative_vll_capped_topmass_085 = (
                        visual_prompt_relative_vll_capped_risks[relative_cost]
                    )
                    if emit_relative_cost_fields:
                        for mode, value in visual_prompt_relative_vll_capped_risks.items():
                            relative_cost_series[mode]["vp_risk_cap"].append(float(value))
                for source_mode, variant_source_dist in source_variant_dists.items():
                    variant_cost_states = (
                        layer_support_states
                        if source_mode.startswith("hmid")
                        else layer_support_prev_states
                    )
                    variant_series = source_variant_series.setdefault(
                        ("visual_prompt_relative_vll", source_mode),
                        {"risk": [], "risk_cap": []},
                    )
                    variant_support = _topk_union_indices(
                        variant_source_dist,
                        visual_prompt_relative_vll_evidence,
                        transport_top_k,
                    )
                    variant_risk = _transport_risk_on_support(
                        source_dist=variant_source_dist,
                        target_dist=visual_prompt_relative_vll_evidence,
                        support_states=variant_cost_states,
                        semantic_probs=semantic_gate_visual_prompt_relative_vll,
                        support=variant_support,
                        cost_mode=relative_cost,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        relative_barrier=visual_prompt_relative_barrier_vll,
                        relative_barrier_lambda=relative_barrier_lambda,
                        ot_solver=ot_solver,
                    )
                    variant_series["risk"].append(float(variant_risk))
                    _append_topk_region_features(
                        series=topk_region_series,
                        stem=f"dgst_t_vp_source_{source_mode}",
                        source_dist=variant_source_dist,
                        target_dist=visual_prompt_relative_vll_evidence,
                        support_states=variant_cost_states,
                        semantic_probs=semantic_gate_visual_prompt_relative_vll,
                        transport_top_k=transport_top_k,
                        lambda_d=lambda_d,
                        lambda_s=lambda_s,
                        lambda_t=lambda_t,
                        lambda_int=lambda_int,
                        ot_solver=ot_solver,
                        union_risk=variant_risk if relative_cost == "geo" else None,
                    )
                    if compute_capped_topmass_085:
                        variant_capped_support = _capped_topmass_union_indices(
                            variant_source_dist,
                            visual_prompt_relative_vll_evidence,
                            capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                        _append_capped_support_series(
                            series=capped_support_series,
                            stem=f"dgst_t_vp_source_{source_mode}",
                            support=variant_capped_support,
                            support_positions=support_positions,
                        )
                        variant_risk_cap = _transport_risk_on_support(
                            source_dist=variant_source_dist,
                            target_dist=visual_prompt_relative_vll_evidence,
                            support_states=variant_cost_states,
                            semantic_probs=semantic_gate_visual_prompt_relative_vll,
                            support=variant_capped_support,
                            cost_mode=relative_cost,
                            lambda_d=lambda_d,
                            lambda_s=lambda_s,
                            lambda_t=lambda_t,
                            lambda_int=lambda_int,
                            relative_barrier=visual_prompt_relative_barrier_vll,
                            relative_barrier_lambda=relative_barrier_lambda,
                            ot_solver=ot_solver,
                        )
                        variant_series["risk_cap"].append(float(variant_risk_cap))

        prompt_last = prompt_last_hidden_states[layer_idx].to(layer_prediction_hidden.device).float()
        prompt_mean = prompt_mean_hidden_states[layer_idx].to(layer_prediction_hidden.device).float()
        prompt_last_cosine = float(
            F.cosine_similarity(
                layer_prediction_hidden.unsqueeze(0),
                prompt_last.unsqueeze(0),
                dim=-1,
            ).item()
        )
        prompt_mean_cosine = float(
            F.cosine_similarity(
                layer_prediction_hidden.unsqueeze(0),
                prompt_mean.unsqueeze(0),
                dim=-1,
            ).item()
        )
        target_visual_hidden_cosine = _target_hidden_topk_visual_cosine(
            target_hidden=layer_prediction_hidden,
            support_output_states=layer_support_output_states,
            target_dist=target_dist,
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            top_k=atarget_visual_top_k,
        )
        target_visual_prompt_hidden_cosine = _target_hidden_topk_support_cosine(
            target_hidden=layer_prediction_hidden,
            support_output_states=layer_support_output_states,
            target_dist=target_dist,
            top_k=atarget_visual_top_k,
        )
        target_visual_hidden_cosine_capped_topmass_085 = None
        target_visual_prompt_hidden_cosine_capped_topmass_085 = None
        if compute_capped_topmass_085:
            target_visual_hidden_cosine_capped_topmass_085 = _target_hidden_capped_topmass_visual_cosine(
                target_hidden=layer_prediction_hidden,
                support_output_states=layer_support_output_states,
                target_dist=target_dist,
                support_positions=support_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                alpha=capped_topmass_alpha,
                min_k=capped_topmass_min_k,
                max_k=capped_topmass_max_k,
            )
            target_visual_prompt_hidden_cosine_capped_topmass_085 = _target_hidden_capped_topmass_support_cosine(
                target_hidden=layer_prediction_hidden,
                support_output_states=layer_support_output_states,
                target_dist=target_dist,
                alpha=capped_topmass_alpha,
                min_k=capped_topmass_min_k,
                max_k=capped_topmass_max_k,
            )
        target_visual_hidden_cosine_relative_vll = None
        target_visual_hidden_cosine_relative_vll_capped_topmass_085 = None
        target_visual_hidden_cosine16_relative_vll = None
        target_visual_hpre_cosine_relative_vll = None
        target_visual_hpre_cosine_relative_vll_capped_topmass_085 = None
        target_visual_hpre_cosine16_relative_vll = None
        target_visual_prompt_hidden_cosine_visual_prompt_relative_vll = None
        target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 = None
        target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll = None
        target_visual_prompt_hpre_cosine_visual_prompt_relative_vll = None
        target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085 = None
        target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll = None
        if compute_relative_vll:
            if relative_vll_evidence is not None:
                target_visual_hidden_cosine_relative_vll = _target_hidden_topk_visual_cosine(
                    target_hidden=layer_prediction_hidden,
                    support_output_states=layer_support_output_states,
                    target_dist=relative_vll_evidence,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    top_k=atarget_visual_top_k,
                )
                target_visual_hidden_cosine16_relative_vll = _target_hidden_topk_visual_cosine(
                    target_hidden=layer_prediction_hidden,
                    support_output_states=layer_support_output_states,
                    target_dist=relative_vll_evidence,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    top_k=COSINE16_TOP_K,
                )
                if compute_capped_topmass_085:
                    target_visual_hidden_cosine_relative_vll_capped_topmass_085 = (
                        _target_hidden_capped_topmass_visual_cosine(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=relative_vll_evidence,
                            support_positions=support_positions,
                            visual_start=visual_start,
                            visual_end=visual_end,
                            alpha=capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                    )
                if layer_prediction_h_prev is not None:
                    target_visual_hpre_cosine_relative_vll = _target_hidden_topk_visual_cosine(
                        target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                        support_output_states=layer_support_prev_states,
                        target_dist=relative_vll_evidence,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        top_k=atarget_visual_top_k,
                    )
                    target_visual_hpre_cosine16_relative_vll = _target_hidden_topk_visual_cosine(
                        target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                        support_output_states=layer_support_prev_states,
                        target_dist=relative_vll_evidence,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        top_k=COSINE16_TOP_K,
                    )
                    if compute_capped_topmass_085:
                        target_visual_hpre_cosine_relative_vll_capped_topmass_085 = (
                            _target_hidden_capped_topmass_visual_cosine(
                                target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                                support_output_states=layer_support_prev_states,
                                target_dist=relative_vll_evidence,
                                support_positions=support_positions,
                                visual_start=visual_start,
                                visual_end=visual_end,
                                alpha=capped_topmass_alpha,
                                min_k=capped_topmass_min_k,
                                max_k=capped_topmass_max_k,
                            )
                        )
            if visual_prompt_relative_vll_evidence is not None:
                target_visual_prompt_hidden_cosine_visual_prompt_relative_vll = _target_hidden_topk_support_cosine(
                    target_hidden=layer_prediction_hidden,
                    support_output_states=layer_support_output_states,
                    target_dist=visual_prompt_relative_vll_evidence,
                    top_k=atarget_visual_top_k,
                )
                target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll = (
                    _target_hidden_topk_support_cosine(
                        target_hidden=layer_prediction_hidden,
                        support_output_states=layer_support_output_states,
                        target_dist=visual_prompt_relative_vll_evidence,
                        top_k=COSINE16_TOP_K,
                    )
                )
                if compute_capped_topmass_085:
                    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 = (
                        _target_hidden_capped_topmass_support_cosine(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=visual_prompt_relative_vll_evidence,
                            alpha=capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                    )
                if layer_prediction_h_prev is not None:
                    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll = _target_hidden_topk_support_cosine(
                        target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                        support_output_states=layer_support_prev_states,
                        target_dist=visual_prompt_relative_vll_evidence,
                        top_k=atarget_visual_top_k,
                    )
                    target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll = (
                        _target_hidden_topk_support_cosine(
                            target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                            support_output_states=layer_support_prev_states,
                            target_dist=visual_prompt_relative_vll_evidence,
                            top_k=COSINE16_TOP_K,
                        )
                    )
                    if compute_capped_topmass_085:
                        target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085 = (
                            _target_hidden_capped_topmass_support_cosine(
                                target_hidden=layer_prediction_h_prev.to(layer_support_prev_states.device),
                                support_output_states=layer_support_prev_states,
                                target_dist=visual_prompt_relative_vll_evidence,
                                alpha=capped_topmass_alpha,
                                min_k=capped_topmass_min_k,
                                max_k=capped_topmass_max_k,
                            )
                        )
        prompt_conf_top3 = _scalar(prompt_confidence_top3[layer_idx])
        prompt_conf_max = _scalar(prompt_confidence_max[layer_idx])
        context_confidence = float(prompt_conf_top3 * target_visual_hidden_cosine)
        context_confidence_max_prompt = float(prompt_conf_max * target_visual_hidden_cosine)

        risk_per_layer.append(float(transport_risk))
        if transport_risk_topmass_085 is not None:
            risk_topmass_085_per_layer.append(float(transport_risk_topmass_085))
        if transport_risk_capped_topmass_085 is not None:
            risk_capped_topmass_085_per_layer.append(float(transport_risk_capped_topmass_085))
        if transport_risk_relative_vll is not None:
            risk_relative_vll_per_layer.append(float(transport_risk_relative_vll))
        if relative_vll_evidence_strength is not None:
            es_value = float(relative_vll_evidence_strength.detach())
            relative_vll_evidence_strength_per_layer.append(es_value)
            risk_for_r_es = (
                float(transport_risk_relative_vll_cost_geo)
                if transport_risk_relative_vll_cost_geo is not None
                else (
                    float(transport_risk_relative_vll)
                    if transport_risk_relative_vll is not None
                    else 0.0
                )
            )
            r_es_relative_vll_cost_geo_per_layer.append(
                float(risk_for_r_es - torch.log(relative_vll_evidence_strength.clamp_min(EPS)).item())
            )
        if divergence_relative_vll is not None:
            js_relative_vll_per_layer.append(float(divergence_relative_vll["js"]))
            kl_target_source_relative_vll_per_layer.append(
                float(divergence_relative_vll["kl_target_source"])
            )
            kl_source_target_relative_vll_per_layer.append(
                float(divergence_relative_vll["kl_source_target"])
            )
        if transport_risk_relative_vll_capped_topmass_085 is not None:
            risk_relative_vll_capped_topmass_085_per_layer.append(
                float(transport_risk_relative_vll_capped_topmass_085)
            )
        if transport_risk_visual_prompt_relative_vll is not None:
            risk_visual_prompt_relative_vll_per_layer.append(float(transport_risk_visual_prompt_relative_vll))
        if divergence_visual_prompt_relative_vll is not None:
            js_visual_prompt_relative_vll_per_layer.append(
                float(divergence_visual_prompt_relative_vll["js"])
            )
            kl_target_source_visual_prompt_relative_vll_per_layer.append(
                float(divergence_visual_prompt_relative_vll["kl_target_source"])
            )
            kl_source_target_visual_prompt_relative_vll_per_layer.append(
                float(divergence_visual_prompt_relative_vll["kl_source_target"])
            )
        if transport_risk_visual_prompt_relative_vll_capped_topmass_085 is not None:
            risk_visual_prompt_relative_vll_capped_topmass_085_per_layer.append(
                float(transport_risk_visual_prompt_relative_vll_capped_topmass_085)
            )
        prompt_last_cosine_per_layer.append(prompt_last_cosine)
        prompt_mean_cosine_per_layer.append(prompt_mean_cosine)
        target_visual_hidden_cosine_per_layer.append(float(target_visual_hidden_cosine))
        target_visual_prompt_hidden_cosine_per_layer.append(float(target_visual_prompt_hidden_cosine))
        if target_visual_hidden_cosine_capped_topmass_085 is not None:
            target_visual_hidden_cosine_capped_topmass_085_per_layer.append(
                float(target_visual_hidden_cosine_capped_topmass_085)
            )
        if target_visual_prompt_hidden_cosine_capped_topmass_085 is not None:
            target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer.append(
                float(target_visual_prompt_hidden_cosine_capped_topmass_085)
            )
        if target_visual_hidden_cosine_relative_vll is not None:
            target_visual_hidden_cosine_relative_vll_per_layer.append(
                float(target_visual_hidden_cosine_relative_vll)
            )
        if target_visual_hidden_cosine16_relative_vll is not None:
            target_visual_hidden_cosine16_relative_vll_per_layer.append(
                float(target_visual_hidden_cosine16_relative_vll)
            )
        if target_visual_hpre_cosine_relative_vll is not None:
            target_visual_hpre_cosine_relative_vll_per_layer.append(
                float(target_visual_hpre_cosine_relative_vll)
            )
        if target_visual_hpre_cosine16_relative_vll is not None:
            target_visual_hpre_cosine16_relative_vll_per_layer.append(
                float(target_visual_hpre_cosine16_relative_vll)
            )
        if target_visual_hidden_cosine_relative_vll_capped_topmass_085 is not None:
            target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_hidden_cosine_relative_vll_capped_topmass_085)
            )
        if target_visual_hpre_cosine_relative_vll_capped_topmass_085 is not None:
            target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_hpre_cosine_relative_vll_capped_topmass_085)
            )
        if target_visual_prompt_hidden_cosine_visual_prompt_relative_vll is not None:
            target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer.append(
                float(target_visual_prompt_hidden_cosine_visual_prompt_relative_vll)
            )
        if target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll is not None:
            target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer.append(
                float(target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll)
            )
        if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll is not None:
            target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer.append(
                float(target_visual_prompt_hpre_cosine_visual_prompt_relative_vll)
            )
        if target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll is not None:
            target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer.append(
                float(target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll)
            )
        if visual_prompt_relative_vll_target_visual_mass is not None:
            visual_prompt_relative_vll_target_visual_mass_per_layer.append(
                float(visual_prompt_relative_vll_target_visual_mass.detach())
            )
        if visual_prompt_relative_vll_target_prompt_mass is not None:
            visual_prompt_relative_vll_target_prompt_mass_per_layer.append(
                float(visual_prompt_relative_vll_target_prompt_mass.detach())
            )
        if visual_prompt_relative_vll_evidence_strength is not None:
            visual_prompt_relative_vll_evidence_strength_per_layer.append(
                float(visual_prompt_relative_vll_evidence_strength.detach())
            )
        if visual_prompt_relative_vll_evidence_visual_mass is not None:
            visual_prompt_relative_vll_evidence_visual_mass_per_layer.append(
                float(visual_prompt_relative_vll_evidence_visual_mass.detach())
            )
        if visual_prompt_relative_vll_evidence_prompt_mass is not None:
            visual_prompt_relative_vll_evidence_prompt_mass_per_layer.append(
                float(visual_prompt_relative_vll_evidence_prompt_mass.detach())
            )
        if visual_prompt_relative_vll_source_visual_mass is not None:
            visual_prompt_relative_vll_source_visual_mass_per_layer.append(
                float(visual_prompt_relative_vll_source_visual_mass.detach())
            )
        if visual_prompt_relative_vll_source_prompt_mass is not None:
            visual_prompt_relative_vll_source_prompt_mass_per_layer.append(
                float(visual_prompt_relative_vll_source_prompt_mass.detach())
            )
        if target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 is not None:
            target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085)
            )
        if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085 is not None:
            target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085)
            )
        prompt_confidence_top3_per_layer.append(float(prompt_conf_top3))
        prompt_confidence_max_per_layer.append(float(prompt_conf_max))
        context_confidence_per_layer.append(context_confidence)
        context_confidence_max_prompt_per_layer.append(context_confidence_max_prompt)
        if ffn_injection is not None:
            gate_value = float(ffn_gate_ratio) if ffn_gate_ratio is not None else 0.0
            risk_geo_value = (
                float(transport_risk_relative_vll_cost_geo)
                if transport_risk_relative_vll_cost_geo is not None
                else 0.0
            )
            fgr_value = float(gate_value * risk_geo_value)
            ffn_gate_ratio_per_layer.append(gate_value)
            ffn_fgr_per_layer.append(fgr_value)
            ffn_injection["ffn_gate_ratio"] = gate_value
            ffn_injection["ffn_fgr"] = fgr_value
        stats = {
            "layer": int(layer_idx + 1),
            "transport_risk": float(transport_risk),
            "prompt_last_cosine": prompt_last_cosine,
            "prompt_mean_cosine": prompt_mean_cosine,
            "prompt_logit_lens_top3_confidence": float(prompt_conf_top3),
            "prompt_logit_lens_max_confidence": float(prompt_conf_max),
            "atarget_top32_visual_cosine": float(target_visual_hidden_cosine),
            "target_hidden_top32_visual_cosine": float(target_visual_hidden_cosine),
            "target_hidden_top32_visual_prompt_cosine": float(target_visual_prompt_hidden_cosine),
            "context_confidence": context_confidence,
            "context_confidence_max_prompt": context_confidence_max_prompt,
            "support_size": int(len(support_positions)),
            "selected_support_size": int(support.numel()),
        }
        if target_visual_hidden_cosine_capped_topmass_085 is not None:
            stats["target_hidden_capped_topmass_085_visual_cosine"] = float(
                target_visual_hidden_cosine_capped_topmass_085
            )
        if target_visual_prompt_hidden_cosine_capped_topmass_085 is not None:
            stats["target_hidden_capped_topmass_085_visual_prompt_cosine"] = float(
                target_visual_prompt_hidden_cosine_capped_topmass_085
            )
        if transport_risk_topmass_085 is not None and topmass_support is not None:
            stats["transport_risk_topmass_085"] = float(transport_risk_topmass_085)
            stats["selected_support_size_topmass_085"] = int(topmass_support.numel())
        if transport_risk_capped_topmass_085 is not None and capped_topmass_support is not None:
            stats["transport_risk_capped_topmass_085"] = float(transport_risk_capped_topmass_085)
            stats["selected_support_size_capped_topmass_085"] = int(capped_topmass_support.numel())
        if transport_risk_relative_vll is not None and relative_vll_support is not None:
            stats["transport_risk_relative_vll"] = float(transport_risk_relative_vll)
            stats["selected_support_size_relative_vll"] = int(relative_vll_support.numel())
            stats["target_hidden_top32_visual_cosine_relative_vll"] = float(
                target_visual_hidden_cosine_relative_vll
            )
            if target_visual_hidden_cosine16_relative_vll is not None:
                stats["target_hidden_top16_visual_cosine_relative_vll"] = float(
                    target_visual_hidden_cosine16_relative_vll
                )
            if target_visual_hpre_cosine_relative_vll is not None:
                stats["target_hpre_top32_visual_cosine_relative_vll"] = float(
                    target_visual_hpre_cosine_relative_vll
                )
            if target_visual_hpre_cosine16_relative_vll is not None:
                stats["target_hpre_top16_visual_cosine_relative_vll"] = float(
                    target_visual_hpre_cosine16_relative_vll
                )
            if relative_vll_evidence_strength is not None:
                stats["relative_vll_evidence_strength"] = float(
                    relative_vll_evidence_strength.detach()
                )
                stats["r_es_relative_vll_cost_geo"] = float(
                    r_es_relative_vll_cost_geo_per_layer[-1]
                )
            if divergence_relative_vll is not None:
                stats["source_target_js_relative_vll"] = float(divergence_relative_vll["js"])
                stats["source_target_kl_target_source_relative_vll"] = float(
                    divergence_relative_vll["kl_target_source"]
                )
                stats["source_target_kl_source_target_relative_vll"] = float(
                    divergence_relative_vll["kl_source_target"]
                )
        if (
            transport_risk_relative_vll_capped_topmass_085 is not None
            and relative_vll_capped_support is not None
        ):
            stats["transport_risk_relative_vll_capped_topmass_085"] = float(
                transport_risk_relative_vll_capped_topmass_085
            )
            stats["selected_support_size_relative_vll_capped_topmass_085"] = int(
                relative_vll_capped_support.numel()
            )
            stats["target_hidden_capped_topmass_085_visual_cosine_relative_vll"] = float(
                target_visual_hidden_cosine_relative_vll_capped_topmass_085
            )
            if target_visual_hpre_cosine_relative_vll_capped_topmass_085 is not None:
                stats["target_hpre_capped_topmass_085_visual_cosine_relative_vll"] = float(
                    target_visual_hpre_cosine_relative_vll_capped_topmass_085
                )
        if relative_vll_stats is not None:
            stats.update(relative_vll_stats)
        if (
            transport_risk_visual_prompt_relative_vll is not None
            and visual_prompt_relative_vll_support is not None
        ):
            stats["transport_risk_visual_prompt_relative_vll"] = float(transport_risk_visual_prompt_relative_vll)
            stats["selected_support_size_visual_prompt_relative_vll"] = int(
                visual_prompt_relative_vll_support.numel()
            )
            stats["target_hidden_top32_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                target_visual_prompt_hidden_cosine_visual_prompt_relative_vll
            )
            if target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll is not None:
                stats["target_hidden_top16_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                    target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll
                )
            if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll is not None:
                stats["target_hpre_top32_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll
                )
            if target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll is not None:
                stats["target_hpre_top16_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                    target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll
                )
            stats["visual_prompt_relative_vll_target_visual_mass"] = float(
                visual_prompt_relative_vll_target_visual_mass.detach()
            )
            stats["visual_prompt_relative_vll_target_prompt_mass"] = float(
                visual_prompt_relative_vll_target_prompt_mass.detach()
            )
            if visual_prompt_relative_vll_evidence_strength is not None:
                stats["visual_prompt_relative_vll_evidence_strength"] = float(
                    visual_prompt_relative_vll_evidence_strength.detach()
                )
            if visual_prompt_relative_vll_evidence_visual_mass is not None:
                stats["visual_prompt_relative_vll_evidence_visual_mass"] = float(
                    visual_prompt_relative_vll_evidence_visual_mass.detach()
                )
            if visual_prompt_relative_vll_evidence_prompt_mass is not None:
                stats["visual_prompt_relative_vll_evidence_prompt_mass"] = float(
                    visual_prompt_relative_vll_evidence_prompt_mass.detach()
                )
            if visual_prompt_relative_vll_source_visual_mass is not None:
                stats["visual_prompt_relative_vll_source_visual_mass"] = float(
                    visual_prompt_relative_vll_source_visual_mass.detach()
                )
            if visual_prompt_relative_vll_source_prompt_mass is not None:
                stats["visual_prompt_relative_vll_source_prompt_mass"] = float(
                    visual_prompt_relative_vll_source_prompt_mass.detach()
                )
            if divergence_visual_prompt_relative_vll is not None:
                stats["source_target_js_visual_prompt_relative_vll"] = float(
                    divergence_visual_prompt_relative_vll["js"]
                )
                stats["source_target_kl_target_source_visual_prompt_relative_vll"] = float(
                    divergence_visual_prompt_relative_vll["kl_target_source"]
                )
                stats["source_target_kl_source_target_visual_prompt_relative_vll"] = float(
                    divergence_visual_prompt_relative_vll["kl_source_target"]
                )
        if (
            transport_risk_visual_prompt_relative_vll_capped_topmass_085 is not None
            and visual_prompt_relative_vll_capped_support is not None
        ):
            stats["transport_risk_visual_prompt_relative_vll_capped_topmass_085"] = float(
                transport_risk_visual_prompt_relative_vll_capped_topmass_085
            )
            stats["selected_support_size_visual_prompt_relative_vll_capped_topmass_085"] = int(
                visual_prompt_relative_vll_capped_support.numel()
            )
            stats["target_hidden_capped_topmass_085_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085
            )
            if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085 is not None:
                stats["target_hpre_capped_topmass_085_visual_prompt_cosine_visual_prompt_relative_vll"] = float(
                    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085
                )
        if visual_prompt_relative_vll_stats is not None:
            stats.update(visual_prompt_relative_vll_stats)
        if ffn_injection is not None:
            stats.update(ffn_injection)
        if delta_layer_stats:
            stats.update(delta_layer_stats)
        layer_stats.append(stats)

    if compute_cost_variants:
        cost_variant_series = _solve_cost_variant_problem_series(
            cost_variant_problem_series,
            ot_solver=ot_solver,
        )

    risk_tensor = torch.tensor(risk_per_layer, dtype=torch.float32)
    final_score = _baseline_excess_score(
        risk_tensor,
        baseline_layers=baseline_layers,
        risk_start_layer=risk_start_layer,
        alpha=alpha,
    )
    target_visual_tensor = torch.tensor(target_visual_hidden_cosine_per_layer, dtype=torch.float32)
    target_visual_prompt_tensor = torch.tensor(
        target_visual_prompt_hidden_cosine_per_layer,
        dtype=torch.float32,
    )
    target_visual_capped_tensor = torch.tensor(
        target_visual_hidden_cosine_capped_topmass_085_per_layer,
        dtype=torch.float32,
    )
    target_visual_prompt_capped_tensor = torch.tensor(
        target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer,
        dtype=torch.float32,
    )

    result = {
        "dgst_t_score": float(final_score),
        "dgst_t_source_distribution_mode": source_distribution,
        "dgst_t_relative_cost_mode": relative_cost,
        "dgst_t_relative_cost_modes": list(emit_relative_cost_modes),
        "dgst_t_relative_cost_state_modes": [
            state_slug for _state_mode, _update_lambda, state_slug in relative_cost_state_specs
        ],
        "dgst_t_relative_barrier_lambda": float(relative_barrier_lambda),
        "dgst_t_relative_barrier_margin": float(relative_barrier_margin),
        "dgst_t_relative_barrier_max": float(relative_barrier_max),
        "dgst_t_per_layer": risk_tensor,
        "dgst_t_transport_risk_per_layer": risk_tensor,
        "dgst_t_prompt_last_cosine_per_layer": torch.tensor(prompt_last_cosine_per_layer, dtype=torch.float32),
        "dgst_t_prompt_mean_cosine_per_layer": torch.tensor(prompt_mean_cosine_per_layer, dtype=torch.float32),
        "dgst_t_atarget_visual_cosine_per_layer": target_visual_tensor,
        "dgst_t_target_visual_hidden_cosine_per_layer": target_visual_tensor,
        "dgst_t_target_visual_prompt_hidden_cosine_per_layer": target_visual_prompt_tensor,
        "dgst_t_target_visual_hidden_cosine_capped_topmass_085_per_layer": target_visual_capped_tensor,
        "dgst_t_target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer": target_visual_prompt_capped_tensor,
        "dgst_t_prompt_confidence_top3_per_layer": torch.tensor(prompt_confidence_top3_per_layer, dtype=torch.float32),
        "dgst_t_prompt_confidence_max_per_layer": torch.tensor(prompt_confidence_max_per_layer, dtype=torch.float32),
        "dgst_t_context_confidence_per_layer": torch.tensor(context_confidence_per_layer, dtype=torch.float32),
        "dgst_t_context_confidence_max_prompt_per_layer": torch.tensor(
            context_confidence_max_prompt_per_layer,
            dtype=torch.float32,
        ),
        "dgst_t_layer_stats": layer_stats,
        "dgst_t_feature_vector": _feature_vector(
            risk_per_layer,
            prompt_last_cosine_per_layer,
            prompt_mean_cosine_per_layer,
            context_confidence_per_layer,
        ),
    }
    if compute_ffn_injection_features:
        result["dgst_t_ffn_attn_dominance_per_layer"] = torch.tensor(
            ffn_attn_dominance_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_evidence_orthogonal_dose_per_layer"] = torch.tensor(
            ffn_evidence_orthogonal_dose_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_logit_lift_per_layer"] = torch.tensor(
            ffn_logit_lift_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_eif_fraction_svd_per_layer"] = torch.tensor(
            ffn_eif_fraction_svd_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_eif_dose_svd_per_layer"] = torch.tensor(
            ffn_eif_dose_svd_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_eif_fraction_pca_per_layer"] = torch.tensor(
            ffn_eif_fraction_pca_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_eif_dose_pca_per_layer"] = torch.tensor(
            ffn_eif_dose_pca_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_gate_ratio_per_layer"] = torch.tensor(
            ffn_gate_ratio_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_fgr_per_layer"] = torch.tensor(
            ffn_fgr_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_ffn_evidence_top_k"] = int(ffn_injection_evidence_top_k)
        result["dgst_t_ffn_evidence_rank"] = int(ffn_injection_evidence_rank)
        result["dgst_t_ffn_injection_eps"] = float(ffn_injection_eps)
    if compute_relative_vll:
        risk_relative_tensor = torch.tensor(risk_relative_vll_per_layer, dtype=torch.float32)
        result["dgst_t_score_relative_vll"] = float(
            _baseline_excess_score(
                risk_relative_tensor,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
            )
        )
        result["dgst_t_transport_risk_relative_vll_per_layer"] = risk_relative_tensor
        result["dgst_t_relative_vll_evidence_strength_per_layer"] = torch.tensor(
            relative_vll_evidence_strength_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_r_es_relative_vll_cost_geo_per_layer"] = torch.tensor(
            r_es_relative_vll_cost_geo_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_target_visual_hidden_cosine_relative_vll_per_layer"] = torch.tensor(
            target_visual_hidden_cosine_relative_vll_per_layer,
            dtype=torch.float32,
        )
        result["dgst_t_target_visual_hidden_cosine16_relative_vll_per_layer"] = torch.tensor(
            target_visual_hidden_cosine16_relative_vll_per_layer,
            dtype=torch.float32,
        )
        if target_visual_hpre_cosine_relative_vll_per_layer:
            result["dgst_t_target_visual_hpre_cosine_relative_vll_per_layer"] = torch.tensor(
                target_visual_hpre_cosine_relative_vll_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_target_visual_hpre_cosine16_relative_vll_per_layer"] = torch.tensor(
                target_visual_hpre_cosine16_relative_vll_per_layer,
                dtype=torch.float32,
            )
        if vv_attention_dist_per_layer:
            result["dgst_t_vv_support_positions"] = [
                int(position) for position in support_positions
            ]
            result["dgst_t_vv_attention_dist_per_layer"] = torch.stack(
                vv_attention_dist_per_layer
            ).to(dtype=torch.float32)
            result["dgst_t_vv_support_attention_per_layer"] = torch.stack(
                vv_support_attention_per_layer
            ).to(dtype=torch.float32)
            result["dgst_t_vv_source_dist_per_layer"] = torch.stack(
                vv_source_dist_per_layer
            ).to(dtype=torch.float32)
            for source_mode, values in vv_source_variant_dist_per_layer.items():
                if values:
                    result[f"dgst_t_vv_source_{source_mode}_dist_per_layer"] = (
                        torch.stack(values).to(dtype=torch.float32)
                    )
            result["dgst_t_vv_semantic_gate_per_layer"] = torch.stack(
                vv_semantic_gate_per_layer
            ).to(dtype=torch.float32)
            if vv_gauss_semantic_gate_per_layer:
                result["dgst_t_vv_gauss_semantic_gate_per_layer"] = torch.stack(
                    vv_gauss_semantic_gate_per_layer
                ).to(dtype=torch.float32)
            result["dgst_t_vv_raw_evidence_strength_per_layer"] = torch.tensor(
                vv_raw_evidence_strength_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_vv_source_entropy_per_layer"] = torch.tensor(
                vv_source_entropy_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_vv_target_entropy_per_layer"] = torch.tensor(
                vv_target_entropy_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_vv_evidence_entropy_per_layer"] = torch.tensor(
                vv_evidence_entropy_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_vv_source_topk_entropy_per_layer"] = torch.tensor(
                vv_source_topk_entropy_per_layer,
                dtype=torch.float32,
            )
        if compute_cost_variants:
            for key, values in cost_variant_series.items():
                if len(values) != layer_count:
                    raise ValueError(
                        f"Incomplete cost variant {key}: {len(values)} of {layer_count} layers."
                    )
                result[f"dgst_t_{key}_per_layer"] = torch.tensor(
                    values,
                    dtype=torch.float32,
                )
            result["dgst_t_cost_variant_mad_scale"] = float(GAUSSIAN_MAD_SCALE)
            result["dgst_t_cost_variant_transport_top_k"] = int(transport_top_k)
            result["dgst_t_cost_variant_hprecosine_top_k"] = int(atarget_visual_top_k)
        if js_relative_vll_per_layer:
            result["dgst_t_js_relative_vll_per_layer"] = torch.tensor(
                js_relative_vll_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_kl_target_source_relative_vll_per_layer"] = torch.tensor(
                kl_target_source_relative_vll_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_kl_source_target_relative_vll_per_layer"] = torch.tensor(
                kl_source_target_relative_vll_per_layer,
                dtype=torch.float32,
            )
        if compute_capped_topmass_085:
            result["dgst_t_transport_risk_relative_vll_capped_topmass_085_per_layer"] = torch.tensor(
                risk_relative_vll_capped_topmass_085_per_layer,
                dtype=torch.float32,
            )
            result[
                "dgst_t_target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer"
            ] = torch.tensor(
                target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer,
                dtype=torch.float32,
            )
            if target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer:
                result[
                    "dgst_t_target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer"
                ] = torch.tensor(
                    target_visual_hpre_cosine_relative_vll_capped_topmass_085_per_layer,
                    dtype=torch.float32,
                )
        if risk_visual_prompt_relative_vll_per_layer:
            risk_visual_prompt_relative_tensor = torch.tensor(
                risk_visual_prompt_relative_vll_per_layer,
                dtype=torch.float32,
            )
            result["dgst_t_score_visual_prompt_relative_vll"] = float(
                _baseline_excess_score(
                    risk_visual_prompt_relative_tensor,
                    baseline_layers=baseline_layers,
                    risk_start_layer=risk_start_layer,
                    alpha=alpha,
                )
            )
            result["dgst_t_transport_risk_visual_prompt_relative_vll_per_layer"] = (
                risk_visual_prompt_relative_tensor
            )
            result["dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer"] = (
                torch.tensor(
                    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
            )
            result[
                "dgst_t_target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer"
            ] = torch.tensor(
                target_visual_prompt_hidden_cosine16_visual_prompt_relative_vll_per_layer,
                dtype=torch.float32,
            )
            if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer:
                result[
                    "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer"
                ] = torch.tensor(
                    target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
                result[
                    "dgst_t_target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer"
                ] = torch.tensor(
                    target_visual_prompt_hpre_cosine16_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
            result["dgst_t_visual_prompt_relative_vll_target_visual_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_target_visual_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_evidence_visual_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_evidence_visual_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_evidence_prompt_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_evidence_prompt_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_evidence_strength_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_evidence_strength_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_source_visual_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_source_visual_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_source_prompt_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_m_p_per_layer"] = result[
                "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer"
            ]
            if vp_attention_dist_per_layer:
                result["dgst_t_vp_support_positions"] = [
                    int(position) for position in support_positions
                ]
                result["dgst_t_vp_attention_dist_per_layer"] = torch.stack(
                    vp_attention_dist_per_layer
                ).to(dtype=torch.float32)
                result["dgst_t_vp_support_attention_per_layer"] = torch.stack(
                    vp_support_attention_per_layer
                ).to(dtype=torch.float32)
                result["dgst_t_vp_source_dist_per_layer"] = torch.stack(
                    vp_source_dist_per_layer
                ).to(dtype=torch.float32)
                for source_mode, values in vp_source_variant_dist_per_layer.items():
                    if values:
                        result[f"dgst_t_vp_source_{source_mode}_dist_per_layer"] = (
                            torch.stack(values).to(dtype=torch.float32)
                        )
                result["dgst_t_vp_semantic_gate_per_layer"] = torch.stack(
                    vp_semantic_gate_per_layer
                ).to(dtype=torch.float32)
                result["dgst_t_vp_raw_evidence_visual_mass_per_layer"] = torch.tensor(
                    vp_raw_evidence_visual_mass_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_raw_evidence_prompt_mass_per_layer"] = torch.tensor(
                    vp_raw_evidence_prompt_mass_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_raw_evidence_strength_per_layer"] = torch.tensor(
                    vp_raw_evidence_strength_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_source_entropy_per_layer"] = torch.tensor(
                    vp_source_entropy_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_target_entropy_per_layer"] = torch.tensor(
                    vp_target_entropy_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_evidence_entropy_per_layer"] = torch.tensor(
                    vp_evidence_entropy_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_vp_source_topk_entropy_per_layer"] = torch.tensor(
                    vp_source_topk_entropy_per_layer,
                    dtype=torch.float32,
                )
            result["dgst_t_visual_prompt_relative_vll_target_prompt_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_target_prompt_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            if js_visual_prompt_relative_vll_per_layer:
                result["dgst_t_js_visual_prompt_relative_vll_per_layer"] = torch.tensor(
                    js_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_kl_target_source_visual_prompt_relative_vll_per_layer"] = torch.tensor(
                    kl_target_source_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
                result["dgst_t_kl_source_target_visual_prompt_relative_vll_per_layer"] = torch.tensor(
                    kl_source_target_visual_prompt_relative_vll_per_layer,
                    dtype=torch.float32,
                )
            if compute_capped_topmass_085:
                result[
                    "dgst_t_transport_risk_visual_prompt_relative_vll_capped_topmass_085_per_layer"
                ] = torch.tensor(
                    risk_visual_prompt_relative_vll_capped_topmass_085_per_layer,
                    dtype=torch.float32,
                )
                result[
                    "dgst_t_target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
                ] = torch.tensor(
                    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer,
                    dtype=torch.float32,
                )
                if target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer:
                    result[
                        "dgst_t_target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer"
                    ] = torch.tensor(
                        target_visual_prompt_hpre_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer,
                        dtype=torch.float32,
                    )
        for mode, values in relative_cost_series.items():
            slug = _relative_cost_slug(mode)
            risk_cost_tensor = torch.tensor(values["risk"], dtype=torch.float32)
            if risk_cost_tensor.numel():
                result[f"dgst_t_score_relative_vll_cost_{slug}"] = float(
                    _baseline_excess_score(
                        risk_cost_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[f"dgst_t_transport_risk_relative_vll_cost_{slug}_per_layer"] = (
                    risk_cost_tensor
                )
            risk_cost_cap_tensor = torch.tensor(values["risk_cap"], dtype=torch.float32)
            if risk_cost_cap_tensor.numel():
                result[f"dgst_t_score_relative_vll_cost_{slug}_capped_topmass_085"] = float(
                    _baseline_excess_score(
                        risk_cost_cap_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_relative_vll_cost_{slug}_capped_topmass_085_per_layer"
                ] = risk_cost_cap_tensor
            vp_risk_cost_tensor = torch.tensor(values["vp_risk"], dtype=torch.float32)
            if vp_risk_cost_tensor.numel():
                result[f"dgst_t_score_visual_prompt_relative_vll_cost_{slug}"] = float(
                    _baseline_excess_score(
                        vp_risk_cost_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_visual_prompt_relative_vll_cost_{slug}_per_layer"
                ] = vp_risk_cost_tensor
            vp_risk_cost_cap_tensor = torch.tensor(values["vp_risk_cap"], dtype=torch.float32)
            if vp_risk_cost_cap_tensor.numel():
                result[
                    f"dgst_t_score_visual_prompt_relative_vll_cost_{slug}_capped_topmass_085"
                ] = float(
                    _baseline_excess_score(
                        vp_risk_cost_cap_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_visual_prompt_relative_vll_cost_{slug}_capped_topmass_085_per_layer"
                ] = vp_risk_cost_cap_tensor
        for (mode, state_slug), values in relative_cost_state_series.items():
            cost_slug = _relative_cost_slug(mode)
            combined_slug = f"{cost_slug}_{state_slug}"
            risk_state_tensor = torch.tensor(values["risk"], dtype=torch.float32)
            if risk_state_tensor.numel():
                result[f"dgst_t_score_relative_vll_cost_{combined_slug}"] = float(
                    _baseline_excess_score(
                        risk_state_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_relative_vll_cost_{combined_slug}_per_layer"
                ] = risk_state_tensor
            vp_risk_state_tensor = torch.tensor(values["vp_risk"], dtype=torch.float32)
            if vp_risk_state_tensor.numel():
                result[f"dgst_t_score_visual_prompt_relative_vll_cost_{combined_slug}"] = float(
                    _baseline_excess_score(
                        vp_risk_state_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_visual_prompt_relative_vll_cost_{combined_slug}_per_layer"
                ] = vp_risk_state_tensor
        for (target_slug, source_mode), values in source_variant_series.items():
            risk_source_tensor = torch.tensor(values["risk"], dtype=torch.float32)
            if risk_source_tensor.numel():
                result[f"dgst_t_score_{target_slug}_source_{source_mode}"] = float(
                    _baseline_excess_score(
                        risk_source_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_{target_slug}_source_{source_mode}_per_layer"
                ] = risk_source_tensor
            risk_source_cap_tensor = torch.tensor(values["risk_cap"], dtype=torch.float32)
            if risk_source_cap_tensor.numel():
                result[
                    f"dgst_t_score_{target_slug}_source_{source_mode}_capped_topmass_085"
                ] = float(
                    _baseline_excess_score(
                        risk_source_cap_tensor,
                        baseline_layers=baseline_layers,
                        risk_start_layer=risk_start_layer,
                        alpha=alpha,
                    )
                )
                result[
                    f"dgst_t_transport_risk_{target_slug}_source_{source_mode}_capped_topmass_085_per_layer"
                ] = risk_source_cap_tensor
        for stem, values in topk_region_series.items():
            for name, series_values in values.items():
                tensor = torch.tensor(series_values, dtype=torch.float32)
                if tensor.numel():
                    result[f"{stem}_{name}_per_layer"] = tensor
        for stem, values in capped_support_series.items():
            result[f"{stem}_capped_topmass_085_support_indices_per_layer"] = values[
                "indices"
            ]
            result[f"{stem}_capped_topmass_085_support_positions_per_layer"] = values[
                "positions"
            ]
    if compute_topmass_085:
        result["dgst_t_transport_risk_topmass_085_per_layer"] = torch.tensor(
            risk_topmass_085_per_layer,
            dtype=torch.float32,
        )
    if compute_capped_topmass_085:
        result["dgst_t_transport_risk_capped_topmass_085_per_layer"] = torch.tensor(
            risk_capped_topmass_085_per_layer,
            dtype=torch.float32,
        )
    for series_key, values in delta_series.items():
        risk_delta_tensor = torch.tensor(values["risk"], dtype=torch.float32)
        result[f"dgst_t_score_{series_key}"] = float(
            _baseline_excess_score(
                risk_delta_tensor,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
            )
        )
        result[f"dgst_t_risk_{series_key}_per_layer"] = risk_delta_tensor
        result[f"dgst_t_cos_{series_key}_per_layer"] = torch.tensor(
            values["cos"],
            dtype=torch.float32,
        )
        if compute_capped_topmass_085:
            result[f"dgst_t_risk_{series_key}_cap085_per_layer"] = torch.tensor(
                values["risk_cap"],
                dtype=torch.float32,
            )
            result[f"dgst_t_cos_{series_key}_cap085_per_layer"] = torch.tensor(
                values["cos_cap"],
                dtype=torch.float32,
            )
    return result


def _compute_visual_scope_result_from_parts(
    *,
    source_ffn_states: Sequence[torch.Tensor],
    source_attn_states: Sequence[torch.Tensor] | None,
    prediction_hidden_states: Sequence[torch.Tensor],
    support_h_prev_states: Sequence[torch.Tensor],
    support_h_mid_states: Sequence[torch.Tensor],
    support_output_states: Sequence[torch.Tensor],
    support_attentions: Sequence[torch.Tensor],
    semantic_probs: Sequence[torch.Tensor],
    relative_vll_logits: Sequence[torch.Tensor] | None,
    prompt_last_hidden_states: Sequence[torch.Tensor],
    prompt_mean_hidden_states: Sequence[torch.Tensor],
    prompt_confidence_top3: Sequence[torch.Tensor],
    prompt_confidence_max: Sequence[torch.Tensor],
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    tau: float,
    source_distribution_mode: str,
    transport_top_k: int,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    baseline_layers: int,
    risk_start_layer: int,
    alpha: float,
    ot_solver: str,
    atarget_visual_top_k: int,
    topmass_alpha: float,
    capped_topmass_alpha: float,
    capped_topmass_min_k: int,
    capped_topmass_max_k: int,
    compute_topmass_085: bool,
    compute_capped_topmass_085: bool,
    target_gate_mode: str,
    relative_vll_mad_epsilon: float,
    relative_cost_mode: str,
    relative_cost_modes: Sequence[str] | str | None,
    relative_cost_state_modes: Sequence[str] | str | None,
    relative_cost_update_lambdas: Sequence[float] | float | None,
    relative_barrier_lambda: float,
    relative_barrier_margin: float,
    relative_barrier_max: float,
    source_modes: Sequence[str] | None,
    target_attention_gammas: Sequence[float] | None,
    target_attention_epsilon: float,
    target_unembedding: torch.Tensor | None,
) -> dict[str, Any]:
    visual_indices = [
        offset
        for offset, position in enumerate(support_positions)
        if int(visual_start) <= int(position) < int(visual_end)
    ]
    if not visual_indices or len(visual_indices) == len(support_positions):
        return {}

    return _compute_dgst_t_from_parts(
        source_ffn_states=source_ffn_states,
        source_attn_states=source_attn_states,
        prediction_hidden_states=prediction_hidden_states,
        support_h_prev_states=_slice_support_layers(support_h_prev_states, visual_indices),
        support_h_mid_states=_slice_support_layers(support_h_mid_states, visual_indices),
        support_output_states=_slice_support_layers(support_output_states, visual_indices),
        support_attentions=_slice_support_layers(support_attentions, visual_indices),
        semantic_probs=_slice_support_layers(semantic_probs, visual_indices),
        relative_vll_logits=(
            None
            if relative_vll_logits is None
            else _slice_support_layers(relative_vll_logits, visual_indices)
        ),
        prompt_last_hidden_states=prompt_last_hidden_states,
        prompt_mean_hidden_states=prompt_mean_hidden_states,
        prompt_confidence_top3=prompt_confidence_top3,
        prompt_confidence_max=prompt_confidence_max,
        support_positions=[int(support_positions[index]) for index in visual_indices],
        visual_start=int(visual_start),
        visual_end=int(visual_end),
        tau=tau,
        source_distribution_mode=source_distribution_mode,
        transport_top_k=transport_top_k,
        cost_mode=cost_mode,
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
        baseline_layers=baseline_layers,
        risk_start_layer=risk_start_layer,
        alpha=alpha,
        ot_solver=ot_solver,
        atarget_visual_top_k=atarget_visual_top_k,
        topmass_alpha=topmass_alpha,
        capped_topmass_alpha=capped_topmass_alpha,
        capped_topmass_min_k=capped_topmass_min_k,
        capped_topmass_max_k=capped_topmass_max_k,
        compute_topmass_085=compute_topmass_085,
        compute_capped_topmass_085=compute_capped_topmass_085,
        target_gate_mode=target_gate_mode,
        relative_vll_mad_epsilon=relative_vll_mad_epsilon,
        relative_cost_mode=relative_cost_mode,
        relative_cost_modes=relative_cost_modes,
        relative_cost_state_modes=relative_cost_state_modes,
        relative_cost_update_lambdas=relative_cost_update_lambdas,
        relative_barrier_lambda=relative_barrier_lambda,
        relative_barrier_margin=relative_barrier_margin,
        relative_barrier_max=relative_barrier_max,
        source_modes=source_modes,
        target_attention_gammas=target_attention_gammas,
        target_attention_epsilon=target_attention_epsilon,
        target_unembedding=target_unembedding,
        compute_ffn_injection_features=False,
    )


def _slice_support_layers(
    values: Sequence[torch.Tensor],
    support_indices: Sequence[int],
) -> list[torch.Tensor]:
    sliced = []
    for value in values:
        index = torch.tensor(support_indices, dtype=torch.long, device=value.device)
        sliced.append(value.index_select(0, index))
    return sliced


def _merge_visual_scope_relative_fields(
    result: dict[str, Any],
    visual_result: dict[str, Any],
) -> None:
    for key, value in visual_result.items():
        if (
            key.startswith("dgst_t_score_relative_vll")
            or key.startswith("dgst_t_transport_risk_relative_vll")
            or key.startswith("dgst_t_target_visual_hidden_cosine_relative_vll")
            or key.startswith("dgst_t_target_visual_hidden_cosine16_relative_vll")
            or key.startswith("dgst_t_target_visual_hpre_cosine_relative_vll")
            or key.startswith("dgst_t_target_visual_hpre_cosine16_relative_vll")
            or key.startswith("dgst_t_relative_vll_evidence_strength")
            or key.startswith("dgst_t_r_es_relative_vll")
            or key.startswith("dgst_t_vv_")
            or key.startswith("dgst_t_js_relative_vll")
            or key.startswith("dgst_t_kl_target_source_relative_vll")
            or key.startswith("dgst_t_kl_source_target_relative_vll")
        ):
            result[key] = value
    if visual_result:
        result["dgst_t_dual_scope_vv_source_target"] = "visual"
        result["dgst_t_dual_scope_vp_source_target"] = "visual_prompt"


def _attach_c_vp_feature(
    result: dict[str, Any],
    *,
    baseline_layers: int,
    risk_start_layer: int,
    alpha: float,
) -> None:
    risk_values = result.get("dgst_t_transport_risk_relative_vll_cost_geo_per_layer")
    if risk_values is None:
        risk_values = result.get("dgst_t_transport_risk_relative_vll_per_layer")
    prompt_mass_values = result.get("dgst_t_m_p_per_layer")
    if prompt_mass_values is None:
        prompt_mass_values = result.get(
            "dgst_t_visual_prompt_relative_vll_source_prompt_mass_per_layer"
        )
    if risk_values is None or prompt_mass_values is None:
        return

    risk_tensor = torch.as_tensor(risk_values, dtype=torch.float32)
    prompt_mass_tensor = torch.as_tensor(prompt_mass_values, dtype=torch.float32)
    if risk_tensor.numel() == 0 or risk_tensor.shape != prompt_mass_tensor.shape:
        return
    c_vp_tensor = risk_tensor + prompt_mass_tensor
    result["dgst_t_c_vp_relative_vll_cost_geo_per_layer"] = c_vp_tensor
    result["dgst_t_score_c_vp_relative_vll_cost_geo"] = float(
        _baseline_excess_score(
            c_vp_tensor,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
        )
    )


def _layer_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value[index] for index in range(int(value.shape[0]))]
    return list(value)


def _optional_layer_tensors(value: Any) -> list[torch.Tensor] | None:
    if value is None:
        return None
    return _layer_tensors(value)


def _normalize_target_gate_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {
        "four_gate",
        "four_gates",
        "four_gate_vv",
        "four_branch",
        "selected_four_gate",
    }:
        return "four_gate"
    if mode in {"legacy", "legacy_prob", "prob", "softmax_prob"}:
        return "legacy_prob"
    if mode in {"relative_vll", "relative", "relative_logit"}:
        return "relative_vll"
    if mode == "dual":
        return "dual"
    if mode in {"cost_variants", "costvariant", "coco500_costvariant"}:
        return "cost_variants"
    if mode in {
        "gate_comparison",
        "gate_compare",
        "softmax_relative_vll_legacy_compare",
        "coco100_gate_comparison",
    }:
        return "gate_comparison"
    raise ValueError(
        "DGST-T target_gate_mode must be 'legacy_prob', 'relative_vll', "
        "'dual', 'cost_variants', 'gate_comparison', or 'four_gate'."
    )


def _normalize_relative_cost_mode(value: str | None, *, legacy_cost_mode: str) -> str:
    if value is None:
        return _normalize_legacy_cost_mode(legacy_cost_mode)
    mode = str(value).strip().lower()
    if mode in {"", "legacy", "inherit", "cost_mode"}:
        return _normalize_legacy_cost_mode(legacy_cost_mode)
    if mode in {"geo", "pure_geo", "geometric", "pure_geometric"}:
        return "geo"
    if mode in {
        "target_barrier_geo",
        "target_barrier",
        "target_relative_barrier",
        "tbar",
    }:
        return "target_barrier_geo"
    if mode in {
        "symmetric_barrier_geo",
        "symmetric_barrier",
        "symmetric_relative_barrier",
        "sbar",
    }:
        return "symmetric_barrier_geo"
    if mode in {
        "target_additive_barrier_geo",
        "target_additive_barrier",
        "target_additive",
        "tadd",
    }:
        return "target_additive_barrier_geo"
    if mode in {
        "source_additive_barrier_geo",
        "source_additive_barrier",
        "source_additive",
        "sadd",
    }:
        return "source_additive_barrier_geo"
    if mode in {
        "two_end_additive_barrier_geo",
        "two_end_additive_barrier",
        "two_end_additive",
        "additive_barrier_geo",
        "additive_barrier",
        "tsadd",
    }:
        return "two_end_additive_barrier_geo"
    if mode in {
        "semantic_match_geo",
        "semantic_match",
        "qmatch",
        "qmatch_geo",
        "qadd",
        "qadd_geo",
        "sqrt_q_geo",
        "sqrtq_geo",
    }:
        return "semantic_match_geo"
    if mode in {"direct", "decomposed"}:
        return _normalize_legacy_cost_mode(mode)
    raise ValueError(
        "relative_cost_mode must be one of: geo, target_barrier_geo, "
        "symmetric_barrier_geo, target_additive_barrier_geo, "
        "source_additive_barrier_geo, two_end_additive_barrier_geo, "
        "semantic_match_geo, direct, decomposed, or inherit."
    )


def _normalize_relative_cost_modes(
    value: Sequence[str] | str | None,
    *,
    primary_cost_mode: str,
    legacy_cost_mode: str,
) -> list[str]:
    if value is None:
        return [str(primary_cost_mode)]
    raw_modes = [value] if isinstance(value, str) else list(value)
    modes = []
    for raw in raw_modes:
        mode = _normalize_relative_cost_mode(raw, legacy_cost_mode=legacy_cost_mode)
        if mode not in modes:
            modes.append(mode)
    if primary_cost_mode not in modes:
        modes.insert(0, str(primary_cost_mode))
    return modes or [str(primary_cost_mode)]


def _relative_cost_slug(mode: str) -> str:
    name = _normalize_relative_cost_mode(mode, legacy_cost_mode="decomposed")
    if name == "target_barrier_geo":
        return "tbar"
    if name == "symmetric_barrier_geo":
        return "sbar"
    if name == "target_additive_barrier_geo":
        return "tadd"
    if name == "source_additive_barrier_geo":
        return "sadd"
    if name == "two_end_additive_barrier_geo":
        return "tsadd"
    if name == "semantic_match_geo":
        return "qmatch"
    return name


def _normalize_relative_cost_state_modes(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return ["mid"]
    raw_modes = [value] if isinstance(value, str) else list(value)
    modes = []
    for raw in raw_modes:
        mode = str(raw).strip().lower().replace("-", "_")
        if mode in {"", "mid", "h_mid", "mid_mid"}:
            canonical = "mid"
        elif mode in {"out", "h_out", "output", "out_out", "post_ffn"}:
            canonical = "out"
        elif mode in {"avg", "average", "mid_out_avg", "mid_out_average"}:
            canonical = "avg"
        elif mode in {
            "state_update",
            "stateupd",
            "state_upd",
            "state+upd",
            "state_plus_update",
            "update",
        }:
            canonical = "state_update"
        else:
            raise ValueError(
                "relative_cost_state_modes entries must be one of: "
                "mid, out, avg, state_update."
            )
        if canonical not in modes:
            modes.append(canonical)
    return modes or ["mid"]


def _normalize_relative_cost_update_lambdas(
    value: Sequence[float] | float | None,
) -> list[float]:
    raw_values = [0.1] if value is None else ([value] if isinstance(value, (float, int)) else list(value))
    lambdas = []
    for raw in raw_values:
        lambda_u = float(raw)
        if lambda_u < 0.0:
            raise ValueError("relative_cost_update_lambdas must be non-negative.")
        if not any(abs(lambda_u - existing) < 1e-12 for existing in lambdas):
            lambdas.append(lambda_u)
    return lambdas or [0.1]


def _relative_cost_state_specs(
    modes_value: Sequence[str] | str | None,
    lambdas_value: Sequence[float] | float | None,
) -> list[tuple[str, float | None, str]]:
    modes = _normalize_relative_cost_state_modes(modes_value)
    lambdas = _normalize_relative_cost_update_lambdas(lambdas_value)
    specs: list[tuple[str, float | None, str]] = []
    for mode in modes:
        if mode == "state_update":
            for lambda_u in lambdas:
                specs.append((mode, float(lambda_u), f"stateupd_{_lambda_slug(lambda_u)}"))
        else:
            specs.append((mode, None, mode))
    return specs


def _lambda_slug(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m")
    if "." in text:
        whole, frac = text.split(".", 1)
        frac = frac.rstrip("0")
        return f"lu{whole}{frac}" if whole != "0" else f"lu0{frac}"
    return f"lu{text}"


def _normalize_legacy_cost_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in {"direct", "decomposed"}:
        return mode
    raise ValueError("DGST-T cost_mode must be 'direct' or 'decomposed'.")


def _normalize_source_modes(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return ["legacy_ffn"]
    raw_modes = [value] if isinstance(value, str) else list(value)
    modes = []
    for raw in raw_modes:
        mode = str(raw).strip().lower()
        if mode in {"legacy", "legacy_ffn", "ffn", "o_ffn"}:
            canonical = "legacy_ffn"
        elif mode in {"delta", "delta_src", "source_delta", "source_delta_visual"}:
            canonical = "delta_src"
        elif mode in {
            "hmid_proj",
            "h_mid_proj",
            "mid_proj",
            "hmid_projection",
            "h_mid_projection",
        }:
            canonical = "hmid_proj"
        elif mode in {"hprev_cos", "h_prev_cos", "prev_cos", "hminus1_cos"}:
            canonical = "hprev_cos"
        elif mode in {
            "hprev_proj",
            "h_prev_proj",
            "prev_proj",
            "hminus1_proj",
            "hprev_projection",
            "h_prev_projection",
        }:
            canonical = "hprev_proj"
        else:
            raise ValueError(
                "DGST-T source_modes entries must be one of: "
                "legacy_ffn, delta_src, hmid_proj, hprev_cos, hprev_proj."
            )
        if canonical not in modes:
            modes.append(canonical)
    return modes or ["legacy_ffn"]


def _normalize_source_distribution_mode(value: str | None) -> str:
    mode = str(value or "softmax").strip().lower().replace("-", "_")
    if mode in {"softmax", "legacy", "legacy_softmax"}:
        return "softmax"
    if mode in {
        "relu_norm",
        "relu",
        "no_softmax",
        "nosoftmax",
        "linear_relu",
        "positive_norm",
        "positive",
    }:
        return "relu_norm"
    raise ValueError("DGST-T source_distribution_mode must be 'softmax' or 'relu_norm'.")


def _normalize_target_attention_gammas(value: Sequence[float] | float | None) -> list[float]:
    if value is None:
        raw_values = [1.0]
    elif isinstance(value, (float, int)):
        raw_values = [float(value)]
    else:
        raw_values = [float(item) for item in value]
    gammas = []
    for raw in raw_values:
        gamma = float(raw)
        if gamma < 0.0:
            raise ValueError("DGST-T target_attention_gammas must be non-negative.")
        if not any(abs(gamma - existing) < 1e-9 for existing in gammas):
            gammas.append(gamma)
    return gammas or [1.0]


def _gamma_slug(gamma: float) -> str:
    if abs(float(gamma)) < 1e-9:
        return "g0"
    if abs(float(gamma) - 0.5) < 1e-9:
        return "g05"
    if abs(float(gamma) - 1.0) < 1e-9:
        return "g1"
    text = f"{float(gamma):g}".replace(".", "p").replace("-", "m")
    return f"g{text}"


def _scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


def _compute_ffn_injection_features(
    *,
    source_ffn: torch.Tensor,
    source_attn: torch.Tensor | None,
    prediction_hidden: torch.Tensor,
    support_states: torch.Tensor,
    target_dist_visual: torch.Tensor | None,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    target_unembedding: torch.Tensor | None,
    evidence_top_k: int = 32,
    evidence_rank: int = 8,
    eps: float = EPS,
) -> dict[str, float]:
    source_ffn_f = source_ffn.float()
    eps_value = max(float(eps), EPS)
    if source_attn is None:
        ffn_attn_dominance = 0.0
    else:
        source_attn_f = source_attn.to(source_ffn_f.device).float()
        ffn_attn_dominance = float(
            torch.log(
                (source_ffn_f.norm(p=2) + eps_value)
                / (source_attn_f.norm(p=2) + eps_value)
            ).item()
        )

    ffn_logit_lift = 0.0
    if target_unembedding is not None:
        target_unembedding_f = target_unembedding.to(source_ffn_f.device).float()
        if target_unembedding_f.numel() == source_ffn_f.numel():
            ffn_logit_lift = float(torch.dot(target_unembedding_f.reshape(-1), source_ffn_f.reshape(-1)).item())

    eifdose = 0.0
    rank_used = 0
    eif_fraction_svd = 0.0
    eif_dose_svd = 0.0
    eif_fraction_pca = 0.0
    eif_dose_pca = 0.0
    rank_svd_used = 0
    rank_pca_used = 0
    if target_dist_visual is not None and target_dist_visual.numel() == support_states.shape[0]:
        support_states_f = support_states.to(source_ffn_f.device).float()
        target_dist_f = target_dist_visual.to(source_ffn_f.device).float()
        visual_indices = _visual_support_indices(
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            device=source_ffn_f.device,
        )
        if visual_indices.numel() > 0:
            visual_scores = target_dist_f.index_select(0, visual_indices)
            k = min(max(int(evidence_top_k), 1), int(visual_scores.numel()))
            if k > 0:
                topk = torch.topk(visual_scores, k=k).indices
                selected = visual_indices.index_select(0, topk)
                weights = target_dist_f.index_select(0, selected).clamp_min(0.0)
                weight_sum = weights.sum()
                if weight_sum > eps_value:
                    h_mid_t = prediction_hidden.to(source_ffn_f.device).float() - source_ffn_f
                    anchor_states = support_states_f.index_select(0, selected)
                    variant_scores = _compute_ffn_evidence_subspace_variants(
                        source_ffn=source_ffn_f,
                        prediction_h_mid=h_mid_t,
                        anchor_states=anchor_states,
                        evidence_rank=evidence_rank,
                        eps=eps_value,
                    )
                    eif_fraction_svd = variant_scores["ffn_eif_fraction_svd"]
                    eif_dose_svd = variant_scores["ffn_eif_dose_svd"]
                    eif_fraction_pca = variant_scores["ffn_eif_fraction_pca"]
                    eif_dose_pca = variant_scores["ffn_eif_dose_pca"]
                    rank_svd_used = int(variant_scores["ffn_evidence_svd_rank_used"])
                    rank_pca_used = int(variant_scores["ffn_evidence_pca_rank_used"])
                    directions = support_states_f.index_select(0, selected) - h_mid_t.unsqueeze(0)
                    directions = F.normalize(directions.float(), dim=-1, eps=eps_value)
                    weights = weights / (weight_sum + eps_value)
                    directions = directions * torch.sqrt(weights).unsqueeze(-1)
                    try:
                        _u, _s, vh = torch.linalg.svd(directions, full_matrices=False)
                        rank_used = min(int(evidence_rank), int(vh.shape[0]))
                        if rank_used > 0:
                            basis = vh[:rank_used].T.contiguous()
                            projection = basis @ (basis.T @ source_ffn_f)
                            orthogonal = source_ffn_f - projection
                            eifdose = float(
                                (orthogonal.norm(p=2) / (h_mid_t.norm(p=2) + eps_value)).item()
                            )
                    except RuntimeError:
                        eifdose = 0.0
                        rank_used = 0

    return {
        "ffn_attn_dominance": float(ffn_attn_dominance),
        "ffn_evidence_orthogonal_dose": float(eifdose),
        "ffn_logit_lift": float(ffn_logit_lift),
        "ffn_evidence_subspace_rank_used": int(rank_used),
        "ffn_eif_fraction_svd": float(eif_fraction_svd),
        "ffn_eif_dose_svd": float(eif_dose_svd),
        "ffn_eif_fraction_pca": float(eif_fraction_pca),
        "ffn_eif_dose_pca": float(eif_dose_pca),
        "ffn_evidence_svd_rank_used": int(rank_svd_used),
        "ffn_evidence_pca_rank_used": int(rank_pca_used),
    }


def _compute_ffn_evidence_subspace_variants(
    *,
    source_ffn: torch.Tensor,
    prediction_h_mid: torch.Tensor,
    anchor_states: torch.Tensor,
    evidence_rank: int,
    eps: float,
) -> dict[str, float | int]:
    source_ffn_f = source_ffn.float()
    prediction_h_mid_f = prediction_h_mid.to(source_ffn_f.device).float()
    anchors = anchor_states.to(source_ffn_f.device).float()
    if anchors.ndim != 2 or anchors.shape[0] == 0 or anchors.shape[-1] != source_ffn_f.numel():
        return _empty_ffn_evidence_variant_scores()

    svd_scores = _compute_ffn_evidence_variant_for_matrix(
        source_ffn=source_ffn_f,
        prediction_h_mid=prediction_h_mid_f,
        evidence_matrix=anchors,
        evidence_rank=evidence_rank,
        rank_cap=int(min(anchors.shape[0], anchors.shape[-1])),
        eps=eps,
    )

    centered = anchors - anchors.mean(dim=0, keepdim=True)
    pca_scores = _compute_ffn_evidence_variant_for_matrix(
        source_ffn=source_ffn_f,
        prediction_h_mid=prediction_h_mid_f,
        evidence_matrix=centered,
        evidence_rank=evidence_rank,
        rank_cap=int(min(max(int(anchors.shape[0]) - 1, 0), anchors.shape[-1])),
        eps=eps,
    )
    if bool(svd_scores.get("failed", False)) or bool(pca_scores.get("failed", False)):
        return _empty_ffn_evidence_variant_scores()

    return {
        "ffn_eif_fraction_svd": float(svd_scores["fraction"]),
        "ffn_eif_dose_svd": float(svd_scores["dose"]),
        "ffn_evidence_svd_rank_used": int(svd_scores["rank_used"]),
        "ffn_eif_fraction_pca": float(pca_scores["fraction"]),
        "ffn_eif_dose_pca": float(pca_scores["dose"]),
        "ffn_evidence_pca_rank_used": int(pca_scores["rank_used"]),
    }


def _compute_ffn_evidence_variant_for_matrix(
    *,
    source_ffn: torch.Tensor,
    prediction_h_mid: torch.Tensor,
    evidence_matrix: torch.Tensor,
    evidence_rank: int,
    rank_cap: int,
    eps: float,
) -> dict[str, float | int]:
    if int(rank_cap) <= 0 or evidence_matrix.numel() == 0:
        orthogonal = source_ffn
        return {
            "fraction": _ffn_eif_fraction(source_ffn, orthogonal, eps),
            "dose": _ffn_eif_dose(prediction_h_mid, orthogonal, eps),
            "rank_used": 0,
        }
    try:
        _u, singular_values, vh = torch.linalg.svd(evidence_matrix.float(), full_matrices=False)
    except RuntimeError:
        return {"fraction": 0.0, "dose": 0.0, "rank_used": 0, "failed": True}

    numerical_rank = int((singular_values > float(eps)).sum().item())
    rank_used = min(int(evidence_rank), int(rank_cap), int(vh.shape[0]), numerical_rank)
    if rank_used <= 0:
        orthogonal = source_ffn
        return {
            "fraction": _ffn_eif_fraction(source_ffn, orthogonal, eps),
            "dose": _ffn_eif_dose(prediction_h_mid, orthogonal, eps),
            "rank_used": 0,
        }

    basis = vh[:rank_used].T.contiguous()
    projection = basis @ (basis.T @ source_ffn)
    orthogonal = source_ffn - projection
    return {
        "fraction": _ffn_eif_fraction(source_ffn, orthogonal, eps),
        "dose": _ffn_eif_dose(prediction_h_mid, orthogonal, eps),
        "rank_used": int(rank_used),
        "failed": False,
    }


def _ffn_eif_fraction(source_ffn: torch.Tensor, orthogonal: torch.Tensor, eps: float) -> float:
    return float(
        (
            orthogonal.norm(p=2).pow(2)
            / (source_ffn.norm(p=2).pow(2) + float(eps))
        ).item()
    )


def _ffn_eif_dose(prediction_h_mid: torch.Tensor, orthogonal: torch.Tensor, eps: float) -> float:
    return float((orthogonal.norm(p=2) / (prediction_h_mid.norm(p=2) + float(eps))).item())


def _ffn_gate_ratio(
    *,
    source_ffn: torch.Tensor,
    prediction_h_mid: torch.Tensor,
    eps: float,
) -> float:
    source_norm = source_ffn.float().norm(p=2)
    h_mid_norm = prediction_h_mid.to(source_ffn.device).float().norm(p=2)
    value = source_norm / (source_norm + h_mid_norm + max(float(eps), EPS))
    return float(value.item())


def _empty_ffn_evidence_variant_scores() -> dict[str, float | int]:
    return {
        "ffn_eif_fraction_svd": 0.0,
        "ffn_eif_dose_svd": 0.0,
        "ffn_evidence_svd_rank_used": 0,
        "ffn_eif_fraction_pca": 0.0,
        "ffn_eif_dose_pca": 0.0,
        "ffn_evidence_pca_rank_used": 0,
    }


def _feature_vector(
    risk_per_layer: Sequence[float],
    prompt_last_cosine_per_layer: Sequence[float],
    prompt_mean_cosine_per_layer: Sequence[float],
    context_confidence_per_layer: Sequence[float],
) -> list[float]:
    return [
        *[float(value) for value in risk_per_layer],
        *[float(value) for value in prompt_last_cosine_per_layer],
        *[float(value) for value in prompt_mean_cosine_per_layer],
        *[float(value) for value in context_confidence_per_layer],
    ]


def _source_distribution(
    *,
    source_update: torch.Tensor,
    support_states: torch.Tensor,
    tau: float,
    mode: str = "softmax",
) -> torch.Tensor:
    update = source_update.float()
    states = support_states.float()
    scores = F.cosine_similarity(update.unsqueeze(0), states, dim=-1)
    return _source_distribution_from_scores(scores, tau=tau, mode=mode)


def _source_delta_distribution(
    *,
    prediction_h_mid: torch.Tensor,
    prediction_h_out: torch.Tensor,
    support_states: torch.Tensor,
    tau: float,
    mode: str = "softmax",
) -> torch.Tensor:
    states = support_states.float()
    before = F.cosine_similarity(prediction_h_mid.float().unsqueeze(0), states, dim=-1)
    after = F.cosine_similarity(prediction_h_out.float().unsqueeze(0), states, dim=-1)
    scores = after - before
    return _source_distribution_from_scores(scores, tau=tau, mode=mode)


def _source_projection_distribution(
    *,
    source_update: torch.Tensor,
    support_states: torch.Tensor,
    tau: float,
    mode: str = "softmax",
) -> torch.Tensor:
    update = source_update.float()
    states = support_states.float()
    scores = torch.matmul(states, update) / states.norm(p=2, dim=-1).clamp_min(EPS)
    return _source_distribution_from_scores(scores, tau=tau, mode=mode)


def _source_distribution_from_scores(
    scores: torch.Tensor,
    *,
    tau: float,
    mode: str,
) -> torch.Tensor:
    distribution_mode = _normalize_source_distribution_mode(mode)
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if distribution_mode == "softmax":
        return torch.softmax(scores / max(float(tau), 1e-6), dim=-1)
    return _renormalize(scores)


def _renormalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / max(int(values.numel()), 1))
    return values / total


def _normalized_entropy(
    values: torch.Tensor,
    *,
    normalizer_count: int | None = None,
) -> float:
    count = int(normalizer_count) if normalizer_count is not None else int(values.numel())
    if count <= 1 or values.numel() == 0:
        return 0.0
    probs = _renormalize(values).float()
    probs = probs[probs > EPS]
    if probs.numel() == 0:
        return 0.0
    entropy = -torch.sum(probs * torch.log(probs))
    normalizer = torch.log(
        torch.tensor(float(max(count, 2)), dtype=torch.float32, device=entropy.device)
    )
    if normalizer <= EPS:
        return 0.0
    value = entropy / normalizer
    return float(value.clamp(min=0.0, max=1.0).item())


def _append_topk_region_features(
    *,
    series: dict[str, dict[str, list[float]]],
    stem: str,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support_states: torch.Tensor,
    semantic_probs: torch.Tensor,
    transport_top_k: int,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    ot_solver: str,
    union_risk: float | None = None,
) -> None:
    stats = _topk_region_stats(
        source_dist=source_dist,
        target_dist=target_dist,
        transport_top_k=transport_top_k,
    )
    if stats is None:
        return

    sk = stats["source_topk"]
    tk = stats["target_topk"]
    union_support = stats["union_support"]
    source_entropy = float(stats["source_entropy"])
    cov_st = float(stats["topk_cov_st"])
    tkm = float(stats["topk_tkm"])
    es = float(stats["topk_es"])

    if union_risk is None:
        union_risk_value = _transport_risk_on_support(
            source_dist=source_dist,
            target_dist=target_dist,
            support_states=support_states,
            semantic_probs=semantic_probs,
            support=union_support,
            cost_mode="geo",
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
            ot_solver=ot_solver,
        )
    else:
        union_risk_value = float(union_risk)

    rec_risk_value = _transport_risk_rectangular_topk(
        source_dist=source_dist,
        target_dist=target_dist,
        support_states=support_states,
        semantic_probs=semantic_probs,
        source_support=sk,
        target_support=tk,
        cost_mode="geo",
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
        ot_solver=ot_solver,
    )
    target_risk_value = _transport_risk_on_support(
        source_dist=source_dist,
        target_dist=target_dist,
        support_states=support_states,
        semantic_probs=semantic_probs,
        support=tk,
        cost_mode="geo",
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
        ot_solver=ot_solver,
    )

    block = series.setdefault(stem, {})
    for name in ("topk_skm", "topk_tkm", "topk_cov_st", "topk_es"):
        block.setdefault(name, []).append(float(stats[name]))
    for selector, risk_value in (
        ("union", union_risk_value),
        ("rec", rec_risk_value),
        ("target", target_risk_value),
    ):
        adjusted = _topk_adjusted_risks(
            risk=float(risk_value),
            cov_st=cov_st,
            tkm=tkm,
            es=es,
            source_entropy=source_entropy,
        )
        for suffix, value in adjusted.items():
            name = f"r_{selector}{suffix}"
            block.setdefault(name, []).append(float(value))


def _topk_region_stats(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    transport_top_k: int,
) -> dict[str, torch.Tensor | float] | None:
    if source_dist.numel() == 0 or target_dist.numel() == 0:
        return None
    if int(source_dist.numel()) != int(target_dist.numel()):
        raise ValueError("Top-k region stats require source and target to share support length.")
    source = torch.nan_to_num(
        source_dist.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    target = torch.nan_to_num(
        target_dist.to(source.device).float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    source_topk, target_topk = _source_target_topk_indices(
        source=source,
        target=target,
        top_k=transport_top_k,
    )
    union_support = torch.unique(torch.cat([source_topk, target_topk], dim=0), sorted=True)
    return {
        "source_topk": source_topk,
        "target_topk": target_topk,
        "union_support": union_support,
        "topk_skm": float(source.index_select(0, source_topk).sum().item()),
        "topk_tkm": float(target.index_select(0, target_topk).sum().item()),
        "topk_cov_st": float(source.index_select(0, target_topk).sum().item()),
        "topk_es": float(target.sum().item()),
        "source_entropy": _normalized_entropy(source),
    }


def _source_target_topk_indices(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_k = min(max(int(top_k // 2), 1), int(source.numel()))
    target_k = min(max(int(top_k // 2), 1), int(target.numel()))
    source_idx = torch.topk(source, k=source_k).indices
    target_idx = torch.topk(target, k=target_k).indices
    return source_idx, target_idx


def _topk_adjusted_risks(
    *,
    risk: float,
    cov_st: float,
    tkm: float,
    es: float,
    source_entropy: float,
) -> dict[str, float]:
    risk_value = float(risk)
    cov_log = torch.log(torch.tensor(max(float(cov_st), EPS), dtype=torch.float32)).item()
    tkm_log = torch.log(torch.tensor(max(float(tkm), EPS), dtype=torch.float32)).item()
    es_log = torch.log(torch.tensor(max(float(es), EPS), dtype=torch.float32)).item()
    return {
        "_lk": float(risk_value - cov_log - tkm_log),
        "": float(risk_value - float(cov_st) - float(tkm)),
        "_la": float(risk_value - cov_log - es_log),
        "_lp": float(risk_value - float(source_entropy) * cov_log - tkm_log),
    }


def _source_target_divergences(source: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    source_prob = _smooth_probability(source)
    target_prob = _smooth_probability(target)
    midpoint = _smooth_probability(0.5 * (source_prob + target_prob))
    kl_target_source = _kl_divergence(target_prob, source_prob)
    kl_source_target = _kl_divergence(source_prob, target_prob)
    js = 0.5 * _kl_divergence(target_prob, midpoint) + 0.5 * _kl_divergence(source_prob, midpoint)
    return {
        "js": float(js.item()),
        "kl_target_source": float(kl_target_source.item()),
        "kl_source_target": float(kl_source_target.item()),
    }


def _smooth_probability(values: torch.Tensor) -> torch.Tensor:
    probs = _renormalize(values).float().clamp_min(EPS)
    return probs / probs.sum()


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = _smooth_probability(p)
    q = _smooth_probability(q)
    return torch.sum(p * (torch.log(p) - torch.log(q)))


def _source_target_divergences_on_support(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, float]:
    return _source_target_divergences(
        source=source.index_select(0, support),
        target=target.index_select(0, support),
    )


def _relative_vll_evidence_signal(
    *,
    attention_signal: torch.Tensor,
    target_logits: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    candidate_scope: str,
    stat_prefix: str,
    epsilon: float,
    barrier_margin: float,
    barrier_max: float,
    attention_gamma: float = 1.0,
    attention_epsilon: float = 0.0,
    mad_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    logits = torch.nan_to_num(target_logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if logits.numel() != attention_signal.numel():
        raise ValueError(
            "relative_vll logits must align with support positions, got "
            f"{int(logits.numel())} logits and {int(attention_signal.numel())} attention values."
        )
    candidate_index = _support_indices_for_scope(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope=candidate_scope,
        device=logits.device,
    )
    if candidate_index.numel() == 0:
        raise ValueError(
            "relative_vll target construction requires at least one "
            f"{candidate_scope} support token."
        )

    candidate_logits = logits.index_select(0, candidate_index)
    median = candidate_logits.median()
    mad = torch.abs(candidate_logits - median).median()
    scale = float(mad_scale)
    if scale <= 0.0:
        raise ValueError("relative-VLL mad_scale must be positive.")
    z = (candidate_logits - median) / (
        scale * mad + max(float(epsilon), EPS)
    )
    candidate_gate = torch.sigmoid(z)
    candidate_barrier = torch.relu(-(z + float(barrier_margin))).clamp_max(
        max(float(barrier_max), 0.0)
    )
    semantic_gate = torch.zeros_like(logits, dtype=torch.float32)
    semantic_gate.index_copy_(0, candidate_index, candidate_gate.float())
    relative_barrier = torch.zeros_like(logits, dtype=torch.float32)
    relative_barrier.index_copy_(0, candidate_index, candidate_barrier.float())

    gamma = float(attention_gamma)
    attention_weight = torch.pow(
        attention_signal.float().clamp_min(0.0) + max(float(attention_epsilon), 0.0),
        gamma,
    )
    evidence = attention_weight * semantic_gate
    evidence_strength = evidence.sum()

    stats = {
        f"{stat_prefix}_logit_median": float(median.item()),
        f"{stat_prefix}_logit_mad": float(mad.item()),
        f"{stat_prefix}_gate_mean": float(candidate_gate.mean().item()),
        f"{stat_prefix}_gate_max": float(candidate_gate.max().item()),
        f"{stat_prefix}_barrier_mean": float(candidate_barrier.mean().item()),
        f"{stat_prefix}_barrier_max": float(candidate_barrier.max().item()),
        f"{stat_prefix}_evidence_strength": float(evidence_strength.item()),
        f"{stat_prefix}_attention_gamma": float(gamma),
        f"{stat_prefix}_mad_scale": float(scale),
    }
    return evidence, semantic_gate, relative_barrier, stats


@torch.inference_mode()
def compute_four_gate_dgst_batch_from_captures(
    *,
    model: Any,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    target_token_ids: Sequence[int],
    prediction_positions: Sequence[int],
    prompt_positions: Sequence[int] | None = None,
    semantic_chunk_size: int = 64,
    tau: float = 0.07,
    transport_top_k: int = 64,
    target_region_top_k: int = 32,
    mad_epsilon: float = RELATIVE_VLL_MAD_EPSILON,
    cost_mode: str = "sqrt_matched_state",
    cost_modes: Sequence[str] | str | None = None,
    enabled_methods: Sequence[str] | None = None,
    compute_dual_scope: bool = False,
    support_modes: Sequence[str] | str | None = None,
    compute_prompt_cafe: bool = False,
    prompt_cafe_temperature: float = 10.0,
    prompt_cafe_layer: int = 22,
    compute_ffn_injection_features: bool = False,
    ffn_injection_eps: float = EPS,
    release_layer_captures: bool = False,
) -> list[dict[str, Any]]:
    """Compute compact VV features and optional VP/VPend/FAD features.

    VV remains the unprefixed, backward-compatible branch. ``support_modes``
    explicitly selects VV, legacy VP, VPend, or any combination. Legacy VP
    uses every non-visual prompt token. VPend instead uses only prompt tokens
    at ``position >= visual_end`` and is stored under ``dgst_t_vpend_*``.
    ``compute_ffn_injection_features`` adds the historical layerwise FAD
    curve without re-enabling the other legacy FFN diagnostics.
    ``compute_prompt_cafe`` adds the InsLen calibration confidence: at every
    layer, project each hpre instruction state through the LM head with
    temperature scaling, read the generated object token probability, and
    take the maximum over prompt positions.
    """
    active_modes = _normalize_four_gate_support_modes(
        support_modes,
        legacy_compute_dual_scope=bool(compute_dual_scope),
    )
    visual_positions = list(range(int(visual_start), int(visual_end)))
    scopes: list[tuple[str, list[int]]] = []
    if "vv" in active_modes:
        scopes.append(("visual", visual_positions))
    if "vp" in active_modes:
        if not prompt_positions:
            raise ValueError(
                "four-gate VP extraction requires at least one prompt support token."
            )
        vp_positions = sorted(
            dict.fromkeys(
                visual_positions + [int(position) for position in prompt_positions]
            )
        )
        scopes.append(("visual_prompt", vp_positions))
    if "vpend" in active_modes:
        post_visual_prompt_positions = [
            int(position)
            for position in prompt_positions or ()
            if int(position) >= int(visual_end)
        ]
        if not post_visual_prompt_positions:
            raise ValueError(
                "four-gate VPend extraction requires at least one prompt token "
                "at position >= visual_end."
            )
        vpend_positions = sorted(
            dict.fromkeys(visual_positions + post_visual_prompt_positions)
        )
        scopes.append(("visual_prompt_end", vpend_positions))

    scoped_results: dict[str, list[dict[str, Any]]] = {}
    for scope_name, support_positions in scopes:
        scoped_results[scope_name] = _compute_four_gate_single_scope_from_captures(
            model=model,
            captures=captures,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            support_positions=support_positions,
            support_scope=scope_name,
            target_token_ids=target_token_ids,
            prediction_positions=prediction_positions,
            semantic_chunk_size=int(semantic_chunk_size),
            tau=float(tau),
            transport_top_k=int(transport_top_k),
            target_region_top_k=int(target_region_top_k),
            mad_epsilon=float(mad_epsilon),
            cost_mode=cost_mode,
            cost_modes=cost_modes,
            enabled_methods=enabled_methods,
            release_layer_captures=False,
        )

    if "visual" in scoped_results:
        results = scoped_results["visual"]
    else:
        # Prompt-scope-only modes intentionally have no unprefixed per-layer
        # values: those names are reserved for the backward-compatible VV
        # branch.
        primary_prompt_scope = scopes[0][0]
        results = [
            {
                key: value
                for key, value in prompt_result.items()
                if not key.endswith("_per_layer")
            }
            for prompt_result in scoped_results[primary_prompt_scope]
        ]
    for result in results:
        result["dgst_t_four_gate_support_scopes"] = [name for name, _ in scopes]
        if "vv" in active_modes:
            result["dgst_t_vv_support_size"] = len(visual_positions)
            result["dgst_t_vv_support_positions"] = list(visual_positions)

    profile_slug = "_".join(active_modes)
    for result in results:
        if len(active_modes) > 1 or active_modes[0] != "vv":
            result["dgst_t_profile"] = f"four_gate_{profile_slug}_v1"

    for mode, scope_name, field_prefix in (
        ("vp", "visual_prompt", "vp"),
        ("vpend", "visual_prompt_end", "vpend"),
    ):
        if scope_name not in scoped_results:
            continue
        prompt_results = scoped_results[scope_name]
        if len(prompt_results) != len(results):
            raise AssertionError(
                f"VV and {mode.upper()} four-gate result counts differ."
            )
        for result, prompt_result in zip(results, prompt_results):
            result[f"dgst_t_{field_prefix}_support_size"] = int(
                prompt_result["dgst_t_support_size"]
            )
            result[f"dgst_t_{field_prefix}_support_positions"] = list(
                prompt_result["dgst_t_support_positions"]
            )
            for key, value in prompt_result.items():
                if key.startswith("dgst_t_") and (
                    key.endswith("_per_layer")
                    or key
                    in {
                        "dgst_t_attention_support_per_layer",
                        "dgst_t_source_dist_per_layer",
                    }
                ):
                    result[
                        f"dgst_t_{field_prefix}_{key[len('dgst_t_'):]}"
                    ] = value

    if compute_prompt_cafe:
        from models.dgst_capture import (
            resolve_output_embedding_layer,
            target_probabilities_multi,
        )

        instruction_positions = [
            int(position)
            for position in prompt_positions or ()
            if int(position) >= int(visual_end)
        ]
        if not instruction_positions:
            raise ValueError(
                "prompt CAFE extraction requires at least one post-visual "
                "instruction token."
            )
        temperature_value = float(prompt_cafe_temperature)
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError(
                "prompt_cafe_temperature must be a finite positive value."
            )
        layer_count = len(captures)
        requested_layer = int(prompt_cafe_layer)
        resolved_layer = (
            requested_layer
            if requested_layer >= 0
            else layer_count + requested_layer
        )
        if resolved_layer < 0 or resolved_layer >= layer_count:
            raise ValueError(
                f"prompt_cafe_layer={requested_layer} is outside {layer_count} "
                "decoder layers."
            )
        output_layer = resolve_output_embedding_layer(model)
        prompt_index = torch.tensor(
            instruction_positions,
            dtype=torch.long,
            device=captures[0]["h_prev"].device,
        )
        cafe_by_target: list[list[float]] = [[] for _ in target_token_ids]
        for layer_index, capture in enumerate(captures):
            h_prev = capture.get("h_prev")
            if h_prev is None:
                raise RuntimeError(
                    f"prompt CAFE layer {layer_index} is missing h_prev capture."
                )
            layer_prompt_index = prompt_index.to(device=h_prev.device)
            prompt_hpre = h_prev[0].index_select(0, layer_prompt_index)
            prompt_target_probs = target_probabilities_multi(
                output_layer=output_layer,
                states=prompt_hpre,
                target_token_ids=target_token_ids,
                chunk_size=int(semantic_chunk_size),
                temperature=temperature_value,
            )
            layer_maxima = prompt_target_probs.float().max(dim=0).values
            for target_offset, value in enumerate(layer_maxima):
                cafe_by_target[target_offset].append(float(value.item()))
        for result, values in zip(results, cafe_by_target):
            curve = torch.tensor(values, dtype=torch.float32)
            result["dgst_t_prompt_cafe_per_layer"] = curve
            result["dgst_t_prompt_cafe"] = float(curve[resolved_layer].item())
            result["dgst_t_prompt_cafe_layer"] = int(resolved_layer)
            result["dgst_t_prompt_cafe_requested_layer"] = requested_layer
            result["dgst_t_prompt_cafe_temperature"] = temperature_value
            result["dgst_t_prompt_cafe_prompt_size"] = int(prompt_index.numel())
            result["dgst_t_prompt_cafe_position_scope"] = (
                "post_visual_instruction_tokens"
            )
            result["dgst_t_prompt_cafe_definition"] = (
                "max_prompt_position_softmax_lm_head_hpre_over_temperature_"
                "target_token_probability"
            )

    if compute_ffn_injection_features:
        pred_positions = [int(position) for position in prediction_positions]
        eps_value = max(float(ffn_injection_eps), EPS)
        fad_by_target: list[list[float]] = [[] for _ in pred_positions]
        for layer_index, capture in enumerate(captures):
            if capture.get("o_attn") is None:
                raise RuntimeError(
                    "ffn_fad requires retained MHSA updates; layer "
                    f"{layer_index} has no o_attn capture."
                )
            if capture.get("o_ffn") is None:
                raise RuntimeError(
                    f"ffn_fad requires FFN updates; layer {layer_index} has none."
                )
            sequence_length = int(capture["o_ffn"].shape[1])
            for target_offset, position in enumerate(pred_positions):
                if position < 0 or position >= sequence_length:
                    raise ValueError(
                        f"Prediction position {position} is outside sequence length "
                        f"{sequence_length}."
                    )
                ffn_update = capture["o_ffn"][0, position].float()
                attn_update = capture["o_attn"][0, position].float()
                fad_by_target[target_offset].append(
                    float(
                        torch.log(
                            (ffn_update.norm(p=2) + eps_value)
                            / (attn_update.norm(p=2) + eps_value)
                        ).item()
                    )
                )
        for result, fad_values in zip(results, fad_by_target):
            result["dgst_t_ffn_attn_dominance_per_layer"] = torch.tensor(
                fad_values, dtype=torch.float32
            )
            result["dgst_t_ffn_attn_dominance_definition"] = (
                "log((l2_norm(o_ffn)+eps)/(l2_norm(o_attn)+eps))"
            )

    if release_layer_captures:
        for capture in captures:
            for key in ("h_prev", "o_attn", "h_mid", "o_ffn", "attn_weights"):
                capture[key] = None
    return results


@torch.inference_mode()
def _compute_four_gate_single_scope_from_captures(
    *,
    model: Any,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    support_positions: Sequence[int],
    support_scope: str,
    target_token_ids: Sequence[int],
    prediction_positions: Sequence[int],
    semantic_chunk_size: int = 64,
    tau: float = 0.07,
    transport_top_k: int = 64,
    target_region_top_k: int = 32,
    mad_epsilon: float = RELATIVE_VLL_MAD_EPSILON,
    cost_mode: str = "sqrt_matched_state",
    cost_modes: Sequence[str] | str | None = None,
    enabled_methods: Sequence[str] | None = None,
    release_layer_captures: bool = False,
) -> list[dict[str, Any]]:
    """Compute one compact four-gate support scope from decoder captures.

    This is intentionally an early, compact path.  It does not build the
    historical ``dgst_t_raw`` payload, legacy gates or FFN diagnostics. For
    h_pre and h_mid,
    one chunked vocabulary projection produces both the target raw logit and
    the target vocabulary-softmax probability.
    """
    from models.dgst_capture import (
        resolve_output_embedding_layer,
    )

    target_ids = [int(token_id) for token_id in target_token_ids]
    pred_positions = [int(position) for position in prediction_positions]
    if len(target_ids) != len(pred_positions):
        raise ValueError("target_token_ids and prediction_positions must have the same length.")
    if not target_ids:
        return []
    if not captures:
        raise ValueError("four-gate DGST requires at least one decoder-layer capture.")
    normalized_support_positions = [int(position) for position in support_positions]
    if not normalized_support_positions:
        raise ValueError("four-gate DGST requires at least one support token.")
    if str(support_scope) not in {
        "visual",
        "visual_prompt",
        "visual_prompt_end",
    }:
        raise ValueError(
            "four-gate support_scope must be visual, visual_prompt, or "
            "visual_prompt_end."
        )
    if int(target_region_top_k) <= 0:
        raise ValueError("target_region_top_k must be a positive integer.")
    methods = _normalize_four_gate_methods(enabled_methods)
    normalized_cost_mode = _normalize_four_gate_cost_mode(cost_mode)
    normalized_cost_modes = _normalize_four_gate_cost_modes(
        cost_modes,
        primary_cost_mode=normalized_cost_mode,
    )
    output_layer = resolve_output_embedding_layer(model)

    records: list[dict[str, Any]] = []
    for _target_id in target_ids:
        records.append(
            {
                "attention": [],
                "source": [],
                "gates": {
                    method: []
                    for method in methods
                    if method != DIRECT_HPRE_SOFTMAX_METHOD
                },
                "direct_hpre_target_probs": [],
                "direct_hpre_target_dist": [],
                "problems": {
                    (method, active_cost): []
                    for method in methods
                    for active_cost in normalized_cost_modes
                },
                "cosines": {method: [] for method in methods},
                "ev": {method: [] for method in methods},
            }
        )

    for layer_index, capture in enumerate(captures):
        missing = [
            key
            for key in ("h_prev", "h_mid", "o_ffn", "attn_weights")
            if capture.get(key) is None
        ]
        if missing:
            raise RuntimeError(
                f"four-gate DGST layer {layer_index} is missing captures: {missing}."
            )
        compact_capture = build_compact_four_gate_layer_capture(
            output_layer=output_layer,
            capture=capture,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            support_positions=normalized_support_positions,
            target_token_ids=target_ids,
            prediction_positions=pred_positions,
            semantic_chunk_size=int(semantic_chunk_size),
            tau=float(tau),
            enabled_methods=methods,
        )
        if tuple(compact_capture) != FOUR_GATE_CAPTURE_FIELDS:
            raise AssertionError("Unexpected fields in compact four-gate capture.")
        visual_hpre = compact_capture["visual_hpre"]
        support_index = torch.tensor(
            normalized_support_positions,
            dtype=torch.long,
            device=capture["h_mid"].device,
        )
        visual_hmid = capture["h_mid"][0].index_select(0, support_index).float()
        visual_hout = visual_hmid + capture["o_ffn"][0].index_select(
            0, support_index
        ).float()
        visual_update = visual_hout - visual_hmid
        sequence_length = int(capture["h_prev"].shape[1])

        for target_offset, prediction_position in enumerate(pred_positions):
            if prediction_position < 0 or prediction_position >= sequence_length:
                raise ValueError(
                    f"Prediction position {prediction_position} is outside sequence length "
                    f"{sequence_length}."
                )
            attention_support = compact_capture["attention_support"][target_offset]
            source_dist = compact_capture["source_dist"][target_offset]
            prediction_hpre = compact_capture["prediction_hpre"][target_offset]
            prediction_hmid = capture["h_mid"][0, prediction_position].float()
            state_views = {
                "hpre": (prediction_hpre, visual_hpre),
                "hmid": (prediction_hmid, visual_hmid),
            }
            cosine_maps = {}
            for state_name, (prediction_state, visual_states) in state_views.items():
                cosine_map = F.cosine_similarity(
                    prediction_state.unsqueeze(0),
                    visual_states,
                    dim=-1,
                )
                cosine_maps[state_name] = torch.nan_to_num(
                    cosine_map.float(), nan=0.0, posinf=1.0, neginf=-1.0
                ).clamp(-1.0, 1.0)
            gate_input_fields = {
                "hpre_raw_logit_gauss": "hpre_raw_target_logits",
                HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD: "hpre_raw_target_logits",
                "hpre_softmax_prob_gauss": "hpre_softmax_target_probs",
                "hmid_raw_logit_gauss": "hmid_raw_target_logits",
                "hmid_softmax_prob_gauss": "hmid_softmax_target_probs",
            }

            record = records[target_offset]
            record["attention"].append(attention_support)
            record["source"].append(source_dist)
            for method in methods:
                state_name = _target_comparison_state(method)
                _prediction_state, cost_states = state_views[state_name]
                cosine_map = cosine_maps[state_name]
                if method == RAW_ATTENTION_METHOD:
                    # This baseline uses the model's post-softmax attention
                    # weights directly.  The all-ones gate keeps the stored
                    # matrix schema aligned with the Gaussian-gated methods.
                    gate = torch.ones_like(attention_support).detach()
                    target_dist = attention_support.detach()
                elif method == DIRECT_HPRE_SOFTMAX_METHOD:
                    target_probs = compact_capture["hpre_softmax_target_probs"]
                    if target_probs is None:
                        raise RuntimeError(
                            "Direct hpre-softmax target distribution requires "
                            "vocabulary-softmax target probabilities."
                        )
                    # The vocabulary softmax is performed independently at every
                    # visual token.  Its target-token probabilities are then
                    # normalized across visual positions to form the OT marginal.
                    target_dist = _renormalize(
                        target_probs[target_offset]
                    ).detach()
                    record["direct_hpre_target_probs"].append(
                        target_probs[target_offset].detach()
                    )
                    record["direct_hpre_target_dist"].append(target_dist)
                else:
                    gate_values = compact_capture[gate_input_fields[method]]
                    if gate_values is None:
                        raise RuntimeError(
                            f"Missing compact target values for enabled method {method}."
                        )
                    if method == HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD:
                        gate = _relative_vll_mad_gate(
                            gate_values[target_offset],
                            epsilon=float(mad_epsilon),
                        ).detach()
                    else:
                        gate = _gaussian_mad_gate(
                            gate_values[target_offset],
                            epsilon=float(mad_epsilon),
                        ).detach()
                    target_dist = _renormalize(attention_support * gate).detach()
                support = _topk_union_indices(
                    source_dist,
                    target_dist,
                    int(transport_top_k),
                )
                for active_cost in normalized_cost_modes:
                    problem = _prepare_four_gate_cost_problem(
                        cost_mode=active_cost,
                        source_dist=source_dist,
                        target_dist=target_dist,
                        matched_states=cost_states,
                        hmid_states=visual_hmid,
                        hout_states=visual_hout,
                        update_states=visual_update,
                        support=support,
                    )
                    record["problems"][(method, active_cost)].append(problem)
                region = _stable_topk_indices(
                    target_dist,
                    int(target_region_top_k),
                )
                if region.numel() == 0:
                    target_cosine = 0.0
                    evidence_value = 0.0
                else:
                    local_cosine = cosine_map.index_select(0, region)
                    target_cosine = float(local_cosine.mean().item())
                    # EV couples two properties of the branch-specific target
                    # region: how much of that branch's target distribution is
                    # concentrated in its top-K tokens, and how well those
                    # tokens align with the matched prediction state.  Keep the
                    # region cosine unweighted so it is exactly the separately
                    # reported target-cosine feature.
                    target_region_mass = target_dist.index_select(0, region).sum()
                    evidence_value = float(
                        (target_region_mass * local_cosine.mean()).item()
                    )
                if method != DIRECT_HPRE_SOFTMAX_METHOD:
                    record["gates"][method].append(gate)
                record["cosines"][method].append(target_cosine)
                record["ev"][method].append(evidence_value)

        del compact_capture, visual_hpre, visual_hmid, visual_hout, visual_update
        if release_layer_captures:
            # Method-only extraction has no downstream consumer for the hook
            # tensors.  Drop every large decoder reference as soon as this
            # layer has been reduced to the compact inputs/results.
            for key in (
                "h_prev",
                "o_attn",
                "h_mid",
                "o_ffn",
                "attn_weights",
            ):
                capture[key] = None

    results: list[dict[str, Any]] = []
    for target_offset, record in enumerate(records):
        # POT's exact EMD is the only solver exposed by this active profile.
        risk_series = _solve_exact_emd_problem_series(record["problems"])
        result: dict[str, Any] = {
            "dgst_t_profile": (
                "target_comparison_v3"
                if DIRECT_HPRE_SOFTMAX_METHOD in methods
                else (
                    "target_comparison_v2"
                    if RAW_ATTENTION_METHOD in methods
                    else "four_gate_vv_v1"
                )
            ),
            "dgst_t_target_token_id": int(target_ids[target_offset]),
            "dgst_t_prediction_position": int(pred_positions[target_offset]),
            "dgst_t_four_gate_methods": list(methods),
            "dgst_t_mad_axis": {
                "visual": "visual_tokens",
                "visual_prompt": "visual_prompt_tokens",
                "visual_prompt_end": "visual_post_prompt_tokens",
            }[str(support_scope)],
            "dgst_t_support_scope": str(support_scope),
            "dgst_t_support_size": len(normalized_support_positions),
            "dgst_t_support_positions": list(normalized_support_positions),
            "dgst_t_mad_scale": float(GAUSSIAN_MAD_SCALE),
            "dgst_t_mad_scale_by_method": {
                method: (
                    1.0
                    if method == HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD
                    else float(GAUSSIAN_MAD_SCALE)
                )
                for method in methods
                if method not in {DIRECT_HPRE_SOFTMAX_METHOD, RAW_ATTENTION_METHOD}
            },
            "dgst_t_softmax_axis": "vocabulary",
            "dgst_t_source_distribution_mode": "softmax",
            "dgst_t_state_by_method": {
                method: _target_comparison_state(method)
                for method in methods
            },
            "dgst_t_transport_top_k": int(transport_top_k),
            "dgst_t_target_region_top_k": int(target_region_top_k),
            "dgst_t_ev_definition": (
                "target_dist_topk_mass_x_mean_target_cosine"
            ),
            "dgst_t_cost": normalized_cost_mode,
            "dgst_t_cost_modes": list(normalized_cost_modes),
            "dgst_t_ot_solver": "emd",
            "dgst_t_attention_support_per_layer": torch.stack(
                record["attention"], dim=0
            ).detach().to(device="cpu", dtype=torch.float32),
            "dgst_t_source_dist_per_layer": torch.stack(
                record["source"], dim=0
            ).detach().to(device="cpu", dtype=torch.float32),
        }
        stateupd_alphas = {
            active_cost: alpha_value
            for active_cost in normalized_cost_modes
            if (alpha_value := _four_gate_stateupd_alpha(active_cost)) is not None
        }
        if stateupd_alphas:
            result["dgst_t_cost_alphas"] = stateupd_alphas
        primary_alpha = _four_gate_stateupd_alpha(normalized_cost_mode)
        if primary_alpha is not None:
            # Historical scalar provenance remains available when a state-update
            # cost is itself the primary mode. Multi-alpha runs use the mapping
            # above and keep sqrt_matched_state as their primary mode.
            result["dgst_t_cost_alpha"] = primary_alpha
        if RAW_ATTENTION_METHOD in methods:
            result["dgst_t_raw_attention_definition"] = (
                "post_softmax_head_mean_visual_support_renormalized"
            )
        if DIRECT_HPRE_SOFTMAX_METHOD in methods:
            result["dgst_t_hpre_softmax_prob_direct_definition"] = (
                "visual_hpre_vocabulary_softmax_target_probability_"
                "renormalized_over_visual_tokens"
            )
            result[
                "dgst_t_hpre_softmax_prob_direct_target_prob_matrix_per_layer"
            ] = torch.stack(
                record["direct_hpre_target_probs"], dim=0
            ).detach().to(device="cpu", dtype=torch.float32)
            result[
                "dgst_t_hpre_softmax_prob_direct_target_dist_per_layer"
            ] = torch.stack(
                record["direct_hpre_target_dist"], dim=0
            ).detach().to(device="cpu", dtype=torch.float32)
        for method in methods:
            state_name = _target_comparison_state(method)
            if method != DIRECT_HPRE_SOFTMAX_METHOD:
                result[f"dgst_t_{method}_gate_per_layer"] = torch.stack(
                    record["gates"][method], dim=0
                ).detach().to(device="cpu", dtype=torch.float32)
            for active_cost in normalized_cost_modes:
                risk_suffix = _four_gate_risk_suffix(active_cost, state_name)
                result[
                    f"dgst_t_{method}_{risk_suffix}_per_layer"
                ] = torch.tensor(
                    risk_series[(method, active_cost)], dtype=torch.float32
                )
            topk_slug = f"topk{int(target_region_top_k)}"
            result[
                f"dgst_t_{method}_target_cosine_{topk_slug}_{state_name}_per_layer"
            ] = torch.tensor(record["cosines"][method], dtype=torch.float32)
            result[
                f"dgst_t_{method}_ev_target_dist_mass_x_cosine_"
                f"{topk_slug}_{state_name}_per_layer"
            ] = torch.tensor(
                record["ev"][method], dtype=torch.float32
            )
        results.append(result)
    return results


@torch.inference_mode()
def build_compact_four_gate_layer_capture(
    *,
    output_layer: Any,
    capture: dict[str, Any],
    visual_start: int,
    visual_end: int,
    target_token_ids: Sequence[int],
    prediction_positions: Sequence[int],
    support_positions: Sequence[int] | None = None,
    semantic_chunk_size: int = 64,
    tau: float = 0.07,
    enabled_methods: Sequence[str] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Reduce one hook capture to the compact active-profile inputs.

    ``visual_hmid`` and the two full-vocabulary chunk matrices are deliberately
    transient.  Target raw logits and target vocabulary-softmax probabilities
    are produced by the same projection for each of hpre/hmid, then only their
    compact ``[T,P]`` columns survive this function.
    """
    from models.dgst_capture import (
        attention_row_from_capture,
        target_logits_and_probabilities_multi,
        target_logits_multi,
    )

    methods = _normalize_four_gate_methods(enabled_methods)

    h_prev = capture["h_prev"][0]
    h_mid = capture["h_mid"][0]
    o_ffn = capture["o_ffn"][0]
    sequence_length = int(h_prev.shape[0])
    if int(visual_start) < 0 or int(visual_end) > sequence_length:
        raise ValueError(
            "Visual support range is outside the captured sequence: "
            f"[{visual_start}, {visual_end}) vs {sequence_length}."
        )
    positions = [int(value) for value in prediction_positions]
    if any(value < 0 or value >= sequence_length for value in positions):
        raise ValueError("A prediction position is outside the captured sequence.")
    selected_support_positions = (
        list(range(int(visual_start), int(visual_end)))
        if support_positions is None
        else [int(position) for position in support_positions]
    )
    if not selected_support_positions:
        raise ValueError("Four-gate support positions must not be empty.")
    if any(
        position < 0 or position >= sequence_length
        for position in selected_support_positions
    ):
        raise ValueError("A four-gate support position is outside the captured sequence.")
    visual_index = torch.tensor(
        selected_support_positions, dtype=torch.long, device=h_prev.device
    )
    position_index = torch.tensor(positions, dtype=torch.long, device=h_prev.device)
    visual_hpre = h_prev.index_select(0, visual_index).float()
    visual_hmid = h_mid.index_select(0, visual_index).float()

    hpre_raw = hpre_prob = hmid_raw = hmid_prob = None
    needs_hpre_raw = (
        "hpre_raw_logit_gauss" in methods
        or HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD in methods
    )
    needs_hpre_prob = (
        "hpre_softmax_prob_gauss" in methods
        or DIRECT_HPRE_SOFTMAX_METHOD in methods
    )
    needs_hmid_raw = "hmid_raw_logit_gauss" in methods
    needs_hmid_prob = "hmid_softmax_prob_gauss" in methods

    if needs_hpre_prob:
        hpre_raw_projected, hpre_prob = target_logits_and_probabilities_multi(
            output_layer=output_layer,
            states=visual_hpre,
            target_token_ids=target_token_ids,
            chunk_size=int(semantic_chunk_size),
        )
        if needs_hpre_raw:
            hpre_raw = hpre_raw_projected
        else:
            del hpre_raw_projected
    elif needs_hpre_raw:
        hpre_raw = target_logits_multi(
            output_layer=output_layer,
            states=visual_hpre,
            target_token_ids=target_token_ids,
            chunk_size=int(semantic_chunk_size),
        )

    if needs_hmid_prob:
        hmid_raw_projected, hmid_prob = target_logits_and_probabilities_multi(
            output_layer=output_layer,
            states=visual_hmid,
            target_token_ids=target_token_ids,
            chunk_size=int(semantic_chunk_size),
        )
        if needs_hmid_raw:
            hmid_raw = hmid_raw_projected
        else:
            del hmid_raw_projected
    elif needs_hmid_raw:
        hmid_raw = target_logits_multi(
            output_layer=output_layer,
            states=visual_hmid,
            target_token_ids=target_token_ids,
            chunk_size=int(semantic_chunk_size),
        )
    attention_support = torch.stack(
        [
            _renormalize(
                attention_row_from_capture(capture, position)
                .index_select(1, visual_index)
                .float()
                .mean(dim=0)
            )
            for position in positions
        ],
        dim=0,
    )
    source_dist = torch.stack(
        [
            _source_distribution(
                source_update=o_ffn[position, :].float(),
                support_states=visual_hmid,
                tau=float(tau),
                mode="softmax",
            )
            for position in positions
        ],
        dim=0,
    )
    result = {
        "prediction_hpre": h_prev.index_select(0, position_index).float(),
        "visual_hpre": visual_hpre,
        "attention_support": attention_support,
        "source_dist": source_dist,
        "hpre_raw_target_logits": (
            hpre_raw.transpose(0, 1).contiguous().detach()
            if hpre_raw is not None else None
        ),
        "hpre_softmax_target_probs": (
            hpre_prob.transpose(0, 1).contiguous().detach()
            if hpre_prob is not None else None
        ),
        "hmid_raw_target_logits": (
            hmid_raw.transpose(0, 1).contiguous().detach()
            if hmid_raw is not None else None
        ),
        "hmid_softmax_target_probs": (
            hmid_prob.transpose(0, 1).contiguous().detach()
            if hmid_prob is not None else None
        ),
    }
    del (
        visual_hmid,
        hpre_raw,
        hpre_prob,
        hmid_raw,
        hmid_prob,
    )
    return result


def _normalize_four_gate_methods(
    values: Sequence[str] | None,
) -> tuple[str, ...]:
    if values is None:
        return FOUR_GATE_METHODS
    requested = [str(value).strip().lower() for value in values]
    unknown = sorted(set(requested) - set(TARGET_COMPARISON_METHODS))
    if unknown:
        raise ValueError(
            f"Unknown four-gate methods {unknown}; expected a subset of "
            f"{list(TARGET_COMPARISON_METHODS)}."
        )
    selected = tuple(
        method for method in TARGET_COMPARISON_METHODS if method in requested
    )
    if not selected:
        raise ValueError("At least one four-gate method must be enabled.")
    return selected


def _normalize_four_gate_support_modes(
    values: Sequence[str] | str | None,
    *,
    legacy_compute_dual_scope: bool = False,
) -> tuple[str, ...]:
    """Normalize the explicit VV/legacy-VP/VPend extraction switches.

    ``compute_dual_scope`` remains a compatibility fallback for older configs;
    an explicit ``support_modes`` value always wins.
    """
    if values is None:
        return ("vv", "vp") if legacy_compute_dual_scope else ("vv",)
    requested = [values] if isinstance(values, str) else list(values)
    aliases = {
        "vv": "vv",
        "visual": "vv",
        "vp": "vp",
        "visual_prompt": "vp",
        "visual+prompt": "vp",
        "vpend": "vpend",
        "vp_end": "vpend",
        "visual_prompt_end": "vpend",
        "visual+prompt_end": "vpend",
        "post_visual_prompt": "vpend",
    }
    normalized: list[str] = []
    unknown: list[str] = []
    for value in requested:
        raw = str(value).strip().lower()
        mode = aliases.get(raw)
        if mode is None:
            unknown.append(raw)
        elif mode not in normalized:
            normalized.append(mode)
    if unknown:
        raise ValueError(
            f"Unknown four-gate support modes {unknown}; expected vv, vp, "
            "and/or vpend."
        )
    if not normalized:
        raise ValueError("At least one four-gate support mode must be enabled.")
    return tuple(mode for mode in ("vv", "vp", "vpend") if mode in normalized)


def _normalize_four_gate_cost_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {
        "sqrt_matched_state",
        "sqrt_cosine_matched_state",
        "matched_state",
        # Before this option was wired into the compact path, the enclosing
        # API's legacy defaults were accepted but ignored.
        "direct",
        "decomposed",
    }:
        return "sqrt_cosine_matched_state"
    if mode in {
        "cosine_matched_state",
        "cos_matched_state",
        "one_minus_cosine_matched_state",
    }:
        return "cosine_matched_state"
    if mode in {
        "geo_stateupd_lu1",
        "geo_state_update_lu1",
        "geo_stateupd_lambda1",
    }:
        return "geo_stateupd_lu1"
    for alpha_tenth in STATEUPD_ALPHA_TENTHS:
        slug = f"0{alpha_tenth}"
        if mode in {
            f"sqrt_stateupd_alpha{slug}",
            f"sqrt_state_update_alpha{slug}",
            f"sqrt_stateupd_a{slug}",
        }:
            return f"sqrt_stateupd_alpha{slug}"
    raise ValueError(
        "four_gate cost_mode must be 'sqrt_matched_state' or "
        "'cosine_matched_state' or "
        "'geo_stateupd_lu1' or one of 'sqrt_stateupd_alpha01' through "
        "'sqrt_stateupd_alpha09'."
    )


def _normalize_four_gate_cost_modes(
    value: Sequence[str] | str | None,
    *,
    primary_cost_mode: str,
) -> tuple[str, ...]:
    raw_modes = [primary_cost_mode] if value is None else (
        [value] if isinstance(value, str) else list(value)
    )
    modes: list[str] = [str(primary_cost_mode)]
    for raw_mode in raw_modes:
        normalized = _normalize_four_gate_cost_mode(raw_mode)
        if normalized not in modes:
            modes.append(normalized)
    return tuple(modes)


def _four_gate_risk_suffix(cost_mode: str, state_name: str) -> str:
    normalized = _normalize_four_gate_cost_mode(cost_mode)
    if normalized == "geo_stateupd_lu1":
        return "risk_geo_stateupd_lu1"
    if _four_gate_stateupd_alpha(normalized) is not None:
        return f"risk_{normalized}"
    if normalized == "cosine_matched_state":
        return f"risk_cosine_{state_name}"
    return f"risk_sqrt_{state_name}"


def _prepare_four_gate_cost_problem(
    *,
    cost_mode: str,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    matched_states: torch.Tensor,
    hmid_states: torch.Tensor,
    hout_states: torch.Tensor,
    update_states: torch.Tensor,
    support: torch.Tensor,
):
    normalized = _normalize_four_gate_cost_mode(cost_mode)
    if normalized == "geo_stateupd_lu1":
        return _prepare_transport_problem_for_state_cost(
            source_dist=source_dist,
            target_dist=target_dist,
            states=hmid_states,
            output_states=hout_states,
            support=support,
            sqrt_cosine=False,
            cost_state_mode="state_update",
            update_lambda=1.0,
            keep_on_device=False,
        )
    stateupd_alpha = _four_gate_stateupd_alpha(normalized)
    if stateupd_alpha is not None:
        return _prepare_transport_problem_for_state_cost(
            source_dist=source_dist,
            target_dist=target_dist,
            states=matched_states,
            state_update_states=update_states,
            support=support,
            sqrt_cosine=False,
            state_update_mix_alpha=stateupd_alpha,
            keep_on_device=False,
        )
    if normalized == "cosine_matched_state":
        return _prepare_transport_problem_for_state_cost(
            source_dist=source_dist,
            target_dist=target_dist,
            states=matched_states,
            support=support,
            sqrt_cosine=False,
            keep_on_device=False,
        )
    return _prepare_transport_problem_for_state_cost(
        source_dist=source_dist,
        target_dist=target_dist,
        states=matched_states,
        support=support,
        sqrt_cosine=True,
        keep_on_device=False,
    )


def _four_gate_stateupd_alpha(cost_mode: str) -> float | None:
    """Return the state/update mixture weight encoded by a canonical mode."""
    normalized = _normalize_four_gate_cost_mode(cost_mode)
    prefix = "sqrt_stateupd_alpha0"
    if not normalized.startswith(prefix):
        return None
    return int(normalized.removeprefix(prefix)) / 10.0


def _target_comparison_state(method: str) -> str:
    """Return the hidden-state family paired with one target construction."""
    name = str(method).strip().lower()
    if name in {"hmid_raw_logit_gauss", "hmid_softmax_prob_gauss"}:
        return "hmid"
    if name in {
        "hpre_raw_logit_gauss",
        HPRE_RAW_LOGIT_RELATIVE_VLL_METHOD,
        "hpre_softmax_prob_gauss",
        DIRECT_HPRE_SOFTMAX_METHOD,
        RAW_ATTENTION_METHOD,
    }:
        return "hpre"
    raise ValueError(f"Unknown target-comparison method: {method!r}")


def _solve_exact_emd_problem_series(
    problem_series: dict[str, list[Any]],
) -> dict[str, list[float]]:
    """Solve the active-profile risks with POT EMD, ignoring solver overrides."""
    workers = _cost_variant_emd_workers(len(problem_series))
    if workers == 1:
        return {
            name: [_solve_transport_problem(problem, "emd") for problem in problems]
            for name, problems in problem_series.items()
        }
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dgst-four-gate-emd") as executor:
        futures = {
            name: executor.submit(_solve_transport_problem_batch, problems, "emd")
            for name, problems in problem_series.items()
        }
        return {name: futures[name].result() for name in problem_series}


@torch.no_grad()
def _relative_vll_mad_gate(values: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    clean = torch.nan_to_num(
        values.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    if clean.numel() == 0:
        return clean
    median = clean.median()
    mad = torch.abs(clean - median).median()
    z = (clean - median) / (mad + max(float(epsilon), EPS))
    return torch.sigmoid(z).detach()


@torch.no_grad()
def _gaussian_mad_gate(values: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    clean = torch.nan_to_num(
        values.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    if clean.numel() == 0:
        return clean
    median = clean.median()
    mad = torch.abs(clean - median).median()
    z = (clean - median) / (
        float(GAUSSIAN_MAD_SCALE) * mad + max(float(epsilon), EPS)
    )
    return torch.sigmoid(z).detach()


def _stable_topk_indices(values: torch.Tensor, top_k: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    k = min(max(int(top_k), 1), int(values.numel()))
    try:
        return torch.argsort(values, descending=True, stable=True)[:k]
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.argsort(values, descending=True)[:k]


def _compute_gate_comparison_from_parts(
    *,
    source_ffn_states: Sequence[torch.Tensor],
    source_attn_states: Sequence[torch.Tensor] | None,
    prediction_hidden_states: Sequence[torch.Tensor],
    support_h_prev_states: Sequence[torch.Tensor],
    support_h_mid_states: Sequence[torch.Tensor],
    support_attentions: Sequence[torch.Tensor],
    semantic_probs: Sequence[torch.Tensor],
    relative_vll_logits: Sequence[torch.Tensor],
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    tau: float,
    source_distribution_mode: str,
    transport_top_k: int,
    ot_solver: str,
    atarget_visual_top_k: int,
    relative_vll_mad_epsilon: float,
    relative_barrier_margin: float,
    relative_barrier_max: float,
) -> dict[str, Any]:
    """Compare matched VV target gates with one shared model capture.

    All methods use the same source distribution, top-k union, sqrt-cosine
    ground cost on h_pre, and h_pre target cosine.  The only changed variable
    is target construction:

    * relative_vll: raw target logits -> Gaussian-scaled MAD -> sigmoid;
    * softmax_relative_vll: full-vocabulary target probability ->
      Gaussian-scaled MAD -> sigmoid;
    * legacy_prob: full-vocabulary target probability, without MAD/sigmoid.
    """
    if _has_prompt_support_tokens(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
    ):
        raise ValueError("gate_comparison requires visual-only (VV) support.")
    if source_attn_states is None:
        raise ValueError("gate_comparison requires source attention states for h_pre.")

    visual_index = _support_indices_for_scope(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope="visual",
        device=support_h_mid_states[0].device,
    )
    if visual_index.numel() == 0:
        raise ValueError("gate_comparison requires at least one visual support token.")

    risk_problems: dict[str, list[Any]] = {
        method: [] for method in GATE_COMPARISON_METHODS
    }
    cosine_series: dict[str, list[float]] = {
        method: [] for method in GATE_COMPARISON_METHODS
    }
    keep_on_device = _effective_ot_solver(ot_solver) == "sinkhorn"

    for layer_idx in range(len(source_ffn_states)):
        hpre_states = support_h_prev_states[layer_idx].float()
        hmid_states = support_h_mid_states[layer_idx].float()
        source_ffn = source_ffn_states[layer_idx].to(hmid_states.device).float()
        source_attn = source_attn_states[layer_idx].to(hmid_states.device).float()
        prediction_hout = prediction_hidden_states[layer_idx].to(hmid_states.device).float()
        prediction_hpre = prediction_hout - source_ffn - source_attn
        attention = support_attentions[layer_idx].to(hmid_states.device).float()
        logits = relative_vll_logits[layer_idx].to(hmid_states.device).float()
        probabilities = semantic_probs[layer_idx].to(hmid_states.device).float()

        source_dist = _source_distribution(
            source_update=source_ffn,
            support_states=hmid_states,
            tau=tau,
            mode=source_distribution_mode,
        )
        relative_target, _relative_gate, _relative_barrier, _relative_stats = (
            _relative_vll_evidence_signal(
                attention_signal=attention,
                target_logits=logits,
                support_positions=support_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                candidate_scope="visual",
                stat_prefix="relative_vll_gauss",
                epsilon=relative_vll_mad_epsilon,
                barrier_margin=relative_barrier_margin,
                barrier_max=relative_barrier_max,
                mad_scale=GAUSSIAN_MAD_SCALE,
            )
        )
        softmax_target, _softmax_gate, _softmax_barrier, _softmax_stats = (
            _relative_vll_evidence_signal(
                attention_signal=attention,
                # semantic_probs contains p(w* | h_i), computed by applying
                # softmax over the full vocabulary independently at every
                # visual position.  MAD is then taken across visual tokens.
                target_logits=probabilities,
                support_positions=support_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                candidate_scope="visual",
                stat_prefix="softmax_relative_vll_gauss",
                epsilon=relative_vll_mad_epsilon,
                barrier_margin=relative_barrier_margin,
                barrier_max=relative_barrier_max,
                mad_scale=GAUSSIAN_MAD_SCALE,
            )
        )

        legacy_gate = torch.zeros_like(probabilities, dtype=torch.float32)
        local_visual_index = visual_index.to(probabilities.device)
        legacy_gate.index_copy_(
            0,
            local_visual_index,
            torch.nan_to_num(
                probabilities.index_select(0, local_visual_index),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0),
        )
        legacy_target = attention.clamp_min(0.0) * legacy_gate

        targets = {
            "relative_vll": relative_target,
            "softmax_relative_vll": softmax_target,
            "legacy_prob": legacy_target,
        }
        for method, target in targets.items():
            target = _renormalize(target)
            support = _topk_union_indices(source_dist, target, transport_top_k)
            risk_problems[method].append(
                _prepare_transport_problem_for_state_cost(
                    source_dist=source_dist,
                    target_dist=target,
                    states=hpre_states,
                    support=support,
                    sqrt_cosine=True,
                    keep_on_device=keep_on_device,
                )
            )
            cosine_series[method].append(
                _target_hidden_topk_visual_cosine(
                    target_hidden=prediction_hpre.to(hpre_states.device),
                    support_output_states=hpre_states,
                    target_dist=target,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    top_k=atarget_visual_top_k,
                )
            )

    risk_series = _solve_cost_variant_problem_series(
        risk_problems,
        ot_solver=ot_solver,
    )
    result: dict[str, Any] = {
        "dgst_t_source_distribution_mode": str(source_distribution_mode),
        "dgst_t_gate_comparison_methods": list(GATE_COMPARISON_METHODS),
        "dgst_t_gate_comparison_cost": "sqrt_cosine_hpre",
        "dgst_t_gate_comparison_target_cosine_state": "hpre",
        "dgst_t_gate_comparison_softmax_axis": "vocabulary",
        "dgst_t_gate_comparison_mad_axis": "visual_tokens",
        "dgst_t_gate_comparison_mad_scale": float(GAUSSIAN_MAD_SCALE),
        "dgst_t_gate_comparison_transport_top_k": int(transport_top_k),
        "dgst_t_gate_comparison_target_cosine_top_k": int(atarget_visual_top_k),
        "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer": torch.tensor(
            risk_series["relative_vll"], dtype=torch.float32
        ),
        "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer": torch.tensor(
            risk_series["softmax_relative_vll"], dtype=torch.float32
        ),
        "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer": torch.tensor(
            risk_series["legacy_prob"], dtype=torch.float32
        ),
        "dgst_t_relative_vll_gauss_target_visual_hpre_cosine_per_layer": torch.tensor(
            cosine_series["relative_vll"], dtype=torch.float32
        ),
        "dgst_t_softmax_relative_vll_gauss_target_visual_hpre_cosine_per_layer": torch.tensor(
            cosine_series["softmax_relative_vll"], dtype=torch.float32
        ),
        "dgst_t_legacy_prob_target_visual_hpre_cosine_per_layer": torch.tensor(
            cosine_series["legacy_prob"], dtype=torch.float32
        ),
    }
    return result


def _compute_cost_variant_risks(
    *,
    source_dist: torch.Tensor,
    relative_target: torch.Tensor,
    gauss_target: torch.Tensor,
    raw_attention_target: torch.Tensor,
    hmid_states: torch.Tensor,
    hpre_states: torch.Tensor,
    transport_top_k: int,
    ot_solver: str,
) -> dict[str, float]:
    problems = _prepare_cost_variant_problems(
        source_dist=source_dist,
        relative_target=relative_target,
        gauss_target=gauss_target,
        raw_attention_target=raw_attention_target,
        hmid_states=hmid_states,
        hpre_states=hpre_states,
        transport_top_k=transport_top_k,
        ot_solver=ot_solver,
    )
    solved = _solve_cost_variant_problem_series(
        {name: [problem] for name, problem in problems.items()},
        ot_solver=ot_solver,
    )
    return {name: values[0] for name, values in solved.items()}


def _prepare_cost_variant_problems(
    *,
    source_dist: torch.Tensor,
    relative_target: torch.Tensor,
    gauss_target: torch.Tensor,
    raw_attention_target: torch.Tensor,
    hmid_states: torch.Tensor,
    hpre_states: torch.Tensor,
    transport_top_k: int,
    ot_solver: str = "emd",
) -> dict[str, Any]:
    targets = {
        "relative": _renormalize(relative_target),
        "gauss": _renormalize(gauss_target),
        "raw_attention": _renormalize(raw_attention_target),
    }
    supports = {
        name: _topk_union_indices(source_dist, target, transport_top_k)
        for name, target in targets.items()
    }

    def problem(target_name: str, state_name: str, *, sqrt: bool = False):
        states = hmid_states if state_name == "hmid" else hpre_states
        return _prepare_transport_problem_for_state_cost(
            source_dist=source_dist,
            target_dist=targets[target_name],
            states=states,
            support=supports[target_name],
            sqrt_cosine=sqrt,
            keep_on_device=_effective_ot_solver(ot_solver) == "sinkhorn",
        )

    return {
        "risk_geo": problem("relative", "hmid"),
        "risk_cosine_hpre": problem("relative", "hpre"),
        "risk_sqrt_hmid": problem("relative", "hmid", sqrt=True),
        "risk_sqrt_hpre": problem("relative", "hpre", sqrt=True),
        "risk_raw_attention_hmid": problem("raw_attention", "hmid"),
        "risk_raw_attention_hpre": problem("raw_attention", "hpre"),
        "gauss_risk_geo": problem("gauss", "hmid"),
        "gauss_risk_cosine_hpre": problem("gauss", "hpre"),
        "gauss_risk_sqrt_hmid": problem("gauss", "hmid", sqrt=True),
        "gauss_risk_sqrt_hpre": problem("gauss", "hpre", sqrt=True),
    }


def _solve_cost_variant_problem_series(
    problem_series: dict[str, list[Any]],
    *,
    ot_solver: str,
) -> dict[str, list[float]]:
    effective_solver = _effective_ot_solver(ot_solver)
    if effective_solver == "sinkhorn":
        return _solve_sinkhorn_problem_series(problem_series)
    workers = _cost_variant_emd_workers(len(problem_series))
    if workers == 1:
        return {
            name: [_solve_transport_problem(problem, effective_solver) for problem in problems]
            for name, problems in problem_series.items()
        }
    backend = os.environ.get("DGST_COST_VARIANT_EMD_BACKEND", "thread").strip().lower()
    if backend == "process":
        executor = _get_emd_process_pool(workers)
        futures = {
            name: executor.submit(
                _solve_transport_problem_batch,
                problems,
                effective_solver,
            )
            for name, problems in problem_series.items()
        }
        return {name: futures[name].result() for name in problem_series}
    if backend != "thread":
        raise ValueError(
            "DGST_COST_VARIANT_EMD_BACKEND must be 'thread' or 'process'."
        )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dgst-emd") as executor:
        futures = {
            name: executor.submit(
                _solve_transport_problem_batch,
                problems,
                effective_solver,
            )
            for name, problems in problem_series.items()
        }
        # Preserve the canonical field order even though solves finish out of order.
        return {name: futures[name].result() for name in problem_series}


def _get_emd_process_pool(workers: int) -> ProcessPoolExecutor:
    global _EMD_PROCESS_POOL, _EMD_PROCESS_POOL_WORKERS
    if _EMD_PROCESS_POOL is None:
        _EMD_PROCESS_POOL = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_emd_worker,
        )
        _EMD_PROCESS_POOL_WORKERS = int(workers)
    elif _EMD_PROCESS_POOL_WORKERS != int(workers):
        raise RuntimeError(
            "DGST_COST_VARIANT_EMD_WORKERS cannot change after the process pool starts."
        )
    return _EMD_PROCESS_POOL


def _initialize_emd_worker() -> None:
    # Prevent a process per EMD task from recursively spawning BLAS/OpenMP
    # threads.  The outer process pool is the intended parallelism level.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _cost_variant_emd_workers(problem_count: int) -> int:
    raw = os.environ.get("DGST_COST_VARIANT_EMD_WORKERS", "1")
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError(
            "DGST_COST_VARIANT_EMD_WORKERS must be a positive integer."
        ) from exc
    if requested < 1:
        raise ValueError("DGST_COST_VARIANT_EMD_WORKERS must be >= 1.")
    return min(int(problem_count), requested)


def _effective_ot_solver(configured_solver: str) -> str:
    return os.environ.get("DGST_OT_SOLVER_OVERRIDE", configured_solver).strip().lower()


def _solve_sinkhorn_problem_series(
    problem_series: dict[str, list[Any]],
) -> dict[str, list[float]]:
    """Experimental batched log-domain Sinkhorn for cost-variant risks."""
    reg = float(os.environ.get("DGST_SINKHORN_REG", "0.05"))
    max_iter = int(os.environ.get("DGST_SINKHORN_MAX_ITER", "500"))
    tol = float(os.environ.get("DGST_SINKHORN_TOL", "1e-6"))
    max_marginal_error = float(
        os.environ.get("DGST_SINKHORN_MAX_MARGINAL_ERROR", "1e-4")
    )
    batch_size = int(os.environ.get("DGST_SINKHORN_BATCH_SIZE", "512"))
    if reg <= 0.0 or max_iter < 1 or tol <= 0.0 or batch_size < 1:
        raise ValueError("Invalid experimental Sinkhorn configuration.")

    output = {
        name: [0.0] * len(problems)
        for name, problems in problem_series.items()
    }
    grouped: dict[tuple[str, int, int], list[tuple[str, int, Any]]] = {}
    for name, problems in problem_series.items():
        for layer_index, problem in enumerate(problems):
            if problem is None:
                continue
            source, target, cost = problem
            if not torch.is_tensor(source):
                source = torch.from_numpy(source)
                target = torch.from_numpy(target)
                cost = torch.from_numpy(cost)
                problem = (source, target, cost)
            grouped.setdefault(
                (str(cost.device), int(cost.shape[0]), int(cost.shape[1])),
                [],
            ).append((name, layer_index, problem))

    for entries in grouped.values():
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            source = torch.stack([item[2][0] for item in chunk]).float()
            target = torch.stack([item[2][1] for item in chunk]).float()
            cost = torch.stack([item[2][2] for item in chunk]).float()
            risks, marginal_error = _sinkhorn_linear_cost_batch(
                source,
                target,
                cost,
                reg=reg,
                max_iter=max_iter,
                tol=tol,
            )
            if marginal_error > max_marginal_error:
                raise RuntimeError(
                    "Experimental Sinkhorn did not converge: "
                    f"marginal_error={marginal_error:.3e} > "
                    f"{max_marginal_error:.3e}."
                )
            for (name, layer_index, _problem), risk in zip(chunk, risks.tolist()):
                output[name][layer_index] = float(risk)
    return output


@torch.no_grad()
def _sinkhorn_linear_cost_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    reg: float,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, float]:
    """Return linear cost <C, Pi_epsilon> and maximum marginal residual."""
    source = _renormalize_batch(source.float())
    target = _renormalize_batch(target.float())
    cost = torch.nan_to_num(
        cost.float(), nan=1e6, posinf=1e6, neginf=1e6
    ).clamp_min(0.0)
    log_source = source.clamp_min(EPS).log()
    log_target = target.clamp_min(EPS).log()
    f = torch.zeros_like(source)
    g = torch.zeros_like(target)
    marginal_error = float("inf")
    plan = None
    for iteration in range(max_iter):
        f = reg * (
            log_source
            - torch.logsumexp((g.unsqueeze(1) - cost) / reg, dim=2)
        )
        g = reg * (
            log_target
            - torch.logsumexp((f.unsqueeze(2) - cost) / reg, dim=1)
        )
        if (iteration + 1) % 10 == 0 or iteration + 1 == max_iter:
            plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / reg)
            row_error = (plan.sum(dim=2) - source).abs().amax()
            col_error = (plan.sum(dim=1) - target).abs().amax()
            marginal_error = float(torch.maximum(row_error, col_error).item())
            if marginal_error <= tol:
                break
    if plan is None:
        plan = torch.exp((f.unsqueeze(2) + g.unsqueeze(1) - cost) / reg)
        marginal_error = float(
            torch.maximum(
                (plan.sum(dim=2) - source).abs().amax(),
                (plan.sum(dim=1) - target).abs().amax(),
            ).item()
        )
    return (plan * cost).sum(dim=(1, 2)), marginal_error


def _renormalize_batch(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(
        values, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    totals = values.sum(dim=1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / max(int(values.shape[1]), 1))
    return torch.where(totals > EPS, values / totals.clamp_min(EPS), uniform)


def _solve_transport_problem(problem, ot_solver: str) -> float:
    if problem is None:
        return 0.0
    local_source, local_target, distance = problem
    if not torch.is_tensor(local_source):
        local_source = torch.from_numpy(local_source)
        local_target = torch.from_numpy(local_target)
        distance = torch.from_numpy(distance)
    transport_risk, _transport_plan = _wasserstein_1_exact(
        local_source,
        local_target,
        distance,
        solver=ot_solver,
    )
    return float(transport_risk)


def _solve_transport_problem_batch(problems, ot_solver: str) -> list[float]:
    return [_solve_transport_problem(problem, ot_solver) for problem in problems]


def _prepare_transport_problem_for_state_cost(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    states: torch.Tensor,
    output_states: torch.Tensor | None = None,
    state_update_states: torch.Tensor | None = None,
    support: torch.Tensor,
    sqrt_cosine: bool,
    cost_state_mode: str = "mid",
    update_lambda: float = 0.0,
    state_update_mix_alpha: float | None = None,
    keep_on_device: bool = False,
):
    """Build one OT problem on the caller thread, then move it to CPU.

    Keeping CUDA indexing/cost construction out of worker threads avoids
    concurrent CUDA stream access.  Only the independent POT EMD solves run in
    parallel.
    """
    if support.numel() == 0:
        return None
    local_source = _renormalize(source_dist.index_select(0, support))
    local_target = _renormalize(target_dist.index_select(0, support))
    local_states = states.index_select(0, support)
    local_output_states = (
        local_states
        if output_states is None
        else output_states.index_select(0, support)
    )
    local_update_states = (
        None
        if state_update_states is None
        else state_update_states.index_select(0, support)
    )
    if state_update_mix_alpha is not None:
        distance = _sqrt_state_update_mixture_distance_matrix(
            local_states,
            local_output_states,
            update_states=local_update_states,
            alpha=float(state_update_mix_alpha),
        )
    else:
        distance = _cost_state_distance_matrix(
            local_states,
            local_output_states,
            mode=cost_state_mode,
            update_lambda=update_lambda,
        )
        if sqrt_cosine:
            distance = torch.sqrt((distance / 2.0).clamp_min(0.0))
    if keep_on_device:
        return (
            local_source.detach(),
            local_target.detach(),
            distance.detach(),
        )
    # NumPy copies use normal pickle payloads for ProcessPool IPC. Passing
    # hundreds of CPU torch tensors would create one shared-memory file
    # descriptor per storage and exhaust the worker's FD limit.
    return (
        local_source.detach().float().cpu().numpy().copy(),
        local_target.detach().float().cpu().numpy().copy(),
        distance.detach().float().cpu().numpy().copy(),
    )


def _transport_risk_for_state_cost(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    states: torch.Tensor,
    support: torch.Tensor,
    sqrt_cosine: bool,
    ot_solver: str,
) -> float:
    problem = _prepare_transport_problem_for_state_cost(
        source_dist=source_dist,
        target_dist=target_dist,
        states=states,
        support=support,
        sqrt_cosine=sqrt_cosine,
    )
    return _solve_transport_problem(problem, ot_solver)


def _has_prompt_support_tokens(
    *,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
) -> bool:
    return any(
        not (int(visual_start) <= int(position) < int(visual_end))
        for position in support_positions
    )


def _visual_support_indices(
    *,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    device: torch.device,
) -> torch.Tensor:
    return _support_indices_for_scope(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope="visual",
        device=device,
    )


def _target_mass_for_scope(
    *,
    target_dist: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    scope: str,
) -> torch.Tensor:
    return _mass_for_scope(
        values=target_dist,
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope=scope,
    )


def _mass_for_scope(
    *,
    values: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    scope: str,
) -> torch.Tensor:
    indices = _support_indices_for_scope(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope=scope,
        device=values.device,
    )
    if indices.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return values.float().index_select(0, indices).sum()


def _local_mass_for_scope(
    *,
    values: torch.Tensor,
    support: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    scope: str,
) -> torch.Tensor:
    if support.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    local_values = _renormalize(values.index_select(0, support))
    selected_positions = [support_positions[int(index)] for index in support.detach().cpu().tolist()]
    local_indices = _support_indices_for_scope(
        support_positions=selected_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope=scope,
        device=values.device,
    )
    if local_indices.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=values.device)
    return local_values.index_select(0, local_indices).sum()


def _support_indices_for_scope(
    *,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    scope: str,
    device: torch.device,
) -> torch.Tensor:
    scope_name = str(scope).strip().lower()
    if scope_name in {"visual_prompt", "support", "all"}:
        return torch.arange(len(support_positions), dtype=torch.long, device=device)
    if scope_name not in {"visual", "prompt"}:
        raise ValueError("DGST-T support scope must be 'visual', 'prompt', or 'visual_prompt'.")
    indices = [
        offset
        for offset, position in enumerate(support_positions)
        if (
            int(visual_start) <= int(position) < int(visual_end)
            if scope_name == "visual"
            else not (int(visual_start) <= int(position) < int(visual_end))
        )
    ]
    return torch.tensor(indices, dtype=torch.long, device=device)


def _topk_union_indices(source: torch.Tensor, target: torch.Tensor, top_k: int) -> torch.Tensor:
    src_idx, tgt_idx = _source_target_topk_indices(
        source=source,
        target=target,
        top_k=top_k,
    )
    return torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)


def _append_capped_support_series(
    *,
    series: dict[str, dict[str, list[list[int]]]],
    stem: str,
    support: torch.Tensor,
    support_positions: Sequence[int],
) -> None:
    block = series.setdefault(stem, {"indices": [], "positions": []})
    indices = [int(index) for index in support.detach().cpu().tolist()]
    positions = [int(support_positions[index]) for index in indices]
    block["indices"].append(indices)
    block["positions"].append(positions)


def _topmass_k(values: torch.Tensor, alpha: float) -> int:
    probs = _renormalize(values)
    n = int(probs.numel())
    if n <= 0:
        return 0
    threshold = min(max(float(alpha), 0.0), 1.0)
    if threshold <= 0.0:
        return 1
    sorted_probs = torch.sort(probs, descending=True).values
    cumulative = torch.cumsum(sorted_probs, dim=0)
    hits = torch.nonzero(cumulative >= threshold, as_tuple=False)
    if hits.numel() == 0:
        return n
    return min(int(hits[0].item()) + 1, n)


def _topmass_indices(values: torch.Tensor, alpha: float) -> torch.Tensor:
    probs = _renormalize(values)
    k = _topmass_k(probs, alpha)
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=values.device)
    return torch.topk(probs, k=k).indices


def _topmass_union_indices(source: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    src_idx = _topmass_indices(source, alpha)
    tgt_idx = _topmass_indices(target, alpha)
    return torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)


def _capped_topmass_indices(
    values: torch.Tensor,
    alpha: float,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    probs = _renormalize(values)
    n = int(probs.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=values.device)
    lower = max(int(min_k), 1)
    upper = max(int(max_k), lower)
    k = min(upper, max(lower, _topmass_k(probs, alpha)))
    k = min(k, n)
    return torch.topk(probs, k=k).indices


def _capped_topmass_union_indices(
    source: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    *,
    min_k: int,
    max_k: int,
) -> torch.Tensor:
    src_idx = _capped_topmass_indices(source, alpha, min_k=min_k, max_k=max_k)
    tgt_idx = _capped_topmass_indices(target, alpha, min_k=min_k, max_k=max_k)
    return torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)


def _transport_risks_by_cost(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support_states: torch.Tensor,
    semantic_probs: torch.Tensor,
    support: torch.Tensor,
    cost_modes: Sequence[str],
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    relative_barrier: torch.Tensor | None,
    relative_barrier_lambda: float,
    ot_solver: str,
) -> dict[str, float]:
    return {
        mode: _transport_risk_on_support(
            source_dist=source_dist,
            target_dist=target_dist,
            support_states=support_states,
            semantic_probs=semantic_probs,
            support=support,
            cost_mode=mode,
            lambda_d=lambda_d,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            lambda_int=lambda_int,
            ot_solver=ot_solver,
            relative_barrier=relative_barrier,
            relative_barrier_lambda=relative_barrier_lambda,
        )
        for mode in cost_modes
    }


def _transport_risks_by_cost_state(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support_states: torch.Tensor,
    support_output_states: torch.Tensor,
    semantic_probs: torch.Tensor,
    support: torch.Tensor,
    cost_modes: Sequence[str],
    cost_state_specs: Sequence[tuple[str, float | None, str]],
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    relative_barrier: torch.Tensor | None,
    relative_barrier_lambda: float,
    ot_solver: str,
) -> dict[tuple[str, str], float]:
    risks: dict[tuple[str, str], float] = {}
    for mode in cost_modes:
        for state_mode, update_lambda, state_slug in cost_state_specs:
            risks[(mode, state_slug)] = _transport_risk_on_support(
                source_dist=source_dist,
                target_dist=target_dist,
                support_states=support_states,
                support_output_states=support_output_states,
                semantic_probs=semantic_probs,
                support=support,
                cost_mode=mode,
                lambda_d=lambda_d,
                lambda_s=lambda_s,
                lambda_t=lambda_t,
                lambda_int=lambda_int,
                ot_solver=ot_solver,
                relative_barrier=relative_barrier,
                relative_barrier_lambda=relative_barrier_lambda,
                cost_state_mode=state_mode,
                relative_cost_update_lambda=0.0 if update_lambda is None else float(update_lambda),
            )
    return risks


def _transport_risk_on_support(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support_states: torch.Tensor,
    semantic_probs: torch.Tensor,
    support: torch.Tensor,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    ot_solver: str,
    support_output_states: torch.Tensor | None = None,
    relative_barrier: torch.Tensor | None = None,
    relative_barrier_lambda: float = 1.0,
    cost_state_mode: str = "mid",
    relative_cost_update_lambda: float = 0.1,
) -> float:
    if support.numel() == 0:
        return 0.0
    local_source = _renormalize(source_dist.index_select(0, support))
    local_target = _renormalize(target_dist.index_select(0, support))
    local_states = support_states.index_select(0, support)
    local_output_states = (
        local_states
        if support_output_states is None
        else support_output_states.index_select(0, support)
    )
    local_semantic = semantic_probs.index_select(0, support)
    local_barrier = (
        None
        if relative_barrier is None
        else relative_barrier.to(support.device).index_select(0, support)
    )

    distance = _cost_state_distance_matrix(
        local_states,
        local_output_states,
        mode=cost_state_mode,
        update_lambda=relative_cost_update_lambda,
    )
    source_penalty = torch.relu(1.0 - local_semantic)
    target_penalty = torch.relu(1.0 - local_semantic)
    cost = _build_cost_matrix(
        distance,
        source_penalty,
        target_penalty,
        cost_mode=cost_mode,
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
        relative_barrier=local_barrier,
        relative_barrier_lambda=relative_barrier_lambda,
    )
    transport_risk, _transport_plan = _wasserstein_1_exact(
        local_source,
        local_target,
        cost,
        solver=ot_solver,
    )
    return float(transport_risk)


def _transport_risk_rectangular_topk(
    *,
    source_dist: torch.Tensor,
    target_dist: torch.Tensor,
    support_states: torch.Tensor,
    semantic_probs: torch.Tensor,
    source_support: torch.Tensor,
    target_support: torch.Tensor,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    ot_solver: str,
) -> float:
    if source_support.numel() == 0 or target_support.numel() == 0:
        return 0.0
    mode = str(cost_mode).strip().lower()
    if mode not in {"geo", "direct", "decomposed"}:
        raise ValueError("Rectangular top-k OT supports geo/direct/decomposed costs.")

    local_source = _renormalize(source_dist.index_select(0, source_support))
    local_target = _renormalize(target_dist.index_select(0, target_support))
    source_states = support_states.index_select(0, source_support)
    target_states = support_states.index_select(0, target_support)
    source_semantic = semantic_probs.index_select(0, source_support)
    target_semantic = semantic_probs.index_select(0, target_support)

    distance = _cross_cosine_distance_matrix(source_states, target_states)
    source_penalty = torch.relu(1.0 - source_semantic)
    target_penalty = torch.relu(1.0 - target_semantic)
    cost = _build_cost_matrix(
        distance,
        source_penalty,
        target_penalty,
        cost_mode=mode,
        lambda_d=lambda_d,
        lambda_s=lambda_s,
        lambda_t=lambda_t,
        lambda_int=lambda_int,
    )
    transport_risk, _transport_plan = _wasserstein_1_exact(
        local_source,
        local_target,
        cost,
        solver=ot_solver,
    )
    return float(transport_risk)


def _cosine_distance_matrix(states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(states.float(), dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(0, 1))
    return torch.nan_to_num((1.0 - cosine).clamp_min(0.0), nan=0.0, posinf=2.0, neginf=0.0)


def _cross_cosine_distance_matrix(source_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    source_norm = F.normalize(source_states.float(), dim=-1)
    target_norm = F.normalize(target_states.float(), dim=-1)
    cosine = torch.matmul(source_norm, target_norm.transpose(0, 1))
    return torch.nan_to_num((1.0 - cosine).clamp_min(0.0), nan=0.0, posinf=2.0, neginf=0.0)


def _cost_state_distance_matrix(
    mid_states: torch.Tensor,
    output_states: torch.Tensor,
    *,
    mode: str,
    update_lambda: float,
) -> torch.Tensor:
    state_mode = _normalize_relative_cost_state_modes(mode)[0]
    mid_distance = _cosine_distance_matrix(mid_states)
    if state_mode == "mid":
        return mid_distance
    out_distance = _cosine_distance_matrix(output_states)
    if state_mode == "out":
        return out_distance
    if state_mode == "avg":
        return 0.5 * (mid_distance + out_distance)
    update_distance = _cosine_distance_matrix(output_states.float() - mid_states.float())
    return mid_distance + float(update_lambda) * update_distance


def _sqrt_state_update_mixture_distance_matrix(
    mid_states: torch.Tensor,
    output_states: torch.Tensor,
    *,
    update_states: torch.Tensor | None = None,
    alpha: float,
) -> torch.Tensor:
    mix = float(alpha)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("state/update mixture alpha must be in [0, 1].")
    state_distance = torch.sqrt(
        (_cosine_distance_matrix(mid_states) / 2.0).clamp_min(0.0)
    )
    updates = (
        output_states.float() - mid_states.float()
        if update_states is None
        else update_states.float()
    )
    update_distance = torch.sqrt(
        (_cosine_distance_matrix(updates) / 2.0).clamp_min(0.0)
    )
    return (1.0 - mix) * state_distance + mix * update_distance


def _build_cost_matrix(
    distance: torch.Tensor,
    source_penalty: torch.Tensor,
    target_penalty: torch.Tensor,
    *,
    cost_mode: str,
    lambda_d: float,
    lambda_s: float,
    lambda_t: float,
    lambda_int: float,
    relative_barrier: torch.Tensor | None = None,
    relative_barrier_lambda: float = 1.0,
) -> torch.Tensor:
    mode = str(cost_mode).strip().lower()
    if mode == "geo":
        return float(lambda_d) * distance
    if mode in {"target_barrier_geo", "symmetric_barrier_geo"}:
        if relative_barrier is None:
            raise ValueError(f"{mode} requires relative_barrier values.")
        barrier = torch.nan_to_num(
            relative_barrier.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        target_barrier = barrier.unsqueeze(0)
        if mode == "target_barrier_geo":
            multiplier = 1.0 + float(relative_barrier_lambda) * target_barrier
        else:
            source_barrier = barrier.unsqueeze(1)
            multiplier = 1.0 + float(relative_barrier_lambda) * (
                source_barrier + target_barrier
            )
        return float(lambda_d) * distance * multiplier
    if mode in {
        "target_additive_barrier_geo",
        "source_additive_barrier_geo",
        "two_end_additive_barrier_geo",
    }:
        if relative_barrier is None:
            raise ValueError(f"{mode} requires relative_barrier values.")
        barrier = torch.nan_to_num(
            relative_barrier.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        cost = float(lambda_d) * distance
        if mode in {"source_additive_barrier_geo", "two_end_additive_barrier_geo"}:
            cost = cost + float(lambda_s) * barrier.unsqueeze(1)
        if mode in {"target_additive_barrier_geo", "two_end_additive_barrier_geo"}:
            cost = cost + float(lambda_t) * barrier.unsqueeze(0)
        return cost
    if mode in {
        "semantic_match_geo",
        "semantic_match",
        "qmatch",
        "qmatch_geo",
        "qadd",
        "qadd_geo",
        "sqrt_q_geo",
        "sqrtq_geo",
    }:
        q_source = torch.nan_to_num(
            1.0 - source_penalty.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        q_target = torch.nan_to_num(
            1.0 - target_penalty.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        q_match = torch.sqrt(
            (q_source.unsqueeze(1) * q_target.unsqueeze(0)).clamp_min(0.0)
        )
        return float(lambda_d) * distance + (1.0 - q_match)

    source = source_penalty.float().unsqueeze(1)
    target = target_penalty.float().unsqueeze(0)
    interaction = distance.float() * (source + target)
    if mode == "decomposed":
        return float(lambda_d) * distance + float(lambda_int) * interaction
    if mode == "direct":
        return (
            float(lambda_d) * distance
            + float(lambda_s) * source
            + float(lambda_t) * target
            + float(lambda_int) * interaction
        )
    raise ValueError("DGST-T only keeps direct/decomposed transport costs.")


def _target_hidden_topk_visual_cosine(
    *,
    target_hidden: torch.Tensor,
    support_output_states: torch.Tensor,
    target_dist: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    top_k: int,
) -> float:
    visual_indices = [
        offset
        for offset, position in enumerate(support_positions)
        if int(visual_start) <= int(position) < int(visual_end)
    ]
    if not visual_indices:
        return 0.0
    visual_index = torch.tensor(visual_indices, dtype=torch.long, device=target_dist.device)
    visual_scores = target_dist.index_select(0, visual_index)
    k = min(max(int(top_k), 1), int(visual_scores.numel()))
    selected = visual_index.index_select(0, torch.topk(visual_scores, k=k).indices)
    selected_states = support_output_states.index_select(0, selected.to(support_output_states.device)).float()
    similarities = F.cosine_similarity(target_hidden.float().unsqueeze(0), selected_states, dim=-1)
    return float(similarities.mean().item())


def _target_hidden_topk_support_cosine(
    *,
    target_hidden: torch.Tensor,
    support_output_states: torch.Tensor,
    target_dist: torch.Tensor,
    top_k: int,
) -> float:
    if target_dist.numel() == 0:
        return 0.0
    k = min(max(int(top_k), 1), int(target_dist.numel()))
    selected = torch.topk(target_dist, k=k).indices
    selected_states = support_output_states.index_select(0, selected.to(support_output_states.device)).float()
    similarities = F.cosine_similarity(target_hidden.float().unsqueeze(0), selected_states, dim=-1)
    return float(similarities.mean().item())


def _target_hidden_capped_topmass_visual_cosine(
    *,
    target_hidden: torch.Tensor,
    support_output_states: torch.Tensor,
    target_dist: torch.Tensor,
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    alpha: float,
    min_k: int,
    max_k: int,
) -> float:
    visual_indices = [
        offset
        for offset, position in enumerate(support_positions)
        if int(visual_start) <= int(position) < int(visual_end)
    ]
    if not visual_indices:
        return 0.0
    visual_index = torch.tensor(visual_indices, dtype=torch.long, device=target_dist.device)
    visual_scores = target_dist.index_select(0, visual_index)
    local_selected = _capped_topmass_indices(visual_scores, alpha, min_k=min_k, max_k=max_k)
    selected = visual_index.index_select(0, local_selected)
    selected_states = support_output_states.index_select(0, selected.to(support_output_states.device)).float()
    similarities = F.cosine_similarity(target_hidden.float().unsqueeze(0), selected_states, dim=-1)
    return float(similarities.mean().item())


def _target_hidden_capped_topmass_support_cosine(
    *,
    target_hidden: torch.Tensor,
    support_output_states: torch.Tensor,
    target_dist: torch.Tensor,
    alpha: float,
    min_k: int,
    max_k: int,
) -> float:
    if target_dist.numel() == 0:
        return 0.0
    selected = _capped_topmass_indices(target_dist, alpha, min_k=min_k, max_k=max_k)
    selected_states = support_output_states.index_select(0, selected.to(support_output_states.device)).float()
    similarities = F.cosine_similarity(target_hidden.float().unsqueeze(0), selected_states, dim=-1)
    return float(similarities.mean().item())


def _baseline_excess_score(
    risk_per_layer: torch.Tensor,
    *,
    baseline_layers: int,
    risk_start_layer: int,
    alpha: float,
) -> float:
    if risk_per_layer.numel() == 0:
        return 0.0
    base_width = max(1, min(int(baseline_layers), int(risk_per_layer.numel())))
    baseline = risk_per_layer[:base_width]
    baseline_mean = float(baseline.mean().item())
    baseline_std = float(baseline.std(unbiased=False).item())
    threshold = baseline_mean + float(alpha) * baseline_std
    score = 0.0
    for layer_number, value in enumerate(risk_per_layer.tolist(), start=1):
        if layer_number >= int(risk_start_layer):
            score += max(0.0, float(value) - threshold)
    return float(score)


def _wasserstein_1_exact(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    *,
    solver: str,
) -> tuple[float, torch.Tensor]:
    solver_name = str(solver).strip().lower()
    if solver_name not in {"linprog", "emd"}:
        raise ValueError("DGST-T ot_solver must be 'linprog' or 'emd'.")

    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires numpy.") from exc

    source_np = _as_probability_numpy(source)
    target_np = _as_probability_numpy(target)
    source_np, target_np = _balance_ot_marginals(source_np, target_np)
    cost_np = torch.nan_to_num(
        cost.detach().cpu().double(),
        nan=1e6,
        posinf=1e6,
        neginf=1e6,
    ).clamp_min(0.0).numpy()
    if cost_np.shape != (int(source_np.shape[0]), int(target_np.shape[0])):
        raise ValueError(
            "OT cost shape must match source/target lengths, got "
            f"{cost_np.shape} for {source_np.shape[0]}x{target_np.shape[0]}."
        )

    if solver_name == "emd":
        try:
            import ot
        except Exception as exc:
            raise ImportError("DGST-T exact EMD solver requires POT.") from exc
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan_array = ot.emd(source_np, target_np, cost_np)
        warning_text = " ".join(str(item.message).lower() for item in caught)
        plan = torch.tensor(plan_array, dtype=torch.float64)
        if (
            "infeasible" not in warning_text
            and "simplex" not in warning_text
            and tuple(plan.shape) == tuple(cost_np.shape)
            and torch.isfinite(plan).all()
        ):
            return float((plan * torch.tensor(cost_np, dtype=torch.float64)).sum().item()), plan
        return _wasserstein_1_exact(
            torch.tensor(source_np, dtype=torch.float64),
            torch.tensor(target_np, dtype=torch.float64),
            torch.tensor(cost_np, dtype=torch.float64),
            solver="linprog",
        )

    try:
        from scipy.optimize import linprog
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires scipy.") from exc

    source_size = int(source_np.shape[0])
    target_size = int(target_np.shape[0])
    result = linprog(
        c=cost_np.reshape(-1),
        A_eq=_transport_constraint_matrix(source_size, target_size),
        b_eq=np.concatenate([source_np, target_np[:-1]]),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        source_np, target_np = _balance_ot_marginals(source_np, target_np, slack=1e-10)
        result = linprog(
            c=cost_np.reshape(-1),
            A_eq=_transport_constraint_matrix(source_size, target_size),
            b_eq=np.concatenate([source_np, target_np[:-1]]),
            bounds=(0.0, None),
            method="highs",
        )
    if not result.success:
        try:
            import ot
            plan_array = ot.emd(source_np, target_np, cost_np)
            plan = torch.tensor(plan_array, dtype=torch.float64)
            return float((plan * torch.tensor(cost_np, dtype=torch.float64)).sum().item()), plan
        except Exception as exc:
            raise RuntimeError(f"DGST-T exact OT solver failed: {result.message}") from exc
    plan = torch.tensor(result.x, dtype=torch.float64).reshape(source_size, target_size)
    return float(result.fun), plan


def _as_probability_numpy(values: torch.Tensor):
    import numpy as np

    arr = values.detach().cpu().double().numpy().astype(np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if total <= 0.0:
        arr.fill(1.0 / max(int(arr.size), 1))
    else:
        arr /= total
    arr[-1] += 1.0 - float(arr.sum())
    if arr[-1] < 0.0:
        arr[arr < 0.0] = 0.0
        arr /= max(float(arr.sum()), EPS)
        arr[-1] += 1.0 - float(arr.sum())
    return arr


def _balance_ot_marginals(source, target, *, slack: float = 1e-12):
    """Make the dropped-last-column OT constraints numerically feasible."""
    import numpy as np

    source = np.array(source, dtype=np.float64, copy=True)
    target = np.array(target, dtype=np.float64, copy=True)
    source[source < 0.0] = 0.0
    target[target < 0.0] = 0.0
    source /= max(float(source.sum()), EPS)
    target /= max(float(target.sum()), EPS)
    source[-1] += 1.0 - float(source.sum())

    if target.size == 1:
        target[0] = float(source.sum())
        return source, target

    prefix_sum = float(target[:-1].sum())
    source_sum = float(source.sum())
    max_prefix = max(0.0, source_sum - float(slack))
    if prefix_sum > max_prefix:
        target[:-1] *= max_prefix / max(prefix_sum, EPS)
    target[-1] = max(0.0, source_sum - float(target[:-1].sum()))
    target /= max(float(target.sum()), EPS)

    prefix_sum = float(target[:-1].sum())
    if prefix_sum > source_sum:
        target[:-1] *= source_sum / max(prefix_sum, EPS)
        target[-1] = 0.0
    return source, target


@lru_cache(maxsize=32)
def _transport_constraint_matrix(source_size: int, target_size: int):
    try:
        import numpy as np
        from scipy import sparse
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires numpy and scipy.") from exc

    m = int(source_size)
    n = int(target_size)
    row_count = m + n - 1
    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_idx in range(m):
        row_indices.extend([row_idx] * n)
        col_indices.extend([row_idx * n + col_idx for col_idx in range(n)])
    for col_idx in range(n - 1):
        row_indices.extend([m + col_idx] * m)
        col_indices.extend([row_idx * n + col_idx for row_idx in range(m)])

    data = np.ones(len(row_indices), dtype=np.float64)
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(row_count, m * n))
