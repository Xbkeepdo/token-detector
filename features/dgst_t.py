"""DGST-T continuous risk features.

This module keeps only the core DGST-T signals requested for the TGD port:
source_dist from FFN update, target_dist from attention times semantic
probability, Wasserstein/OT transport risk, prompt cosine features, and
context-confidence features.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import torch
import torch.nn.functional as F

EPS = 1e-12
TOPMASS_ALPHA_085 = 0.85
CAPPED_TOPMASS_MIN_K = 32
CAPPED_TOPMASS_MAX_K = 64
RELATIVE_VLL_MAD_EPSILON = 1e-6


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
    support_states = dgst_t_raw["support_h_mid_states"]
    support_output_states = dgst_t_raw.get("support_output_states", support_states)
    prompt_confidence_max = dgst_t_raw.get(
        "prompt_logit_lens_max_confidence",
        dgst_t_raw["prompt_logit_lens_top3_confidence"],
    )

    result = _compute_dgst_t_from_parts(
        source_ffn_states=_layer_tensors(dgst_t_raw["source_ffn_states"]),
        source_attn_states=_optional_layer_tensors(dgst_t_raw.get("source_attn_states")),
        prediction_hidden_states=_layer_tensors(dgst_t_raw["prediction_hidden_states"]),
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
) -> list[dict[str, Any]]:
    """Compute DGST-T for several target tokens directly from shared captures."""
    from models.dgst_capture import (
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

    for capture in captures:
        h_mid = capture["h_mid"][0]
        o_attn = capture["o_attn"][0]
        o_ffn = capture["o_ffn"][0]
        layer_hidden = h_mid + o_ffn
        device = h_mid.device
        support_index = torch.tensor(support_positions, dtype=torch.long, device=device)
        prompt_index = torch.tensor(prompt_positions, dtype=torch.long, device=device)

        support_states = h_mid.index_select(0, support_index)
        support_relative_states = (
            apply_decoder_final_norm(final_norm_layer, support_states)
            if final_norm_layer is not None
            else support_states
        )
        support_output_states = layer_hidden.index_select(0, support_index)
        prompt_states = layer_hidden.index_select(0, prompt_index)

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
            attention_row = capture["attn_weights"][0, :, int(prediction_position), :]
            support_attention = attention_row.index_select(
                1,
                support_index.to(attention_row.device),
            ).mean(dim=0)
            support_attention = support_attention.to(device=device, dtype=torch.float32)

            part = parts[target_offset]
            part["source_ffn_states"].append(o_ffn[int(prediction_position), :])
            part["source_attn_states"].append(o_attn[int(prediction_position), :])
            part["prediction_hidden_states"].append(layer_hidden[int(prediction_position), :])
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
        result["dgst_t_relative_vll_logit_source"] = relative_source
        result["dgst_t_dual_scope"] = bool(compute_dual_scope)
        results.append(result)
    return results


def _compute_dgst_t_from_parts(
    *,
    source_ffn_states: Sequence[torch.Tensor],
    source_attn_states: Sequence[torch.Tensor] | None = None,
    prediction_hidden_states: Sequence[torch.Tensor],
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
    compute_relative_vll = gate_mode in {"relative_vll", "dual"}
    if compute_relative_vll and relative_vll_logits is None:
        raise ValueError("target_gate_mode requires relative_vll_logits, but they are missing.")
    for name, values in {
        "prediction_hidden_states": prediction_hidden_states,
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
    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer = []
    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer = []
    visual_prompt_relative_vll_target_visual_mass_per_layer = []
    visual_prompt_relative_vll_target_prompt_mass_per_layer = []
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
    if compute_relative_vll and emit_relative_cost_state_fields:
        for mode in emit_relative_cost_modes:
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
        layer_support_states = support_h_mid_states[layer_idx].float()
        layer_support_output_states = support_output_states[layer_idx].float()
        layer_source_ffn = source_ffn_states[layer_idx].float()
        layer_prediction_hidden = prediction_hidden_states[layer_idx].float()
        layer_semantic_probs = semantic_probs[layer_idx].to(layer_support_states.device).float()
        layer_support_attentions = support_attentions[layer_idx].to(layer_support_states.device).float()

        source_dist = _source_distribution(
            source_update=layer_source_ffn,
            support_states=layer_support_states,
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
        target_dist_relative_vll = None
        semantic_gate_relative_vll = None
        relative_barrier_vll = None
        relative_vll_stats = None
        target_dist_visual_prompt_relative_vll = None
        semantic_gate_visual_prompt_relative_vll = None
        visual_prompt_relative_barrier_vll = None
        visual_prompt_relative_vll_stats = None
        if compute_relative_vll:
            layer_relative_logits = relative_vll_logits[layer_idx].to(layer_support_states.device).float()
            if not skip_relative_vll_visual_branch:
                (
                    target_dist_relative_vll,
                    semantic_gate_relative_vll,
                    relative_barrier_vll,
                    relative_vll_stats,
                ) = _relative_vll_target_distribution(
                    attention_dist=attention_dist,
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
            if has_prompt_support:
                (
                    target_dist_visual_prompt_relative_vll,
                    semantic_gate_visual_prompt_relative_vll,
                    visual_prompt_relative_barrier_vll,
                    visual_prompt_relative_vll_stats,
                ) = _relative_vll_target_distribution(
                    attention_dist=attention_dist,
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

        ffn_injection = None
        ffn_gate_ratio = None
        if compute_ffn_injection_features:
            layer_source_attn = (
                source_attn_states[layer_idx].float()
                if source_attn_states is not None
                else None
            )
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
                target_dist_visual=target_dist_relative_vll,
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
                    target_dist_delta, semantic_gate_delta, barrier_delta, target_stats = (
                        _relative_vll_target_distribution(
                            attention_dist=attention_dist,
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
                        target_dist_delta,
                        transport_top_k,
                    )
                    delta_risk = _transport_risk_on_support(
                        source_dist=source_dist_delta,
                        target_dist=target_dist_delta,
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
                            target_dist=target_dist_delta,
                            support_positions=support_positions,
                            visual_start=visual_start,
                            visual_end=visual_end,
                            top_k=atarget_visual_top_k,
                        )
                    else:
                        delta_cos = cosine_fn(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=target_dist_delta,
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
                            target_dist_delta,
                            capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                        delta_risk_cap = _transport_risk_on_support(
                            source_dist=source_dist_delta,
                            target_dist=target_dist_delta,
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
                                target_dist=target_dist_delta,
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
                                target_dist=target_dist_delta,
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
        if compute_relative_vll:
            if target_dist_relative_vll is not None:
                relative_vll_support = _topk_union_indices(
                    source_dist,
                    target_dist_relative_vll,
                    transport_top_k,
                )
                relative_vll_risks = _transport_risks_by_cost(
                    source_dist=source_dist,
                    target_dist=target_dist_relative_vll,
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
                if emit_relative_cost_fields:
                    for mode, value in relative_vll_risks.items():
                        relative_cost_series[mode]["risk"].append(float(value))
                if emit_relative_cost_state_fields:
                    relative_vll_state_risks = _transport_risks_by_cost_state(
                        source_dist=source_dist,
                        target_dist=target_dist_relative_vll,
                        support_states=layer_support_states,
                        support_output_states=layer_support_output_states,
                        semantic_probs=semantic_gate_relative_vll,
                        support=relative_vll_support,
                        cost_modes=emit_relative_cost_modes,
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
                        target_dist_relative_vll,
                        capped_topmass_alpha,
                        min_k=capped_topmass_min_k,
                        max_k=capped_topmass_max_k,
                    )
                    relative_vll_capped_risks = _transport_risks_by_cost(
                        source_dist=source_dist,
                        target_dist=target_dist_relative_vll,
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
            if target_dist_visual_prompt_relative_vll is not None:
                visual_prompt_relative_vll_target_visual_mass = (
                    _target_mass_for_scope(
                        target_dist=target_dist_visual_prompt_relative_vll,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        scope="visual",
                    )
                )
                visual_prompt_relative_vll_target_prompt_mass = (
                    _target_mass_for_scope(
                        target_dist=target_dist_visual_prompt_relative_vll,
                        support_positions=support_positions,
                        visual_start=visual_start,
                        visual_end=visual_end,
                        scope="prompt",
                    )
                )
                visual_prompt_relative_vll_support = _topk_union_indices(
                    source_dist,
                    target_dist_visual_prompt_relative_vll,
                    transport_top_k,
                )
                visual_prompt_relative_vll_risks = _transport_risks_by_cost(
                    source_dist=source_dist,
                    target_dist=target_dist_visual_prompt_relative_vll,
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
                if emit_relative_cost_fields:
                    for mode, value in visual_prompt_relative_vll_risks.items():
                        relative_cost_series[mode]["vp_risk"].append(float(value))
                if emit_relative_cost_state_fields:
                    visual_prompt_relative_vll_state_risks = _transport_risks_by_cost_state(
                        source_dist=source_dist,
                        target_dist=target_dist_visual_prompt_relative_vll,
                        support_states=layer_support_states,
                        support_output_states=layer_support_output_states,
                        semantic_probs=semantic_gate_visual_prompt_relative_vll,
                        support=visual_prompt_relative_vll_support,
                        cost_modes=emit_relative_cost_modes,
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
                        target_dist_visual_prompt_relative_vll,
                        capped_topmass_alpha,
                        min_k=capped_topmass_min_k,
                        max_k=capped_topmass_max_k,
                    )
                    visual_prompt_relative_vll_capped_risks = _transport_risks_by_cost(
                        source_dist=source_dist,
                        target_dist=target_dist_visual_prompt_relative_vll,
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
        target_visual_prompt_hidden_cosine_visual_prompt_relative_vll = None
        target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 = None
        if compute_relative_vll:
            if target_dist_relative_vll is not None:
                target_visual_hidden_cosine_relative_vll = _target_hidden_topk_visual_cosine(
                    target_hidden=layer_prediction_hidden,
                    support_output_states=layer_support_output_states,
                    target_dist=target_dist_relative_vll,
                    support_positions=support_positions,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    top_k=atarget_visual_top_k,
                )
                if compute_capped_topmass_085:
                    target_visual_hidden_cosine_relative_vll_capped_topmass_085 = (
                        _target_hidden_capped_topmass_visual_cosine(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=target_dist_relative_vll,
                            support_positions=support_positions,
                            visual_start=visual_start,
                            visual_end=visual_end,
                            alpha=capped_topmass_alpha,
                            min_k=capped_topmass_min_k,
                            max_k=capped_topmass_max_k,
                        )
                    )
            if target_dist_visual_prompt_relative_vll is not None:
                target_visual_prompt_hidden_cosine_visual_prompt_relative_vll = _target_hidden_topk_support_cosine(
                    target_hidden=layer_prediction_hidden,
                    support_output_states=layer_support_output_states,
                    target_dist=target_dist_visual_prompt_relative_vll,
                    top_k=atarget_visual_top_k,
                )
                if compute_capped_topmass_085:
                    target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 = (
                        _target_hidden_capped_topmass_support_cosine(
                            target_hidden=layer_prediction_hidden,
                            support_output_states=layer_support_output_states,
                            target_dist=target_dist_visual_prompt_relative_vll,
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
        if transport_risk_relative_vll_capped_topmass_085 is not None:
            risk_relative_vll_capped_topmass_085_per_layer.append(
                float(transport_risk_relative_vll_capped_topmass_085)
            )
        if transport_risk_visual_prompt_relative_vll is not None:
            risk_visual_prompt_relative_vll_per_layer.append(float(transport_risk_visual_prompt_relative_vll))
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
        if target_visual_hidden_cosine_relative_vll_capped_topmass_085 is not None:
            target_visual_hidden_cosine_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_hidden_cosine_relative_vll_capped_topmass_085)
            )
        if target_visual_prompt_hidden_cosine_visual_prompt_relative_vll is not None:
            target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_per_layer.append(
                float(target_visual_prompt_hidden_cosine_visual_prompt_relative_vll)
            )
        if visual_prompt_relative_vll_target_visual_mass is not None:
            visual_prompt_relative_vll_target_visual_mass_per_layer.append(
                float(visual_prompt_relative_vll_target_visual_mass.detach())
            )
        if visual_prompt_relative_vll_target_prompt_mass is not None:
            visual_prompt_relative_vll_target_prompt_mass_per_layer.append(
                float(visual_prompt_relative_vll_target_prompt_mass.detach())
            )
        if target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 is not None:
            target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085_per_layer.append(
                float(target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085)
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
            stats["visual_prompt_relative_vll_target_visual_mass"] = float(
                visual_prompt_relative_vll_target_visual_mass.detach()
            )
            stats["visual_prompt_relative_vll_target_prompt_mass"] = float(
                visual_prompt_relative_vll_target_prompt_mass.detach()
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
        if visual_prompt_relative_vll_stats is not None:
            stats.update(visual_prompt_relative_vll_stats)
        if ffn_injection is not None:
            stats.update(ffn_injection)
        if delta_layer_stats:
            stats.update(delta_layer_stats)
        layer_stats.append(stats)

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
        result["dgst_t_target_visual_hidden_cosine_relative_vll_per_layer"] = torch.tensor(
            target_visual_hidden_cosine_relative_vll_per_layer,
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
            result["dgst_t_visual_prompt_relative_vll_target_visual_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_target_visual_mass_per_layer,
                    dtype=torch.float32,
                )
            )
            result["dgst_t_visual_prompt_relative_vll_target_prompt_mass_per_layer"] = (
                torch.tensor(
                    visual_prompt_relative_vll_target_prompt_mass_per_layer,
                    dtype=torch.float32,
                )
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
        ):
            result[key] = value
    if visual_result:
        result["dgst_t_dual_scope_vv_source_target"] = "visual"
        result["dgst_t_dual_scope_vp_source_target"] = "visual_prompt"


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
    if mode in {"legacy", "legacy_prob", "prob", "softmax_prob"}:
        return "legacy_prob"
    if mode in {"relative_vll", "relative", "relative_logit"}:
        return "relative_vll"
    if mode == "dual":
        return "dual"
    raise ValueError("DGST-T target_gate_mode must be 'legacy_prob', 'relative_vll', or 'dual'.")


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
    if mode in {"direct", "decomposed"}:
        return _normalize_legacy_cost_mode(mode)
    raise ValueError(
        "relative_cost_mode must be one of: geo, target_barrier_geo, "
        "symmetric_barrier_geo, target_additive_barrier_geo, "
        "source_additive_barrier_geo, two_end_additive_barrier_geo, "
        "direct, decomposed, or inherit."
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
        else:
            raise ValueError("DGST-T source_modes entries must be 'legacy_ffn' or 'delta_src'.")
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


def _relative_vll_target_distribution(
    *,
    attention_dist: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    logits = torch.nan_to_num(target_logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if logits.numel() != attention_dist.numel():
        raise ValueError(
            "relative_vll logits must align with support positions, got "
            f"{int(logits.numel())} logits and {int(attention_dist.numel())} attention values."
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
    z = (candidate_logits - median) / (mad + max(float(epsilon), EPS))
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
        attention_dist.float().clamp_min(0.0) + max(float(attention_epsilon), 0.0),
        gamma,
    )
    weighted = attention_weight * semantic_gate
    denominator = weighted.sum()
    if denominator <= EPS:
        target_dist = torch.zeros_like(weighted, dtype=torch.float32)
        target_dist.index_fill_(0, candidate_index, 1.0 / max(int(candidate_index.numel()), 1))
    else:
        target_dist = weighted / denominator

    stats = {
        f"{stat_prefix}_logit_median": float(median.item()),
        f"{stat_prefix}_logit_mad": float(mad.item()),
        f"{stat_prefix}_gate_mean": float(candidate_gate.mean().item()),
        f"{stat_prefix}_gate_max": float(candidate_gate.max().item()),
        f"{stat_prefix}_barrier_mean": float(candidate_barrier.mean().item()),
        f"{stat_prefix}_barrier_max": float(candidate_barrier.max().item()),
        f"{stat_prefix}_target_denominator": float(denominator.item()),
        f"{stat_prefix}_attention_gamma": float(gamma),
    }
    return target_dist, semantic_gate, relative_barrier, stats


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
    indices = _support_indices_for_scope(
        support_positions=support_positions,
        visual_start=visual_start,
        visual_end=visual_end,
        scope=scope,
        device=target_dist.device,
    )
    if indices.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=target_dist.device)
    return target_dist.float().index_select(0, indices).sum()


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
    k_half = min(max(int(top_k // 2), 1), int(source.numel()))
    src_idx = torch.topk(source, k=k_half).indices
    tgt_idx = torch.topk(target, k=k_half).indices
    return torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)


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


def _cosine_distance_matrix(states: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(states.float(), dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(0, 1))
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

    if solver_name == "emd":
        try:
            import ot
        except Exception as exc:
            raise ImportError("DGST-T exact EMD solver requires POT.") from exc
        plan_array = ot.emd(source_np, target_np, cost_np)
        plan = torch.tensor(plan_array, dtype=torch.float64)
        return float((plan * torch.tensor(cost_np, dtype=torch.float64)).sum().item()), plan

    try:
        from scipy.optimize import linprog
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires scipy.") from exc

    n = int(source_np.shape[0])
    result = linprog(
        c=cost_np.reshape(-1),
        A_eq=_transport_constraint_matrix(n),
        b_eq=np.concatenate([source_np, target_np[:-1]]),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        source_np, target_np = _balance_ot_marginals(source_np, target_np, slack=1e-10)
        result = linprog(
            c=cost_np.reshape(-1),
            A_eq=_transport_constraint_matrix(n),
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
    plan = torch.tensor(result.x, dtype=torch.float64).reshape(n, n)
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
def _transport_constraint_matrix(size: int):
    try:
        import numpy as np
        from scipy import sparse
    except Exception as exc:
        raise ImportError("DGST-T exact OT requires numpy and scipy.") from exc

    n = int(size)
    row_count = (2 * n) - 1
    row_indices: list[int] = []
    col_indices: list[int] = []

    for row_idx in range(n):
        row_indices.extend([row_idx] * n)
        col_indices.extend([row_idx * n + col_idx for col_idx in range(n)])
    for col_idx in range(n - 1):
        row_indices.extend([n + col_idx] * n)
        col_indices.extend([row_idx * n + col_idx for row_idx in range(n)])

    data = np.ones(len(row_indices), dtype=np.float64)
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(row_count, n * n))
