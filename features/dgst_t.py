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


def compute_dgst_t(
    dgst_t_raw: dict[str, Any],
    *,
    tau: float = 0.07,
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
) -> dict[str, Any]:
    """Compute DGST-T layer features from raw wrapper captures."""
    support_states = dgst_t_raw["support_h_mid_states"]
    support_output_states = dgst_t_raw.get("support_output_states", support_states)
    prompt_confidence_max = dgst_t_raw.get(
        "prompt_logit_lens_max_confidence",
        dgst_t_raw["prompt_logit_lens_top3_confidence"],
    )

    return _compute_dgst_t_from_parts(
        source_ffn_states=_layer_tensors(dgst_t_raw["source_ffn_states"]),
        prediction_hidden_states=_layer_tensors(dgst_t_raw["prediction_hidden_states"]),
        support_h_mid_states=_layer_tensors(support_states),
        support_output_states=_layer_tensors(support_output_states),
        support_attentions=_layer_tensors(dgst_t_raw["support_attentions"]),
        semantic_probs=_layer_tensors(dgst_t_raw["semantic_probs"]),
        prompt_last_hidden_states=_layer_tensors(dgst_t_raw["prompt_last_hidden_states"]),
        prompt_mean_hidden_states=_layer_tensors(dgst_t_raw["prompt_mean_hidden_states"]),
        prompt_confidence_top3=_layer_tensors(dgst_t_raw["prompt_logit_lens_top3_confidence"]),
        prompt_confidence_max=_layer_tensors(prompt_confidence_max),
        support_positions=[int(pos) for pos in dgst_t_raw["support_positions"]],
        visual_start=int(dgst_t_raw["visual_start"]),
        visual_end=int(dgst_t_raw["visual_end"]),
        tau=tau,
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
    )


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
    tau: float = 0.07,
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
) -> list[dict[str, Any]]:
    """Compute DGST-T for several target tokens directly from shared captures."""
    from models.dgst_capture import (
        resolve_output_embedding_layer,
        resolve_prompt_positions,
        resolve_support_positions,
        target_probabilities_multi,
    )

    target_ids = [int(token_id) for token_id in target_token_ids]
    pred_positions = [int(position) for position in prediction_positions]
    if len(target_ids) != len(pred_positions):
        raise ValueError("target_token_ids and prediction_positions must have the same length.")
    if not target_ids:
        return []

    prompt_positions = resolve_prompt_positions(
        full_input_ids=full_input_ids,
        prompt_tokenized_length=prompt_tokenized_length,
        image_token_id=int(image_token_id),
        visual_start=int(visual_start),
        visual_end=int(visual_end),
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
    parts = [
        {
            "source_ffn_states": [],
            "prediction_hidden_states": [],
            "support_h_mid_states": [],
            "support_output_states": [],
            "support_attentions": [],
            "semantic_probs": [],
            "prompt_last_hidden_states": [],
            "prompt_mean_hidden_states": [],
            "prompt_logit_lens_top3_confidence": [],
            "prompt_logit_lens_max_confidence": [],
        }
        for _ in target_ids
    ]

    for capture in captures:
        h_mid = capture["h_mid"][0]
        o_ffn = capture["o_ffn"][0]
        layer_hidden = h_mid + o_ffn
        device = h_mid.device
        support_index = torch.tensor(support_positions, dtype=torch.long, device=device)
        prompt_index = torch.tensor(prompt_positions, dtype=torch.long, device=device)

        support_states = h_mid.index_select(0, support_index)
        support_output_states = layer_hidden.index_select(0, support_index)
        prompt_states = layer_hidden.index_select(0, prompt_index)

        support_semantic_all = target_probabilities_multi(
            output_layer=output_layer,
            states=support_states,
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
            part["prediction_hidden_states"].append(layer_hidden[int(prediction_position), :])
            part["support_h_mid_states"].append(support_states)
            part["support_output_states"].append(support_output_states)
            part["support_attentions"].append(support_attention)
            part["semantic_probs"].append(support_semantic_all[:, target_offset])
            part["prompt_last_hidden_states"].append(prompt_last_state)
            part["prompt_mean_hidden_states"].append(prompt_mean_state)
            part["prompt_logit_lens_top3_confidence"].append(prompt_conf_top3_all[target_offset])
            part["prompt_logit_lens_max_confidence"].append(prompt_conf_max_all[target_offset])

    results: list[dict[str, Any]] = []
    for part in parts:
        results.append(
            _compute_dgst_t_from_parts(
                source_ffn_states=part["source_ffn_states"],
                prediction_hidden_states=part["prediction_hidden_states"],
                support_h_mid_states=part["support_h_mid_states"],
                support_output_states=part["support_output_states"],
                support_attentions=part["support_attentions"],
                semantic_probs=part["semantic_probs"],
                prompt_last_hidden_states=part["prompt_last_hidden_states"],
                prompt_mean_hidden_states=part["prompt_mean_hidden_states"],
                prompt_confidence_top3=part["prompt_logit_lens_top3_confidence"],
                prompt_confidence_max=part["prompt_logit_lens_max_confidence"],
                support_positions=[int(position) for position in support_positions],
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                tau=tau,
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
            )
        )
    return results


def _compute_dgst_t_from_parts(
    *,
    source_ffn_states: Sequence[torch.Tensor],
    prediction_hidden_states: Sequence[torch.Tensor],
    support_h_mid_states: Sequence[torch.Tensor],
    support_output_states: Sequence[torch.Tensor],
    support_attentions: Sequence[torch.Tensor],
    semantic_probs: Sequence[torch.Tensor],
    prompt_last_hidden_states: Sequence[torch.Tensor],
    prompt_mean_hidden_states: Sequence[torch.Tensor],
    prompt_confidence_top3: Sequence[torch.Tensor],
    prompt_confidence_max: Sequence[torch.Tensor],
    support_positions: Sequence[int],
    visual_start: int,
    visual_end: int,
    tau: float,
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
) -> dict[str, Any]:
    layer_count = len(source_ffn_states)
    if layer_count == 0:
        raise ValueError("DGST-T requires at least one captured layer.")
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

    layer_stats = []
    risk_per_layer = []
    risk_topmass_085_per_layer = []
    risk_capped_topmass_085_per_layer = []
    prompt_last_cosine_per_layer = []
    prompt_mean_cosine_per_layer = []
    target_visual_hidden_cosine_per_layer = []
    target_visual_prompt_hidden_cosine_per_layer = []
    target_visual_hidden_cosine_capped_topmass_085_per_layer = []
    target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer = []
    prompt_confidence_top3_per_layer = []
    prompt_confidence_max_per_layer = []
    context_confidence_per_layer = []
    context_confidence_max_prompt_per_layer = []

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
        )
        attention_dist = _renormalize(layer_support_attentions)
        target_dist = _renormalize(attention_dist * layer_semantic_probs)

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
        prompt_conf_top3 = _scalar(prompt_confidence_top3[layer_idx])
        prompt_conf_max = _scalar(prompt_confidence_max[layer_idx])
        context_confidence = float(prompt_conf_top3 * target_visual_hidden_cosine)
        context_confidence_max_prompt = float(prompt_conf_max * target_visual_hidden_cosine)

        risk_per_layer.append(float(transport_risk))
        if transport_risk_topmass_085 is not None:
            risk_topmass_085_per_layer.append(float(transport_risk_topmass_085))
        if transport_risk_capped_topmass_085 is not None:
            risk_capped_topmass_085_per_layer.append(float(transport_risk_capped_topmass_085))
        prompt_last_cosine_per_layer.append(prompt_last_cosine)
        prompt_mean_cosine_per_layer.append(prompt_mean_cosine)
        target_visual_hidden_cosine_per_layer.append(float(target_visual_hidden_cosine))
        target_visual_prompt_hidden_cosine_per_layer.append(float(target_visual_prompt_hidden_cosine))
        target_visual_hidden_cosine_capped_topmass_085_per_layer.append(
            float(target_visual_hidden_cosine_capped_topmass_085)
        )
        target_visual_prompt_hidden_cosine_capped_topmass_085_per_layer.append(
            float(target_visual_prompt_hidden_cosine_capped_topmass_085)
        )
        prompt_confidence_top3_per_layer.append(float(prompt_conf_top3))
        prompt_confidence_max_per_layer.append(float(prompt_conf_max))
        context_confidence_per_layer.append(context_confidence)
        context_confidence_max_prompt_per_layer.append(context_confidence_max_prompt)
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
            "target_hidden_capped_topmass_085_visual_cosine": float(
                target_visual_hidden_cosine_capped_topmass_085
            ),
            "target_hidden_capped_topmass_085_visual_prompt_cosine": float(
                target_visual_prompt_hidden_cosine_capped_topmass_085
            ),
            "context_confidence": context_confidence,
            "context_confidence_max_prompt": context_confidence_max_prompt,
            "support_size": int(len(support_positions)),
            "selected_support_size": int(support.numel()),
        }
        if transport_risk_topmass_085 is not None and topmass_support is not None:
            stats["transport_risk_topmass_085"] = float(transport_risk_topmass_085)
            stats["selected_support_size_topmass_085"] = int(topmass_support.numel())
        if transport_risk_capped_topmass_085 is not None and capped_topmass_support is not None:
            stats["transport_risk_capped_topmass_085"] = float(transport_risk_capped_topmass_085)
            stats["selected_support_size_capped_topmass_085"] = int(capped_topmass_support.numel())
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
    return result


def _layer_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value[index] for index in range(int(value.shape[0]))]
    return list(value)


def _scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


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
) -> torch.Tensor:
    update = source_update.float()
    states = support_states.float()
    scores = F.cosine_similarity(update.unsqueeze(0), states, dim=-1)
    return torch.softmax(scores / max(float(tau), 1e-6), dim=-1)


def _renormalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / max(int(values.numel()), 1))
    return values / total


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
) -> float:
    if support.numel() == 0:
        return 0.0
    local_source = _renormalize(source_dist.index_select(0, support))
    local_target = _renormalize(target_dist.index_select(0, support))
    local_states = support_states.index_select(0, support)
    local_semantic = semantic_probs.index_select(0, support)

    distance = _cosine_distance_matrix(local_states)
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
    return (1.0 - cosine).clamp_min(0.0)


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
) -> torch.Tensor:
    mode = str(cost_mode).strip().lower()
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
