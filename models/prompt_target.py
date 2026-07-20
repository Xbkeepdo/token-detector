"""Causal extraction for a token that occurs inside a multimodal prompt.

The generated-response API intentionally enforces that every requested target
is an actual saved response token.  VQA object probes are different: their
target occurs in the question itself.  This module keeps that use case on a
separate path and derives both the target ID and the predictor row from the
real, context-tokenized multimodal prompt.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from features.dgst_t import compute_dgst_t_batch_from_captures
from models.base_wrapper import (
    AttentionRequirement,
    ExtractionRequirements,
    ModelOutput,
    PromptTargetAlignment,
    PromptTargetRequest,
)
from models.dgst_capture import (
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    merged_position_for_tokenized_position,
    run_forward_with_dgst_captures,
    run_forward_with_layer_hidden_captures,
)
from models.prompt_support import resolve_prompt_support_positions
from utils.token_alignment import (
    TokenAlignmentError,
    build_response_token_offsets,
    token_indices_for_char_span,
    validate_token_surface,
)


def resolve_prompt_target_alignment(
    *,
    tokenizer: Any,
    full_input_ids: Sequence[int],
    request: PromptTargetRequest,
    image_token_id: int,
    visual_token_count: int,
) -> PromptTargetAlignment:
    """Locate a prompt surface in actual model input IDs and resolve ``j-1``.

    Image placeholder IDs are omitted only while reconstructing visible text;
    every returned token position remains in the original model-input index
    space.  This avoids the standalone-word BPE/SentencePiece mismatch.
    """

    input_ids = [int(value) for value in full_input_ids]
    if not input_ids:
        raise ValueError("cannot align a prompt target in empty model input IDs")

    visible_positions = [
        index
        for index, token_id in enumerate(input_ids)
        if int(token_id) != int(image_token_id)
    ]
    visible_ids = [input_ids[index] for index in visible_positions]
    if not visible_ids:
        raise ValueError("prompt contains no non-image tokens")
    rendered_text = _decode(tokenizer, visible_ids)
    rendered_start, rendered_end = _resolve_rendered_target_span(
        rendered_text=rendered_text,
        request=request,
    )
    try:
        visible_offsets = build_response_token_offsets(
            tokenizer,
            visible_ids,
            rendered_text,
        )
        visible_token_indices = token_indices_for_char_span(
            visible_offsets,
            rendered_start,
            rendered_end,
        )
        validate_token_surface(
            tokenizer=tokenizer,
            response_token_ids=visible_ids,
            token_indices=visible_token_indices,
            offsets=visible_offsets,
            caption=rendered_text,
            char_start=rendered_start,
            char_end=rendered_end,
        )
    except TokenAlignmentError as exc:
        raise ValueError(f"failed to align prompt target: {exc}") from exc
    if not visible_token_indices:
        raise ValueError(
            f"no actual prompt token overlaps target {request.target_text!r}"
        )

    tokenized_span = tuple(
        int(visible_positions[index]) for index in visible_token_indices
    )
    if list(tokenized_span) != list(
        range(tokenized_span[0], tokenized_span[-1] + 1)
    ):
        raise ValueError(
            "prompt target token span is not contiguous after image-token mapping: "
            f"{tokenized_span}"
        )
    target_tokenized_position = int(tokenized_span[0])
    actual_target_id = int(input_ids[target_tokenized_position])
    if request.expected_target_token_id is not None and actual_target_id != int(
        request.expected_target_token_id
    ):
        raise ValueError(
            "expected prompt target token ID does not match the actual "
            "context-tokenized prompt ID: "
            f"expected={int(request.expected_target_token_id)}, "
            f"actual={actual_target_id}, position={target_tokenized_position}"
        )

    target_expanded_position = merged_position_for_tokenized_position(
        full_input_ids=input_ids,
        tokenized_position=target_tokenized_position,
        image_token_id=int(image_token_id),
        visual_token_count=int(visual_token_count),
    )
    prediction_position = int(target_expanded_position) - 1
    if prediction_position < 0:
        raise ValueError("the first prompt token has no causal predictor row")
    return PromptTargetAlignment(
        target_tokenized_position=target_tokenized_position,
        target_expanded_position=int(target_expanded_position),
        prediction_position=prediction_position,
        target_token_id=actual_target_id,
        tokenized_span=tokenized_span,
        rendered_char_start=int(rendered_start),
        rendered_char_end=int(rendered_end),
    )


def truncate_multimodal_inputs(
    inputs: Mapping[str, Any],
    *,
    tokenized_end: int,
) -> dict[str, Any]:
    """Keep the strict causal prefix ``input_ids[:tokenized_end]``.

    Image tensors and layout metadata remain unchanged.  Sequence-shaped masks
    are sliced with the IDs; cached position tensors are discarded so each
    model reconstructs them for the shortened sequence.
    """

    end = int(tokenized_end)
    result = dict(inputs)
    input_ids = result.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("prompt-target inputs require rank-2 input_ids")
    full_length = int(input_ids.shape[-1])
    if end <= 0 or end > full_length:
        raise ValueError(
            f"invalid prompt-target prefix end {end} for input length {full_length}"
        )
    result["input_ids"] = input_ids[..., :end]
    for key in ("attention_mask", "token_type_ids"):
        value = result.get(key)
        if torch.is_tensor(value) and int(value.shape[-1]) == full_length:
            result[key] = value[..., :end]
    for stale_key in ("position_ids", "cache_position", "labels", "inputs_embeds"):
        result.pop(stale_key, None)
    return result


def extract_prompt_target_from_inputs(
    *,
    wrapper: Any,
    inputs: Mapping[str, Any],
    full_input_ids: Sequence[int],
    alignment: PromptTargetAlignment,
    visual_start: int,
    visual_end: int,
    image_token_id: int,
    visual_grid: Optional[Tuple[int, int]],
    cfg_dgst_t: Optional[dict[str, Any]],
    requirements: Optional[ExtractionRequirements],
    model_name: str,
    support_scope: str,
    semantic_chunk_size_default: int = 64,
) -> ModelOutput:
    """Run the shortened true prompt and build every feature from row ``j-1``."""

    expected_full_ids = [int(value) for value in full_input_ids]
    supplied_ids = [
        int(value) for value in inputs["input_ids"][0].detach().cpu().tolist()
    ]
    if supplied_ids != expected_full_ids:
        raise ValueError("full_input_ids do not match the supplied multimodal inputs")
    target_position = int(alignment.target_tokenized_position)
    if target_position >= len(expected_full_ids) or expected_full_ids[
        target_position
    ] != int(alignment.target_token_id):
        raise ValueError(
            "prompt-target alignment does not point to the actual target ID in inputs"
        )

    requirements_were_explicit = requirements is not None
    resolved_requirements = wrapper.resolve_extraction_requirements(
        requirements,
        dgst_enabled=cfg_dgst_t is not None,
    )
    prefix_inputs = truncate_multimodal_inputs(
        inputs,
        tokenized_end=alignment.target_tokenized_position,
    )
    prefix_ids = [
        int(value) for value in prefix_inputs["input_ids"][0].detach().cpu().tolist()
    ]
    use_dgst = bool(
        resolved_requirements.dgst_capture and cfg_dgst_t is not None
    )
    layer_outputs = None
    if use_dgst:
        out, captures = run_forward_with_dgst_captures(
            wrapper.model,
            output_hidden_states=False,
            retain_attention_updates=(
                str(cfg_dgst_t.get("target_gate_mode", "four_gate"))
                .strip()
                .lower()
                != "four_gate"
                or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
            ),
            **prefix_inputs,
        )
    elif resolved_requirements.needs_hidden_states:
        captures = None
        out, layer_outputs = run_forward_with_layer_hidden_captures(
            wrapper.model,
            output_attentions=resolved_requirements.needs_attention_weights,
            capture_all_layers=(
                resolved_requirements.token_hidden_states
                or resolved_requirements.patch_hidden_states
            ),
            **prefix_inputs,
        )
    else:
        captures = None
        with torch.inference_mode():
            out = wrapper.model(
                **prefix_inputs,
                output_attentions=resolved_requirements.needs_attention_weights,
                output_hidden_states=resolved_requirements.needs_hidden_states,
                return_dict=True,
                use_cache=False,
            )

    seq_len = int(out.logits.shape[1])
    prediction_position = seq_len - 1
    if prediction_position != int(alignment.prediction_position):
        raise RuntimeError(
            "prompt-target expanded predictor row mismatch: "
            f"expected={alignment.prediction_position}, actual={prediction_position}; "
            "the image-token expansion mapping is inconsistent"
        )
    if int(visual_end) > seq_len:
        raise RuntimeError(
            "prompt target occurs before the complete visual span; object-token "
            "prediction requires the image to precede the object in the prompt"
        )

    compact_profile = bool(
        cfg_dgst_t is not None
        and cfg_dgst_t.get("feature_output_profile")
        in {"costvariant_vv", "gate_comparison_vv", "four_gate_vv"}
    )
    keep_attention = (
        (not compact_profile or requirements_were_explicit)
        and resolved_requirements.attention is not AttentionRequirement.NONE
    )
    keep_hidden = (not compact_profile or requirements_were_explicit) and (
        resolved_requirements.token_hidden_states
        or resolved_requirements.patch_hidden_states
    )
    if keep_attention:
        if out.attentions is None or not out.attentions:
            raise RuntimeError(
                f"{model_name} prompt-target attention was requested but the "
                "model returned no eager attention weights"
            )
        text_to_patch_attn, text_to_text_attn = _attention_at_position(
            out.attentions,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            seq_len=seq_len,
            prediction_position=prediction_position,
        )
        if resolved_requirements.attention is AttentionRequirement.HEAD_MEAN:
            text_to_patch_attn = text_to_patch_attn.mean(dim=1, keepdim=True)
            text_to_text_attn = text_to_text_attn.mean(dim=1, keepdim=True)
    else:
        text_to_patch_attn = torch.empty(0)
        text_to_text_attn = torch.empty(0)
    text_to_patch_attn = text_to_patch_attn.cpu()
    text_to_text_attn = text_to_text_attn.cpu()
    out.attentions = None

    if keep_hidden:
        if captures is not None:
            token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                captures,
                token_position=prediction_position,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
            )
        elif layer_outputs is not None:
            token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                layer_outputs,
                token_position=prediction_position,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
            )
        elif out.hidden_states is not None:
            token_hidden_states, patch_hidden_states = _hidden_at_position(
                out.hidden_states,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                prediction_position=prediction_position,
            )
        else:
            raise RuntimeError(
                f"{model_name} prompt-target hidden states were requested but not returned"
            )
        if not resolved_requirements.token_hidden_states:
            token_hidden_states = torch.empty(0, device=patch_hidden_states.device)
        if not resolved_requirements.patch_hidden_states:
            patch_hidden_states = torch.empty(0, device=token_hidden_states.device)
    else:
        token_hidden_states = torch.empty(0)
        patch_hidden_states = torch.empty(0)
    token_hidden_states = token_hidden_states.cpu()
    patch_hidden_states = patch_hidden_states.cpu()

    row_logits = out.logits[0, prediction_position]
    pred_token_id = int(row_logits.argmax().item())
    token_logits = (
        row_logits.float().cpu() if resolved_requirements.logits else None
    )
    pred_token_str = wrapper.tokenizer.decode(
        [pred_token_id],
        skip_special_tokens=False,
    )
    out.logits = None

    dgst_result = None
    if use_dgst:
        support_prompt_positions = resolve_prompt_support_positions(
            tokenizer=wrapper.tokenizer,
            full_input_ids=prefix_ids,
            prompt_tokenized_length=len(prefix_ids),
            image_token_id=int(image_token_id),
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            cfg_dgst_t=cfg_dgst_t,
            model_name=model_name,
        )
        results = compute_dgst_t_batch_from_captures(
            model=wrapper.model,
            full_input_ids=prefix_ids,
            prompt_tokenized_length=len(prefix_ids),
            captures=captures,
            visual_start=int(visual_start),
            visual_end=int(visual_end),
            image_token_id=int(image_token_id),
            target_token_ids=[int(alignment.target_token_id)],
            prediction_positions=[prediction_position],
            support_scope=str(support_scope),
            semantic_chunk_size=int(
                cfg_dgst_t.get("semantic_chunk_size", semantic_chunk_size_default)
            ),
            prompt_positions_override=support_prompt_positions,
            **_dgst_options(cfg_dgst_t),
            release_layer_captures=True,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"expected one prompt-target DGST result, received {len(results)}"
            )
        dgst_result = results[0]

    return ModelOutput(
        token_id=pred_token_id,
        token_str=str(pred_token_str),
        text_to_patch_attn=text_to_patch_attn,
        text_to_text_attn=text_to_text_attn,
        token_hidden_states=token_hidden_states,
        patch_hidden_states=patch_hidden_states,
        response_token_idx=int(alignment.target_tokenized_position),
        token_logits=token_logits,
        dgst_t_raw=None,
        dgst_t_result=dgst_result,
        visual_grid=visual_grid if resolved_requirements.visual_layout else None,
        response_hidden_states=None,
        baseline_capture={
            "prompt_target": True,
            "target_token_id": int(alignment.target_token_id),
            "target_tokenized_position": int(alignment.target_tokenized_position),
            "target_expanded_position": int(alignment.target_expanded_position),
            "prediction_position": int(prediction_position),
            "visual_start": int(visual_start),
            "visual_end": int(visual_end),
            "attention_requirement": resolved_requirements.attention.value,
            "prompt_target_alignment": {
                "target_tokenized_position": int(alignment.target_tokenized_position),
                "target_expanded_position": int(alignment.target_expanded_position),
                "prediction_position": int(prediction_position),
                "target_token_id": int(alignment.target_token_id),
                "tokenized_span": [int(value) for value in alignment.tokenized_span],
                "rendered_char_start": int(alignment.rendered_char_start),
                "rendered_char_end": int(alignment.rendered_char_end),
            },
        },
    )


def _resolve_rendered_target_span(
    *, rendered_text: str, request: PromptTargetRequest
) -> tuple[int, int]:
    prompt_candidates = [str(request.prompt)]
    without_image = str(request.prompt).replace("<image>", "")
    if without_image not in prompt_candidates:
        prompt_candidates.append(without_image)

    if request.target_char_start is not None:
        start_in_prompt = int(request.target_char_start)
        end_in_prompt = int(request.target_char_end)
        # First prefer an exact embedding of the raw prompt in decoded input.
        for candidate in prompt_candidates:
            if not candidate:
                continue
            prompt_starts = _all_occurrences(rendered_text, candidate)
            if not prompt_starts:
                continue
            selected = min(int(request.occurrence), len(prompt_starts) - 1)
            candidate_delta = 0
            if candidate != request.prompt:
                image_pos = str(request.prompt).find("<image>")
                if image_pos >= 0 and start_in_prompt > image_pos:
                    candidate_delta = -len("<image>")
            start = prompt_starts[selected] + start_in_prompt + candidate_delta
            end = start + len(request.target_text)
            if rendered_text[start:end] == request.target_text:
                return start, end

        # Tokenizer decoders may normalize runs of spaces/newlines. Preserve
        # the supplied question-relative character span by mapping the whole
        # normalized prompt into the normalized rendered template. This must
        # stay fail-closed: falling back to the first surface occurrence can
        # silently select an earlier CLEVR reference object when the queried
        # object word is repeated later in the question.
        for candidate in prompt_candidates:
            if not candidate:
                continue
            candidate_delta = 0
            if candidate != request.prompt:
                image_pos = str(request.prompt).find("<image>")
                if image_pos >= 0:
                    if end_in_prompt <= image_pos:
                        pass
                    elif start_in_prompt >= image_pos + len("<image>"):
                        candidate_delta = -len("<image>")
                    else:
                        continue
            candidate_start = start_in_prompt + candidate_delta
            candidate_end = end_in_prompt + candidate_delta
            mapped = _map_normalized_prompt_span(
                rendered_text=rendered_text,
                prompt_text=candidate,
                prompt_char_start=candidate_start,
                prompt_char_end=candidate_end,
            )
            if mapped is not None:
                mapped_start, mapped_end = mapped
                if _normalize_whitespace(
                    rendered_text[mapped_start:mapped_end]
                ) == _normalize_whitespace(str(request.target_text)):
                    return mapped_start, mapped_end

        raise ValueError(
            "could not map the exact question object span into the rendered "
            "multimodal prompt; refusing first-occurrence fallback: "
            f"target={request.target_text!r}, span="
            f"[{start_in_prompt}, {end_in_prompt})"
        )

    matches = _all_occurrences(rendered_text, str(request.target_text))
    occurrence = int(request.occurrence)
    if occurrence >= len(matches):
        raise ValueError(
            "could not locate requested target surface in the rendered prompt: "
            f"target={request.target_text!r}, occurrence={occurrence}, "
            f"matches={len(matches)}"
        )
    start = int(matches[occurrence])
    return start, start + len(request.target_text)


def _map_normalized_prompt_span(
    *,
    rendered_text: str,
    prompt_text: str,
    prompt_char_start: int,
    prompt_char_end: int,
) -> tuple[int, int] | None:
    normalized_rendered, rendered_map = _normalize_whitespace_with_map(rendered_text)
    normalized_prompt, prompt_map = _normalize_whitespace_with_map(prompt_text)
    if not normalized_prompt:
        return None
    prompt_starts = _all_occurrences(normalized_rendered, normalized_prompt)
    if not prompt_starts:
        return None
    target_indices = [
        index
        for index, original_index in enumerate(prompt_map)
        if int(prompt_char_start) <= original_index < int(prompt_char_end)
    ]
    if not target_indices:
        return None
    relative_start = int(target_indices[0])
    relative_end = int(target_indices[-1]) + 1
    for prompt_start in prompt_starts:
        normalized_start = int(prompt_start) + relative_start
        normalized_end = int(prompt_start) + relative_end
        if normalized_end > len(rendered_map):
            continue
        rendered_start = int(rendered_map[normalized_start])
        rendered_end = int(rendered_map[normalized_end - 1]) + 1
        return rendered_start, rendered_end
    return None


def _normalize_whitespace(text: str) -> str:
    normalized, _mapping = _normalize_whitespace_with_map(text)
    return normalized


def _normalize_whitespace_with_map(text: str) -> tuple[str, list[int]]:
    characters: list[str] = []
    original_positions: list[int] = []
    in_whitespace = False
    for index, character in enumerate(str(text)):
        if character.isspace():
            if not in_whitespace:
                characters.append(" ")
                original_positions.append(int(index))
            in_whitespace = True
            continue
        characters.append(character)
        original_positions.append(int(index))
        in_whitespace = False
    return "".join(characters), original_positions

def _all_occurrences(text: str, query: str) -> list[int]:
    if not query:
        return []
    result: list[int] = []
    start = 0
    while True:
        found = str(text).find(str(query), start)
        if found < 0:
            return result
        result.append(int(found))
        start = int(found) + 1


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    ids = [int(value) for value in token_ids]
    try:
        return str(
            tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(ids, skip_special_tokens=True))


def _attention_at_position(
    attentions: Sequence[torch.Tensor],
    *,
    visual_start: int,
    visual_end: int,
    seq_len: int,
    prediction_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    visual = set(range(int(visual_start), int(visual_end)))
    text_indices = [
        index
        for index in range(int(seq_len))
        if index not in visual and index != int(prediction_position)
    ]
    patch_layers = []
    text_layers = []
    for layer_attention in attentions:
        row = layer_attention[0, :, int(prediction_position), :]
        index = torch.tensor(text_indices, dtype=torch.long, device=row.device)
        patch_layers.append(row[:, int(visual_start) : int(visual_end)])
        text_layers.append(row.index_select(-1, index))
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _hidden_at_position(
    hidden_states: Sequence[torch.Tensor],
    *,
    visual_start: int,
    visual_end: int,
    prediction_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.stack(
        [state[0, int(prediction_position), :] for state in hidden_states[1:]],
        dim=0,
    )
    patches = torch.stack(
        [
            state[0, int(visual_start) : int(visual_end), :]
            for state in hidden_states[1:]
        ],
        dim=0,
    )
    return token, patches


def _dgst_options(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tau": float(cfg.get("tau", 0.07)),
        "source_distribution_mode": cfg.get("source_distribution_mode", "softmax"),
        "transport_top_k": int(cfg.get("transport_top_k", 64)),
        "cost_mode": cfg.get("cost_mode", "direct"),
        "lambda_d": float(cfg.get("lambda_d", 1.0)),
        "lambda_s": float(cfg.get("lambda_s", 1.0)),
        "lambda_t": float(cfg.get("lambda_t", 1.0)),
        "lambda_int": float(cfg.get("lambda_int", 1.0)),
        "baseline_layers": int(cfg.get("baseline_layers", 10)),
        "risk_start_layer": int(cfg.get("risk_start_layer", 15)),
        "alpha": float(cfg.get("alpha", 2.0)),
        "ot_solver": cfg.get("ot_solver", "linprog"),
        "atarget_visual_top_k": int(cfg.get("atarget_visual_top_k", 32)),
        "topmass_alpha": float(cfg.get("topmass_085_alpha", 0.85)),
        "capped_topmass_alpha": float(cfg.get("capped_topmass_085_alpha", 0.85)),
        "capped_topmass_min_k": int(cfg.get("capped_topmass_085_min_k", 32)),
        "capped_topmass_max_k": int(cfg.get("capped_topmass_085_max_k", 64)),
        "compute_topmass_085": bool(cfg.get("compute_topmass_085", True)),
        "compute_capped_topmass_085": bool(cfg.get("compute_capped_topmass_085", True)),
        "target_gate_mode": cfg.get("target_gate_mode", "four_gate"),
        "relative_vll_mad_epsilon": float(cfg.get("relative_vll_mad_epsilon", 1e-6)),
        "relative_vll_logit_source": cfg.get("relative_vll_logit_source", "h_mid"),
        "relative_cost_mode": cfg.get("relative_cost_mode"),
        "relative_cost_modes": cfg.get("relative_cost_modes"),
        "relative_cost_state_modes": cfg.get("relative_cost_state_modes"),
        "relative_cost_update_lambdas": cfg.get("relative_cost_update_lambdas"),
        "relative_barrier_lambda": float(cfg.get("relative_barrier_lambda", 1.0)),
        "relative_barrier_margin": float(cfg.get("relative_barrier_margin", 0.5)),
        "relative_barrier_max": float(cfg.get("relative_barrier_max", 3.0)),
        "source_modes": cfg.get("source_modes"),
        "target_attention_gammas": cfg.get("target_attention_gammas"),
        "target_attention_epsilon": float(cfg.get("target_attention_epsilon", 1e-12)),
        "compute_ffn_injection_features": bool(
            cfg.get("compute_ffn_injection_features", False)
        ),
        "ffn_injection_evidence_top_k": int(
            cfg.get("ffn_injection_evidence_top_k", 32)
        ),
        "ffn_injection_evidence_rank": int(
            cfg.get("ffn_injection_evidence_rank", 8)
        ),
        "ffn_injection_eps": float(cfg.get("ffn_injection_eps", 1e-12)),
        "compute_dual_scope": bool(
            cfg.get("dgst_t_dual_scope", cfg.get("compute_dual_scope", False))
        ),
        "four_gate_methods": cfg.get("four_gate_methods"),
        "four_gate_cost_modes": cfg.get("cost_modes"),
        "four_gate_support_modes": cfg.get("support_modes"),
    }
