"""Raw DGST-T capture helpers for decoder-only LVLM backbones."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F


def resolve_decoder_layers(model: Any):
    """Return the LM decoder layers for LLaVA, Qwen2.5-VL, or InternVL."""
    language_model = getattr(model, "language_model", None)
    candidates = []
    if language_model is not None:
        candidates.extend(
            [
                getattr(getattr(language_model, "model", None), "layers", None),
                getattr(language_model, "layers", None),
            ]
        )
    candidates.extend(
        [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(model, "layers", None),
        ]
    )
    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        candidates.extend(
            [
                getattr(getattr(decoder, "model", None), "layers", None),
                getattr(decoder, "layers", None),
            ]
        )

    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError(f"Cannot resolve decoder layers for {type(model).__name__}.")


def resolve_output_embedding_layer(model: Any):
    for module in (model, getattr(model, "language_model", None)):
        get_output_embeddings = getattr(module, "get_output_embeddings", None)
        if callable(get_output_embeddings):
            layer = get_output_embeddings()
            if layer is not None and getattr(layer, "weight", None) is not None:
                return layer
    raise ValueError("Model does not expose output embeddings.")


def resolve_input_embedding_layer(model: Any):
    for module in (model, getattr(model, "language_model", None)):
        get_input_embeddings = getattr(module, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            layer = get_input_embeddings()
            if layer is not None and getattr(layer, "weight", None) is not None:
                return layer
    raise ValueError("Model does not expose input embeddings.")


def run_forward_with_dgst_captures(
    model: Any,
    *,
    output_hidden_states: bool = True,
    **forward_kwargs,
):
    """Run a forward pass while capturing decoder attention and FFN updates."""
    layers = resolve_decoder_layers(model)
    captures: list[dict[str, Any]] = [
        {"h_prev": None, "o_attn": None, "attn_weights": None, "o_ffn": None}
        for _ in range(len(layers))
    ]
    handles = []

    def layer_pre_hook(index: int):
        def hook(_module, args):
            captures[index]["h_prev"] = args[0].detach()

        return hook

    def attention_hook(index: int):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                captures[index]["o_attn"] = output[0].detach()
                if len(output) > 1 and output[1] is not None:
                    captures[index]["attn_weights"] = output[1].detach()
            else:
                captures[index]["o_attn"] = output.detach()

        return hook

    def mlp_hook(index: int):
        def hook(_module, _args, output):
            captures[index]["o_ffn"] = output.detach()

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(layer_pre_hook(index)))
        handles.append(_layer_attention_module(layer).register_forward_hook(attention_hook(index)))
        handles.append(_layer_mlp_module(layer).register_forward_hook(mlp_hook(index)))

    try:
        with torch.no_grad():
            outputs = model(
                **forward_kwargs,
                output_attentions=True,
                output_hidden_states=bool(output_hidden_states),
                return_dict=True,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    attentions = getattr(outputs, "attentions", None)
    for index, capture in enumerate(captures):
        if capture["h_prev"] is None or capture["o_attn"] is None or capture["o_ffn"] is None:
            raise RuntimeError("DGST-T hooks did not capture h_prev, o_attn, and o_ffn for every layer.")
        if capture["attn_weights"] is None and attentions is not None and index < len(attentions):
            capture["attn_weights"] = attentions[index]
        if capture["attn_weights"] is None:
            raise RuntimeError("DGST-T requires attention weights; load the model with eager attention.")

        target_device = capture["o_attn"].device
        for key in ("h_prev", "o_ffn", "attn_weights"):
            if capture[key].device != target_device:
                capture[key] = capture[key].to(target_device)
        capture["h_mid"] = capture["h_prev"] + capture["o_attn"]

    return outputs, captures


def build_dgst_t_raw(
    *,
    model: Any,
    full_input_ids: Sequence[int],
    prompt_tokenized_length: int,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    image_token_id: int,
    target_token_id: int,
    prediction_position: int,
    support_scope: str = "visual_prompt",
    semantic_chunk_size: int = 64,
) -> dict[str, Any]:
    """Build the raw tensors consumed by features.dgst_t.compute_dgst_t."""
    return build_dgst_t_raw_batch(
        model=model,
        full_input_ids=full_input_ids,
        prompt_tokenized_length=prompt_tokenized_length,
        captures=captures,
        visual_start=visual_start,
        visual_end=visual_end,
        image_token_id=image_token_id,
        target_token_ids=[int(target_token_id)],
        prediction_positions=[int(prediction_position)],
        support_scope=support_scope,
        semantic_chunk_size=semantic_chunk_size,
    )[0]


def build_dgst_t_raw_batch(
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
) -> list[dict[str, Any]]:
    """Build per-token DGST-T raw tensors from one shared decoder forward."""
    target_ids = [int(token_id) for token_id in target_token_ids]
    pred_positions = [int(position) for position in prediction_positions]
    if len(target_ids) != len(pred_positions):
        raise ValueError("target_token_ids and prediction_positions must have the same length.")
    if not target_ids:
        return []

    prompt_positions = resolve_prompt_positions(
        full_input_ids=full_input_ids,
        prompt_tokenized_length=prompt_tokenized_length,
        image_token_id=image_token_id,
        visual_start=visual_start,
        visual_end=visual_end,
    )
    support_positions = resolve_support_positions(
        visual_start=visual_start,
        visual_end=visual_end,
        prompt_positions=prompt_positions,
        support_scope=support_scope,
    )
    if not support_positions:
        raise ValueError("DGST-T support is empty.")
    if not prompt_positions:
        raise ValueError("DGST-T prompt positions are empty.")

    output_layer = resolve_output_embedding_layer(model)
    input_layer = resolve_input_embedding_layer(model)
    hidden_size = int(captures[0]["h_mid"].shape[-1])
    target_embeddings = [
        embedding_for_token(
            input_layer=input_layer,
            output_layer=output_layer,
            target_token_id=token_id,
            hidden_size=hidden_size,
            device=captures[0]["h_mid"].device,
        )
        for token_id in target_ids
    ]

    raw_parts = [
        {
            "source_ffn_states": [],
            "prediction_hidden_states": [],
            "support_h_mid_states": [],
            "support_attentions": [],
            "semantic_probs": [],
            "prompt_last_hidden_states": [],
            "prompt_mean_hidden_states": [],
            "prompt_logit_lens_top3_confidence": [],
        }
        for _ in target_ids
    ]

    for layer_offset, capture in enumerate(captures):
        h_mid = capture["h_mid"][0]
        o_ffn = capture["o_ffn"][0]
        layer_hidden = h_mid + o_ffn
        device = h_mid.device
        support_index = torch.tensor(support_positions, dtype=torch.long, device=device)
        prompt_index = torch.tensor(prompt_positions, dtype=torch.long, device=device)

        support_states = h_mid.index_select(0, support_index)
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
        prompt_conf_all = torch.topk(prompt_probs_all.float(), k=top_k, dim=0).values.mean(dim=0)
        prompt_last_state = prompt_states[-1].detach().cpu()
        prompt_mean_state = prompt_states.mean(dim=0).detach().cpu()
        support_states_cpu = support_states.detach().cpu()

        for target_offset, prediction_position in enumerate(pred_positions):
            attention_row = capture["attn_weights"][0, :, int(prediction_position), :]
            support_attention = attention_row.index_select(
                1, support_index.to(attention_row.device)
            ).mean(dim=0)

            part = raw_parts[target_offset]
            part["source_ffn_states"].append(o_ffn[int(prediction_position), :].detach().cpu())
            part["prediction_hidden_states"].append(layer_hidden[int(prediction_position), :].detach().cpu())
            part["support_h_mid_states"].append(support_states_cpu)
            part["support_attentions"].append(support_attention.detach().cpu())
            part["semantic_probs"].append(support_semantic_all[:, target_offset].detach().cpu())
            part["prompt_last_hidden_states"].append(prompt_last_state)
            part["prompt_mean_hidden_states"].append(prompt_mean_state)
            part["prompt_logit_lens_top3_confidence"].append(
                prompt_conf_all[target_offset].detach().cpu()
            )

    raws: list[dict[str, Any]] = []
    for target_offset, part in enumerate(raw_parts):
        raws.append(
            {
                "target_token_id": int(target_ids[target_offset]),
                "prediction_position": int(pred_positions[target_offset]),
                "visual_start": int(visual_start),
                "visual_end": int(visual_end),
                "support_scope": str(support_scope),
                "support_positions": [int(position) for position in support_positions],
                "prompt_positions": [int(position) for position in prompt_positions],
                "source_ffn_states": torch.stack(part["source_ffn_states"], dim=0),
                "prediction_hidden_states": torch.stack(part["prediction_hidden_states"], dim=0),
                "support_h_mid_states": torch.stack(part["support_h_mid_states"], dim=0),
                "support_attentions": torch.stack(part["support_attentions"], dim=0),
                "semantic_probs": torch.stack(part["semantic_probs"], dim=0),
                "prompt_last_hidden_states": torch.stack(part["prompt_last_hidden_states"], dim=0),
                "prompt_mean_hidden_states": torch.stack(part["prompt_mean_hidden_states"], dim=0),
                "prompt_logit_lens_top3_confidence": torch.stack(
                    part["prompt_logit_lens_top3_confidence"], dim=0
                ),
                "target_embedding": target_embeddings[target_offset].detach().cpu(),
            }
        )
    return raws


def resolve_prompt_positions(
    *,
    full_input_ids: Sequence[int],
    prompt_tokenized_length: int,
    image_token_id: int,
    visual_start: int,
    visual_end: int,
) -> list[int]:
    """Map tokenized prompt positions to merged decoder positions."""
    ids = [int(token_id) for token_id in full_input_ids]
    image_positions = [index for index, token_id in enumerate(ids) if token_id == int(image_token_id)]
    prompt_len = int(prompt_tokenized_length)
    visual_count = int(visual_end) - int(visual_start)

    if len(image_positions) == 1 and visual_count > 1:
        image_pos = image_positions[0]
        positions = []
        for pos in range(prompt_len):
            if pos == image_pos:
                continue
            positions.append(pos if pos < image_pos else pos + visual_count - 1)
        return [int(position) for position in positions]

    visual_set = set(range(int(visual_start), int(visual_end)))
    return [
        int(pos)
        for pos in range(prompt_len)
        if pos not in visual_set and (pos >= len(ids) or ids[pos] != int(image_token_id))
    ]


def resolve_support_positions(
    *,
    visual_start: int,
    visual_end: int,
    prompt_positions: Sequence[int],
    support_scope: str,
) -> list[int]:
    scope = str(support_scope).strip().lower()
    if scope not in {"visual", "visual_prompt"}:
        raise ValueError("DGST-T support_scope must be 'visual' or 'visual_prompt'.")
    visual_positions = list(range(int(visual_start), int(visual_end)))
    if scope == "visual":
        return visual_positions
    return sorted(dict.fromkeys(visual_positions + [int(pos) for pos in prompt_positions]))


def target_probabilities(
    *,
    output_layer: Any,
    states: torch.Tensor,
    target_token_id: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute p(target_token | state) without keeping full-vocab logits."""
    return target_probabilities_multi(
        output_layer=output_layer,
        states=states,
        target_token_ids=[int(target_token_id)],
        chunk_size=chunk_size,
    ).squeeze(-1)


def target_probabilities_multi(
    *,
    output_layer: Any,
    states: torch.Tensor,
    target_token_ids: Sequence[int],
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute p(target_token | state) for several targets with one logsumexp pass."""
    weight = output_layer.weight
    bias = getattr(output_layer, "bias", None)
    token_ids = [int(token_id) for token_id in target_token_ids]
    if not token_ids:
        return torch.empty(*states.shape[:-1], 0, dtype=torch.float32, device=states.device)

    valid_columns = [
        (column, token_id)
        for column, token_id in enumerate(token_ids)
        if 0 <= token_id < int(weight.shape[0])
    ]
    if not valid_columns:
        return torch.zeros(*states.shape[:-1], len(token_ids), dtype=torch.float32, device=states.device)

    probs = []
    flat_states = states.reshape(-1, states.shape[-1])
    for start in range(0, int(flat_states.shape[0]), max(1, int(chunk_size))):
        chunk = flat_states[start : start + int(chunk_size)].to(device=weight.device, dtype=weight.dtype)
        logits = F.linear(chunk, weight, bias.to(dtype=weight.dtype) if bias is not None else None)
        log_denominator = torch.logsumexp(logits.float(), dim=-1)
        chunk_probs = torch.zeros(
            int(chunk.shape[0]),
            len(token_ids),
            dtype=torch.float32,
            device=states.device,
        )
        valid_token_index = torch.tensor(
            [token_id for _column, token_id in valid_columns],
            dtype=torch.long,
            device=logits.device,
        )
        target_logits = logits.index_select(dim=1, index=valid_token_index).float()
        valid_probs = torch.exp((target_logits - log_denominator.unsqueeze(1)).clamp(max=0.0)).to(states.device)
        for valid_offset, (column, _token_id) in enumerate(valid_columns):
            chunk_probs[:, column] = valid_probs[:, valid_offset]
        probs.append(chunk_probs)
    return torch.cat(probs, dim=0).reshape(*states.shape[:-1], len(token_ids)).float()


def merged_position_for_tokenized_position(
    *,
    full_input_ids: Sequence[int],
    tokenized_position: int,
    image_token_id: int,
    visual_token_count: int,
) -> int:
    image_positions = [
        index for index, token_id in enumerate(full_input_ids) if int(token_id) == int(image_token_id)
    ]
    if len(image_positions) == 1 and int(visual_token_count) > 1:
        image_position = int(image_positions[0])
        position = int(tokenized_position)
        return position if position < image_position else position + int(visual_token_count) - 1
    return int(tokenized_position)


def pre_token_prediction_positions(
    *,
    full_input_ids: Sequence[int],
    prompt_tokenized_length: int,
    response_token_indices: Sequence[int],
    image_token_id: int,
    visual_token_count: int,
    prompt_positions: Sequence[int],
) -> list[int]:
    if not prompt_positions:
        raise ValueError("Cannot resolve pre-token prediction positions without prompt positions.")

    positions: list[int] = []
    answer_positions = [
        merged_position_for_tokenized_position(
            full_input_ids=full_input_ids,
            tokenized_position=int(prompt_tokenized_length) + offset,
            image_token_id=image_token_id,
            visual_token_count=visual_token_count,
        )
        for offset in range(
            max([int(index) for index in response_token_indices], default=-1) + 1
        )
    ]
    for response_index in response_token_indices:
        index = int(response_index)
        if index <= 0:
            positions.append(int(prompt_positions[-1]))
        else:
            positions.append(int(answer_positions[index - 1]))
    return positions


def hidden_states_from_captures(
    captures: Sequence[dict[str, Any]],
    *,
    token_position: int,
    visual_start: int,
    visual_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_states = []
    patch_states = []
    for capture in captures:
        layer_hidden = capture["h_mid"] + capture["o_ffn"]
        token_states.append(layer_hidden[0, int(token_position), :])
        patch_states.append(layer_hidden[0, int(visual_start) : int(visual_end), :])
    return torch.stack(token_states, dim=0), torch.stack(patch_states, dim=0)


def embedding_for_token(
    *,
    input_layer: Any,
    output_layer: Any,
    target_token_id: int,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    token_id = int(target_token_id)
    for layer in (input_layer, output_layer):
        weight = getattr(layer, "weight", None)
        if weight is None:
            continue
        if 0 <= token_id < int(weight.shape[0]) and int(weight.shape[-1]) == int(hidden_size):
            return weight[token_id].detach().to(device=device, dtype=torch.float32)
    raise ValueError(f"Cannot resolve target token embedding for token_id={target_token_id}.")


def _layer_attention_module(layer: Any):
    for name in ("self_attn", "attention"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError(f"Cannot resolve attention module for decoder layer {type(layer).__name__}.")


def _layer_mlp_module(layer: Any):
    for name in ("mlp", "feed_forward"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise ValueError(f"Cannot resolve MLP module for decoder layer {type(layer).__name__}.")
