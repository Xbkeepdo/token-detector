"""MetaToken feature equations used by the paper baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch


@dataclass(frozen=True)
class MetaTokenFeatures:
    vector: np.ndarray
    names: tuple[str, ...]

    def as_payload(self) -> dict:
        return {
            "vector": self.vector.astype(np.float32, copy=False),
            "num_attention_heads": sum(
                name.startswith("mean_absolute_visual_attention_head_")
                for name in self.names
            ),
            "feature_names": list(self.names),
            "probability_difference_adaptation": "greedy_target_probability",
        }


def compute_metatoken_features(
    *,
    response_token_ids: Sequence[int],
    token_logits: Union[torch.Tensor, np.ndarray],
    visual_attention: Union[torch.Tensor, np.ndarray],
    span_start: int,
    span_end: int,
    occurrence_count: int,
    length_penalty: float = 1.0,
    attention_layer: Union[int, str] = -1,
) -> MetaTokenFeatures:
    """Compute the exact ``10 + H`` MetaToken feature vector.

    ``token_logits[t]`` must be the next-token vocabulary logits that produced
    ``response_token_ids[t]``.  ``span_end`` is inclusive.  In the greedy
    decoding protocol used by *Beyond the Global Scores*, Eq. 11's probability
    difference is identically zero, so its slot stores the target-token
    probability as specified by that paper's supplement.

    The original feature cardinality contains heads but not layers.  Therefore
    a ``[L,H,P]`` attention tensor uses the final decoder layer by default;
    callers can select another layer or pass ``attention_layer='mean'``.
    """

    ids = torch.as_tensor(response_token_ids, dtype=torch.long).reshape(-1)
    logits = torch.as_tensor(token_logits).detach().float()
    if logits.ndim != 2:
        raise ValueError(f"token_logits must have shape [T,V], got {tuple(logits.shape)}")
    if ids.numel() != logits.shape[0]:
        raise ValueError(
            f"response ids/logits length mismatch: {ids.numel()} vs {logits.shape[0]}"
        )
    if logits.shape[1] < 2:
        raise ValueError("MetaToken requires a vocabulary with at least two entries")
    if ids.numel() == 0:
        raise ValueError("response_token_ids cannot be empty")
    if int(ids.min()) < 0 or int(ids.max()) >= logits.shape[1]:
        raise ValueError("response_token_ids contain an id outside token_logits vocabulary")

    start, end = int(span_start), int(span_end)
    if start < 0 or end < start or end >= ids.numel():
        raise ValueError(f"Invalid inclusive span [{start}, {end}] for T={ids.numel()}")
    if int(occurrence_count) < 1:
        raise ValueError("occurrence_count must be at least one")
    if float(length_penalty) < 0:
        raise ValueError("length_penalty must be non-negative")

    selected_attention = _select_attention_layer(visual_attention, attention_layer)
    head_attention = selected_attention.abs().mean(dim=-1)

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    rows = torch.arange(ids.numel())
    generated_log_probs = log_probs[rows, ids]

    object_log_probability = generated_log_probs[start : end + 1].sum()
    cumulative_log_probability = generated_log_probs[: end + 1].sum()
    generated_length = max(1, end + 1)
    sequence_score = cumulative_log_probability / (
        float(generated_length) ** float(length_penalty)
    )

    start_log_probs = log_probs[start]
    start_probs = probs[start]
    log_probability_variance = start_log_probs.var(unbiased=False)
    normalized_entropy = -(
        start_probs * start_log_probs
    ).sum() / np.log(float(start_probs.numel()))
    top2 = torch.topk(start_probs, k=2).values
    variation_ratio = 1.0 - top2[0]
    probability_margin = variation_ratio + top2[1]
    target_token_probability = start_probs[ids[start]]

    names = (
        "relative_position",
        "absolute_occurrence",
        *tuple(
            f"mean_absolute_visual_attention_head_{head}"
            for head in range(head_attention.numel())
        ),
        "object_log_probability",
        "cumulative_log_probability",
        "sequence_score",
        "log_probability_variance",
        "normalized_entropy",
        "variation_ratio",
        "probability_margin",
        "target_token_probability",
    )
    values = torch.cat(
        (
            torch.tensor(
                [start / float(ids.numel()), float(occurrence_count)],
                device=logits.device,
            ),
            head_attention,
            torch.stack(
                (
                    object_log_probability,
                    cumulative_log_probability,
                    sequence_score,
                    log_probability_variance,
                    normalized_entropy,
                    variation_ratio,
                    probability_margin,
                    target_token_probability,
                )
            ),
        )
    )
    vector = values.detach().cpu().numpy().astype(np.float32, copy=False)
    if vector.size != 10 + head_attention.numel():
        raise AssertionError("MetaToken feature cardinality must be 10 + H")
    if not np.isfinite(vector).all():
        raise ValueError("MetaToken features contain a non-finite value")
    return MetaTokenFeatures(vector=vector, names=names)


def compute_metatoken_features_from_stats(
    *,
    response_token_ids: Sequence[int],
    visual_attention: Union[torch.Tensor, np.ndarray],
    span_start: int,
    span_end: int,
    occurrence_count: int,
    response_target_logprobs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    response_target_probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    response_logprob_variances: Union[torch.Tensor, np.ndarray, Sequence[float]],
    response_normalized_entropies: Union[torch.Tensor, np.ndarray, Sequence[float]],
    response_top1_probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    response_top2_probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    length_penalty: float = 1.0,
    attention_layer: Union[int, str] = -1,
) -> MetaTokenFeatures:
    """Build the same ``10+H`` vector from compact per-step statistics.

    This avoids retaining ``[T,V]`` logits.  The six input vectors are enough
    to exactly reconstruct every MetaToken term used by the greedy-decoding
    adaptation in *Beyond the Global Scores*.
    """

    response_length = len(response_token_ids)
    start, end = int(span_start), int(span_end)
    if response_length <= 0 or start < 0 or end < start or end >= response_length:
        raise ValueError(
            f"Invalid inclusive span [{start}, {end}] for T={response_length}"
        )
    if int(occurrence_count) < 1:
        raise ValueError("occurrence_count must be at least one")
    if float(length_penalty) < 0:
        raise ValueError("length_penalty must be non-negative")

    raw_statistics = {
        "response_target_logprobs": response_target_logprobs,
        "response_target_probs": response_target_probs,
        "response_logprob_variances": response_logprob_variances,
        "response_normalized_entropies": response_normalized_entropies,
        "response_top1_probs": response_top1_probs,
        "response_top2_probs": response_top2_probs,
    }
    statistics = {
        name: torch.as_tensor(value).detach().float().reshape(-1)
        for name, value in raw_statistics.items()
    }
    for name, value in statistics.items():
        if value.numel() != response_length:
            raise ValueError(
                f"{name} must have T={response_length} entries, got {value.numel()}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("response_target_probs", "response_top1_probs", "response_top2_probs"):
        value = statistics[name]
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name} must contain probabilities in [0,1]")

    head_attention = _select_attention_layer(
        visual_attention, attention_layer
    ).abs().mean(dim=-1).cpu()
    target_logprobs = statistics["response_target_logprobs"]
    object_log_probability = target_logprobs[start : end + 1].sum()
    cumulative_log_probability = target_logprobs[: end + 1].sum()
    sequence_score = cumulative_log_probability / (
        float(max(1, end + 1)) ** float(length_penalty)
    )
    variation_ratio = 1.0 - statistics["response_top1_probs"][start]
    probability_margin = variation_ratio + statistics["response_top2_probs"][start]
    names = (
        "relative_position",
        "absolute_occurrence",
        *tuple(
            f"mean_absolute_visual_attention_head_{head}"
            for head in range(head_attention.numel())
        ),
        "object_log_probability",
        "cumulative_log_probability",
        "sequence_score",
        "log_probability_variance",
        "normalized_entropy",
        "variation_ratio",
        "probability_margin",
        "target_token_probability",
    )
    values = torch.cat(
        (
            torch.tensor(
                [start / float(response_length), float(occurrence_count)],
                dtype=torch.float32,
            ),
            head_attention,
            torch.stack(
                (
                    object_log_probability,
                    cumulative_log_probability,
                    sequence_score,
                    statistics["response_logprob_variances"][start],
                    statistics["response_normalized_entropies"][start],
                    variation_ratio,
                    probability_margin,
                    statistics["response_target_probs"][start],
                )
            ).cpu(),
        )
    )
    vector = values.numpy().astype(np.float32, copy=False)
    if vector.size != 10 + head_attention.numel():
        raise AssertionError("MetaToken feature cardinality must be 10 + H")
    if not np.isfinite(vector).all():
        raise ValueError("MetaToken features contain a non-finite value")
    return MetaTokenFeatures(vector=vector, names=names)


def _select_attention_layer(
    visual_attention: Union[torch.Tensor, np.ndarray],
    layer: Union[int, str],
) -> torch.Tensor:
    attention = torch.as_tensor(visual_attention).detach().float()
    if attention.ndim == 2:
        selected = attention
    elif attention.ndim == 3:
        if isinstance(layer, str):
            if layer.strip().lower() != "mean":
                raise ValueError("attention_layer string must be 'mean'")
            selected = attention.mean(dim=0)
        else:
            index = int(layer)
            if index < 0:
                index += attention.shape[0]
            if not 0 <= index < attention.shape[0]:
                raise IndexError(
                    f"attention_layer {layer} is outside L={attention.shape[0]}"
                )
            selected = attention[index]
    else:
        raise ValueError(
            "visual_attention must have shape [H,P] or [L,H,P], got "
            f"{tuple(attention.shape)}"
        )
    if selected.shape[-1] == 0 or not torch.isfinite(selected).all():
        raise ValueError("visual_attention is empty or contains non-finite values")
    return selected
