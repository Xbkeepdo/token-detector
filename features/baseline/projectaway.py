"""Training-free ProjectAway internal-confidence detector core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch


@dataclass(frozen=True)
class ProjectAwayConfidence:
    internal_confidence: float
    hallucination_score: float
    per_layer_internal_confidence: np.ndarray
    target_token_ids: tuple[int, ...]

    def as_payload(self) -> dict:
        return {
            "internal_confidence": float(self.internal_confidence),
            "hallucination_score": float(self.hallucination_score),
            "per_layer_internal_confidence": self.per_layer_internal_confidence.astype(
                np.float32, copy=False
            ),
            "target_token_ids": list(self.target_token_ids),
            "score_orientation": "higher_is_more_hallucinatory",
        }


@dataclass(frozen=True)
class ProjectAwayProbabilityCache:
    """Exact probabilities for a small union of target tokens in one image."""

    probabilities: torch.Tensor
    target_token_ids: tuple[int, ...]

    def confidence(self, target_token_ids: Sequence[int]) -> ProjectAwayConfidence:
        requested = tuple(dict.fromkeys(int(value) for value in target_token_ids))
        if not requested:
            raise ValueError("target_token_ids cannot be empty")
        lookup = {token_id: index for index, token_id in enumerate(self.target_token_ids)}
        missing = [token_id for token_id in requested if token_id not in lookup]
        if missing:
            raise KeyError(f"Target token ids were not precomputed: {missing}")
        indices = torch.tensor(
            [lookup[token_id] for token_id in requested],
            device=self.probabilities.device,
            dtype=torch.long,
        )
        selected = self.probabilities.index_select(-1, indices)
        per_layer = selected.amax(dim=(1, 2))
        confidence = float(per_layer.max().item())
        return ProjectAwayConfidence(
            internal_confidence=confidence,
            hallucination_score=1.0 - confidence,
            per_layer_internal_confidence=per_layer.detach().cpu().numpy().astype(
                np.float32, copy=False
            ),
            target_token_ids=requested,
        )


@torch.no_grad()
def compute_projectaway_internal_confidence(
    visual_hidden_states: Union[torch.Tensor, np.ndarray],
    unembedding_weight: Union[torch.Tensor, np.ndarray],
    target_token_ids: Sequence[int],
    *,
    unembedding_bias: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    vocab_chunk_size: int = 8192,
    row_chunk_size: int = 256,
) -> ProjectAwayConfidence:
    """Compute exact target softmax confidence without materializing ``L*P*V``.

    The maximum follows ProjectAway: target subtokens are compared across all
    visual patches and decoder layers.  Vocabulary chunks are combined with
    ``logaddexp(logsumexp(chunk))``, which is mathematically identical to a
    full-vocabulary softmax and only changes peak memory.
    """

    cache = compute_projectaway_probability_cache(
        visual_hidden_states,
        unembedding_weight,
        target_token_ids,
        unembedding_bias=unembedding_bias,
        normalizer=normalizer,
        vocab_chunk_size=vocab_chunk_size,
        row_chunk_size=row_chunk_size,
    )
    return cache.confidence(target_token_ids)


@torch.no_grad()
def compute_projectaway_probability_cache(
    visual_hidden_states: Union[torch.Tensor, np.ndarray],
    unembedding_weight: Union[torch.Tensor, np.ndarray],
    target_token_ids: Sequence[int],
    *,
    unembedding_bias: Optional[Union[torch.Tensor, np.ndarray]] = None,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    vocab_chunk_size: int = 8192,
    row_chunk_size: int = 256,
) -> ProjectAwayProbabilityCache:
    """Project one image once for the union of all object-subtoken ids."""

    raw_weight = torch.as_tensor(unembedding_weight)
    projection_device = raw_weight.device
    hidden = torch.as_tensor(
        visual_hidden_states, device=projection_device
    ).detach()
    weight = raw_weight.detach()
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported unembedding dtype for ProjectAway: {weight.dtype}")
    if hidden.ndim != 3:
        raise ValueError(
            f"visual_hidden_states must have shape [L,P,D], got {tuple(hidden.shape)}"
        )
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            "unembedding_weight must have shape [V,D] matching visual hidden size"
        )
    if not torch.isfinite(hidden).all():
        raise ValueError("ProjectAway hidden states contain non-finite values")
    token_ids = tuple(dict.fromkeys(int(token_id) for token_id in target_token_ids))
    if not token_ids:
        raise ValueError("target_token_ids cannot be empty")
    if min(token_ids) < 0 or max(token_ids) >= weight.shape[0]:
        raise ValueError("target_token_ids contain an id outside the vocabulary")
    chunk_size = int(vocab_chunk_size)
    if chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    rows_per_chunk = int(row_chunk_size)
    if rows_per_chunk <= 0:
        raise ValueError("row_chunk_size must be positive")

    normalized = normalizer(hidden) if normalizer is not None else hidden
    if normalized.shape != hidden.shape:
        raise ValueError("normalizer must preserve visual hidden shape [L,P,D]")
    flat = normalized.reshape(-1, normalized.shape[-1]).to(
        device=projection_device, dtype=weight.dtype
    )
    bias = None
    if unembedding_bias is not None:
        bias = torch.as_tensor(unembedding_bias, device=flat.device).detach().float()
        if bias.shape != (weight.shape[0],):
            raise ValueError("unembedding_bias must have shape [V]")

    target_index = torch.tensor(token_ids, device=flat.device, dtype=torch.long)
    target_weight = weight.index_select(0, target_index)
    if not torch.isfinite(target_weight).all():
        raise ValueError("ProjectAway target unembedding rows contain non-finite values")
    target_bias = bias.index_select(0, target_index) if bias is not None else None
    probabilities = torch.empty(
        (flat.shape[0], len(token_ids)),
        device=flat.device,
        dtype=torch.float32,
    )
    # Bound both temporary dimensions.  Casting the whole fp16/bf16 LM head to
    # fp32 would add hundreds of MB for common 7B/8B vocabularies; each vocab
    # slice is converted only for the current row block instead.
    for row_start in range(0, flat.shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, flat.shape[0])
        row_hidden = flat[row_start:row_end]
        target_logits = (row_hidden @ target_weight.T).float()
        if target_bias is not None:
            target_logits = target_logits + target_bias
        denominator = torch.full(
            (row_hidden.shape[0],),
            -torch.inf,
            device=flat.device,
            dtype=torch.float32,
        )
        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            chunk_weight = weight[start:end]
            if not torch.isfinite(chunk_weight).all():
                raise ValueError("ProjectAway unembedding contains non-finite values")
            chunk_logits = (row_hidden @ chunk_weight.T).float()
            if bias is not None:
                chunk_logits = chunk_logits + bias[start:end]
            denominator = torch.logaddexp(
                denominator, torch.logsumexp(chunk_logits, dim=-1)
            )
            del chunk_weight, chunk_logits
        probabilities[row_start:row_end] = torch.exp(
            target_logits - denominator[:, None]
        )
    probabilities = probabilities.reshape(
        hidden.shape[0], hidden.shape[1], len(token_ids)
    )
    return ProjectAwayProbabilityCache(
        probabilities=probabilities.cpu().float(),
        target_token_ids=token_ids,
    )
