"""Layer/head Visual Attention Ratio features for the SVAR baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Union

import numpy as np
import torch


@dataclass(frozen=True)
class SVARFeatures:
    vector: np.ndarray
    visual_attention_ratio: np.ndarray
    score: float
    layer_start: int
    layer_end: int

    def as_payload(self) -> dict:
        return {
            "vector": self.vector.astype(np.float32, copy=False),
            "visual_attention_ratio": self.visual_attention_ratio.astype(
                np.float32, copy=False
            ),
            "score": float(self.score),
            "layer_start": int(self.layer_start),
            "layer_end_exclusive": int(self.layer_end),
        }


def compute_svar_features(
    visual_attention: Union[torch.Tensor, np.ndarray],
    *,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    start_fraction: float = 0.15,
    end_fraction: float = 0.55,
) -> SVARFeatures:
    """Return the concatenated per-layer/per-head VAR detector input.

    The input is ``[L,H,P]`` and VAR is the attention mass allocated to all
    visual tokens for every layer/head.  The selected middle-layer matrix is
    flattened for the paper's one-hidden-layer MLP.  ``score`` is the classic
    scalar SVAR (sum over layers after averaging heads).
    """

    attention = torch.as_tensor(visual_attention).detach().float()
    if attention.ndim != 3:
        raise ValueError(
            f"visual_attention must have shape [L,H,P], got {tuple(attention.shape)}"
        )
    num_layers, num_heads, num_patches = attention.shape
    if min(num_layers, num_heads, num_patches) <= 0:
        raise ValueError("visual_attention dimensions must all be positive")
    if not torch.isfinite(attention).all():
        raise ValueError("visual_attention contains non-finite values")

    start = (
        int(layer_start)
        if layer_start is not None
        else int(math.floor(num_layers * float(start_fraction)))
    )
    end = (
        int(layer_end)
        if layer_end is not None
        else int(math.ceil(num_layers * float(end_fraction)))
    )
    start = max(0, min(start, num_layers - 1))
    end = max(start + 1, min(end, num_layers))

    ratio = attention.sum(dim=-1)[start:end]
    vector = ratio.reshape(-1).cpu().numpy().astype(np.float32, copy=False)
    matrix = ratio.cpu().numpy().astype(np.float32, copy=False)
    score = float(ratio.mean(dim=-1).sum().item())
    return SVARFeatures(
        vector=vector,
        visual_attention_ratio=matrix,
        score=score,
        layer_start=start,
        layer_end=end,
    )
