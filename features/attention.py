"""Attention aggregation helpers."""

from __future__ import annotations

import torch


def compute_alpha_img_alpha_text(
    text_to_patch_attn: torch.Tensor,
    text_to_text_attn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer image/text attention mass for the target token."""
    alpha_img = text_to_patch_attn.sum(dim=-1).mean(dim=-1)
    alpha_text = text_to_text_attn.sum(dim=-1).mean(dim=-1)
    return alpha_img, alpha_text
