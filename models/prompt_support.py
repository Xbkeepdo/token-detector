"""Prompt support selection helpers for DGST-T."""

from __future__ import annotations

from typing import Sequence

from models.dgst_capture import merged_position_for_tokenized_position


def resolve_prompt_support_positions(
    *,
    tokenizer,
    full_input_ids: Sequence[int],
    prompt_tokenized_length: int,
    image_token_id: int,
    visual_start: int,
    visual_end: int,
    cfg_dgst_t: dict | None,
    model_name: str,
) -> list[int] | None:
    """Return prompt support override positions, or None for full prompt."""
    if cfg_dgst_t is None:
        return None
    mode = str(
        cfg_dgst_t.get(
            "dgst_t_prompt_support_mode",
            cfg_dgst_t.get("prompt_support_mode", "full"),
        )
    ).strip().lower()
    if mode in {"full", "all", "template"}:
        return None
    if mode not in {"user_text", "user", "semantic"}:
        raise ValueError(
            "dgst_t_prompt_support_mode must be 'full' or 'user_text', "
            f"got {mode!r}."
        )

    user_text = str(
        cfg_dgst_t.get(
            "dgst_t_user_prompt_text",
            cfg_dgst_t.get("user_prompt_text", "Describe this image."),
        )
    )
    prompt_ids = [int(token_id) for token_id in full_input_ids[: int(prompt_tokenized_length)]]
    span = find_user_text_token_span(
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        user_text=user_text,
        image_token_id=int(image_token_id),
        model_name=model_name,
    )
    visual_count = int(visual_end) - int(visual_start)
    return [
        int(
            merged_position_for_tokenized_position(
                full_input_ids=full_input_ids,
                tokenized_position=tokenized_position,
                image_token_id=int(image_token_id),
                visual_token_count=visual_count,
            )
        )
        for tokenized_position in range(span[0], span[1])
    ]


def find_user_text_token_span(
    *,
    tokenizer,
    prompt_ids: Sequence[int],
    user_text: str,
    image_token_id: int,
    model_name: str,
) -> tuple[int, int]:
    target = normalize_prompt_text(user_text)
    if not target:
        raise ValueError("dgst_t_user_prompt_text must be non-empty in user_text mode.")

    matches: list[tuple[int, int]] = []
    max_span = min(32, int(len(prompt_ids)))
    for start in range(int(len(prompt_ids))):
        for end in range(start + 1, min(int(len(prompt_ids)), start + max_span) + 1):
            span_ids = [int(token_id) for token_id in prompt_ids[start:end]]
            if int(image_token_id) in span_ids:
                continue
            decoded = tokenizer.decode(span_ids, skip_special_tokens=False)
            if normalize_prompt_text(decoded) == target:
                matches.append((start, end))

    if not matches:
        decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        raise ValueError(
            "Could not locate dgst_t_user_prompt_text in prompt tokens for "
            f"{model_name}: {user_text!r}. Decoded prompt={decoded_prompt!r}"
        )
    return min(matches, key=lambda item: (item[1] - item[0], item[0]))


def normalize_prompt_text(text: str) -> str:
    return " ".join(str(text).strip().split())
