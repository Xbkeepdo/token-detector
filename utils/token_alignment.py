"""Align generated response token IDs with character spans in decoded text.

The important invariant in this module is that every returned token index is an
index into the *actual* ``response_token_ids`` emitted by generation.  Re-
encoding caption fragments is deliberately avoided because byte-level BPE and
SentencePiece tokenization are not additive across whitespace boundaries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Optional, Sequence


class TokenAlignmentError(ValueError):
    """Raised when generated token IDs cannot be aligned to their caption."""


def build_response_token_offsets(
    tokenizer,
    response_token_ids: Sequence[int],
    caption: str,
) -> list[Optional[tuple[int, int]]]:
    """Return caption character offsets for each actual response token.

    Fast-tokenizer offsets are preferred when their encoded IDs form the
    visible subsequence of the generated IDs.  Slow SentencePiece tokenizers
    use the processor's immutable decode proto, whose pieces retain exact
    begin/end offsets.  A decoder-prefix reconstruction is the final fallback.
    Special tokens remain in the returned list as ``None`` so original response
    indices are never shifted.
    """

    token_ids = [int(value) for value in response_token_ids]
    if not token_ids:
        if caption:
            raise TokenAlignmentError(
                "response_token_ids is empty but generated_text is non-empty"
            )
        return []

    decoded = _decode(
        tokenizer,
        token_ids,
        skip_special_tokens=True,
    )
    _text_core_alignment(decoded, caption)

    offsets = _fast_token_offsets(tokenizer, token_ids, caption)
    if offsets is not None:
        _validate_offsets(offsets, len(caption))
        return offsets

    offsets = _sentencepiece_token_offsets(
        tokenizer,
        token_ids,
        caption,
    )
    if offsets is not None:
        _validate_offsets(offsets, len(caption))
        return offsets

    offsets = _decoder_prefix_offsets(
        tokenizer,
        token_ids,
        caption,
        decoded=decoded,
    )
    _validate_offsets(offsets, len(caption))
    return offsets


def token_indices_for_char_span(
    offsets: Sequence[Optional[tuple[int, int]]],
    char_start: int,
    char_end: int,
) -> list[int]:
    """Select actual response indices whose character ranges overlap a span."""

    start = int(char_start)
    end = int(char_end)
    if start < 0 or end <= start:
        raise TokenAlignmentError(
            f"invalid character span [{char_start}, {char_end})"
        )
    return [
        index
        for index, offset in enumerate(offsets)
        if offset is not None and offset[0] < end and offset[1] > start
    ]


def validate_token_surface(
    *,
    tokenizer,
    response_token_ids: Sequence[int],
    token_indices: Sequence[int],
    offsets: Sequence[Optional[tuple[int, int]]],
    caption: str,
    char_start: int,
    char_end: int,
) -> None:
    """Ensure selected response tokens really cover the CHAIR object surface."""

    indices = [int(value) for value in token_indices]
    if not indices:
        raise TokenAlignmentError(
            f"no response tokens overlap character span [{char_start}, {char_end})"
        )
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise TokenAlignmentError(
            f"aligned response token indices are not contiguous: {indices}"
        )
    selected_offsets = [offsets[index] for index in indices]
    if any(offset is None for offset in selected_offsets):
        raise TokenAlignmentError(
            f"aligned response token indices contain a special token: {indices}"
        )
    concrete = [offset for offset in selected_offsets if offset is not None]
    if min(offset[0] for offset in concrete) > int(char_start) or max(
        offset[1] for offset in concrete
    ) < int(char_end):
        raise TokenAlignmentError(
            "aligned response tokens do not cover the complete object surface "
            f"[{char_start}, {char_end})"
        )

    surface = caption[int(char_start):int(char_end)]
    selected_ids = [int(response_token_ids[index]) for index in indices]
    selected_text = _decode(
        tokenizer,
        selected_ids,
        skip_special_tokens=True,
    )
    normalized_surface = _normalize_for_validation(surface)
    normalized_selected = _normalize_for_validation(selected_text)
    if normalized_surface and normalized_surface not in normalized_selected:
        raise TokenAlignmentError(
            "decoded response tokens do not contain the CHAIR object surface: "
            f"surface={surface!r}, decoded_tokens={selected_text!r}, "
            f"indices={indices}"
        )


def locate_first_token_id(
    *,
    tokenizer,
    response_token_ids: Sequence[int],
    query: str,
    pluralize: Optional[Callable[[str], str]] = None,
) -> dict:
    """Reproduce SVAR's first-token-ID / first-occurrence lookup.

    When the singular query token is absent, the official implementation uses
    ``inflect.engine().plural`` and repeats the same first-ID lookup.
    """

    requested_query = str(query).strip()
    initial_ids = _encode(tokenizer, requested_query)
    initial_token_id = initial_ids[0] if initial_ids else None
    location = _first_index(response_token_ids, initial_token_id)
    if location is not None:
        return {
            "status": "found",
            "query": requested_query,
            "matched_query": requested_query,
            "query_token_id": initial_token_id,
            "matched_token_id": initial_token_id,
            "token_indices": [location],
            "used_plural_fallback": False,
        }

    plural_query = None
    plural_token_id = None
    if requested_query and pluralize is not None:
        plural_query = str(pluralize(requested_query))
        plural_ids = _encode(tokenizer, plural_query)
        plural_token_id = plural_ids[0] if plural_ids else None
        plural_location = _first_index(response_token_ids, plural_token_id)
        if plural_location is not None:
            return {
                "status": "found",
                "query": requested_query,
                "matched_query": plural_query,
                "query_token_id": initial_token_id,
                "matched_token_id": plural_token_id,
                "token_indices": [plural_location],
                "used_plural_fallback": True,
            }

    return {
        "status": "not_found",
        "query": requested_query,
        "matched_query": None,
        "query_token_id": initial_token_id,
        "matched_token_id": plural_token_id,
        "plural_query": plural_query,
        "token_indices": [],
        "used_plural_fallback": bool(plural_query),
    }


def _fast_token_offsets(
    tokenizer,
    response_token_ids: list[int],
    caption: str,
) -> Optional[list[Optional[tuple[int, int]]]]:
    if not bool(getattr(tokenizer, "is_fast", False)):
        return None
    try:
        encoded = tokenizer(
            caption,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = _flatten_ints(encoded["input_ids"])
        encoded_offsets = _flatten_offsets(encoded["offset_mapping"])
    except Exception:
        return None
    if len(encoded_ids) != len(encoded_offsets):
        return None

    match_start = _find_visible_subsequence(
        tokenizer,
        response_token_ids,
        encoded_ids,
    )
    if match_start is None:
        return None

    result: list[Optional[tuple[int, int]]] = [None] * len(response_token_ids)
    for local_index, (start, end) in enumerate(encoded_offsets):
        if end <= start:
            continue
        result[match_start + local_index] = (int(start), int(end))
    return result


def _sentencepiece_token_offsets(
    tokenizer,
    response_token_ids: list[int],
    caption: str,
) -> Optional[list[Optional[tuple[int, int]]]]:
    processor = getattr(tokenizer, "sp_model", None)
    decoder = getattr(processor, "decode_ids_as_immutable_proto", None)
    if decoder is None:
        decoder = getattr(processor, "DecodeIdsAsImmutableProto", None)
    if decoder is None:
        return None

    visible_indices = _visible_response_indices(tokenizer, response_token_ids)
    visible_ids = [response_token_ids[index] for index in visible_indices]
    if not visible_ids:
        return [None] * len(response_token_ids)
    try:
        proto = decoder(visible_ids)
    except Exception:
        return None
    pieces = list(getattr(proto, "pieces", []))
    if len(pieces) != len(visible_indices):
        return None

    decoded = str(getattr(proto, "text", ""))
    decoded_core_start, caption_core_start, _ = _text_core_alignment(
        decoded,
        caption,
    )
    decoded_core_end = len(decoded.rstrip())

    result: list[Optional[tuple[int, int]]] = [None] * len(response_token_ids)
    for response_index, piece in zip(visible_indices, pieces):
        mapped = _map_decoded_interval_to_caption(
            int(piece.begin),
            int(piece.end),
            decoded_core_start=decoded_core_start,
            decoded_core_end=decoded_core_end,
            caption_core_start=caption_core_start,
        )
        result[response_index] = mapped
    # SentencePiece byte fallback may emit several visible pieces whose proto
    # intervals have begin == end until the final byte completes a Unicode
    # character. Returning those pieces as None loses their response indices
    # and breaks surface validation, so reconstruct the group from cumulative
    # decoder progress instead.
    if any(result[index] is None for index in visible_indices):
        return None
    return result


def _decoder_prefix_offsets(
    tokenizer,
    response_token_ids: list[int],
    caption: str,
    *,
    decoded: str,
) -> list[Optional[tuple[int, int]]]:
    decoded_core_start, caption_core_start, _ = _text_core_alignment(
        decoded,
        caption,
    )
    decoded_core_end = len(decoded.rstrip())
    prefixes = [response_token_ids[: index + 1] for index in range(len(response_token_ids))]
    try:
        decoded_prefixes = tokenizer.batch_decode(
            prefixes,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        decoded_prefixes = [
            _decode(tokenizer, prefix, skip_special_tokens=True)
            for prefix in prefixes
        ]

    result: list[Optional[tuple[int, int]]] = [None] * len(response_token_ids)
    previous_progress = 0
    pending: list[int] = []
    special_ids = _tokenizer_special_ids(tokenizer)
    for response_index, prefix_text in enumerate(decoded_prefixes):
        progress = _common_prefix_length(str(prefix_text), decoded)
        if progress < previous_progress:
            raise TokenAlignmentError(
                "tokenizer decoder prefix is not monotonic at response index "
                f"{response_index}: {progress} < {previous_progress}"
            )
        token_id = response_token_ids[response_index]
        visible = token_id not in special_ids or bool(
            _decode(tokenizer, [token_id], skip_special_tokens=True)
        )
        if progress == previous_progress:
            if visible:
                pending.append(response_index)
            continue

        group = pending + ([response_index] if visible else [])
        for index in group:
            result[index] = _map_decoded_interval_to_caption(
                previous_progress,
                progress,
                decoded_core_start=decoded_core_start,
                decoded_core_end=decoded_core_end,
                caption_core_start=caption_core_start,
            )
        pending = []
        previous_progress = progress

    if previous_progress < decoded_core_end:
        raise TokenAlignmentError(
            "decoder-prefix reconstruction did not cover generated_text: "
            f"covered={previous_progress}, required={decoded_core_end}"
        )
    if pending:
        raise TokenAlignmentError(
            "visible response tokens decode to no text at the end of the response: "
            f"indices={pending}"
        )
    return result


def _find_visible_subsequence(
    tokenizer,
    response_ids: list[int],
    encoded_ids: list[int],
) -> Optional[int]:
    if not encoded_ids:
        return 0 if not str(_decode(tokenizer, response_ids, True)).strip() else None
    width = len(encoded_ids)
    for start in range(len(response_ids) - width + 1):
        if response_ids[start:start + width] != encoded_ids:
            continue
        prefix = response_ids[:start]
        suffix = response_ids[start + width:]
        if not _decode(tokenizer, prefix, True).strip() and not _decode(
            tokenizer,
            suffix,
            True,
        ).strip():
            return start
    return None


def _visible_response_indices(tokenizer, response_ids: Sequence[int]) -> list[int]:
    special_ids = _tokenizer_special_ids(tokenizer)
    result = []
    for index, token_id in enumerate(response_ids):
        value = int(token_id)
        if value in special_ids and not _decode(
            tokenizer,
            [value],
            skip_special_tokens=True,
        ):
            continue
        result.append(index)
    return result


def _tokenizer_special_ids(tokenizer) -> set[int]:
    """Return every token the tokenizer backend marks as special.

    Some Llama-3 checkpoints keep ``<|eot_id|>`` as a special AddedToken but
    omit it from ``all_special_ids`` because it is not assigned to a named
    tokenizer role.  The decoder still removes it for
    ``skip_special_tokens=True``, so alignment must include both sources.
    """

    special_ids = {
        int(value) for value in getattr(tokenizer, "all_special_ids", [])
    }
    added_tokens = getattr(tokenizer, "added_tokens_decoder", {})
    try:
        items = added_tokens.items()
    except AttributeError:
        return special_ids
    for token_id, token in items:
        if bool(getattr(token, "special", False)):
            special_ids.add(int(token_id))
    return special_ids


def _map_decoded_interval_to_caption(
    start: int,
    end: int,
    *,
    decoded_core_start: int,
    decoded_core_end: int,
    caption_core_start: int,
) -> Optional[tuple[int, int]]:
    clipped_start = max(int(start), int(decoded_core_start))
    clipped_end = min(int(end), int(decoded_core_end))
    if clipped_end <= clipped_start:
        return None
    return (
        caption_core_start + clipped_start - decoded_core_start,
        caption_core_start + clipped_end - decoded_core_start,
    )


def _text_core_alignment(decoded: str, caption: str) -> tuple[int, int, int]:
    decoded_text = str(decoded)
    caption_text = str(caption)
    decoded_core = decoded_text.strip()
    caption_core = caption_text.strip()
    if decoded_core != caption_core:
        raise TokenAlignmentError(
            "decoded response_token_ids do not match generated_text apart from "
            "leading/trailing whitespace: "
            f"decoded={decoded_text!r}, generated_text={caption_text!r}"
        )
    decoded_start = len(decoded_text) - len(decoded_text.lstrip())
    caption_start = len(caption_text) - len(caption_text.lstrip())
    return decoded_start, caption_start, len(decoded_core)


def _validate_offsets(
    offsets: Sequence[Optional[tuple[int, int]]],
    caption_length: int,
) -> None:
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end <= start or end > int(caption_length):
            raise TokenAlignmentError(
                f"invalid response token offset at index {index}: {offset}"
            )


def _normalize_for_validation(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _decode(tokenizer, token_ids: Sequence[int], skip_special_tokens: bool) -> str:
    ids = [int(value) for value in token_ids]
    try:
        return str(
            tokenizer.decode(
                ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))


def _encode(tokenizer, text: str) -> list[int]:
    try:
        encoded = tokenizer(
            str(text),
            add_special_tokens=False,
        )["input_ids"]
    except Exception:
        encoded = tokenizer.encode(str(text), add_special_tokens=False)
    return _flatten_ints(encoded)


def _flatten_ints(values) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise TokenAlignmentError("expected one tokenized sequence")
        values = values[0]
    return [int(value) for value in values]


def _flatten_offsets(values) -> list[tuple[int, int]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if values and len(values) == 1 and isinstance(values[0], (list, tuple)):
        first = values[0]
        if first and isinstance(first[0], (list, tuple)):
            values = first
    return [(int(value[0]), int(value[1])) for value in values]


def _first_index(values: Sequence[int], target: Optional[int]) -> Optional[int]:
    if target is None:
        return None
    for index, value in enumerate(values):
        if int(value) == int(target):
            return index
    return None


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index
