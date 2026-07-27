"""Layer/head Visual Attention Ratio features for the SVAR baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch


SUPPORTED_SVAR_PROTOCOLS = frozenset({"controlled", "official"})


def normalize_svar_protocols(
    value: Sequence[str] | str | None,
) -> tuple[str, ...]:
    """Normalize the configured SVAR sample protocols.

    Historical configs did not have a protocol switch and therefore mean
    ``controlled``: SVAR uses the same exact object spans as every other
    detector.  ``official`` is opt-in because it has a different sample list
    and is intentionally written and trained in an isolated directory.
    """

    if value is None:
        values = ["controlled"]
    elif isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    normalized = [
        str(item).strip().lower().replace("-", "_") for item in values
    ]
    aliases = {
        "fair": "controlled",
        "shared": "controlled",
        "paper": "official",
        "svar_official": "official",
    }
    normalized = [aliases.get(item, item) for item in normalized]
    unknown = set(normalized) - SUPPORTED_SVAR_PROTOCOLS
    if unknown:
        raise ValueError(
            f"Unknown SVAR protocols: {sorted(unknown)}; expected a subset of "
            f"{sorted(SUPPORTED_SVAR_PROTOCOLS)}"
        )
    result = tuple(dict.fromkeys(normalized))
    if not result:
        raise ValueError("SVAR protocols cannot be empty")
    return result


def prepare_official_svar_spans(
    samples: Sequence[Mapping[str, Any]] | None,
    response_token_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return found, in-range official-SVAR samples as ordinary span records.

    Labeling owns the official first-token-ID lookup.  This function only
    accepts its resolved result and deliberately skips ``not_found`` entries;
    it never falls back to a neighbouring token.  A few equivalent field
    spellings are accepted so schema-v2 labeling can evolve without coupling
    extraction to one serialization detail.
    """

    response_ids = [int(value) for value in response_token_ids]
    result: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, str]] = set()
    for sample_index, raw_sample in enumerate(samples or ()):
        if not isinstance(raw_sample, Mapping):
            continue
        sample = dict(raw_sample)
        token_location = sample.get("token_location")
        token_location = (
            dict(token_location)
            if isinstance(token_location, Mapping)
            else {}
        )
        schema_v2_resolved = (
            "status" in sample and isinstance(sample.get("token_location"), Mapping)
        )
        status = str(
            sample.get("status")
            or sample.get("match_status")
            or token_location.get("status")
            or "found"
        ).strip().lower()
        if status == "not_found":
            continue
        if status in {"missing", "skipped", "skip", "invalid", "error"}:
            if schema_v2_resolved:
                raise ValueError(
                    f"Official SVAR sample {sample_index} has unsupported "
                    f"schema-v2 status {status!r}."
                )
            continue

        token_indices = _resolved_token_indices(sample)
        if schema_v2_resolved and status != "found":
            raise ValueError(
                f"Official SVAR sample {sample_index} has unsupported status "
                f"{status!r}; only explicit 'found' or 'not_found' is valid."
            )
        if schema_v2_resolved and len(token_indices) != 1:
            raise ValueError(
                f"Official SVAR sample {sample_index} is marked found but does "
                "not contain exactly one saved response token index."
            )
        if not token_indices:
            first_token_id = _first_int(
                sample,
                (
                    "matched_first_token_id",
                    "first_token_id",
                    "search_token_id",
                    "target_token_id",
                    "token_id",
                ),
            )
            if first_token_id is None:
                first_token_id = _first_int(
                    token_location,
                    ("matched_token_id", "query_token_id"),
                )
            if first_token_id is not None:
                try:
                    token_indices = [response_ids.index(first_token_id)]
                except ValueError:
                    plural_token_id = _first_int(
                        sample,
                        (
                            "plural_first_token_id",
                            "plural_token_id",
                        ),
                    )
                    if plural_token_id is not None:
                        try:
                            token_indices = [response_ids.index(plural_token_id)]
                        except ValueError:
                            token_indices = []
        if not token_indices:
            continue
        first_index = int(token_indices[0])
        if first_index < 0 or first_index >= len(response_ids):
            if schema_v2_resolved:
                raise ValueError(
                    f"Official SVAR sample {sample_index} index {first_index} "
                    f"is outside response length {len(response_ids)}."
                )
            continue
        if schema_v2_resolved:
            matched_token_id = _first_int(
                token_location,
                ("matched_token_id", "query_token_id"),
            )
            if matched_token_id is None:
                raise ValueError(
                    f"Official SVAR sample {sample_index} has no saved matched "
                    "token ID."
                )
            if response_ids[first_index] != matched_token_id:
                raise ValueError(
                    f"Official SVAR sample {sample_index} index/token mismatch: "
                    f"response[{first_index}]={response_ids[first_index]} != "
                    f"{matched_token_id}."
                )
            if response_ids.index(matched_token_id) != first_index:
                raise ValueError(
                    f"Official SVAR sample {sample_index} does not point to the "
                    "first occurrence of its matched token ID."
                )

        label = sample.get("label")
        if label is None or isinstance(label, (bool, np.bool_)):
            if schema_v2_resolved:
                raise ValueError(
                    f"Official SVAR sample {sample_index} has no valid label."
                )
            continue
        label = int(label)
        if label not in (0, 1):
            if schema_v2_resolved:
                raise ValueError(
                    f"Official SVAR sample {sample_index} has invalid label {label}."
                )
            continue
        search_term = str(
            sample.get("search_term")
            or sample.get("query")
            or sample.get("word")
            or sample.get("surface")
            or sample.get("canonical")
            or ""
        )
        search_source = str(
            sample.get("search_source")
            or sample.get("source")
            or sample.get("kind")
            or "unknown"
        )
        key = (first_index, search_term, label, search_source)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "word": search_term,
                "label": label,
                # Official SVAR takes the hidden/attention state that predicts
                # the first matched token, even for a multi-token search term.
                "token_indices": [first_index],
                "svar_official": {
                    "sample_index": int(sample_index),
                    "status": "found",
                    "search_term": search_term,
                    "search_source": search_source,
                    "surface": sample.get("surface"),
                    "canonical": (
                        sample.get("canonical")
                        or sample.get("canonical_object")
                    ),
                    "matched_form": (
                        sample.get("matched_form")
                        or token_location.get("matched_query")
                    ),
                    "used_plural_fallback": bool(
                        sample.get("used_plural_fallback")
                        or token_location.get("used_plural_fallback")
                    ),
                    "first_token_id": int(response_ids[first_index]),
                },
            }
        )
    return result


def _resolved_token_indices(sample: Mapping[str, Any]) -> list[int]:
    direct_fields = (
        "token_indices",
        "response_token_indices",
        "matched_token_indices",
    )
    for field in direct_fields:
        values = sample.get(field)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            try:
                parsed = [int(value) for value in values]
            except (TypeError, ValueError):
                parsed = []
            if parsed:
                return parsed
    for field in (
        "response_token_idx",
        "token_index",
        "first_token_index",
        "matched_token_index",
    ):
        value = sample.get(field)
        if value is not None:
            try:
                return [int(value)]
            except (TypeError, ValueError):
                pass

    locations = sample.get("token_locations")
    if isinstance(locations, Mapping):
        preferred = (
            "svar_official",
            "svar_canonical_first_token_id",
            "svar_surface_first_token_id",
        )
        for name in preferred:
            location = locations.get(name)
            if isinstance(location, Mapping):
                nested = _resolved_token_indices(location)
                if nested:
                    return nested
            elif isinstance(location, Sequence) and not isinstance(
                location, (str, bytes)
            ):
                try:
                    parsed = [int(value) for value in location]
                except (TypeError, ValueError):
                    parsed = []
                if parsed:
                    return parsed
    location = sample.get("token_location")
    if isinstance(location, Mapping):
        nested = _resolved_token_indices(location)
        if nested:
            return nested
    return []


def _first_int(
    mapping: Mapping[str, Any],
    fields: Sequence[str],
) -> Optional[int]:
    for field in fields:
        value = mapping.get(field)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


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
    all_layers: bool = False,
) -> SVARFeatures:
    """Return the concatenated per-layer/per-head VAR detector input.

    The input is ``[L,H,P]`` and VAR is the attention mass allocated to all
    visual tokens for every layer/head.  The selected matrix is flattened for
    the detector MLP.  Production extraction sets ``all_layers=True`` so the
    saved artifact remains reusable; training applies its configured layer
    slice later.  ``score`` is the classic scalar SVAR (sum over layers after
    averaging heads).
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

    if all_layers:
        start, end = 0, int(num_layers)
    else:
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


def svar_training_vector(
    payload: Mapping[str, Any],
    *,
    layer_start: int = 5,
    layer_end: int = 19,
) -> np.ndarray:
    """Slice a saved SVAR matrix by absolute layer number for training.

    New artifacts contain every decoder layer.  Legacy artifacts that contain
    only layers 5--18 remain valid because their ``layer_start`` metadata maps
    the stored rows back to absolute model-layer indices.
    """

    start = int(layer_start)
    end = int(layer_end)
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid SVAR training layer range [{start},{end}); expected "
            "0 <= layer_start < layer_end"
        )
    matrix_value = payload.get("visual_attention_ratio")
    stored_start = int(payload.get("layer_start", 0))
    if matrix_value is None:
        vector_value = payload.get("vector")
        if vector_value is None:
            raise ValueError("SVAR payload has no training vector")
        vector = np.asarray(vector_value, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("SVAR training vector is empty or non-finite")
        if (
            "layer_start" not in payload
            and "layer_end_exclusive" not in payload
        ):
            # Very old/test payloads predate layer metadata; their vector is
            # already the detector input and cannot be resliced further.
            return vector
        stored_end = int(payload.get("layer_end_exclusive", end))
        if (start, end) != (stored_start, stored_end):
            raise ValueError(
                "Legacy SVAR payload has no per-layer matrix and cannot be "
                f"resliced from [{stored_start},{stored_end}) to [{start},{end})"
            )
        return vector

    matrix = np.asarray(matrix_value, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError(
            "SVAR visual_attention_ratio must have shape [layers,heads], got "
            f"{matrix.shape}"
        )
    stored_end = int(
        payload.get("layer_end_exclusive", stored_start + matrix.shape[0])
    )
    if stored_end - stored_start != matrix.shape[0]:
        raise ValueError(
            "SVAR layer metadata does not match visual_attention_ratio rows: "
            f"[{stored_start},{stored_end}) versus {matrix.shape[0]} rows"
        )
    if start < stored_start or end > stored_end:
        raise ValueError(
            f"Requested SVAR training layers [{start},{end}) are outside saved "
            f"layers [{stored_start},{stored_end}); re-extract all-layer SVAR "
            "features or use a compatible legacy range"
        )
    selected = matrix[start - stored_start : end - stored_start]
    if not np.isfinite(selected).all():
        raise ValueError("SVAR training slice contains non-finite values")
    return selected.reshape(-1).astype(np.float32, copy=False)
