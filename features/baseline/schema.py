"""Stable on-disk schema for token-level hallucination baselines.

The main extractor owns image/model inference.  This module deliberately only
defines the small, pickle-friendly record exchanged with the independent
baseline feature builders and trainers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import numpy as np


BASELINE_SCHEMA_VERSION = "1.0"
SUPPORTED_BASELINES = frozenset(
    {"metatoken", "svar", "dhcp", "projectaway", "halloc"}
)


def make_baseline_record(
    *,
    image_id: int,
    token_str: str,
    response_token_idx: int,
    label: int,
    target_token_id: Optional[int] = None,
    span_start: Optional[int] = None,
    span_end: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create one record while preserving the project's raw label convention.

    ``label=0`` means hallucinated and ``label=1`` means real.  Classifiers
    convert this field explicitly when using hallucination as the positive
    class; the stored annotation is never silently inverted.
    """

    _validate_binary_label(label)
    response_token_idx = int(response_token_idx)
    if response_token_idx < 0:
        raise ValueError("response_token_idx must be non-negative")
    start = response_token_idx if span_start is None else int(span_start)
    end = start if span_end is None else int(span_end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid token span [{start}, {end}]")

    return {
        "baseline_schema_version": BASELINE_SCHEMA_VERSION,
        "image_id": int(image_id),
        "token_str": str(token_str),
        "response_token_idx": response_token_idx,
        "target_token_id": (
            None if target_token_id is None else int(target_token_id)
        ),
        "span_start": start,
        "span_end": end,
        "label": int(label),
        "label_semantics": {"0": "hallucination", "1": "real"},
        "baselines": {},
        "metadata": deepcopy(dict(metadata or {})),
    }


def attach_baseline(
    record: MutableMapping[str, Any],
    name: str,
    payload: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Attach a validated feature payload and return ``record`` for chaining."""

    normalized = str(name).strip().lower()
    if normalized not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unknown baseline {name!r}; expected one of {sorted(SUPPORTED_BASELINES)}"
        )
    if not isinstance(payload, Mapping):
        raise TypeError(f"{normalized} payload must be a mapping")
    if "baselines" not in record or not isinstance(record["baselines"], dict):
        record["baselines"] = {}
    record["baselines"][normalized] = deepcopy(dict(payload))
    return record


def validate_baseline_record(
    record: Mapping[str, Any],
    required: Iterable[str] = (),
) -> None:
    """Raise a descriptive exception when a baseline record is malformed."""

    if record.get("baseline_schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported baseline schema version "
            f"{record.get('baseline_schema_version')!r}; expected "
            f"{BASELINE_SCHEMA_VERSION!r}"
        )
    for key in ("image_id", "response_token_idx", "label", "baselines"):
        if key not in record:
            raise KeyError(f"Missing required baseline record field: {key}")
    _validate_binary_label(record["label"])
    if not isinstance(record["baselines"], Mapping):
        raise TypeError("record['baselines'] must be a mapping")
    required_names = {str(name).strip().lower() for name in required}
    unknown = required_names - SUPPORTED_BASELINES
    if unknown:
        raise ValueError(f"Unknown required baselines: {sorted(unknown)}")
    missing = required_names - set(record["baselines"])
    if missing:
        raise KeyError(f"Baseline record is missing payloads: {sorted(missing)}")
    _assert_finite(record["baselines"], path="baselines")


def get_baseline_payload(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    normalized = str(name).strip().lower()
    validate_baseline_record(record, required=(normalized,))
    return record["baselines"][normalized]


def baseline_vector(record: Mapping[str, Any], name: str) -> np.ndarray:
    """Return the canonical dense vector for MetaToken or SVAR records."""

    normalized = str(name).strip().lower()
    payload = get_baseline_payload(record, normalized)
    if normalized == "metatoken":
        definition = payload.get("metatoken_feature_definition")
        probability_definition = payload.get("probability_difference_definition")
        if (
            definition != "original_paper_equations_1_to_12"
            or probability_definition != "paper_eq_11"
        ):
            raise ValueError(
                "MetaToken payload predates the original-paper feature definition; "
                "re-extract baseline/features.pkl before training"
            )
    if "vector" not in payload:
        raise KeyError(f"Baseline {name!r} does not contain a dense 'vector'")
    vector = np.asarray(payload["vector"], dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"Baseline {name!r} has an empty or non-finite vector")
    return vector


def _validate_binary_label(label: Any) -> None:
    if isinstance(label, (bool, np.bool_)) or int(label) not in (0, 1):
        raise ValueError(
            f"label must be 0 (hallucination) or 1 (real), got {label!r}"
        )


def _assert_finite(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and not np.isfinite(array).all():
            raise ValueError(f"Non-finite numeric value at {path}")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"Non-finite numeric value at {path}")
