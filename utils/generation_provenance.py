"""Stable provenance for generated captions and their actual response IDs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


GENERATION_MANIFEST_NAME = "generation_manifest.json"
GENERATION_MANIFEST_VERSION = 2
_NON_GENERATION_MODEL_KEYS = {
    "dgst_t_support_scope",
    "extraction_mode",
    "generation_prompt",
    "prompt",
}


def stable_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_model_config_sha256(model_cfg: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in model_cfg.items()
        if str(key) not in _NON_GENERATION_MODEL_KEYS
    }
    return stable_sha256(payload)


def _normalized_image_ids(image_ids: Sequence[int] | set[int]) -> list[int]:
    normalized = sorted({int(value) for value in image_ids})
    if len(normalized) != len(image_ids):
        raise ValueError("Selected generation image IDs must be unique")
    if not normalized:
        raise ValueError("Selected generation image cohort cannot be empty")
    return normalized


def build_generation_run_manifest(
    *,
    model: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    expected_image_ids: Sequence[int] | set[int],
) -> dict[str, object]:
    """Build the identity written before the first generation shard."""

    image_ids = _normalized_image_ids(expected_image_ids)
    return {
        "manifest_version": GENERATION_MANIFEST_VERSION,
        "status": "in_progress",
        "model": str(model),
        "model_config_sha256": generation_model_config_sha256(model_cfg),
        "prompt": str(prompt),
        "selected_image_ids_sha256": stable_sha256(image_ids),
        "num_images": len(image_ids),
    }


def canonical_generation_payload(
    generations: Mapping[str, Any],
    *,
    expected_image_ids: Sequence[int] | set[int] | None = None,
) -> dict[str, dict[str, object]]:
    if not isinstance(generations, Mapping):
        raise ValueError("generations.json must contain a JSON object")
    normalized_rows: dict[str, Any] = {}
    for raw_image_id, row in generations.items():
        try:
            image_id = str(int(raw_image_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid generation image ID {raw_image_id!r}"
            ) from exc
        if image_id in normalized_rows:
            raise ValueError(f"Duplicate normalized generation image ID {image_id}")
        normalized_rows[image_id] = row

    if expected_image_ids is not None:
        expected = {str(value) for value in _normalized_image_ids(expected_image_ids)}
        actual = set(normalized_rows)
        if actual != expected:
            missing = sorted(expected - actual, key=int)[:10]
            extra = sorted(actual - expected, key=int)[:10]
            raise ValueError(
                "generations.json does not match the selected image cohort; "
                f"missing={missing}, extra={extra}"
            )

    result: dict[str, dict[str, object]] = {}
    for image_id in sorted(normalized_rows, key=int):
        row = normalized_rows[image_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"Invalid generation row for image {image_id}")
        generated_text = row.get("generated_text")
        raw_ids = row.get("response_token_ids")
        if not isinstance(generated_text, str) or not generated_text:
            raise ValueError(
                f"Generation row {image_id} has no non-empty generated_text"
            )
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(
                f"Generation row {image_id} has no actual response_token_ids"
            )
        try:
            token_ids = [int(value) for value in raw_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Generation row {image_id} has invalid response_token_ids"
            ) from exc
        result[image_id] = {
            "generated_text": generated_text,
            "response_token_ids": token_ids,
        }
    return result


def build_generation_manifest(
    *,
    model: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    generations: Mapping[str, Any],
    expected_image_ids: Sequence[int] | set[int] | None = None,
) -> dict[str, object]:
    payload = canonical_generation_payload(
        generations, expected_image_ids=expected_image_ids
    )
    image_ids = (
        [int(value) for value in payload]
        if expected_image_ids is None
        else expected_image_ids
    )
    manifest = build_generation_run_manifest(
        model=model,
        model_cfg=model_cfg,
        prompt=prompt,
        expected_image_ids=image_ids,
    )
    manifest.update(
        {
            "status": "complete",
            "generation_sha256": stable_sha256(payload),
        }
    )
    return manifest


def validate_generation_identity(
    manifest: Mapping[str, Any],
    *,
    model: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    expected_image_ids: Sequence[int] | set[int],
) -> dict[str, object]:
    expected = build_generation_run_manifest(
        model=model,
        model_cfg=model_cfg,
        prompt=prompt,
        expected_image_ids=expected_image_ids,
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("generation_manifest.json must contain a JSON object")
    identity_keys = tuple(key for key in expected if key != "status")
    mismatches = {
        key: (manifest.get(key), expected[key])
        for key in identity_keys
        if manifest.get(key) != expected[key]
    }
    if manifest.get("status") not in {"in_progress", "complete"}:
        mismatches["status"] = (
            manifest.get("status"),
            "in_progress|complete",
        )
    if mismatches:
        raise ValueError(
            "generation_manifest.json is incompatible with the requested "
            f"model/prompt/cohort: {mismatches}"
        )
    return expected


def validate_generation_manifest(
    manifest: Mapping[str, Any],
    *,
    model: str,
    model_cfg: Mapping[str, Any],
    prompt: str,
    generations: Mapping[str, Any],
    expected_image_ids: Sequence[int] | set[int] | None = None,
) -> dict[str, object]:
    expected = build_generation_manifest(
        model=model,
        model_cfg=model_cfg,
        prompt=prompt,
        generations=generations,
        expected_image_ids=expected_image_ids,
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("generation_manifest.json must contain a JSON object")
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "generation_manifest.json is incompatible with the requested "
            f"model/prompt/content: {mismatches}"
        )
    return expected
