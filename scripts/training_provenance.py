"""Strict provenance and completeness checks for training feature artifacts."""

from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

from features.baseline.svar import prepare_official_svar_spans
from scripts.extract_features import (
    _combined_baseline_config,
    _controlled_baseline_feature_config,
    _feature_provenance,
    _official_svar_feature_config,
    _resolve_extraction_mode,
)
from utils.config_utils import (
    extraction_mode_flags,
    get_dgst_t_cfg,
    get_model_cfg,
)
from utils.io_utils import load_json, load_pkl
from utils.split_utils import validate_strict_811_split


SUPPORTED_ARTIFACT_FAMILIES = frozenset(
    {"root", "baseline_controlled", "baseline_svar_official"}
)


def load_validated_training_features(
    *,
    feature_path: Union[Path, str],
    artifact_family: str,
    model_key: str,
    config: Mapping[str, Any],
    output_dir: Union[Path, str],
    image_splits: Mapping[str, Sequence[int]],
) -> list[Mapping[str, Any]]:
    """Load a complete feature artifact only after strict provenance checks."""

    path = Path(feature_path)
    if not path.exists():
        raise FileNotFoundError(path)
    family = str(artifact_family)
    if family not in SUPPORTED_ARTIFACT_FAMILIES:
        raise ValueError(
            f"Unsupported feature artifact family {family!r}; expected one of "
            f"{sorted(SUPPORTED_ARTIFACT_FAMILIES)}."
        )

    current = expected_feature_provenance(
        artifact_family=family,
        model_key=model_key,
        config=config,
        output_dir=output_dir,
    )
    manifest_path = path.parent / "features_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing feature provenance manifest: {manifest_path}. "
            "Refusing to train on an unverified artifact."
        )
    manifest = load_json(str(manifest_path))
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"Invalid feature provenance manifest: {manifest_path}")
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in current.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Feature provenance mismatch for {family}: {mismatches}. "
            "Re-run feature extraction with the current model, prompt, "
            "labeling, generations and feature configuration."
        )

    records = load_pkl(str(path))
    if not isinstance(records, list):
        raise RuntimeError(f"Feature artifact must contain a list: {path}")
    _validate_feature_completeness(
        records=records,
        artifact_family=family,
        output_dir=Path(output_dir),
        image_splits=image_splits,
    )
    return records


def expected_feature_provenance(
    *,
    artifact_family: str,
    model_key: str,
    config: Mapping[str, Any],
    output_dir: Union[Path, str],
) -> dict[str, Any]:
    """Recompute the exact extraction manifest payload for one artifact."""

    family = str(artifact_family)
    if family not in SUPPORTED_ARTIFACT_FAMILIES:
        raise ValueError(f"Unsupported feature artifact family {family!r}.")
    effective_config = copy.deepcopy(dict(config))
    model_cfg = get_model_cfg(effective_config, model_key)
    run_cfg = effective_config.get("run") or {}
    prompt = str(
        (run_cfg.get("prompt") if isinstance(run_cfg, Mapping) else None)
        or model_cfg.get("prompt")
        or "Describe this image."
    )

    if family == "root":
        feature_cfg = _resolved_feature_extraction_config(effective_config)
        effective_config["feature_extraction"] = feature_cfg
        dgst_t_cfg = copy.deepcopy(get_dgst_t_cfg(effective_config))
        feature_config: object = {
            "method": feature_cfg.get("method"),
            "ads_cgc": feature_cfg.get("ads_cgc"),
            "dgst_t": dgst_t_cfg,
            "ads": feature_cfg.get("ads"),
            "cgc": feature_cfg.get("cgc"),
        }
    else:
        baseline_cfg = _combined_baseline_config(effective_config)
        feature_config = (
            _controlled_baseline_feature_config(baseline_cfg)
            if family == "baseline_controlled"
            else _official_svar_feature_config(baseline_cfg)
        )

    return _feature_provenance(
        artifact_family=family,
        model_key=model_key,
        model_cfg=model_cfg,
        prompt=prompt,
        feature_config=feature_config,
        output_dir=str(output_dir),
        config=effective_config,
    )


def _resolved_feature_extraction_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    feature_cfg = copy.deepcopy(dict(config.get("feature_extraction") or {}))
    mode = _resolve_extraction_mode(dict(config), None)
    if mode is None:
        return feature_cfg
    flags = extraction_mode_flags(mode)
    for family in ("method", "ads_cgc", "baseline"):
        section = feature_cfg.get(family)
        if isinstance(section, Mapping):
            section = copy.deepcopy(dict(section))
            configured = bool(section.get("enabled", True))
        else:
            configured = True if section is None else bool(section)
            section = {}
        section["enabled"] = bool(flags[family] and configured)
        feature_cfg[family] = section
    return feature_cfg


def _validate_feature_completeness(
    *,
    records: Sequence[Mapping[str, Any]],
    artifact_family: str,
    output_dir: Path,
    image_splits: Mapping[str, Sequence[int]],
) -> None:
    labeling_path = output_dir / "labeling.json"
    generations_path = output_dir / "generations.json"
    if not labeling_path.exists():
        raise FileNotFoundError(labeling_path)
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)
    labeling = load_json(str(labeling_path))
    generations = load_json(str(generations_path))
    if not isinstance(labeling, Mapping):
        raise RuntimeError(f"Invalid schema-v2 labeling file: {labeling_path}")
    if not isinstance(generations, Mapping):
        raise RuntimeError(f"Invalid generations file: {generations_path}")

    label_image_ids = {int(value) for value in labeling}
    validate_strict_811_split(
        image_splits,
        expected_image_ids=label_image_ids,
    )
    for raw_image_id, row in labeling.items():
        if not isinstance(row, Mapping) or int(row.get("schema_version", -1)) != 2:
            raise RuntimeError(
                f"Image {raw_image_id} is not a schema-v2 labeling row."
            )

    if artifact_family == "baseline_svar_official":
        expected = _expected_official_keys(labeling, generations)
        actual = Counter(_official_record_key(record) for record in records)
    else:
        expected = _expected_controlled_keys(labeling)
        actual = Counter(_controlled_record_key(record) for record in records)
    if actual != expected:
        missing = list((expected - actual).elements())
        extra = list((actual - expected).elements())
        raise RuntimeError(
            f"Refusing to train on partial or incompatible {artifact_family} "
            f"features: expected={sum(expected.values())}, "
            f"actual={sum(actual.values())}, missing={missing[:10]}, "
            f"extra={extra[:10]}."
        )


def _expected_controlled_keys(
    labeling: Mapping[str, Any],
) -> Counter[tuple[int, int, str, int]]:
    result: Counter[tuple[int, int, str, int]] = Counter()
    for raw_image_id, row in labeling.items():
        image_id = int(raw_image_id)
        for span in row.get("object_token_spans") or []:
            if not isinstance(span, Mapping):
                raise RuntimeError(
                    f"Image {image_id} contains an invalid controlled span."
                )
            indices = _required_indices(span, image_id=image_id)
            if not indices:
                continue
            label = _required_binary_label(span, image_id=image_id)
            token_str = str(
                span.get("word")
                or span.get("canonical_object")
                or span.get("surface")
                or ""
            )
            if not token_str:
                raise RuntimeError(
                    f"Image {image_id} controlled span has no object name."
                )
            result[(image_id, indices[0], token_str, label)] += 1
    return result


def _expected_official_keys(
    labeling: Mapping[str, Any],
    generations: Mapping[str, Any],
) -> Counter[tuple[int, int, str, int, str]]:
    result: Counter[tuple[int, int, str, int, str]] = Counter()
    for raw_image_id, row in labeling.items():
        image_id = int(raw_image_id)
        generation = generations.get(str(image_id))
        if not isinstance(generation, Mapping):
            raise RuntimeError(f"Missing generation row for image {image_id}.")
        response_ids = generation.get("response_token_ids") or []
        spans = prepare_official_svar_spans(
            row.get("official_svar_samples") or [],
            [int(value) for value in response_ids],
        )
        for span in spans:
            index = _required_indices(span, image_id=image_id)[0]
            label = _required_binary_label(span, image_id=image_id)
            token_str = str(span.get("word") or "")
            metadata = span.get("svar_official") or {}
            search_source = str(metadata.get("search_source") or "unknown")
            result[(image_id, index, token_str, label, search_source)] += 1
    return result


def _controlled_record_key(
    record: Mapping[str, Any],
) -> tuple[int, int, str, int]:
    if not isinstance(record, Mapping):
        raise RuntimeError("Feature artifact contains a non-mapping record.")
    return (
        int(record.get("image_id", -1)),
        int(record.get("response_token_idx", -1)),
        str(record.get("token_str") or ""),
        _required_binary_label(record, image_id=record.get("image_id", -1)),
    )


def _official_record_key(
    record: Mapping[str, Any],
) -> tuple[int, int, str, int, str]:
    controlled = _controlled_record_key(record)
    metadata = record.get("metadata") or {}
    official = metadata.get("svar_official") or {}
    return (*controlled, str(official.get("search_source") or "unknown"))


def _required_indices(
    value: Mapping[str, Any],
    *,
    image_id: Any,
) -> list[int]:
    raw = value.get("token_indices") or []
    try:
        indices = [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Image {image_id} contains invalid token indices {raw!r}."
        ) from exc
    if any(index < 0 for index in indices):
        raise RuntimeError(
            f"Image {image_id} contains negative token indices {indices}."
        )
    return indices


def _required_binary_label(value: Mapping[str, Any], *, image_id: Any) -> int:
    try:
        label = int(value.get("label", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Image {image_id} contains an invalid label.") from exc
    if label not in (0, 1):
        raise RuntimeError(
            f"Image {image_id} contains a non-binary label {label}."
        )
    return label
