"""Deterministic, leak-free image-level train/val/test splits."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union


SPLIT_NAMES = ("train", "val", "test")


def build_strict_811_split(
    image_ids: Iterable[int],
    *,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Build a deterministic 80/10/10 split from unique image IDs.

    Sorting before the seeded shuffle makes the result independent of JSON or
    filesystem iteration order.  For 4,000 IDs this returns exactly
    3,200/400/400 IDs.
    """

    ids = [int(image_id) for image_id in image_ids]
    unique_ids = sorted(set(ids))
    if len(unique_ids) != len(ids):
        raise ValueError("image_ids contains duplicate IDs")
    if len(unique_ids) < 3:
        raise ValueError("At least three unique image IDs are required")

    rng = random.Random(int(seed))
    rng.shuffle(unique_ids)
    count = len(unique_ids)
    n_train = int(count * 0.8)
    n_val = int(count * 0.1)
    n_test = count - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"80/10/10 split would contain an empty partition for {count} IDs"
        )

    return {
        "train": unique_ids[:n_train],
        "val": unique_ids[n_train : n_train + n_val],
        "test": unique_ids[n_train + n_val :],
    }


def validate_strict_811_split(
    splits: Mapping[str, Sequence[int]],
    *,
    expected_image_ids: Optional[Iterable[int]] = None,
) -> dict[str, int]:
    """Validate keys, counts, uniqueness, disjointness, and ID coverage.

    Returns the partition counts on success and raises ``ValueError`` with a
    useful diagnostic otherwise.
    """

    if not isinstance(splits, Mapping):
        raise ValueError("image split must be a mapping")
    missing = [name for name in SPLIT_NAMES if name not in splits]
    extra = sorted(set(splits) - set(SPLIT_NAMES))
    if missing or extra:
        raise ValueError(f"image split keys invalid: missing={missing}, extra={extra}")

    normalized: dict[str, list[int]] = {}
    for name in SPLIT_NAMES:
        values = splits[name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"split {name!r} must be a list of image IDs")
        normalized[name] = [int(value) for value in values]
        if len(set(normalized[name])) != len(normalized[name]):
            raise ValueError(f"split {name!r} contains duplicate image IDs")

    sets = {name: set(values) for name, values in normalized.items()}
    overlaps = {
        "train_val": sets["train"] & sets["val"],
        "train_test": sets["train"] & sets["test"],
        "val_test": sets["val"] & sets["test"],
    }
    bad_overlaps = {name: values for name, values in overlaps.items() if values}
    if bad_overlaps:
        summary = {name: len(values) for name, values in bad_overlaps.items()}
        raise ValueError(f"image splits overlap: {summary}")

    all_ids = sets["train"] | sets["val"] | sets["test"]
    count = len(all_ids)
    expected_counts = {
        "train": int(count * 0.8),
        "val": int(count * 0.1),
    }
    expected_counts["test"] = count - expected_counts["train"] - expected_counts["val"]
    actual_counts = {name: len(normalized[name]) for name in SPLIT_NAMES}
    if actual_counts != expected_counts:
        raise ValueError(
            f"image split is not strict 80/10/10: "
            f"actual={actual_counts}, expected={expected_counts}"
        )

    if expected_image_ids is not None:
        expected = {int(value) for value in expected_image_ids}
        if all_ids != expected:
            raise ValueError(
                "image split ID coverage does not match the selected dataset: "
                f"missing={len(expected - all_ids)}, extra={len(all_ids - expected)}"
            )

    return actual_counts


def ensure_strict_811_split(
    splits_path: Union[os.PathLike[str], str],
    image_ids: Iterable[int],
    *,
    seed: int = 42,
    shared_splits_path: Optional[Union[os.PathLike[str], str]] = None,
) -> tuple[dict[str, list[int]], Optional[Path]]:
    """Atomically install the canonical split, backing up an old split.

    If ``shared_splits_path`` is provided, it acts as a cross-model master.
    An existing master with a different image-ID universe is rejected instead
    of silently changing the evaluation cohort.

    Returns ``(splits, backup_path)`` for the output split.  ``backup_path`` is
    ``None`` when no replacement was needed.
    """

    ids = [int(image_id) for image_id in image_ids]
    canonical = build_strict_811_split(ids, seed=seed)
    validate_strict_811_split(canonical, expected_image_ids=ids)

    if shared_splits_path is not None:
        shared_path = Path(shared_splits_path)
        if shared_path.exists():
            shared = _load_json(shared_path)
            shared_ids = _split_id_union(shared)
            if shared_ids != set(ids):
                raise ValueError(
                    f"Shared split {shared_path} belongs to a different COCO cohort: "
                    f"selected={len(set(ids))}, shared={len(shared_ids)}, "
                    f"missing={len(set(ids) - shared_ids)}, "
                    f"extra={len(shared_ids - set(ids))}"
                )
        _install_if_needed(shared_path, canonical, ids)
        canonical = _load_json(shared_path)

    output_path = Path(splits_path)
    backup_path = _install_if_needed(output_path, canonical, ids)
    installed = _load_json(output_path)
    validate_strict_811_split(installed, expected_image_ids=ids)
    return installed, backup_path


def _install_if_needed(
    path: Path,
    canonical: Mapping[str, Sequence[int]],
    expected_ids: Iterable[int],
) -> Optional[Path]:
    if path.exists():
        current = _load_json(path)
        if _same_split(current, canonical):
            validate_strict_811_split(current, expected_image_ids=expected_ids)
            return None
        backup_path = _next_backup_path(path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
    else:
        backup_path = None

    _atomic_write_json(path, canonical)
    return backup_path


def _same_split(
    left: Mapping[str, Sequence[int]],
    right: Mapping[str, Sequence[int]],
) -> bool:
    try:
        return all(
            [int(value) for value in left[name]]
            == [int(value) for value in right[name]]
            for name in SPLIT_NAMES
        ) and set(left) == set(SPLIT_NAMES)
    except (KeyError, TypeError, ValueError):
        return False


def _split_id_union(splits: Mapping[str, Sequence[int]]) -> set[int]:
    try:
        return {
            int(value)
            for name in SPLIT_NAMES
            for value in splits.get(name, [])
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Shared image split is not a valid mapping") from exc


def _next_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.stem}.pre_811{path.suffix}.bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(
            f"{path.stem}.pre_811.{index}{path.suffix}.bak"
        )
        index += 1
    return candidate


def _atomic_write_json(path: Path, value: Mapping[str, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value
