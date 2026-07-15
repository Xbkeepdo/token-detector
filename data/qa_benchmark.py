"""Shared POPE/CLEVR-Exist question schema, deterministic splits, and labels."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


YES_NO_RE = re.compile(r"(?<![a-z])(yes|no)(?![a-z])", re.IGNORECASE)
POPE_OBJECT_RE = re.compile(r"Is there (?:a|an)\s+(.+?)\s+in the image", re.IGNORECASE)


def question_key(row: dict) -> str:
    """Return the stable key required by every pipeline stage."""
    return f"{row['dataset']}::{row['source_split']}::{row['question_id']}"


def normalize_yes_no(text: str | None) -> str | None:
    """Parse a generated answer without treating substrings such as 'nobody' as no."""
    if text is None:
        return None
    match = YES_NO_RE.search(str(text).strip().lower())
    return match.group(1).lower() if match else None


def label_answer(prediction: str | None, gt_answer: str) -> tuple[int, str]:
    """Return (1=real/0=hallucination, error_type)."""
    gt = normalize_yes_no(gt_answer)
    if gt not in ("yes", "no"):
        raise ValueError(f"GT must be yes/no, got {gt_answer!r}")
    pred = normalize_yes_no(prediction)
    if pred is None:
        return 0, "invalid"
    if pred == gt:
        return 1, f"correct_{gt}"
    return 0, "false_positive" if pred == "yes" else "false_negative"


def atomic_write_json(path: str | os.PathLike, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_jsonl(path: str | os.PathLike) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pope(
    pope_dir: str,
    coco_image_dir: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict]:
    strategies = ("random", "popular", "adversarial")
    by_strategy: dict[str, list[dict]] = {}
    image_sets = []
    for strategy in strategies:
        path = _find_pope_file(pope_dir, strategy)
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item["text"]
                match = POPE_OBJECT_RE.search(question)
                if match is None:
                    raise ValueError(f"Cannot parse queried object: {question!r}")
                gt = normalize_yes_no(item.get("label"))
                if gt is None:
                    raise ValueError(f"Missing yes/no GT in {path}:{line_idx + 1}")
                image_file = item["image"]
                image_id = _parse_image_id(image_file)
                row = {
                    "dataset": "pope",
                    "source_split": strategy,
                    "question_id": int(item.get("question_id", line_idx)),
                    "image_id": image_id,
                    "image_file": image_file,
                    "image_path": os.path.join(coco_image_dir, image_file),
                    "question": question,
                    "gt_answer": gt,
                    "object_word": match.group(1).strip().lower(),
                    "question_family_index": None,
                }
                row["key"] = question_key(row)
                rows.append(row)
        by_strategy[strategy] = rows
        image_sets.append({row["image_id"] for row in rows})

    if any(len(rows) != 3000 for rows in by_strategy.values()):
        raise ValueError({key: len(value) for key, value in by_strategy.items()})
    if not all(images == image_sets[0] for images in image_sets[1:]):
        raise ValueError("POPE strategies do not share exactly the same image IDs")
    images = sorted(image_sets[0])
    if len(images) != 500:
        raise ValueError(f"Expected 500 shared POPE images, got {len(images)}")
    random.Random(seed).shuffle(images)
    image_splits = {
        "train": sorted(images[:400]),
        "val": sorted(images[400:450]),
        "test": sorted(images[450:]),
        "seed": seed,
        "unit": "image_id",
    }
    split_lookup = {
        image_id: split
        for split in ("train", "val", "test")
        for image_id in image_splits[split]
    }
    questions = []
    for strategy in strategies:
        for row in by_strategy[strategy]:
            row["probe_split"] = split_lookup[row["image_id"]]
            questions.append(row)
    _validate_counts(questions, {"train": 7200, "val": 900, "test": 900})
    _write_prepared(output_dir, questions, image_splits, seed, "POPE official 9K")
    return questions


def prepare_clevr_exist(
    clevr_root: str,
    output_dir: str,
    seed: int = 42,
    train_count: int = 4000,
    val_count: int = 500,
    test_count: int = 500,
) -> list[dict]:
    root = Path(clevr_root)
    train = _load_clevr_exist(root, "train")
    official_val = _load_clevr_exist(root, "val")
    rng = random.Random(seed)
    selected_train = rng.sample(train, train_count)

    val_images = sorted({row["image_id"] for row in official_val})
    rng.shuffle(val_images)
    midpoint = len(val_images) // 2
    val_pool, test_pool = set(val_images[:midpoint]), set(val_images[midpoint:])
    val_candidates = [row for row in official_val if row["image_id"] in val_pool]
    test_candidates = [row for row in official_val if row["image_id"] in test_pool]
    selected_val = rng.sample(val_candidates, val_count)
    selected_test = rng.sample(test_candidates, test_count)

    questions = []
    for split, selected in (
        ("train", selected_train),
        ("val", selected_val),
        ("test", selected_test),
    ):
        for row in selected:
            row["probe_split"] = split
            questions.append(row)
    image_splits = {
        "train": sorted({row["image_id"] for row in selected_train}),
        "val": sorted({row["image_id"] for row in selected_val}),
        "test": sorted({row["image_id"] for row in selected_test}),
        "official_val_pool": {
            "val": sorted(val_pool),
            "test": sorted(test_pool),
        },
        "seed": seed,
        "unit": "official_split_then_image_pool",
    }
    _validate_counts(questions, {"train": train_count, "val": val_count, "test": test_count})
    assert_no_image_leakage(questions)
    _write_prepared(
        output_dir,
        questions,
        image_splits,
        seed,
        "CLEVR official exist 5K subset",
    )
    return questions


def assert_no_image_leakage(rows: Sequence[dict]) -> None:
    split_images = {
        split: {(row["source_split"], int(row["image_id"])) for row in rows if row["probe_split"] == split}
        for split in ("train", "val", "test")
    }
    # Official CLEVR train and val are physically disjoint; keep source split in identity.
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_images[left] & split_images[right]
        if overlap:
            raise ValueError(f"Image leakage between {left}/{right}: {sorted(overlap)[:5]}")


def _load_clevr_exist(root: Path, source_split: str) -> list[dict]:
    path = root / "questions" / f"CLEVR_{source_split}_questions.json"
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = []
    for item in raw["questions"]:
        program = item.get("program") or []
        if not program or program[-1].get("function") != "exist":
            continue
        gt = normalize_yes_no(item.get("answer"))
        if gt is None:
            continue
        row = {
            "dataset": "clevr_exist_5k",
            "source_split": source_split,
            "question_id": int(item["question_index"]),
            "image_id": int(item["image_index"]),
            "image_file": item["image_filename"],
            "image_path": str(root / "images" / source_split / item["image_filename"]),
            "question": item["question"],
            "gt_answer": gt,
            "object_word": None,
            "question_family_index": int(item["question_family_index"]),
        }
        row["key"] = question_key(row)
        rows.append(row)
    return rows


def _write_prepared(
    output_dir: str,
    questions: list[dict],
    image_splits: dict,
    seed: int,
    name: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len({row["key"] for row in questions}) != len(questions):
        raise ValueError("Question keys are not unique")
    atomic_write_jsonl(output / "questions.jsonl", questions)
    atomic_write_json(output / "image_splits.json", image_splits)
    atomic_write_json(
        output / "manifest.json",
        {
            "name": name,
            "seed": seed,
            "num_questions": len(questions),
            "counts": _counts(questions),
            "questions_sha256": sha256_file(output / "questions.jsonl"),
            "image_splits_sha256": sha256_file(output / "image_splits.json"),
        },
    )


def _find_pope_file(pope_dir: str, strategy: str) -> str:
    for name in (f"coco_pope_{strategy}.json", f"pope_{strategy}.json", f"coco_pope_{strategy}.jsonl"):
        path = os.path.join(pope_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"POPE {strategy} file not found in {pope_dir}")


def _parse_image_id(filename: str) -> int:
    match = re.search(r"(\d{12})", filename)
    if not match:
        raise ValueError(f"Cannot parse COCO image ID: {filename}")
    return int(match.group(1))


def _counts(rows: Sequence[dict]) -> dict[str, int]:
    return {split: sum(row["probe_split"] == split for row in rows) for split in ("train", "val", "test")}


def _validate_counts(rows: Sequence[dict], expected: dict[str, int]) -> None:
    actual = _counts(rows)
    if actual != expected:
        raise ValueError(f"Split counts mismatch: expected={expected}, actual={actual}")
