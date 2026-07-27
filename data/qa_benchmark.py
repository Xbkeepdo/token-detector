"""Shared POPE/CLEVR/AMBER question schema, deterministic splits, and labels."""

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

_CLEVR_EXISTENTIAL_RE = re.compile(
    r"\b(?:is|are)\s+there\b|\bare\s+(?:any|some)\b", re.IGNORECASE
)
_CLEVR_GENERIC_OBJECT_RE = re.compile(
    r"\b(?:anything|something|objects?|things?|shapes?|cubes?|blocks?|"
    r"spheres?|balls?|cylinders?)\b",
    re.IGNORECASE,
)
_CLEVR_SHAPE_SURFACES = {
    "cube": ("cube", "cubes", "block", "blocks"),
    "sphere": ("sphere", "spheres", "ball", "balls"),
    "cylinder": ("cylinder", "cylinders"),
}


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
                    "object_surface": match.group(1),
                    "object_char_start": int(match.start(1)),
                    "object_char_end": int(match.end(1)),
                    "object_span_status": "found",
                    "object_span_protocol": "pope_official_query_surface_v1",
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
        "val": [],
        "test": sorted(images[400:]),
        "seed": seed,
        "unit": "image_id",
        "protocol": "strict_outer_image_level_82",
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
    _validate_counts(questions, {"train": 7200, "val": 0, "test": 1800})
    _write_prepared(output_dir, questions, image_splits, seed, "POPE official 9K")
    return questions


def prepare_clevr_exist(
    clevr_root: str,
    output_dir: str,
    seed: int = 42,
    train_count: int = 7200,
    val_count: int = 0,
    test_count: int = 1800,
    dataset_name: str = "clevr_exist_9k",
) -> list[dict]:
    if int(val_count) != 0:
        raise ValueError(
            "Strict outer 8:2 CLEVR protocol requires val_count=0; "
            "this protocol does not use a validation split"
        )
    dataset_name = str(dataset_name).strip()
    if not dataset_name:
        raise ValueError("CLEVR dataset_name cannot be empty")
    root = Path(clevr_root)
    train = _load_clevr_exist(root, "train", dataset_name=dataset_name)
    official_val = _load_clevr_exist(root, "val", dataset_name=dataset_name)
    rng = random.Random(seed)
    selected_train = rng.sample(train, train_count)

    selected_val: list[dict] = []
    selected_test = rng.sample(official_val, test_count)

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
            "val": [],
            "test": sorted({row["image_id"] for row in selected_test}),
        },
        "seed": seed,
        "unit": "official_source_split_outer_82",
        "protocol": "strict_outer_image_level_82",
    }
    _validate_counts(questions, {"train": train_count, "val": val_count, "test": test_count})
    assert_no_image_leakage(questions)
    _write_prepared(
        output_dir,
        questions,
        image_splits,
        seed,
        f"CLEVR official exist {len(questions)}-question subset",
    )
    return questions


def prepare_amber_discriminative(
    amber_root: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict]:
    """Prepare all official AMBER discriminative Yes/No questions.

    AMBER has a variable number of questions per image.  The outer 8:2 split
    is therefore defined over the 1004 physical images, keeping every
    existence, attribute, and relation question for one image in one split.
    """

    root = Path(amber_root)
    query_path = root / "data" / "query" / "query_discriminative.json"
    annotation_path = root / "data" / "annotations.json"
    image_root = root / "images"
    with query_path.open(encoding="utf-8") as handle:
        queries = json.load(handle)
    with annotation_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(queries, list) or len(queries) != 14216:
        raise ValueError(
            f"Expected 14216 AMBER discriminative queries, got {len(queries)}"
        )
    if not isinstance(annotations, list) or len(annotations) < 15220:
        raise ValueError("AMBER annotations.json is incomplete")

    questions: list[dict] = []
    image_ids: set[int] = set()
    for item in queries:
        official_id = int(item["id"])
        annotation = annotations[official_id - 1]
        if int(annotation.get("id", official_id)) != official_id:
            raise ValueError(f"AMBER annotation ID mismatch at {official_id}")
        gt = normalize_yes_no(annotation.get("truth"))
        if gt is None:
            raise ValueError(f"AMBER question {official_id} has no Yes/No truth")
        image_file = str(item["image"])
        match = re.fullmatch(r"AMBER_(\d+)\.jpg", image_file)
        if match is None:
            raise ValueError(f"Unexpected AMBER image filename: {image_file!r}")
        image_id = int(match.group(1))
        image_path = image_root / image_file
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        amber_type = str(annotation.get("type") or "").strip()
        dimension = _amber_dimension(amber_type)
        row = {
            "dataset": "amber_discriminative",
            "source_split": dimension,
            "question_id": official_id,
            "image_id": image_id,
            "image_file": image_file,
            "image_path": str(image_path),
            "question": str(item["query"]),
            "gt_answer": gt,
            "object_word": None,
            "object_surface": None,
            "object_char_start": None,
            "object_char_end": None,
            "object_span_status": "unavailable",
            "object_span_protocol": "amber_prompt_last_only_v1",
            "question_family_index": None,
            "amber_question_type": amber_type,
            "amber_dimension": dimension,
        }
        row["key"] = question_key(row)
        questions.append(row)
        image_ids.add(image_id)

    if image_ids != set(range(1, 1005)):
        raise ValueError(
            "AMBER discriminative queries must cover image IDs 1..1004"
        )
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * 0.8)
    image_splits = {
        "train": sorted(shuffled[:train_count]),
        "val": [],
        "test": sorted(shuffled[train_count:]),
        "seed": seed,
        "unit": "amber_physical_image_id",
        "protocol": "strict_outer_image_level_82",
    }
    split_lookup = {
        image_id: split
        for split in ("train", "test")
        for image_id in image_splits[split]
    }
    for row in questions:
        row["probe_split"] = split_lookup[int(row["image_id"])]
    assert_no_image_leakage(questions)
    _write_prepared(
        output_dir,
        questions,
        image_splits,
        seed,
        "AMBER official discriminative VQA 14,216",
    )
    return questions


def _amber_dimension(question_type: str) -> str:
    value = str(question_type).strip()
    if value == "discriminative-hallucination":
        return "existence"
    if value.startswith("discriminative-attribute-"):
        return "attribute"
    if value in {"discriminative-relation", "relation"}:
        return "relation"
    raise ValueError(f"Unknown AMBER discriminative type: {value!r}")


def assert_no_image_leakage(rows: Sequence[dict]) -> None:
    split_images = {
        split: {
            _physical_image_identity(row)
            for row in rows
            if row["probe_split"] == split
        }
        for split in ("train", "val", "test")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_images[left] & split_images[right]
        if overlap:
            raise ValueError(f"Image leakage between {left}/{right}: {sorted(overlap)[:5]}")


def _physical_image_identity(row: dict) -> tuple:
    dataset = str(row.get("dataset") or "")
    image_id = int(row["image_id"])
    if dataset in {"pope", "amber_discriminative"}:
        return dataset, image_id
    # Official CLEVR train and val are separate physical namespaces.
    return dataset, str(row.get("source_split") or ""), image_id


def _load_clevr_exist(
    root: Path,
    source_split: str,
    dataset_name: str = "clevr_exist_9k",
) -> list[dict]:
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
        query_span = infer_clevr_query_object_span(
            str(item["question"]),
            program,
        )
        row = {
            "dataset": str(dataset_name),
            "source_split": source_split,
            "question_id": int(item["question_index"]),
            "image_id": int(item["image_index"]),
            "image_file": item["image_filename"],
            "image_path": str(root / "images" / source_split / item["image_filename"]),
            "question": item["question"],
            "gt_answer": gt,
            "object_word": query_span.get("surface"),
            "object_surface": query_span.get("surface"),
            "object_char_start": query_span.get("char_start"),
            "object_char_end": query_span.get("char_end"),
            "object_span_status": query_span["status"],
            "object_span_protocol": query_span["protocol"],
            "object_query_shape": query_span.get("query_shape"),
            "question_family_index": int(item["question_family_index"]),
        }
        row["key"] = question_key(row)
        rows.append(row)
    return rows



def infer_clevr_query_object_span(question: str, program: Sequence[dict]) -> dict:
    """Locate the queried entity head for a terminal CLEVR ``exist`` program.

    Search starts after the final existential phrase so an earlier reference
    object cannot be selected accidentally.
    """

    text = str(question)
    protocol = "clevr_terminal_exist_query_head_v1"
    if not program or str(program[-1].get("function")) != "exist":
        return {
            "status": "ambiguous",
            "protocol": protocol,
            "reason": "terminal_program_is_not_exist",
        }

    query_shape = _terminal_query_shape(program)
    triggers = list(_CLEVR_EXISTENTIAL_RE.finditer(text))
    if not triggers:
        return {
            "status": "ambiguous",
            "protocol": protocol,
            "query_shape": query_shape,
            "reason": "no_existential_trigger",
        }
    clause_start = triggers[-1].end()
    clause = text[clause_start:]

    candidates: list[re.Match[str]] = []
    if query_shape is not None:
        surfaces = _CLEVR_SHAPE_SURFACES.get(query_shape, (query_shape,))
        pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(value) for value in surfaces) + r")\b",
            re.IGNORECASE,
        )
        candidates = list(pattern.finditer(clause))
    if not candidates:
        candidates = list(_CLEVR_GENERIC_OBJECT_RE.finditer(clause))
    if not candidates:
        return {
            "status": "ambiguous",
            "protocol": protocol,
            "query_shape": query_shape,
            "reason": "no_query_head_after_existential_trigger",
        }

    selected = candidates[0]
    start = clause_start + selected.start()
    end = clause_start + selected.end()
    return {
        "status": "found",
        "protocol": protocol,
        "surface": text[start:end],
        "char_start": int(start),
        "char_end": int(end),
        "query_shape": query_shape,
    }


def _terminal_query_shape(program: Sequence[dict]) -> str | None:
    """Return an explicit shape filter on the set consumed by ``exist``."""

    try:
        inputs = list(program[-1].get("inputs") or [])
        current = int(inputs[0]) if inputs else len(program) - 2
    except (TypeError, ValueError, IndexError):
        return None
    visited: set[int] = set()
    while 0 <= current < len(program) and current not in visited:
        visited.add(current)
        node = program[current]
        function = str(node.get("function", ""))
        if function == "filter_shape":
            values = list(node.get("value_inputs") or [])
            return str(values[0]).lower() if values else None
        if function.startswith("filter_"):
            node_inputs = list(node.get("inputs") or [])
            if not node_inputs:
                return None
            current = int(node_inputs[0])
            continue
        if function.startswith("same_") or function in {
            "relate", "unique", "intersect", "union", "scene",
        }:
            return None
        node_inputs = list(node.get("inputs") or [])
        if len(node_inputs) != 1:
            return None
        current = int(node_inputs[0])
    return None


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
