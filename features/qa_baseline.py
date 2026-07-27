"""QA adapters for the token-level paper baseline subsystem.

The caption baselines operate on one object-token span. For QA we adapt that
single-token span to the first generated token, whose prediction state is the
last token of the complete prompt. Labels preserve the raw convention
(``0=hallucination, 1=real``).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
from typing import Any, Mapping, Optional, Sequence

from models.base_wrapper import ModelOutput

from .baseline.runtime import BaselineRuntime, normalize_baseline_methods
from .baseline.schema import get_baseline_payload, validate_baseline_record


QA_BASELINE_PROTOCOL = "qa_prompt_last_token_image_level_probe_split_v3"
QA_CACHE_ID_SCHEME = "sha256(question_key)-63bit-v1"
QA_LABEL_PROTOCOLS = (
    "answer_correctness_all",
    "object_hallucination_yes_only",
)
DEFAULT_QA_LABEL_PROTOCOL = "answer_correctness_all"
QA_LABEL_FIELDS = {
    "answer_correctness_all": "label",
    "object_hallucination_yes_only": "object_hallucination_yes_only_label",
}
DEFAULT_QA_BASELINES = (
    "metatoken",
    "svar",
    "dhcp",
    "projectaway",
    "halloc",
)
QA_SPLITS = ("train", "val", "test")


def normalize_qa_label_protocol(value: str) -> str:
    protocol = str(value).strip().lower().replace("-", "_")
    if protocol not in QA_LABEL_PROTOCOLS:
        raise ValueError(
            f"Unknown QA label protocol {value!r}; expected one of "
            f"{list(QA_LABEL_PROTOCOLS)}"
        )
    return protocol


def qa_label_for_protocol(
    label_row: Mapping[str, Any],
    protocol: str,
) -> Optional[int]:
    """Return the selected raw label, or ``None`` for excluded yes-only rows."""

    normalized = normalize_qa_label_protocol(protocol)
    field = QA_LABEL_FIELDS[normalized]
    if field not in label_row:
        raise KeyError(
            f"QA label row {label_row.get('key')!r} is missing {field!r} "
            f"required by protocol {normalized!r}"
        )
    value = label_row[field]
    if value is None:
        if normalized == "object_hallucination_yes_only":
            return None
        raise ValueError(
            f"QA label row {label_row.get('key')!r} has no {field!r} value"
        )
    if isinstance(value, (bool, np.bool_)) or int(value) not in (0, 1):
        raise ValueError(
            f"QA label {field!r} for {label_row.get('key')!r} must be "
            f"0=hallucination/1=real or None, got {value!r}"
        )
    return int(value)


def qa_image_identity(row: Mapping[str, Any]) -> str:
    """Return the physical-image identity used for leakage checks.

    POPE strategies share the same COCO image pool, so strategy is excluded.
    CLEVR train/val indices occupy different namespaces and retain source_split.
    """

    dataset = str(row.get("dataset") or "").strip()
    if not dataset:
        raise ValueError("QA row is missing dataset")
    try:
        image_id = int(row["image_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("QA row has no valid image_id") from exc
    if dataset in {"pope", "amber_discriminative"}:
        return f"{dataset}::{image_id}"
    source_split = str(row.get("source_split") or "").strip()
    if not source_split:
        raise ValueError(f"{dataset} QA row is missing source_split")
    return f"{dataset}::{source_split}::{image_id}"


def qa_cache_id(question_key: str) -> int:
    """Map a question key to a stable positive integer cache namespace."""

    key = str(question_key).strip()
    if not key:
        raise ValueError("question key cannot be empty")
    value = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)
    return value or 1


def validate_qa_cache_ids(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Reject duplicate keys or the astronomically unlikely 63-bit collision."""

    result: dict[str, int] = {}
    owners: dict[int, str] = {}
    for row in rows:
        key = str(row.get("key") or "").strip()
        if not key:
            raise ValueError("Every QA row must have a non-empty key")
        if key in result:
            raise ValueError(f"Duplicate QA question key: {key}")
        cache_id = qa_cache_id(key)
        previous = owners.get(cache_id)
        if previous is not None and previous != key:
            raise RuntimeError(
                "Question cache-ID collision; choose a wider cache scheme: "
                f"{previous!r} and {key!r} -> {cache_id}"
            )
        result[key] = cache_id
        owners[cache_id] = key
    return result


def resolve_qa_answer_index(
    generation: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[list[int], int]:
    """Validate or recover the actual generated yes/no response token index."""

    raw_ids = generation.get("response_token_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("QA generation has no actual response_token_ids")
    try:
        response_ids = [int(value) for value in raw_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("QA generation has invalid response_token_ids") from exc

    saved_index = generation.get("answer_token_index")
    if saved_index is not None:
        index = int(saved_index)
        if index < 0 or index >= len(response_ids):
            raise ValueError(
                f"Saved answer_token_index {index} is outside response length "
                f"{len(response_ids)}"
            )
    else:
        # Reuse the exact semantic-token locator used by generation without
        # changing the protected QA extractor.
        from features.qa_extractor import find_answer_semantic_token

        index = find_answer_semantic_token(
            response_ids,
            tokenizer,
            generation.get("prediction"),
        )
        if index is None:
            raise ValueError(
                "No actual generated yes/no answer token is available; arbitrary "
                "content-token fallback is disabled"
            )
        index = int(index)

    saved_token_id = generation.get("answer_token_id")
    if saved_token_id is not None and int(saved_token_id) != response_ids[index]:
        raise ValueError(
            "Saved answer_token_id does not match response_token_ids at the "
            f"answer index: {saved_token_id} != {response_ids[index]}"
        )
    from features.qa_extractor import find_answer_semantic_token

    semantic_index = find_answer_semantic_token(
        response_ids, tokenizer, generation.get("prediction")
    )
    if semantic_index != index:
        raise ValueError(
            "Saved answer_token_index is not the actual first generated yes/no "
            f"token: saved={index}, recovered={semantic_index}"
        )
    return response_ids, index


class QABaselineAdapter:
    """Build one enriched baseline record from the prompt-last causal state."""

    def __init__(
        self,
        runtime: BaselineRuntime,
        *,
        label_protocol: str = DEFAULT_QA_LABEL_PROTOCOL,
    ) -> None:
        self.runtime = runtime
        self.label_protocol = normalize_qa_label_protocol(label_protocol)
        if runtime.official_svar_enabled:
            raise ValueError(
                "QA baselines support controlled SVAR only; official SVAR's "
                "caption object-search protocol is not defined for yes/no QA."
            )

    @property
    def requirements(self):
        return self.runtime.requirements_for(controlled=True, official=False)

    def build_record(
        self,
        *,
        image: Any,
        question: Mapping[str, Any],
        generation: Mapping[str, Any],
        label_row: Mapping[str, Any],
        response_token_ids: Sequence[int],
        target_index: int,
        model_output: ModelOutput,
        cache_id: Optional[int] = None,
    ) -> dict[str, Any]:
        key = _matching_question_key(question, generation, label_row)
        label = qa_label_for_protocol(label_row, self.label_protocol)
        if label is None:
            raise ValueError(
                f"QA question {key} is excluded by label protocol "
                f"{self.label_protocol!r}"
            )
        probe_split = str(question.get("probe_split") or "").strip()
        if probe_split not in QA_SPLITS:
            raise ValueError(f"Invalid probe_split for {key}: {probe_split!r}")
        if str(label_row.get("probe_split") or probe_split) != probe_split:
            raise ValueError(f"Question/label probe_split mismatch for {key}")

        response_ids = [int(value) for value in response_token_ids]
        target_index = int(target_index)
        if target_index != 0:
            raise ValueError(
                "Prompt-last QA baseline requires response target index 0, "
                f"got {target_index}"
            )
        if not response_ids:
            raise ValueError("Prompt-last QA baseline requires a response token")
        if int(model_output.response_token_idx) != target_index:
            raise ValueError(
                "Wrapper output does not represent the requested prompt-last "
                f"prediction position: {model_output.response_token_idx} != "
                f"{target_index}"
            )
        synthetic_cache_id = int(cache_id if cache_id is not None else qa_cache_id(key))
        word = str(model_output.token_str or "first_response_token")
        records = self.runtime.build_image_records(
            image=image,
            image_id=synthetic_cache_id,
            response_token_ids=response_ids,
            spans=(
                {
                    "word": word,
                    "token_indices": [target_index],
                    "occurrence_count": 1,
                    "label": label,
                },
            ),
            model_outputs=(model_output,),
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected one QA baseline record, got {len(records)}")
        record = records[0]
        # Runtime uses image_id for its legacy HalLoc cache filename. Restore
        # the real image ID after the unique question cache has been written.
        record["image_id"] = int(question["image_id"])
        record.update(
            {
                "key": key,
                "dataset": str(question["dataset"]),
                "source_split": str(question["source_split"]),
                "question_id": int(question["question_id"]),
                "probe_split": probe_split,
                "qa_image_identity": qa_image_identity(question),
                "qa_label_protocol": self.label_protocol,
            }
        )
        metadata = record.setdefault("metadata", {})
        metadata["qa"] = {
            "protocol": QA_BASELINE_PROTOCOL,
            "target": "first_generated_token_from_prompt_last_state",
            "question_key": key,
            "cache_id": synthetic_cache_id,
            "cache_id_scheme": QA_CACHE_ID_SCHEME,
            "image_identity": record["qa_image_identity"],
            "probe_split": probe_split,
            "prompt_last_response_target_index": target_index,
            "semantic_answer_token_index": generation.get("answer_token_index"),
            "target_token_id": int(response_ids[target_index]),
            "predicted_token_id": int(model_output.token_id),
            "predicted_token": str(model_output.token_str),
            "label_protocol": self.label_protocol,
            "label_field": QA_LABEL_FIELDS[self.label_protocol],
        }
        if "svar" in record.get("baselines", {}):
            metadata["svar_protocol"] = "qa_controlled_prompt_last_token"
        validate_baseline_record(record, required=self.runtime.methods)
        return record


class QABaselineFeatureStore:
    """Atomic, append-only question shards with deterministic consolidation."""

    def __init__(
        self,
        baseline_dir: str | os.PathLike[str],
        *,
        shard_size: int = 25,
        resume: bool = True,
        part_prefix: str = "part",
    ) -> None:
        self.baseline_dir = Path(baseline_dir)
        self.parts_dir = self.baseline_dir / "feature_parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self.part_prefix = str(part_prefix).strip()
        if not self.part_prefix or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in self.part_prefix
        ):
            raise ValueError(f"Invalid QA baseline shard prefix: {part_prefix!r}")
        existing_paths = sorted(self.parts_dir.glob("part-*.pkl"))
        if existing_paths and not resume:
            raise FileExistsError(
                f"QA baseline feature parts already exist: {self.parts_dir}"
            )
        self.rows: dict[str, dict[str, Any]] = {}
        for path in existing_paths:
            with path.open("rb") as handle:
                values = pickle.load(handle)
            if not isinstance(values, list):
                raise ValueError(f"QA baseline shard must contain a list: {path}")
            for raw_record in values:
                record = dict(raw_record)
                key = str(record.get("key") or "")
                if not key:
                    raise ValueError(f"QA baseline shard record has no key: {path}")
                if key in self.rows:
                    raise ValueError(f"Duplicate QA baseline feature key: {key}")
                self.rows[key] = record
        self.pending: list[dict[str, Any]] = []
        self.next_part = len(
            list(self.parts_dir.glob(f"{self.part_prefix}-*.pkl"))
        )

    def add(self, record: Mapping[str, Any]) -> bool:
        key = str(record.get("key") or "").strip()
        if not key:
            raise ValueError("QA baseline feature record has no key")
        if key in self.rows:
            return False
        value = deepcopy(dict(record))
        self.rows[key] = value
        self.pending.append(value)
        if len(self.pending) >= self.shard_size:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.parts_dir / (
            f"{self.part_prefix}-{self.next_part:06d}.pkl"
        )
        _atomic_pickle(path, self.pending)
        self.pending = []
        self.next_part += 1

    def consolidate(self) -> Path:
        self.flush()
        path = self.baseline_dir / "features.pkl"
        _atomic_pickle(path, [self.rows[key] for key in sorted(self.rows)])
        return path


def split_qa_baseline_records(
    records: Sequence[Mapping[str, Any]],
    *,
    label_protocol: Optional[str] = None,
) -> dict[str, list[Mapping[str, Any]]]:
    """Split by authoritative per-question ``probe_split`` with image audit."""

    expected_protocol = (
        normalize_qa_label_protocol(label_protocol)
        if label_protocol is not None
        else None
    )
    result: dict[str, list[Mapping[str, Any]]] = {name: [] for name in QA_SPLITS}
    key_owner: dict[str, str] = {}
    image_owner: dict[str, str] = {}
    for record in records:
        validate_baseline_record(record)
        key = str(record.get("key") or "").strip()
        if not key:
            raise ValueError("QA baseline record is missing key")
        split = str(record.get("probe_split") or "").strip()
        if split not in result:
            raise ValueError(f"QA baseline record {key} has invalid split {split!r}")
        if key in key_owner:
            raise ValueError(f"Duplicate QA baseline record key: {key}")
        if (
            expected_protocol is not None
            and str(record.get("qa_label_protocol") or "") != expected_protocol
        ):
            raise ValueError(
                f"QA baseline record {key} does not belong to label protocol "
                f"{expected_protocol!r}"
            )
        identity = str(record.get("qa_image_identity") or "").strip()
        if not identity:
            identity = qa_image_identity(record)
        previous_split = image_owner.get(identity)
        if previous_split is not None and previous_split != split:
            raise ValueError(
                "QA image leakage across probe splits: "
                f"{identity} occurs in {previous_split} and {split}"
            )
        key_owner[key] = split
        image_owner[identity] = split
        result[split].append(record)
    if result["val"]:
        raise ValueError("Strict QA 8:2 requires an empty validation split")
    for split in ("train", "test"):
        if not result[split]:
            raise ValueError(f"QA baseline split {split!r} is empty")
    return result


def build_qa_probe_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    label_protocol: Optional[str] = None,
) -> dict[str, Any]:
    split_records = split_qa_baseline_records(
        records,
        label_protocol=label_protocol,
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "unit": "strict_82_question_split_with_physical_image_leakage_audit",
        "protocol": QA_BASELINE_PROTOCOL,
        "selection_protocol": "fixed_epochs_last_checkpoint_threshold_0.5",
    }
    if label_protocol is not None:
        payload["label_protocol"] = normalize_qa_label_protocol(label_protocol)
    image_counts: dict[str, int] = {}
    for split, values in split_records.items():
        payload[split] = sorted(str(value["key"]) for value in values)
        image_counts[split] = len(
            {str(value["qa_image_identity"]) for value in values}
        )
    payload["question_counts"] = {
        split: len(split_records[split]) for split in QA_SPLITS
    }
    payload["image_counts"] = image_counts
    return payload


def validate_halloc_cache_uniqueness(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Ensure no two QA questions address the same HalLoc response cache."""

    owners: dict[str, str] = {}
    for record in records:
        if "halloc" not in (record.get("baselines") or {}):
            continue
        payload = get_baseline_payload(record, "halloc")
        cache_file = payload.get("cache_file")
        if cache_file is None:
            continue
        cache_name = str(cache_file)
        key = str(record.get("key") or "")
        previous = owners.get(cache_name)
        if previous is not None and previous != key:
            raise ValueError(
                "HalLoc QA cache collision would overwrite question-specific "
                f"response states: {cache_name!r} belongs to {previous!r} and "
                f"{key!r}"
            )
        owners[cache_name] = key


def qa_baseline_feature_config(
    config: Mapping[str, Any],
    methods: Sequence[str] | str,
) -> dict[str, Any]:
    """Return only settings that affect extracted QA baseline payloads."""

    selected = normalize_baseline_methods(methods)
    allowed = {
        "metatoken": ("length_penalty", "attention_layer"),
        "svar": (),
        "dhcp": ("target_grid", "spatial_size", "shard_size"),
        "projectaway": ("vocab_chunk_size", "row_chunk_size"),
        "halloc": ("clip_model",),
    }
    payload: dict[str, Any] = {"methods": list(selected)}
    for method in selected:
        section = config.get(method) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"QA baseline config {method!r} must be a mapping")
        payload[method] = {
            key: deepcopy(section[key])
            for key in allowed[method]
            if key in section
        }
    if "svar" in selected:
        payload["svar"]["protocols"] = ["controlled"]
        payload["svar"]["extraction_layers"] = "all"
    return payload


def _matching_question_key(*rows: Mapping[str, Any]) -> str:
    keys = {str(row.get("key") or "").strip() for row in rows}
    if "" in keys or len(keys) != 1:
        raise ValueError(f"Question/generation/label keys do not match: {keys}")
    return next(iter(keys))


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
