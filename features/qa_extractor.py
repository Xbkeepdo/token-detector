"""Unified Generate -> Label -> Extract implementation for yes/no QA benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.qa_benchmark import (
    atomic_write_jsonl,
    label_answer,
    load_jsonl,
    normalize_yes_no,
)
from features.ads import compute_ads
from features.cgc import compute_cgc
from features.dgst_t import _target_comparison_state
from features.extractor import (
    _compute_dgst_t_result,
    build_extraction_requirements,
)
from models.base_wrapper import PromptTargetRequest


QA_FEATURE_SCHEMA_VERSION = "qa-prompt-last-token-v6"
_IMAGE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}
DIRECT_SOFTMAX_METHOD = "hpre_softmax_prob_direct"


class JSONLCheckpointStore:
    """Deduplicated JSONL store whose checkpoints are temp-file + rename atomic."""

    def __init__(self, path: str, checkpoint_every: int = 10):
        self.path = path
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.rows = {row["key"]: row for row in load_jsonl(path)}
        self.dirty = 0

    def add(self, row: dict) -> None:
        if self.rows.get(row["key"]) == row:
            return
        self.rows[row["key"]] = row
        self.dirty += 1
        if self.dirty >= self.checkpoint_every:
            self.flush()

    def flush(self) -> None:
        if self.dirty:
            atomic_write_jsonl(self.path, self.rows.values())
            self.dirty = 0


class AtomicFeatureShards:
    """Resume-safe feature shards plus the required consolidated features.pkl."""

    def __init__(self, output_dir: str, shard_size: int = 25):
        self.output_dir = Path(output_dir)
        self.parts_dir = self.output_dir / "features.parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = max(1, int(shard_size))
        self.rows: dict[str, dict] = {}
        self.pending: list[dict] = []
        for path in sorted(self.parts_dir.glob("part-*.pkl")):
            with open(path, "rb") as handle:
                for row in pickle.load(handle):
                    self.rows[row["key"]] = row
        self.next_part = len(list(self.parts_dir.glob("part-*.pkl")))

    def add(self, row: dict) -> None:
        if row["key"] in self.rows:
            return
        self.rows[row["key"]] = row
        self.pending.append(row)
        if len(self.pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.parts_dir / f"part-{self.next_part:06d}.pkl"
        _atomic_pickle(path, self.pending)
        self.next_part += 1
        self.pending = []

    def consolidate(self) -> None:
        self.flush()
        _atomic_pickle(self.output_dir / "features.pkl", list(self.rows.values()))


def qa_prompt(model_key: str, question: str) -> str:
    """Return the raw user instruction; each wrapper renders its own template."""
    del model_key  # Kept in the public signature for compatible callers.
    return f"{question}\nAnswer only yes or no."


def generate_questions(
    model_wrapper,
    model_key: str,
    questions: list[dict],
    output_dir: str,
    checkpoint_every: int = 10,
) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    store = JSONLCheckpointStore(str(output / "generations.jsonl"), checkpoint_every)
    failures = JSONLCheckpointStore(str(output / "generation_failures.jsonl"), 1)
    for question in tqdm(questions, desc="Generate yes/no"):
        key = question["key"]
        prompt = qa_prompt(model_key, question["question"])
        existing = store.rows.get(key)
        if existing is not None and qa_generation_record_is_complete(
            existing, prompt
        ):
            continue
        try:
            with Image.open(question["image_path"]) as raw_image:
                image = raw_image.convert("RGB")
            generated = model_wrapper.generate(image, prompt=prompt)
            prediction = normalize_yes_no(generated.generated_text)
            semantic_index = find_answer_semantic_token(
                generated.response_token_ids,
                model_wrapper.tokenizer,
                prediction,
            )
            if semantic_index is None:
                raise ValueError(
                    "Model output does not contain a locatable yes/no answer token: "
                    f"{generated.generated_text!r}"
                )
            row = {
                "key": key,
                "dataset": question["dataset"],
                "source_split": question["source_split"],
                "question_id": question["question_id"],
                "image_id": question["image_id"],
                "probe_split": question["probe_split"],
                "prompt": prompt,
                "generation_protocol": "raw_question_yes_no_v1",
                "generated_text": generated.generated_text,
                "prediction": prediction,
                "response_token_ids": [int(x) for x in generated.response_token_ids],
                "response_tokens": list(generated.response_tokens),
                "answer_token_index": semantic_index,
                "answer_token_id": (
                    int(generated.response_token_ids[semantic_index])
                    if semantic_index is not None else None
                ),
            }
            store.add(row)
        except Exception as exc:
            failures.add({"key": key, "error": repr(exc), "traceback": traceback.format_exc()})
    store.flush()
    failures.flush()
    return list(store.rows.values())


def label_generations(
    questions: list[dict],
    output_dir: str,
    checkpoint_every: int = 100,
) -> list[dict]:
    output = Path(output_dir)
    generations = {row["key"]: row for row in load_jsonl(output / "generations.jsonl")}
    store = JSONLCheckpointStore(str(output / "labels.jsonl"), checkpoint_every)
    for question in questions:
        generation = generations.get(question["key"])
        if generation is None:
            continue
        prediction = generation.get("prediction")
        label, error_type = label_answer(prediction, question["gt_answer"])
        gt_answer = normalize_yes_no(question["gt_answer"])
        yes_only_label = (
            (1 if gt_answer == "yes" else 0)
            if prediction == "yes"
            else None
        )
        store.add({
            "key": question["key"],
            "dataset": question["dataset"],
            "source_split": question["source_split"],
            "question_id": question["question_id"],
            "image_id": question["image_id"],
            "probe_split": question["probe_split"],
            "question_family_index": question.get("question_family_index"),
            "gt_answer": question["gt_answer"],
            "prediction": prediction,
            "label": label,
            "class_name": "real" if label == 1 else "hallucination",
            "error_type": error_type,
            "answer_correctness_all_label": label,
            "object_hallucination_yes_only_label": yes_only_label,
            "object_hallucination_yes_only_class_name": (
                None
                if yes_only_label is None
                else ("real" if yes_only_label == 1 else "hallucination")
            ),
            "label_protocols": {
                "answer_correctness_all": label,
                "object_hallucination_yes_only": yes_only_label,
            },
        })
    store.flush()
    return list(store.rows.values())


def extract_questions(
    model_wrapper,
    model_key: str,
    questions: list[dict],
    output_dir: str,
    cfg_dgst_t: dict,
    cfg_ads: dict,
    cfg_cgc: dict,
    shard_size: int = 25,
    position_protocols: Iterable[str] = ("prompt_last_token",),
    extraction_fingerprint: str = "",
    method_enabled: bool = True,
    ads_cgc_enabled: bool = True,
    baseline_consumers: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Extract every enabled QA family from one shared prompt-last forward.

    ``prompt_last_token`` is the final causal state in the complete prompt and
    therefore predicts ``response_token_ids[0]``.  It deliberately does not
    move when a model emits a preamble before its semantic yes/no answer.  The
    optional ``question_object_pre_token`` ablation predicts the first
    contextual sub-token of the queried object word in the question.  The
    default QA YAML enables only ``prompt_last_token`` so no object forward is
    run in the first VQA experiment. ``baseline_consumers`` maps each active
    label protocol to an adapter/store pair. Their requirements are merged
    with DGST/ADS+CGC before the wrapper call, so enabling baselines does not
    trigger a second LVLM forward.
    """

    active_positions = tuple(dict.fromkeys(str(value) for value in position_protocols))
    allowed_positions = {"prompt_last_token", "question_object_pre_token"}
    unknown_positions = sorted(set(active_positions) - allowed_positions)
    if unknown_positions:
        raise ValueError(f"Unknown QA extraction positions: {unknown_positions}")
    if "prompt_last_token" not in active_positions:
        raise ValueError("QA extraction currently requires prompt_last_token")
    object_position_enabled = "question_object_pre_token" in active_positions
    extraction_fingerprint = str(extraction_fingerprint).strip()
    if not extraction_fingerprint:
        raise ValueError("QA extraction requires a non-empty extraction fingerprint")
    method_enabled = bool(method_enabled)
    ads_cgc_enabled = bool(ads_cgc_enabled)
    root_enabled = method_enabled or ads_cgc_enabled
    consumers = dict(baseline_consumers or {})
    if not root_enabled and not consumers:
        raise ValueError("QA extraction has no enabled feature family")
    for protocol, consumer in consumers.items():
        if "adapter" not in consumer or "store" not in consumer:
            raise ValueError(
                f"QA baseline consumer {protocol!r} requires adapter and store"
            )
    output = Path(output_dir)
    generations = {row["key"]: row for row in load_jsonl(output / "generations.jsonl")}
    labels = {row["key"]: row for row in load_jsonl(output / "labels.jsonl")}
    shards = AtomicFeatureShards(output_dir, shard_size) if root_enabled else None
    failures = JSONLCheckpointStore(str(output / "extraction_failures.jsonl"), 1)

    for question in tqdm(questions, desc="Extract QA features"):
        key = question["key"]
        generation = generations.get(key)
        label_row = labels.get(key)
        if generation is None or label_row is None:
            continue
        prompt = qa_prompt(model_key, question["question"])
        generation_fingerprint = qa_generation_fingerprint(generation, prompt)
        label_fingerprint = qa_label_fingerprint(label_row)
        question_input_fingerprint = qa_question_input_fingerprint(question, prompt)
        root_pending = bool(root_enabled)
        if shards is not None and key in shards.rows:
            existing = shards.rows[key]
            existing_schema = existing.get("feature_schema_version")
            if existing_schema != QA_FEATURE_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Cannot resume {key} from schema {existing_schema!r}; "
                    f"use a new output directory for {QA_FEATURE_SCHEMA_VERSION}."
                )
            if existing.get("generation_fingerprint") != generation_fingerprint:
                raise RuntimeError(
                    f"Cannot resume {key}: generation/prompt fingerprint changed; "
                    "use a new output directory or remove this experiment output."
                )
            if existing.get("label_fingerprint") != label_fingerprint:
                raise RuntimeError(
                    f"Cannot resume {key}: QA labels changed; use a new output "
                    "directory or remove this experiment output."
                )
            if existing.get("question_input_fingerprint") != question_input_fingerprint:
                raise RuntimeError(
                    f"Cannot resume {key}: question, image, or split changed; use "
                    "a new output directory or remove this experiment output."
                )
            if existing.get("extraction_fingerprint") != extraction_fingerprint:
                raise RuntimeError(
                    f"Cannot resume {key}: model or extraction configuration changed; "
                    "use a new output directory or remove this experiment output."
                )
            if tuple(existing.get("position_protocols") or ()) != active_positions:
                raise RuntimeError(
                    f"Cannot resume {key}: extraction positions changed from "
                    f"{existing.get('position_protocols')!r} to {list(active_positions)!r}; "
                    "use a new output directory or remove this experiment output."
                )
            object_retry = (
                object_position_enabled
                and str(
                    (existing.get("question_object_position") or {}).get(
                        "status"
                    )
                )
                == "extraction_failed"
            )
            if not object_retry:
                root_pending = False
            else:
                # A later part with the same key supersedes the failed record
                # when shards are reloaded in sorted order. Recompute both
                # positions so the row stays internally consistent.
                del shards.rows[key]
        pending_baselines: dict[str, Mapping[str, Any]] = {}
        for protocol, consumer in consumers.items():
            adapter = consumer["adapter"]
            store = consumer["store"]
            label = adapter.label_protocol
            from features.qa_baseline import qa_label_for_protocol

            if qa_label_for_protocol(label_row, label) is None:
                continue
            if key not in store.rows:
                pending_baselines[protocol] = consumer
        if not root_pending and not pending_baselines:
            continue
        response_ids = [int(x) for x in generation.get("response_token_ids", [])]
        if not qa_generation_record_is_complete(generation, prompt):
            failures.add({
                "key": key,
                "error": "Generation lacks a valid saved yes/no answer token index",
            })
            continue
        answer_index = int(generation["answer_token_index"])
        recovered_index = find_answer_semantic_token(
            response_ids, model_wrapper.tokenizer, generation.get("prediction")
        )
        if recovered_index != answer_index:
            failures.add({
                "key": key,
                "error": (
                    "Saved answer token is not the actual first generated yes/no token: "
                    f"saved={answer_index}, recovered={recovered_index}"
                ),
            })
            continue
        # response index 0 is predicted by the prompt's final causal row.  Do
        # not follow ``answer_index`` here: a later semantic yes/no token would
        # include generated preamble tokens and would no longer be a
        # prompt-last feature.
        prompt_last_response_index = 0
        prompt_last_target_token_id = int(response_ids[prompt_last_response_index])
        object_status = {
            "status": (
                str(question.get("object_span_status") or "unavailable")
                if object_position_enabled
                else "disabled_by_config"
            ),
            "protocol": question.get("object_span_protocol"),
            "surface": question.get("object_surface"),
            "char_start": question.get("object_char_start"),
            "char_end": question.get("object_char_end"),
        }
        positions: dict[str, dict] = {}

        try:
            with Image.open(question["image_path"]) as raw_image:
                image = raw_image.convert("RGB")

            requirements = build_extraction_requirements(
                method=bool(method_enabled and root_pending),
                ads_cgc=bool(ads_cgc_enabled and root_pending),
                baseline=False,
            )
            for consumer in pending_baselines.values():
                requirements = requirements.merged(
                    consumer["adapter"].requirements
                )
            prompt_last_outputs = model_wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=response_ids,
                response_token_indices=[prompt_last_response_index],
                target_token_ids=[prompt_last_target_token_id],
                cfg_dgst_t=(cfg_dgst_t if method_enabled and root_pending else None),
                prompt=prompt,
                requirements=requirements,
            )
            if len(prompt_last_outputs) != 1:
                raise RuntimeError(
                    "Expected one prompt-last-position output, got "
                    f"{len(prompt_last_outputs)}"
                )
            prompt_last_output = prompt_last_outputs[0]
            if root_pending:
                positions["prompt_last_token"] = _build_position_record(
                    model_out=prompt_last_output,
                    cfg_dgst_t=cfg_dgst_t,
                    cfg_ads=cfg_ads,
                    cfg_cgc=cfg_cgc,
                    method_enabled=method_enabled,
                    ads_cgc_enabled=ads_cgc_enabled,
                    target_metadata={
                        "protocol": "prompt_last_token_v1",
                        "target_token_id": prompt_last_target_token_id,
                        "response_token_index": prompt_last_response_index,
                        "semantic_answer_token_index": answer_index,
                        "prediction_source": "prompt_last_causal_row",
                    },
                )

            for protocol, consumer in pending_baselines.items():
                cache_ids = consumer.get("cache_ids") or {}
                baseline_record = consumer["adapter"].build_record(
                    image=image,
                    question=question,
                    generation=generation,
                    label_row=label_row,
                    response_token_ids=response_ids,
                    target_index=prompt_last_response_index,
                    model_output=prompt_last_output,
                    cache_id=cache_ids.get(key),
                )
                consumer["store"].add(baseline_record)

            if (
                root_pending
                and object_position_enabled
                and object_status["status"] == "found"
            ):
                try:
                    surface = str(question["object_surface"])
                    char_start = int(question["object_char_start"])
                    char_end = int(question["object_char_end"])
                    if prompt[char_start:char_end] != surface:
                        raise ValueError(
                            "Prepared object span does not match the raw QA prompt: "
                            f"{prompt[char_start:char_end]!r} != {surface!r}"
                        )
                    request = PromptTargetRequest(
                        prompt=prompt,
                        target_text=surface,
                        target_char_start=char_start,
                        target_char_end=char_end,
                    )
                    object_output = model_wrapper.extract_prompt_target_features(
                        image=image,
                        request=request,
                        cfg_dgst_t=(cfg_dgst_t if method_enabled else None),
                        requirements=requirements,
                    )
                    alignment = _prompt_alignment_metadata(object_output)
                    if "target_token_id" not in alignment:
                        raise RuntimeError(
                            "Prompt-target wrapper omitted the actual contextual target ID"
                        )
                    positions["question_object_pre_token"] = _build_position_record(
                        model_out=object_output,
                        cfg_dgst_t=cfg_dgst_t,
                        cfg_ads=cfg_ads,
                        cfg_cgc=cfg_cgc,
                        method_enabled=method_enabled,
                        ads_cgc_enabled=ads_cgc_enabled,
                        target_metadata={
                            "protocol": "question_object_contextual_first_subtoken_v1",
                            "surface": surface,
                            "question_char_start": char_start,
                            "question_char_end": char_end,
                            **alignment,
                        },
                    )
                    object_status.update(
                        {
                            "status": "extracted",
                            **alignment,
                        }
                    )
                    del object_output
                except Exception as object_exc:
                    object_status["status"] = "extraction_failed"
                    object_status["error"] = repr(object_exc)
                    failures.add(
                        {
                            "key": key,
                            "component": "question_object_pre_token",
                            "error": repr(object_exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

            if root_pending:
                yes_only_label = label_row.get("object_hallucination_yes_only_label")
                feature = {
                    "feature_schema_version": QA_FEATURE_SCHEMA_VERSION,
                    "generation_fingerprint": generation_fingerprint,
                    "label_fingerprint": label_fingerprint,
                    "question_input_fingerprint": question_input_fingerprint,
                    "extraction_fingerprint": extraction_fingerprint,
                    "feature_families": {
                        "method": method_enabled,
                        "ads_cgc": ads_cgc_enabled,
                    },
                    "position_protocols": list(active_positions),
                    "key": key,
                    "dataset": question["dataset"],
                    "source_split": question["source_split"],
                    "question_id": question["question_id"],
                    "image_id": question["image_id"],
                    "probe_split": question["probe_split"],
                    "question_family_index": question.get("question_family_index"),
                    "label": int(label_row["label"]),
                    "answer_correctness_all_label": int(label_row["label"]),
                    "object_hallucination_yes_only_label": (
                        None if yes_only_label is None else int(yes_only_label)
                    ),
                    "class_name": label_row["class_name"],
                    "error_type": label_row["error_type"],
                    "prediction": label_row.get("prediction"),
                    "generated_text": generation.get("generated_text"),
                    "positions": positions,
                    "question_object_position": object_status,
                    "ot_solver": (
                        str(cfg_dgst_t.get("ot_solver", "emd"))
                        if method_enabled
                        else None
                    ),
                }
                if "gt_answer" in feature or _contains_gt_key(feature):
                    raise AssertionError("GT leakage: feature record contains a GT field")
                _assert_finite(feature)
                assert shards is not None
                shards.add(feature)
            del prompt_last_output, prompt_last_outputs
        except Exception as exc:
            failures.add({"key": key, "error": repr(exc), "traceback": traceback.format_exc()})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if shards is not None:
        shards.consolidate()
    for consumer in consumers.values():
        consumer["store"].flush()
    failures.flush()
    return list(shards.rows.values()) if shards is not None else []


def qa_generation_fingerprint(generation: dict, prompt: str) -> str:
    """Fingerprint the exact prompt/response pair consumed by extraction."""

    payload = {
        "prompt": str(prompt),
        "response_token_ids": [
            int(value) for value in generation.get("response_token_ids", [])
        ],
        "generated_text": str(generation.get("generated_text") or ""),
        "prediction": generation.get("prediction"),
        "generation_protocol": generation.get("generation_protocol"),
        "answer_token_index": generation.get("answer_token_index"),
        "answer_token_id": generation.get("answer_token_id"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qa_generation_record_is_complete(generation: dict, prompt: str) -> bool:
    """Return whether a row names one actual saved yes/no response token."""

    if generation.get("prompt") != prompt:
        return False
    if normalize_yes_no(generation.get("prediction")) not in ("yes", "no"):
        return False
    raw_ids = generation.get("response_token_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return False
    try:
        response_ids = [int(value) for value in raw_ids]
        index = int(generation["answer_token_index"])
        token_id = int(generation["answer_token_id"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= index < len(response_ids) and response_ids[index] == token_id


def qa_question_input_fingerprint(question: dict, prompt: str) -> str:
    """Fingerprint the exact question, image bytes, and authoritative split."""

    image_path = Path(str(question.get("image_path") or "")).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"QA image is missing: {image_path}")
    stat = image_path.stat()
    cache_key = (str(image_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    image_sha256 = _IMAGE_SHA256_CACHE.get(cache_key)
    if image_sha256 is None:
        digest = hashlib.sha256()
        with image_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        image_sha256 = digest.hexdigest()
        _IMAGE_SHA256_CACHE[cache_key] = image_sha256
    fields = (
        "key", "dataset", "source_split", "question_id", "image_id",
        "probe_split", "question_family_index", "question", "image_path",
        "object_span_status", "object_span_protocol", "object_surface",
        "object_char_start", "object_char_end",
    )
    payload = {
        "prompt": str(prompt),
        "question": {field: question.get(field) for field in fields},
        "image": {
            "resolved_path": cache_key[0],
            "size": cache_key[1],
            "sha256": image_sha256,
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qa_label_fingerprint(label_row: dict) -> str:
    """Fingerprint the labels embedded into one QA feature row."""

    payload = {
        "label": label_row.get("label"),
        "answer_correctness_all_label": label_row.get(
            "answer_correctness_all_label", label_row.get("label")
        ),
        "object_hallucination_yes_only_label": label_row.get(
            "object_hallucination_yes_only_label"
        ),
        "prediction": label_row.get("prediction"),
        "class_name": label_row.get("class_name"),
        "error_type": label_row.get("error_type"),
        "label_protocols": label_row.get("label_protocols"),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_answer_semantic_token(response_ids: list[int], tokenizer, answer: str | None) -> int | None:
    if not response_ids:
        return None
    if answer in ("yes", "no"):
        candidates = []
        for text in (answer, " " + answer, "\n" + answer, answer.capitalize(), " " + answer.capitalize()):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids and ids not in candidates:
                candidates.append(ids)
        for candidate in candidates:
            for start in range(0, len(response_ids) - len(candidate) + 1):
                if response_ids[start : start + len(candidate)] == candidate:
                    return start
        for index, token_id in enumerate(response_ids):
            if normalize_yes_no(tokenizer.decode([token_id], skip_special_tokens=True)) == answer:
                return index
    return None


def _compact_dgst(result: dict) -> dict:
    """Serialize every active VV/VP/VPend DGST scope without aliases."""

    methods = tuple(str(value) for value in result.get("dgst_t_four_gate_methods") or ())
    if not methods:
        raise KeyError("Missing dgst_t_four_gate_methods in QA DGST result")
    metadata_keys = (
        "dgst_t_profile",
        "dgst_t_mad_axis",
        "dgst_t_mad_scale",
        "dgst_t_softmax_axis",
        "dgst_t_source_distribution_mode",
        "dgst_t_state_by_method",
        "dgst_t_transport_top_k",
        "dgst_t_target_region_top_k",
        "dgst_t_ev_definition",
        "dgst_t_cost",
        "dgst_t_ot_solver",
    )
    missing = [key for key in metadata_keys if key not in result]
    if missing:
        raise KeyError(f"Missing QA DGST metadata fields: {missing}")

    raw_scopes = tuple(
        str(value)
        for value in result.get("dgst_t_four_gate_support_scopes") or ()
    )
    if not raw_scopes:
        # Compatibility with captures created before support scopes were
        # serialized explicitly.
        inferred = []
        if "dgst_t_attention_support_per_layer" in result:
            inferred.append("visual")
        if "dgst_t_vp_attention_support_per_layer" in result:
            inferred.append("visual_prompt")
        if "dgst_t_vpend_attention_support_per_layer" in result:
            inferred.append("visual_prompt_end")
        raw_scopes = tuple(inferred)
    unknown_scopes = sorted(
        set(raw_scopes)
        - {"visual", "visual_prompt", "visual_prompt_end"}
    )
    if unknown_scopes:
        raise ValueError(f"Unknown QA DGST support scopes: {unknown_scopes}")
    scope_specs = []
    if "visual" in raw_scopes:
        scope_specs.append(("vv", "", ""))
    if "visual_prompt" in raw_scopes:
        scope_specs.append(("vp", "vp_", "vp_"))
    if "visual_prompt_end" in raw_scopes:
        scope_specs.append(("vpend", "vpend_", "vpend_"))
    if not scope_specs:
        raise KeyError("Missing QA DGST VV/VP/VPend support scope")

    matrices_by_scope: dict[str, dict[str, np.ndarray]] = {}
    for scope_name, raw_prefix, _ in scope_specs:
        attention_key = f"dgst_t_{raw_prefix}attention_support_per_layer"
        source_key = f"dgst_t_{raw_prefix}source_dist_per_layer"
        scope_missing = [
            key for key in (attention_key, source_key) if key not in result
        ]
        if scope_missing:
            raise KeyError(
                f"Missing QA DGST {scope_name.upper()} shared fields: "
                f"{scope_missing}"
            )
        matrices_by_scope[scope_name] = {
            "attention_support": _as_float32_array(result[attention_key]),
            "source_dist": _as_float32_array(result[source_key]),
        }

    primary_scope = next(
        scope
        for scope in ("vv", "vpend", "vp")
        if scope in matrices_by_scope
    )
    compact: dict = {
        "schema_version": "dgst-target-comparison-v3",
        "methods": list(methods),
        "support_modes": [scope[0] for scope in scope_specs],
        "metadata": {
            **{key: result[key] for key in metadata_keys},
            "dgst_t_four_gate_support_scopes": list(raw_scopes),
        },
        # Preserve the historical primary matrices field for VV consumers. In
        # VP-only mode it deliberately points at VP so spatial diagnostics can
        # still use the same path.
        "matrices": matrices_by_scope[primary_scope],
        "matrices_by_scope": matrices_by_scope,
    }
    for scope_name, _raw_prefix, _compact_prefix in scope_specs:
        for suffix in ("support_size", "support_positions"):
            key = f"dgst_t_{scope_name}_{suffix}"
            if key in result:
                compact["metadata"][key] = result[key]
    if "dgst_t_prompt_cafe" in result:
        compact["prompt_cafe"] = float(result["dgst_t_prompt_cafe"])
        compact["prompt_cafe_per_layer"] = _as_float32_array(
            result["dgst_t_prompt_cafe_per_layer"]
        )
        for key in (
            "dgst_t_prompt_cafe_layer",
            "dgst_t_prompt_cafe_requested_layer",
            "dgst_t_prompt_cafe_temperature",
            "dgst_t_prompt_cafe_prompt_size",
            "dgst_t_prompt_cafe_position_scope",
            "dgst_t_prompt_cafe_definition",
        ):
            if key in result:
                compact["metadata"][key] = result[key]
    target_region_top_k = int(result["dgst_t_target_region_top_k"])
    topk_slug = f"topk{target_region_top_k}"
    scoped_methods: list[str] = []
    for scope_name, raw_prefix, compact_prefix in scope_specs:
        for method in methods:
            compact_method = f"{compact_prefix}{method}"
            state = _target_comparison_state(method)
            keys = {
                "risk": (
                    f"dgst_t_{raw_prefix}{method}_risk_sqrt_{state}_per_layer"
                ),
                "target_cosine": (
                    f"dgst_t_{raw_prefix}{method}_target_cosine_"
                    f"{topk_slug}_{state}_per_layer"
                ),
                "ev": (
                    f"dgst_t_{raw_prefix}{method}_"
                    "ev_target_dist_mass_x_cosine_"
                    f"{topk_slug}_{state}_per_layer"
                ),
            }
            branch_missing = [
                key for key in keys.values() if key not in result
            ]
            if branch_missing:
                raise KeyError(
                    f"Missing QA DGST branch fields for {compact_method}: "
                    f"{branch_missing}"
                )
            branch = {
                "state": state,
                "support_mode": scope_name,
                **{
                    name: _as_float32_array(result[key])
                    for name, key in keys.items()
                },
            }
            risk_prefix = f"dgst_t_{raw_prefix}{method}_risk_"
            for key, value in result.items():
                if not (
                    key.startswith(risk_prefix)
                    and key.endswith("_per_layer")
                ):
                    continue
                component = key[
                    len(f"dgst_t_{raw_prefix}{method}_")
                    : -len("_per_layer")
                ]
                if component == f"risk_sqrt_{state}":
                    component = "risk_sqrt_matched_state"
                elif component == f"risk_cosine_{state}":
                    component = "risk_cosine_matched_state"
                branch[component] = _as_float32_array(value)
            if method == DIRECT_SOFTMAX_METHOD:
                for name, suffix in (
                    (
                        "target_prob_matrix",
                        "hpre_softmax_prob_direct_target_prob_matrix_per_layer",
                    ),
                    (
                        "target_dist",
                        "hpre_softmax_prob_direct_target_dist_per_layer",
                    ),
                ):
                    key = f"dgst_t_{raw_prefix}{suffix}"
                    if key not in result:
                        raise KeyError(
                            f"Missing direct QA DGST matrix: {key}"
                        )
                    branch[name] = _as_float32_array(result[key])
            else:
                gate_key = f"dgst_t_{raw_prefix}{method}_gate_per_layer"
                if gate_key not in result:
                    raise KeyError(f"Missing QA DGST gate: {gate_key}")
                branch["gate"] = _as_float32_array(result[gate_key])
            compact[compact_method] = branch
            scoped_methods.append(compact_method)
    compact["scoped_methods"] = scoped_methods
    return compact


def _build_position_record(
    *,
    model_out,
    cfg_dgst_t: dict,
    cfg_ads: dict,
    cfg_cgc: dict,
    target_metadata: dict,
    method_enabled: bool = True,
    ads_cgc_enabled: bool = True,
) -> dict:
    if not method_enabled and not ads_cgc_enabled:
        raise ValueError("Position record requires DGST or ADS+CGC")
    record = {
        "target": {
            "predicted_token_id": int(model_out.token_id),
            "predicted_token": str(model_out.token_str),
            **target_metadata,
        },
    }
    if method_enabled:
        record["dgst"] = _compact_dgst(
            _compute_dgst_t_result(model_out, cfg_dgst_t)
        )
    if not ads_cgc_enabled:
        return record
    ads_score, ads_layers = compute_ads(
        model_out.text_to_patch_attn,
        top_patch_pct=float(cfg_ads.get("top_patch_pct", 0.10)),
        connectivity=int(cfg_ads.get("connectivity", 8)),
        min_blob_area=int(cfg_ads.get("min_blob_area", 3)),
        top_k_layers=int(cfg_ads.get("top_k_layers", 10)),
        per_head_min=bool(cfg_ads.get("per_head_min", False)),
        top_k_heads=int(cfg_ads.get("top_k_heads", 0)),
        grid_shape=model_out.visual_grid,
    )
    cgc_score, cgc_layers = compute_cgc(
        model_out.token_hidden_states,
        model_out.patch_hidden_states,
        top_k_patches=int(cfg_cgc.get("top_k_patches", 5)),
        top_k_pct=float(cfg_cgc.get("top_k_pct", 0.05)),
        text_to_patch_attn=(
            model_out.text_to_patch_attn
            if bool(cfg_cgc.get("use_attn_weighting", False))
            else None
        ),
        mid_layer_pct=tuple(cfg_cgc.get("mid_layer_pct", (0.25, 0.75))),
    )
    record.update({
        "ads_score": float(ads_score),
        "ads_per_layer": _as_float32_array(ads_layers),
        "cgc_score": float(cgc_score),
        "cgc_per_layer": _as_float32_array(cgc_layers),
    })
    return record


def _prompt_alignment_metadata(model_out) -> dict:
    capture = model_out.baseline_capture
    if not isinstance(capture, dict):
        return {}
    raw = capture.get("prompt_target_alignment")
    if raw is None:
        return {}
    if hasattr(raw, "__dict__"):
        raw = vars(raw)
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, value in raw.items():
        if isinstance(value, tuple):
            value = [int(item) for item in value]
        elif isinstance(value, (np.integer, int)):
            value = int(value)
        result[str(key)] = value
    return result


def _as_float32_array(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _as_float_list(value) -> list[float]:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.astype(np.float32).reshape(-1).tolist()
    return [float(x) for x in value]


def _contains_gt_key(value) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower().startswith("gt") or _contains_gt_key(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_gt_key(child) for child in value)
    return False


def _assert_finite(value, path: str = "feature") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"Non-finite value at {path}: {value}")


def _atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
