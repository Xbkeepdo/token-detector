"""Unified Generate -> Label -> Extract implementation for yes/no QA benchmarks."""

from __future__ import annotations

import math
import os
import pickle
import tempfile
import traceback
from pathlib import Path
from typing import Iterable

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
from features.dgst_t import COST_VARIANT_RISK_KEYS
from features.extractor import _compute_dgst_t_result
from features.pope_extractor import _extract_pope_forward


HPRE_KEY = "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer"


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
    instruction = f"{question}\nAnswer only yes or no."
    if model_key.startswith("llava"):
        return f"USER: <image>\n{instruction}\nASSISTANT:"
    return instruction


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
        if key in store.rows:
            continue
        try:
            with Image.open(question["image_path"]) as raw_image:
                image = raw_image.convert("RGB")
            prompt = qa_prompt(model_key, question["question"])
            generated = model_wrapper.generate(image, prompt=prompt)
            prediction = normalize_yes_no(generated.generated_text)
            semantic_index = find_answer_semantic_token(
                generated.response_token_ids,
                model_wrapper.tokenizer,
                prediction,
            )
            row = {
                "key": key,
                "dataset": question["dataset"],
                "source_split": question["source_split"],
                "question_id": question["question_id"],
                "image_id": question["image_id"],
                "probe_split": question["probe_split"],
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
        label, error_type = label_answer(generation.get("prediction"), question["gt_answer"])
        store.add({
            "key": question["key"],
            "dataset": question["dataset"],
            "source_split": question["source_split"],
            "question_id": question["question_id"],
            "image_id": question["image_id"],
            "probe_split": question["probe_split"],
            "question_family_index": question.get("question_family_index"),
            "gt_answer": question["gt_answer"],
            "prediction": generation.get("prediction"),
            "label": label,
            "class_name": "real" if label == 1 else "hallucination",
            "error_type": error_type,
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
    include_object_cgc: bool = True,
) -> list[dict]:
    output = Path(output_dir)
    generations = {row["key"]: row for row in load_jsonl(output / "generations.jsonl")}
    labels = {row["key"]: row for row in load_jsonl(output / "labels.jsonl")}
    shards = AtomicFeatureShards(output_dir, shard_size)
    failures = JSONLCheckpointStore(str(output / "extraction_failures.jsonl"), 1)

    for question in tqdm(questions, desc="Extract QA features"):
        key = question["key"]
        if key in shards.rows:
            continue
        generation = generations.get(key)
        label_row = labels.get(key)
        if generation is None or label_row is None:
            continue
        response_ids = [int(x) for x in generation.get("response_token_ids", [])]
        answer_index = generation.get("answer_token_index")
        if answer_index is None:
            answer_index = _first_content_token(response_ids, model_wrapper.tokenizer)
        if answer_index is None or not response_ids:
            failures.add({"key": key, "error": "No response token available for extraction"})
            continue
        answer_index = int(answer_index)
        answer_token_id = int(response_ids[answer_index])
        target_ids = [answer_token_id]
        target_names = ["answer"]
        object_token_id = None
        if question["dataset"] == "pope":
            object_ids = model_wrapper.tokenizer.encode(
                question["object_word"], add_special_tokens=False
            )
            if object_ids:
                object_token_id = int(object_ids[0])
                target_ids.append(object_token_id)
                target_names.append("object")
        prompt = qa_prompt(model_key, question["question"])
        try:
            with Image.open(question["image_path"]) as raw_image:
                image = raw_image.convert("RGB")
            outputs = model_wrapper.extract_token_features_batch(
                image=image,
                response_token_ids=response_ids,
                response_token_indices=[answer_index] * len(target_ids),
                target_token_ids=target_ids,
                cfg_dgst_t=cfg_dgst_t,
                prompt=prompt,
            )
            if len(outputs) != len(target_ids):
                raise RuntimeError(f"Expected {len(target_ids)} target outputs, got {len(outputs)}")
            baseline = _baseline_features(outputs[0], answer_token_id, cfg_ads, cfg_cgc)
            targets = {
                name: _compact_dgst(_compute_dgst_t_result(model_out, cfg_dgst_t))
                for name, model_out in zip(target_names, outputs)
            }
            object_cgc = None
            object_cgc_per_layer = None
            if question["dataset"] == "pope" and include_object_cgc:
                try:
                    legacy = _extract_pope_forward(
                        model_wrapper, image, prompt, question["object_word"]
                    )
                    if legacy is not None:
                        object_cgc, object_layers = compute_cgc(
                            legacy["object_hidden_states"],
                            legacy["patch_hidden_states"],
                            top_k_patches=cfg_cgc.get("top_k_patches", 5),
                            top_k_pct=cfg_cgc.get("top_k_pct", 0.0),
                            text_to_patch_attn=(
                                legacy["text_to_patch_attn"]
                                if cfg_cgc.get("use_attn_weighting", False) else None
                            ),
                            mid_layer_pct=tuple(cfg_cgc.get("mid_layer_pct", [0.25, 0.75])),
                        )
                        object_cgc_per_layer = object_layers.tolist()
                except Exception as object_exc:
                    failures.add({
                        "key": key,
                        "component": "object_position_cgc",
                        "error": repr(object_exc),
                        "traceback": traceback.format_exc(),
                    })

            feature = {
                "key": key,
                "dataset": question["dataset"],
                "source_split": question["source_split"],
                "question_id": question["question_id"],
                "image_id": question["image_id"],
                "probe_split": question["probe_split"],
                "question_family_index": question.get("question_family_index"),
                "label": int(label_row["label"]),
                "class_name": label_row["class_name"],
                "error_type": label_row["error_type"],
                "prediction": label_row.get("prediction"),
                "generated_text": generation.get("generated_text"),
                "answer_token_index": answer_index,
                "answer_token_id": answer_token_id,
                "predicted_token_id": int(outputs[0].token_id),
                "predicted_token": outputs[0].token_str,
                "object_token_id": object_token_id,
                "ot_solver": os.environ.get(
                    "DGST_OT_SOLVER_OVERRIDE",
                    str(cfg_dgst_t.get("ot_solver", "emd")),
                ),
                "sinkhorn_reg": (
                    float(os.environ.get("DGST_SINKHORN_REG", "0.05"))
                    if os.environ.get("DGST_OT_SOLVER_OVERRIDE", "").lower() == "sinkhorn"
                    else None
                ),
                "targets": targets,
                **baseline,
                "object_cgc_score": None if object_cgc is None else float(object_cgc),
                "object_cgc_per_layer": object_cgc_per_layer,
            }
            if "gt_answer" in feature or _contains_gt_key(feature):
                raise AssertionError("GT leakage: feature record contains a GT field")
            _assert_finite(feature)
            shards.add(feature)
        except Exception as exc:
            failures.add({"key": key, "error": repr(exc), "traceback": traceback.format_exc()})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    shards.consolidate()
    failures.flush()
    return list(shards.rows.values())


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
    return _first_content_token(response_ids, tokenizer)


def _compact_dgst(result: dict) -> dict:
    compact = {}
    for risk_name in COST_VARIANT_RISK_KEYS:
        key = f"dgst_t_{risk_name}_per_layer"
        if key not in result:
            raise KeyError(f"Missing required QA DGST feature: {key}")
        compact[risk_name] = _as_float_list(result[key])
    if HPRE_KEY not in result:
        raise KeyError(f"Missing required QA DGST feature: {HPRE_KEY}")
    compact["hprecosine"] = _as_float_list(result[HPRE_KEY])
    return compact


def _baseline_features(model_out, target_token_id: int, cfg_ads: dict, cfg_cgc: dict) -> dict:
    ads_score, ads_layers = compute_ads(
        model_out.text_to_patch_attn,
        top_patch_pct=cfg_ads.get("top_patch_pct", 0.10),
        connectivity=cfg_ads.get("connectivity", 8),
        min_blob_area=cfg_ads.get("min_blob_area", 3),
        top_k_layers=cfg_ads.get("top_k_layers", 10),
        per_head_min=cfg_ads.get("per_head_min", False),
        top_k_heads=cfg_ads.get("top_k_heads", 0),
    )
    cgc_score, cgc_layers = compute_cgc(
        model_out.token_hidden_states,
        model_out.patch_hidden_states,
        top_k_patches=cfg_cgc.get("top_k_patches", 5),
        top_k_pct=cfg_cgc.get("top_k_pct", 0.0),
        text_to_patch_attn=(
            model_out.text_to_patch_attn if cfg_cgc.get("use_attn_weighting", False) else None
        ),
        mid_layer_pct=tuple(cfg_cgc.get("mid_layer_pct", [0.25, 0.75])),
    )
    if model_out.token_logits is None:
        raise RuntimeError("Wrapper did not return token logits")
    logits = model_out.token_logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    token_log_prob = float(log_probs[target_token_id].item())
    entropy = float((-(probs * log_probs).sum() / math.log(probs.numel())).item())
    attn = model_out.text_to_patch_attn.float()
    n_layers = int(attn.shape[0])
    visual_mass = attn.sum(dim=-1).mean(dim=-1)
    start = max(0, int(n_layers * 0.15))
    end = min(n_layers, int(n_layers * 0.55))
    return {
        "ads_score": float(ads_score),
        "ads_per_layer": _as_float_list(ads_layers),
        "answer_cgc_score": float(cgc_score),
        "answer_cgc_per_layer": _as_float_list(cgc_layers),
        "token_log_probability": token_log_prob,
        "token_entropy": entropy,
        "token_nll": -token_log_prob,
        "svar_score": float(visual_mass[start:end].sum().item()),
        "attention_per_head_mid": _as_float_list(attn[n_layers // 2].mean(dim=-1)),
    }


def _as_float_list(value) -> list[float]:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.astype(np.float32).reshape(-1).tolist()
    return [float(x) for x in value]


def _first_content_token(response_ids: list[int], tokenizer) -> int | None:
    for index, token_id in enumerate(response_ids):
        if tokenizer.decode([token_id], skip_special_tokens=True).strip():
            return index
    return 0 if response_ids else None


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
