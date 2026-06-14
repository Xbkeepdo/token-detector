#!/usr/bin/env python3
"""Create TGD labels for COCO captions with DGST's CHAIR evaluator."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/apulis-dev/code")

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from data.coco_loader import load_coco_samples, train_val_split
from dgst.data.chair import CocoChairEvaluator
from models import build_model
from utils.config_utils import load_config, get_dataset_cfg, get_model_cfg
from utils.io_utils import save_json, load_json


WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="configs/model_configs.yaml")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--caption-file", default=None)
    p.add_argument("--chair-cache", default=None)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    seed = args.seed if args.seed is not None else dataset_cfg["seed"]
    num_images = args.num_images or dataset_cfg["num_images"]

    samples = load_coco_samples(
        images_dir=os.path.join(dataset_cfg["coco_root"], "val2014"),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=num_images if args.caption_file is None else None,
        seed=seed,
    )
    sample_by_id = {int(s["image_id"]): s for s in samples}

    if args.caption_file:
        caption_rows = _load_caption_rows(Path(args.caption_file), sample_by_id)
        if num_images is not None and num_images < len(caption_rows):
            random.seed(seed)
            caption_rows = random.sample(caption_rows, num_images)
        caption_ids = {int(row["image_id"]) for row in caption_rows}
        selected_samples = [s for s in samples if int(s["image_id"]) in caption_ids]
    else:
        caption_rows = []
        selected_samples = samples

    splits_path = os.path.join(args.output_dir, "image_splits.json")
    if args.resume and os.path.exists(splits_path):
        splits = load_json(splits_path)
        print(
            f"[CHAIR] Loaded existing split: "
            f"{len(splits['train'])} train, {len(splits['val'])} val."
        )
    else:
        train_samples, val_samples = train_val_split(
            selected_samples,
            train_ratio=dataset_cfg["train_ratio"],
            seed=seed,
        )
        splits = {
            "train": [s["image_id"] for s in train_samples],
            "val": [s["image_id"] for s in val_samples],
            "test": [s["image_id"] for s in val_samples],
        }
        save_json(splits, splits_path)

    chair_cache = args.chair_cache or os.path.join(args.output_dir, "chair.pkl")

    print("[CHAIR] Loading DGST CHAIR evaluator")
    evaluator = CocoChairEvaluator.from_cache(
        Path(dataset_cfg["annotation_file"]).parent,
        chair_cache,
    )

    generations_path = os.path.join(args.output_dir, "generations.json")
    labeling_path = os.path.join(args.output_dir, "labeling.json")
    generations = {}
    labeling = {}
    if args.resume and os.path.exists(generations_path):
        generations = load_json(generations_path)
    if args.resume and os.path.exists(labeling_path):
        labeling = load_json(labeling_path)

    if args.caption_file:
        print(f"[CHAIR] Loading tokenizer for {model_cfg['hf_name']}")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
    elif _all_generations_available(selected_samples, generations):
        print("[CHAIR] Reusing existing generations and loading tokenizer only")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
        caption_rows = _caption_rows_from_generations(selected_samples, generations)
    else:
        print(f"[CHAIR] Loading model '{args.model}' for fresh caption generation")
        wrapper = build_model(args.model, model_cfg, device=args.device)
        tokenizer = wrapper.tokenizer
        caption_rows = _generate_caption_rows(
            wrapper=wrapper,
            samples=selected_samples,
            generations=generations,
            generations_path=generations_path,
        )

    for row in caption_rows:
        image_id = int(row["image_id"])
        if args.resume and str(image_id) in labeling:
            continue

        caption = str(row["caption"])
        token_ids = tokenizer.encode(caption, add_special_tokens=False)
        spans = _chair_token_spans(evaluator, tokenizer, image_id, caption, token_ids)
        eval_info = evaluator.evaluate_caption(image_id, caption)

        generations[str(image_id)] = {
            "generated_text": caption,
            "response_token_ids": token_ids,
        }
        labeling[str(image_id)] = {
            "image_id": image_id,
            "generated_text": caption,
            "hallucinated_words": [
                m["canonical_name"] for m in eval_info["hallucinated_mentions"]
            ],
            "object_token_spans": spans,
            "chair_s": eval_info["chair_s"],
            "chair_i": eval_info["chair_i"],
            "ground_truth_objects": eval_info["ground_truth_objects"],
        }

    save_json(generations, generations_path)
    save_json(labeling, labeling_path)
    _save_summary(labeling, os.path.join(args.output_dir, "chair_summary.json"))
    print(f"[CHAIR] Saved {len(labeling)} labels to {labeling_path}")


def _load_tokenizer(hf_name: str):
    try:
        return AutoProcessor.from_pretrained(hf_name).tokenizer
    except Exception:
        return AutoTokenizer.from_pretrained(hf_name)


def _load_caption_rows(path: Path, sample_by_id: dict[int, dict]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id not in sample_by_id:
                continue
            if not row.get("caption"):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable captions found in {path}")
    return rows


def _generate_caption_rows(wrapper, samples: list[dict], generations: dict, generations_path: str) -> list[dict]:
    rows = []
    for sample in samples:
        image_id = int(sample["image_id"])
        key = str(image_id)
        if key in generations and generations[key].get("generated_text"):
            caption = generations[key]["generated_text"]
        else:
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image)
            caption = gen_out.generated_text
            generations[key] = {
                "generated_text": caption,
                "response_token_ids": gen_out.response_token_ids,
            }
            save_json(generations, generations_path)
        rows.append({
            "image_id": image_id,
            "image_path": sample["image_path"],
            "caption": caption,
        })
    return rows


def _all_generations_available(samples: list[dict], generations: dict) -> bool:
    if not generations:
        return False
    return all(
        str(int(sample["image_id"])) in generations
        and generations[str(int(sample["image_id"]))].get("generated_text")
        for sample in samples
    )


def _caption_rows_from_generations(samples: list[dict], generations: dict) -> list[dict]:
    rows = []
    for sample in samples:
        image_id = int(sample["image_id"])
        rows.append({
            "image_id": image_id,
            "image_path": sample["image_path"],
            "caption": generations[str(image_id)]["generated_text"],
        })
    return rows


def _chair_token_spans(evaluator, tokenizer, image_id: int, caption: str, token_ids: list[int]) -> list[dict]:
    mentions = _caption_mentions_with_chars(evaluator, image_id, caption)
    spans = []
    for mention in mentions:
        char_start = mention["char_start"]
        char_end = mention["char_end"]
        prefix_ids = tokenizer.encode(caption[:char_start], add_special_tokens=False)
        mention_ids = tokenizer.encode(caption[char_start:char_end], add_special_tokens=False)
        if not mention_ids:
            continue
        first_idx = len(prefix_ids)
        last_idx = first_idx + len(mention_ids) - 1
        if last_idx >= len(token_ids):
            continue
        spans.append({
            "word": mention["canonical_name"],
            "surface": caption[char_start:char_end],
            "token_indices": list(range(first_idx, first_idx + len(mention_ids))),
            "label": int(mention["hallucinated"]),
        })
    spans.sort(key=lambda item: item["token_indices"][0])
    return spans


def _caption_mentions_with_chars(evaluator, image_id: int, caption: str) -> list[dict]:
    if hasattr(evaluator, "caption_to_mentions") and not hasattr(evaluator, "_normalize_token"):
        return _adapter_mentions_with_chars(evaluator, image_id, caption)

    raw_tokens = [
        {
            "text": m.group(0),
            "norm": evaluator._normalize_token(m.group(0).lower()),
            "char_start": m.start(),
            "char_end": m.end(),
        }
        for m in WORD_RE.finditer(caption)
    ]
    gt_words = evaluator.get_ground_truth_objects(image_id)
    mentions = []
    i = 0
    mention_index = 0
    while i < len(raw_tokens):
        phrase = None
        token_span = 1
        if i + 1 < len(raw_tokens):
            double_word = raw_tokens[i]["norm"] + " " + raw_tokens[i + 1]["norm"]
            if double_word in evaluator.double_word_dict:
                phrase = evaluator.double_word_dict[double_word]
                token_span = 2
        if phrase is None:
            phrase = raw_tokens[i]["norm"]
        if phrase in evaluator.mscoco_objects:
            canonical = evaluator.inverse_synonym_dict[phrase]
            char_start = raw_tokens[i]["char_start"]
            char_end = raw_tokens[i + token_span - 1]["char_end"]
            mentions.append({
                "surface": caption[char_start:char_end],
                "canonical_name": canonical,
                "char_start": char_start,
                "char_end": char_end,
                "mention_index": mention_index,
                "hallucinated": int(canonical not in gt_words),
            })
            mention_index += 1
        i += token_span

    return mentions


def _adapter_mentions_with_chars(evaluator, image_id: int, caption: str) -> list[dict]:
    raw_tokens = [
        {
            "text": m.group(0),
            "char_start": m.start(),
            "char_end": m.end(),
        }
        for m in WORD_RE.finditer(caption)
    ]
    eval_info = evaluator.evaluate_caption(image_id, caption)
    mentions = []
    for mention in eval_info["object_mentions"]:
        token_start = mention.get("token_start", mention.get("word_index"))
        token_end = mention.get("token_end")
        if token_start is None:
            continue
        token_start = int(token_start)
        token_end = int(token_end) if token_end is not None else token_start + 1
        if token_start < 0 or token_end <= token_start or token_end > len(raw_tokens):
            continue
        char_start = raw_tokens[token_start]["char_start"]
        char_end = raw_tokens[token_end - 1]["char_end"]
        mentions.append({
            "surface": caption[char_start:char_end],
            "canonical_name": mention["canonical_name"],
            "char_start": char_start,
            "char_end": char_end,
            "mention_index": mention.get("mention_index", len(mentions)),
            "hallucinated": int(mention.get("hallucinated", 0)),
        })
    return mentions


def _save_summary(labeling: dict, path: str) -> None:
    total_images = len(labeling)
    total_mentions = 0
    hall_mentions = 0
    hall_images = 0
    for info in labeling.values():
        spans = info.get("object_token_spans", [])
        n_hall = sum(1 for span in spans if span.get("label") == 1)
        total_mentions += len(spans)
        hall_mentions += n_hall
        hall_images += int(n_hall > 0)
    save_json({
        "images": total_images,
        "object_mentions": total_mentions,
        "hallucinated_mentions": hall_mentions,
        "images_with_hallucination": hall_images,
        "chair_s": hall_images / total_images if total_images else 0.0,
        "chair_i": hall_mentions / total_mentions if total_mentions else 0.0,
    }, path)


if __name__ == "__main__":
    main()
