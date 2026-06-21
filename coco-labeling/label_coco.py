#!/usr/bin/env python3
"""Generate/reuse COCO captions and build CHAIR-style token labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from tqdm import tqdm

from data.coco_loader import load_coco_samples, train_val_split
from utils.io_utils import load_json, save_json

_CHAIR_PATH = Path(__file__).with_name("coco_chair.py")
_SPEC = importlib.util.spec_from_file_location("coco_chair", _CHAIR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot import {_CHAIR_PATH}")
_CHAIR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHAIR)
CocoChairEvaluator = _CHAIR.CocoChairEvaluator
chair_summary = _CHAIR.chair_summary
dedupe_mentions_by_object = _CHAIR.dedupe_mentions_by_object
iter_ground_truth_entries = _CHAIR.iter_ground_truth_entries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caption-file", default=None, help="Optional JSONL rows with image_id and caption.")
    parser.add_argument("--chair-cache", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-devices", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from utils.config_utils import get_dataset_cfg, get_model_cfg, load_config

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    seed = args.seed if args.seed is not None else int(dataset_cfg["seed"])
    num_images = args.num_images or int(dataset_cfg["num_images"])

    samples = load_coco_samples(
        images_dir=os.path.join(dataset_cfg["coco_root"], "val2014"),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=num_images if args.caption_file is None else None,
        seed=seed,
    )
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}

    if args.caption_file:
        caption_rows = _load_caption_rows(Path(args.caption_file), sample_by_id)
        if num_images is not None and num_images < len(caption_rows):
            random.seed(seed)
            caption_rows = random.sample(caption_rows, num_images)
        selected_ids = {int(row["image_id"]) for row in caption_rows}
        selected_samples = [sample for sample in samples if int(sample["image_id"]) in selected_ids]
    else:
        caption_rows = []
        selected_samples = samples

    splits = _load_or_create_splits(
        output_dir=args.output_dir,
        samples=selected_samples,
        train_ratio=float(dataset_cfg["train_ratio"]),
        seed=seed,
        resume=args.resume,
    )

    evaluator = CocoChairEvaluator.from_cache(
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg.get("captions_file"),
        cache_path=args.chair_cache or os.path.join(args.output_dir, "chair.pkl"),
    )

    generations_path = os.path.join(args.output_dir, "generations.json")
    labeling_path = os.path.join(args.output_dir, "labeling.json")
    generations = load_json(generations_path) if args.resume and os.path.exists(generations_path) else {}

    if args.caption_file:
        print(f"[COCO-CHAIR] Loading tokenizer for {model_cfg['hf_name']}")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
    else:
        caption_rows, tokenizer = _caption_rows_from_generation(
            args=args,
            model_cfg=model_cfg,
            samples=selected_samples,
            generations=generations,
            generations_path=generations_path,
        )

    labeling: dict[str, dict] = {}
    for row in tqdm(caption_rows, desc="COCO-CHAIR labeling"):
        image_id = int(row["image_id"])
        caption = str(row["caption"])
        token_ids = _generation_token_ids(generations, image_id, caption, tokenizer)
        spans = _chair_token_spans(
            evaluator=evaluator,
            tokenizer=tokenizer,
            image_id=image_id,
            caption=caption,
            token_ids=token_ids,
        )
        eval_info = evaluator.evaluate_caption(image_id, caption)
        generations[str(image_id)] = {
            "generated_text": caption,
            "response_token_ids": [int(token_id) for token_id in token_ids],
        }
        labeling[str(image_id)] = {
            "image_id": image_id,
            "generated_text": caption,
            "hallucinated_words": [
                span["word"] for span in spans if int(span.get("label", 0)) == 1
            ],
            "object_token_spans": spans,
            "chair_s": eval_info["chair_s"],
            "chair_i": eval_info["chair_i"],
            "ground_truth_objects": eval_info["ground_truth_objects"],
        }

    save_json(generations, generations_path)
    save_json(labeling, labeling_path)
    save_json(chair_summary(labeling), os.path.join(args.output_dir, "chair_summary.json"))
    _save_ground_truth(evaluator, splits, os.path.join(args.output_dir, "coco_ground_truth.jsonl"))
    print(f"[COCO-CHAIR] Saved {len(labeling)} labels to {labeling_path}")


def _load_or_create_splits(
    *,
    output_dir: str,
    samples: list[dict],
    train_ratio: float,
    seed: int,
    resume: bool,
) -> dict:
    splits_path = os.path.join(output_dir, "image_splits.json")
    if resume and os.path.exists(splits_path):
        splits = load_json(splits_path)
        print(
            f"[COCO-CHAIR] Loaded existing split: "
            f"{len(splits['train'])} train, {len(splits['val'])} val."
        )
        return splits
    train_samples, val_samples = train_val_split(samples, train_ratio=train_ratio, seed=seed)
    splits = {
        "train": [int(sample["image_id"]) for sample in train_samples],
        "val": [int(sample["image_id"]) for sample in val_samples],
        "test": [int(sample["image_id"]) for sample in val_samples],
    }
    save_json(splits, splits_path)
    return splits


def _caption_rows_from_generation(
    *,
    args,
    model_cfg: dict,
    samples: list[dict],
    generations: dict,
    generations_path: str,
):
    from models import build_model

    if _all_generations_available(samples, generations):
        print("[COCO-CHAIR] Reusing existing generations and loading tokenizer only.")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
        return _rows_from_generations(samples, generations), tokenizer

    pending = [
        sample for sample in samples
        if not generations.get(str(int(sample["image_id"])), {}).get("generated_text")
    ]
    devices = args.generation_devices or [args.device]
    if len(devices) > 1 and pending:
        print(
            f"[COCO-CHAIR] Parallel generation on {', '.join(devices)} "
            f"for {len(pending)} images."
        )
        for image_id, caption, token_ids in _parallel_generate(
            model_key=args.model,
            model_cfg=model_cfg,
            samples=pending,
            devices=devices,
        ):
            generations[str(image_id)] = {
                "generated_text": caption,
                "response_token_ids": [int(token_id) for token_id in token_ids],
            }
        save_json(generations, generations_path)
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
    else:
        print(f"[COCO-CHAIR] Loading model '{args.model}' on {devices[0]}.")
        wrapper = build_model(args.model, model_cfg, device=devices[0])
        tokenizer = wrapper.tokenizer
        for sample in tqdm(pending, desc="Generating"):
            image_id = int(sample["image_id"])
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image)
            generations[str(image_id)] = {
                "generated_text": gen_out.generated_text,
                "response_token_ids": [int(token_id) for token_id in gen_out.response_token_ids],
            }
            save_json(generations, generations_path)

    return _rows_from_generations(samples, generations), tokenizer


def _parallel_generate(
    *,
    model_key: str,
    model_cfg: dict,
    samples: list[dict],
    devices: list[str],
) -> list[tuple[int, str, list[int]]]:
    chunks = [[] for _ in devices]
    for index, sample in enumerate(samples):
        chunks[index % len(devices)].append(sample)
    ctx = get_context("spawn")
    results = []
    with ctx.Pool(processes=len(devices)) as pool:
        jobs = [
            pool.apply_async(_generate_worker, (worker_id, model_key, model_cfg, device, chunk))
            for worker_id, (device, chunk) in enumerate(zip(devices, chunks))
            if chunk
        ]
        for job in jobs:
            results.extend(job.get())
    return results


def _generate_worker(
    worker_id: int,
    model_key: str,
    model_cfg: dict,
    device: str,
    samples: list[dict],
) -> list[tuple[int, str, list[int]]]:
    from models import build_model

    print(f"[COCO-CHAIR worker {worker_id}] Loading model '{model_key}' on {device}.")
    wrapper = build_model(model_key, model_cfg, device=device)
    rows = []
    for sample in tqdm(samples, desc=f"Generating worker {worker_id}"):
        image_id = int(sample["image_id"])
        image = Image.open(sample["image_path"]).convert("RGB")
        gen_out = wrapper.generate(image)
        rows.append((image_id, gen_out.generated_text, [int(token_id) for token_id in gen_out.response_token_ids]))
    return rows


def _all_generations_available(samples: list[dict], generations: dict) -> bool:
    return bool(generations) and all(
        generations.get(str(int(sample["image_id"])), {}).get("generated_text")
        for sample in samples
    )


def _rows_from_generations(samples: list[dict], generations: dict) -> list[dict]:
    rows = []
    for sample in samples:
        image_id = int(sample["image_id"])
        entry = generations.get(str(image_id), {})
        if not entry.get("generated_text"):
            continue
        rows.append(
            {
                "image_id": image_id,
                "image_path": sample["image_path"],
                "caption": entry["generated_text"],
            }
        )
    return rows


def _load_caption_rows(path: Path, sample_by_id: dict[int, dict]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = int(row["image_id"])
            caption = str(row.get("caption") or row.get("generated_text") or "")
            if image_id not in sample_by_id or not caption:
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": sample_by_id[image_id]["image_path"],
                    "caption": caption,
                }
            )
    if not rows:
        raise ValueError(f"No usable captions found in {path}")
    return rows


def _load_tokenizer(hf_name: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoProcessor.from_pretrained(hf_name).tokenizer
    except Exception:
        return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)


def _generation_token_ids(generations: dict, image_id: int, caption: str, tokenizer) -> list[int]:
    entry = generations.get(str(int(image_id)), {})
    token_ids = entry.get("response_token_ids")
    if token_ids:
        return [int(token_id) for token_id in token_ids]
    return [int(token_id) for token_id in tokenizer.encode(caption, add_special_tokens=False)]


def _chair_token_spans(
    *,
    evaluator,
    tokenizer,
    image_id: int,
    caption: str,
    token_ids: list[int],
) -> list[dict]:
    eval_info = evaluator.evaluate_caption(image_id, caption)
    mentions = dedupe_mentions_by_object(eval_info["object_mentions"])
    spans = []
    for mention in mentions:
        char_start = int(mention["char_start"])
        char_end = int(mention["char_end"])
        prefix_ids = tokenizer.encode(caption[:char_start], add_special_tokens=False)
        mention_ids = tokenizer.encode(caption[char_start:char_end], add_special_tokens=False)
        if not mention_ids:
            continue
        first_idx = len(prefix_ids)
        last_idx = first_idx + len(mention_ids) - 1
        if first_idx < 0 or last_idx >= len(token_ids):
            continue
        spans.append(
            {
                "word": mention["canonical_name"],
                "surface": caption[char_start:char_end],
                "token_indices": list(range(first_idx, first_idx + len(mention_ids))),
                "label": int(mention.get("hallucinated", 0)),
            }
        )
    spans.sort(key=lambda item: item["token_indices"][0])
    return spans


def _save_ground_truth(evaluator, splits: dict, path: str) -> None:
    image_ids = sorted({int(image_id) for values in splits.values() for image_id in values})
    with open(path, "w", encoding="utf-8") as handle:
        for entry in iter_ground_truth_entries(evaluator, image_ids):
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
