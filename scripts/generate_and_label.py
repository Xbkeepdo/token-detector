#!/usr/bin/env python3
"""Generate image descriptions with an LVLM and label hallucinated tokens via GPT-4o."""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import get_context
from tqdm import tqdm

from models import build_model
from data.coco_loader import load_coco_samples, train_val_split
from utils.io_utils import save_json, load_json
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   help="Model key, e.g. llava_1_5_7b")
    p.add_argument("--config",      default="configs/model_configs.yaml")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--openai-key",  default=None,
                   help="OpenAI API key, or GitHub Models PAT when using GitHub Models")
    p.add_argument("--openai-base-url", default=None,
                   help="OpenAI-compatible API base URL, e.g. https://models.github.ai/inference")
    p.add_argument("--openai-model", default=None,
                   help="Chat model name, e.g. gpt-4o or openai/gpt-4o")
    p.add_argument("--openai-proxy", default=None,
                   help="HTTP(S) proxy for OpenAI API calls, e.g. http://127.0.0.1:12596")
    p.add_argument("--num-images",  type=int, default=None,
                   help="Override num_images from config")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--generation-devices", nargs="+", default=None,
                   help="Run generation in parallel, one worker per device, e.g. cuda:0 cuda:1")
    p.add_argument("--resume",      action="store_true",
                   help="Skip already-processed images")
    return p.parse_args()


def main():
    args = parse_args()
    from utils.config_utils import load_config, get_model_cfg, get_dataset_cfg

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)

    num_images = args.num_images or dataset_cfg["num_images"]

    images_dir = os.path.join(dataset_cfg["coco_root"], "val2014")
    samples = load_coco_samples(
        images_dir=images_dir,
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=num_images,
        seed=dataset_cfg["seed"],
    )

    splits_path = os.path.join(args.output_dir, "image_splits.json")
    if os.path.exists(splits_path):
        splits = load_json(splits_path)
        train_samples = [s for s in samples if s["image_id"] in set(splits["train"])]
        val_samples   = [s for s in samples if s["image_id"] in set(splits["val"])]
        print(f"[Generate] Loaded existing split: "
              f"{len(splits['train'])} train, {len(splits['val'])} val.")
    else:
        train_samples, val_samples = train_val_split(
            samples,
            train_ratio=dataset_cfg["train_ratio"],
            seed=dataset_cfg["seed"],
        )
        splits = {
            "train": [s["image_id"] for s in train_samples],
            "val":   [s["image_id"] for s in val_samples],
            "test":  [s["image_id"] for s in val_samples],
        }
        save_json(splits, splits_path)
        print(f"[Generate] Split saved: "
              f"{len(train_samples)} train, {len(val_samples)} val/test.")

    gen_path = os.path.join(args.output_dir, "generations.json")
    generation_results = {}
    generation_token_ids = {}

    if args.resume and os.path.exists(gen_path):
        raw = load_json(gen_path)
        generation_results   = {int(k): v["generated_text"]    for k, v in raw.items()}
        generation_token_ids = {int(k): v["response_token_ids"] for k, v in raw.items()}
        print(f"[Generate] Loaded {len(generation_results)} existing generations.")

    pending = [s for s in samples if s["image_id"] not in generation_results]

    devices = args.generation_devices or [args.device]
    if pending and len(devices) > 1:
        print(
            f"[Generate] Running parallel generation on "
            f"{', '.join(devices)} for {len(pending)} images."
        )
        worker_results = _parallel_generate(
            model_key=args.model,
            model_cfg=model_cfg,
            pending_samples=pending,
            devices=devices,
        )
        for image_id, generated_text, response_token_ids in worker_results:
            generation_results[int(image_id)] = generated_text
            generation_token_ids[int(image_id)] = [int(token_id) for token_id in response_token_ids]
        _save_generations(generation_results, generation_token_ids, gen_path)
        print(f"[Generate] Generations done. Saved to {gen_path}")
        print(f"[Generate] Loading tokenizer for {model_cfg['hf_name']} …")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])
    elif pending:
        print(f"[Generate] Loading model '{args.model}' for generation on {devices[0]} …")
        wrapper = build_model(args.model, model_cfg, device=devices[0])
        tokenizer = wrapper.tokenizer

        for sample in tqdm(pending, desc="Generating"):
            image_id = sample["image_id"]
            image = Image.open(sample["image_path"]).convert("RGB")
            gen_out = wrapper.generate(image)

            generation_results[image_id] = gen_out.generated_text
            generation_token_ids[image_id] = gen_out.response_token_ids
            _save_generations(generation_results, generation_token_ids, gen_path)

        print(f"[Generate] Generations done. Saved to {gen_path}")
    else:
        print("[Generate] All generations already complete.")
        print(f"[Generate] Loading tokenizer for {model_cfg['hf_name']} …")
        tokenizer = _load_tokenizer(model_cfg["hf_name"])

    label_path = os.path.join(args.output_dir, "labeling.json")
    print("[Generate] Running GPT-4o labeling …")
    from labeling.gpt4_labeler import label_dataset

    label_dataset(
        samples=samples,
        generation_results=generation_results,
        generation_token_ids=generation_token_ids,
        tokenizer=tokenizer,
        output_path=label_path,
        openai_api_key=args.openai_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        openai_proxy=args.openai_proxy,
        resume=args.resume,
    )
    print(f"[Generate] Labeling saved to {label_path}")


def _save_generations(
    generation_results: dict[int, str],
    generation_token_ids: dict[int, list[int]],
    path: str,
) -> None:
    save_json(
        {
            str(k): {
                "generated_text": generation_results[k],
                "response_token_ids": generation_token_ids[k],
            }
            for k in sorted(generation_results)
        },
        path,
    )


def _parallel_generate(
    *,
    model_key: str,
    model_cfg: dict,
    pending_samples: list[dict],
    devices: list[str],
) -> list[tuple[int, str, list[int]]]:
    chunks = [[] for _ in devices]
    for index, sample in enumerate(pending_samples):
        chunks[index % len(devices)].append(sample)

    ctx = get_context("spawn")
    jobs = []
    with ctx.Pool(processes=len(devices)) as pool:
        for worker_id, (device, chunk) in enumerate(zip(devices, chunks)):
            if not chunk:
                continue
            jobs.append(
                pool.apply_async(
                    _generate_worker,
                    (worker_id, model_key, model_cfg, device, chunk),
                )
            )
        results = []
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
    print(
        f"[Generate worker {worker_id}] Loading model '{model_key}' on {device} "
        f"for {len(samples)} images."
    )
    wrapper = build_model(model_key, model_cfg, device=device)
    results = []
    for sample in tqdm(samples, desc=f"Generating worker {worker_id}"):
        image_id = int(sample["image_id"])
        image = Image.open(sample["image_path"]).convert("RGB")
        gen_out = wrapper.generate(image)
        results.append((image_id, gen_out.generated_text, [int(t) for t in gen_out.response_token_ids]))
    return results


def _load_tokenizer(hf_name: str):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    except Exception:
        return AutoProcessor.from_pretrained(hf_name, trust_remote_code=True).tokenizer


if __name__ == "__main__":
    main()
