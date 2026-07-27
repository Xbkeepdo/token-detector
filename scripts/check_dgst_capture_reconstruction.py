#!/usr/bin/env python3
"""Compare DGST hook reconstruction against model hidden_states."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.dgst_capture import (
    hidden_states_from_captures,
    pre_token_prediction_positions,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
)
from utils.config_utils import get_dataset_cfg, get_model_cfg, load_config
from utils.io_utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/model_configs.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-response-index", type=int, default=80)
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[Check] Loading config and sample metadata", flush=True)
    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    output_dir = Path(args.output_dir)

    label_path = output_dir / "labeling.json"
    generations_path = output_dir / "generations.json"
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)

    labels = {int(k): v for k, v in load_json(label_path).items()}
    generations = {int(k): v for k, v in load_json(generations_path).items()}
    images_dir = Path(dataset_cfg["coco_root"]) / "val2014"
    selected = _select_cases(labels, generations, images_dir, args.num_samples, args.max_response_index)
    if not selected:
        raise RuntimeError("No usable labeled object-token cases found.")

    print(f"[Check] Selected {len(selected)} cases", flush=True)
    for case in selected:
        print(
            f"  image={case['image_id']} word={case['word']!r} "
            f"idx={case['response_index']} label={case['label']}",
            flush=True,
        )
    print(f"[Check] Loading {args.model} on {args.device}", flush=True)
    wrapper = build_model(args.model, model_cfg, device=args.device)
    print("[Check] Model loaded", flush=True)

    rows = []
    for case in selected:
        print(f"[Check] Running case image={case['image_id']}", flush=True)
        row = _check_case(wrapper, args.model, case)
        rows.append(row)
        _print_row(row)

    summary = _summary(rows)
    print("\n[Check] Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
        print(f"[Check] Saved {path}")

    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _select_cases(labels: dict, generations: dict, images_dir: Path, num_samples: int, max_index: int):
    cases = []
    for image_id in sorted(labels):
        image_path = images_dir / f"COCO_val2014_{int(image_id):012d}.jpg"
        if image_id not in generations or not image_path.exists():
            continue
        response_ids = generations[image_id].get("response_token_ids") or []
        spans = labels[image_id].get("object_token_spans") or []
        usable = []
        for span in spans:
            token_indices = span.get("token_indices") or []
            if not token_indices:
                continue
            first_idx = int(token_indices[0])
            if first_idx >= len(response_ids) or first_idx > int(max_index):
                continue
            usable.append((first_idx, span))
        if not usable:
            continue
        first_idx, span = sorted(usable, key=lambda item: item[0])[0]
        cases.append(
            {
                "image_id": int(image_id),
                "image_path": str(image_path),
                "response_ids": [int(token_id) for token_id in response_ids],
                "response_index": int(first_idx),
                "target_token_id": int(response_ids[first_idx]),
                "word": str(span.get("word", "")),
                "label": int(span.get("label", -1)),
            }
        )
        if len(cases) >= int(num_samples):
            break
    return cases


def _check_case(wrapper, model_key: str, case: dict) -> dict:
    image = Image.open(case["image_path"]).convert("RGB")
    response_ids = case["response_ids"]
    response_index = int(case["response_index"])
    target_token_id = int(case["target_token_id"])

    if model_key == "llava_1_5_7b":
        forward = _forward_llava
    elif model_key == "qwen2_5_vl_7b":
        forward = _forward_qwen
    elif model_key == "internvl_2_5_8b":
        forward = _forward_internvl
    else:
        raise ValueError(f"Unsupported model for check: {model_key}")

    with torch.no_grad():
        data = forward(wrapper, image, response_ids)

    prompt_positions = resolve_prompt_positions(
        full_input_ids=data["input_ids"],
        prompt_tokenized_length=data["prompt_tokenized_length"],
        image_token_id=data["image_token_id"],
        visual_start=data["visual_start"],
        visual_end=data["visual_end"],
    )
    prediction_position = pre_token_prediction_positions(
        full_input_ids=data["input_ids"],
        prompt_tokenized_length=data["prompt_tokenized_length"],
        response_token_indices=[response_index],
        image_token_id=data["image_token_id"],
        visual_token_count=int(data["visual_end"] - data["visual_start"]),
        prompt_positions=prompt_positions,
    )[0]

    hook_token_hs, hook_patch_hs = hidden_states_from_captures(
        data["captures"],
        token_position=int(prediction_position),
        visual_start=data["visual_start"],
        visual_end=data["visual_end"],
    )
    out_hidden = data["out"].hidden_states
    model_token_hs = torch.stack(
        [hs[0, int(prediction_position), :] for hs in out_hidden[1:]],
        dim=0,
    )
    model_patch_hs = torch.stack(
        [hs[0, data["visual_start"] : data["visual_end"], :] for hs in out_hidden[1:]],
        dim=0,
    )

    token_stats = _compare_tensors(hook_token_hs, model_token_hs)
    patch_stats = _compare_tensors(hook_patch_hs, model_patch_hs)
    raw_token_stats = _compare_tensors(hook_token_hs[:-1], model_token_hs[:-1])
    raw_patch_stats = _compare_tensors(hook_patch_hs[:-1], model_patch_hs[:-1])
    final_norm_stats = _final_norm_compare(wrapper, hook_token_hs, hook_patch_hs, out_hidden, prediction_position, data)
    logits = data["out"].logits[0, int(prediction_position)]
    pred_token_id = int(logits.argmax().item())
    return {
        "image_id": int(case["image_id"]),
        "word": case["word"],
        "label": int(case["label"]),
        "response_index": int(response_index),
        "prediction_position": int(prediction_position),
        "target_token_id": int(target_token_id),
        "pred_token_id": pred_token_id,
        "target_token": wrapper.tokenizer.decode([target_token_id], skip_special_tokens=False),
        "pred_token": wrapper.tokenizer.decode([pred_token_id], skip_special_tokens=False),
        "visual_tokens": int(data["visual_end"] - data["visual_start"]),
        "seq_len": int(data["out"].attentions[0].shape[-1]),
        "num_layers": int(hook_token_hs.shape[0]),
        "token": token_stats,
        "patch": patch_stats,
        "raw_no_final_norm_token": raw_token_stats,
        "raw_no_final_norm_patch": raw_patch_stats,
        "final_norm": final_norm_stats,
    }


def _forward_llava(wrapper, image, response_ids):
    prompt_text = wrapper.cfg["prompt_template"]
    prefix_inputs = wrapper.processor(text=prompt_text, images=image, return_tensors="pt")
    prompt_len = int(prefix_inputs["input_ids"].shape[1])
    answer_ids = torch.tensor(response_ids, dtype=prefix_inputs["input_ids"].dtype).unsqueeze(0)
    full_inputs = dict(prefix_inputs)
    full_inputs["input_ids"] = torch.cat([prefix_inputs["input_ids"], answer_ids], dim=1)
    if "attention_mask" in prefix_inputs:
        full_inputs["attention_mask"] = torch.cat([prefix_inputs["attention_mask"], torch.ones_like(answer_ids)], dim=1)
    full_inputs = _to_device_dtype(full_inputs, wrapper.device, torch.float16)
    input_ids = full_inputs["input_ids"][0]
    image_token_id = int(getattr(wrapper.model.config, "image_token_index", -200))
    mask = input_ids == image_token_id
    if mask.any():
        img_start = int(mask.nonzero(as_tuple=True)[0][0].item())
        img_end = img_start + int(wrapper.num_visual_tokens)
    else:
        img_start, img_end = wrapper._find_img_range_from_embeds(full_inputs)
    out, captures = run_forward_with_dgst_captures(wrapper.model, **full_inputs)
    return {
        "out": out,
        "captures": captures,
        "input_ids": input_ids.tolist(),
        "prompt_tokenized_length": prompt_len,
        "visual_start": int(img_start),
        "visual_end": int(img_end),
        "image_token_id": image_token_id,
    }


def _forward_qwen(wrapper, image, response_ids):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = wrapper.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_inputs = wrapper.processor(text=[text], images=[image], return_tensors="pt")
    prompt_len = int(prefix_inputs["input_ids"].shape[1])
    answer_ids = torch.tensor(response_ids, dtype=prefix_inputs["input_ids"].dtype).unsqueeze(0)
    full_inputs = dict(prefix_inputs)
    full_inputs["input_ids"] = torch.cat([prefix_inputs["input_ids"], answer_ids], dim=1)
    if "attention_mask" in prefix_inputs:
        full_inputs["attention_mask"] = torch.cat([prefix_inputs["attention_mask"], torch.ones_like(answer_ids)], dim=1)
    full_inputs = _to_device_dtype(full_inputs, wrapper.device)
    input_ids = full_inputs["input_ids"][0]
    img_start, img_end = wrapper._find_vision_token_range(input_ids)
    out, captures = run_forward_with_dgst_captures(wrapper.model, **full_inputs)
    return {
        "out": out,
        "captures": captures,
        "input_ids": input_ids.tolist(),
        "prompt_tokenized_length": prompt_len,
        "visual_start": int(img_start),
        "visual_end": int(img_end),
        "image_token_id": int(wrapper.model.config.image_token_id),
    }


def _forward_internvl(wrapper, image, response_ids):
    pixel_values = wrapper._preprocess_image(image)
    prompt_input_ids, _, _ = wrapper._build_input_ids_with_image(pixel_values, prefix_token_ids=[])
    prompt_len = int(prompt_input_ids.shape[1])
    input_ids, img_start, img_end = wrapper._build_input_ids_with_image(pixel_values, prefix_token_ids=response_ids)
    attention_mask = torch.ones_like(input_ids)
    image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long, device=wrapper.device)
    out, captures = run_forward_with_dgst_captures(
        wrapper.model,
        input_ids=input_ids.to(wrapper.device),
        attention_mask=attention_mask.to(wrapper.device),
        pixel_values=pixel_values.to(wrapper.device),
        image_flags=image_flags,
    )
    return {
        "out": out,
        "captures": captures,
        "input_ids": input_ids[0].tolist(),
        "prompt_tokenized_length": prompt_len,
        "visual_start": int(img_start),
        "visual_end": int(img_end),
        "image_token_id": int(wrapper._img_ctx_id),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = a - b
    cosine = F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=-1)
    denom = b.norm(dim=-1) if b.ndim == 2 else b.reshape(b.shape[0], -1).norm(dim=-1)
    rel = diff.reshape(diff.shape[0], -1).norm(dim=-1) / denom.clamp_min(1e-12)
    return {
        "cosine_mean": float(cosine.mean().item()),
        "cosine_min": float(cosine.min().item()),
        "rel_l2_mean": float(rel.mean().item()),
        "rel_l2_max": float(rel.max().item()),
        "max_abs": float(diff.abs().max().item()),
    }


def _final_norm_compare(wrapper, hook_token_hs, hook_patch_hs, out_hidden, prediction_position: int, data: dict):
    norm = _resolve_final_norm(wrapper)
    if norm is None:
        return None
    with torch.no_grad():
        token = hook_token_hs[-1:].to(_module_device(norm))
        patch = hook_patch_hs[-1:].to(_module_device(norm))
        normed_token = norm(token)
        normed_patch = norm(patch)
    model_token = out_hidden[-1][0, int(prediction_position), :].unsqueeze(0)
    model_patch = out_hidden[-1][0, data["visual_start"] : data["visual_end"], :].unsqueeze(0)
    return {
        "token": _compare_tensors(normed_token, model_token),
        "patch": _compare_tensors(normed_patch, model_patch),
    }


def _resolve_final_norm(wrapper):
    candidates = []
    model = getattr(wrapper, "model", None)
    language_model = getattr(model, "language_model", None)
    for module in (language_model, model):
        candidates.extend(
            [
                getattr(getattr(module, "model", None), "norm", None),
                getattr(module, "norm", None),
            ]
        )
    for norm in candidates:
        if norm is not None:
            return norm
    return None


def _module_device(module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_row(row: dict) -> None:
    print(
        f"[Check] image={row['image_id']} word={row['word']!r} idx={row['response_index']} "
        f"pos={row['prediction_position']} visual={row['visual_tokens']} "
        f"target={row['target_token']!r} pred={row['pred_token']!r}"
    )
    print(
        f"  token cosine mean/min={row['token']['cosine_mean']:.8f}/"
        f"{row['token']['cosine_min']:.8f}, rel_l2 mean/max="
        f"{row['token']['rel_l2_mean']:.3e}/{row['token']['rel_l2_max']:.3e}, "
        f"max_abs={row['token']['max_abs']:.3e}"
    )
    print(
        f"  patch cosine mean/min={row['patch']['cosine_mean']:.8f}/"
        f"{row['patch']['cosine_min']:.8f}, rel_l2 mean/max="
        f"{row['patch']['rel_l2_mean']:.3e}/{row['patch']['rel_l2_max']:.3e}, "
        f"max_abs={row['patch']['max_abs']:.3e}"
    )
    print(
        f"  raw(no final norm) token cosine mean/min="
        f"{row['raw_no_final_norm_token']['cosine_mean']:.8f}/"
        f"{row['raw_no_final_norm_token']['cosine_min']:.8f}, rel_l2 mean/max="
        f"{row['raw_no_final_norm_token']['rel_l2_mean']:.3e}/"
        f"{row['raw_no_final_norm_token']['rel_l2_max']:.3e}"
    )
    print(
        f"  raw(no final norm) patch cosine mean/min="
        f"{row['raw_no_final_norm_patch']['cosine_mean']:.8f}/"
        f"{row['raw_no_final_norm_patch']['cosine_min']:.8f}, rel_l2 mean/max="
        f"{row['raw_no_final_norm_patch']['rel_l2_mean']:.3e}/"
        f"{row['raw_no_final_norm_patch']['rel_l2_max']:.3e}"
    )
    if row["final_norm"] is not None:
        print(
            f"  final-norm token cosine mean/min="
            f"{row['final_norm']['token']['cosine_mean']:.8f}/"
            f"{row['final_norm']['token']['cosine_min']:.8f}, rel_l2 mean/max="
            f"{row['final_norm']['token']['rel_l2_mean']:.3e}/"
            f"{row['final_norm']['token']['rel_l2_max']:.3e}"
        )
        print(
            f"  final-norm patch cosine mean/min="
            f"{row['final_norm']['patch']['cosine_mean']:.8f}/"
            f"{row['final_norm']['patch']['cosine_min']:.8f}, rel_l2 mean/max="
            f"{row['final_norm']['patch']['rel_l2_mean']:.3e}/"
            f"{row['final_norm']['patch']['rel_l2_max']:.3e}"
        )


def _summary(rows: list[dict]) -> dict[str, float]:
    result = {}
    for group in ("token", "patch", "raw_no_final_norm_token", "raw_no_final_norm_patch"):
        for metric in ("cosine_mean", "cosine_min", "rel_l2_mean", "rel_l2_max", "max_abs"):
            values = [float(row[group][metric]) for row in rows]
            result[f"{group}_{metric}_avg"] = float(sum(values) / len(values))
            result[f"{group}_{metric}_worst"] = float(min(values) if "cosine" in metric else max(values))
    if all(row.get("final_norm") is not None for row in rows):
        for group in ("token", "patch"):
            for metric in ("cosine_mean", "cosine_min", "rel_l2_mean", "rel_l2_max", "max_abs"):
                values = [float(row["final_norm"][group][metric]) for row in rows]
                result[f"final_norm_{group}_{metric}_avg"] = float(sum(values) / len(values))
                result[f"final_norm_{group}_{metric}_worst"] = float(min(values) if "cosine" in metric else max(values))
    return result


def _to_device_dtype(inputs: dict[str, Any], device: str, dtype=None) -> dict[str, Any]:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if dtype is not None and value.is_floating_point():
                result[key] = value.to(device=device, dtype=dtype)
            else:
                result[key] = value.to(device=device)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()
