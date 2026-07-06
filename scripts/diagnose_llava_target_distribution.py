#!/usr/bin/env python3
"""Diagnose LLaVA VP target distributions on a small COCO sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data.coco_loader import load_coco_samples
from features.dgst_t import (
    _relative_vll_target_distribution,
    _renormalize,
    _source_distribution,
    _topk_union_indices,
)
from models import build_model
from models.dgst_capture import (
    pre_token_prediction_positions,
    resolve_output_embedding_layer,
    resolve_prompt_positions,
    resolve_support_positions,
    run_forward_with_dgst_captures,
    target_logits_multi,
)
from models.llava_wrapper import IMAGE_TOKEN_INDEX, NUM_VISUAL_TOKENS, _to_device_dtype
from utils.config_utils import get_dataset_cfg, get_dgst_t_cfg, get_model_cfg, load_config
from utils.io_utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_configs_visualprompt_relativevll_cost_geo.yaml")
    parser.add_argument("--model", default="llava_1_5_7b")
    parser.add_argument(
        "--source-dir",
        default="outputs/llava_1_5_7b/COCO500-visualprompt-relativevll-cost-3way",
        help="Directory containing generations.json and labeling.json.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/llava_target_distribution_diagnostic",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--hall-images", type=int, default=5)
    parser.add_argument("--ranking-layers", nargs="+", type=int, default=[0, 8, 16, 24, 31])
    parser.add_argument("--rank-top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model_cfg = get_model_cfg(config, args.model)
    dataset_cfg = get_dataset_cfg(config)
    dgst_cfg = get_dgst_t_cfg(config)
    source_dir = Path(args.source_dir)
    labels = {int(k): v for k, v in load_json(source_dir / "labeling.json").items()}
    generations = {int(k): v for k, v in load_json(source_dir / "generations.json").items()}

    samples = load_coco_samples(
        images_dir=str(Path(dataset_cfg["coco_root"]) / "val2014"),
        instances_file=dataset_cfg["annotation_file"],
        captions_file=dataset_cfg["captions_file"],
        num_images=None,
        seed=dataset_cfg["seed"],
    )
    sample_by_id = {int(sample["image_id"]): sample for sample in samples}
    cases = _select_cases(labels, sample_by_id, args.num_images, args.hall_images)
    if not cases:
        raise RuntimeError("No suitable LLaVA diagnostic cases found.")

    print(f"[Diag] Loading {args.model} on {args.device}...")
    wrapper = build_model(args.model, model_cfg, device=args.device)

    layer_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    heatmap_paths: list[str] = []

    for case_index, case in enumerate(cases, start=1):
        print(
            f"[Diag] {case_index}/{len(cases)} image={case['image_id']} "
            f"word={case['word']!r} label={case['label']}"
        )
        case_layer_rows, case_rank_rows, case_meta, case_heatmap_paths = _diagnose_case(
            wrapper=wrapper,
            case=case,
            generations=generations,
            dgst_cfg=dgst_cfg,
            out_dir=out_dir,
            ranking_layers=args.ranking_layers,
            rank_top_k=args.rank_top_k,
        )
        layer_rows.extend(case_layer_rows)
        rank_rows.extend(case_rank_rows)
        selected_rows.append(case_meta)
        heatmap_paths.extend(case_heatmap_paths)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_csv = out_dir / "llava_10image_target_distribution_layerwise.csv"
    rank_csv = out_dir / "llava_10image_target_endpoint_ranking.csv"
    selected_csv = out_dir / "llava_10image_selected_cases.csv"
    _write_csv(layer_csv, layer_rows)
    _write_csv(rank_csv, rank_rows)
    _write_csv(selected_csv, selected_rows)

    mass_plot = out_dir / "llava_10image_target_mass_split_by_layer.png"
    mass_plot_pdf = mass_plot.with_suffix(".pdf")
    quality_plot = out_dir / "llava_10image_quality_rank_prompt_share.png"
    quality_plot_pdf = quality_plot.with_suffix(".pdf")
    _plot_mass_split(layer_rows, selected_rows, mass_plot, mass_plot_pdf)
    _plot_quality_prompt_share(layer_rows, selected_rows, quality_plot, quality_plot_pdf)

    summary_path = out_dir / "llava_10image_target_distribution_summary.md"
    _write_summary(
        summary_path=summary_path,
        selected_rows=selected_rows,
        layer_rows=layer_rows,
        mass_plot=mass_plot,
        quality_plot=quality_plot,
        layer_csv=layer_csv,
        rank_csv=rank_csv,
        selected_csv=selected_csv,
        heatmap_paths=heatmap_paths,
    )
    print(f"[Diag] Wrote {summary_path}")


def _select_cases(
    labels: dict[int, dict[str, Any]],
    sample_by_id: dict[int, dict[str, Any]],
    num_images: int,
    hall_images: int,
) -> list[dict[str, Any]]:
    hall_cases = []
    non_cases = []
    for image_id in sorted(labels):
        if image_id not in sample_by_id:
            continue
        spans = labels[image_id].get("object_token_spans") or []
        hall_span = next((span for span in spans if int(span.get("label", 0)) == 1), None)
        non_span = next((span for span in spans if int(span.get("label", 0)) == 0), None)
        if hall_span is not None:
            hall_cases.append(_case_from_span(image_id, sample_by_id[image_id], labels[image_id], hall_span))
        elif non_span is not None:
            non_cases.append(_case_from_span(image_id, sample_by_id[image_id], labels[image_id], non_span))

    hall_count = min(max(int(hall_images), 0), int(num_images), len(hall_cases))
    non_count = max(0, int(num_images) - hall_count)
    return [*hall_cases[:hall_count], *non_cases[:non_count]]


def _case_from_span(
    image_id: int,
    sample: dict[str, Any],
    label_info: dict[str, Any],
    span: dict[str, Any],
) -> dict[str, Any]:
    token_indices = [int(x) for x in (span.get("token_indices") or [])]
    return {
        "image_id": int(image_id),
        "image_path": sample["image_path"],
        "generated_text": label_info.get("generated_text", ""),
        "word": str(span.get("word", "")),
        "response_index": int(token_indices[0]),
        "label": int(span.get("label", 0)),
    }


def _diagnose_case(
    *,
    wrapper,
    case: dict[str, Any],
    generations: dict[int, dict[str, Any]],
    dgst_cfg: dict[str, Any],
    out_dir: Path,
    ranking_layers: list[int],
    rank_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
    image = Image.open(case["image_path"]).convert("RGB")
    response_ids = generations.get(int(case["image_id"]), {}).get("response_token_ids") or []
    if not response_ids:
        response_ids = wrapper.tokenizer.encode(case["generated_text"], add_special_tokens=False)
    response_ids = [int(token_id) for token_id in response_ids]
    response_index = int(case["response_index"])
    if response_index < 0 or response_index >= len(response_ids):
        raise ValueError(f"Invalid response index for image {case['image_id']}: {response_index}")
    target_token_id = int(response_ids[response_index])

    prompt_text = wrapper.cfg["prompt_template"]
    prefix_inputs = wrapper.processor(text=prompt_text, images=image, return_tensors="pt")
    prompt_len = int(prefix_inputs["input_ids"].shape[1])
    answer_ids = torch.tensor(response_ids, dtype=prefix_inputs["input_ids"].dtype).unsqueeze(0)
    full_inputs = dict(prefix_inputs)
    full_inputs["input_ids"] = torch.cat([prefix_inputs["input_ids"], answer_ids], dim=1)
    if "attention_mask" in prefix_inputs:
        full_inputs["attention_mask"] = torch.cat(
            [prefix_inputs["attention_mask"], torch.ones_like(answer_ids)],
            dim=1,
        )
    full_inputs = _to_device_dtype(full_inputs, wrapper.device, torch.float16)
    input_ids = full_inputs["input_ids"][0].tolist()

    image_token_id = int(getattr(wrapper.model.config, "image_token_index", IMAGE_TOKEN_INDEX))
    image_positions = [idx for idx, token_id in enumerate(input_ids) if int(token_id) == image_token_id]
    if image_positions:
        visual_start = int(image_positions[0])
        visual_end = visual_start + NUM_VISUAL_TOKENS
    else:
        visual_start, visual_end = wrapper._find_img_range_from_embeds(full_inputs)
    visual_count = int(visual_end - visual_start)

    outputs, captures = run_forward_with_dgst_captures(
        wrapper.model,
        output_hidden_states=False,
        **full_inputs,
    )
    full_prompt_positions = resolve_prompt_positions(
        full_input_ids=input_ids,
        prompt_tokenized_length=prompt_len,
        image_token_id=image_token_id,
        visual_start=visual_start,
        visual_end=visual_end,
    )
    support_prompt_positions = wrapper._resolve_dgst_prompt_support_positions(
        full_input_ids=input_ids,
        prompt_tokenized_length=prompt_len,
        image_token_id=image_token_id,
        visual_start=visual_start,
        visual_end=visual_end,
        cfg_dgst_t=dgst_cfg,
    )
    if support_prompt_positions is None:
        support_prompt_positions = full_prompt_positions
    support_positions = resolve_support_positions(
        visual_start=visual_start,
        visual_end=visual_end,
        prompt_positions=support_prompt_positions,
        support_scope="visual_prompt",
    )
    prediction_position = pre_token_prediction_positions(
        full_input_ids=input_ids,
        prompt_tokenized_length=prompt_len,
        response_token_indices=[response_index],
        image_token_id=image_token_id,
        visual_token_count=visual_count,
        prompt_positions=full_prompt_positions,
    )[0]

    output_layer = resolve_output_embedding_layer(wrapper.model)
    support_index = torch.tensor(support_positions, dtype=torch.long, device=full_inputs["input_ids"].device)
    prompt_set = set(int(pos) for pos in support_prompt_positions)
    prompt_support_token_texts = _prompt_support_token_texts(
        support_prompt_positions=support_prompt_positions,
        input_ids=input_ids,
        tokenizer=wrapper.tokenizer,
        visual_start=visual_start,
        visual_end=visual_end,
        visual_count=visual_count,
    )
    visual_mask_np = np.array(
        [visual_start <= int(pos) < visual_end for pos in support_positions],
        dtype=bool,
    )
    prompt_mask_np = ~visual_mask_np

    layer_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    heatmap_paths: list[str] = []
    last_layer_payload = None

    for layer_idx, capture in enumerate(captures):
        h_mid = capture["h_mid"][0]
        o_ffn = capture["o_ffn"][0]
        support_states = h_mid.index_select(0, support_index).float()
        attention_row = capture["attn_weights"][0, :, int(prediction_position), :]
        support_attention = attention_row.index_select(1, support_index.to(attention_row.device)).mean(dim=0)
        attention_dist = _renormalize(support_attention.float())
        target_logits = target_logits_multi(
            output_layer=output_layer,
            states=support_states,
            target_token_ids=[target_token_id],
            chunk_size=128,
        )[:, 0]
        target_vp, gate_vp, barrier_vp, stats_vp = _relative_vll_target_distribution(
            attention_dist=attention_dist,
            target_logits=target_logits,
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            candidate_scope="visual_prompt",
            stat_prefix="vp",
            epsilon=float(dgst_cfg.get("relative_vll_mad_epsilon", 1e-6)),
            barrier_margin=float(dgst_cfg.get("relative_barrier_margin", 0.5)),
            barrier_max=float(dgst_cfg.get("relative_barrier_max", 3.0)),
        )
        target_v, gate_v, _barrier_v, stats_v = _relative_vll_target_distribution(
            attention_dist=attention_dist,
            target_logits=target_logits,
            support_positions=support_positions,
            visual_start=visual_start,
            visual_end=visual_end,
            candidate_scope="visual",
            stat_prefix="v",
            epsilon=float(dgst_cfg.get("relative_vll_mad_epsilon", 1e-6)),
            barrier_margin=float(dgst_cfg.get("relative_barrier_margin", 0.5)),
            barrier_max=float(dgst_cfg.get("relative_barrier_max", 3.0)),
        )
        source_dist = _source_distribution(
            source_update=o_ffn[int(prediction_position), :].float(),
            support_states=support_states,
            tau=float(dgst_cfg.get("tau", 0.07)),
            mode=str(dgst_cfg.get("source_distribution_mode", "softmax")),
        )
        union_support = _topk_union_indices(
            source_dist,
            target_vp,
            int(dgst_cfg.get("transport_top_k", 64)),
        )

        target_np = target_vp.detach().float().cpu().numpy()
        target_v_np = target_v.detach().float().cpu().numpy()
        gate_np = gate_vp.detach().float().cpu().numpy()
        barrier_np = barrier_vp.detach().float().cpu().numpy()
        attention_np = attention_dist.detach().float().cpu().numpy()
        logits_np = target_logits.detach().float().cpu().numpy()
        source_np = source_dist.detach().float().cpu().numpy()
        union_np = union_support.detach().cpu().numpy()

        entropy = _entropy(target_np)
        sorted_target = np.argsort(-target_np)
        sorted_gate = np.argsort(-gate_np)
        top_target = sorted_target[: min(rank_top_k, len(sorted_target))]
        top_gate = sorted_gate[: min(rank_top_k, len(sorted_gate))]
        row = {
            **_case_prefix(case, target_token_id, prediction_position),
            "layer": int(layer_idx),
            "vp_target_visual_mass": float(target_np[visual_mask_np].sum()),
            "vp_target_prompt_mass": float(target_np[prompt_mask_np].sum()),
            "v_target_visual_mass": float(target_v_np[visual_mask_np].sum()),
            "attention_visual_mass": float(attention_np[visual_mask_np].sum()),
            "attention_prompt_mass": float(attention_np[prompt_mask_np].sum()),
            "source_visual_mass": float(source_np[visual_mask_np].sum()),
            "source_prompt_mass": float(source_np[prompt_mask_np].sum()),
            "vp_gate_visual_mean": float(gate_np[visual_mask_np].mean()),
            "vp_gate_prompt_mean": float(gate_np[prompt_mask_np].mean()),
            "vp_gate_visual_max": float(gate_np[visual_mask_np].max()),
            "vp_gate_prompt_max": float(gate_np[prompt_mask_np].max()),
            "vp_target_entropy": float(entropy),
            "vp_target_effective_n": float(math.exp(entropy)),
            "vp_top1_target_type": _endpoint_type(support_positions[int(sorted_target[0])], visual_start, visual_end),
            "vp_top1_gate_type": _endpoint_type(support_positions[int(sorted_gate[0])], visual_start, visual_end),
            "vp_top16_target_prompt_count": int(sum(prompt_mask_np[top_target[:16]])),
            "vp_top16_gate_prompt_count": int(sum(prompt_mask_np[top_gate[:16]])),
            "top64_union_size": int(len(union_np)),
            "top64_union_prompt_count": int(sum(prompt_mask_np[union_np])),
            "top64_union_visual_count": int(sum(visual_mask_np[union_np])),
            "vp_logit_median": float(stats_vp["vp_logit_median"]),
            "vp_logit_mad": float(stats_vp["vp_logit_mad"]),
            "v_logit_median": float(stats_v["v_logit_median"]),
            "v_logit_mad": float(stats_v["v_logit_mad"]),
        }
        layer_rows.append(row)

        if layer_idx in ranking_layers:
            rank_rows.extend(
                _ranking_rows(
                    case=case,
                    target_token_id=target_token_id,
                    prediction_position=prediction_position,
                    layer_idx=layer_idx,
                    support_positions=support_positions,
                    input_ids=input_ids,
                    tokenizer=wrapper.tokenizer,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    prompt_positions=prompt_set,
                    visual_count=visual_count,
                    target_np=target_np,
                    attention_np=attention_np,
                    gate_np=gate_np,
                    barrier_np=barrier_np,
                    logits_np=logits_np,
                    source_np=source_np,
                    sort_name="target_mass",
                    order=sorted_target,
                    top_k=rank_top_k,
                )
            )
            rank_rows.extend(
                _ranking_rows(
                    case=case,
                    target_token_id=target_token_id,
                    prediction_position=prediction_position,
                    layer_idx=layer_idx,
                    support_positions=support_positions,
                    input_ids=input_ids,
                    tokenizer=wrapper.tokenizer,
                    visual_start=visual_start,
                    visual_end=visual_end,
                    prompt_positions=prompt_set,
                    visual_count=visual_count,
                    target_np=target_np,
                    attention_np=attention_np,
                    gate_np=gate_np,
                    barrier_np=barrier_np,
                    logits_np=logits_np,
                    source_np=source_np,
                    sort_name="quality_gate",
                    order=sorted_gate,
                    top_k=rank_top_k,
                )
            )

        if layer_idx == len(captures) - 1:
            last_layer_payload = {
                "target": target_np.copy(),
                "gate": gate_np.copy(),
                "attention": attention_np.copy(),
                "visual_mask": visual_mask_np.copy(),
                "support_positions": support_positions,
            }

    if last_layer_payload is not None:
        heatmap_paths = _plot_case_heatmaps(
            image=image,
            case=case,
            payload=last_layer_payload,
            visual_start=visual_start,
            visual_end=visual_end,
            out_dir=out_dir,
        )

    final_layer = layer_rows[-1]
    meta = {
        **_case_prefix(case, target_token_id, prediction_position),
        "image_path": case["image_path"],
        "final_vp_target_visual_mass": final_layer["vp_target_visual_mass"],
        "final_vp_target_prompt_mass": final_layer["vp_target_prompt_mass"],
        "final_attention_visual_mass": final_layer["attention_visual_mass"],
        "final_attention_prompt_mass": final_layer["attention_prompt_mass"],
        "final_source_visual_mass": final_layer["source_visual_mass"],
        "final_source_prompt_mass": final_layer["source_prompt_mass"],
        "final_vp_top1_target_type": final_layer["vp_top1_target_type"],
        "final_vp_top1_gate_type": final_layer["vp_top1_gate_type"],
        "prompt_support_mode": str(
            dgst_cfg.get(
                "dgst_t_prompt_support_mode",
                dgst_cfg.get("prompt_support_mode", "full"),
            )
        ),
        "prompt_support_text": str(
            dgst_cfg.get(
                "dgst_t_user_prompt_text",
                dgst_cfg.get("user_prompt_text", ""),
            )
        ),
        "prompt_support_token_count": int(len(prompt_support_token_texts)),
        "prompt_support_tokens": " / ".join(prompt_support_token_texts),
    }
    return layer_rows, rank_rows, meta, heatmap_paths


def _case_prefix(case: dict[str, Any], target_token_id: int, prediction_position: int) -> dict[str, Any]:
    return {
        "image_id": int(case["image_id"]),
        "word": str(case["word"]),
        "label": int(case["label"]),
        "response_index": int(case["response_index"]),
        "target_token_id": int(target_token_id),
        "prediction_position": int(prediction_position),
    }


def _ranking_rows(
    *,
    case: dict[str, Any],
    target_token_id: int,
    prediction_position: int,
    layer_idx: int,
    support_positions: list[int],
    input_ids: list[int],
    tokenizer,
    visual_start: int,
    visual_end: int,
    prompt_positions: set[int],
    visual_count: int,
    target_np: np.ndarray,
    attention_np: np.ndarray,
    gate_np: np.ndarray,
    barrier_np: np.ndarray,
    logits_np: np.ndarray,
    source_np: np.ndarray,
    sort_name: str,
    order: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for rank, local_idx in enumerate(order[: min(int(top_k), len(order))], start=1):
        local_idx = int(local_idx)
        support_pos = int(support_positions[local_idx])
        endpoint_type = _endpoint_type(support_pos, visual_start, visual_end)
        token_text = ""
        visual_index = ""
        visual_row = ""
        visual_col = ""
        if endpoint_type == "visual":
            visual_index = int(support_pos - visual_start)
            visual_row = int(visual_index) // 24
            visual_col = int(visual_index) % 24
        else:
            tokenized_pos = _tokenized_position_from_merged(
                support_pos,
                input_ids=input_ids,
                visual_start=visual_start,
                visual_end=visual_end,
                visual_count=visual_count,
            )
            if 0 <= tokenized_pos < len(input_ids):
                token_text = tokenizer.decode([int(input_ids[tokenized_pos])], skip_special_tokens=False)
        rows.append(
            {
                **_case_prefix(case, target_token_id, prediction_position),
                "layer": int(layer_idx),
                "sort_by": sort_name,
                "rank": int(rank),
                "local_support_index": int(local_idx),
                "support_position": int(support_pos),
                "endpoint_type": endpoint_type,
                "is_prompt_position": int(support_pos in prompt_positions),
                "prompt_token_text": token_text,
                "visual_index": visual_index,
                "visual_row": visual_row,
                "visual_col": visual_col,
                "target_mass": float(target_np[local_idx]),
                "attention_mass": float(attention_np[local_idx]),
                "quality_gate": float(gate_np[local_idx]),
                "relative_logit": float(logits_np[local_idx]),
                "barrier": float(barrier_np[local_idx]),
                "source_mass": float(source_np[local_idx]),
            }
        )
    return rows


def _prompt_support_token_texts(
    *,
    support_prompt_positions: list[int],
    input_ids: list[int],
    tokenizer,
    visual_start: int,
    visual_end: int,
    visual_count: int,
) -> list[str]:
    texts = []
    for support_pos in support_prompt_positions:
        tokenized_pos = _tokenized_position_from_merged(
            int(support_pos),
            input_ids=input_ids,
            visual_start=visual_start,
            visual_end=visual_end,
            visual_count=visual_count,
        )
        if 0 <= tokenized_pos < len(input_ids):
            texts.append(tokenizer.decode([int(input_ids[tokenized_pos])], skip_special_tokens=False))
    return texts


def _endpoint_type(position: int, visual_start: int, visual_end: int) -> str:
    return "visual" if int(visual_start) <= int(position) < int(visual_end) else "prompt"


def _tokenized_position_from_merged(
    position: int,
    *,
    input_ids: list[int],
    visual_start: int,
    visual_end: int,
    visual_count: int,
) -> int:
    image_positions = [idx for idx, token_id in enumerate(input_ids) if int(token_id) == IMAGE_TOKEN_INDEX]
    if image_positions and int(position) >= int(visual_end):
        return int(position) - int(visual_count) + 1
    return int(position)


def _entropy(values: np.ndarray) -> float:
    probs = np.asarray(values, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _plot_case_heatmaps(
    *,
    image: Image.Image,
    case: dict[str, Any],
    payload: dict[str, Any],
    visual_start: int,
    visual_end: int,
    out_dir: Path,
) -> list[str]:
    support_positions = payload["support_positions"]
    visual_values = {}
    for key in ("target", "gate", "attention"):
        grid = np.zeros((24, 24), dtype=np.float32)
        for local_idx, position in enumerate(support_positions):
            if visual_start <= int(position) < visual_end:
                visual_idx = int(position) - int(visual_start)
                grid[visual_idx // 24, visual_idx % 24] = float(payload[key][local_idx])
        visual_values[key] = grid

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")
    for ax, key, title in zip(
        axes[1:],
        ("target", "gate", "attention"),
        ("VP target mass", "quality gate", "attention"),
    ):
        ax.imshow(image.resize((336, 336)))
        heat = ax.imshow(
            np.kron(visual_values[key], np.ones((14, 14))),
            cmap="magma",
            alpha=0.58,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"image={case['image_id']} word={case['word']} label={case['label']} final layer",
        fontsize=11,
    )
    fig.tight_layout()
    safe_word = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case["word"]))[:40]
    path = out_dir / f"llava_target_heatmap_{case['image_id']}_{safe_word}.png"
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [str(path), str(pdf_path)]


def _plot_mass_split(
    layer_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    rows_by_case = _rows_by_case(layer_rows)
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, meta in zip(axes, selected_rows):
        key = (int(meta["image_id"]), int(meta["response_index"]))
        rows = rows_by_case[key]
        layers = [int(row["layer"]) for row in rows]
        ax.plot(layers, [row["vp_target_visual_mass"] for row in rows], label="VP target visual mass")
        ax.plot(layers, [row["vp_target_prompt_mass"] for row in rows], label="VP target prompt mass")
        ax.plot(layers, [row["attention_visual_mass"] for row in rows], "--", label="attention visual mass")
        ax.set_title(f"{meta['image_id']} {meta['word']} y={meta['label']}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes[len(selected_rows):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="best")
    fig.supxlabel("layer")
    fig.supylabel("mass")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def _plot_quality_prompt_share(
    layer_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    rows_by_case = _rows_by_case(layer_rows)
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, meta in zip(axes, selected_rows):
        key = (int(meta["image_id"]), int(meta["response_index"]))
        rows = rows_by_case[key]
        layers = [int(row["layer"]) for row in rows]
        ax.plot(layers, [row["vp_top16_target_prompt_count"] / 16.0 for row in rows], label="top16 target prompt share")
        ax.plot(layers, [row["vp_top16_gate_prompt_count"] / 16.0 for row in rows], label="top16 quality prompt share")
        ax.plot(layers, [row["top64_union_prompt_count"] / max(row["top64_union_size"], 1) for row in rows], label="top64 union prompt share")
        ax.set_title(f"{meta['image_id']} {meta['word']} y={meta['label']}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    for ax in axes[len(selected_rows):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="best")
    fig.supxlabel("layer")
    fig.supylabel("prompt share")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)


def _rows_by_case(layer_rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in layer_rows:
        key = (int(row["image_id"]), int(row["response_index"]))
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["layer"]))
    return grouped


def _write_summary(
    *,
    summary_path: Path,
    selected_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    mass_plot: Path,
    quality_plot: Path,
    layer_csv: Path,
    rank_csv: Path,
    selected_csv: Path,
    heatmap_paths: list[str],
) -> None:
    final_rows = [row for row in layer_rows if int(row["layer"]) == max(int(r["layer"]) for r in layer_rows)]
    by_label = {}
    for label in (0, 1):
        rows = [row for row in final_rows if int(row["label"]) == label]
        if rows:
            by_label[label] = {
                "n": len(rows),
                "target_prompt_mass": float(np.mean([row["vp_target_prompt_mass"] for row in rows])),
                "attention_prompt_mass": float(np.mean([row["attention_prompt_mass"] for row in rows])),
                "source_prompt_mass": float(np.mean([row["source_prompt_mass"] for row in rows])),
                "top16_target_prompt_share": float(np.mean([row["vp_top16_target_prompt_count"] / 16.0 for row in rows])),
                "top16_quality_prompt_share": float(np.mean([row["vp_top16_gate_prompt_count"] / 16.0 for row in rows])),
            }

    lines = [
        "# LLaVA Target Distribution Diagnostic",
        "",
        "Scope: 10 LLaVA COCO500 images, VP/VP support, relative-VLL target distribution.",
        "",
        "## Final-layer Aggregate",
        "",
        "| Label | n | target prompt mass | attention prompt mass | source prompt mass | top16 target prompt share | top16 quality prompt share |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in by_label.items():
        lines.append(
            "| {label} | {n} | {target_prompt_mass:.4f} | {attention_prompt_mass:.4f} | "
            "{source_prompt_mass:.4f} | {top16_target_prompt_share:.4f} | "
            "{top16_quality_prompt_share:.4f} |".format(label=label, **stats)
        )

    if selected_rows:
        first = selected_rows[0]
        lines.extend(
            [
                "",
                "## Prompt Support",
                "",
                f"- mode: `{first.get('prompt_support_mode', 'full')}`",
                f"- text: `{first.get('prompt_support_text', '')}`",
                f"- selected token count: `{first.get('prompt_support_token_count', '')}`",
                f"- selected tokens: `{first.get('prompt_support_tokens', '')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Selected Cases",
            "",
            "| image_id | word | label | final target prompt mass | final source prompt mass | top target type | top quality type |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in selected_rows:
        lines.append(
            f"| {row['image_id']} | {row['word']} | {row['label']} | "
            f"{float(row['final_vp_target_prompt_mass']):.4f} | "
            f"{float(row['final_source_prompt_mass']):.4f} | "
            f"{row['final_vp_top1_target_type']} | {row['final_vp_top1_gate_type']} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"- `{mass_plot.name}`",
            f"- `{mass_plot.with_suffix('.pdf').name}`",
            f"- `{quality_plot.name}`",
            f"- `{quality_plot.with_suffix('.pdf').name}`",
            "- `llava_target_heatmap_*_{word}.{png,pdf}`",
            "",
            "## CSV",
            "",
            f"- `{selected_csv.name}`",
            f"- `{layer_csv.name}`",
            f"- `{rank_csv.name}`",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
