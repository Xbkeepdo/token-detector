#!/usr/bin/env python3
"""Create GLSim-style spatial overlays for shared COCO examples across LVLMs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

from plot_attention_gate_source_heatmaps import (
    LABEL_NAMES,
    MATRIX_KEYS,
    MODEL_NAMES,
    iter_pickle_records,
)


MODELS = ("llava_1_5_7b", "qwen2_5_vl_7b", "internvl_2_5_8b")
ROW_NAMES = ("Raw attention", "Gate-weighted attention", "Source distribution")


@dataclass
class Candidate:
    score: float
    record: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", default="COCO4000-all")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 15, 20, 25])
    parser.add_argument("--samples-per-label", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--coco-images",
        type=Path,
        default=Path("/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco/val2014"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco/annotations/instances_val2014.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/coco4000_all_glsim_style_spatial_maps_shared10"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sanitize(values: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = sanitize(values)
    totals = values.sum(dim=1, keepdim=True)
    return torch.where(totals > 0, values / totals.clamp_min(1e-12), torch.zeros_like(values))


def matrices_for_plot(record: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = MATRIX_KEYS["vv"]
    attention = sanitize(torch.as_tensor(record[keys["attention"]]))
    gate = torch.nan_to_num(
        torch.as_tensor(record[keys["gate"]]).float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    gated = attention * gate
    source = sanitize(torch.as_tensor(record[keys["source"]]))
    return attention, gated, source


def gate_effect_score(record: dict) -> float:
    attention, gated, _source = matrices_for_plot(record)
    attention = normalize_rows(attention)
    gated = normalize_rows(gated)
    return float((0.5 * torch.abs(attention - gated).sum(dim=1)).mean().item())


def freeze_record(record: dict) -> dict:
    keys = MATRIX_KEYS["vv"]
    frozen = {
        "image_id": int(record["image_id"]),
        "label": int(record["label"]),
        "token_str": str(record.get("token_str", "")),
        "response_token_idx": int(record.get("response_token_idx", -1)),
        keys["positions"]: [int(value) for value in record[keys["positions"]]],
    }
    for name in ("attention", "gate", "source"):
        key = keys[name]
        frozen[key] = torch.as_tensor(record[key]).detach().cpu().clone()
    return frozen


def label_image_sets(outputs_root: Path, experiment: str, models: list[str]) -> dict[str, dict[int, set[int]]]:
    result: dict[str, dict[int, set[int]]] = {}
    for model in models:
        path = outputs_root / model / experiment / "labeling.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        by_label = {0: set(), 1: set()}
        for image_id, item in data.items():
            for span in item.get("object_token_spans", []):
                label = int(span.get("label", -1))
                if label in by_label:
                    by_label[label].add(int(image_id))
        result[model] = by_label
    return result


def sampled_common_image_ids(
    *,
    outputs_root: Path,
    experiment: str,
    models: list[str],
    pool_size: int,
    seed: int,
) -> dict[int, set[int]]:
    label_sets = label_image_sets(outputs_root, experiment, models)
    rng = random.Random(seed)
    sampled = {}
    for label in (0, 1):
        common = sorted(set.intersection(*(label_sets[model][label] for model in models)))
        if len(common) < pool_size:
            raise ValueError(f"Only {len(common)} shared images available for label {label}; need {pool_size}.")
        sampled[label] = set(rng.sample(common, pool_size))
    return sampled


def collect_candidates(
    *,
    part_paths: list[Path],
    candidate_image_ids: dict[int, set[int]],
) -> tuple[dict[tuple[int, int], Candidate], int]:
    best: dict[tuple[int, int], Candidate] = {}
    record_count = 0
    required = {
        MATRIX_KEYS["vv"]["attention"],
        MATRIX_KEYS["vv"]["gate"],
        MATRIX_KEYS["vv"]["source"],
        MATRIX_KEYS["vv"]["positions"],
    }
    for record in iter_pickle_records(part_paths):
        record_count += 1
        label = int(record.get("label", -1))
        image_id = int(record.get("image_id", -1))
        if label not in candidate_image_ids or image_id not in candidate_image_ids[label]:
            continue
        if not required.issubset(record):
            continue
        score = gate_effect_score(record)
        key = (label, image_id)
        if key not in best or score > best[key].score:
            best[key] = Candidate(score=score, record=freeze_record(record))
    return best, record_count


def select_shared_images(
    candidates: dict[str, dict[tuple[int, int], Candidate]], models: list[str], count: int) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = {}
    for label in (0, 1):
        common = set.intersection(
            *(set(image_id for sample_label, image_id in candidates[model] if sample_label == label) for model in models)
        )
        if len(common) < count:
            raise RuntimeError(f"Only {len(common)} fully available shared feature examples for label {label}; need {count}.")
        model_rank = {}
        for model in models:
            scores = sorted(
                ((image_id, candidates[model][(label, image_id)].score) for image_id in common),
                key=lambda item: item[1],
                reverse=True,
            )
            denominator = max(1, len(scores) - 1)
            model_rank[model] = {image_id: 1.0 - rank / denominator for rank, (image_id, _score) in enumerate(scores)}
        ranked = sorted(
            common,
            key=lambda image_id: sum(model_rank[model][image_id] for model in models) / len(models),
            reverse=True,
        )
        selected[label] = ranked[:count]
    return selected


def target_info(record: dict, labeling: dict[str, dict]) -> tuple[str, str]:
    canonical = str(record["token_str"])
    response_index = int(record["response_token_idx"])
    item = labeling.get(str(record["image_id"]), {})
    for span in item.get("object_token_spans", []):
        if response_index in [int(value) for value in span.get("token_indices", [])]:
            return str(span.get("surface", canonical)), str(span.get("word", canonical))
    return canonical, canonical


def center_square_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)), (left, top)


def model_view(image: Image.Image, model: str) -> tuple[Image.Image, tuple[int, int]]:
    image = image.convert("RGB")
    return center_square_crop(image) if model == "llava_1_5_7b" else (image, (0, 0))


def closest_factor_grid(token_count: int, aspect_ratio: float) -> tuple[int, int]:
    candidates = []
    for height in range(1, int(math.sqrt(token_count)) + 1):
        if token_count % height:
            continue
        width = token_count // height
        for grid_height, grid_width in ((height, width), (width, height)):
            candidates.append((abs(math.log((grid_width / grid_height) / aspect_ratio)), grid_height, grid_width))
    if not candidates:
        raise ValueError(f"Cannot factor {token_count} visual tokens into a patch grid.")
    _error, grid_height, grid_width = min(candidates)
    return grid_height, grid_width


def visual_grid(model: str, token_count: int, image_size: tuple[int, int]) -> tuple[int, int]:
    if model == "llava_1_5_7b":
        grid = (24, 24)
    elif model == "internvl_2_5_8b":
        grid = (16, 16)
    else:
        width, height = image_size
        grid = closest_factor_grid(token_count, width / height)
    if grid[0] * grid[1] != token_count:
        raise ValueError(f"{model} grid {grid} does not match {token_count} visual tokens.")
    return grid


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(heatmap.astype(np.float32))
    return np.asarray(image.resize(size, resample=Image.Resampling.BICUBIC), dtype=np.float32).clip(0.0, 1.0)


def thermal_overlay(image: Image.Image, heatmap: np.ndarray, scale_max: float) -> tuple[np.ndarray, np.ndarray]:
    relative = np.clip(heatmap / max(scale_max, 1e-12), 0.0, 1.0)
    smooth = resize_heatmap(relative, image.size)
    base = np.asarray(image, dtype=np.float32) / 255.0
    color = plt.get_cmap("turbo")(smooth)[..., :3]
    alpha = (0.78 * np.power(smooth, 0.68))[..., None]
    return np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 1.0), smooth


def load_annotations(path: Path) -> tuple[dict[str, int], dict[int, list[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    category_ids = {str(item["name"]): int(item["id"]) for item in data["categories"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    return category_ids, by_image


def projected_boxes(
    *,
    image_id: int,
    canonical: str,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
    crop_offset: tuple[int, int],
    view_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    category_id = category_ids.get(canonical)
    if category_id is None:
        return []
    left, top = crop_offset
    view_width, view_height = view_size
    boxes = []
    for annotation in annotations_by_image.get(image_id, []):
        if int(annotation["category_id"]) != category_id:
            continue
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        x -= left
        y -= top
        x2 = min(view_width, x + width)
        y2 = min(view_height, y + height)
        x = max(0.0, x)
        y = max(0.0, y)
        if x2 > x and y2 > y:
            boxes.append((x, y, x2 - x, y2 - y))
    return boxes


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "token"


def plot_record(
    *,
    model: str,
    candidate: Candidate,
    labeling: dict[str, dict],
    image_path: Path,
    layers: list[int],
    output_stem: Path,
    dpi: int,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
) -> dict:
    record = candidate.record
    raw_image = Image.open(image_path)
    image, crop_offset = model_view(raw_image, model)
    raw_image.close()
    attention, gated, source = matrices_for_plot(record)
    matrices = (attention.cpu().numpy(), gated.cpu().numpy(), source.cpu().numpy())
    token_count = int(matrices[0].shape[1])
    grid_height, grid_width = visual_grid(model, token_count, image.size)
    layer_indices = [layer - 1 for layer in layers]
    if min(layer_indices) < 0 or max(layer_indices) >= matrices[0].shape[0]:
        raise ValueError(f"Layers {layers} outside 1..{matrices[0].shape[0]} for {model}.")

    surface, canonical = target_info(record, labeling)
    boxes = projected_boxes(
        image_id=int(record["image_id"]),
        canonical=canonical,
        category_ids=category_ids,
        annotations_by_image=annotations_by_image,
        crop_offset=crop_offset,
        view_size=image.size,
    )
    figure, axes = plt.subplots(3, len(layers), figsize=(3.7 * len(layers), 10.0), constrained_layout=True)
    if len(layers) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    retention = gated.sum(axis=1) / np.clip(attention.sum(axis=1), 1e-12, None)

    for column, (layer, layer_index) in enumerate(zip(layers, layer_indices)):
        raw_patch = matrices[0][layer_index].reshape(grid_height, grid_width)
        raw_max = float(raw_patch.max(initial=0.0))
        source_patch = matrices[2][layer_index].reshape(grid_height, grid_width)
        source_max = float(source_patch.max(initial=0.0))
        for row, matrix in enumerate(matrices):
            patch = matrix[layer_index].reshape(grid_height, grid_width)
            scale = raw_max if row < 2 else source_max
            overlay, _smooth = thermal_overlay(image, patch, scale)
            axis = axes[row, column]
            axis.imshow(overlay, interpolation="nearest")
            for x, y, width, height in boxes:
                axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=1.5))
            if row == 0:
                axis.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(ROW_NAMES[row], fontsize=11, fontweight="semibold", labelpad=8)
            if row == 1:
                axis.text(
                    0.02,
                    0.035,
                    f"retained mass: {retention[layer_index]:.3f}",
                    transform=axis.transAxes,
                    color="white",
                    fontsize=8.5,
                    bbox={"boxstyle": "square,pad=0.2", "facecolor": "black", "edgecolor": "none", "alpha": 0.6},
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#303030")

    label = int(record["label"])
    title = (
        f"{MODEL_NAMES[model]} | Target: '{surface}'"
        + (f" (COCO class: {canonical})" if surface.lower() != canonical.lower() else "")
        + f" | {LABEL_NAMES[label]} | Image {int(record['image_id'])}"
    )
    figure.suptitle(
        f"{title}\nVV visual scope; raw and gated rows share raw-attention scale per layer; red boxes are matching COCO ground truth.",
        fontsize=13.2,
        fontweight="bold",
        linespacing=1.35,
    )
    colorbar = figure.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo"), ax=axes, location="right", shrink=0.84, pad=0.012)
    colorbar.set_label("Relative spatial intensity", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "model": model,
        "model_display": MODEL_NAMES[model],
        "label": label,
        "label_name": LABEL_NAMES[label],
        "image_id": int(record["image_id"]),
        "response_token_idx": int(record["response_token_idx"]),
        "surface": surface,
        "canonical_word": canonical,
        "gate_effect_tv": candidate.score,
        "layers": layers,
        "visual_grid": [grid_height, grid_width],
        "gt_box_count": len(boxes),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def write_outputs(output_dir: Path, rows: list[dict], shared_images: dict[int, list[int]], scanned: dict[str, int]) -> None:
    (output_dir / "selected_samples.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "shared_image_ids.json").write_text(json.dumps(shared_images, indent=2), encoding="utf-8")
    csv_rows = [{**row, "layers": ",".join(map(str, row["layers"])), "visual_grid": "x".join(map(str, row["visual_grid"]))} for row in rows]
    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# GLSim-style shared-image spatial maps",
        "",
        "The same five hallucinated and five non-hallucinated COCO images are used for all three LVLMs.",
        "Each figure overlays VV visual-token maps at layers 5, 15, 20, and 25. Raw and gate-weighted maps share the raw-attention maximum within each layer, so the gate row preserves absolute attenuation. Source maps use their own maximum for localization. Red boxes mark matching COCO ground-truth boxes, when present.",
        "",
        "## Shared Image IDs",
        "",
        f"- Hallucinated: {', '.join(map(str, shared_images[0]))}",
        f"- Non-hallucinated: {', '.join(map(str, shared_images[1]))}",
        "",
        "| Image ID | Label | LLaVA target | Qwen target | InternVL target |",
        "|---:|---|---|---|---|",
    ]
    by_model_image = {(row["model"], row["image_id"]): row for row in rows}
    for label in (0, 1):
        for image_id in shared_images[label]:
            lines.append(
                f"| {image_id} | {LABEL_NAMES[label]} | "
                f"{by_model_image[('llava_1_5_7b', image_id)]['surface']} | "
                f"{by_model_image[('qwen2_5_vl_7b', image_id)]['surface']} | "
                f"{by_model_image[('internvl_2_5_8b', image_id)]['surface']} |"
            )
    lines.append("")
    for model in dict.fromkeys(row["model"] for row in rows):
        lines.extend([f"## {MODEL_NAMES[model]}", "", f"Scanned feature records: {scanned[model]}", ""])
        for row in [row for row in rows if row["model"] == model]:
            relative = Path(row["png"]).relative_to(output_dir)
            lines.append(f"- Image {row['image_id']} ({row['label_name']}, target `{row['surface']}`): [PNG]({relative.as_posix()})")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if set(args.models) != set(MODELS):
        raise ValueError("Shared-image comparison requires all three models.")
    if args.samples_per_label <= 0 or args.candidate_pool_size < args.samples_per_label:
        raise ValueError("candidate pool size must be at least samples per label.")
    candidate_ids = sampled_common_image_ids(
        outputs_root=args.outputs_root,
        experiment=args.experiment,
        models=args.models,
        pool_size=args.candidate_pool_size,
        seed=args.seed,
    )
    candidates = {}
    scanned = {}
    labelings = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        part_paths = sorted(experiment_dir.glob("features.part*.pkl"))
        print(f"[{model}] Scanning {len(part_paths)} feature parts for shared candidates...", flush=True)
        candidates[model], scanned[model] = collect_candidates(part_paths=part_paths, candidate_image_ids=candidate_ids)
        labelings[model] = json.loads((experiment_dir / "labeling.json").read_text(encoding="utf-8"))
        gc.collect()
    shared_images = select_shared_images(candidates, args.models, args.samples_per_label)
    category_ids, annotations_by_image = load_annotations(args.annotations)
    rows = []
    for model in args.models:
        for label in (0, 1):
            for rank, image_id in enumerate(shared_images[label], start=1):
                candidate = candidates[model][(label, image_id)]
                record = candidate.record
                surface, _canonical = target_info(record, labelings[model])
                image_path = args.coco_images / f"COCO_val2014_{image_id:012d}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                stem = args.output_dir / model / f"{LABEL_NAMES[label].lower().replace('-', '_')}_{rank:02d}_{safe_slug(surface)}_image{image_id}"
                print(f"[{model}] Plotting image {image_id}, {LABEL_NAMES[label]}, target {surface}...", flush=True)
                rows.append(
                    plot_record(
                        model=model,
                        candidate=candidate,
                        labeling=labelings[model],
                        image_path=image_path,
                        layers=args.layers,
                        output_stem=stem,
                        dpi=args.dpi,
                        category_ids=category_ids,
                        annotations_by_image=annotations_by_image,
                    )
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, rows, shared_images, scanned)
    print(f"Wrote {len(rows)} shared-image spatial figures to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
