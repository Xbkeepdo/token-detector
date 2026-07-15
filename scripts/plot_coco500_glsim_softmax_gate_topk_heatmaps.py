#!/usr/bin/env python3
"""Plot COCO500 GLSim-style Top-K maps for softmax(gate), attention, and source."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
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
from plot_spatial_attention_gate_source_heatmaps import (
    MODELS,
    model_view,
    projected_boxes,
    safe_slug,
    thermal_overlay,
    visual_grid,
)


SIGNALS = (
    ("gate_softmax", "Softmax(gate) Top-K"),
    ("attention", "Attention Top-K"),
    ("source", "Source distribution Top-K"),
)

EXCLUDED_LARGE_CATEGORIES = {
    "airplane",
    "bear",
    "bed",
    "bus",
    "couch",
    "cow",
    "dining table",
    "elephant",
    "giraffe",
    "horse",
    "person",
    "refrigerator",
    "train",
    "truck",
}


@dataclass(frozen=True)
class TargetSpec:
    model: str
    label: int
    image_id: int
    response_token_idx: int
    canonical_word: str
    surface: str
    object_fraction: float
    size_basis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", default="COCO500-mass-dist-topk")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 15, 20, 25])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--softmax-within-topk",
        action="store_true",
        help="After selecting each signal's Top-K, softmax only those retained values for visualization.",
    )
    parser.add_argument("--topk-softmax-temperature", type=float, default=1.0)
    parser.add_argument("--samples-per-label", type=int, default=5)
    parser.add_argument(
        "--max-object-fraction",
        type=float,
        default=0.02,
        help="Maximum object box/image area fraction; hallucinations use the category median.",
    )
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.001,
        help="Minimum GT-box/image area for non-hallucinated targets so the object remains visible.",
    )
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
        default=Path("outputs/coco500_glsim_softmax_gate_attention_source_topk32_small_shared10"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_coco_metadata(path: Path) -> tuple[
    dict[str, int],
    dict[int, list[dict]],
    dict[int, tuple[int, int]],
    dict[str, float],
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    category_ids = {str(item["name"]).lower(): int(item["id"]) for item in data["categories"]}
    category_names = {int(item["id"]): str(item["name"]).lower() for item in data["categories"]}
    image_sizes = {
        int(item["id"]): (int(item["width"]), int(item["height"]))
        for item in data["images"]
    }
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    category_fractions: dict[str, list[float]] = defaultdict(list)
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)
        width, height = image_sizes[image_id]
        area = float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
        category = category_names[int(annotation["category_id"])]
        category_fractions[category].append(area / max(float(width * height), 1.0))
    category_medians = {
        category: float(statistics.median(values))
        for category, values in category_fractions.items()
        if values
    }
    return category_ids, annotations_by_image, image_sizes, category_medians


def actual_object_fraction(
    *,
    image_id: int,
    canonical: str,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
    image_sizes: dict[int, tuple[int, int]],
) -> float | None:
    category_id = category_ids.get(canonical)
    if category_id is None:
        return None
    width, height = image_sizes[image_id]
    fractions = []
    for annotation in annotations_by_image.get(image_id, []):
        if int(annotation["category_id"]) != category_id:
            continue
        area = float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]))
        fractions.append(area / max(float(width * height), 1.0))
    return min(fractions) if fractions else None


def candidate_targets(
    *,
    model: str,
    labeling: dict[str, dict],
    max_object_fraction: float,
    min_visible_fraction: float,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
    image_sizes: dict[int, tuple[int, int]],
    category_medians: dict[str, float],
) -> dict[int, dict[int, list[TargetSpec]]]:
    candidates: dict[int, dict[int, list[TargetSpec]]] = {0: defaultdict(list), 1: defaultdict(list)}
    seen = set()
    for image_id_text, item in labeling.items():
        image_id = int(image_id_text)
        for span in item.get("object_token_spans", []):
            label = int(span.get("label", -1))
            token_indices = [int(value) for value in span.get("token_indices", [])]
            canonical = str(span.get("word", "")).strip().lower()
            if (
                label not in candidates
                or not token_indices
                or canonical not in category_ids
                or canonical in EXCLUDED_LARGE_CATEGORIES
            ):
                continue
            actual_fraction = actual_object_fraction(
                image_id=image_id,
                canonical=canonical,
                category_ids=category_ids,
                annotations_by_image=annotations_by_image,
                image_sizes=image_sizes,
            )
            if label == 0:
                # A hallucinated category should not have a matching COCO box in the image.
                if actual_fraction is not None:
                    continue
                object_fraction = category_medians.get(canonical, math.inf)
                size_basis = "COCO category median"
            else:
                if actual_fraction is None:
                    continue
                object_fraction = actual_fraction
                size_basis = "smallest matching GT box"
            if (
                not math.isfinite(object_fraction)
                or object_fraction > max_object_fraction
                or (label == 1 and object_fraction < min_visible_fraction)
            ):
                continue
            key = (label, image_id, token_indices[0], canonical)
            if key in seen:
                continue
            seen.add(key)
            candidates[label][image_id].append(
                TargetSpec(
                    model=model,
                    label=label,
                    image_id=image_id,
                    response_token_idx=token_indices[0],
                    canonical_word=canonical,
                    surface=str(span.get("surface", canonical)),
                    object_fraction=float(object_fraction),
                    size_basis=size_basis,
                )
            )
    for label in candidates:
        for image_id in candidates[label]:
            candidates[label][image_id].sort(
                key=lambda value: (value.object_fraction, value.canonical_word, value.response_token_idx)
            )
    return candidates


def select_shared_targets(
    candidates: dict[str, dict[int, dict[int, list[TargetSpec]]]],
    models: list[str],
    samples_per_label: int,
) -> tuple[dict[int, list[int]], dict[tuple[str, int, int], TargetSpec]]:
    selected_images: dict[int, list[int]] = {}
    target_specs: dict[tuple[str, int, int], TargetSpec] = {}
    for label in (0, 1):
        common_images = set.intersection(
            *(set(candidates[model][label]) for model in models)
        )
        ranked = []
        for image_id in common_images:
            per_model = {
                model: candidates[model][label][image_id][0]
                for model in models
            }
            signature = tuple(per_model[model].canonical_word for model in models)
            worst_fraction = max(spec.object_fraction for spec in per_model.values())
            mean_fraction = float(np.mean([spec.object_fraction for spec in per_model.values()]))
            ranked.append((worst_fraction, mean_fraction, signature, image_id, per_model))
        ranked.sort(key=lambda row: (row[0], row[1], row[3]))

        chosen = []
        seen_signatures = set()
        for row in ranked:
            if row[2] in seen_signatures:
                continue
            chosen.append(row)
            seen_signatures.add(row[2])
            if len(chosen) == samples_per_label:
                break
        if len(chosen) < samples_per_label:
            chosen_ids = {row[3] for row in chosen}
            chosen.extend(row for row in ranked if row[3] not in chosen_ids)
            chosen = chosen[:samples_per_label]
        if len(chosen) < samples_per_label:
            raise RuntimeError(
                f"Only {len(chosen)} shared small-object images for label {label}; need {samples_per_label}."
            )
        selected_images[label] = [int(row[3]) for row in chosen]
        for row in chosen:
            image_id = int(row[3])
            for model, spec in row[4].items():
                target_specs[(model, label, image_id)] = spec
    return selected_images, target_specs


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


def restore_records(feature_path: Path, specs: list[TargetSpec]) -> tuple[dict[tuple[int, int], dict], int]:
    desired = {(spec.image_id, spec.response_token_idx): spec for spec in specs}
    required = {
        MATRIX_KEYS["vv"]["attention"],
        MATRIX_KEYS["vv"]["gate"],
        MATRIX_KEYS["vv"]["source"],
        MATRIX_KEYS["vv"]["positions"],
    }
    found = {}
    scanned = 0
    for record in iter_pickle_records([feature_path]):
        scanned += 1
        key = (int(record.get("image_id", -1)), int(record.get("response_token_idx", -1)))
        if key not in desired or key in found or not required.issubset(record):
            continue
        if int(record.get("label", -1)) != desired[key].label:
            continue
        found[key] = freeze_record(record)
    missing = set(desired) - set(found)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} selected feature records in {feature_path}: {sorted(missing)}")
    return found, scanned


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    totals = values.sum(dim=1, keepdim=True)
    return torch.where(totals > 0.0, values / totals.clamp_min(1e-12), torch.zeros_like(values))


def signal_matrices(record: dict) -> dict[str, torch.Tensor]:
    keys = MATRIX_KEYS["vv"]
    gate = torch.nan_to_num(
        torch.as_tensor(record[keys["gate"]]).float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return {
        "gate_softmax": torch.softmax(gate, dim=1),
        "attention": normalize_rows(torch.as_tensor(record[keys["attention"]])),
        "source": normalize_rows(torch.as_tensor(record[keys["source"]])),
    }


def retain_topk(
    values: torch.Tensor,
    top_k: int,
    *,
    softmax_within_topk: bool = False,
    softmax_temperature: float = 1.0,
) -> tuple[torch.Tensor, list[int], float]:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0).flatten()
    k = min(max(int(top_k), 1), int(values.numel()))
    selected, indices = torch.topk(values, k=k, largest=True, sorted=True)
    full_mass = float(values.sum().item())
    coverage = float(selected.sum().item()) / full_mass if full_mass > 0.0 else 0.0
    display_values = (
        torch.softmax(selected / softmax_temperature, dim=0)
        if softmax_within_topk
        else selected
    )
    retained = torch.zeros_like(values)
    retained.index_copy_(0, indices, display_values)
    return retained, [int(value) for value in indices.tolist()], coverage


def plot_record(
    *,
    model: str,
    spec: TargetSpec,
    record: dict,
    image_path: Path,
    layers: list[int],
    top_k: int,
    softmax_within_topk: bool,
    topk_softmax_temperature: float,
    output_stem: Path,
    dpi: int,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
) -> tuple[dict, list[dict]]:
    original = Image.open(image_path)
    image, crop_offset = model_view(original, model)
    original.close()
    matrices = signal_matrices(record)
    token_count = int(next(iter(matrices.values())).shape[1])
    grid_height, grid_width = visual_grid(model, token_count, image.size)
    layer_indices = [layer - 1 for layer in layers]
    layer_count = int(next(iter(matrices.values())).shape[0])
    if min(layer_indices) < 0 or max(layer_indices) >= layer_count:
        raise ValueError(f"Layers {layers} outside 1..{layer_count} for {model}.")
    boxes = projected_boxes(
        image_id=spec.image_id,
        canonical=spec.canonical_word,
        category_ids=category_ids,
        annotations_by_image=annotations_by_image,
        crop_offset=crop_offset,
        view_size=image.size,
    )

    retained: dict[tuple[str, int], torch.Tensor] = {}
    indices: dict[tuple[str, int], list[int]] = {}
    coverages: dict[tuple[str, int], float] = {}
    row_scales = {}
    for signal, _title in SIGNALS:
        maxima = []
        for layer, layer_index in zip(layers, layer_indices):
            values, top_indices, coverage = retain_topk(
                matrices[signal][layer_index],
                top_k,
                softmax_within_topk=softmax_within_topk,
                softmax_temperature=topk_softmax_temperature,
            )
            retained[(signal, layer)] = values
            indices[(signal, layer)] = top_indices
            coverages[(signal, layer)] = coverage
            maxima.append(float(values.max().item()))
        row_scales[signal] = max(max(maxima), 1e-12)

    figure, axes = plt.subplots(3, len(layers), figsize=(3.9 * len(layers), 10.2), constrained_layout=True)
    if len(layers) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    stats = []
    for row, (signal, row_title) in enumerate(SIGNALS):
        for column, layer in enumerate(layers):
            patch = retained[(signal, layer)].reshape(grid_height, grid_width).cpu().numpy()
            overlay, _smooth = thermal_overlay(image, patch, row_scales[signal])
            axis = axes[row, column]
            axis.imshow(overlay, interpolation="nearest")
            for x, y, width, height in boxes:
                axis.add_patch(
                    Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=1.5)
                )
            if row == 0:
                axis.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
            if column == 0:
                ylabel = row_title + ("\nTop-K local softmax" if softmax_within_topk else "")
                axis.set_ylabel(ylabel, fontsize=10.2, fontweight="semibold", labelpad=8)
            axis.text(
                0.02,
                0.035,
                (
                    f"Pre-softmax Top-{top_k} mass: {coverages[(signal, layer)]:.3f}"
                    if softmax_within_topk
                    else f"Top-{top_k} mass: {coverages[(signal, layer)]:.3f}"
                ),
                transform=axis.transAxes,
                color="white",
                fontsize=8.2,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#303030")
            stats.append(
                {
                    "model": model,
                    "model_display": MODEL_NAMES[model],
                    "label": spec.label,
                    "label_name": LABEL_NAMES[spec.label],
                    "image_id": spec.image_id,
                    "response_token_idx": spec.response_token_idx,
                    "surface": spec.surface,
                    "canonical_word": spec.canonical_word,
                    "object_fraction": spec.object_fraction,
                    "size_basis": spec.size_basis,
                    "layer": layer,
                    "signal": signal,
                    "visual_tokens": token_count,
                    "top_k": top_k,
                    "topk_mass": coverages[(signal, layer)],
                    "softmax_within_topk": softmax_within_topk,
                    "topk_softmax_temperature": topk_softmax_temperature,
                    "display_topk_sum": float(retained[(signal, layer)].sum().item()),
                    "row_scale_max": row_scales[signal],
                    "topk_indices": indices[(signal, layer)],
                }
            )
        colorbar = figure.colorbar(
            ScalarMappable(norm=Normalize(0.0, row_scales[signal]), cmap="turbo"),
            ax=axes[row, :],
            location="right",
            shrink=0.88,
            pad=0.012,
        )
        colorbar.set_label(f"{row_title} value", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.5)

    title = (
        f"{MODEL_NAMES[model]} | Target: '{spec.surface}'"
        + (f" (COCO class: {spec.canonical_word})" if spec.surface.lower() != spec.canonical_word else "")
        + f" | {LABEL_NAMES[spec.label]} | Image {spec.image_id}"
    )
    subtitle = (
        f"Layers {','.join(map(str, layers))}; each signal has its own scale across layers; "
        f"small-object fraction={100.0 * spec.object_fraction:.2f}% ({spec.size_basis})."
    )
    if softmax_within_topk:
        subtitle += f" Top-{top_k} values use local softmax (T={topk_softmax_temperature:g})."
    figure.suptitle(f"{title}\n{subtitle}", fontsize=12.8, fontweight="bold", linespacing=1.3)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return (
        {
            "model": model,
            "model_display": MODEL_NAMES[model],
            "label": spec.label,
            "label_name": LABEL_NAMES[spec.label],
            "image_id": spec.image_id,
            "response_token_idx": spec.response_token_idx,
            "surface": spec.surface,
            "canonical_word": spec.canonical_word,
            "object_fraction": spec.object_fraction,
            "size_basis": spec.size_basis,
            "layers": layers,
            "top_k": top_k,
            "softmax_within_topk": softmax_within_topk,
            "topk_softmax_temperature": topk_softmax_temperature,
            "visual_grid": [grid_height, grid_width],
            "gt_box_count": len(boxes),
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        stats,
    )


def write_outputs(
    output_dir: Path,
    figure_rows: list[dict],
    stats: list[dict],
    shared_images: dict[int, list[int]],
    scanned: dict[str, int],
    softmax_within_topk: bool,
    topk_softmax_temperature: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_samples.json").write_text(
        json.dumps(figure_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "shared_image_ids.json").write_text(
        json.dumps(shared_images, indent=2), encoding="utf-8"
    )
    (output_dir / "topk_values.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_rows = []
    for row in stats:
        csv_rows.append({**row, "topk_indices": ",".join(map(str, row["topk_indices"]))})
    with (output_dir / "topk_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    by_model_image = {(row["model"], int(row["image_id"])): row for row in figure_rows}
    display_description = (
        f"After Top-K selection, the retained values are softmaxed locally with temperature "
        f"{topk_softmax_temperature:g}; the annotated Top-K mass is measured before this visualization-only softmax."
        if softmax_within_topk
        else "Top-K values retain their original normalized mass without a second normalization."
    )
    lines = [
        "# COCO500 softmax-gate / attention / source Top-K GLSim maps",
        "",
        "The same five hallucinated and five non-hallucinated COCO500 images are used for all three models. Gate values are softmaxed over VV visual tokens before Top-K selection. Attention and source are row-normalized before their own Top-K selection. Each signal has an independent numeric color scale shared only across layers 5/15/20/25 within one figure.",
        "",
        display_description,
        "",
        "Small non-hallucinated targets are selected by the smallest matching GT-box/image area fraction. Hallucinated targets have no matching GT box, so selection uses the COCO category's median box/image area fraction. Repeated three-model target signatures are skipped when possible.",
        "",
        "## Shared images and targets",
        "",
        "| Image | Label | LLaVA target | Qwen target | InternVL target |",
        "|---:|---|---|---|---|",
    ]
    for label in (0, 1):
        for image_id in shared_images[label]:
            cells = []
            for model in MODELS:
                row = by_model_image[(model, image_id)]
                cells.append(f"{row['surface']} ({100.0 * row['object_fraction']:.2f}%)")
            lines.append(
                f"| {image_id} | {LABEL_NAMES[label]} | " + " | ".join(cells) + " |"
            )
    lines.extend(["", "## Figures", ""])
    for model in MODELS:
        lines.extend([f"### {MODEL_NAMES[model]}", "", f"Scanned feature rows: {scanned[model]}", ""])
        for row in [value for value in figure_rows if value["model"] == model]:
            relative = Path(row["png"]).relative_to(output_dir)
            lines.append(
                f"- Image {row['image_id']} ({row['label_name']}, `{row['surface']}`): "
                f"[PNG]({relative.as_posix()})"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if set(args.models) != set(MODELS):
        raise ValueError("Shared-image comparison requires all three models.")
    if args.top_k <= 0 or args.samples_per_label <= 0:
        raise ValueError("--top-k and --samples-per-label must be positive.")
    if args.topk_softmax_temperature <= 0.0:
        raise ValueError("--topk-softmax-temperature must be positive.")
    category_ids, annotations_by_image, image_sizes, category_medians = load_coco_metadata(args.annotations)
    labelings = {}
    candidates = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        labelings[model] = json.loads((experiment_dir / "labeling.json").read_text(encoding="utf-8"))
        candidates[model] = candidate_targets(
            model=model,
            labeling=labelings[model],
            max_object_fraction=args.max_object_fraction,
            min_visible_fraction=args.min_visible_fraction,
            category_ids=category_ids,
            annotations_by_image=annotations_by_image,
            image_sizes=image_sizes,
            category_medians=category_medians,
        )
    shared_images, target_specs = select_shared_targets(candidates, args.models, args.samples_per_label)
    print(f"Selected shared images: {shared_images}", flush=True)

    figure_rows = []
    all_stats = []
    scanned = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        specs = [
            target_specs[(model, label, image_id)]
            for label in (0, 1)
            for image_id in shared_images[label]
        ]
        feature_path = experiment_dir / "features.pkl"
        print(f"[{model}] Restoring 10 records from {feature_path}...", flush=True)
        records, scanned[model] = restore_records(feature_path, specs)
        label_ranks = {0: 0, 1: 0}
        for spec in specs:
            label_ranks[spec.label] += 1
            image_path = args.coco_images / f"COCO_val2014_{spec.image_id:012d}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            output_stem = args.output_dir / model / (
                f"{LABEL_NAMES[spec.label].lower().replace('-', '_')}_{label_ranks[spec.label]:02d}_"
                f"{safe_slug(spec.surface)}_image{spec.image_id}_topk{args.top_k}"
            )
            print(
                f"[{model}] Plotting image {spec.image_id}, {LABEL_NAMES[spec.label]}, target {spec.surface}...",
                flush=True,
            )
            figure_row, stats = plot_record(
                model=model,
                spec=spec,
                record=records[(spec.image_id, spec.response_token_idx)],
                image_path=image_path,
                layers=args.layers,
                top_k=args.top_k,
                softmax_within_topk=args.softmax_within_topk,
                topk_softmax_temperature=args.topk_softmax_temperature,
                output_stem=output_stem,
                dpi=args.dpi,
                category_ids=category_ids,
                annotations_by_image=annotations_by_image,
            )
            figure_rows.append(figure_row)
            all_stats.extend(stats)
        del records
        gc.collect()
    write_outputs(
        args.output_dir,
        figure_rows,
        all_stats,
        shared_images,
        scanned,
        args.softmax_within_topk,
        args.topk_softmax_temperature,
    )
    print(
        f"Wrote {len(figure_rows)} figures and {len(all_stats)} signal/layer rows to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
