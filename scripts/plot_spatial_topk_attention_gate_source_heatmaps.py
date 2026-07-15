#!/usr/bin/env python3
"""Plot shared-image Top-K VV attention, gated attention, and source overlays."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import defaultdict
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

from plot_attention_gate_source_heatmaps import LABEL_NAMES, MATRIX_KEYS, MODEL_NAMES, restore_samples
from plot_spatial_attention_gate_source_heatmaps import (
    MODELS,
    load_annotations,
    matrices_for_plot,
    model_view,
    projected_boxes,
    safe_slug,
    target_info,
    thermal_overlay,
    visual_grid,
)


ROW_NAMES = ("Raw attention Top-32", "Gate-weighted Top-32", "Source Top-32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", default="COCO4000-all")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 15, 20, 25])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path("outputs/coco4000_all_glsim_style_spatial_maps_shared10/selected_samples.json"),
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
        default=Path("outputs/coco4000_all_glsim_style_spatial_topk32_shared10"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def sanitize(values: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def topk_l1(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    values = sanitize(values).flatten()
    k = min(max(int(top_k), 1), int(values.numel()))
    indices = torch.topk(values, k=k, largest=True, sorted=False).indices
    selected = values.index_select(0, indices)
    kept = torch.zeros_like(values)
    kept.index_copy_(0, indices, selected)
    full_mass = float(values.sum().item())
    topk_mass = float(selected.sum().item())
    coverage = topk_mass / full_mass if full_mass > 0.0 else 0.0
    if topk_mass > 0.0:
        kept = kept / topk_mass
    return kept, indices, coverage


def gate_statistics(
    *,
    model: str,
    record: dict,
    surface: str,
    canonical: str,
    layers: list[int],
    top_k: int,
) -> list[dict]:
    attention, gated, source = matrices_for_plot(record)
    gate = torch.nan_to_num(
        torch.as_tensor(record[MATRIX_KEYS["vv"]["gate"]]).float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    rows = []
    for layer in layers:
        index = layer - 1
        raw_row = sanitize(attention[index])
        gated_row = sanitize(gated[index])
        source_row = sanitize(source[index])
        gate_row = gate[index]
        _raw_map, raw_indices, raw_coverage = topk_l1(raw_row, top_k)
        _gated_map, gated_indices, gated_coverage = topk_l1(gated_row, top_k)
        _source_map, _source_indices, source_coverage = topk_l1(source_row, top_k)
        raw_set = set(int(value) for value in raw_indices.tolist())
        gated_set = set(int(value) for value in gated_indices.tolist())
        overlap = len(raw_set & gated_set)
        raw_total = float(raw_row.sum().item())
        gated_total = float(gated_row.sum().item())
        rows.append(
            {
                "model": model,
                "model_display": MODEL_NAMES[model],
                "label": int(record["label"]),
                "label_name": LABEL_NAMES[int(record["label"])],
                "image_id": int(record["image_id"]),
                "response_token_idx": int(record["response_token_idx"]),
                "surface": surface,
                "canonical_word": canonical,
                "layer": layer,
                "visual_tokens": int(gate_row.numel()),
                "gate_mean": float(gate_row.mean().item()),
                "gate_median": float(gate_row.median().item()),
                "gate_std": float(gate_row.std(unbiased=False).item()),
                "gate_min": float(gate_row.min().item()),
                "gate_max": float(gate_row.max().item()),
                "gate_above_0_5_fraction": float((gate_row >= 0.5).float().mean().item()),
                "attention_weighted_gate": gated_total / raw_total if raw_total > 0.0 else 0.0,
                "gate_mean_on_raw_topk": float(gate_row.index_select(0, raw_indices).mean().item()),
                "gate_mean_on_gated_topk": float(gate_row.index_select(0, gated_indices).mean().item()),
                "raw_topk_coverage": raw_coverage,
                "gated_topk_coverage": gated_coverage,
                "source_topk_coverage": source_coverage,
                "raw_gated_topk_overlap": overlap,
                "raw_gated_topk_overlap_fraction": overlap / min(top_k, int(gate_row.numel())),
            }
        )
    return rows


def plot_record(
    *,
    model: str,
    candidate,
    labeling: dict[str, dict],
    image_path: Path,
    layers: list[int],
    top_k: int,
    output_stem: Path,
    dpi: int,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
) -> tuple[dict, list[dict]]:
    record = candidate.record
    original = Image.open(image_path)
    image, crop_offset = model_view(original, model)
    original.close()
    attention, gated, source = matrices_for_plot(record)
    matrices = (attention, gated, source)
    token_count = int(attention.shape[1])
    grid_height, grid_width = visual_grid(model, token_count, image.size)
    surface, canonical = target_info(record, labeling)
    boxes = projected_boxes(
        image_id=int(record["image_id"]),
        canonical=canonical,
        category_ids=category_ids,
        annotations_by_image=annotations_by_image,
        crop_offset=crop_offset,
        view_size=image.size,
    )
    stats = gate_statistics(
        model=model,
        record=record,
        surface=surface,
        canonical=canonical,
        layers=layers,
        top_k=top_k,
    )
    stats_by_layer = {int(row["layer"]): row for row in stats}

    figure, axes = plt.subplots(3, len(layers), figsize=(3.7 * len(layers), 10.0), constrained_layout=True)
    if len(layers) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for column, layer in enumerate(layers):
        layer_index = layer - 1
        if layer_index < 0 or layer_index >= attention.shape[0]:
            raise ValueError(f"Layer {layer} outside 1..{attention.shape[0]} for {model}.")
        layer_stats = stats_by_layer[layer]
        coverages = (
            layer_stats["raw_topk_coverage"],
            layer_stats["gated_topk_coverage"],
            layer_stats["source_topk_coverage"],
        )
        for row, matrix in enumerate(matrices):
            topk_map, _indices, _coverage = topk_l1(matrix[layer_index], top_k)
            patch = topk_map.reshape(grid_height, grid_width).cpu().numpy()
            overlay, _smooth = thermal_overlay(image, patch, float(patch.max(initial=0.0)))
            axis = axes[row, column]
            axis.imshow(overlay, interpolation="nearest")
            for x, y, width, height in boxes:
                axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=1.5))
            if row == 0:
                axis.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(ROW_NAMES[row], fontsize=11, fontweight="semibold", labelpad=8)
            note = f"Top-{top_k} mass: {coverages[row]:.3f}"
            if row == 1:
                note += f"\nretained: {layer_stats['attention_weighted_gate']:.3f}"
            axis.text(
                0.02,
                0.035,
                note,
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

    label = int(record["label"])
    title = (
        f"{MODEL_NAMES[model]} | Target: '{surface}'"
        + (f" (COCO class: {canonical})" if surface.lower() != canonical.lower() else "")
        + f" | {LABEL_NAMES[label]} | Image {int(record['image_id'])}"
    )
    figure.suptitle(
        f"{title}\nEach row selects its own Top-{top_k} visual patches and L1-normalizes within Top-{top_k}.",
        fontsize=13.2,
        fontweight="bold",
        linespacing=1.35,
    )
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo"),
        ax=axes,
        location="right",
        shrink=0.84,
        pad=0.012,
    )
    colorbar.set_label(f"Relative intensity within Top-{top_k}", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
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
            "label": label,
            "label_name": LABEL_NAMES[label],
            "image_id": int(record["image_id"]),
            "response_token_idx": int(record["response_token_idx"]),
            "surface": surface,
            "canonical_word": canonical,
            "layers": layers,
            "top_k": top_k,
            "visual_grid": [grid_height, grid_width],
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        stats,
    )


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def write_outputs(output_dir: Path, figure_rows: list[dict], gate_rows: list[dict], scanned: dict[str, int]) -> None:
    (output_dir / "selected_samples.json").write_text(
        json.dumps(figure_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "gate_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0]))
        writer.writeheader()
        writer.writerows(gate_rows)
    with (output_dir / "gate_values.json").open("w", encoding="utf-8") as handle:
        json.dump(gate_rows, handle, indent=2, ensure_ascii=False)

    lines = [
        "# Top-32 spatial maps and gate analysis",
        "",
        "Raw attention, gate-weighted attention (`A * gate`), and source distribution each select their own Top-32 visual patches. Each retained Top-32 vector is L1-normalized before visualization.",
        "",
        "`attention_weighted_gate = sum(A * gate) / sum(A)` is the absolute retained attention mass shown in the previous full-region figures. Values below 1 therefore make the previous gate overlays lighter.",
        "",
        "## Aggregate gate values",
        "",
        "| Model | Label | Layer | Gate mean | Attention-weighted gate | Gate on raw Top-32 | Raw/Gated Top-32 overlap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for label in (0, 1):
            for layer in sorted({int(row["layer"]) for row in gate_rows}):
                subset = [
                    row
                    for row in gate_rows
                    if row["model"] == model and int(row["label"]) == label and int(row["layer"]) == layer
                ]
                lines.append(
                    f"| {MODEL_NAMES[model]} | {LABEL_NAMES[label]} | {layer} | "
                    f"{mean([row['gate_mean'] for row in subset]):.3f} | "
                    f"{mean([row['attention_weighted_gate'] for row in subset]):.3f} | "
                    f"{mean([row['gate_mean_on_raw_topk'] for row in subset]):.3f} | "
                    f"{mean([row['raw_gated_topk_overlap_fraction'] for row in subset]):.3f} |"
                )
    lines.extend(
        [
            "",
            "## Per-sample retained mass",
            "",
            "| Model | Label | Image | Target | L5 | L15 | L20 | L25 | Figure |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    gate_by_key = defaultdict(dict)
    for row in gate_rows:
        gate_by_key[(row["model"], int(row["image_id"]))][int(row["layer"])] = row
    for figure_row in figure_rows:
        key = (figure_row["model"], int(figure_row["image_id"]))
        layer_rows = gate_by_key[key]
        relative = Path(figure_row["png"]).relative_to(output_dir)
        lines.append(
            f"| {figure_row['model_display']} | {figure_row['label_name']} | {figure_row['image_id']} | "
            f"{figure_row['surface']} | "
            + " | ".join(f"{layer_rows[layer]['attention_weighted_gate']:.3f}" for layer in (5, 15, 20, 25))
            + f" | [PNG]({relative.as_posix()}) |"
        )
    lines.extend(["", "## Scan counts", ""])
    for model in MODELS:
        lines.append(f"- {MODEL_NAMES[model]}: {scanned[model]} feature records")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if set(args.models) != set(MODELS):
        raise ValueError("Shared-image comparison requires all three models.")
    manifest_rows = json.loads(args.selection_json.read_text(encoding="utf-8"))
    category_ids, annotations_by_image = load_annotations(args.annotations)
    figure_rows = []
    gate_rows = []
    scanned = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        part_paths = sorted(experiment_dir.glob("features.part*.pkl"))
        print(f"[{model}] Restoring 10 selected records from {len(part_paths)} feature parts...", flush=True)
        candidates, scanned[model] = restore_samples(part_paths, model=model, manifest_rows=manifest_rows)
        labeling = json.loads((experiment_dir / "labeling.json").read_text(encoding="utf-8"))
        label_ranks = {0: 0, 1: 0}
        for candidate in candidates:
            record = candidate.record
            label = int(record["label"])
            label_ranks[label] += 1
            surface, _canonical = target_info(record, labeling)
            image_id = int(record["image_id"])
            image_path = args.coco_images / f"COCO_val2014_{image_id:012d}.jpg"
            stem = args.output_dir / model / (
                f"{LABEL_NAMES[label].lower().replace('-', '_')}_{label_ranks[label]:02d}_"
                f"{safe_slug(surface)}_image{image_id}_topk{args.top_k}"
            )
            print(f"[{model}] Plotting Top-{args.top_k} image {image_id}, target {surface}...", flush=True)
            figure_row, sample_gate_rows = plot_record(
                model=model,
                candidate=candidate,
                labeling=labeling,
                image_path=image_path,
                layers=args.layers,
                top_k=args.top_k,
                output_stem=stem,
                dpi=args.dpi,
                category_ids=category_ids,
                annotations_by_image=annotations_by_image,
            )
            figure_rows.append(figure_row)
            gate_rows.extend(sample_gate_rows)
        del candidates, labeling
        gc.collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, figure_rows, gate_rows, scanned)
    print(f"Wrote {len(figure_rows)} Top-{args.top_k} figures and {len(gate_rows)} gate rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
