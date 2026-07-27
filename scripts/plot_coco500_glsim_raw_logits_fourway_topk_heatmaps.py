#!/usr/bin/env python3
"""Plot COCO500 raw relative-VLL logits against gate, attention, and source Top-K maps."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image

from plot_attention_gate_source_heatmaps import LABEL_NAMES, MATRIX_KEYS, MODEL_NAMES, iter_pickle_records
from plot_coco500_glsim_softmax_gate_topk_heatmaps import (
    TargetSpec,
    candidate_targets,
    load_coco_metadata,
    normalize_rows,
    retain_topk,
    select_shared_targets,
    signal_matrices,
)
from plot_spatial_attention_gate_source_heatmaps import (
    MODELS,
    model_view,
    projected_boxes,
    resize_heatmap,
    safe_slug,
    thermal_overlay,
    visual_grid,
)


BASE_ROWS = (
    ("raw_logits", "Raw relative-VLL logits Top-K"),
    ("gate_softmax", "Softmax(gate) Top-K\nTop-K local softmax"),
    ("attention", "Attention Top-K\nTop-K local softmax"),
)
GATED_ATTENTION_ROW = (
    "gated_attention",
    "GateAttention Top-K\nnorm(attention × raw gate)\nTop-K local softmax",
)
SOURCE_ROW = ("source", "Source distribution Top-K\nTop-K local softmax")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", default="COCO500-mass-dist-topk")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 15, 20, 25])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--include-gated-attention",
        action="store_true",
        help="Add norm(support_attention * raw sigmoid semantic gate) as a comparison row.",
    )
    parser.add_argument(
        "--include-last-layer",
        action="store_true",
        help="Append each model's actual final decoder layer to --layers.",
    )
    parser.add_argument("--samples-per-label", type=int, default=5)
    parser.add_argument("--topk-softmax-temperature", type=float, default=1.0)
    parser.add_argument("--relative-vll-mad-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-object-fraction", type=float, default=0.02)
    parser.add_argument("--min-visible-fraction", type=float, default=0.001)
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
        default=Path(
            "outputs/coco500_glsim_raw_logits_gate_attention_source_topk32_localsoftmax_small_shared10"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def freeze_record(record: dict) -> dict:
    vv = MATRIX_KEYS["vv"]
    vp = MATRIX_KEYS["vp"]
    frozen = {
        "image_id": int(record["image_id"]),
        "label": int(record["label"]),
        "token_str": str(record.get("token_str", "")),
        "response_token_idx": int(record.get("response_token_idx", -1)),
        "dgst_t_layer_stats": [dict(value) for value in record["dgst_t_layer_stats"]],
        vv["positions"]: [int(value) for value in record[vv["positions"]]],
        vp["positions"]: [int(value) for value in record[vp["positions"]]],
    }
    for key in (vv["attention"], vv["gate"], vv["source"], vp["gate"]):
        frozen[key] = torch.as_tensor(record[key]).detach().cpu().clone()
    return frozen


def restore_records(feature_path: Path, specs: list[TargetSpec]) -> tuple[dict[tuple[int, int], dict], int]:
    desired = {(spec.image_id, spec.response_token_idx): spec for spec in specs}
    vv = MATRIX_KEYS["vv"]
    vp = MATRIX_KEYS["vp"]
    required = {
        vv["attention"],
        vv["gate"],
        vv["source"],
        vv["positions"],
        vp["gate"],
        vp["positions"],
        "dgst_t_layer_stats",
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
        raise RuntimeError(f"Missing {len(missing)} selected records in {feature_path}: {sorted(missing)}")
    return found, scanned


def reconstruct_visual_raw_logits(
    record: dict,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, list[dict]]:
    vv = MATRIX_KEYS["vv"]
    vp = MATRIX_KEYS["vp"]
    vp_gate = torch.nan_to_num(
        torch.as_tensor(record[vp["gate"]]).float(), nan=0.5, posinf=1.0, neginf=0.0
    )
    vp_positions = [int(value) for value in record[vp["positions"]]]
    vv_positions = [int(value) for value in record[vv["positions"]]]
    vp_index = {position: index for index, position in enumerate(vp_positions)}
    missing = [position for position in vv_positions if position not in vp_index]
    if missing:
        raise ValueError(f"VV positions missing from VP support: {missing[:10]}")
    visual_index = torch.tensor([vp_index[position] for position in vv_positions], dtype=torch.long)
    if vp_gate.shape[1] != len(vp_positions):
        raise ValueError("VP gate width does not match VP support positions.")

    lower_bound = torch.nextafter(
        torch.tensor(0.0, dtype=vp_gate.dtype),
        torch.tensor(1.0, dtype=vp_gate.dtype),
    )
    upper_bound = torch.nextafter(
        torch.tensor(1.0, dtype=vp_gate.dtype),
        torch.tensor(0.0, dtype=vp_gate.dtype),
    )
    clamped_gate = vp_gate.clamp(lower_bound, upper_bound)
    layer_stats = record["dgst_t_layer_stats"]
    if len(layer_stats) != int(vp_gate.shape[0]):
        raise ValueError("Layer stats count does not match VP gate layers.")
    visual_logits = []
    diagnostics = []
    for layer_index, stats in enumerate(layer_stats):
        median = float(stats["visual_prompt_relative_vll_logit_median"])
        mad = float(stats["visual_prompt_relative_vll_logit_mad"])
        raw_support = median + (mad + max(float(epsilon), 1e-12)) * torch.logit(clamped_gate[layer_index])
        reconstructed_median = float(raw_support.median().item())
        reconstructed_mad = float(torch.abs(raw_support - raw_support.median()).median().item())
        visual_logits.append(raw_support.index_select(0, visual_index))
        diagnostics.append(
            {
                "layer": layer_index + 1,
                "saved_median": median,
                "saved_mad": mad,
                "reconstructed_median": reconstructed_median,
                "reconstructed_mad": reconstructed_mad,
                "median_abs_error": abs(reconstructed_median - median),
                "mad_abs_error": abs(reconstructed_mad - mad),
                "gate_clipped_fraction": float(
                    ((vp_gate[layer_index] <= 0.0) | (vp_gate[layer_index] >= 1.0))
                    .float()
                    .mean()
                    .item()
                ),
            }
        )
    return torch.stack(visual_logits), diagnostics


def retain_raw_topk(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, list[int], float, float]:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0).flatten()
    k = min(max(int(top_k), 1), int(values.numel()))
    selected, indices = torch.topk(values, k=k, largest=True, sorted=True)
    retained = torch.zeros_like(values)
    retained.index_copy_(0, indices, selected)
    return retained, [int(value) for value in indices.tolist()], float(selected.min()), float(selected.max())


def raw_norm(values: list[torch.Tensor]) -> Normalize:
    selected = torch.cat([value[value != 0.0] for value in values])
    if selected.numel() == 0:
        return Normalize(-1.0, 1.0)
    minimum = float(selected.min().item())
    maximum = float(selected.max().item())
    if math.isclose(minimum, maximum, rel_tol=1e-9, abs_tol=1e-12):
        delta = max(abs(minimum) * 0.05, 1e-3)
        return Normalize(minimum - delta, maximum + delta)
    if minimum < 0.0 < maximum:
        return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return Normalize(vmin=minimum, vmax=maximum)


def numeric_masked_overlay(
    image: Image.Image,
    values: np.ndarray,
    mask: np.ndarray,
    norm: Normalize,
) -> np.ndarray:
    normalized = np.asarray(norm(values), dtype=np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    smooth_values = resize_heatmap(normalized, image.size)
    smooth_mask = resize_heatmap(mask.astype(np.float32), image.size)
    base = np.asarray(image, dtype=np.float32) / 255.0
    color = plt.get_cmap("coolwarm")(smooth_values)[..., :3]
    alpha = (0.82 * np.power(smooth_mask, 0.68))[..., None]
    return np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 1.0)


def plot_record(
    *,
    model: str,
    spec: TargetSpec,
    record: dict,
    image_path: Path,
    layers: list[int],
    top_k: int,
    topk_softmax_temperature: float,
    relative_vll_mad_epsilon: float,
    include_gated_attention: bool,
    include_last_layer: bool,
    output_stem: Path,
    dpi: int,
    category_ids: dict[str, int],
    annotations_by_image: dict[int, list[dict]],
) -> tuple[dict, list[dict], list[dict]]:
    original = Image.open(image_path)
    image, crop_offset = model_view(original, model)
    original.close()
    raw_logits, reconstruction = reconstruct_visual_raw_logits(
        record, epsilon=relative_vll_mad_epsilon
    )
    if include_last_layer:
        layers = list(dict.fromkeys([*layers, int(raw_logits.shape[0])]))
    probability_signals = signal_matrices(record)
    gated_attention_retained = None
    if include_gated_attention:
        vv = MATRIX_KEYS["vv"]
        raw_gate = torch.nan_to_num(
            torch.as_tensor(record[vv["gate"]]).float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        raw_attention = probability_signals["attention"]
        gated_unnormalized = raw_attention * raw_gate
        gated_attention_retained = gated_unnormalized.sum(dim=1)
        probability_signals["gated_attention"] = normalize_rows(gated_unnormalized)
    token_count = int(raw_logits.shape[1])
    grid_height, grid_width = visual_grid(model, token_count, image.size)
    layer_indices = [layer - 1 for layer in layers]
    if min(layer_indices) < 0 or max(layer_indices) >= raw_logits.shape[0]:
        raise ValueError(f"Layers {layers} outside 1..{raw_logits.shape[0]} for {model}.")
    boxes = projected_boxes(
        image_id=spec.image_id,
        canonical=spec.canonical_word,
        category_ids=category_ids,
        annotations_by_image=annotations_by_image,
        crop_offset=crop_offset,
        view_size=image.size,
    )

    retained = {}
    top_indices = {}
    annotations = {}
    raw_maps = []
    for layer, layer_index in zip(layers, layer_indices):
        values, indices, minimum, maximum = retain_raw_topk(raw_logits[layer_index], top_k)
        retained[("raw_logits", layer)] = values
        top_indices[("raw_logits", layer)] = indices
        annotations[("raw_logits", layer)] = f"Top-{top_k} range: [{minimum:.2f}, {maximum:.2f}]"
        raw_maps.append(values)
        probability_signal_names = ["gate_softmax", "attention"]
        if include_gated_attention:
            probability_signal_names.append("gated_attention")
        probability_signal_names.append("source")
        for signal in probability_signal_names:
            values, indices, coverage = retain_topk(
                probability_signals[signal][layer_index],
                top_k,
                softmax_within_topk=True,
                softmax_temperature=topk_softmax_temperature,
            )
            retained[(signal, layer)] = values
            top_indices[(signal, layer)] = indices
            note = f"Pre-softmax Top-{top_k} mass: {coverage:.3f}"
            if signal == "gated_attention" and gated_attention_retained is not None:
                note += f"\nA×g retained: {float(gated_attention_retained[layer_index]):.3f}"
            annotations[(signal, layer)] = note

    raw_color_norm = raw_norm(raw_maps)
    rows = list(BASE_ROWS)
    if include_gated_attention:
        rows.append(GATED_ATTENTION_ROW)
    rows.append(SOURCE_ROW)
    probability_signal_names = [signal for signal, _title in rows if signal != "raw_logits"]
    row_scales = {
        signal: max(
            max(float(retained[(signal, layer)].max().item()) for layer in layers),
            1e-12,
        )
        for signal in probability_signal_names
    }
    figure, axes = plt.subplots(
        len(rows),
        len(layers),
        figsize=(3.9 * len(layers), 3.25 * len(rows) + 0.3),
        constrained_layout=True,
    )
    if len(layers) == 1:
        axes = np.asarray(axes).reshape(len(rows), 1)
    stats_rows = []
    for row, (signal, row_title) in enumerate(rows):
        for column, layer in enumerate(layers):
            vector = retained[(signal, layer)]
            patch = vector.reshape(grid_height, grid_width).cpu().numpy()
            if signal == "raw_logits":
                overlay = numeric_masked_overlay(
                    image,
                    patch,
                    (vector != 0.0).reshape(grid_height, grid_width).cpu().numpy(),
                    raw_color_norm,
                )
            else:
                overlay, _smooth = thermal_overlay(image, patch, row_scales[signal])
            axis = axes[row, column]
            axis.imshow(overlay, interpolation="nearest")
            for x, y, width, height in boxes:
                axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#ff2020", linewidth=1.5))
            if row == 0:
                axis.set_title(f"Layer {layer}", fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(row_title, fontsize=9.8, fontweight="semibold", labelpad=8)
            axis.text(
                0.02,
                0.035,
                annotations[(signal, layer)],
                transform=axis.transAxes,
                color="white",
                fontsize=7.8,
                bbox={"boxstyle": "square,pad=0.2", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#303030")
            stats_rows.append(
                {
                    "model": model,
                    "model_display": MODEL_NAMES[model],
                    "label": spec.label,
                    "label_name": LABEL_NAMES[spec.label],
                    "image_id": spec.image_id,
                    "response_token_idx": spec.response_token_idx,
                    "surface": spec.surface,
                    "canonical_word": spec.canonical_word,
                    "layer": layer,
                    "signal": signal,
                    "visual_tokens": token_count,
                    "top_k": top_k,
                    "topk_indices": top_indices[(signal, layer)],
                    "display_min": float(vector[vector != 0.0].min().item()),
                    "display_max": float(vector.max().item()),
                    "raw_gate_topk_overlap": len(
                        set(top_indices[("raw_logits", layer)])
                        & set(top_indices[("gate_softmax", layer)])
                    ) / min(top_k, token_count),
                    "attention_gated_topk_overlap": (
                        len(
                            set(top_indices[("attention", layer)])
                            & set(top_indices[("gated_attention", layer)])
                        )
                        / min(top_k, token_count)
                        if include_gated_attention
                        else None
                    ),
                    "gate_gated_topk_overlap": (
                        len(
                            set(top_indices[("gate_softmax", layer)])
                            & set(top_indices[("gated_attention", layer)])
                        )
                        / min(top_k, token_count)
                        if include_gated_attention
                        else None
                    ),
                    "gated_attention_retained_mass": (
                        float(gated_attention_retained[layer - 1].item())
                        if gated_attention_retained is not None
                        else None
                    ),
                }
            )
        if signal == "raw_logits":
            colorbar = figure.colorbar(
                ScalarMappable(norm=raw_color_norm, cmap="coolwarm"),
                ax=axes[row, :],
                location="right",
                shrink=0.88,
                pad=0.012,
            )
            colorbar.set_label("Raw relative-VLL logit", fontsize=8.5)
        else:
            colorbar = figure.colorbar(
                ScalarMappable(norm=Normalize(0.0, row_scales[signal]), cmap="turbo"),
                ax=axes[row, :],
                location="right",
                shrink=0.88,
                pad=0.012,
            )
            colorbar.set_label(f"{signal} local-softmax value", fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.5)

    title = (
        f"{MODEL_NAMES[model]} | Target: '{spec.surface}'"
        + (f" (COCO class: {spec.canonical_word})" if spec.surface.lower() != spec.canonical_word else "")
        + f" | {LABEL_NAMES[spec.label]} | Image {spec.image_id}"
    )
    figure.suptitle(
        f"{title}\nRaw logits vs gate / attention"
        + (" / GateAttention" if include_gated_attention else "")
        + f" / source Top-{top_k}; layers {','.join(map(str, layers))}; each row has its own scale.",
        fontsize=12.5,
        fontweight="bold",
        linespacing=1.3,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    figure_row = {
        "model": model,
        "model_display": MODEL_NAMES[model],
        "label": spec.label,
        "label_name": LABEL_NAMES[spec.label],
        "image_id": spec.image_id,
        "response_token_idx": spec.response_token_idx,
        "surface": spec.surface,
        "canonical_word": spec.canonical_word,
        "object_fraction": spec.object_fraction,
        "layers": layers,
        "top_k": top_k,
        "visual_grid": [grid_height, grid_width],
        "gt_box_count": len(boxes),
        "include_gated_attention": include_gated_attention,
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    reconstruction_rows = [
        {
            "model": model,
            "label": spec.label,
            "image_id": spec.image_id,
            "response_token_idx": spec.response_token_idx,
            "surface": spec.surface,
            **row,
        }
        for row in reconstruction
    ]
    return figure_row, stats_rows, reconstruction_rows


def write_outputs(
    output_dir: Path,
    figure_rows: list[dict],
    stats_rows: list[dict],
    reconstruction_rows: list[dict],
    shared_images: dict[int, list[int]],
    scanned: dict[str, int],
    include_gated_attention: bool,
    include_last_layer: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_samples.json").write_text(json.dumps(figure_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "shared_image_ids.json").write_text(json.dumps(shared_images, indent=2), encoding="utf-8")
    (output_dir / "topk_values.json").write_text(json.dumps(stats_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "raw_logit_reconstruction.json").write_text(
        json.dumps(reconstruction_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, rows in (
        ("topk_values.csv", stats_rows),
        ("raw_logit_reconstruction.csv", reconstruction_rows),
    ):
        csv_rows = []
        for row in rows:
            csv_rows.append(
                {
                    **row,
                    **(
                        {"topk_indices": ",".join(map(str, row["topk_indices"]))}
                        if "topk_indices" in row
                        else {}
                    ),
                }
            )
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    by_model_image = {(row["model"], row["image_id"]): row for row in figure_rows}
    layer_summary = "; ".join(
        f"{MODEL_NAMES[model]}: "
        + ",".join(f"L{layer}" for layer in by_model_image[(model, shared_images[0][0])]["layers"])
        for model in MODELS
    )
    lines = [
        "# COCO500 raw-logit / gate / attention"
        + (" / GateAttention" if include_gated_attention else "")
        + " / source Top-32 GLSim maps",
        "",
        "Raw visual-token relative-VLL logits are reconstructed from the saved VP semantic gate and per-layer `visual_prompt_relative_vll_logit_median/MAD`: `raw = median + (MAD + 1e-6) * logit(gate)`. The VP support positions are then sliced to the VV visual positions.",
        "",
        "Raw logits retain signed numeric values and use a `coolwarm` scale. Gate, attention, "
        + ("GateAttention, " if include_gated_attention else "")
        + "and source each select their own Top-32 and use visualization-only local softmax (T=1). Every row has an independent scale shared across the displayed layers.",
        "",
        "Displayed layers"
        + (" (including each model's final decoder layer)" if include_last_layer else "")
        + ": "
        + layer_summary
        + ".",
        "",
        *(
            [
                "GateAttention is the original DGST-T gated attention: `norm(support_attention * sigmoid semantic gate)`. It does not multiply attention by `softmax(gate)`. The figures also report `A×g retained = sum(A*g)/sum(A)` before normalization.",
                "",
            ]
            if include_gated_attention
            else []
        ),
        "Because the VV gate is a monotonic per-layer transform of the same raw logits, their Top-32 spatial indices should coincide; `topk_values` records the overlap.",
        "",
        "## Shared images and targets",
        "",
        "| Image | Label | LLaVA | Qwen | InternVL |",
        "|---:|---|---|---|---|",
    ]
    for label in (0, 1):
        for image_id in shared_images[label]:
            targets = [by_model_image[(model, image_id)]["surface"] for model in MODELS]
            lines.append(f"| {image_id} | {LABEL_NAMES[label]} | " + " | ".join(targets) + " |")
    lines.extend(["", "## Figures", ""])
    for model in MODELS:
        lines.extend([f"### {MODEL_NAMES[model]}", "", f"Scanned feature rows: {scanned[model]}", ""])
        for row in [value for value in figure_rows if value["model"] == model]:
            relative = Path(row["png"]).relative_to(output_dir)
            lines.append(
                f"- Image {row['image_id']} ({row['label_name']}, `{row['surface']}`): [PNG]({relative.as_posix()})"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if set(args.models) != set(MODELS):
        raise ValueError("Shared-image comparison requires all three models.")
    if args.top_k <= 0 or args.samples_per_label <= 0:
        raise ValueError("--top-k and --samples-per-label must be positive.")
    if args.topk_softmax_temperature <= 0.0 or args.relative_vll_mad_epsilon <= 0.0:
        raise ValueError("Softmax temperature and MAD epsilon must be positive.")
    category_ids, annotations_by_image, image_sizes, category_medians = load_coco_metadata(args.annotations)
    candidates = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        labeling = json.loads((experiment_dir / "labeling.json").read_text(encoding="utf-8"))
        candidates[model] = candidate_targets(
            model=model,
            labeling=labeling,
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
    stats_rows = []
    reconstruction_rows = []
    scanned = {}
    for model in args.models:
        specs = [
            target_specs[(model, label, image_id)]
            for label in (0, 1)
            for image_id in shared_images[label]
        ]
        feature_path = args.outputs_root / model / args.experiment / "features.pkl"
        print(f"[{model}] Restoring 10 records from {feature_path}...", flush=True)
        records, scanned[model] = restore_records(feature_path, specs)
        ranks = {0: 0, 1: 0}
        for spec in specs:
            ranks[spec.label] += 1
            image_path = args.coco_images / f"COCO_val2014_{spec.image_id:012d}.jpg"
            output_stem = args.output_dir / model / (
                f"{LABEL_NAMES[spec.label].lower().replace('-', '_')}_{ranks[spec.label]:02d}_"
                f"{safe_slug(spec.surface)}_image{spec.image_id}_topk{args.top_k}"
            )
            print(f"[{model}] Plotting image {spec.image_id}, target {spec.surface}...", flush=True)
            figure_row, sample_stats, sample_reconstruction = plot_record(
                model=model,
                spec=spec,
                record=records[(spec.image_id, spec.response_token_idx)],
                image_path=image_path,
                layers=args.layers,
                top_k=args.top_k,
                topk_softmax_temperature=args.topk_softmax_temperature,
                relative_vll_mad_epsilon=args.relative_vll_mad_epsilon,
                include_gated_attention=args.include_gated_attention,
                include_last_layer=args.include_last_layer,
                output_stem=output_stem,
                dpi=args.dpi,
                category_ids=category_ids,
                annotations_by_image=annotations_by_image,
            )
            figure_rows.append(figure_row)
            stats_rows.extend(sample_stats)
            reconstruction_rows.extend(sample_reconstruction)
        del records
        gc.collect()
    write_outputs(
        args.output_dir,
        figure_rows,
        stats_rows,
        reconstruction_rows,
        shared_images,
        scanned,
        args.include_gated_attention,
        args.include_last_layer,
    )
    print(
        f"Wrote {len(figure_rows)} figures, {len(stats_rows)} Top-K rows, and "
        f"{len(reconstruction_rows)} reconstruction rows to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
