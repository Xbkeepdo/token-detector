#!/usr/bin/env python3
"""Select and render publication-style VV/VP Top-32 COCO heatmap cases.

The selected Gaussian target distribution is reconstructed exactly as in DGST-T:

    target_dist = renormalize(support_attention * gaussian_gate)

VV contains the 576 visual patches. VP contains prompt and visual support; only
its decoder positions that correspond to the 24 x 24 visual span are projected
back onto the image. Prompt mass is retained in the quantitative report but is
never painted as an image patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


LAYERS_1BASED = (18, 21, 24, 27, 30)
LAYERS_0BASED = tuple(layer - 1 for layer in LAYERS_1BASED)
GRID = (24, 24)
VISUAL_COUNT = GRID[0] * GRID[1]
TARGET_METHOD_TITLES = {
    "hpre_raw_logit_gauss": "Raw-logit Gaussian target",
    "hpre_softmax_prob_gauss": "Softmax-prob Gaussian target",
}
SMALL_OBJECT_NAMES = {
    "airplane", "apple", "baseball bat", "baseball glove", "bird", "book",
    "bottle", "bowl", "cell phone", "clock", "cup", "donut", "fork",
    "frisbee", "hair drier", "handbag", "kite", "knife", "mouse", "orange",
    "remote", "scissors", "skateboard", "spoon", "sports ball", "tie",
    "toaster", "toothbrush", "traffic light", "umbrella", "vase", "wine glass",
}

# Quantitatively shortlisted, then manually checked in both VV and VP.  These
# examples are intentionally fixed so the paper figures are reproducible.
CURATED_HALL_KEYS = ((164475, 17), (546226, 45), (48917, 80), (105552, 105))
CURATED_REAL_KEYS = ((462289, 5), (272440, 27), (497801, 26))
CURATED_PAIR_KEYS = ((122602, 29, 103),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/llava_1_5_7b/COCO4000-512-ENDAC-SOFT"),
    )
    parser.add_argument(
        "--coco-image-dir",
        type=Path,
        default=Path("/root/rivermind-data/dataset/coco/val2014"),
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=Path("/root/rivermind-data/dataset/coco/annotations/instances_val2014.json"),
    )
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--vp-visual-start", type=int, default=4)
    parser.add_argument(
        "--target-method",
        choices=tuple(TARGET_METHOD_TITLES),
        default="hpre_raw_logit_gauss",
        help="Choose which already-extracted hpre Gaussian gate builds target_dist.",
    )
    parser.add_argument("--hall-count", type=int, default=4)
    parser.add_argument("--real-count", type=int, default=5)
    parser.add_argument("--paired-count", type=int, default=4)
    parser.add_argument("--shortlist-count", type=int, default=30)
    parser.add_argument(
        "--selection-mode", choices=("auto", "curated"), default="auto",
        help="Use automatic scoring or the manually verified paper manifest.",
    )
    parser.add_argument("--dpi", type=int, default=190)
    parser.add_argument(
        "--show-boxes", action="store_true",
        help="Draw COCO boxes for diagnostics; final paper figures omit them.",
    )
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=-1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def normalize_vector(values: np.ndarray) -> np.ndarray:
    return normalize_rows(np.asarray(values).reshape(1, -1))[0]


def signal_specs(target_method: str) -> tuple[tuple[str, str], ...]:
    return (
        ("support", "Support attention"),
        ("target", TARGET_METHOD_TITLES[target_method]),
        ("source", "Source distribution"),
    )


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower() or "token"


def find_image(image_dir: Path, image_id: int) -> Path:
    path = image_dir / f"COCO_val2014_{int(image_id):012d}.jpg"
    if not path.exists():
        matches = sorted(image_dir.glob(f"COCO_val2014_{int(image_id):012d}.*"))
        if not matches:
            raise FileNotFoundError(f"COCO image {image_id} not found in {image_dir}")
        path = matches[0]
    return path


def center_square_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)), (left, top)


def load_coco_annotations(path: Path) -> tuple[dict[str, int], dict[int, list[dict[str, Any]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    category_ids = {str(row["name"]).lower(): int(row["id"]) for row in data["categories"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in data["annotations"]:
        annotations[int(row["image_id"])].append(row)
    return category_ids, annotations


def projected_boxes(
    image_id: int,
    canonical: str,
    category_ids: dict[str, int],
    annotations: dict[int, list[dict[str, Any]]],
    crop_offset: tuple[int, int],
    view_size: tuple[int, int],
) -> list[tuple[float, float, float, float]]:
    category_id = category_ids.get(str(canonical).lower())
    if category_id is None:
        return []
    left, top = crop_offset
    view_width, view_height = view_size
    result = []
    for row in annotations.get(int(image_id), []):
        if int(row["category_id"]) != category_id:
            continue
        x, y, width, height = [float(value) for value in row["bbox"]]
        x1 = max(0.0, x - left)
        y1 = max(0.0, y - top)
        x2 = min(float(view_width), x + width - left)
        y2 = min(float(view_height), y + height - top)
        if x2 > x1 and y2 > y1:
            result.append((x1, y1, x2 - x1, y2 - y1))
    return result


def patch_box_coverage(
    boxes: list[tuple[float, float, float, float]],
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    patch_w = width / GRID[1]
    patch_h = height / GRID[0]
    patch_area = patch_w * patch_h
    coverage = np.zeros(VISUAL_COUNT, dtype=np.float64)
    for row in range(GRID[0]):
        py1, py2 = row * patch_h, (row + 1) * patch_h
        for col in range(GRID[1]):
            px1, px2 = col * patch_w, (col + 1) * patch_w
            covered = 0.0
            for x, y, box_w, box_h in boxes:
                ix = max(0.0, min(px2, x + box_w) - max(px1, x))
                iy = max(0.0, min(py2, y + box_h) - max(py1, y))
                covered += ix * iy
            coverage[row * GRID[1] + col] = min(1.0, covered / max(patch_area, 1e-12))
    return coverage


def span_lookup(labeling: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for image_key, item in labeling.items():
        image_id = int(image_key)
        spans = item.get("all_object_token_spans") or item.get("object_token_spans") or []
        for span in spans:
            for token_idx in span.get("token_indices") or []:
                result.setdefault((image_id, int(token_idx)), span)
    return result


def caption_snippet(text: str, span: dict[str, Any], flank: int = 86) -> str:
    start = max(0, int(span.get("char_start", 0)))
    end = max(start, int(span.get("char_end", start)))
    left, right = max(0, start - flank), min(len(text), end + flank)
    before = re.sub(r"\s+", " ", text[left:start]).strip()
    token = re.sub(r"\s+", " ", text[start:end]).strip()
    after = re.sub(r"\s+", " ", text[end:right]).strip()
    return f"{'…' if left else ''}{before} [{token}] {after}{'…' if right < len(text) else ''}".strip()


def scope_distributions(
    record: dict[str, Any],
    scope: str,
    vp_visual_start: int,
    target_method: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    prefix = "" if scope == "vv" else "vp_"
    attention_key = f"dgst_t_{prefix}attention_support_per_layer"
    gate_key = f"dgst_t_{prefix}{target_method}_gate_per_layer"
    source_key = f"dgst_t_{prefix}source_dist_per_layer"
    attention_full = normalize_rows(np.asarray(record[attention_key])[list(LAYERS_0BASED)])
    gate_full = np.asarray(record[gate_key], dtype=np.float64)[list(LAYERS_0BASED)]
    gate_full = np.clip(np.nan_to_num(gate_full, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    target_full = normalize_rows(attention_full * gate_full)
    source_full = normalize_rows(np.asarray(record[source_key])[list(LAYERS_0BASED)])
    support_size = attention_full.shape[1]
    if scope == "vv":
        visual_slice = slice(0, VISUAL_COUNT)
    else:
        visual_slice = slice(int(vp_visual_start), int(vp_visual_start) + VISUAL_COUNT)
        if visual_slice.stop > support_size:
            raise ValueError(
                f"VP visual slice {visual_slice.start}:{visual_slice.stop} exceeds support size {support_size}"
            )
    distributions = {
        "support": attention_full[:, visual_slice],
        "target": target_full[:, visual_slice],
        "source": source_full[:, visual_slice],
    }
    masses = {key: float(values.sum(axis=1).mean()) for key, values in distributions.items()}
    masses["support_size"] = float(support_size)
    return distributions, masses


def top_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    k = min(int(top_k), int(values.size))
    return np.argpartition(values, -k)[-k:]


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize_vector(left)
    right = normalize_vector(right)
    middle = 0.5 * (left + right)
    value = 0.5 * np.sum(left * np.log((left + 1e-12) / (middle + 1e-12)))
    value += 0.5 * np.sum(right * np.log((right + 1e-12) / (middle + 1e-12)))
    return float(value)


def spatial_compactness(values: np.ndarray, indices: np.ndarray) -> float:
    weights = normalize_vector(values[indices])
    rows, cols = np.divmod(indices, GRID[1])
    points = np.stack(((cols + 0.5) / GRID[1], (rows + 0.5) / GRID[0]), axis=1)
    center = np.sum(points * weights[:, None], axis=0)
    rms = math.sqrt(float(np.sum(weights * np.sum((points - center) ** 2, axis=1))))
    return float(np.clip(1.0 - rms / 0.58, 0.0, 1.0))


def distribution_metrics(
    distributions: dict[str, np.ndarray],
    visual_masses: dict[str, float],
    top_k: int,
    coverage: np.ndarray | None,
) -> dict[str, float]:
    layer_rows: list[dict[str, float]] = []
    for layer_offset in range(len(LAYERS_1BASED)):
        conditional = {
            key: normalize_vector(values[layer_offset])
            for key, values in distributions.items()
        }
        indices = {key: top_indices(values, top_k) for key, values in conditional.items()}
        target_set, source_set = set(indices["target"].tolist()), set(indices["source"].tolist())
        union = target_set | source_set
        overlap = len(target_set & source_set) / max(1, len(union))
        row: dict[str, float] = {
            "target_source_topk_jaccard": float(overlap),
            "target_source_js": js_divergence(conditional["target"], conditional["source"]),
            "source_mass_on_target_topk": float(conditional["source"][indices["target"]].sum()),
        }
        for key in ("support", "target", "source"):
            values = conditional[key]
            idx = indices[key]
            entropy = -float(np.sum(values * np.log(values + 1e-12))) / math.log(VISUAL_COUNT)
            row[f"{key}_topk_mass"] = float(values[idx].sum())
            row[f"{key}_entropy"] = entropy
            row[f"{key}_compactness"] = spatial_compactness(values, idx)
            if coverage is not None:
                row[f"{key}_box_topk_hit"] = float(np.mean(coverage[idx] > 0.02))
                row[f"{key}_box_mass"] = float(np.sum(values * coverage))
        layer_rows.append(row)
    result = {
        key: float(np.mean([row[key] for row in layer_rows]))
        for key in layer_rows[0]
    }
    for key, value in visual_masses.items():
        result[f"{key}_visual_mass"] = float(value)
    if coverage is not None:
        target_gain = np.asarray([
            row["target_box_topk_hit"] - row["support_box_topk_hit"] for row in layer_rows
        ])
        result["target_support_hit_gain_positive_layers"] = float(np.mean(target_gain > 0.0))
        result["target_support_hit_gain_min"] = float(np.min(target_gain))
    return result


def score_hall(metrics: dict[str, float], canonical: str) -> float:
    score = (
        1.05 * metrics["support_topk_mass"]
        + 0.45 * metrics["support_compactness"]
        + 0.45 * metrics["target_topk_mass"]
        + 1.55 * (1.0 - metrics["target_source_topk_jaccard"])
        + 1.50 * metrics["target_source_js"]
        + 0.80 * (1.0 - metrics["source_mass_on_target_topk"])
    )
    if str(canonical).lower() in SMALL_OBJECT_NAMES:
        score += 0.35
    return float(score)


def score_real(metrics: dict[str, float], area_ratio: float) -> float:
    hit_gain = metrics["target_box_topk_hit"] - metrics["support_box_topk_hit"]
    mass_gain = metrics["target_box_mass"] - metrics["support_box_mass"]
    small_bonus = float(np.clip(-math.log10(max(area_ratio, 1e-5)) / 4.0, 0.0, 1.0))
    return float(
        2.35 * hit_gain
        + 2.00 * mass_gain
        + 0.85 * metrics["target_box_topk_hit"]
        + 1.00 * metrics["source_box_topk_hit"]
        + 1.00 * metrics["source_box_mass"]
        + 0.70 * metrics["target_source_topk_jaccard"]
        + 0.45 * (1.0 - metrics["target_source_js"])
        + 0.70 * metrics["target_support_hit_gain_positive_layers"]
        + 0.90 * small_bonus
    )


def rank_fraction(rows: list[dict[str, Any]], field: str) -> dict[tuple[int, int], float]:
    ordered = sorted(rows, key=lambda row: float(row[field]), reverse=True)
    denominator = max(1, len(ordered) - 1)
    return {
        (int(row["image_id"]), int(row["token_idx"])): 1.0 - rank / denominator
        for rank, row in enumerate(ordered)
    }


def add_combined_ranks(rows: list[dict[str, Any]], kind: str) -> None:
    vv = rank_fraction(rows, f"{kind}_score_vv")
    vp = rank_fraction(rows, f"{kind}_score_vp")
    for row in rows:
        key = (int(row["image_id"]), int(row["token_idx"]))
        row[f"{kind}_rank_vv"] = vv[key]
        row[f"{kind}_rank_vp"] = vp[key]
        row[f"{kind}_combined_rank"] = 0.5 * (vv[key] + vp[key])


def diverse_top(rows: list[dict[str, Any]], score_field: str, count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_images: set[int] = set()
    used_categories: defaultdict[str, int] = defaultdict(int)
    for row in sorted(rows, key=lambda item: float(item[score_field]), reverse=True):
        image_id = int(row["image_id"])
        canonical = str(row["canonical"])
        if image_id in used_images or used_categories[canonical] >= 1:
            continue
        selected.append(row)
        used_images.add(image_id)
        used_categories[canonical] += 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        for row in sorted(rows, key=lambda item: float(item[score_field]), reverse=True):
            if row not in selected and int(row["image_id"]) not in used_images:
                selected.append(row)
                used_images.add(int(row["image_id"]))
            if len(selected) >= count:
                break
    return selected


def build_candidates(
    records: list[dict[str, Any]],
    spans: dict[tuple[int, int], dict[str, Any]],
    labeling: dict[str, Any],
    image_dir: Path,
    category_ids: dict[str, int],
    annotations: dict[int, list[dict[str, Any]]],
    top_k: int,
    vp_visual_start: int,
    target_method: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    record_index: dict[tuple[int, int], dict[str, Any]] = {}
    required = {
        "dgst_t_attention_support_per_layer",
        f"dgst_t_{target_method}_gate_per_layer",
        "dgst_t_source_dist_per_layer",
        "dgst_t_vp_attention_support_per_layer",
        f"dgst_t_vp_{target_method}_gate_per_layer",
        "dgst_t_vp_source_dist_per_layer",
    }
    image_geometry: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for number, record in enumerate(records, start=1):
        if not required.issubset(record):
            continue
        image_id = int(record["image_id"])
        token_idx = int(record["response_token_idx"])
        key = (image_id, token_idx)
        span = spans.get(key)
        if span is None:
            continue
        label = int(record.get("label", span.get("label", -1)))
        canonical = str(span.get("canonical_object") or span.get("word") or record.get("token_str"))
        surface = str(span.get("surface") or span.get("surface_word") or canonical)
        coverage = None
        boxes: list[tuple[float, float, float, float]] = []
        area_ratio = 1.0
        max_box_area_ratio = 1.0
        box_count = 0
        if label == 1:
            if image_id not in image_geometry:
                with Image.open(find_image(image_dir, image_id)) as raw:
                    view, offset = center_square_crop(raw.convert("RGB"))
                    image_geometry[image_id] = (view.size, offset)
            view_size, offset = image_geometry[image_id]
            boxes = projected_boxes(
                image_id, canonical, category_ids, annotations, offset, view_size
            )
            if not boxes:
                continue
            coverage = patch_box_coverage(boxes, view_size)
            box_count = len(boxes)
            # The token names a COCO category, so when several instances are
            # present the relevant target is their union rather than whichever
            # annotation happens to have the smallest box.
            area_ratio = float(np.mean(coverage))
            max_box_area_ratio = max(
                (width * height) / max(1.0, float(view_size[0] * view_size[1]))
                for _x, _y, width, height in boxes
            )
        item = labeling.get(str(image_id), {})
        row: dict[str, Any] = {
            "image_id": image_id,
            "token_idx": token_idx,
            "label": label,
            "surface": surface,
            "canonical": canonical,
            "char_start": int(span.get("char_start", 0)),
            "char_end": int(span.get("char_end", 0)),
            "caption": str(item.get("generated_text", "")),
            "area_ratio": float(area_ratio),
            "max_box_area_ratio": float(max_box_area_ratio),
            "box_count": int(box_count),
            "is_small": (
                bool(area_ratio < 0.08 and box_count <= 3)
                if label == 1 else canonical.lower() in SMALL_OBJECT_NAMES
            ),
        }
        for scope in ("vv", "vp"):
            distributions, visual_masses = scope_distributions(
                record, scope, vp_visual_start, target_method
            )
            metrics = distribution_metrics(distributions, visual_masses, top_k, coverage)
            for metric_name, value in metrics.items():
                row[f"{scope}_{metric_name}"] = float(value)
            if label == 0:
                row[f"hall_score_{scope}"] = score_hall(metrics, canonical)
            else:
                row[f"real_score_{scope}"] = score_real(metrics, area_ratio)
        rows.append(row)
        record_index[key] = record
        if number % 2000 == 0:
            print(f"[score] processed {number}/{len(records)} records; kept {len(rows)}")
    hall_rows = [row for row in rows if int(row["label"]) == 0]
    real_rows = [row for row in rows if int(row["label"]) == 1]
    add_combined_ranks(hall_rows, "hall")
    add_combined_ranks(real_rows, "real")
    return rows, record_index


def select_pairs(rows: list[dict[str, Any]], count: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_image: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_image[int(row["image_id"])][int(row["label"])].append(row)
    candidates = []
    for image_id, groups in by_image.items():
        if not groups[0] or not groups[1]:
            continue
        preferred_real = [
            row for row in groups[1]
            if bool(row["is_small"])
            and all(
                row[f"{scope}_target_box_topk_hit"] - row[f"{scope}_support_box_topk_hit"] >= 0.035
                and row[f"{scope}_source_box_topk_hit"] >= 0.60
                for scope in ("vv", "vp")
            )
        ]
        real = max(
            preferred_real or groups[1],
            key=lambda row: float(row["real_combined_rank"]),
        )
        hall = max(groups[0], key=lambda row: float(row["hall_combined_rank"]))
        contrast = 0.0
        for scope in ("vv", "vp"):
            contrast += (
                real[f"{scope}_target_source_topk_jaccard"]
                - hall[f"{scope}_target_source_topk_jaccard"]
                + hall[f"{scope}_target_source_js"]
                - real[f"{scope}_target_source_js"]
            )
        pair_score = (
            float(real["real_combined_rank"])
            + float(hall["hall_combined_rank"])
            + 0.30 * contrast
            + (0.45 if bool(real["is_small"]) else 0.0)
        )
        candidates.append((pair_score, image_id, real, hall))
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = []
    used_real: set[str] = set()
    used_hall: set[str] = set()
    for _score, _image_id, real, hall in candidates:
        if str(real["canonical"]) in used_real and str(hall["canonical"]) in used_hall:
            continue
        selected.append((real, hall))
        used_real.add(str(real["canonical"]))
        used_hall.add(str(hall["canonical"]))
        if len(selected) >= count:
            break
    return selected


def topk_sparse(values: np.ndarray, top_k: int) -> np.ndarray:
    values = normalize_vector(values)
    indices = top_indices(values, top_k)
    sparse = np.zeros_like(values)
    sparse[indices] = values[indices]
    return sparse.reshape(GRID)


def resize_heat(sparse: np.ndarray, width: int, height: int) -> np.ndarray:
    peak = float(np.max(sparse))
    if peak <= 0.0:
        return np.zeros((height, width), dtype=np.float64)
    grid = Image.fromarray(np.uint8(np.round(np.clip(sparse / peak, 0, 1) * 255)), mode="L")
    heat = grid.resize((width, height), resample=Image.Resampling.BICUBIC)
    heat = heat.filter(ImageFilter.GaussianBlur(radius=max(1.2, min(width, height) / 105.0)))
    return np.asarray(heat, dtype=np.float64) / 255.0


def cool_heat_overlay(image: Image.Image, sparse: np.ndarray, global_peak: float) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    height, width = base.shape[:2]
    local_peak = float(np.max(sparse))
    heat = resize_heat(sparse, width, height)
    if global_peak > 0 and local_peak > 0:
        heat *= local_peak / global_peak
    heat = np.clip(heat, 0.0, 1.0)
    cold = np.array([0.035, 0.075, 0.285], dtype=np.float64)
    cold_base = 0.40 * base + 0.60 * cold
    hot = plt.colormaps["turbo"](heat)[..., :3]
    alpha = (0.94 * np.power(heat, 0.56))[..., None]
    return np.clip(cold_base * (1.0 - alpha) + hot * alpha, 0.0, 1.0)


def draw_boxes(ax: plt.Axes, boxes: list[tuple[float, float, float, float]]) -> None:
    for x, y, width, height in boxes:
        patch = Rectangle(
            (x, y), width, height, fill=False, edgecolor="#fff26b", linewidth=1.55,
        )
        patch.set_path_effects([path_effects.withStroke(linewidth=2.8, foreground="#111827")])
        ax.add_patch(patch)


def layer_label(ax: plt.Axes, layer: int) -> None:
    label = ax.text(
        0.035, 0.95, f"Layer {layer}", transform=ax.transAxes,
        ha="left", va="top", color="white", fontsize=9.5, weight="medium",
    )
    label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="black")])


def case_payload(
    row: dict[str, Any],
    scope: str,
    record_index: dict[tuple[int, int], dict[str, Any]],
    image_dir: Path,
    category_ids: dict[str, int],
    annotations: dict[int, list[dict[str, Any]]],
    vp_visual_start: int,
    target_method: str,
    show_boxes: bool = False,
) -> dict[str, Any]:
    image_id, token_idx = int(row["image_id"]), int(row["token_idx"])
    with Image.open(find_image(image_dir, image_id)) as raw:
        image, offset = center_square_crop(raw.convert("RGB"))
    boxes = (
        projected_boxes(image_id, str(row["canonical"]), category_ids, annotations, offset, image.size)
        if int(row["label"]) == 1 else []
    )
    distributions, visual_masses = scope_distributions(
        record_index[(image_id, token_idx)], scope, vp_visual_start, target_method
    )
    span = {"char_start": row["char_start"], "char_end": row["char_end"]}
    return {
        "row": row,
        "scope": scope,
        "image": image,
        "boxes": boxes,
        "plot_boxes": bool(show_boxes),
        "distributions": distributions,
        "visual_masses": visual_masses,
        "signals": signal_specs(target_method),
        "target_method": target_method,
        "snippet": caption_snippet(str(row["caption"]), span),
    }


def save_original(case: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = case["image"].copy()
    draw = ImageDraw.Draw(image)
    if case.get("plot_boxes", False):
        for x, y, width, height in case["boxes"]:
            draw.rectangle((x, y, x + width, y + height), outline="#fff26b", width=3)
    image.save(path)


def signal_sparse(case: dict[str, Any], signal: str, top_k: int) -> list[np.ndarray]:
    return [topk_sparse(values, top_k) for values in case["distributions"][signal]]


def save_individual_heatmaps(case: dict[str, Any], root: Path, top_k: int) -> None:
    row, scope = case["row"], case["scope"]
    label_name = "real" if int(row["label"]) else "hall"
    stem = f"{label_name}-{safe_slug(row['surface'])}-idx{int(row['token_idx'])}"
    for signal, _title in case["signals"]:
        sparse_layers = signal_sparse(case, signal, top_k)
        global_peak = max(float(layer.max()) for layer in sparse_layers)
        output_dir = root / str(row["image_id"]) / scope / signal
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer, sparse in zip(LAYERS_1BASED, sparse_layers):
            overlay = cool_heat_overlay(case["image"], sparse, global_peak)
            figure, ax = plt.subplots(figsize=(3.1, 3.1))
            ax.imshow(overlay)
            if case.get("plot_boxes", False):
                draw_boxes(ax, case["boxes"])
            layer_label(ax, layer)
            ax.axis("off")
            figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
            figure.savefig(output_dir / f"{stem}-layer{layer}.png", dpi=160, bbox_inches="tight", pad_inches=0)
            plt.close(figure)


def plot_single(case: dict[str, Any], path: Path, top_k: int, dpi: int) -> None:
    figure = plt.figure(figsize=(18.2, 7.45), facecolor="white")
    grid = figure.add_gridspec(
        3, 6, width_ratios=(1.14, 1, 1, 1, 1, 1),
        left=0.025, right=0.995, bottom=0.035, top=0.835, wspace=0.04, hspace=0.08,
    )
    original_ax = figure.add_subplot(grid[:, 0])
    original_ax.imshow(case["image"])
    if case.get("plot_boxes", False):
        draw_boxes(original_ax, case["boxes"])
    original_ax.set_title("Original image", fontsize=11, weight="bold", pad=7)
    original_ax.axis("off")
    for row_index, (signal, title) in enumerate(case["signals"]):
        sparse_layers = signal_sparse(case, signal, top_k)
        global_peak = max(float(layer.max()) for layer in sparse_layers)
        for col_index, (layer, sparse) in enumerate(zip(LAYERS_1BASED, sparse_layers), start=1):
            ax = figure.add_subplot(grid[row_index, col_index])
            ax.imshow(cool_heat_overlay(case["image"], sparse, global_peak))
            if case.get("plot_boxes", False):
                draw_boxes(ax, case["boxes"])
            layer_label(ax, layer)
            ax.axis("off")
            if col_index == 1:
                ax.set_ylabel(title, fontsize=10.4, weight="bold", labelpad=8)
    row = case["row"]
    label_name = "REAL" if int(row["label"]) else "HALL"
    scope = str(case["scope"]).upper()
    figure.suptitle(
        f"{scope} · {label_name} token ‘{row['surface']}’ (canonical: {row['canonical']}) · Top-{top_k}",
        fontsize=16.5, weight="bold", y=0.985,
    )
    figure.text(0.5, 0.905, case["snippet"], ha="center", va="top", fontsize=10.2, wrap=True)
    if case["scope"] == "vp":
        masses = case["visual_masses"]
        figure.text(
            0.985, 0.895,
            f"VP visual mass: A={masses['support']:.3f}, T={masses['target']:.3f}, S={masses['source']:.3f}",
            ha="right", va="top", fontsize=8.5, color="#374151",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_pair(cases: tuple[dict[str, Any], dict[str, Any]], path: Path, top_k: int, dpi: int) -> None:
    figure = plt.figure(figsize=(18.2, 12.7), facecolor="white")
    grid = figure.add_gridspec(
        6, 6, width_ratios=(1.14, 1, 1, 1, 1, 1),
        left=0.025, right=0.995, bottom=0.025, top=0.84, wspace=0.04, hspace=0.075,
    )
    original_ax = figure.add_subplot(grid[:, 0])
    original_ax.imshow(cases[0]["image"])
    if cases[0].get("plot_boxes", False):
        draw_boxes(original_ax, cases[0]["boxes"])
    original_ax.set_title("Original image", fontsize=11, weight="bold", pad=7)
    original_ax.axis("off")
    for block, case in enumerate(cases):
        row = case["row"]
        label_name = "REAL" if int(row["label"]) else "HALL"
        for signal_offset, (signal, title) in enumerate(case["signals"]):
            row_index = block * 3 + signal_offset
            sparse_layers = signal_sparse(case, signal, top_k)
            global_peak = max(float(layer.max()) for layer in sparse_layers)
            for col_index, (layer, sparse) in enumerate(zip(LAYERS_1BASED, sparse_layers), start=1):
                ax = figure.add_subplot(grid[row_index, col_index])
                ax.imshow(cool_heat_overlay(case["image"], sparse, global_peak))
                if case.get("plot_boxes", False):
                    draw_boxes(ax, case["boxes"])
                layer_label(ax, layer)
                ax.axis("off")
                if col_index == 1:
                    ax.set_ylabel(
                        f"{label_name} ‘{row['surface']}’\n{title}",
                        fontsize=9.6, weight="bold", labelpad=8,
                    )
    real_case, hall_case = cases
    scope = str(real_case["scope"]).upper()
    figure.suptitle(
        f"{scope} same-image contrast · REAL ‘{real_case['row']['surface']}’ vs HALL ‘{hall_case['row']['surface']}’ · Top-{top_k}",
        fontsize=16.5, weight="bold", y=0.99,
    )
    figure.text(0.5, 0.928, f"REAL: {real_case['snippet']}", ha="center", fontsize=9.7, wrap=True)
    figure.text(0.5, 0.895, f"HALL: {hall_case['snippet']}", ha="center", fontsize=9.7, wrap=True)
    separator = plt.Line2D([0.24, 0.995], [0.43, 0.43], transform=figure.transFigure, color="#6b7280", lw=1.0)
    figure.add_artist(separator)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_contact_sheet(paths: Iterable[Path], output_path: Path, title: str) -> None:
    paths = list(paths)
    if not paths:
        return
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((900, 380), Image.Resampling.LANCZOS)
            thumbs.append((path.stem, thumb.copy()))
    columns = 2
    cell_w, cell_h = 930, 430
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * cell_w, 55 + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 15), title, fill="#111827")
    for index, (name, thumb) in enumerate(thumbs):
        x, y = (index % columns) * cell_w, 55 + (index // columns) * cell_h
        sheet.paste(thumb, (x + 10, y + 28))
        draw.text((x + 12, y + 5), name, fill="#111827")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (bool(value) if isinstance(value, np.bool_) else value.item() if isinstance(value, np.generic) else value)
        for key, value in row.items()
    }


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.top_k > VISUAL_COUNT:
        raise ValueError(f"top-k must be within 1..{VISUAL_COUNT}")
    output_root = args.output_root.resolve()
    save_dir = (
        args.save_dir.resolve()
        if args.save_dir is not None
        else output_root / f"analysis/paper_heatmap_cases_top{args.top_k}_vv_vp"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    labeling = json.loads((output_root / "labeling.json").read_text(encoding="utf-8"))
    spans = span_lookup(labeling)
    category_ids, annotations = load_coco_annotations(args.instances_file)
    print(f"[load] reading {output_root / 'features.pkl'}")
    with (output_root / "features.pkl").open("rb") as handle:
        records = pickle.load(handle)
    print(f"[load] {len(records)} feature records")
    if records:
        vp_size = int(np.asarray(records[0]["dgst_t_vp_attention_support_per_layer"]).shape[1])
        expected_end = int(args.vp_visual_start) + VISUAL_COUNT
        if expected_end > vp_size:
            raise ValueError(f"VP visual end {expected_end} exceeds support size {vp_size}")
        print(
            f"[VP mapping] support={vp_size}; visual columns="
            f"{args.vp_visual_start}:{expected_end} (decoder visual positions)"
        )
    rows, record_index = build_candidates(
        records, spans, labeling, args.coco_image_dir, category_ids, annotations,
        args.top_k, args.vp_visual_start, args.target_method,
    )
    hall_rows = [row for row in rows if int(row["label"]) == 0]
    real_rows = [row for row in rows if int(row["label"]) == 1]
    selected_hall = diverse_top(hall_rows, "hall_combined_rank", args.hall_count)
    strict_real_rows = [
        row for row in real_rows
        if bool(row["is_small"])
        and all(
            row[f"{scope}_target_box_topk_hit"] - row[f"{scope}_support_box_topk_hit"] >= 0.05
            and row[f"{scope}_target_box_mass"] - row[f"{scope}_support_box_mass"] >= 0.025
            and row[f"{scope}_source_box_topk_hit"] >= 0.65
            for scope in ("vv", "vp")
        )
    ]
    real_selection_pool = strict_real_rows if len(strict_real_rows) >= args.real_count else real_rows
    selected_real = diverse_top(real_selection_pool, "real_combined_rank", args.real_count)
    selected_pairs = select_pairs(rows, args.paired_count)
    if args.selection_mode == "curated":
        by_key = {
            (int(row["image_id"]), int(row["token_idx"])): row for row in rows
        }
        selected_hall = [by_key[key] for key in CURATED_HALL_KEYS]
        selected_real = [by_key[key] for key in CURATED_REAL_KEYS]
        selected_pairs = [
            (by_key[(image_id, real_idx)], by_key[(image_id, hall_idx)])
            for image_id, real_idx, hall_idx in CURATED_PAIR_KEYS
        ]
    shortlist_hall = sorted(hall_rows, key=lambda row: row["hall_combined_rank"], reverse=True)[: args.shortlist_count]
    shortlist_real = sorted(real_rows, key=lambda row: row["real_combined_rank"], reverse=True)[: args.shortlist_count]
    write_csv(shortlist_hall, save_dir / "shortlist_hall.csv")
    write_csv(shortlist_real, save_dir / "shortlist_real.csv")
    write_csv(rows, save_dir / "candidate_metrics_all.csv")

    selection_payload = {
        "settings": {
            "output_root": str(output_root), "top_k": args.top_k,
            "layers_1based": list(LAYERS_1BASED), "vp_visual_start": args.vp_visual_start,
            "vp_visual_end_exclusive": args.vp_visual_start + VISUAL_COUNT,
            "target_method": args.target_method,
            "target_formula": (
                f"renormalize(support_attention * {args.target_method}_gate)"
            ),
            "selection_mode": args.selection_mode,
        },
        "hall": [serializable_row(row) for row in selected_hall],
        "real": [serializable_row(row) for row in selected_real],
        "paired": [
            {"real": serializable_row(real), "hall": serializable_row(hall)}
            for real, hall in selected_pairs
        ],
    }
    (save_dir / "selected_cases.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat_selected = selected_hall + selected_real
    write_csv(flat_selected, save_dir / "selected_single_cases.csv")

    panel_paths: dict[str, list[Path]] = defaultdict(list)
    for kind, selected in (("hall", selected_hall), ("real", selected_real)):
        for rank, row in enumerate(selected, start=1):
            image_root = save_dir / str(row["image_id"])
            first_case = case_payload(
                row, "vv", record_index, args.coco_image_dir, category_ids, annotations,
                args.vp_visual_start, args.target_method, args.show_boxes,
            )
            save_original(first_case, image_root / "original.png")
            for scope in ("vv", "vp"):
                case = case_payload(
                    row, scope, record_index, args.coco_image_dir, category_ids, annotations,
                    args.vp_visual_start, args.target_method, args.show_boxes,
                )
                stem = f"{kind}{rank:02d}-{safe_slug(row['surface'])}-idx{int(row['token_idx'])}-{scope}"
                panel = save_dir / "panels" / kind / f"{stem}.png"
                plot_single(case, panel, args.top_k, args.dpi)
                save_individual_heatmaps(case, save_dir, args.top_k)
                panel_paths[f"{kind}_{scope}"].append(panel)
                print(f"[render] {panel}")

    for rank, (real, hall) in enumerate(selected_pairs, start=1):
        image_root = save_dir / str(real["image_id"])
        first_case = case_payload(
            real, "vv", record_index, args.coco_image_dir, category_ids, annotations,
            args.vp_visual_start, args.target_method, args.show_boxes,
        )
        save_original(first_case, image_root / "original.png")
        for scope in ("vv", "vp"):
            real_case = case_payload(
                real, scope, record_index, args.coco_image_dir, category_ids, annotations,
                args.vp_visual_start, args.target_method, args.show_boxes,
            )
            hall_case = case_payload(
                hall, scope, record_index, args.coco_image_dir, category_ids, annotations,
                args.vp_visual_start, args.target_method, args.show_boxes,
            )
            stem = (
                f"pair{rank:02d}-real-{safe_slug(real['surface'])}-hall-"
                f"{safe_slug(hall['surface'])}-{scope}"
            )
            panel = save_dir / "panels" / "paired" / f"{stem}.png"
            plot_pair((real_case, hall_case), panel, args.top_k, args.dpi)
            save_individual_heatmaps(real_case, save_dir, args.top_k)
            save_individual_heatmaps(hall_case, save_dir, args.top_k)
            panel_paths[f"paired_{scope}"].append(panel)
            print(f"[render] {panel}")

    for name, paths in panel_paths.items():
        make_contact_sheet(
            paths,
            save_dir / f"contact_sheet_{name}.jpg",
            f"{name} | {args.target_method} | Top-{args.top_k}",
        )

    readme = [
        f"# VV/VP publication heatmap candidates (Top-{args.top_k})",
        "",
        f"- Layers (paper 1-based): {', '.join(map(str, LAYERS_1BASED))}.",
        f"- Branch: `{args.target_method}`.",
        f"- Target: `renormalize(support_attention * {args.target_method}_gate)`.",
        f"- Every spatial map retains only the largest {args.top_k} visual regions.",
        "- VP prompt tokens are excluded from spatial projection; VP visual mass is reported on each panel.",
        (
            "- Yellow boxes are diagnostic COCO ground-truth boxes."
            if args.show_boxes else
            "- COCO boxes are used only for selection metrics and are omitted from the figures."
        ),
        "- Candidate selection combines VV and VP quantitative ranks, then enforces image/category diversity.",
        "",
        "## Selected hallucinated tokens",
        "",
    ]
    for row in selected_hall:
        readme.append(
            f"- image `{row['image_id']}`, token `{row['surface']}` / `{row['canonical']}`, "
            f"index `{row['token_idx']}`: {caption_snippet(row['caption'], row)}"
        )
    readme.extend(["", "## Selected real tokens", ""])
    for row in selected_real:
        readme.append(
            f"- image `{row['image_id']}`, token `{row['surface']}` / `{row['canonical']}`, "
            f"index `{row['token_idx']}`, box area ratio `{row['area_ratio']:.4f}`: "
            f"{caption_snippet(row['caption'], row)}"
        )
    readme.extend(["", "## Selected same-image pairs", ""])
    for real, hall in selected_pairs:
        readme.append(
            f"- image `{real['image_id']}`: REAL `{real['surface']}` (idx {real['token_idx']}) vs "
            f"HALL `{hall['surface']}` (idx {hall['token_idx']})."
        )
    (save_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[done] outputs: {save_dir}")


if __name__ == "__main__":
    main()
