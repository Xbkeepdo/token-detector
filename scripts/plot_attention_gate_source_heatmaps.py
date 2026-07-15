#!/usr/bin/env python3
"""Plot paper-style VV/VP attention, gated attention, and source heatmaps."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


LABEL_NAMES = {0: "Hallucinated", 1: "Non-hallucinated"}
MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
SCOPES = ("vv", "vp")
MATRIX_KEYS = {
    "vv": {
        "attention": "dgst_t_vv_support_attention_per_layer",
        "gate": "dgst_t_vv_semantic_gate_per_layer",
        "source": "dgst_t_vv_source_dist_per_layer",
        "positions": "dgst_t_vv_support_positions",
    },
    "vp": {
        "attention": "dgst_t_vp_support_attention_per_layer",
        "gate": "dgst_t_vp_semantic_gate_per_layer",
        "source": "dgst_t_vp_source_dist_per_layer",
        "positions": "dgst_t_vp_support_positions",
    },
}


@dataclass
class Candidate:
    score: float
    record: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment", default="COCO4000-all")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=list(MODEL_NAMES),
    )
    parser.add_argument("--samples-per-label", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/coco4000_all_attention_gate_source_heatmaps"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--selection-json",
        type=Path,
        help="Reuse sample identities from a previous selected_samples.json instead of rescoring.",
    )
    return parser.parse_args()


def iter_pickle_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open("rb") as handle:
            while True:
                try:
                    obj = pickle.load(handle)
                except EOFError:
                    break
                records = obj if isinstance(obj, list) else [obj]
                yield from records


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = values.clamp_min(0.0)
    totals = values.sum(dim=1, keepdim=True)
    return torch.where(totals > 0.0, values / totals.clamp_min(1e-12), torch.zeros_like(values))


def normalized_matrices(record: dict, scope: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = MATRIX_KEYS[scope]
    attention = normalize_rows(torch.as_tensor(record[keys["attention"]]))
    gate = torch.nan_to_num(
        torch.as_tensor(record[keys["gate"]]).float(),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    gated = normalize_rows(attention * gate)
    source = normalize_rows(torch.as_tensor(record[keys["source"]]))
    return attention, gated, source


def gate_effect_score(record: dict) -> float:
    scope_scores = []
    for scope in SCOPES:
        attention, gated, _source = normalized_matrices(record, scope)
        total_variation = 0.5 * torch.abs(attention - gated).sum(dim=1)
        scope_scores.append(float(total_variation.mean().item()))
    return float(np.mean(scope_scores))


def has_required_fields(record: dict) -> bool:
    return all(
        key in record
        for scope in SCOPES
        for key in MATRIX_KEYS[scope].values()
    )


def retain_candidate(
    selected: dict[int, dict[int, Candidate]],
    *,
    label: int,
    image_id: int,
    candidate: Candidate,
    limit: int,
) -> None:
    by_image = selected[label]
    current = by_image.get(image_id)
    if current is not None:
        if candidate.score > current.score:
            by_image[image_id] = candidate
        return
    if len(by_image) < limit:
        by_image[image_id] = candidate
        return
    weakest_image, weakest = min(by_image.items(), key=lambda item: item[1].score)
    if candidate.score > weakest.score:
        del by_image[weakest_image]
        by_image[image_id] = candidate


def select_samples(part_paths: list[Path], samples_per_label: int) -> tuple[list[Candidate], int]:
    selected: dict[int, dict[int, Candidate]] = {0: {}, 1: {}}
    record_count = 0
    for record in iter_pickle_records(part_paths):
        record_count += 1
        label = int(record.get("label", -1))
        if label not in selected or not has_required_fields(record):
            continue
        score = gate_effect_score(record)
        if not math.isfinite(score):
            continue
        retain_candidate(
            selected,
            label=label,
            image_id=int(record["image_id"]),
            candidate=Candidate(score=score, record=record),
            limit=samples_per_label,
        )

    samples = []
    for label in (0, 1):
        if len(selected[label]) < samples_per_label:
            raise RuntimeError(
                f"Only found {len(selected[label])} valid records for label {label}; "
                f"need {samples_per_label}."
            )
        samples.extend(sorted(selected[label].values(), key=lambda item: item.score, reverse=True))
    return samples, record_count


def restore_samples(
    part_paths: list[Path],
    *,
    model: str,
    manifest_rows: list[dict],
) -> tuple[list[Candidate], int]:
    model_rows = [row for row in manifest_rows if row["model"] == model]
    desired = {
        (
            int(row["image_id"]),
            int(row["response_token_idx"]),
            str(row["canonical_word"]),
        ): row
        for row in model_rows
    }
    found: dict[tuple[int, int, str], Candidate] = {}
    record_count = 0
    for record in iter_pickle_records(part_paths):
        record_count += 1
        key = (
            int(record.get("image_id", -1)),
            int(record.get("response_token_idx", -1)),
            str(record.get("token_str", "")),
        )
        if key in desired and key not in found:
            found[key] = Candidate(
                score=float(desired[key]["gate_effect_tv"]),
                record=record,
            )
    missing = desired.keys() - found.keys()
    if missing:
        raise RuntimeError(f"Could not restore {len(missing)} selected records for {model}: {sorted(missing)}")
    ordered = []
    for row in model_rows:
        key = (int(row["image_id"]), int(row["response_token_idx"]), str(row["canonical_word"]))
        ordered.append(found[key])
    return ordered, record_count


def load_labeling(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): value for key, value in data.items()}


def target_surface(record: dict, labeling: dict[str, dict]) -> tuple[str, str]:
    canonical = str(record.get("token_str", "unknown"))
    response_index = int(record.get("response_token_idx", -1))
    item = labeling.get(str(int(record["image_id"])), {})
    for span in item.get("object_token_spans", []):
        indices = [int(value) for value in span.get("token_indices", [])]
        if response_index in indices:
            return str(span.get("surface", canonical)), str(span.get("word", canonical))
    return canonical, canonical


def visual_runs(vv_positions: list[int], vp_positions: list[int]) -> list[tuple[bool, int, int]]:
    visual = set(int(value) for value in vv_positions)
    flags = [int(position) in visual for position in vp_positions]
    runs: list[tuple[bool, int, int]] = []
    for index, flag in enumerate(flags):
        if not runs or runs[-1][0] != flag:
            runs.append((flag, index, index))
        else:
            previous = runs[-1]
            runs[-1] = (previous[0], previous[1], index)
    return runs


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "token"


def plot_sample(
    *,
    model: str,
    candidate: Candidate,
    labeling: dict[str, dict],
    output_stem: Path,
    dpi: int,
) -> dict:
    record = candidate.record
    label = int(record["label"])
    surface, canonical = target_surface(record, labeling)

    figure, axes = plt.subplots(2, 3, figsize=(16.2, 8.2), constrained_layout=True)
    images = []
    for row, scope in enumerate(SCOPES):
        attention, gated, source = normalized_matrices(record, scope)
        matrices = (attention, gated, source)
        for column, (matrix, title) in enumerate(
            zip(
                matrices,
                ("Raw attention", r"Gated attention: norm($A \odot g$)", "Source distribution"),
            )
        ):
            log_probability = torch.log10(matrix.clamp_min(1e-6)).cpu().numpy()
            axis = axes[row, column]
            image = axis.imshow(
                log_probability,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=-6.0,
                vmax=0.0,
                rasterized=True,
            )
            images.append(image)
            axis.set_title(title, fontsize=11, fontweight="semibold", pad=7)
            axis.set_xlabel("Support token index", fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{scope.upper()} scope\nDecoder layer", fontsize=10)
            else:
                axis.set_ylabel("Decoder layer", fontsize=9)
            layer_count = matrix.shape[0]
            y_step = max(1, layer_count // 7)
            ticks = np.arange(0, layer_count, y_step)
            axis.set_yticks(ticks, labels=[str(int(value) + 1) for value in ticks])
            axis.tick_params(axis="both", labelsize=8, length=2)
            for spine in axis.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#404040")

        if scope == "vp":
            vv_positions = record[MATRIX_KEYS["vv"]["positions"]]
            vp_positions = record[MATRIX_KEYS["vp"]["positions"]]
            runs = visual_runs(vv_positions, vp_positions)
            for axis in axes[row]:
                for _is_visual, start, _end in runs[1:]:
                    axis.axvline(start - 0.5, color="white", linewidth=0.8, alpha=0.85)
                for is_visual, start, end in runs:
                    axis.text(
                        (start + end) / 2.0,
                        0.985,
                        "Visual" if is_visual else "Prompt",
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=7.5,
                        color="white",
                        bbox={
                            "boxstyle": "square,pad=0.12",
                            "facecolor": "black",
                            "edgecolor": "none",
                            "alpha": 0.42,
                        },
                    )

    colorbar = figure.colorbar(images[-1], ax=axes, location="right", shrink=0.88, pad=0.012)
    colorbar.set_label("Normalized mass (log scale)", fontsize=9)
    colorbar.set_ticks([-6, -5, -4, -3, -2, -1, 0])
    colorbar.set_ticklabels([r"$10^{-6}$", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    colorbar.ax.tick_params(labelsize=8)

    title = (
        f"{MODEL_NAMES[model]} | Target: '{surface}'"
        + (f" (COCO class: {canonical})" if surface.lower() != canonical.lower() else "")
        + f" | {LABEL_NAMES[label]} | Image {int(record['image_id'])}"
    )
    subtitle = (
        f"Response token index: {int(record.get('response_token_idx', -1))}"
        f" | Mean gate effect (TV): {candidate.score:.4f}"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=13.5, fontweight="bold", linespacing=1.35)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "model": model,
        "model_display": MODEL_NAMES[model],
        "label": label,
        "label_name": LABEL_NAMES[label],
        "image_id": int(record["image_id"]),
        "response_token_idx": int(record.get("response_token_idx", -1)),
        "surface": surface,
        "canonical_word": canonical,
        "gate_effect_tv": candidate.score,
        "vv_layers": int(torch.as_tensor(record[MATRIX_KEYS["vv"]["attention"]]).shape[0]),
        "vv_support_tokens": len(record[MATRIX_KEYS["vv"]["positions"]]),
        "vp_support_tokens": len(record[MATRIX_KEYS["vp"]["positions"]]),
        "png": str(output_stem.with_suffix(".png")),
        "pdf": str(output_stem.with_suffix(".pdf")),
    }


def write_summary(output_dir: Path, rows: list[dict], scanned: dict[str, int]) -> None:
    json_path = output_dir / "selected_samples.json"
    csv_path = output_dir / "selected_samples.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# COCO4000-all attention/gate/source heatmaps",
        "",
        "Each figure compares row-normalized raw support attention, row-normalized gated attention "
        "(`attention * semantic_gate`), and the saved source distribution across decoder layers.",
        "Values use one fixed log-mass color scale (`1e-6` to `1`) across all models and panels.",
        "Samples are selected independently per model and label by the largest mean total-variation "
        "change from raw to gated attention, with unique image IDs.",
        "",
    ]
    for model in dict.fromkeys(row["model"] for row in rows):
        lines.extend(
            [
                f"## {MODEL_NAMES[model]}",
                "",
                f"Scanned feature records: {scanned[model]}",
                "",
                "| # | Label | Target word | COCO class | Image ID | Gate effect (TV) | Figure |",
                "|---:|---|---|---|---:|---:|---|",
            ]
        )
        model_rows = [row for row in rows if row["model"] == model]
        for index, row in enumerate(model_rows, start=1):
            relative_png = Path(row["png"]).relative_to(output_dir)
            lines.append(
                f"| {index} | {row['label_name']} | {row['surface']} | "
                f"{row['canonical_word']} | {row['image_id']} | "
                f"{row['gate_effect_tv']:.4f} | [PNG]({relative_png.as_posix()}) |"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = None
    if args.selection_json is not None:
        manifest_rows = json.loads(args.selection_json.read_text(encoding="utf-8"))

    rows: list[dict] = []
    scanned: dict[str, int] = {}
    for model in args.models:
        experiment_dir = args.outputs_root / model / args.experiment
        part_paths = sorted(experiment_dir.glob("features.part*.pkl"))
        if not part_paths:
            raise FileNotFoundError(f"No feature parts found under {experiment_dir}")
        if manifest_rows is None:
            print(f"[{model}] Scanning and scoring {len(part_paths)} feature parts...", flush=True)
            samples, record_count = select_samples(part_paths, args.samples_per_label)
        else:
            print(f"[{model}] Restoring selected records from {len(part_paths)} feature parts...", flush=True)
            samples, record_count = restore_samples(
                part_paths,
                model=model,
                manifest_rows=manifest_rows,
            )
        scanned[model] = record_count
        labeling = load_labeling(experiment_dir / "labeling.json")
        model_dir = args.output_dir / model
        for label in (0, 1):
            label_samples = [item for item in samples if int(item.record["label"]) == label]
            for rank, candidate in enumerate(label_samples, start=1):
                surface, _canonical = target_surface(candidate.record, labeling)
                stem = model_dir / (
                    f"{LABEL_NAMES[label].lower().replace('-', '_')}_{rank:02d}_"
                    f"{safe_slug(surface)}_image{int(candidate.record['image_id'])}"
                )
                print(
                    f"[{model}] Plotting {LABEL_NAMES[label]} {rank}/{args.samples_per_label}: "
                    f"{surface} (image {int(candidate.record['image_id'])})",
                    flush=True,
                )
                rows.append(
                    plot_sample(
                        model=model,
                        candidate=candidate,
                        labeling=labeling,
                        output_stem=stem,
                        dpi=args.dpi,
                    )
                )
        del samples, labeling
        gc.collect()

    write_summary(args.output_dir, rows, scanned)
    print(f"Wrote {len(rows)} figures and metadata to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
