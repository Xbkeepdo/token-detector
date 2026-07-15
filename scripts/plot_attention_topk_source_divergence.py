#!/usr/bin/env python3
"""Compare support-attention and source distributions on attention Top-K.

For each object-token row and decoder layer, the region is selected using the
raw support attention's Top-K positions.  Attention and source values on that
same region are independently L1-normalized before computing JS divergence,
KL(attention || source), and KL(source || attention).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


MODEL_NAMES = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl_2_5_8b": "InternVL2.5-8B",
}
LABEL_NAMES = {0: "Hallucination", 1: "Non-hallucination"}
LABEL_COLORS = {0: "#d55e00", 1: "#0072b2"}
SCOPE_NAMES = {"vv": "VV (visual support)", "vp": "VP (visual + prompt support)"}
METRIC_NAMES = {
    "js": "JS(attention, source)",
    "kl_attention_source": "KL(attention || source)",
    "kl_source_attention": "KL(source || attention)",
}
METRIC_ORDER = tuple(METRIC_NAMES)


@dataclass
class RunningLayerStats:
    count: int = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("Divergence curve contains NaN or infinity.")
        if self.total is None:
            self.total = np.zeros_like(array)
            self.total_sq = np.zeros_like(array)
        if array.shape != self.total.shape:
            raise ValueError(
                f"Layer count changed from {self.total.size} to {array.size} within one group."
            )
        self.count += 1
        self.total += array
        assert self.total_sq is not None
        self.total_sq += array * array

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.count <= 0 or self.total is None or self.total_sq is None:
            raise ValueError("Cannot finalize empty layer statistics.")
        mean = self.total / float(self.count)
        if self.count > 1:
            variance = (self.total_sq - float(self.count) * mean * mean) / float(
                self.count - 1
            )
            std = np.sqrt(np.maximum(variance, 0.0))
        else:
            std = np.zeros_like(mean)
        sem = std / math.sqrt(float(self.count))
        return {"count": self.count, "mean": mean, "std": std, "sem": sem}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_NAMES),
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--scopes", nargs="+", choices=list(SCOPE_NAMES), default=["vv", "vp"])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/coco500_attention_topk32_source_divergence"),
    )
    return parser.parse_args()


def smooth_probability(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    values = torch.nan_to_num(
        values.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    totals = values.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(values, 1.0 / float(values.shape[-1]))
    probabilities = torch.where(totals > 0.0, values / totals.clamp_min(epsilon), uniform)
    probabilities = probabilities.clamp_min(epsilon)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def divergence_curves(
    support_attention: torch.Tensor,
    source_distribution: torch.Tensor,
    top_k: int,
    epsilon: float,
) -> dict[str, np.ndarray]:
    attention = torch.as_tensor(support_attention).detach().cpu()
    source = torch.as_tensor(source_distribution).detach().cpu()
    if attention.ndim != 2 or source.ndim != 2:
        raise ValueError(
            "Expected [layers, support] tensors, got "
            f"attention={tuple(attention.shape)}, source={tuple(source.shape)}."
        )
    if attention.shape != source.shape:
        raise ValueError(
            f"Attention/source shape mismatch: {tuple(attention.shape)} vs {tuple(source.shape)}."
        )
    if attention.shape[-1] <= 0:
        raise ValueError("Support dimension must be non-empty.")

    attention = torch.nan_to_num(
        attention.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    source = torch.nan_to_num(
        source.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    k = min(int(top_k), int(attention.shape[-1]))
    indices = torch.topk(attention, k=k, dim=-1, largest=True, sorted=False).indices
    attention_tk = torch.gather(attention, dim=-1, index=indices)
    source_tk = torch.gather(source, dim=-1, index=indices)

    attention_prob = smooth_probability(attention_tk, epsilon)
    source_prob = smooth_probability(source_tk, epsilon)
    midpoint = 0.5 * (attention_prob + source_prob)

    kl_attention_source = torch.sum(
        attention_prob * (torch.log(attention_prob) - torch.log(source_prob)), dim=-1
    )
    kl_source_attention = torch.sum(
        source_prob * (torch.log(source_prob) - torch.log(attention_prob)), dim=-1
    )
    js = 0.5 * torch.sum(
        attention_prob * (torch.log(attention_prob) - torch.log(midpoint)), dim=-1
    ) + 0.5 * torch.sum(
        source_prob * (torch.log(source_prob) - torch.log(midpoint)), dim=-1
    )
    return {
        "js": js.numpy(),
        "kl_attention_source": kl_attention_source.numpy(),
        "kl_source_attention": kl_source_attention.numpy(),
    }


def process_model(
    model: str,
    features_path: Path,
    scopes: list[str],
    top_k: int,
    epsilon: float,
) -> tuple[dict[str, dict[str, dict[int, dict[str, np.ndarray | int]]]], dict[str, int]]:
    print(f"Loading {MODEL_NAMES[model]}: {features_path}", flush=True)
    features = load_pkl(features_path)
    if not isinstance(features, list):
        raise TypeError(f"Expected a list in {features_path}, got {type(features).__name__}.")

    accumulators = {
        scope: {
            metric: {label: RunningLayerStats() for label in LABEL_NAMES}
            for metric in METRIC_ORDER
        }
        for scope in scopes
    }
    label_counts = {label: 0 for label in LABEL_NAMES}
    for row in features:
        label = int(row.get("label", -1))
        if label not in LABEL_NAMES:
            continue
        label_counts[label] += 1
        for scope in scopes:
            attention_key = f"dgst_t_{scope}_support_attention_per_layer"
            source_key = f"dgst_t_{scope}_source_dist_per_layer"
            if attention_key not in row or source_key not in row:
                raise KeyError(
                    f"{model} feature row is missing {attention_key!r} or {source_key!r}."
                )
            curves = divergence_curves(
                row[attention_key], row[source_key], top_k=top_k, epsilon=epsilon
            )
            for metric, values in curves.items():
                accumulators[scope][metric][label].update(values)

    finalized = {
        scope: {
            metric: {
                label: accumulators[scope][metric][label].finalize()
                for label in LABEL_NAMES
            }
            for metric in METRIC_ORDER
        }
        for scope in scopes
    }
    print(
        f"Processed {MODEL_NAMES[model]}: rows={len(features)}, "
        f"hall={label_counts[0]}, non_hall={label_counts[1]}",
        flush=True,
    )
    del features
    gc.collect()
    return finalized, label_counts


def aggregate_rows(
    all_stats: dict[str, dict[str, dict[str, dict[int, dict[str, np.ndarray | int]]]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, model_stats in all_stats.items():
        for scope, scope_stats in model_stats.items():
            for metric, metric_stats in scope_stats.items():
                for label, stats in metric_stats.items():
                    mean = np.asarray(stats["mean"], dtype=np.float64)
                    std = np.asarray(stats["std"], dtype=np.float64)
                    sem = np.asarray(stats["sem"], dtype=np.float64)
                    for layer_idx in range(mean.size):
                        rows.append(
                            {
                                "model": model,
                                "model_display": MODEL_NAMES[model],
                                "scope": scope,
                                "scope_display": SCOPE_NAMES[scope],
                                "metric": metric,
                                "metric_display": METRIC_NAMES[metric],
                                "layer": layer_idx + 1,
                                "label": label,
                                "label_display": LABEL_NAMES[label],
                                "count": int(stats["count"]),
                                "mean": float(mean[layer_idx]),
                                "std": float(std[layer_idx]),
                                "sem": float(sem[layer_idx]),
                            }
                        )
    return rows


def summary_rows(
    all_stats: dict[str, dict[str, dict[str, dict[int, dict[str, np.ndarray | int]]]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, model_stats in all_stats.items():
        for scope, scope_stats in model_stats.items():
            for metric, metric_stats in scope_stats.items():
                hall = np.asarray(metric_stats[0]["mean"], dtype=np.float64)
                non = np.asarray(metric_stats[1]["mean"], dtype=np.float64)
                diff = hall - non
                peak_idx = int(np.argmax(np.abs(diff)))
                rows.append(
                    {
                        "model": model,
                        "model_display": MODEL_NAMES[model],
                        "scope": scope,
                        "scope_display": SCOPE_NAMES[scope],
                        "metric": metric,
                        "metric_display": METRIC_NAMES[metric],
                        "hall_count": int(metric_stats[0]["count"]),
                        "non_hall_count": int(metric_stats[1]["count"]),
                        "hall_layer_average": float(hall.mean()),
                        "non_hall_layer_average": float(non.mean()),
                        "hall_minus_non_average": float(diff.mean()),
                        "peak_abs_gap_layer": peak_idx + 1,
                        "peak_hall_minus_non_gap": float(diff[peak_idx]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scope(
    output_dir: Path,
    scope: str,
    models: list[str],
    all_stats: dict[str, dict[str, dict[str, dict[int, dict[str, np.ndarray | int]]]]],
    top_k: int,
) -> None:
    fig, axes = plt.subplots(
        len(METRIC_ORDER),
        len(models),
        figsize=(5.4 * len(models), 3.7 * len(METRIC_ORDER)),
        sharey="row",
        squeeze=False,
    )
    for row_idx, metric in enumerate(METRIC_ORDER):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            metric_stats = all_stats[model][scope][metric]
            for label in LABEL_NAMES:
                stats = metric_stats[label]
                mean = np.asarray(stats["mean"], dtype=np.float64)
                sem = np.asarray(stats["sem"], dtype=np.float64)
                layers = np.arange(1, mean.size + 1)
                ax.plot(
                    layers,
                    mean,
                    color=LABEL_COLORS[label],
                    linewidth=2.0,
                    label=f"{LABEL_NAMES[label]} (n={stats['count']})",
                )
                ax.fill_between(
                    layers,
                    mean - sem,
                    mean + sem,
                    color=LABEL_COLORS[label],
                    alpha=0.18,
                    linewidth=0,
                )
            if row_idx == 0:
                ax.set_title(MODEL_NAMES[model], fontsize=12, fontweight="semibold")
            if col_idx == 0:
                ax.set_ylabel(METRIC_NAMES[metric])
            if row_idx == len(METRIC_ORDER) - 1:
                ax.set_xlabel("Decoder layer")
            ax.set_xlim(1, int(np.asarray(metric_stats[0]["mean"]).size))
            ax.grid(True, alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle(
        f"{SCOPE_NAMES[scope]}: attention Top-{top_k} vs source distribution",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    stem = output_dir / f"{scope}_attention_topk{top_k}_source_js_kl_by_label"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    summaries: list[dict[str, object]],
    models: list[str],
    scopes: list[str],
    label_counts: dict[str, dict[int, int]],
    top_k: int,
    epsilon: float,
) -> None:
    lines = [
        "# COCO500 Attention-TK / Source JS-KL 曲线摘要",
        "",
        "## 计算口径",
        "",
        f"- 每层由原始 `support_attention` 选择 Top-{top_k}，不使用 semantic gate。",
        "- Attention 与 source distribution 截取相同 TK 位置后，各自在区域内做 L1 归一化。",
        f"- JS 与双向 KL 使用自然对数；概率下限为 `{epsilon:g}`。",
        "- `VV` 使用 visual support，`VP` 使用 visual+prompt support。",
        "- 曲线按 object-token row 分组求均值，图中阴影为 mean ± SEM；`H-N` 表示幻觉减非幻觉。",
        "",
        "## 样本数",
        "",
    ]
    for model in models:
        lines.append(
            f"- {MODEL_NAMES[model]}：hall={label_counts[model][0]}，"
            f"non-hall={label_counts[model][1]}"
        )
    for scope in scopes:
        lines.extend(
            [
                "",
                f"## {SCOPE_NAMES[scope]}",
                "",
                "| Model | Metric | Hall avg | Non-hall avg | H-N avg | Peak | Peak H-N gap |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in summaries:
            if item["scope"] != scope:
                continue
            lines.append(
                "| {model_display} | {metric_display} | {hall_layer_average:.6f} | "
                "{non_hall_layer_average:.6f} | {hall_minus_non_average:+.6f} | "
                "L{peak_abs_gap_layer} | {peak_hall_minus_non_gap:+.6f} |".format(**item)
            )
    path.write_text("\n".join(lines) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be positive.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models contains duplicates.")
    if len(set(args.scopes)) != len(args.scopes):
        raise ValueError("--scopes contains duplicates.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    all_stats = {}
    label_counts = {}
    for model in args.models:
        features_path = args.input_root / model / "COCO500-mass-dist-topk" / "features.pkl"
        if not features_path.is_file():
            raise FileNotFoundError(features_path)
        all_stats[model], label_counts[model] = process_model(
            model=model,
            features_path=features_path,
            scopes=args.scopes,
            top_k=args.top_k,
            epsilon=args.epsilon,
        )

    layer_rows = aggregate_rows(all_stats)
    summaries = summary_rows(all_stats)
    write_csv(
        args.output_dir / f"attention_topk{args.top_k}_source_js_kl_layerwise.csv",
        layer_rows,
    )
    write_csv(
        args.output_dir / f"attention_topk{args.top_k}_source_js_kl_summary.csv",
        summaries,
    )
    for scope in args.scopes:
        plot_scope(args.output_dir, scope, args.models, all_stats, args.top_k)

    metadata = {
        "top_k": args.top_k,
        "epsilon": args.epsilon,
        "log_base": "natural",
        "region_selector": "raw support_attention Top-K",
        "local_normalization": "attention and source independently L1-normalized on the same attention Top-K",
        "models": args.models,
        "scopes": args.scopes,
        "label_semantics": {"0": "hallucination", "1": "non-hallucination"},
        "label_counts": {model: {str(k): v for k, v in counts.items()} for model, counts in label_counts.items()},
        "summary": summaries,
    }
    (args.output_dir / f"attention_topk{args.top_k}_source_js_kl_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_summary(
        args.output_dir / "summary.md",
        summaries=summaries,
        models=args.models,
        scopes=args.scopes,
        label_counts=label_counts,
        top_k=args.top_k,
        epsilon=args.epsilon,
    )
    print(f"Wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
