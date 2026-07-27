#!/usr/bin/env python3
"""Compute full-distribution source/target KL and JS curves by CHAIR label."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import load_pkl


HALLUCINATION = 0
REAL = 1
EPS = 1.0e-12

BRANCHES = (
    ("hpre_raw_logit_gauss", "hpre raw-logit Gaussian"),
    ("hpre_softmax_prob_gauss", "hpre softmax-prob Gaussian"),
    ("hmid_raw_logit_gauss", "hmid raw-logit Gaussian"),
    ("hmid_softmax_prob_gauss", "hmid softmax-prob Gaussian"),
    ("hpre_softmax_prob_direct", "hpre direct target probability"),
    ("raw_attention", "raw attention"),
)

METRICS = (
    ("kl_target_source", "KL(target || source)"),
    ("kl_source_target", "KL(source || target)"),
    ("js", "Jensen-Shannon divergence"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    features_path = output_dir / "features.pkl"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{args.model}_dgst_source_target_divergence_by_label"

    rows = load_pkl(str(features_path))
    stats = collect_stats(rows)

    csv_path = results_dir / f"{stem}.csv"
    md_path = results_dir / f"{stem}.md"
    write_csv(stats, csv_path)
    figure_paths = plot_metrics(stats, results_dir, stem)
    write_markdown(args.model, features_path, stats, csv_path, figure_paths, md_path)

    print(f"markdown {md_path.resolve()}")
    print(f"csv {csv_path.resolve()}")
    for path in figure_paths:
        print(f"figure {path.resolve()}")
    for branch, title in BRANCHES:
        for metric, _ in METRICS:
            item = stats[branch][metric]
            gap = item[HALLUCINATION]["mean"] - item[REAL]["mean"]
            peak = int(np.argmax(np.abs(gap)))
            print(
                f"{branch} {metric}: peak_layer={peak + 1} "
                f"hall_minus_real={gap[peak]:.8f}"
            )


def collect_stats(rows: list[dict]) -> dict:
    values = {
        branch: {
            metric: {HALLUCINATION: [], REAL: []}
            for metric, _ in METRICS
        }
        for branch, _ in BRANCHES
    }
    used = 0
    for row in rows:
        label = row.get("label")
        if label not in (HALLUCINATION, REAL):
            continue
        source = _matrix(row, "dgst_t_source_dist_per_layer")
        attention = _matrix(row, "dgst_t_attention_support_per_layer")
        if source.shape != attention.shape:
            raise ValueError(
                f"source/attention shape mismatch: {source.shape} != {attention.shape}"
            )
        used += 1
        for branch, _ in BRANCHES:
            target = target_distribution(row, branch, attention)
            divergence = divergences(source, target)
            for metric, _ in METRICS:
                values[branch][metric][int(label)].append(divergence[metric])

    if not used:
        raise ValueError("No binary-labeled DGST rows were found.")

    stats = {}
    for branch, _ in BRANCHES:
        stats[branch] = {}
        for metric, _ in METRICS:
            stats[branch][metric] = {}
            for label in (HALLUCINATION, REAL):
                curves = values[branch][metric][label]
                if not curves:
                    raise ValueError(f"No rows for branch={branch}, metric={metric}, label={label}")
                matrix = np.stack(curves).astype(np.float64, copy=False)
                stats[branch][metric][label] = {
                    "n": int(matrix.shape[0]),
                    "mean": matrix.mean(axis=0),
                    "sem": matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
                }
    return stats


def target_distribution(row: dict, branch: str, attention: np.ndarray) -> np.ndarray:
    if branch == "raw_attention":
        return normalize(attention)
    if branch == "hpre_softmax_prob_direct":
        return normalize(
            _matrix(row, "dgst_t_hpre_softmax_prob_direct_target_dist_per_layer")
        )
    gate = _matrix(row, f"dgst_t_{branch}_gate_per_layer")
    if gate.shape != attention.shape:
        raise ValueError(f"{branch} gate shape mismatch: {gate.shape} != {attention.shape}")
    return normalize(attention * gate)


def divergences(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source = smooth(source)
    target = smooth(target)
    midpoint = smooth(0.5 * (source + target))
    kl_target_source = np.sum(target * (np.log(target) - np.log(source)), axis=1)
    kl_source_target = np.sum(source * (np.log(source) - np.log(target)), axis=1)
    js = 0.5 * np.sum(target * (np.log(target) - np.log(midpoint)), axis=1)
    js += 0.5 * np.sum(source * (np.log(source) - np.log(midpoint)), axis=1)
    return {
        "kl_target_source": kl_target_source,
        "kl_source_target": kl_source_target,
        "js": js,
    }


def _matrix(row: dict, key: str) -> np.ndarray:
    if key not in row:
        raise KeyError(f"Missing required DGST field {key!r}")
    value = np.asarray(row[key], dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"DGST field {key!r} must be [layers, visual_tokens], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"DGST field {key!r} contains non-finite values")
    return value


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = np.maximum(value, 0.0)
    total = value.sum(axis=1, keepdims=True)
    uniform = np.full_like(value, 1.0 / max(value.shape[1], 1))
    return np.divide(value, total, out=uniform, where=total > EPS)


def smooth(value: np.ndarray) -> np.ndarray:
    value = np.maximum(normalize(value), EPS)
    return value / value.sum(axis=1, keepdims=True)


def plot_metrics(stats: dict, results_dir: Path, stem: str) -> list[Path]:
    paths = []
    colors = {HALLUCINATION: "#d55e00", REAL: "#0072b2"}
    label_names = {HALLUCINATION: "Hallucination", REAL: "Real / non-hallucination"}
    for metric, metric_title in METRICS:
        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
        for ax, (branch, branch_title) in zip(axes.reshape(-1), BRANCHES):
            item = stats[branch][metric]
            layer_count = len(item[HALLUCINATION]["mean"])
            layers = np.arange(1, layer_count + 1)
            for label in (HALLUCINATION, REAL):
                mean = item[label]["mean"]
                sem = item[label]["sem"]
                ax.plot(
                    layers,
                    mean,
                    color=colors[label],
                    linewidth=2,
                    label=f"{label_names[label]} (n={item[label]['n']})",
                )
                ax.fill_between(
                    layers, mean - sem, mean + sem,
                    color=colors[label], alpha=0.18, linewidth=0,
                )
            ax.set_title(branch_title)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_title)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(1, layer_count)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Full target-distribution vs source-distribution: {metric_title}")
        fig.tight_layout()
        png = results_dir / f"{stem}_{metric}.png"
        pdf = results_dir / f"{stem}_{metric}.pdf"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        paths.extend((png, pdf))
    return paths


def write_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "branch", "metric", "layer", "n_hallucination", "n_real",
            "hallucination_mean", "hallucination_sem", "real_mean", "real_sem",
            "hallucination_minus_real",
        ])
        for branch, _ in BRANCHES:
            for metric, _ in METRICS:
                hall = stats[branch][metric][HALLUCINATION]
                real = stats[branch][metric][REAL]
                for index in range(len(hall["mean"])):
                    writer.writerow([
                        branch, metric, index + 1, hall["n"], real["n"],
                        f"{hall['mean'][index]:.10g}", f"{hall['sem'][index]:.10g}",
                        f"{real['mean'][index]:.10g}", f"{real['sem'][index]:.10g}",
                        f"{hall['mean'][index] - real['mean'][index]:.10g}",
                    ])


def write_markdown(
    model: str,
    features_path: Path,
    stats: dict,
    csv_path: Path,
    figure_paths: list[Path],
    path: Path,
) -> None:
    lines = [
        f"# {model} target-dist / source-dist 散度分标签曲线",
        "",
        "- 在完整视觉 token 分布上计算，不使用 transport top-k 截断。",
        "- `KL(target||source)`、`KL(source||target)` 和自然对数定义的 JS。",
        "- 概率采用与 DGST 实现一致的非负归一化和 `1e-12` 平滑。",
        "- Gaussian target-dist = normalize(attention × gate)；direct 使用已保存 target-dist；raw-attention 使用 attention。",
        "- 标签：`0=hallucination`，`1=real/non-hallucination`；阴影为 SEM。",
        f"- 输入：`{features_path}`。",
        f"- 逐层数据：`{csv_path.name}`。",
        "- 图片：" + ", ".join(f"`{item.name}`" for item in figure_paths if item.suffix == ".png") + "。",
        "",
        "| 分支 | 指标 | Hall 全层均值 | Real 全层均值 | 最大绝对差层 | 该层 Hall-Real |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for branch, branch_title in BRANCHES:
        for metric, metric_title in METRICS:
            hall = stats[branch][metric][HALLUCINATION]["mean"]
            real = stats[branch][metric][REAL]["mean"]
            gap = hall - real
            peak = int(np.argmax(np.abs(gap)))
            lines.append(
                f"| {branch_title} | {metric_title} | {hall.mean():.6f} | "
                f"{real.mean():.6f} | {peak + 1} | {gap[peak]:.6f} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
