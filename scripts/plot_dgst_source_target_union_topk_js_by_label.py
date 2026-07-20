#!/usr/bin/env python3
"""Plot source/target JS on the DGST union-topK support by CHAIR label."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_dgst_source_target_divergence_by_label import BRANCHES
from train_feature_sets import feature_block
from utils.io_utils import load_pkl


HALLUCINATION = 0
REAL = 1


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
    stem = args.name or f"{args.model}_dgst_source_target_union_topk_js_by_label"

    rows = load_pkl(str(features_path))
    stats, transport_top_ks = collect_stats(rows)
    csv_path = results_dir / f"{stem}.csv"
    png_path = results_dir / f"{stem}.png"
    pdf_path = results_dir / f"{stem}.pdf"
    md_path = results_dir / f"{stem}.md"
    write_csv(stats, csv_path)
    plot_curves(stats, transport_top_ks, png_path, pdf_path)
    write_markdown(
        args.model,
        features_path,
        stats,
        transport_top_ks,
        csv_path,
        png_path,
        md_path,
    )

    print(f"markdown {md_path.resolve()}")
    print(f"csv {csv_path.resolve()}")
    print(f"png {png_path.resolve()}")
    print(f"pdf {pdf_path.resolve()}")
    for branch, _ in BRANCHES:
        hall = stats[branch][HALLUCINATION]["mean"]
        real = stats[branch][REAL]["mean"]
        gap = hall - real
        peak = int(np.argmax(np.abs(gap)))
        print(
            f"{branch}: peak_layer={peak + 1} "
            f"hall_minus_real={gap[peak]:.8f}"
        )


def collect_stats(rows: list[dict]) -> tuple[dict, tuple[int, ...]]:
    values = {
        branch: {HALLUCINATION: [], REAL: []}
        for branch, _ in BRANCHES
    }
    transport_top_ks = set()
    for row in rows:
        label = row.get("label")
        if label not in (HALLUCINATION, REAL):
            continue
        transport_top_ks.add(int(row.get("dgst_t_transport_top_k", 32)))
        for branch, _ in BRANCHES:
            curve = feature_block(
                row, f"{branch}_source_target_union_topk_js"
            )
            values[branch][int(label)].append(curve)

    if not transport_top_ks:
        raise ValueError("No binary-labeled DGST rows were found.")
    stats = {}
    for branch, _ in BRANCHES:
        stats[branch] = {}
        for label in (HALLUCINATION, REAL):
            curves = values[branch][label]
            if not curves:
                raise ValueError(f"No rows for branch={branch}, label={label}")
            matrix = np.stack(curves).astype(np.float64, copy=False)
            stats[branch][label] = {
                "n": int(matrix.shape[0]),
                "mean": matrix.mean(axis=0),
                "sem": matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
            }
    return stats, tuple(sorted(transport_top_ks))


def plot_curves(
    stats: dict,
    transport_top_ks: tuple[int, ...],
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
    colors = {HALLUCINATION: "#d55e00", REAL: "#0072b2"}
    names = {HALLUCINATION: "Hallucination", REAL: "Real / non-hallucination"}
    for ax, (branch, title) in zip(axes.reshape(-1), BRANCHES):
        item = stats[branch]
        layers = np.arange(1, len(item[HALLUCINATION]["mean"]) + 1)
        for label in (HALLUCINATION, REAL):
            mean = item[label]["mean"]
            sem = item[label]["sem"]
            ax.plot(
                layers,
                mean,
                color=colors[label],
                linewidth=2,
                label=f"{names[label]} (n={item[label]['n']})",
            )
            ax.fill_between(
                layers,
                mean - sem,
                mean + sem,
                color=colors[label],
                alpha=0.18,
                linewidth=0,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Union-topK Jensen-Shannon divergence")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(1, len(layers))
        ax.legend(frameon=False, fontsize=8)
    topk_text = ", ".join(str(value) for value in transport_top_ks)
    fig.suptitle(
        "Target vs source JS within DGST union-topK support "
        f"(transport K={topk_text})"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "branch",
            "layer",
            "n_hallucination",
            "n_real",
            "hallucination_mean",
            "hallucination_sem",
            "real_mean",
            "real_sem",
            "hallucination_minus_real",
        ])
        for branch, _ in BRANCHES:
            hall = stats[branch][HALLUCINATION]
            real = stats[branch][REAL]
            for index in range(len(hall["mean"])):
                writer.writerow([
                    branch,
                    index + 1,
                    hall["n"],
                    real["n"],
                    f"{hall['mean'][index]:.10g}",
                    f"{hall['sem'][index]:.10g}",
                    f"{real['mean'][index]:.10g}",
                    f"{real['sem'][index]:.10g}",
                    f"{hall['mean'][index] - real['mean'][index]:.10g}",
                ])


def write_markdown(
    model: str,
    features_path: Path,
    stats: dict,
    transport_top_ks: tuple[int, ...],
    csv_path: Path,
    png_path: Path,
    path: Path,
) -> None:
    topk_text = ", ".join(str(value) for value in transport_top_ks)
    lines = [
        f"# {model} 联合 top-K 区域 source-target JS 分标签曲线",
        "",
        f"- transport top-K 总预算：`{topk_text}`。每层分别取 source/target 的 `K//2` 个 token 后求并集。",
        "- source 和对应分支 target 在联合区域内分别重新归一化，再计算自然对数 JS。",
        "- Gaussian target = normalize(attention × gate)；direct 使用保存的 target-dist；raw-attention 使用 attention。",
        "- 标签：`0=hallucination`，`1=real/non-hallucination`；阴影为 SEM。",
        f"- 输入：`{features_path}`。",
        f"- 逐层数据：`{csv_path.name}`。",
        f"- 图片：`{png_path.name}`。",
        "",
        "| 分支 | Hall 全层均值 | Real 全层均值 | 最大绝对差层 | 该层 Hall-Real |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch, title in BRANCHES:
        hall = stats[branch][HALLUCINATION]["mean"]
        real = stats[branch][REAL]["mean"]
        gap = hall - real
        peak = int(np.argmax(np.abs(gap)))
        lines.append(
            f"| {title} | {hall.mean():.6f} | {real.mean():.6f} | "
            f"{peak + 1} | {gap[peak]:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
