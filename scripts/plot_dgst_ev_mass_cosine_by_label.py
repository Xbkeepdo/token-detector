#!/usr/bin/env python3
"""Plot all active DGST EV=target-mass x target-cosine curves by label."""

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


LABEL_HALLUCINATION = 0
LABEL_REAL = 1

FEATURE_SPECS = (
    (
        "hpre_raw_logit_gauss",
        "hpre raw-logit Gaussian",
        "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "hpre_softmax_prob_gauss",
        "hpre softmax-prob Gaussian",
        "dgst_t_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "hmid_raw_logit_gauss",
        "hmid raw-logit Gaussian",
        "dgst_t_hmid_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hmid_per_layer",
    ),
    (
        "hmid_softmax_prob_gauss",
        "hmid softmax-prob Gaussian",
        "dgst_t_hmid_softmax_prob_gauss_ev_target_dist_mass_x_cosine_"
        "topk32_hmid_per_layer",
    ),
    (
        "hpre_softmax_prob_direct",
        "hpre direct target probability",
        "dgst_t_hpre_softmax_prob_direct_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
    (
        "raw_attention",
        "raw attention",
        "dgst_t_raw_attention_ev_target_dist_mass_x_cosine_"
        "topk32_hpre_per_layer",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename stem; defaults to '<model>_dgst_ev_mass_x_cosine_by_label'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    features_path = output_dir / "features.pkl"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    features = load_pkl(str(features_path))
    stats = _collect_stats(features)

    stem = args.name or f"{args.model}_dgst_ev_mass_x_cosine_by_label"
    png_path = results_dir / f"{stem}.png"
    pdf_path = results_dir / f"{stem}.pdf"
    csv_path = results_dir / f"{stem}.csv"
    md_path = results_dir / f"{stem}.md"

    _plot(stats, png_path, pdf_path)
    _write_csv(stats, csv_path)
    _write_markdown(args.model, features_path, stats, png_path, csv_path, md_path)

    print("png", png_path.resolve())
    print("pdf", pdf_path.resolve())
    print("csv", csv_path.resolve())
    print("markdown", md_path.resolve())
    for slug, title, _ in FEATURE_SPECS:
        item = stats[slug]
        gap = item[LABEL_HALLUCINATION]["mean"] - item[LABEL_REAL]["mean"]
        peak = int(np.argmax(np.abs(gap)))
        print(
            f"{slug}: n_hall={item[LABEL_HALLUCINATION]['n']} "
            f"n_real={item[LABEL_REAL]['n']} peak_layer={peak} "
            f"peak_hall_minus_real={gap[peak]:.8f}"
        )


def _collect_stats(features: list[dict]) -> dict[str, dict[int, dict[str, np.ndarray | int]]]:
    curves: dict[str, dict[int, list[np.ndarray]]] = {
        slug: {LABEL_HALLUCINATION: [], LABEL_REAL: []}
        for slug, _, _ in FEATURE_SPECS
    }
    missing_keys = []
    for _, _, key in FEATURE_SPECS:
        if not any(key in row for row in features):
            missing_keys.append(key)
    if missing_keys:
        raise KeyError("Missing EV feature keys: " + ", ".join(missing_keys))

    for row in features:
        label = row.get("label")
        if label not in (LABEL_HALLUCINATION, LABEL_REAL):
            continue
        for slug, _, key in FEATURE_SPECS:
            if key not in row:
                continue
            values = np.asarray(row[key], dtype=np.float32).reshape(-1)
            if values.size:
                curves[slug][int(label)].append(values)

    result = {}
    expected_layers = None
    for slug, _, _ in FEATURE_SPECS:
        result[slug] = {}
        for label in (LABEL_HALLUCINATION, LABEL_REAL):
            values = curves[slug][label]
            if not values:
                raise ValueError(f"Feature {slug!r} has no rows for label={label}.")
            matrix = np.stack(values, axis=0).astype(np.float64, copy=False)
            if expected_layers is None:
                expected_layers = int(matrix.shape[1])
            elif matrix.shape[1] != expected_layers:
                raise ValueError(
                    f"Feature {slug!r} has {matrix.shape[1]} layers; expected {expected_layers}."
                )
            result[slug][label] = {
                "n": int(matrix.shape[0]),
                "mean": matrix.mean(axis=0),
                "sem": matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]),
            }
    return result


def _plot(stats: dict, png_path: Path, pdf_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
    axes = axes.reshape(-1)
    colors = {LABEL_HALLUCINATION: "#d55e00", LABEL_REAL: "#0072b2"}
    labels = {
        LABEL_HALLUCINATION: "Hallucination (label=0)",
        LABEL_REAL: "Non-hallucination / real (label=1)",
    }

    for ax, (slug, title, _) in zip(axes, FEATURE_SPECS):
        item = stats[slug]
        layer_count = len(item[LABEL_HALLUCINATION]["mean"])
        layers = np.arange(layer_count)
        for label in (LABEL_HALLUCINATION, LABEL_REAL):
            mean = item[label]["mean"]
            sem = item[label]["sem"]
            ax.plot(
                layers,
                mean,
                color=colors[label],
                linewidth=2.0,
                label=f"{labels[label]} (n={item[label]['n']})",
            )
            ax.fill_between(
                layers,
                mean - sem,
                mean + sem,
                color=colors[label],
                alpha=0.18,
                linewidth=0,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel("EV = target top-k mass × target cosine")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, layer_count - 1)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "DGST EV (target-distribution top-k mass × target cosine) by label",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(stats: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "layer",
                "n_hallucination",
                "n_real",
                "hallucination_mean",
                "hallucination_sem",
                "real_mean",
                "real_sem",
                "hallucination_minus_real",
            ]
        )
        for slug, _, _ in FEATURE_SPECS:
            item = stats[slug]
            hall = item[LABEL_HALLUCINATION]
            real = item[LABEL_REAL]
            for layer, (hall_mean, hall_sem, real_mean, real_sem) in enumerate(
                zip(hall["mean"], hall["sem"], real["mean"], real["sem"])
            ):
                writer.writerow(
                    [
                        slug,
                        layer,
                        hall["n"],
                        real["n"],
                        f"{hall_mean:.10g}",
                        f"{hall_sem:.10g}",
                        f"{real_mean:.10g}",
                        f"{real_sem:.10g}",
                        f"{hall_mean - real_mean:.10g}",
                    ]
                )


def _write_markdown(
    model: str,
    features_path: Path,
    stats: dict,
    png_path: Path,
    csv_path: Path,
    path: Path,
) -> None:
    lines = [
        f"# {model} EV mass × cosine 分标签曲线",
        "",
        "- 标签：`0=hallucination`，`1=real/non-hallucination`。",
        "- 曲线：每层样本均值；阴影：均值的标准误（SEM）。",
        "- 定义：目标分布 top-k 区域的 mass × 同一区域的 target cosine。",
        f"- 输入特征：`{features_path}`。",
        f"- 总图：`{png_path.name}`。",
        f"- 逐层数值：`{csv_path.name}`。",
        "",
        "| 分支 | 幻觉样本 | 真实样本 | 幻觉全层均值 | 真实全层均值 | 最大绝对差层 | 该层 Hall-Real |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, title, _ in FEATURE_SPECS:
        item = stats[slug]
        hall = item[LABEL_HALLUCINATION]
        real = item[LABEL_REAL]
        gap = hall["mean"] - real["mean"]
        peak = int(np.argmax(np.abs(gap)))
        lines.append(
            f"| {title} | {hall['n']} | {real['n']} | "
            f"{np.mean(hall['mean']):.6f} | {np.mean(real['mean']):.6f} | "
            f"{peak} | {gap[peak]:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
