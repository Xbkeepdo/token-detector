from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_baselines import (  # noqa: E402
    _build_threshold_reports,
    _configured_baseline_trainers,
    _normalize_baseline_trainer,
    _run_shared_mlp_protocol,
    _run_training_protocol,
    _shared_mlp_baseline_matrix,
    _shared_probe_config,
    _write_baseline_markdown,
    _write_trainer_comparison,
    aggregate_baseline_outputs,
)


def _metrics(value: float) -> dict:
    return {
        "accuracy": value,
        "hallucination_positive": {
            "precision": value,
            "recall": value,
            "f1": value,
            "auc": value,
            "aupr": value,
        },
        "real_positive": {
            "precision": value + 0.05,
            "recall": value + 0.05,
            "f1": value + 0.05,
            "auc": value + 0.05,
            "aupr": value + 0.05,
        },
    }


def _output(seed: int, value: float) -> dict:
    result = {
        "train_metrics": _metrics(value - 0.1),
        "test_metrics": _metrics(value),
    }
    return {
        "model": "qwen3_vl_8b",
        "seed": seed,
        "configured_methods": ["metatoken", "svar"],
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "label_protocol": "test",
        "counts": {"train": 80, "val": 0, "test": 20},
        "image_split_counts": {"train": 8, "val": 0, "test": 2},
        "split_protocol": "strict_82_no_validation",
        "checkpoint_selection": "last_epoch",
        "threshold_selection": "train_f1",
        "methods": {
            "metatoken": {"lr": result},
            "svar": result,
        },
    }


class BaselineReportingTests(unittest.TestCase):
    def test_config_can_run_native_and_yaml_mlp_together(self) -> None:
        args = argparse.Namespace(trainer=None, trainers=None)
        self.assertEqual(
            _configured_baseline_trainers(
                args,
                {"trainers": ["native_paper", "shared_torch_mlp"]},
            ),
            ("native_paper", "shared_torch_mlp"),
        )
        args.trainers = ["shared_torch_mlp", "shared_torch_mlp"]
        self.assertEqual(
            _configured_baseline_trainers(args, {}),
            ("shared_torch_mlp",),
        )
        self.assertIn(
            "trainer_namespace",
            inspect.signature(_run_training_protocol).parameters,
        )
        self.assertNotIn(
            "trainer_namespace",
            inspect.signature(_run_shared_mlp_protocol).parameters,
        )

    def test_native_head_reports_same_dual_class_metrics_and_thresholds(self) -> None:
        reports = _build_threshold_reports(
            train_labels=[0, 1, 0, 1],
            train_scores=[0.9, 0.1, 0.8, 0.2],
            test_labels=[0, 1, 0, 1],
            test_scores=[0.95, 0.05, 0.75, 0.25],
            selected_threshold=0.7,
            positive_class="hallucination",
        )
        self.assertEqual(set(reports), {"fixed_0.5", "train_f1"})
        for report in reports.values():
            metrics = report["test_metrics"]
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["real_positive"]["f1"], 1.0)
            self.assertEqual(metrics["hallucination_positive"]["f1"], 1.0)

    def test_writes_native_vs_yaml_mlp_comparison(self) -> None:
        outputs = [_output(43, 0.7), _output(44, 0.8)]
        for output in outputs:
            for result in (
                output["methods"]["metatoken"]["lr"],
                output["methods"]["svar"],
            ):
                result["threshold_reports"] = {
                    mode: {
                        "threshold": threshold,
                        "train_metrics": result["train_metrics"],
                        "test_metrics": result["test_metrics"],
                    }
                    for mode, threshold in (
                        ("fixed_0.5", 0.5),
                        ("train_f1", 0.4),
                    )
                }
        native = aggregate_baseline_outputs(outputs)
        native["trainer"] = "native_paper"
        shared = aggregate_baseline_outputs(outputs)
        shared["trainer"] = "shared_torch_mlp"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_trainer_comparison(
                result_root=root,
                result_stem="test_baselines",
                trainer_runs={
                    "native_paper": (native, [Path("native43.json")]),
                    "shared_torch_mlp": (shared, [Path("shared43.json")]),
                },
                probe_cfg={
                    "hidden_sizes": [128, 64, 32],
                    "batch_norm": True,
                    "dropout": 0.3,
                    "drop_last": True,
                },
            )
            output_dir = root / "results" / "comparison"
            markdown = next(output_dir.glob("*_summary.md")).read_text(
                encoding="utf-8"
            )
            payload = next(output_dir.glob("*.json")).read_text(
                encoding="utf-8"
            )
        self.assertIn("Baseline 原方法", markdown)
        self.assertIn("YAML 三层 MLP", markdown)
        self.assertIn("native_paper_vs_yaml_shared_three_layer_mlp", payload)

    def test_shared_mlp_trainer_aliases_and_three_layer_config(self) -> None:
        self.assertEqual(
            _normalize_baseline_trainer("3layer_mlp"),
            "shared_torch_mlp",
        )
        config = _shared_probe_config(
            {"hidden_sizes": [128, 64, 32], "max_epochs": 7},
            seed=44,
            positive_class="real",
        )
        self.assertEqual(config.hidden_sizes, (128, 64, 32))
        self.assertEqual(config.num_epochs, 7)
        self.assertEqual(config.seed, 44)
        with self.assertRaisesRegex(ValueError, "exactly three"):
            _shared_probe_config(
                {"hidden_sizes": [128, 64]},
                seed=44,
                positive_class="real",
            )

    def test_shared_mlp_dense_vectors_include_projectaway_curve(self) -> None:
        records = []
        for index, label in enumerate((0, 1)):
            records.append({
                "baseline_schema_version": "1.0",
                "image_id": index,
                "response_token_idx": 0,
                "label": label,
                "baselines": {
                    "metatoken": {"vector": [1.0, 2.0]},
                    "svar": {"vector": [3.0, 4.0, 5.0]},
                    "projectaway": {
                        "internal_confidence": 0.8,
                        "per_layer_internal_confidence": [0.1, 0.2, 0.3],
                    },
                },
            })
        metatoken, labels = _shared_mlp_baseline_matrix(records, "metatoken")
        svar, _ = _shared_mlp_baseline_matrix(records, "svar")
        projectaway, _ = _shared_mlp_baseline_matrix(records, "projectaway")
        self.assertEqual(metatoken.shape, (2, 2))
        self.assertEqual(svar.shape, (2, 3))
        self.assertEqual(projectaway.shape, (2, 4))
        np.testing.assert_allclose(projectaway[0], [0.8, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(labels, [0, 1])

    def test_shared_mlp_aggregate_and_markdown_report_both_thresholds(self) -> None:
        outputs = [_output(seed, value) for seed, value in zip(
            (43, 44, 45), (0.6, 0.7, 0.8)
        )]
        for output in outputs:
            output["checkpoint_selection"] = "minimum_train_loss"
            output["threshold_reporting"] = ["fixed_0.5", "train_f1"]
            for result in (
                output["methods"]["metatoken"]["lr"],
                output["methods"]["svar"],
            ):
                result["threshold_reports"] = {
                    "fixed_0.5": {
                        "threshold": 0.5,
                        "train_metrics": result["train_metrics"],
                        "test_metrics": result["test_metrics"],
                    },
                    "train_f1": {
                        "threshold": 0.4,
                        "train_metrics": result["train_metrics"],
                        "test_metrics": result["test_metrics"],
                    },
                }
        summary = aggregate_baseline_outputs(outputs)
        self.assertEqual(
            summary["methods"]["svar"]["threshold_reports"]
            ["fixed_0.5"]["threshold"]["mean"],
            0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_baseline_markdown(
                path,
                summary,
                source_paths=[Path(f"seed{seed}.json") for seed in (43, 44, 45)],
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("固定阈值 0.5", text)
        self.assertIn("Train Real-F1 搜索阈值", text)

    def test_three_seed_population_mean_std_and_markdown(self) -> None:
        outputs = [
            _output(42, 0.6),
            _output(43, 0.7),
            _output(44, 0.8),
        ]
        summary = aggregate_baseline_outputs(outputs)
        self.assertEqual(summary["seeds"], [42, 43, 44])
        self.assertEqual(summary["num_seeds"], 3)
        stats = summary["methods"]["svar"]["test_metrics"]["f1"]
        self.assertEqual(summary["headline_positive_class"], "real")
        self.assertAlmostEqual(
            summary["methods"]["svar"]["test_metrics"]
            ["hallucination_positive"]["aupr"]["mean"],
            0.7,
        )
        self.assertAlmostEqual(
            summary["methods"]["svar"]["test_metrics"]
            ["real_positive"]["aupr"]["mean"],
            0.75,
        )
        self.assertAlmostEqual(stats["mean"], 0.75)
        self.assertAlmostEqual(stats["std"], (2 / 300) ** 0.5)
        self.assertEqual(stats["values"], [0.65, 0.75, 0.8500000000000001])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_baseline_markdown(
                path,
                summary,
                source_paths=[Path(f"seed{seed}.json") for seed in (42, 43, 44)],
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("0.7500 ± 0.0816", text)
        self.assertIn("headline 正类：real", text)
        self.assertIn("严格 8:2 无验证集", text)
        self.assertIn("各随机种子的 Test Real F1", text)
        self.assertIn("seed 42", text)

    def test_rejects_duplicate_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_baseline_outputs([_output(42, 0.6), _output(42, 0.7)])


if __name__ == "__main__":
    unittest.main()
