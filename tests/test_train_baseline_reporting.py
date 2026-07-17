from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_baselines import (  # noqa: E402
    _write_baseline_markdown,
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
        "val_metrics": _metrics(value - 0.1),
        "test_metrics": _metrics(value),
    }
    return {
        "model": "qwen3_vl_8b",
        "seed": seed,
        "configured_methods": ["metatoken", "svar"],
        "stored_label_semantics": {"0": "hallucination", "1": "real"},
        "detector_target_semantics": {"0": "real", "1": "hallucination"},
        "label_protocol": "test",
        "counts": {"train": 80, "val": 10, "test": 10},
        "image_split_counts": {"train": 8, "val": 1, "test": 1},
        "methods": {
            "metatoken": {"lr": result},
            "svar": result,
        },
    }


class BaselineReportingTests(unittest.TestCase):
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
        self.assertIn("各随机种子的 Test Real F1", text)
        self.assertIn("seed 42", text)

    def test_rejects_duplicate_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            aggregate_baseline_outputs([_output(42, 0.6), _output(42, 0.7)])


if __name__ == "__main__":
    unittest.main()
