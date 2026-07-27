from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_feature_set_results import _to_markdown
from scripts.summarize_torch_probe_seed_runs import (
    CLASS_METRICS,
    _validate_seed_metadata,
    _write_markdown,
)


def _class_metrics(value: float) -> dict:
    return {
        "precision": value,
        "recall": value + 0.01,
        "f1": value + 0.02,
        "auc": value + 0.03,
        "aupr": value + 0.04,
    }


class TorchProbeReportingTests(unittest.TestCase):
    def test_single_seed_table_places_real_and_hallucination_side_by_side(self) -> None:
        row = {
            "feature_set": "risk+ev",
            "classifier": "torch_probe",
            "accuracy": 0.8,
            "dual_class_metrics": True,
        }
        for prefix, values in (
            ("real", _class_metrics(0.8)),
            ("hallucination", _class_metrics(0.5)),
        ):
            for metric, value in values.items():
                row[f"{prefix}_{metric}"] = value

        markdown = _to_markdown([row], dual_class=True)
        self.assertIn("Real F1", markdown)
        self.assertIn("Hall. F1", markdown)
        self.assertIn("| risk+ev | torch_probe | 0.800 |", markdown)

    def test_three_seed_summary_requires_real_headline_and_both_classes(self) -> None:
        metrics = {}
        for seed in (43, 44, 45):
            payload = {
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "accuracy": 0.8,
                "auc": 0.8,
                "aupr": 0.8,
                "reported_positive_class": "real",
                "real_positive": _class_metrics(0.8),
                "hallucination_positive": _class_metrics(0.5),
                "best_params": {"seed": seed},
                "split_protocol": "strict_82_no_validation",
                "checkpoint_selection": "minimum_train_loss",
                "threshold_selection": "train_f1",
                "threshold_reporting": ["fixed_0.5", "train_f1"],
            }
            payload["threshold_reports"] = {
                mode: {
                    "threshold": 0.5 if mode == "fixed_0.5" else 0.4,
                    "test_metrics": {
                        key: value
                        for key, value in payload.items()
                        if key in {
                            "precision", "recall", "f1", "accuracy", "auc",
                            "aupr", "reported_positive_class",
                            "real_positive", "hallucination_positive",
                        }
                    },
                }
                for mode in ("fixed_0.5", "train_f1")
            }
            metrics[seed] = payload
        _validate_seed_metadata("model", "risk+ev", metrics)

        metrics[45]["reported_positive_class"] = "hallucination"
        with self.assertRaisesRegex(ValueError, "must be real-positive"):
            _validate_seed_metadata("model", "risk+ev", metrics)

    def test_three_seed_markdown_reports_real_headline_and_hall_metrics(self) -> None:
        base_row = {
            "model": "model",
            "model_label": "Model",
            "feature_set": "risk+ev",
            "rank": 1,
            "threshold_mean": 0.5,
            "threshold_std": 0.01,
            "accuracy_mean": 0.8,
            "accuracy_std": 0.01,
        }
        for prefix, base in (
            ("real_positive", 0.8),
            ("hallucination_positive", 0.5),
        ):
            for offset, metric in enumerate(CLASS_METRICS):
                base_row[f"{prefix}_{metric}_mean"] = base + offset * 0.01
                base_row[f"{prefix}_{metric}_std"] = 0.01
        rows = [
            {**base_row, "threshold_mode": "fixed_0.5"},
            {**base_row, "threshold_mode": "train_f1"},
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_markdown(path, "Summary", [43, 44, 45], rows)
            markdown = path.read_text(encoding="utf-8")

        self.assertIn("Real is the headline positive class", markdown)
        self.assertIn("Real F1", markdown)
        self.assertIn("Hall. F1", markdown)


if __name__ == "__main__":
    unittest.main()
