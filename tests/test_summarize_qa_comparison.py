from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.summarize_qa_comparison import (  # noqa: E402
    METRIC_KEYS,
    resolve_output_root,
    build_comparison,
    render_markdown,
    write_comparison,
)


SEEDS = (42, 43, 44)


def _stat(mean: float, std: float = 0.01) -> dict:
    return {"mean": mean, "std": std}


def _probe_method(
    value: float,
    counts: tuple[int, int, int] = (80, 10, 10),
) -> dict:
    result = {
        "seeds": list(SEEDS),
        "counts": dict(
            zip(("train", "val", "test"), counts)
        ),
        "auroc": _stat(value),
        "real_aupr": _stat(value - 0.01),
        "hallucination_aupr": _stat(value - 0.02),
    }
    for label, offset in (("real", 0.0), ("hallucination", -0.1)):
        for metric in ("precision", "recall", "f1"):
            result[f"{label}.{metric}"] = _stat(value + offset)
    return result


def _statistics(values) -> dict:
    mean = sum(values) / len(values)
    std = math.sqrt(
        sum((value - mean) ** 2 for value in values) / len(values)
    )
    return {"mean": mean, "std": std, "values": list(values)}


def _class_metrics(value: float) -> dict:
    return {
        "auc": value,
        "aupr": value,
        "f1": value,
        "precision": value,
        "recall": value,
    }


def _baseline_seed(seed: int, real: float, hall: float) -> dict:
    return {
        "model": "tiny",
        "seed": seed,
        "methods": {
            "projectaway": {
                "test_metrics": {
                    "accuracy": 0.75,
                    "real_positive": _class_metrics(real),
                    "hallucination_positive": _class_metrics(hall),
                }
            }
        },
    }


def _baseline_summary(seed_paths: list[str]) -> dict:
    real_values = [0.7, 0.8, 0.9]
    hall_values = [0.4, 0.5, 0.6]
    return {
        "model": "tiny",
        "seeds": list(SEEDS),
        "num_seeds": 3,
        "headline_positive_class": "real",
        "label_protocol": (
            "qa_answer_correctness_all_0hall_1real_question_probe_split_"
            "physical_image_disjoint"
        ),
        "counts": {"train": 80, "val": 10, "test": 10},
        "image_split_counts": {"train": 8, "val": 1, "test": 1},
        "seed_result_paths": seed_paths,
        "methods": {
            "projectaway": {
                "display_name": "ProjectAway",
                "test_metrics": {
                    "accuracy": _statistics([0.75, 0.75, 0.75]),
                    "precision": _statistics(real_values),
                    "recall": _statistics(real_values),
                    "f1": _statistics(real_values),
                    "auc": _statistics(real_values),
                    "aupr": _statistics(real_values),
                    "other_f1": _statistics(hall_values),
                },
            }
        },
    }


class QAComparisonSummaryTests(unittest.TestCase):
    def _comparison(self, directory: Path) -> dict:
        seed_paths = []
        for seed, real, hall in zip(
            SEEDS, (0.7, 0.8, 0.9), (0.4, 0.5, 0.6)
        ):
            path = directory / f"seed-{seed}.json"
            path.write_text(
                json.dumps(_baseline_seed(seed, real, hall)),
                encoding="utf-8",
            )
            seed_paths.append(str(path))
        probe = {
            "schema_version": 2,
            "positive_class": "real",
            "seeds": list(SEEDS),
            "positions": [
                "prompt_last_token",
                "question_object_pre_token",
            ],
            "image_counts": {"train": 80, "val": 10, "test": 10},
            "coverage": {
                "positions": {
                    "prompt_last_token": {
                        "available": 100,
                        "total": 100,
                        "rate": 1.0,
                    },
                    "question_object_pre_token": {
                        "available": 55,
                        "total": 100,
                        "rate": 0.55,
                    },
                }
            },
            "label_protocols": {
                "answer_correctness_all": {
                    "ads+cgc@prompt_last_token": _probe_method(0.82),
                    (
                        "hpre_softmax_prob_gauss_risk+"
                        "hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine@"
                        "question_object_pre_token"
                    ): _probe_method(0.84, (44, 5, 6)),
                }
            },
        }
        return build_comparison(
            model="tiny",
            dataset="pope",
            label_protocol="answer_correctness_all",
            expected_seeds=SEEDS,
            probe_summary=probe,
            probe_summary_path=directory / "probe.json",
            baseline_summary=_baseline_summary(seed_paths),
            baseline_summary_path=directory / "baseline_3seed.json",
            feature_summary={
                "num_questions": 100,
                "prompt_last_token": 100,
                "question_object_pre_token": 60,
            },
            feature_summary_path=directory / "qa_feature_summary.json",
        )

    def _answer_only_comparison(self, directory: Path) -> dict:
        seed_paths = []
        for seed, real, hall in zip(
            SEEDS, (0.7, 0.8, 0.9), (0.4, 0.5, 0.6)
        ):
            path = directory / f"answer-only-seed-{seed}.json"
            path.write_text(
                json.dumps(_baseline_seed(seed, real, hall)),
                encoding="utf-8",
            )
            seed_paths.append(str(path))
        probe = {
            "schema_version": 2,
            "positive_class": "real",
            "seeds": list(SEEDS),
            "positions": ["prompt_last_token"],
            # Include stale object coverage deliberately: actual result rows,
            # not unrelated coverage metadata, define report positions.
            "coverage": {
                "positions": {
                    "prompt_last_token": {
                        "available": 100,
                        "total": 100,
                        "rate": 1.0,
                    },
                    "question_object_pre_token": {
                        "available": 60,
                        "total": 100,
                        "rate": 0.6,
                    },
                }
            },
            "label_protocols": {
                "answer_correctness_all": {
                    "ads+cgc@prompt_last_token": _probe_method(0.82),
                }
            },
        }
        return build_comparison(
            model="tiny",
            dataset="pope",
            label_protocol="answer_correctness_all",
            expected_seeds=SEEDS,
            probe_summary=probe,
            probe_summary_path=directory / "answer-only-probe.json",
            baseline_summary=_baseline_summary(seed_paths),
            baseline_summary_path=directory / "answer-only-baseline_3seed.json",
            feature_summary={
                "num_questions": 100,
                "prompt_last_token": 100,
                "question_object_pre_token": 60,
            },
            feature_summary_path=directory / "answer-only-feature-summary.json",
        )

    def test_answer_only_report_omits_inactive_object_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            comparison = self._answer_only_comparison(Path(temporary))
            self.assertEqual(
                list(comparison["position_metadata"]),
                ["prompt_last_token"],
            )
            self.assertEqual(
                list(comparison["coverage_metadata"]["feature_extraction"]),
                ["prompt_last_token"],
            )
            self.assertEqual(
                list(
                    comparison["coverage_metadata"]["probe_summary"][
                        "positions"
                    ]
                ),
                ["prompt_last_token"],
            )
            markdown = render_markdown(comparison)
            self.assertIn("prompt_last_token", markdown)
            self.assertNotIn("question_object_pre_token", markdown)
            self.assertNotIn("object word", markdown)

    def test_combines_dual_class_metrics_positions_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = self._comparison(root)
            self.assertEqual(comparison["seeds"], [42, 43, 44])
            self.assertEqual(
                list(comparison["position_metadata"]),
                ["prompt_last_token", "question_object_pre_token"],
            )
            self.assertEqual(
                [row["family"] for row in comparison["rows"]],
                ["ads_cgc", "dgst_method", "native_baseline"],
            )
            object_row = comparison["rows"][1]
            self.assertEqual(
                object_row["position"], "question_object_pre_token"
            )
            self.assertIn(
                "actual position in the question",
                object_row["position_definition"],
            )
            self.assertEqual(object_row["coverage"]["available"], 55)

            baseline = comparison["rows"][2]
            self.assertEqual(baseline["position"], "prompt_last_token")
            self.assertAlmostEqual(
                baseline["metrics"]["real_aupr"]["mean"], 0.8
            )
            self.assertAlmostEqual(
                baseline["metrics"]["hallucination_precision"]["mean"],
                0.5,
            )
            self.assertEqual(
                set(baseline["metrics"]),
                set(METRIC_KEYS),
            )
            self.assertEqual(
                comparison["report_policy"],
                (
                    "read existing test metrics only; report fixed-0.5 and "
                    "train-F1-selected thresholds from the same minimum-train-loss "
                    "checkpoint; no test-set method selection or ranking"
                ),
            )

    def test_writes_markdown_without_test_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = self._comparison(root)
            json_path = root / "comparison.json"
            markdown_path = root / "comparison.md"
            write_comparison(comparison, json_path, markdown_path)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], "qa-comparison-v1")
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Real AUPR", markdown)
            self.assertIn("Hall. AUPR", markdown)
            self.assertIn("object word", markdown)
            self.assertIn("0.8000 ± 0.0816", markdown)
            self.assertNotIn("最高", markdown)
            self.assertNotIn("best method", markdown.lower())
            table_rows = [
                line for line in markdown.splitlines()
                if line.startswith("| native_baseline")
            ]
            self.assertEqual(len(table_rows), 1)
            self.assertEqual(
                table_rows[0].count("|"),
                14,
            )

    def test_rejects_unscoped_or_mismatched_inputs(self) -> None:
        bare = {"ads+cgc@prompt_last_token": _probe_method(0.8)}
        with self.assertRaisesRegex(ValueError, "no scoped methods wrapper"):
            build_comparison(
                model="tiny",
                dataset="pope",
                label_protocol="answer_correctness_all",
                expected_seeds=SEEDS,
                probe_summary=bare,
                probe_summary_path=Path("probe.json"),
                baseline_summary=_baseline_summary([]),
                baseline_summary_path=Path("baseline_3seed.json"),
            )

        scoped = {
            "model": "tiny",
            "dataset": "pope",
            "label_protocol": "object_hallucination_yes_only",
            "seeds": list(SEEDS),
            "methods": {"ads+cgc@prompt_last_token": _probe_method(0.8)},
        }
        with self.assertRaisesRegex(ValueError, "label_protocol"):
            build_comparison(
                model="tiny",
                dataset="pope",
                label_protocol="answer_correctness_all",
                expected_seeds=SEEDS,
                probe_summary=scoped,
                probe_summary_path=Path("probe.json"),
                baseline_summary=_baseline_summary([]),
                baseline_summary_path=Path("baseline_3seed.json"),
            )

    def test_output_root_uses_yaml_unless_cli_overrides(self) -> None:
        config = {
            "qa_benchmarks": {
                "output_root": "/yaml/qa-results",
            }
        }
        self.assertEqual(
            resolve_output_root(config),
            Path("/yaml/qa-results"),
        )
        self.assertEqual(
            resolve_output_root(config, "/cli/qa-results"),
            Path("/cli/qa-results"),
        )
        with self.assertRaisesRegex(ValueError, "output root"):
            resolve_output_root({})


    def test_render_is_deterministic_and_does_not_mutate_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            comparison = self._comparison(Path(temporary))
            before = json.dumps(comparison, sort_keys=True)
            first = render_markdown(comparison)
            second = render_markdown(comparison)
            self.assertEqual(first, second)
            self.assertEqual(before, json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
