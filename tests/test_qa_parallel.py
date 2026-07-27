import pickle
import tempfile
import unittest
from pathlib import Path

from features.qa_baseline import QABaselineFeatureStore
from scripts.extract_qa_baselines import (
    _normalize_devices as normalize_baseline_devices,
)
from scripts.extract_qa_baselines import (
    _question_partitions as baseline_question_partitions,
)
from scripts.qa_pipeline import _normalize_devices, _question_partitions


class QAParallelTests(unittest.TestCase):
    def test_device_lists_are_deduplicated_and_partitions_are_disjoint(self):
        self.assertEqual(
            _normalize_devices(["cuda:0", "cuda:1", "cuda:0"], "cpu"),
            ("cuda:0", "cuda:1"),
        )
        self.assertEqual(
            normalize_baseline_devices(None, "cuda:0"),
            ("cuda:0",),
        )
        questions = [{"key": f"q{index}"} for index in range(7)]
        expected = [["q0", "q2", "q4", "q6"], ["q1", "q3", "q5"]]
        for partitioner in (_question_partitions, baseline_question_partitions):
            partitions = partitioner(questions, 2)
            self.assertEqual(
                [[row["key"] for row in partition] for partition in partitions],
                expected,
            )

    def test_baseline_worker_shards_merge_without_filename_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker_zero = QABaselineFeatureStore(
                root,
                shard_size=1,
                part_prefix="part-worker000",
            )
            worker_one = QABaselineFeatureStore(
                root,
                shard_size=1,
                part_prefix="part-worker001",
            )
            worker_zero.add({"key": "q0", "value": 0})
            worker_one.add({"key": "q1", "value": 1})

            parts = sorted((root / "feature_parts").glob("part-*.pkl"))
            self.assertEqual(len(parts), 2)
            self.assertNotEqual(parts[0].name, parts[1].name)

            merged = QABaselineFeatureStore(root, resume=True)
            output = merged.consolidate()
            with output.open("rb") as handle:
                rows = pickle.load(handle)
            self.assertEqual([row["key"] for row in rows], ["q0", "q1"])


if __name__ == "__main__":
    unittest.main()
