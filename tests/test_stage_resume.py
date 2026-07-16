from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_baselines import (
    _has_extractable_object_spans as baseline_has_extractable_spans,
)
from scripts.extract_features import (
    _has_extractable_object_spans,
    _pending_samples_for_resume,
)
from utils.io_utils import save_pkl


def _load_label_coco_module():
    path = ROOT / "coco-labeling" / "label_coco.py"
    spec = importlib.util.spec_from_file_location("label_coco_resume_tests", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.label_coco = _load_label_coco_module()

    def test_complete_matching_labels_skip_the_labeling_stage(self) -> None:
        samples = [{"image_id": 1}, {"image_id": 2}]
        generations = {
            "1": {"generated_text": "a chair"},
            "2": {"generated_text": "a table"},
        }
        labeling = {
            image_id: {
                "image_id": int(image_id),
                "generated_text": generation["generated_text"],
                "hallucinated_words": [],
                "real_words": [],
                "object_token_spans": [],
                "chair_s": 0,
                "chair_i": 0.0,
            }
            for image_id, generation in generations.items()
        }
        self.assertTrue(
            self.label_coco._all_labeling_available(
                samples,
                generations,
                labeling,
            )
        )

        mismatched = dict(labeling)
        mismatched["2"] = dict(mismatched["2"], generated_text="changed")
        self.assertFalse(
            self.label_coco._all_labeling_available(
                samples,
                generations,
                mismatched,
            )
        )
        self.assertFalse(
            self.label_coco._all_labeling_available(
                samples,
                generations,
                {"1": labeling["1"]},
            )
        )

    def test_non_extractable_spans_do_not_stay_pending_forever(self) -> None:
        valid = {
            "generated_text": "a chair",
            "object_token_spans": [{"token_indices": [1]}],
        }
        empty = {
            "generated_text": "a chair",
            "object_token_spans": [],
        }
        out_of_range = {
            "generated_text": "a chair",
            "object_token_spans": [{"token_indices": [3]}],
        }
        generation = {"response_token_ids": [10, 11]}

        for helper in (
            _has_extractable_object_spans,
            baseline_has_extractable_spans,
        ):
            self.assertTrue(helper(valid, generation))
            self.assertFalse(helper(empty, generation))
            self.assertFalse(helper(out_of_range, generation))

    def test_resume_requires_root_and_baseline_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_output = root / "features.pkl"
            baseline_output = root / "baseline" / "features.pkl"
            samples = [{"image_id": 1}, {"image_id": 2}]
            save_pkl([{"image_id": 1}, {"image_id": 2}], str(root_output))
            save_pkl([{"image_id": 1}], str(baseline_output))

            pending, complete = _pending_samples_for_resume(
                samples=samples,
                root_output_path=str(root_output),
                root_part_paths=[],
                baseline_output_path=str(baseline_output),
                baseline_part_paths=[],
            )
            self.assertEqual([sample["image_id"] for sample in pending], [2])
            self.assertEqual(complete, 1)

            save_pkl(
                [{"image_id": 1}, {"image_id": 2}],
                str(baseline_output),
            )
            pending, complete = _pending_samples_for_resume(
                samples=samples,
                root_output_path=str(root_output),
                root_part_paths=[],
                baseline_output_path=str(baseline_output),
                baseline_part_paths=[],
            )
            self.assertEqual(pending, [])
            self.assertEqual(complete, 2)


if __name__ == "__main__":
    unittest.main()
