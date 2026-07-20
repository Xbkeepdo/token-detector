from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.qa_benchmark import atomic_write_jsonl
from features.qa_extractor import (
    QA_FEATURE_SCHEMA_VERSION,
    extract_questions,
    qa_generation_fingerprint,
    qa_label_fingerprint,
    qa_prompt,
    qa_question_input_fingerprint,
)
from models.base_wrapper import AttentionRequirement, ExtractionRequirements
from scripts.qa_pipeline import _feature_keys_complete, _qa_model_cfg
from scripts.train_qa_probes import (
    _configured_qa_feature_sets,
    _qa_training_input_fingerprint,
    _validate_reusable_result,
    _validate_training_artifacts,
)
from utils.config_utils import load_config, qa_extraction_family_flags


class QAAnswerOnlyConfigTests(unittest.TestCase):
    def test_active_yaml_runs_only_prompt_last_position_by_default(self) -> None:
        config = load_config(str(ROOT / "configs/model_configs_unified.yaml"))
        positions = config["qa_benchmarks"]["position_protocols"]
        self.assertEqual(positions, ["prompt_last_token"])
        self.assertEqual(config["models"]["qwen3_vl_8b"]["max_new_tokens"], 512)
        self.assertEqual(_qa_model_cfg(config, "qwen3_vl_8b")["max_new_tokens"], 8)
        self.assertNotIn("max_pixels", _qa_model_cfg(config, "qwen3_vl_8b", "pope"))
        self.assertEqual(
            _qa_model_cfg(config, "qwen3_vl_8b", "amber_discriminative")[
                "max_pixels"
            ],
            200704,
        )
        feature_sets = _configured_qa_feature_sets(
            config["training"],
            tuple(positions),
            qa_extraction_family_flags(config),
        )
        expected_count = sum(
            len(config["training"]["feature_sets"][family])
            for family in ("method", "ads_cgc")
        )
        self.assertEqual(len(feature_sets), expected_count)
        self.assertTrue(all(name.endswith("@prompt_last_token") for name in feature_sets))
        self.assertFalse(any("_target_cosine" in name for name in feature_sets))
        self.assertIn(
            "hmid_softmax_prob_gauss_risk+"
            "hmid_softmax_prob_gauss_ev_target_dist_mass_x_cosine@"
            "prompt_last_token",
            feature_sets,
        )
        self.assertEqual(
            qa_extraction_family_flags(config),
            {"mode": "all", "method": True, "ads_cgc": True, "baseline": True},
        )

    def test_qa_extraction_modes_match_coco_family_semantics(self) -> None:
        base = {
            "feature_extraction": {
                "method": {"enabled": True},
                "dgst_t": {"enabled": True},
                "ads_cgc": {"enabled": True},
                "baseline": {"enabled": True},
            }
        }
        expected = {
            "all": (True, True, True),
            "method_only": (True, False, False),
            "ads_cgc_only": (False, True, False),
            "baseline_only": (False, False, True),
        }
        for mode, values in expected.items():
            config = {**base, "qa_benchmarks": {"extraction_mode": mode}}
            flags = qa_extraction_family_flags(config)
            self.assertEqual(
                (flags["method"], flags["ads_cgc"], flags["baseline"]),
                values,
            )

    def test_prompt_last_extraction_never_calls_object_forward(self) -> None:
        class TinyTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [7] if str(text).strip().lower() == "yes" else []

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return "yes" if 7 in token_ids else ""

        class Wrapper:
            tokenizer = TinyTokenizer()

            def __init__(self) -> None:
                self.answer_calls = 0
                self.object_calls = 0

            def extract_token_features_batch(self, **kwargs):
                self.answer_calls += 1
                self.request = kwargs
                return [object()]

            def extract_prompt_target_features(self, **kwargs):
                self.object_calls += 1
                raise AssertionError("object forward must be disabled")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            question = {
                "key": "pope::random::1",
                "dataset": "pope",
                "source_split": "random",
                "question_id": 1,
                "image_id": 1,
                "probe_split": "train",
                "question_family_index": None,
                "question": "Is there a cat in the image?",
                "image_path": str(image_path),
                "object_span_status": "found",
                "object_span_protocol": "pope_official_query_surface_v1",
                "object_surface": "cat",
                "object_char_start": 11,
                "object_char_end": 14,
            }
            prompt = qa_prompt("qwen3_vl_8b", question["question"])
            atomic_write_jsonl(
                root / "generations.jsonl",
                [{
                    "key": question["key"],
                    "prompt": prompt,
                    "response_token_ids": [99, 7],
                    "answer_token_index": 1,
                    "answer_token_id": 7,
                    "generated_text": "Well yes",
                    "prediction": "yes",
                    "generation_protocol": "raw_question_yes_no_v1",
                }],
            )
            atomic_write_jsonl(
                root / "labels.jsonl",
                [{
                    "key": question["key"],
                    "label": 1,
                    "class_name": "real",
                    "error_type": "correct_yes",
                    "prediction": "yes",
                    "object_hallucination_yes_only_label": 1,
                }],
            )
            wrapper = Wrapper()
            with patch(
                "features.qa_extractor._build_position_record",
                return_value={"synthetic": True},
            ):
                rows = extract_questions(
                    wrapper,
                    "qwen3_vl_8b",
                    [question],
                    str(root),
                    {"ot_solver": "emd"},
                    {},
                    {},
                    shard_size=1,
                    position_protocols=("prompt_last_token",),
                    extraction_fingerprint="test-extraction",
                )
            self.assertEqual(wrapper.answer_calls, 1)
            self.assertEqual(wrapper.object_calls, 0)
            self.assertEqual(rows[0]["position_protocols"], ["prompt_last_token"])
            self.assertEqual(set(rows[0]["positions"]), {"prompt_last_token"})
            self.assertEqual(
                rows[0]["question_object_position"]["status"],
                "disabled_by_config",
            )
            self.assertEqual(wrapper.request["response_token_indices"], [0])
            self.assertEqual(wrapper.request["target_token_ids"], [99])

    def test_joint_and_baseline_only_modes_share_one_forward(self) -> None:
        class TinyTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return [7] if str(text).strip().lower() == "yes" else []

            @staticmethod
            def decode(token_ids, skip_special_tokens=True):
                return "yes" if token_ids else ""

        output_sentinel = object()

        class Wrapper:
            tokenizer = TinyTokenizer()

            def __init__(self) -> None:
                self.calls = []

            def extract_token_features_batch(self, **kwargs):
                self.calls.append(kwargs)
                return [output_sentinel]

        class Store:
            def __init__(self) -> None:
                self.rows = {}
                self.flushes = 0

            def add(self, row):
                self.rows[row["key"]] = row

            def flush(self):
                self.flushes += 1

        class Adapter:
            label_protocol = "object_hallucination_yes_only"
            requirements = ExtractionRequirements(
                attention=AttentionRequirement.HEAD_MEAN,
                logits=False,
                token_hidden_states=False,
                patch_hidden_states=False,
                response_hidden_states=False,
                visual_layout=False,
                dgst_capture=False,
            )

            def __init__(self) -> None:
                self.outputs = []

            def build_record(self, **kwargs):
                self.outputs.append(kwargs["model_output"])
                return {"key": kwargs["question"]["key"]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            question = {
                "key": "pope::random::1",
                "dataset": "pope",
                "source_split": "random",
                "question_id": 1,
                "image_id": 1,
                "probe_split": "train",
                "question": "Is there a cat in the image?",
                "image_path": str(image_path),
            }
            prompt = qa_prompt("qwen3_vl_8b", question["question"])

            def prepare(path: Path) -> None:
                path.mkdir()
                atomic_write_jsonl(path / "generations.jsonl", [{
                    "key": question["key"],
                    "prompt": prompt,
                    "response_token_ids": [7],
                    "answer_token_index": 0,
                    "answer_token_id": 7,
                    "generated_text": "yes",
                    "prediction": "yes",
                    "generation_protocol": "raw_question_yes_no_v1",
                }])
                atomic_write_jsonl(path / "labels.jsonl", [{
                    "key": question["key"],
                    "label": 1,
                    "class_name": "real",
                    "error_type": "correct_yes",
                    "prediction": "yes",
                    "object_hallucination_yes_only_label": 1,
                }])

            joint_dir = root / "joint"
            prepare(joint_dir)
            joint_wrapper = Wrapper()
            joint_store = Store()
            joint_adapter = Adapter()
            with patch(
                "features.qa_extractor._build_position_record",
                return_value={"synthetic": True},
            ):
                joint_rows = extract_questions(
                    joint_wrapper,
                    "qwen3_vl_8b",
                    [question],
                    str(joint_dir),
                    {"ot_solver": "emd"},
                    {},
                    {},
                    shard_size=1,
                    extraction_fingerprint="joint",
                    baseline_consumers={
                        joint_adapter.label_protocol: {
                            "adapter": joint_adapter,
                            "store": joint_store,
                        }
                    },
                )
            self.assertEqual(len(joint_wrapper.calls), 1)
            self.assertIs(joint_adapter.outputs[0], output_sentinel)
            self.assertEqual(len(joint_rows), 1)
            self.assertTrue((joint_dir / "features.pkl").is_file())
            self.assertEqual(set(joint_store.rows), {question["key"]})

            baseline_dir = root / "baseline_only"
            prepare(baseline_dir)
            baseline_wrapper = Wrapper()
            baseline_store = Store()
            baseline_adapter = Adapter()
            rows = extract_questions(
                baseline_wrapper,
                "qwen3_vl_8b",
                [question],
                str(baseline_dir),
                {},
                {},
                {},
                shard_size=1,
                extraction_fingerprint="baseline-only",
                method_enabled=False,
                ads_cgc_enabled=False,
                baseline_consumers={
                    baseline_adapter.label_protocol: {
                        "adapter": baseline_adapter,
                        "store": baseline_store,
                    }
                },
            )
            self.assertEqual(rows, [])
            self.assertEqual(len(baseline_wrapper.calls), 1)
            self.assertIs(baseline_adapter.outputs[0], output_sentinel)
            self.assertIsNone(baseline_wrapper.calls[0]["cfg_dgst_t"])
            self.assertFalse((baseline_dir / "features.pkl").exists())
            self.assertEqual(set(baseline_store.rows), {question["key"]})

    def test_resume_requires_the_same_selected_position_profile(self) -> None:
        question = {
            "key": "pope::random::1",
            "dataset": "pope",
            "source_split": "random",
            "question_id": 1,
            "image_id": 1,
            "probe_split": "train",
            "question": "Is there a cat in the image?",
        }
        prompt = qa_prompt("qwen3_vl_8b", question["question"])
        generation = {
            "key": question["key"],
            "prompt": prompt,
            "response_token_ids": [42],
            "generated_text": "yes",
            "prediction": "yes",
            "generation_protocol": "raw_question_yes_no_v1",
            "answer_token_index": 0,
            "answer_token_id": 42,
        }
        label_row = {
            "key": question["key"],
            "label": 1,
            "answer_correctness_all_label": 1,
            "object_hallucination_yes_only_label": 1,
            "prediction": "yes",
            "class_name": "real",
            "error_type": "correct_yes",
            "probe_split": "train",
            "image_id": 1,
            "source_split": "random",
        }
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            question["image_path"] = str(image_path)
            row = {
                "key": question["key"],
                "feature_schema_version": QA_FEATURE_SCHEMA_VERSION,
                "generation_fingerprint": qa_generation_fingerprint(generation, prompt),
                "label_fingerprint": qa_label_fingerprint(label_row),
                "question_input_fingerprint": qa_question_input_fingerprint(
                    question, prompt
                ),
                "extraction_fingerprint": "test-extraction",
                "feature_families": {"method": True, "ads_cgc": True},
                "position_protocols": ["prompt_last_token"],
                "positions": {
                    "prompt_last_token": {
                        "dgst": {},
                        "ads_score": 0.0,
                        "ads_per_layer": [],
                        "cgc_score": 0.0,
                        "cgc_per_layer": [],
                    }
                },
            }
            path = Path(directory) / "features.pkl"
            with path.open("wb") as handle:
                pickle.dump([row], handle)
            args = (
                str(path),
                {question["key"]},
                {question["key"]: generation},
                {question["key"]: label_row},
                [question],
                "qwen3_vl_8b",
            )
            self.assertTrue(
                _feature_keys_complete(
                    *args, ("prompt_last_token",), "test-extraction"
                )
            )
            moved_answer = {
                **generation,
                "response_token_ids": [9, 42],
                "answer_token_index": 1,
            }
            self.assertFalse(
                _feature_keys_complete(
                    str(path),
                    {question["key"]},
                    {question["key"]: moved_answer},
                    {question["key"]: label_row},
                    [question],
                    "qwen3_vl_8b",
                    ("prompt_last_token",),
                    "test-extraction",
                )
            )
            self.assertFalse(
                _feature_keys_complete(
                    *args,
                    ("prompt_last_token", "question_object_pre_token"),
                    "test-extraction",
                )
            )
            changed_label = {**label_row, "label": 0}
            self.assertFalse(
                _feature_keys_complete(
                    str(path),
                    {question["key"]},
                    {question["key"]: generation},
                    {question["key"]: changed_label},
                    [question],
                    "qwen3_vl_8b",
                    ("prompt_last_token",),
                    "test-extraction",
                )
            )
            self.assertFalse(
                _feature_keys_complete(
                    *args, ("prompt_last_token",), "changed-extraction"
                )
            )

    def test_probe_resume_fingerprints_inputs_and_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "features.pkl").open("wb") as handle:
                pickle.dump([{"key": "row", "value": 1}], handle)
            (root / "labels.jsonl").write_text(
                "{\"key\":\"row\",\"label\":1}\n", encoding="utf-8"
            )
            (root / "image_splits.json").write_text(
                "{\"train\":[1],\"val\":[],\"test\":[2,3]}\n",
                encoding="utf-8",
            )
            fingerprint, _ = _qa_training_input_fingerprint(
                root, {"hidden_sizes": [8]}, ("prompt_last_token",)
            )
            changed_cfg_fingerprint, _ = _qa_training_input_fingerprint(
                root, {"hidden_sizes": [16]}, ("prompt_last_token",)
            )
            self.assertNotEqual(fingerprint, changed_cfg_fingerprint)

            label = {
                "key": "row",
                "label": 1,
                "answer_correctness_all_label": 1,
                "object_hallucination_yes_only_label": 1,
                "prediction": "yes",
                "class_name": "real",
                "error_type": "correct_yes",
                "probe_split": "train",
                "image_id": 1,
                "source_split": "random",
            }
            feature = {
                **label,
                "label_fingerprint": qa_label_fingerprint(label),
            }
            _validate_training_artifacts(
                [feature], {"row": label},
                {"train": [1], "val": [], "test": [2, 3]},
            )
            with self.assertRaises(ValueError):
                _validate_training_artifacts(
                    [{**feature, "probe_split": "test"}],
                    {"row": label},
                    {"train": [1], "val": [], "test": [2, 3]},
                )

            result = {
                "feature_set": "ads@prompt_last_token",
                "seed": 42,
                "label_protocol": "answer_correctness_all",
                "position": "prompt_last_token",
                "positive_class": "real",
                "split_protocol": "strict_82_no_validation",
                "checkpoint_selection": "minimum_train_loss",
                "threshold_selection": "train_f1",
                "training_input_fingerprint": fingerprint,
            }
            _validate_reusable_result(
                result,
                "ads@prompt_last_token",
                42,
                "answer_correctness_all",
                "prompt_last_token",
                fingerprint,
            )
            with self.assertRaises(RuntimeError):
                _validate_reusable_result(
                    result,
                    "ads@prompt_last_token",
                    42,
                    "answer_correctness_all",
                    "prompt_last_token",
                    changed_cfg_fingerprint,
                )


if __name__ == "__main__":
    unittest.main()
