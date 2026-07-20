from __future__ import annotations

import json
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from detection.qa_probe import baseline_probe_vector, label_for_protocol
from features.baseline import attach_baseline, make_baseline_record
from features.qa_baseline import (
    DEFAULT_QA_LABEL_PROTOCOL,
    QA_BASELINE_PROTOCOL,
    QABaselineAdapter,
    QABaselineFeatureStore,
    build_qa_probe_split_manifest,
    qa_baseline_feature_config,
    qa_cache_id,
    qa_image_identity,
    qa_label_for_protocol,
    resolve_qa_answer_index,
    split_qa_baseline_records,
    validate_halloc_cache_uniqueness,
)
from models.base_wrapper import ModelOutput
from scripts.train_qa_baselines import (
    DEFAULT_SEEDS,
    _normalize_qa_baseline_trainer,
    run_qa_baseline_training,
)


class _Tokenizer:
    all_special_ids = [99]

    def encode(self, text, add_special_tokens=False):
        return [7] if "yes" in text.lower() else []

    def decode(self, ids, skip_special_tokens=True):
        return " yes" if 7 in ids else ""


class _FakeRuntime:
    official_svar_enabled = False
    methods = ("halloc",)

    def requirements_for(self, *, controlled, official):
        assert controlled and not official
        return "requirements"

    def build_image_records(
        self, *, image, image_id, response_token_ids, spans, model_outputs
    ):
        span = spans[0]
        record = make_baseline_record(
            image_id=image_id,
            token_str=span["word"],
            response_token_idx=span["token_indices"][0],
            target_token_id=response_token_ids[span["token_indices"][0]],
            label=span["label"],
        )
        attach_baseline(
            record,
            "halloc",
            {
                "cache_file": f"halloc/cache/{image_id}.npz",
                "object_index": span["token_indices"][0],
            },
        )
        return [record]


def _model_output(index=0, token_id=99):
    return ModelOutput(
        token_id=token_id,
        token_str="first",
        text_to_patch_attn=torch.empty(0),
        text_to_text_attn=torch.empty(0),
        token_hidden_states=torch.empty(0),
        patch_hidden_states=torch.empty(0),
        response_token_idx=index,
    )


def _question(key, source, image_id, split, dataset="pope", question_id=1):
    return {
        "key": key,
        "dataset": dataset,
        "source_split": source,
        "question_id": question_id,
        "image_id": image_id,
        "probe_split": split,
    }


def _baseline_record(
    key,
    source,
    image_id,
    split,
    label,
    dataset="pope",
    cache_file=None,
    label_protocol=None,
):
    record = make_baseline_record(
        image_id=image_id,
        token_str="yes",
        response_token_idx=0,
        target_token_id=7,
        label=label,
    )
    attach_baseline(record, "svar", {"vector": np.asarray([1.0], np.float32)})
    if cache_file is not None:
        attach_baseline(
            record,
            "halloc",
            {"cache_file": cache_file, "object_index": 0},
        )
    record.update(
        {
            "key": key,
            "dataset": dataset,
            "source_split": source,
            "question_id": 1,
            "image_id": image_id,
            "probe_split": split,
            "qa_image_identity": qa_image_identity(
                {
                    "dataset": dataset,
                    "source_split": source,
                    "image_id": image_id,
                }
            ),
        }
    )
    if label_protocol is not None:
        record["qa_label_protocol"] = label_protocol
    return record


def _projectaway_record(key, image_id, split, label, protocol):
    record = _baseline_record(
        key,
        "random",
        image_id,
        split,
        label,
        label_protocol=protocol,
    )
    attach_baseline(
        record,
        "projectaway",
        {
            "hallucination_score": 0.9 if label == 0 else 0.1,
            "score_orientation": "higher_is_more_hallucinatory",
        },
    )
    return record


class QABaselineTests(unittest.TestCase):
    def test_qa_baseline_trainer_is_explicit(self):
        self.assertEqual(
            _normalize_qa_baseline_trainer("shared_mlp"),
            "shared_torch_mlp",
        )
        self.assertEqual(
            _normalize_qa_baseline_trainer("paper"),
            "native_paper",
        )
        with self.assertRaisesRegex(ValueError, "baseline_trainer"):
            _normalize_qa_baseline_trainer("mystery")

    def test_shared_mlp_vectors_and_protocol_specific_label(self):
        record = _baseline_record(
            "shared",
            "attribute",
            1,
            "train",
            0,
            dataset="amber_discriminative",
            label_protocol="object_hallucination_yes_only",
        )
        attach_baseline(
            record,
            "metatoken",
            {"vector": np.asarray([1.0, 2.0], np.float32)},
        )
        attach_baseline(
            record,
            "projectaway",
            {
                "internal_confidence": 0.4,
                "hallucination_score": 0.6,
                "per_layer_internal_confidence": np.asarray(
                    [0.1, 0.4], np.float32
                ),
            },
        )
        svar_matrix = np.arange(8, dtype=np.float32).reshape(4, 2)
        attach_baseline(
            record,
            "svar",
            {
                "vector": svar_matrix.reshape(-1),
                "visual_attention_ratio": svar_matrix,
                "layer_start": 0,
                "layer_end_exclusive": 4,
            },
        )
        np.testing.assert_array_equal(
            baseline_probe_vector(record, "metatoken"),
            np.asarray([1.0, 2.0], np.float32),
        )
        np.testing.assert_allclose(
            baseline_probe_vector(record, "projectaway"),
            np.asarray([0.4, 0.1, 0.4], np.float32),
        )
        np.testing.assert_array_equal(
            baseline_probe_vector(
                record,
                "svar",
                svar_layer_start=1,
                svar_layer_end=3,
            ),
            svar_matrix[1:3].reshape(-1),
        )
        self.assertEqual(
            label_for_protocol(record, "object_hallucination_yes_only"),
            0,
        )

    def test_answer_index_uses_actual_saved_response_token(self):
        response_ids, index = resolve_qa_answer_index(
            {
                "response_token_ids": [99, 7, 8],
                "answer_token_index": 1,
                "answer_token_id": 7,
                "prediction": "yes",
            },
            _Tokenizer(),
        )
        self.assertEqual(response_ids, [99, 7, 8])
        self.assertEqual(index, 1)
        with self.assertRaisesRegex(ValueError, "answer_token_id"):
            resolve_qa_answer_index(
                {
                    "response_token_ids": [99, 7],
                    "answer_token_index": 1,
                    "answer_token_id": 8,
                },
                _Tokenizer(),
            )
        with self.assertRaisesRegex(ValueError, "actual first generated yes/no"):
            resolve_qa_answer_index(
                {
                    "response_token_ids": [7, 8],
                    "answer_token_index": 1,
                    "answer_token_id": 8,
                    "prediction": "yes",
                },
                _Tokenizer(),
            )

    def test_question_cache_prevents_same_image_multi_question_halloc_overwrite(self):
        adapter = QABaselineAdapter(_FakeRuntime())
        generation_a = {"key": "pope::random::1", "prediction": "yes"}
        generation_b = {"key": "pope::popular::2", "prediction": "yes"}
        question_a = _question(
            generation_a["key"], "random", 17, "train", question_id=1
        )
        question_b = _question(
            generation_b["key"], "popular", 17, "train", question_id=2
        )
        label_a = {
            "key": generation_a["key"],
            "label": 1,
            "probe_split": "train",
        }
        label_b = {
            "key": generation_b["key"],
            "label": 0,
            "probe_split": "train",
        }
        record_a = adapter.build_record(
            image=SimpleNamespace(),
            question=question_a,
            generation=generation_a,
            label_row=label_a,
            response_token_ids=[99, 7],
            target_index=0,
            model_output=_model_output(),
        )
        record_b = adapter.build_record(
            image=SimpleNamespace(),
            question=question_b,
            generation=generation_b,
            label_row=label_b,
            response_token_ids=[99, 7],
            target_index=0,
            model_output=_model_output(),
        )
        self.assertEqual(record_a["image_id"], 17)
        self.assertEqual(record_b["image_id"], 17)
        self.assertEqual(
            record_a["qa_image_identity"], record_b["qa_image_identity"]
        )
        self.assertNotEqual(qa_cache_id(record_a["key"]), qa_cache_id(record_b["key"]))
        cache_a = record_a["baselines"]["halloc"]["cache_file"]
        cache_b = record_b["baselines"]["halloc"]["cache_file"]
        self.assertNotEqual(cache_a, cache_b)
        validate_halloc_cache_uniqueness([record_a, record_b])
        self.assertEqual(
            record_a["metadata"]["qa"]["protocol"], QA_BASELINE_PROTOCOL
        )
        self.assertEqual(
            record_a["metadata"]["qa"]["prompt_last_response_target_index"],
            0,
        )
        self.assertFalse(any("gt" in key.lower() for key in record_a))

    def test_probe_split_is_question_authoritative_and_image_disjoint(self):
        records = [
            _baseline_record("p-r", "random", 1, "train", 0),
            _baseline_record("p-p", "popular", 1, "train", 1),
            _baseline_record("p-t", "random", 3, "test", 1),
        ]
        split = split_qa_baseline_records(records)
        self.assertEqual(
            [row["key"] for row in split["train"]], ["p-r", "p-p"]
        )
        manifest = build_qa_probe_split_manifest(records)
        self.assertEqual(
            manifest["question_counts"], {"train": 2, "val": 0, "test": 1}
        )
        self.assertEqual(
            manifest["image_counts"], {"train": 1, "val": 0, "test": 1}
        )

        leaked = list(records)
        leaked[-1] = _baseline_record("p-t", "popular", 1, "test", 0)
        with self.assertRaisesRegex(ValueError, "image leakage"):
            split_qa_baseline_records(leaked)

    def test_clevr_image_identity_keeps_official_source_namespace(self):
        self.assertNotEqual(
            qa_image_identity(
                {
                    "dataset": "clevr_exist_5k",
                    "source_split": "train",
                    "image_id": 1,
                }
            ),
            qa_image_identity(
                {
                    "dataset": "clevr_exist_5k",
                    "source_split": "val",
                    "image_id": 1,
                }
            ),
        )

    def test_halloc_cache_collision_is_rejected(self):
        records = [
            _baseline_record(
                "a", "random", 1, "train", 0, cache_file="same.npz"
            ),
            _baseline_record(
                "b", "random", 1, "train", 1, cache_file="same.npz"
            ),
        ]
        with self.assertRaisesRegex(ValueError, "cache collision"):
            validate_halloc_cache_uniqueness(records)

    def test_feature_store_resume_and_training_defaults(self):
        record = _baseline_record("a", "random", 1, "train", 0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = QABaselineFeatureStore(root, shard_size=1)
            self.assertTrue(store.add(record))
            path = store.consolidate()
            resumed = QABaselineFeatureStore(root, shard_size=1, resume=True)
            self.assertIn("a", resumed.rows)
            self.assertFalse(resumed.add(record))
            with path.open("rb") as handle:
                self.assertEqual(pickle.load(handle)[0]["key"], "a")
        self.assertEqual(DEFAULT_SEEDS, (43, 44, 45))

    def test_two_label_protocols_are_explicit_and_yes_only_filters_null(self):
        all_row = {
            "key": "q-all",
            "label": 1,
            "object_hallucination_yes_only_label": None,
        }
        yes_row = {
            "key": "q-yes",
            "label": 0,
            "object_hallucination_yes_only_label": 1,
        }
        self.assertEqual(
            qa_label_for_protocol(all_row, "answer_correctness_all"),
            1,
        )
        self.assertIsNone(
            qa_label_for_protocol(
                all_row,
                "object_hallucination_yes_only",
            )
        )
        self.assertEqual(
            qa_label_for_protocol(
                yes_row,
                "object_hallucination_yes_only",
            ),
            1,
        )
        with self.assertRaisesRegex(KeyError, "object_hallucination"):
            qa_label_for_protocol(
                {"key": "missing", "label": 1},
                "object_hallucination_yes_only",
            )

        adapter = QABaselineAdapter(
            _FakeRuntime(),
            label_protocol="object_hallucination_yes_only",
        )
        question = _question("q-yes", "random", 19, "train")
        record = adapter.build_record(
            image=SimpleNamespace(),
            question=question,
            generation={"key": "q-yes", "prediction": "yes"},
            label_row={
                "key": "q-yes",
                "label": 0,
                "object_hallucination_yes_only_label": 1,
                "probe_split": "train",
            },
            response_token_ids=[99, 7],
            target_index=0,
            model_output=_model_output(),
        )
        self.assertEqual(record["label"], 1)
        self.assertEqual(
            record["qa_label_protocol"],
            "object_hallucination_yes_only",
        )

    def test_three_seed_projectaway_training_writes_protocol_isolated_summary(self):
        protocol = "object_hallucination_yes_only"
        records = []
        next_image = 1
        for split in ("train", "test"):
            for label in (0, 1):
                records.append(
                    _projectaway_record(
                        f"{split}-{label}",
                        next_image,
                        split,
                        label,
                        protocol,
                    )
                )
                next_image += 1
        split_records = split_qa_baseline_records(
            records,
            label_protocol=protocol,
        )
        split_manifest = build_qa_probe_split_manifest(
            records,
            label_protocol=protocol,
        )
        with tempfile.TemporaryDirectory() as directory:
            baseline_dir = Path(directory) / "baseline" / protocol
            feature_path = baseline_dir / "features.pkl"
            split_path = baseline_dir / "qa_probe_splits.json"
            baseline_dir.mkdir(parents=True)
            feature_path.write_bytes(b"test")
            split_path.write_text("{}\n", encoding="utf-8")
            run_qa_baseline_training(
                model="tiny",
                dataset="pope",
                label_protocol=protocol,
                seeds=DEFAULT_SEEDS,
                methods=("projectaway",),
                split_records=split_records,
                image_split_counts=split_manifest["image_counts"],
                feature_path=feature_path,
                split_path=split_path,
                baseline_dir=baseline_dir,
                baseline_cfg={},
                device="cpu",
            )
            stem = f"tiny_pope_{protocol}_qa_baselines"
            aggregate_path = (
                baseline_dir / "results" / f"{stem}_3seed.json"
            )
            markdown_path = (
                baseline_dir / "results" / f"{stem}_3seed_summary.md"
            )
            self.assertTrue(aggregate_path.is_file())
            self.assertTrue(markdown_path.is_file())
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(aggregate["seeds"], [43, 44, 45])
            self.assertEqual(aggregate["headline_positive_class"], "real")
            seed_output = json.loads(
                (
                    baseline_dir
                    / "results"
                    / "seed43"
                    / f"{stem}.json"
                ).read_text(encoding="utf-8")
            )
            metrics = seed_output["methods"]["projectaway"]["test_metrics"]
            self.assertIn("real_positive", metrics)
            self.assertIn("hallucination_positive", metrics)
            self.assertEqual(seed_output["threshold_selection"], "train_f1")
            self.assertAlmostEqual(
                seed_output["methods"]["projectaway"]["threshold"],
                0.9,
            )
            self.assertIn(
                "train_metrics",
                seed_output["methods"]["projectaway"],
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("3 个随机种子", markdown)
            self.assertIn("headline 正类：real", markdown)

    def test_feature_fingerprint_is_extraction_only_and_svar_controlled(self):
        config = {
            "svar": {
                "layer_start": 5,
                "epochs": 99,
                "protocols": ["official"],
            },
            "halloc": {
                "clip_model": "clip",
                "visualbert_model": "bert",
                "epochs": 9,
            },
        }
        payload = qa_baseline_feature_config(config, ["svar", "halloc"])
        self.assertEqual(
            payload["svar"],
            {"protocols": ["controlled"], "extraction_layers": "all"},
        )
        self.assertEqual(payload["halloc"], {"clip_model": "clip"})


if __name__ == "__main__":
    unittest.main()
