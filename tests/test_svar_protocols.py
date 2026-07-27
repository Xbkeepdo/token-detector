from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from features.baseline import (
    BaselineRuntime,
    attach_baseline,
    make_baseline_record,
    normalize_svar_protocols,
    prepare_official_svar_spans,
    validate_baseline_record,
)
from features.extractor import extract_features_for_dataset
from models.base_wrapper import AttentionRequirement, ModelOutput
from scripts.extract_baselines import (
    _extract_worker,
    _has_extractable_official_svar_samples,
    _pending_samples_for_protocols,
)
from scripts.train_baselines import (
    _official_svar_sample_audit,
    _train_one_seed,
    _write_training_summaries,
)
from utils.io_utils import load_pkl


def _model_output(index: int, token_id: int, *, compact=None) -> ModelOutput:
    return ModelOutput(
        token_id=token_id,
        token_str=f"token-{token_id}",
        text_to_patch_attn=torch.full((2, 2, 4), 0.125),
        text_to_text_attn=torch.empty(0),
        token_hidden_states=torch.empty(0),
        patch_hidden_states=torch.empty(0),
        response_token_idx=index,
        visual_grid=(2, 2),
        baseline_capture=compact,
    )


class SVARProtocolExtractionTests(unittest.TestCase):
    def test_protocol_normalization_is_backward_compatible(self) -> None:
        self.assertEqual(normalize_svar_protocols(None), ("controlled",))
        self.assertEqual(
            normalize_svar_protocols(["fair", "paper"]),
            ("controlled", "official"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown SVAR"):
            normalize_svar_protocols(["unknown"])

    def test_official_samples_skip_not_found_and_use_first_token_id(self) -> None:
        response_ids = [10, 20, 30, 20]
        spans = prepare_official_svar_spans(
            [
                {
                    "search_term": "person",
                    "label": 1,
                    "status": "not_found",
                },
                {
                    "search_term": "officer",
                    "search_source": "surface",
                    "label": 1,
                    "first_token_id": 20,
                },
                {
                    "search_term": "phones",
                    "label": 0,
                    "first_token_id": 999,
                    "plural_first_token_id": 30,
                },
                {
                    "search_term": "late",
                    "label": 0,
                    "token_indices": [3],
                },
                {
                    "search_term": "invalid",
                    "label": 0,
                    "token_indices": [9],
                },
            ],
            response_ids,
        )
        self.assertEqual(
            [span["token_indices"] for span in spans],
            [[1], [2], [3]],
        )
        self.assertEqual(spans[0]["word"], "officer")
        self.assertEqual(
            spans[0]["svar_official"]["first_token_id"],
            response_ids[1],
        )

    def test_runtime_isolates_official_records(self) -> None:
        class _Wrapper:
            device = "cpu"
            model = None

        response_ids = [10, 20]
        label_info = {
            "official_svar_samples": [
                {
                    "search_term": "officer",
                    "search_source": "surface",
                    "label": 1,
                    "token_indices": [1],
                },
                {"search_term": "person", "label": 1, "status": "not_found"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime = BaselineRuntime(
                wrapper=_Wrapper(),
                methods=("svar",),
                baseline_dir=directory,
                config={
                    "svar": {
                        "protocols": ["official"],
                        "layer_start": 0,
                        "layer_end": 2,
                    }
                },
                device="cpu",
            )
            self.assertEqual(runtime.methods, ())
            self.assertTrue(runtime.official_svar_enabled)
            self.assertEqual(
                runtime.requirements.attention,
                AttentionRequirement.PER_HEAD,
            )
            spans = runtime.prepare_official_svar_spans(
                label_info, response_ids
            )
            records = runtime.build_official_svar_records(
                image_id=7,
                response_token_ids=response_ids,
                spans=spans,
                model_outputs=[_model_output(1, 20)],
            )
            runtime.close()
        self.assertEqual(len(records), 1)
        validate_baseline_record(records[0], required=("svar",))
        self.assertEqual(records[0]["response_token_idx"], 1)
        self.assertEqual(records[0]["metadata"]["svar_protocol"], "official")
        self.assertEqual(
            records[0]["metadata"]["svar_official"]["search_term"],
            "officer",
        )

    def test_official_only_resume_uses_svar_capture_and_no_heavy_resources(
        self,
    ) -> None:
        class _Wrapper:
            device = "cpu"
            model = None

            def __init__(self):
                self.requirements = []

            def extract_token_features_batch(
                self,
                *,
                response_token_indices,
                target_token_ids,
                requirements,
                **kwargs,
            ):
                self.requirements.append(requirements)
                return [
                    _model_output(index, target)
                    for index, target in zip(
                        response_token_indices, target_token_ids
                    )
                ]

        wrapper = _Wrapper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            official_path = (
                root / "baseline" / "svar_official" / "features.pkl"
            )
            with (
                mock.patch("models.build_model", return_value=wrapper),
                mock.patch(
                    "features.baseline.runtime.DHCPShardWriter",
                    side_effect=AssertionError("DHCP must stay lazy"),
                ) as dhcp_cls,
                mock.patch(
                    "features.baseline.runtime.HalLocCLIPFeatureExtractor",
                    side_effect=AssertionError("HalLoc must stay lazy"),
                ) as clip_cls,
                mock.patch(
                    "features.baseline.runtime.resolve_output_embedding_layer",
                    side_effect=AssertionError("ProjectAway must stay lazy"),
                ) as resolve_output,
            ):
                _extract_worker(
                    worker_id=0,
                    model_key="demo",
                    model_cfg={},
                    device="cpu",
                    samples=[
                        {
                            "image_id": 1,
                            "image_path": str(image_path),
                            "_baseline_protocols_needed": {
                                "controlled": False,
                                "official": True,
                            },
                        }
                    ],
                    labeling={
                        1: {
                            "generated_text": "a dog",
                            "object_token_spans": [
                                {
                                    "word": "dog",
                                    "label": 1,
                                    "token_indices": [1],
                                }
                            ],
                            "official_svar_samples": [
                                {
                                    "search_term": "dog",
                                    "search_source": "canonical",
                                    "label": 1,
                                    "token_indices": [1],
                                }
                            ],
                        }
                    },
                    generations={
                        1: {
                            "generated_text": "a dog",
                            "response_token_ids": [10, 20],
                        }
                    },
                    baseline_cfg={
                        "svar": {
                            "protocols": ["controlled", "official"],
                            "layer_start": 0,
                            "layer_end": 2,
                        }
                    },
                    baseline_dir=str(root / "baseline"),
                    part_path=str(root / "baseline" / "features.pkl"),
                    official_part_path=str(official_path),
                    methods="all",
                    prompt="Describe this image.",
                    resume=False,
                    parallel=False,
                )
                dhcp_cls.assert_not_called()
                clip_cls.assert_not_called()
                resolve_output.assert_not_called()

            official = load_pkl(str(official_path))

        self.assertEqual(len(official), 1)
        self.assertEqual(len(wrapper.requirements), 1)
        requirements = wrapper.requirements[0]
        self.assertEqual(
            requirements.attention, AttentionRequirement.PER_HEAD
        )
        self.assertFalse(requirements.patch_hidden_states)
        self.assertFalse(requirements.response_hidden_states)
        self.assertFalse(requirements.visual_layout)
        self.assertFalse(requirements.logits)
        self.assertFalse(requirements.dgst_capture)

    def test_joint_resume_only_missing_official_skips_root_and_heavy_baselines(
        self,
    ) -> None:
        class _Tokenizer:
            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                return f"token-{int(token_ids[0])}"

        class _Wrapper:
            device = "cpu"
            model = None
            tokenizer = _Tokenizer()

            def __init__(self):
                self.requirements = []

            def extract_token_features_batch(
                self,
                *,
                response_token_indices,
                target_token_ids,
                requirements,
                **kwargs,
            ):
                self.requirements.append(requirements)
                return [
                    _model_output(index, target)
                    for index, target in zip(
                        response_token_indices, target_token_ids
                    )
                ]

        wrapper = _Wrapper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            (root / "generations.json").write_text(
                json.dumps(
                    {
                        "1": {
                            "generated_text": "a dog",
                            "response_token_ids": [10, 20],
                        }
                    }
                ),
                encoding="utf-8",
            )
            root_part = root / "features.part0.pkl"
            controlled_part = root / "baseline" / "features.part0.pkl"
            official_part = (
                root / "baseline" / "svar_official" / "features.part0.pkl"
            )
            baseline_cfg = {
                "svar": {
                    "protocols": ["controlled", "official"],
                    "layer_start": 0,
                    "layer_end": 2,
                }
            }
            with (
                mock.patch(
                    "features.extractor.compute_dgst_t",
                    side_effect=AssertionError("DGST must not rerun"),
                ) as dgst,
                mock.patch(
                    "features.baseline.runtime.DHCPShardWriter",
                    side_effect=AssertionError("DHCP must stay lazy"),
                ) as dhcp_cls,
                mock.patch(
                    "features.baseline.runtime.HalLocCLIPFeatureExtractor",
                    side_effect=AssertionError("HalLoc must stay lazy"),
                ) as clip_cls,
                mock.patch(
                    "features.baseline.runtime.resolve_output_embedding_layer",
                    side_effect=AssertionError("ProjectAway must stay lazy"),
                ) as resolve_output,
            ):
                with BaselineRuntime(
                    wrapper=wrapper,
                    methods="all",
                    baseline_dir=root / "baseline",
                    config=baseline_cfg,
                    device="cpu",
                    resume=True,
                ) as runtime:
                    extract_features_for_dataset(
                        model_wrapper=wrapper,
                        coco_samples=[
                            {
                                "image_id": 1,
                                "image_path": str(image_path),
                                "_feature_families_needed": {
                                    "root": False,
                                    "controlled": False,
                                    "official": True,
                                },
                            }
                        ],
                        labeling_results={
                            1: {
                                "generated_text": "a dog",
                                "object_token_spans": [
                                    {
                                        "word": "dog",
                                        "label": 1,
                                        "token_indices": [1],
                                    }
                                ],
                                "official_svar_samples": [
                                    {
                                        "search_term": "dog",
                                        "label": 1,
                                        "status": "found",
                                        "token_indices": [1],
                                        "token_location": {
                                            "matched_token_id": 20,
                                        },
                                    }
                                ],
                            }
                        },
                        cfg_dgst_t={"enabled": True},
                        output_path=str(root_part),
                        resume=True,
                        prompt="Describe this image.",
                        cfg_feature_extraction={
                            "method": {"enabled": True},
                            "ads_cgc": {"enabled": True},
                            "baseline": {"enabled": True},
                        },
                        baseline_runtime=runtime,
                        baseline_output_path=str(controlled_part),
                        baseline_official_output_path=str(official_part),
                    )
                dgst.assert_not_called()
                dhcp_cls.assert_not_called()
                clip_cls.assert_not_called()
                resolve_output.assert_not_called()

            official = load_pkl(str(official_part))
        self.assertEqual(len(official), 1)
        self.assertEqual(len(wrapper.requirements), 1)
        requirements = wrapper.requirements[0]
        self.assertEqual(requirements.attention, AttentionRequirement.PER_HEAD)
        self.assertFalse(requirements.dgst_capture)
        self.assertFalse(requirements.patch_hidden_states)
        self.assertFalse(requirements.response_hidden_states)
        self.assertFalse(requirements.visual_layout)

    def test_baseline_only_worker_rejects_wrong_response_index(self) -> None:
        class _Wrapper:
            device = "cpu"
            model = None

            def extract_token_features_batch(
                self,
                *,
                response_token_indices,
                target_token_ids,
                **kwargs,
            ):
                return [
                    _model_output(index + 1, target)
                    for index, target in zip(
                        response_token_indices, target_token_ids
                    )
                ]

        wrapper = _Wrapper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            with mock.patch("models.build_model", return_value=wrapper):
                with self.assertRaisesRegex(
                    AssertionError, "requested causal position 1"
                ):
                    _extract_worker(
                        worker_id=0,
                        model_key="demo",
                        model_cfg={},
                        device="cpu",
                        samples=[
                            {
                                "image_id": 1,
                                "image_path": str(image_path),
                            }
                        ],
                        labeling={
                            1: {
                                "generated_text": "a dog",
                                "object_token_spans": [
                                    {
                                        "word": "dog",
                                        "label": 1,
                                        "token_indices": [1],
                                    }
                                ],
                            }
                        },
                        generations={
                            1: {
                                "generated_text": "a dog",
                                "response_token_ids": [10, 20],
                            }
                        },
                        baseline_cfg={
                            "svar": {
                                "protocols": ["controlled"],
                                "layer_start": 0,
                                "layer_end": 2,
                            }
                        },
                        baseline_dir=str(root / "baseline"),
                        part_path=str(root / "baseline" / "features.pkl"),
                        official_part_path=None,
                        methods=("svar",),
                        prompt="Describe this image.",
                        resume=False,
                        parallel=False,
                    )

    def test_metatoken_prefers_saved_pre_deduplication_occurrence_count(self) -> None:
        class _Wrapper:
            device = "cpu"
            model = None

        compact = {
            "response_target_logprobs": torch.tensor([-1.0, -0.5]),
            "response_target_probs": torch.tensor([0.3, 0.6]),
            "response_logprob_variances": torch.tensor([0.2, 0.1]),
            "response_normalized_entropies": torch.tensor([0.8, 0.7]),
            "response_top1_probs": torch.tensor([0.5, 0.7]),
            "response_top2_probs": torch.tensor([0.2, 0.2]),
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime = BaselineRuntime(
                wrapper=_Wrapper(),
                methods=("metatoken",),
                baseline_dir=directory,
                device="cpu",
            )
            records = runtime.build_image_records(
                image=Image.new("RGB", (4, 4)),
                image_id=9,
                response_token_ids=[10, 20],
                spans=[
                    {
                        "word": "dog",
                        "label": 1,
                        "token_indices": [1],
                        "occurrence_count": 7,
                    }
                ],
                model_outputs=[_model_output(1, 20, compact=compact)],
            )
            runtime.close()
        self.assertEqual(
            float(records[0]["baselines"]["metatoken"]["vector"][1]),
            7.0,
        )

    def test_protocol_resume_only_requeues_missing_output(self) -> None:
        samples = [{"image_id": 1}, {"image_id": 2}]
        labeling = {
            image_id: {
                "generated_text": "a dog",
                "object_token_spans": [
                    {"word": "dog", "label": 1, "token_indices": [1]}
                ],
                "official_svar_samples": [
                    {
                        "search_term": "dog",
                        "label": 1,
                        "token_indices": [1],
                    }
                ],
            }
            for image_id in (1, 2)
        }
        generations = {
            image_id: {"generated_text": "a dog", "response_token_ids": [10, 20]}
            for image_id in (1, 2)
        }
        self.assertTrue(
            _has_extractable_official_svar_samples(
                labeling[1], generations[1]
            )
        )
        pending = _pending_samples_for_protocols(
            samples=samples,
            labeling=labeling,
            generations=generations,
            controlled_enabled=True,
            official_enabled=True,
            controlled_done={1, 2},
            official_done={1},
        )
        self.assertEqual([sample["image_id"] for sample in pending], [2])
        self.assertEqual(
            pending[0]["_baseline_protocols_needed"],
            {"controlled": False, "official": True},
        )

    def test_baseline_only_worker_shares_one_forward_between_protocols(self) -> None:
        class _Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [10, 20, 30]

        class _Wrapper:
            device = "cpu"
            model = None
            tokenizer = _Tokenizer()

            def __init__(self):
                self.calls = []

            def extract_token_features_batch(
                self,
                *,
                response_token_indices,
                target_token_ids,
                **kwargs,
            ):
                self.calls.append(
                    (list(response_token_indices), list(target_token_ids))
                )
                return [
                    _model_output(index, target)
                    for index, target in zip(
                        response_token_indices, target_token_ids
                    )
                ]

        wrapper = _Wrapper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 8)).save(image_path)
            controlled_path = root / "baseline" / "features.pkl"
            official_path = (
                root / "baseline" / "svar_official" / "features.pkl"
            )
            with mock.patch("models.build_model", return_value=wrapper):
                _extract_worker(
                    worker_id=0,
                    model_key="demo",
                    model_cfg={},
                    device="cpu",
                    samples=[{"image_id": 1, "image_path": str(image_path)}],
                    labeling={
                        1: {
                            "generated_text": "a dog",
                            "object_token_spans": [
                                {
                                    "word": "dog",
                                    "label": 1,
                                    "token_indices": [1],
                                }
                            ],
                            "official_svar_samples": [
                                {
                                    "search_term": "dog",
                                    "search_source": "canonical",
                                    "label": 1,
                                    "token_indices": [2],
                                    "status": "found",
                                }
                            ],
                        }
                    },
                    generations={
                        1: {"generated_text": "a dog", "response_token_ids": [10, 20, 30]}
                    },
                    baseline_cfg={
                        "svar": {
                            "protocols": ["controlled", "official"],
                            "layer_start": 0,
                            "layer_end": 2,
                        }
                    },
                    baseline_dir=str(root / "baseline"),
                    part_path=str(controlled_path),
                    official_part_path=str(official_path),
                    methods=("svar",),
                    prompt="Describe this image.",
                    resume=False,
                    parallel=False,
                )
            controlled = load_pkl(str(controlled_path))
            official = load_pkl(str(official_path))
        self.assertEqual(wrapper.calls, [([1, 2], [20, 30])])
        self.assertEqual(controlled[0]["response_token_idx"], 1)
        self.assertEqual(
            controlled[0]["metadata"]["svar_protocol"], "controlled"
        )
        self.assertEqual(official[0]["response_token_idx"], 2)
        self.assertEqual(
            official[0]["metadata"]["svar_protocol"], "official"
        )


class SVAROfficialTrainingTests(unittest.TestCase):
    def test_official_sample_audit_counts_status_labels_and_splits(self) -> None:
        labeling = {
            "1": {
                "official_svar_samples": [
                    {"status": "found", "label": 0},
                    {"status": "not_found", "label": 1},
                ]
            },
            "2": {
                "official_svar_samples": [
                    {"status": "found", "label": 1}
                ]
            },
            "3": {
                "official_svar_samples": [
                    {"status": "not_found", "label": 0}
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeling.json"
            path.write_text(json.dumps(labeling), encoding="utf-8")
            audit = _official_svar_sample_audit(
                path,
                {"train": [1], "val": [2], "test": [3]},
            )
        self.assertEqual(audit["overall"]["total"], 4)
        self.assertEqual(audit["overall"]["found"], 2)
        self.assertEqual(audit["overall"]["not_found"], 2)
        self.assertEqual(audit["overall"]["hallucination_found"], 1)
        self.assertEqual(audit["overall"]["real_found"], 1)
        self.assertEqual(audit["by_split"]["train"]["found"], 1)
        self.assertEqual(audit["by_split"]["test"]["not_found"], 1)

    def test_official_training_writes_isolated_json_checkpoint_and_markdown(self) -> None:
        def record(image_id: int, label: int, value: float):
            item = make_baseline_record(
                image_id=image_id,
                token_str="object",
                response_token_idx=0,
                target_token_id=1,
                label=label,
                metadata={"svar_protocol": "official"},
            )
            attach_baseline(
                item,
                "svar",
                {"vector": np.asarray([value, 1.0 - value], np.float32)},
            )
            return item

        split_records = {
            "train": [record(1, 0, 0.9), record(2, 1, 0.1)],
            "val": [record(3, 0, 0.8), record(4, 1, 0.2)],
            "test": [record(5, 0, 0.7), record(6, 1, 0.3)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "baseline" / "svar_official"
            output, result_path = _train_one_seed(
                model="demo",
                seed=42,
                methods=("svar",),
                split_records=split_records,
                image_split_counts={"train": 8, "val": 1, "test": 1},
                feature_path=root / "features.pkl",
                split_path=Path(directory) / "image_splits.json",
                baseline_dir=root,
                baseline_cfg={
                    "seed": 42,
                    "svar": {
                        "hidden_dim": 4,
                        "epochs": 1,
                        "batch_size": 2,
                    },
                },
                device="cpu",
                run_name=None,
                result_stem="demo_svar_official",
                label_protocol=(
                    "official_svar_set_first_token_id_first_occurrence"
                ),
                sample_audit={
                    "overall": {
                        "total": 10,
                        "found": 6,
                        "not_found": 4,
                        "hallucination_found": 3,
                        "real_found": 3,
                    },
                    "by_split": {
                        "train": {"found": 2},
                        "val": {"found": 2},
                        "test": {"found": 2},
                    },
                },
            )
            self.assertEqual(
                result_path,
                root / "results" / "demo_svar_official.json",
            )
            self.assertTrue(result_path.exists())
            self.assertTrue(
                (root / "checkpoints" / "svar.pt").exists()
            )
            _write_training_summaries(
                model="demo",
                baseline_dir=root,
                outputs=[output],
                result_paths=[result_path],
                result_stem="demo_svar_official",
            )
            markdown = (
                root / "results" / "demo_svar_official_summary.md"
            ).read_text(encoding="utf-8")
        self.assertIn("SVAR Official 结果汇总", markdown)
        self.assertIn("official_svar_set_first_token_id", markdown)
        self.assertIn("total=10，found=6，not_found=4", markdown)
        self.assertIn("hallucination=3，real=3", markdown)
        self.assertIn("train=2，val=2，test=2", markdown)


if __name__ == "__main__":
    unittest.main()
