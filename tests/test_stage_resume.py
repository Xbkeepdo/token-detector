from __future__ import annotations

import importlib.util
import json
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
    _controlled_baseline_feature_config,
    _feature_model_config_payload,
    _feature_provenance,
    _has_extractable_object_spans,
    _official_svar_feature_config,
    _sample_has_any_extractable_protocol,
    _pending_samples_for_resume,
    _resolve_prompt,
    _stable_sha256,
    _validate_or_write_feature_manifest,
)
from utils.generation_provenance import build_generation_manifest
from utils.io_utils import save_json, save_pkl


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

    def test_prompt_resolution_prefers_cli_then_unified_run_yaml(self) -> None:
        config = {"run": {"prompt": "run prompt"}}
        model_cfg = {"prompt": "model prompt"}
        self.assertEqual(
            _resolve_prompt(
                cli_prompt="CLI prompt",
                config=config,
                model_cfg=model_cfg,
            ),
            "CLI prompt",
        )
        self.assertEqual(
            _resolve_prompt(
                cli_prompt=None,
                config=config,
                model_cfg=model_cfg,
            ),
            "run prompt",
        )
        self.assertEqual(
            _resolve_prompt(
                cli_prompt=None,
                config={},
                model_cfg=model_cfg,
            ),
            "model prompt",
        )
        self.assertEqual(
            _resolve_prompt(cli_prompt=None, config={}, model_cfg={}),
            "Describe this image.",
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
        generation = {"generated_text": "a chair", "response_token_ids": [10, 11]}

        for helper in (_has_extractable_object_spans, baseline_has_extractable_spans):
            self.assertTrue(helper(valid, generation))
            self.assertFalse(helper(empty, generation))
        with self.assertRaisesRegex(ValueError, "outside response length"):
            _has_extractable_object_spans(out_of_range, generation)
        with self.assertRaisesRegex(ValueError, "outside response length"):
            baseline_has_extractable_spans(out_of_range, generation)

    def test_official_found_survives_controlled_skip_prefilter(self) -> None:
        label = {
            "image_id": 1,
            "generated_text": "officer",
            "object_token_spans": [],
            "official_svar_samples": [
                {
                    "word": "officer",
                    "label": 1,
                    "status": "found",
                    "token_indices": [0],
                    "token_location": {
                        "status": "found",
                        "token_indices": [0],
                        "query_token_id": 17,
                    },
                }
            ],
        }
        generation = {
            "generated_text": "officer",
            "response_token_ids": [17],
        }
        self.assertFalse(
            _sample_has_any_extractable_protocol(
                label_info=label,
                generation=generation,
                controlled_enabled=True,
                official_enabled=False,
            )
        )
        self.assertTrue(
            _sample_has_any_extractable_protocol(
                label_info=label,
                generation=generation,
                controlled_enabled=True,
                official_enabled=True,
            )
        )

    def test_ground_truth_hash_guards_labeling_resume(self) -> None:
        samples = [{"image_id": 1}]
        generations = {
            "1": {
                "generated_text": "a chair",
                "response_token_ids": [7, 8],
            }
        }
        labeling = {
            "1": {
                "schema_version": 2,
                "image_id": 1,
                "generated_text": "a chair",
                "hallucinated_words": [],
                "real_words": ["chair"],
                "object_token_spans": [],
                "all_object_token_spans": [],
                "official_svar_samples": [],
                "chair_s": 0,
                "chair_i": 0.0,
            }
        }
        expected_manifest = {
            "label_schema_version": 2,
            "sample_unit": "first_canonical_mention",
            "primary_locator": "exact_response_offsets",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "labeling_manifest.json"
            manifest = {
                **expected_manifest,
                "labeling_sha256": self.label_coco._json_sha256(labeling),
            }
            save_json(manifest, str(manifest_path))
            ground_truth_path = root / "coco_ground_truth.jsonl"
            ground_truth_path.write_text(
                json.dumps({"image_id": 1, "objects": ["chair"]}) + "\n",
                encoding="utf-8",
            )
            kwargs = {
                "samples": samples,
                "generations": generations,
                "labeling": labeling,
                "generations_path": str(root / "generations.json"),
                "summary_path": str(root / "chair_summary.json"),
                "manifest_path": str(manifest_path),
                "expected_manifest": expected_manifest,
                "ground_truth_path": str(ground_truth_path),
            }
            self.assertFalse(
                self.label_coco._reuse_complete_labeling(**kwargs)
            )
            self.assertTrue(
                self.label_coco._reuse_complete_labeling(
                    **kwargs,
                    adopt_legacy_ground_truth=True,
                )
            )
            migrated = self.label_coco.load_json(str(manifest_path))
            self.assertIn("ground_truth_sha256", migrated)
            ground_truth_path.write_text(
                json.dumps({"image_id": 2, "objects": []}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(
                self.label_coco._reuse_complete_labeling(**kwargs)
            )

    def test_feature_model_hash_is_composable_across_extraction_modes(self) -> None:
        base = {
            "hf_name": "model",
            "num_layers": 32,
            "prompt": "Describe this image.",
            "generation_prompt": "Describe this image.",
            "dgst_t_support_scope": "visual",
        }
        all_mode = {**base, "extraction_mode": "all"}
        baseline_only = {**base, "extraction_mode": "baseline_only"}
        method_only = {**base, "extraction_mode": "method_only"}
        self.assertEqual(
            _feature_model_config_payload(
                all_mode, "baseline_controlled"
            ),
            _feature_model_config_payload(
                baseline_only, "baseline_controlled"
            ),
        )
        self.assertEqual(
            _feature_model_config_payload(all_mode, "root"),
            _feature_model_config_payload(method_only, "root"),
        )
        changed_scope = {**all_mode, "dgst_t_support_scope": "visual_prompt"}
        self.assertNotEqual(
            _feature_model_config_payload(all_mode, "root"),
            _feature_model_config_payload(changed_scope, "root"),
        )
        self.assertEqual(
            _feature_model_config_payload(
                all_mode, "baseline_svar_official"
            ),
            _feature_model_config_payload(
                changed_scope, "baseline_svar_official"
            ),
        )

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
            self.assertEqual(
                pending[0]["_feature_families_needed"],
                {"root": False, "controlled": True, "official": False},
            )
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

    def test_official_svar_resume_only_requires_images_with_found_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{"image_id": 1}, {"image_id": 2}]
            root_output = root / "features.pkl"
            baseline_output = root / "baseline" / "features.pkl"
            official_output = root / "baseline" / "svar_official" / "features.pkl"
            complete = [{"image_id": 1}, {"image_id": 2}]
            save_pkl(complete, str(root_output))
            save_pkl(complete, str(baseline_output))
            save_pkl([{"image_id": 1}], str(official_output))

            pending, complete_count = _pending_samples_for_resume(
                samples=samples,
                root_output_path=str(root_output),
                root_part_paths=[],
                baseline_output_path=str(baseline_output),
                baseline_part_paths=[],
                baseline_official_output_path=str(official_output),
                baseline_official_part_paths=[],
                baseline_official_required_image_ids={1},
            )
            self.assertEqual(pending, [])
            self.assertEqual(complete_count, 2)

            pending, complete_count = _pending_samples_for_resume(
                samples=samples,
                root_output_path=str(root_output),
                root_part_paths=[],
                baseline_output_path=str(baseline_output),
                baseline_part_paths=[],
                baseline_official_output_path=str(official_output),
                baseline_official_part_paths=[],
                baseline_official_required_image_ids={1, 2},
            )
            self.assertEqual([sample["image_id"] for sample in pending], [2])
            self.assertEqual(
                pending[0]["_feature_families_needed"],
                {"root": False, "controlled": False, "official": True},
            )
            self.assertEqual(complete_count, 1)

    def test_baseline_feature_fingerprint_ignores_training_only_settings(self) -> None:
        baseline_cfg = {
            "enabled": True,
            "output_subdir": "baseline",
            "methods": ["metatoken", "svar", "dhcp", "projectaway", "halloc"],
            "seed": 42,
            "metatoken": {
                "classifiers": ["lr", "gb"],
                "gb_n_estimators": 100,
                "length_penalty": 1.0,
                "attention_layer": -1,
            },
            "svar": {
                "protocols": ["controlled", "official"],
                "layer_start": 5,
                "layer_end": 19,
                "hidden_dim": 248,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "early_stopping_patience": 5,
            },
            "dhcp": {
                "spatial_size": [12, 12],
                "shard_size": 256,
                "hidden_dim": 128,
                "learning_rate": 0.001,
                "batch_size": 1024,
                "epochs": 30,
                "early_stopping_patience": 5,
            },
            "projectaway": {"detection_only": True},
            "halloc": {
                "clip_model": "clip-a",
                "visualbert_model": "visualbert-a",
                "freeze_clip": True,
                "learning_rate": 1e-6,
                "batch_size": 16,
                "epochs": 25,
                "early_stopping_patience": 3,
            },
        }
        changed_training = json.loads(json.dumps(baseline_cfg))
        changed_training["seed"] = 44
        changed_training["metatoken"]["classifiers"] = ["mlp"]
        changed_training["metatoken"]["gb_n_estimators"] = 999
        changed_training["svar"].update(
            hidden_dim=999,
            learning_rate=0.5,
            batch_size=3,
            epochs=2,
            early_stopping_patience=1,
        )
        changed_training["dhcp"].update(
            hidden_dim=999,
            learning_rate=0.5,
            batch_size=3,
            epochs=2,
            early_stopping_patience=1,
        )
        changed_training["projectaway"]["detection_only"] = False
        changed_training["halloc"].update(
            visualbert_model="visualbert-b",
            freeze_clip=False,
            learning_rate=0.5,
            batch_size=3,
            epochs=2,
            early_stopping_patience=1,
        )
        self.assertEqual(
            _controlled_baseline_feature_config(baseline_cfg),
            _controlled_baseline_feature_config(changed_training),
        )
        self.assertEqual(
            _official_svar_feature_config(baseline_cfg),
            _official_svar_feature_config(changed_training),
        )

        changed_training_layers = json.loads(json.dumps(baseline_cfg))
        changed_training_layers["svar"]["layer_start"] = 6
        self.assertEqual(
            _controlled_baseline_feature_config(baseline_cfg),
            _controlled_baseline_feature_config(changed_training_layers),
        )
        self.assertEqual(
            _official_svar_feature_config(baseline_cfg),
            _official_svar_feature_config(changed_training_layers),
        )

        official_only = {
            "methods": ["svar"],
            "svar": {"protocols": ["official"]},
        }
        controlled_enabled = {
            "methods": ["svar"],
            "svar": {"protocols": ["controlled", "official"]},
        }
        self.assertEqual(
            _controlled_baseline_feature_config(official_only)["methods"], []
        )
        self.assertEqual(
            _controlled_baseline_feature_config(controlled_enabled)["methods"],
            ["svar"],
        )

    def test_feature_manifest_rejects_changed_labeling_content_and_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labeling_payload = {
                "1": {
                    "schema_version": 2,
                    "generated_text": "chair",
                }
            }
            generation_payload = {
                "1": {
                    "generated_text": "chair",
                    "response_token_ids": [7],
                }
            }
            (root / "labeling.json").write_text(json.dumps(labeling_payload))
            (root / "generations.json").write_text(json.dumps(generation_payload))
            (root / "generation_manifest.json").write_text(
                json.dumps(
                    build_generation_manifest(
                        model="model",
                        model_cfg={"hf_name": "unused"},
                        prompt="Describe this image.",
                        generations=generation_payload,
                        expected_image_ids={1},
                    )
                )
            )
            (root / "labeling_manifest.json").write_text(
                json.dumps(
                    {
                        "label_schema_version": 2,
                        "primary_locator": "exact_response_offsets",
                        "sample_unit": "first_canonical_mention",
                        "labeling_sha256": _stable_sha256(labeling_payload),
                        "generation_sha256": _stable_sha256(generation_payload),
                    }
                )
            )
            config = {
                "labeling": {
                    "schema_version": 2,
                    "primary_locator": "exact_response_offsets",
                    "sample_unit": "first_canonical_mention",
                }
            }
            provenance = _feature_provenance(
                artifact_family="root",
                model_key="model",
                model_cfg={"hf_name": "unused"},
                prompt="Describe this image.",
                feature_config={"method": True},
                output_dir=str(root),
                config=config,
            )
            with self.assertRaisesRegex(RuntimeError, "generation_manifest.json"):
                _feature_provenance(
                    artifact_family="root",
                    model_key="model",
                    model_cfg={"hf_name": "unused"},
                    prompt="A different prompt.",
                    feature_config={"method": True},
                    output_dir=str(root),
                    config=config,
                )
            manifest = root / "features_manifest.json"
            output = root / "features.pkl"
            _validate_or_write_feature_manifest(
                str(manifest),
                provenance,
                artifact_paths=[str(output)],
                resume=True,
                adopt_legacy=False,
            )
            save_pkl([{"image_id": 1}], str(output))

            changed_labeling_payload = {
                "1": {
                    "schema_version": 2,
                    "generated_text": "chair",
                    "x": 1,
                }
            }
            (root / "labeling.json").write_text(
                json.dumps(changed_labeling_payload)
            )
            changed_labeling_manifest = {
                "label_schema_version": 2,
                "primary_locator": "exact_response_offsets",
                "sample_unit": "first_canonical_mention",
                "labeling_sha256": _stable_sha256(changed_labeling_payload),
                "generation_sha256": _stable_sha256(generation_payload),
            }
            (root / "labeling_manifest.json").write_text(
                json.dumps(changed_labeling_manifest)
            )
            changed_content = _feature_provenance(
                artifact_family="root",
                model_key="model",
                model_cfg={"hf_name": "unused"},
                prompt="Describe this image.",
                feature_config={"method": True},
                output_dir=str(root),
                config=config,
            )
            with self.assertRaisesRegex(ValueError, "labeling_sha256"):
                _validate_or_write_feature_manifest(
                    str(manifest),
                    changed_content,
                    artifact_paths=[str(output)],
                    resume=True,
                    adopt_legacy=False,
                )

            changed_config = {
                "labeling": {
                    **config["labeling"],
                    "primary_locator": "first_token_id",
                }
            }
            changed_locator = _feature_provenance(
                artifact_family="root",
                model_key="model",
                model_cfg={"hf_name": "unused"},
                prompt="Describe this image.",
                feature_config={"method": True},
                output_dir=str(root),
                config=changed_config,
            )
            self.assertNotEqual(
                provenance["labeling_config_sha256"],
                changed_locator["labeling_config_sha256"],
            )

    def test_feature_provenance_never_trusts_yaml_without_label_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labeling.json").write_text(
                '{"1":{"schema_version":2}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "labeling_manifest.json"):
                _feature_provenance(
                    artifact_family="root",
                    model_key="model",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    feature_config={"method": True},
                    output_dir=str(root),
                    config={
                        "labeling": {
                            "schema_version": 2,
                            "primary_locator": "exact_response_offsets",
                        }
                    },
                )

    def test_legacy_feature_file_cannot_be_adopted_into_v2_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "features.pkl"
            save_pkl([{"image_id": 1}], str(output))
            with self.assertRaisesRegex(ValueError, "adoption is intentionally disabled"):
                _validate_or_write_feature_manifest(
                    str(root / "features_manifest.json"),
                    {
                        "manifest_version": 1,
                        "artifact_family": "root",
                        "labeling_schema_version": "2",
                        "labeling_primary_locator": "exact_response_offsets",
                    },
                    artifact_paths=[str(output)],
                    resume=True,
                    adopt_legacy=True,
                )


    def test_atomic_json_failure_keeps_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labeling.json"
            path.write_text("{\"old\": true}", encoding="utf-8")
            circular = []
            circular.append(circular)
            with self.assertRaises(ValueError):
                save_json(circular, str(path))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"old": True},
            )
            self.assertEqual(list(Path(directory).glob(".tmp-*.json")), [])


if __name__ == "__main__":
    unittest.main()
