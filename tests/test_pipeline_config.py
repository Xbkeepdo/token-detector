from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.config_utils import extraction_mode_flags, resolve_run_config
from utils.generation_provenance import build_generation_manifest
from utils.split_utils import (
    build_strict_82_split,
    build_strict_811_split,
    ensure_strict_811_split,
    validate_strict_811_split,
)
from scripts.run_pipeline import (
    _artifact_config_sha256,
    _apply_effective_feature_switches,
    _apply_runtime_overrides,
    _baseline_features_config_sha256,
    _effective_feature_flags,
    _feature_sets,
    _labeling_config_sha256,
    _prepare_fresh_artifacts,
    _require_complete_generations,
    _reuse_generation_artifacts,
    _root_features_config_sha256,
    _validate_or_write_manifest,
    build_baseline_extract_command,
    build_root_extract_command,
)
from scripts.extract_features import _resolve_extraction_mode


def _generation_rows(image_ids: list[int]) -> dict[str, dict[str, object]]:
    return {
        str(image_id): {
            "generated_text": f"caption {image_id}",
            "response_token_ids": [1000 + image_id, 2000 + image_id],
        }
        for image_id in image_ids
    }


def _minimal_config() -> dict:
    return {
        "run": {
            "model": "model_a",
            "output_dir": "outputs/model_a/COCO4000-vv",
            "prompt": "Describe this image.",
            "resume": True,
            "extraction_mode": "all",
        },
        "models": {"model_a": {"hf_name": "unused"}},
    }


class PipelineConfigTests(unittest.TestCase):
    def test_strict_82_is_deterministic_and_has_no_validation(self) -> None:
        image_ids = list(range(10_000, 14_000))
        shuffled = image_ids.copy()
        random.Random(999).shuffle(shuffled)
        first = build_strict_82_split(image_ids, seed=42)
        second = build_strict_82_split(shuffled, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            {"train": 3200, "val": 0, "test": 800},
        )

    def test_strict_811_is_deterministic_and_order_independent(self) -> None:
        image_ids = list(range(10_000, 14_000))
        shuffled = image_ids.copy()
        random.Random(999).shuffle(shuffled)

        first = build_strict_811_split(image_ids, seed=42)
        second = build_strict_811_split(shuffled, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            {"train": 3200, "val": 400, "test": 400},
        )
        self.assertEqual(
            validate_strict_811_split(first, expected_image_ids=image_ids),
            {"train": 3200, "val": 400, "test": 400},
        )

    def test_legacy_9010_split_is_backed_up_and_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            image_ids = list(range(4_000))
            legacy = {
                "train": image_ids[:3600],
                "val": image_ids[3600:],
                "test": image_ids[3600:],
            }
            path = tmp_path / "image_splits.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")

            installed, backup = ensure_strict_811_split(path, image_ids, seed=42)

            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertTrue(backup.exists())
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), legacy)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), installed)
            self.assertEqual(
                validate_strict_811_split(installed, expected_image_ids=image_ids),
                {"train": 3200, "val": 400, "test": 400},
            )

            unchanged, second_backup = ensure_strict_811_split(path, image_ids, seed=42)
            self.assertEqual(unchanged, installed)
            self.assertIsNone(second_backup)

    def test_shared_split_rejects_a_different_image_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            shared = tmp_path / "shared.json"
            output_a = tmp_path / "model_a.json"
            output_b = tmp_path / "model_b.json"
            image_ids = list(range(4_000))
            ensure_strict_811_split(
                output_a,
                image_ids,
                seed=42,
                shared_splits_path=shared,
            )

            other_ids = list(range(1, 4_001))
            with self.assertRaisesRegex(ValueError, "different COCO cohort"):
                ensure_strict_811_split(
                    output_b,
                    other_ids,
                    seed=42,
                    shared_splits_path=shared,
                )

    def test_split_validation_rejects_val_test_leakage(self) -> None:
        legacy = {
            "train": list(range(80)),
            "val": list(range(80, 90)),
            "test": list(range(80, 90)),
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_strict_811_split(legacy)

    def test_yaml_defaults_and_explicit_environment_overrides(self) -> None:
        config = _minimal_config()
        resolved = resolve_run_config(
            config,
            environ={
                "MODEL": "model_a",
                "OUTPUT": "outputs/override",
                "PROMPT": "Caption precisely.",
                "RESUME": "false",
                "EXTRACTION_MODE": "baseline_only",
                "DEVICE": "cuda:2",
                "GENERATION_DEVICES": "cuda:2 cuda:3",
                "FEATURE_DEVICES": "cuda:4,cuda:5",
                "STAGES": "feature_extraction training",
                "FEATURE_SETS": "risk cosine risk+cosine",
            },
        )

        self.assertEqual(resolved["output_dir"], "outputs/override")
        self.assertEqual(resolved["prompt"], "Caption precisely.")
        self.assertFalse(resolved["resume"])
        self.assertEqual(resolved["extraction_mode"], "baseline_only")
        self.assertEqual(resolved["devices"]["primary"], "cuda:2")
        self.assertEqual(resolved["devices"]["generation"], ["cuda:2", "cuda:3"])
        self.assertEqual(
            resolved["devices"]["feature_extraction"], ["cuda:4", "cuda:5"]
        )
        self.assertTrue(resolved["stages"]["feature_extraction"])
        self.assertTrue(resolved["stages"]["training"])
        self.assertFalse(resolved["stages"]["generation"])
        self.assertEqual(resolved["feature_sets"], ["risk", "cosine", "risk+cosine"])

    def test_thin_shell_runtime_overrides_are_resolved_in_python(self) -> None:
        config = _minimal_config()
        config["feature_extraction"] = {
            "dgst_t": {
                "four_gate_methods": ["hpre_raw_logit_gauss"],
                "branches": {"hpre_raw_logit_gauss": True},
            }
        }
        resolved = resolve_run_config(
            config,
            environ={
                "POSITIVE_CLASS": "hallucination",
                "ADOPT_LEGACY_ARTIFACTS": "true",
                "DGST_BRANCHES": "hpre_softmax_prob_gauss raw_attention",
                "MAX_PIXELS": "200704",
                "RUN_GENERATION_LABELING": "false",
                "RUN_FEATURE_EXTRACTION": "true",
                "RUN_TRAIN_EVAL": "false",
            },
        )
        self.assertEqual(resolved["positive_class"], "hallucination")
        self.assertTrue(resolved["adopt_legacy_artifacts"])
        self.assertEqual(
            resolved["dgst_branches"],
            ["hpre_softmax_prob_gauss", "raw_attention"],
        )
        self.assertEqual(resolved["max_pixels"], 200704)
        self.assertFalse(resolved["stages"]["generation"])
        self.assertFalse(resolved["stages"]["labeling"])
        self.assertTrue(resolved["stages"]["feature_extraction"])
        self.assertFalse(resolved["stages"]["training"])

        _apply_runtime_overrides(config, resolved)
        self.assertEqual(config["models"]["model_a"]["max_pixels"], 200704)
        dgst = config["feature_extraction"]["dgst_t"]
        self.assertEqual(
            dgst["four_gate_methods"],
            ["hpre_softmax_prob_gauss", "raw_attention"],
        )
        self.assertFalse(dgst["branches"]["hpre_raw_logit_gauss"])
        self.assertTrue(dgst["branches"]["hpre_softmax_prob_gauss"])
        self.assertTrue(dgst["branches"]["raw_attention"])

        command = build_root_extract_command(
            resolved,
            Path("resolved.yaml"),
            Path(resolved["output_dir"]),
        )
        self.assertEqual(command[command.index("--max-pixels") + 1], "200704")
        branch_index = command.index("--dgst-branches")
        self.assertEqual(
            command[branch_index + 1 : branch_index + 3],
            ["hpre_softmax_prob_gauss", "raw_attention"],
        )

    def test_thin_shell_rejects_unknown_dgst_branch_and_bad_pixel_cap(self) -> None:
        config = _minimal_config()
        with self.assertRaisesRegex(ValueError, "Unknown DGST branches"):
            resolve_run_config(config, environ={"DGST_BRANCHES": "not_a_method"})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            resolve_run_config(config, environ={"MAX_PIXELS": "0"})

    def test_extraction_mode_flags(self) -> None:
        cases = [
            ("all", {"method": True, "ads_cgc": True, "baseline": True}),
            ("method_only", {"method": True, "ads_cgc": False, "baseline": False}),
            ("ads_cgc_only", {"method": False, "ads_cgc": True, "baseline": False}),
            ("baseline_only", {"method": False, "ads_cgc": False, "baseline": True}),
        ]
        for mode, expected in cases:
            with self.subTest(mode=mode):
                self.assertEqual(extraction_mode_flags(mode), expected)

    def test_extract_stage_reads_mode_from_yaml_with_cli_override(self) -> None:
        config = {"run": {"extraction_mode": "all"}}
        self.assertEqual(_resolve_extraction_mode(config, None), "all")
        self.assertEqual(
            _resolve_extraction_mode(config, "method_only"),
            "method_only",
        )
        self.assertIsNone(_resolve_extraction_mode({}, None))

    def test_pipeline_modes_and_feature_sets_stay_isolated(self) -> None:
        configured = {
            "feature_extraction": {
                "method": {"enabled": True},
                "ads_cgc": {"enabled": True},
                "baseline": {"enabled": True},
            },
            "training": {
                "feature_sets": {
                    "method": ["method_risk"],
                    "ads_cgc": ["ads", "cgc", "ads+cgc"],
                }
            },
        }
        for mode, expected_flags, expected_sets in (
            (
                "all",
                {"method": True, "ads_cgc": True, "baseline": True},
                ["method_risk", "ads", "cgc", "ads+cgc"],
            ),
            (
                "method_only",
                {"method": True, "ads_cgc": False, "baseline": False},
                ["method_risk"],
            ),
            (
                "ads_cgc_only",
                {"method": False, "ads_cgc": True, "baseline": False},
                ["ads", "cgc", "ads+cgc"],
            ),
            (
                "baseline_only",
                {"method": False, "ads_cgc": False, "baseline": True},
                ["risk", "target_cosine", "risk+target_cosine"],
            ),
        ):
            with self.subTest(mode=mode):
                import copy

                config = copy.deepcopy(configured)
                _apply_effective_feature_switches(config, mode)
                self.assertEqual(_effective_feature_flags(config), expected_flags)
                self.assertEqual(_feature_sets(config, {"extraction_mode": mode}), expected_sets)

    def test_root_extract_command_has_one_well_formed_model_option(self) -> None:
        run = resolve_run_config(_minimal_config(), environ={})
        command = build_root_extract_command(
            run,
            Path("resolved.yaml"),
            Path("outputs/model_a/COCO4000-vv"),
        )
        self.assertEqual(command.count("--model"), 1)
        model_index = command.index("--model")
        self.assertEqual(command[model_index + 1], "model_a")

    def test_baseline_only_uses_the_provenance_protected_entrypoint(self) -> None:
        run = resolve_run_config(_minimal_config(), environ={})
        command = build_baseline_extract_command(
            run,
            Path("resolved.yaml"),
            Path("outputs/model_a/COCO4000-vv"),
        )
        self.assertEqual(command[1], "scripts/extract_features.py")
        self.assertEqual(
            command[command.index("--extraction-mode") + 1],
            "baseline_only",
        )

    def test_disabled_four_gate_branch_is_not_sent_to_training(self) -> None:
        config = {
            "feature_extraction": {
                "dgst_t": {
                    "branches": {
                        "hpre_raw_logit_gauss": True,
                        "hmid_raw_logit_gauss": False,
                    }
                }
            },
            "training": {
                "feature_sets": {
                    "method": [
                        "hpre_raw_logit_gauss_risk",
                        "hmid_raw_logit_gauss_risk",
                        "hpre_raw_logit_gauss_risk+hmid_raw_logit_gauss_ev",
                    ],
                    "ads_cgc": [],
                }
            },
        }
        selected = _feature_sets(config, {"extraction_mode": "method_only"})
        self.assertEqual(selected, ["hpre_raw_logit_gauss_risk"])

        # Removing a method from four_gate_methods is also an extraction switch,
        # even when its boolean branch entry remains true.
        config["feature_extraction"]["dgst_t"]["four_gate_methods"] = [
            "hmid_raw_logit_gauss"
        ]
        config["feature_extraction"]["dgst_t"]["branches"][
            "hmid_raw_logit_gauss"
        ] = True
        selected = _feature_sets(config, {"extraction_mode": "method_only"})
        self.assertEqual(selected, ["hmid_raw_logit_gauss_risk"])

    def test_resume_manifest_fingerprints_feature_configuration(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {"enabled": False, "output_subdir": "baseline"},
            "dgst_t": {"branches": {"hpre_raw_logit_gauss": True}},
        }
        run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "features.pkl").write_bytes(b"existing")
            _validate_or_write_manifest(output, run, config)

            changed = json.loads(json.dumps(config))
            changed["feature_extraction"]["dgst_t"]["branches"][
                "hpre_raw_logit_gauss"
            ] = False
            self.assertNotEqual(
                _artifact_config_sha256(config, run),
                _artifact_config_sha256(changed, run),
            )
            self.assertNotEqual(
                _root_features_config_sha256(config, run),
                _root_features_config_sha256(changed, run),
            )
            with self.assertRaisesRegex(ValueError, "Refusing to resume"):
                _validate_or_write_manifest(output, run, changed)

    def test_labeling_protocol_and_artifact_content_are_resume_keys(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 1, "seed": 42}
        config["labeling"] = {
            "schema_version": 2,
            "sample_unit": "first_canonical_mention",
            "primary_locator": "exact_response_offsets",
        }
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {"enabled": False, "output_subdir": "baseline"},
        }
        run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        changed_locator = json.loads(json.dumps(config))
        changed_locator["labeling"]["primary_locator"] = "first_token_id"
        self.assertNotEqual(
            _labeling_config_sha256(config, run),
            _labeling_config_sha256(changed_locator, run),
        )
        changed_protocols = json.loads(json.dumps(config))
        changed_protocols["feature_extraction"]["baseline"]["svar"] = {
            "protocols": ["controlled", "official"]
        }
        self.assertEqual(
            _labeling_config_sha256(config, run),
            _labeling_config_sha256(changed_protocols, run),
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "labeling.json").write_text(
                '{"1":{"schema_version":2}}',
                encoding="utf-8",
            )
            (output / "labeling_manifest.json").write_text(
                json.dumps(
                    {
                        "label_schema_version": 2,
                        "sample_unit": "first_canonical_mention",
                        "primary_locator": "exact_response_offsets",
                    }
                ),
                encoding="utf-8",
            )
            _validate_or_write_manifest(output, run, config)
            first = json.loads(
                (output / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first["manifest_version"], 4)
            self.assertEqual(first["labeling_schema_version"], "2")
            self.assertEqual(
                first["labeling_primary_locator"],
                "exact_response_offsets",
            )

            (output / "labeling.json").write_text(
                '{"1":{"schema_version":2,"changed":true}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "labeling_sha256"):
                _validate_or_write_manifest(output, run, config)

    def test_manifest_accepts_new_label_artifact_from_enabled_stage(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 1, "seed": 42}
        config["labeling"] = {
            "schema_version": 2,
            "sample_unit": "first_canonical_mention",
            "primary_locator": "exact_response_offsets",
        }
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {"enabled": False},
        }
        run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _validate_or_write_manifest(output, run, config)
            (output / "labeling.json").write_text(
                '{"1":{"schema_version":2}}',
                encoding="utf-8",
            )
            _validate_or_write_manifest(
                output,
                run,
                config,
                allow_labeling_update=True,
            )
            manifest = json.loads(
                (output / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNotNone(manifest["labeling_sha256"])

    def test_reuse_generations_copies_only_validated_generation_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            expected_ids = list(range(10, 20))
            generations = _generation_rows(expected_ids)
            model_cfg = {"hf_name": "unused"}
            (source / "baseline").mkdir(parents=True)
            (source / "generations.json").write_text(
                json.dumps(generations),
                encoding="utf-8",
            )
            (source / "generation_manifest.json").write_text(
                json.dumps(
                    build_generation_manifest(
                        model="model_a",
                        model_cfg=model_cfg,
                        prompt="Describe this image.",
                        generations=generations,
                        expected_image_ids=expected_ids,
                    )
                ),
                encoding="utf-8",
            )
            (source / "image_splits.json").write_text(
                json.dumps(build_strict_82_split(expected_ids, seed=42)),
                encoding="utf-8",
            )
            (source / "labeling.json").write_text("{}", encoding="utf-8")
            (source / "features.pkl").write_bytes(b"root")
            (source / "baseline" / "features.pkl").write_bytes(b"baseline")

            _reuse_generation_artifacts(
                source,
                destination,
                model="model_a",
                model_cfg=model_cfg,
                prompt="Describe this image.",
                expected_image_ids=expected_ids,
            )

            self.assertTrue((destination / "generations.json").exists())
            self.assertTrue((destination / "generation_manifest.json").exists())
            self.assertTrue((destination / "image_splits.json").exists())
            self.assertFalse((destination / "labeling.json").exists())
            self.assertFalse((destination / "features.pkl").exists())
            self.assertFalse((destination / "baseline").exists())

            config = _minimal_config()
            config["dataset"] = {"num_images": len(expected_ids), "seed": 42}
            config["feature_extraction"] = {
                "method": {"enabled": True},
                "ads_cgc": {"enabled": False},
                "baseline": {"enabled": False},
            }
            run = {
                **config["run"],
                "extraction_mode": "method_only",
                "resume": True,
                "adopt_legacy_artifacts": False,
                "reuse_generations_from": str(source),
            }
            _validate_or_write_manifest(destination, run, config)
            manifest = json.loads(
                (destination / "pipeline_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["generation_reuse_source"],
                str(source),
            )

    def test_pipeline_reuse_requires_matching_manifest_cohort_and_response_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            expected_ids = [1, 2, 3]
            generations = _generation_rows(expected_ids)
            (source / "generations.json").write_text(
                json.dumps(generations), encoding="utf-8"
            )
            (source / "generation_manifest.json").write_text(
                json.dumps(
                    build_generation_manifest(
                        model="model_a",
                        model_cfg={"hf_name": "unused"},
                        prompt="Wrong prompt.",
                        generations=generations,
                        expected_image_ids=expected_ids,
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model/prompt/content"):
                _reuse_generation_artifacts(
                    source,
                    root / "wrong_prompt",
                    model="model_a",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    expected_image_ids=expected_ids,
                )

            (source / "generation_manifest.json").write_text(
                json.dumps(
                    build_generation_manifest(
                        model="model_a",
                        model_cfg={"hf_name": "different-checkpoint"},
                        prompt="Describe this image.",
                        generations=generations,
                        expected_image_ids=expected_ids,
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model/prompt/content"):
                _reuse_generation_artifacts(
                    source,
                    root / "wrong_model_config",
                    model="model_a",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    expected_image_ids=expected_ids,
                )

            bad_cohort = dict(generations)
            bad_cohort["4"] = bad_cohort.pop("3")
            (source / "generations.json").write_text(
                json.dumps(bad_cohort), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "selected image cohort"):
                _reuse_generation_artifacts(
                    source,
                    root / "wrong_cohort",
                    model="model_a",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    expected_image_ids=expected_ids,
                )

            no_ids = _generation_rows(expected_ids)
            no_ids["2"]["response_token_ids"] = []
            (source / "generations.json").write_text(
                json.dumps(no_ids), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "response_token_ids"):
                _reuse_generation_artifacts(
                    source,
                    root / "missing_ids",
                    model="model_a",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    expected_image_ids=expected_ids,
                )

    def test_pipeline_reuse_complete_generations_auto_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            expected_ids = [1, 2, 3]
            (source / "generations.json").write_text(
                json.dumps(_generation_rows(expected_ids)), encoding="utf-8"
            )
            common = {
                "model": "model_a",
                "model_cfg": {"hf_name": "unused"},
                "prompt": "Describe this image.",
                "expected_image_ids": expected_ids,
            }

            destination = root / "reused"
            _reuse_generation_artifacts(
                source,
                destination,
                **common,
            )
            manifest = json.loads(
                (destination / "generation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertNotIn("adopted_legacy_source", manifest)

    def test_generation_disabled_labeling_validates_complete_manifest_and_content(
        self,
    ) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 3, "seed": 42}
        run = {
            **config["run"],
            "adopt_legacy_artifacts": False,
        }
        expected_ids = [1, 2, 3]
        generations = _generation_rows(expected_ids)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "generations.json").write_text(
                json.dumps(generations), encoding="utf-8"
            )
            registered = _require_complete_generations(
                output,
                config=config,
                run=run,
                expected_image_ids=expected_ids,
            )
            self.assertEqual(registered["status"], "complete")
            self.assertTrue((output / "generation_manifest.json").is_file())

            tampered = _generation_rows(expected_ids)
            tampered["2"]["response_token_ids"][0] += 1
            (output / "generations.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "model/prompt/content"):
                _require_complete_generations(
                    output,
                    config=config,
                    run=run,
                    expected_image_ids=expected_ids,
                )

    def test_labeling_update_cannot_bless_retained_features(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 1, "seed": 42}
        config["labeling"] = {
            "schema_version": 2,
            "sample_unit": "first_canonical_mention",
            "primary_locator": "exact_response_offsets",
        }
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {"enabled": False},
        }
        run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _validate_or_write_manifest(output, run, config)
            (output / "features.pkl").write_bytes(b"stale")
            (output / "labeling.json").write_text(
                '{"1":{"schema_version":2}}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "retained feature artifacts"):
                _validate_or_write_manifest(
                    output,
                    run,
                    config,
                    allow_labeling_update=True,
                )

    def test_pipeline_reuse_refuses_existing_downstream_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            expected_ids = [1, 2, 3]
            (source / "generations.json").write_text(
                json.dumps(_generation_rows(expected_ids)), encoding="utf-8"
            )
            (destination / "labeling.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "downstream artifacts"):
                _reuse_generation_artifacts(
                    source,
                    destination,
                    model="model_a",
                    model_cfg={"hf_name": "unused"},
                    prompt="Describe this image.",
                    expected_image_ids=expected_ids,
                    resume=False,
                )

    def test_pipeline_baseline_hash_ignores_training_only_settings(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "baseline": {
                "enabled": True,
                "methods": ["metatoken", "svar", "dhcp", "projectaway", "halloc"],
                "metatoken": {"classifiers": ["lr"], "gb_n_estimators": 100},
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
                    "hidden_dim": 128,
                    "batch_size": 1024,
                    "epochs": 30,
                },
                "projectaway": {"detection_only": True},
                "halloc": {
                    "clip_model": "clip-a",
                    "visualbert_model": "visualbert-a",
                    "learning_rate": 1e-6,
                    "batch_size": 16,
                    "epochs": 25,
                },
            }
        }
        run = {**config["run"], "extraction_mode": "all"}
        original = _baseline_features_config_sha256(config, run)

        training_change = json.loads(json.dumps(config))
        baseline = training_change["feature_extraction"]["baseline"]
        baseline["metatoken"]["classifiers"] = ["gb", "mlp"]
        baseline["svar"].update(
            hidden_dim=999,
            learning_rate=0.5,
            batch_size=3,
            epochs=2,
            early_stopping_patience=1,
        )
        baseline["dhcp"].update(hidden_dim=999, batch_size=3, epochs=2)
        baseline["projectaway"]["detection_only"] = False
        baseline["halloc"].update(
            visualbert_model="visualbert-b",
            learning_rate=0.5,
            batch_size=3,
            epochs=2,
        )
        self.assertEqual(
            original,
            _baseline_features_config_sha256(training_change, run),
        )

        extraction_change = json.loads(json.dumps(config))
        extraction_change["feature_extraction"]["baseline"]["svar"][
            "layer_start"
        ] = 6
        self.assertNotEqual(
            original,
            _baseline_features_config_sha256(extraction_change, run),
        )

    def test_manifest_keeps_root_and_baseline_provenance_independent(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {
                "enabled": False,
                "output_subdir": "baseline",
                "methods": ["metatoken"],
            },
            "dgst_t": {"branches": {"hpre_raw_logit_gauss": True}},
        }
        method_run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "generations.json").write_text("{}", encoding="utf-8")
            (output / "labeling.json").write_text("{}", encoding="utf-8")
            (output / "features.pkl").write_bytes(b"method")
            _validate_or_write_manifest(output, method_run, config)
            first = json.loads(
                (output / "pipeline_manifest.json").read_text(encoding="utf-8")
            )

            baseline_config = json.loads(json.dumps(config))
            baseline_config["feature_extraction"]["method"]["enabled"] = False
            baseline_config["feature_extraction"]["baseline"]["enabled"] = True
            baseline_run = {
                **method_run,
                "extraction_mode": "baseline_only",
            }
            _validate_or_write_manifest(output, baseline_run, baseline_config)
            second = json.loads(
                (output / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                second["root_features_config_sha256"],
                first["root_features_config_sha256"],
            )
            self.assertEqual(
                second["baseline_features_config_sha256"],
                _baseline_features_config_sha256(baseline_config, baseline_run),
            )

            baseline_dir = output / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "features.pkl").write_bytes(b"baseline")
            _validate_or_write_manifest(output, method_run, config)
            third = json.loads(
                (output / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                third["root_features_config_sha256"],
                first["root_features_config_sha256"],
            )
            self.assertEqual(
                third["baseline_features_config_sha256"],
                second["baseline_features_config_sha256"],
            )

    def test_generation_label_manifest_can_later_add_method_features(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": True},
            "baseline": {"enabled": True, "output_subdir": "baseline"},
            "dgst_t": {"branches": {"hpre_raw_logit_gauss": True}},
        }
        all_run = {
            **config["run"],
            "extraction_mode": "all",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "generations.json").write_text("{}", encoding="utf-8")
            (output / "labeling.json").write_text("{}", encoding="utf-8")
            _validate_or_write_manifest(output, all_run, config)

            method_config = json.loads(json.dumps(config))
            _apply_effective_feature_switches(method_config, "method_only")
            method_run = {**all_run, "extraction_mode": "method_only"}
            # No root artifact exists yet, so changing the active extraction
            # family is valid while generations/labels are reused.
            _validate_or_write_manifest(output, method_run, method_config)

    def test_manifest_fingerprints_partial_extraction_and_generation_shards(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": False},
            "baseline": {"enabled": False, "output_subdir": "baseline"},
            "dgst_t": {"branches": {"hpre_raw_logit_gauss": True}},
        }
        run = {
            **config["run"],
            "extraction_mode": "method_only",
            "resume": True,
            "adopt_legacy_artifacts": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "generation_shards").mkdir()
            (output / "generation_shards" / "worker_0.jsonl").write_text(
                '{"image_id": 1}\n', encoding="utf-8"
            )
            (output / "features.part0.pkl").write_bytes(b"partial")
            _validate_or_write_manifest(output, run, config)

            changed_branch = json.loads(json.dumps(config))
            changed_branch["feature_extraction"]["dgst_t"]["branches"][
                "hpre_raw_logit_gauss"
            ] = False
            with self.assertRaisesRegex(ValueError, "root_features_config_sha256"):
                _validate_or_write_manifest(output, run, changed_branch)

            changed_prompt_run = {**run, "prompt": "A different prompt."}
            with self.assertRaisesRegex(ValueError, "generation_config_sha256"):
                _validate_or_write_manifest(output, changed_prompt_run, config)

    def test_legacy_resume_requires_explicit_adoption(self) -> None:
        config = _minimal_config()
        config["dataset"] = {"num_images": 4000, "seed": 42}
        config["feature_extraction"] = {
            "baseline": {"output_subdir": "baseline"}
        }
        run = {
            **config["run"],
            "resume": True,
            "adopt_legacy_artifacts": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "generations.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no pipeline_manifest"):
                _validate_or_write_manifest(output, run, config)

    def test_no_resume_generation_cleanup_removes_generation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in (
                "generations.json",
                "generation_manifest.json",
                "labeling.json",
                "labeling_manifest.json",
            ):
                (output / name).write_text("{}", encoding="utf-8")
            (output / "generation_shards").mkdir()
            (output / "generation_shards" / "worker_0.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            run = {
                "stages": {
                    "generation": True,
                    "labeling": True,
                    "feature_extraction": False,
                    "training": False,
                    "plotting": False,
                }
            }
            config = {
                "feature_extraction": {
                    "method": {"enabled": False},
                    "ads_cgc": {"enabled": False},
                    "baseline": {"enabled": False},
                }
            }
            _prepare_fresh_artifacts(output, run, config)
            self.assertFalse((output / "generations.json").exists())
            self.assertFalse((output / "generation_manifest.json").exists())
            self.assertFalse((output / "labeling.json").exists())
            self.assertFalse((output / "labeling_manifest.json").exists())
            self.assertFalse((output / "generation_shards").exists())

    def test_no_resume_baseline_cleanup_preserves_root_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root_features = output / "features.pkl"
            root_features.write_bytes(b"root")
            baseline = output / "baseline"
            (baseline / "feature_parts").mkdir(parents=True)
            (baseline / "feature_parts" / "worker_0.pkl").write_bytes(b"part")
            (baseline / "features.pkl").write_bytes(b"baseline")
            (baseline / "dhcp").mkdir()
            (baseline / "halloc").mkdir()
            run = {
                "stages": {
                    "generation": False,
                    "labeling": False,
                    "feature_extraction": True,
                    "training": False,
                    "plotting": False,
                }
            }
            config = {
                "feature_extraction": {
                    "method": {"enabled": False},
                    "ads_cgc": {"enabled": False},
                    "baseline": {
                        "enabled": True,
                        "output_subdir": "baseline",
                    },
                }
            }
            _prepare_fresh_artifacts(output, run, config)
            self.assertEqual(root_features.read_bytes(), b"root")
            self.assertFalse((baseline / "features.pkl").exists())
            self.assertFalse((baseline / "feature_parts").exists())
            self.assertFalse((baseline / "dhcp").exists())
            self.assertFalse((baseline / "halloc").exists())


if __name__ == "__main__":
    unittest.main()
