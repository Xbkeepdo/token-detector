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
from utils.split_utils import (
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
    _prepare_fresh_artifacts,
    _root_features_config_sha256,
    _validate_or_write_manifest,
    build_root_extract_command,
)
from scripts.extract_features import _resolve_extraction_mode


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
