from __future__ import annotations

import copy
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_and_eval import _torch_probe_cli_args, build_training_commands
from utils.config_utils import load_config


class UnifiedTrainStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            os.path.join(ROOT, "configs/model_configs_unified.yaml")
        )

    def test_all_mode_builds_root_and_baseline_training_commands(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "all"
        commands = build_training_commands(
            config=config,
            model="qwen3_vl_8b",
            config_path="configs/model_configs_unified.yaml",
            output_dir="outputs/qwen3_vl_8b/COCO4000-all",
            device="cuda:0",
        )
        self.assertEqual(len(commands), 5)
        roots, summary, baseline = commands[:3], commands[3], commands[4]
        self.assertEqual(
            [command[command.index("--seed") + 1] for command in roots],
            ["43", "44", "45"],
        )
        self.assertEqual(
            [command[command.index("--run-name") + 1] for command in roots],
            ["seed43", "seed44", "seed45"],
        )
        for root in roots:
            self.assertEqual(root[1], "scripts/train_torch_probe_feature_sets.py")
            self.assertIn("vpend_hpre_raw_logit_gauss_risk_sqrt_matched_state", root)
            self.assertIn("ads+cgc", root)
            self.assertEqual(root[root.index("--positive-class") + 1], "real")
            self.assertEqual(root[root.index("--batch-size") + 1], "256")
            self.assertEqual(root[root.index("--num-epochs") + 1], "100")
            self.assertEqual(
                root[root.index("--early-stopping-patience") + 1], "10"
            )
            hidden_start = root.index("--hidden-sizes") + 1
            self.assertEqual(
                root[hidden_start : hidden_start + 3], ["128", "64", "32"]
            )
        self.assertEqual(summary[1], "scripts/summarize_torch_probe_seed_runs.py")
        self.assertTrue(
            any(
                argument.endswith("/{model}_selected_feature_sets.json")
                for argument in summary
            )
        )
        self.assertEqual(baseline[1], "scripts/train_baselines.py")
        self.assertEqual(baseline[baseline.index("--device") + 1], "cuda:0")
        baseline_training = self.config["training"]["baseline"]
        self.assertEqual(baseline_training["seeds"], [43, 44, 45])
        self.assertNotIn("halloc", baseline_training["methods"])

    def test_method_mode_and_branch_switch_filter_training(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "method_only"
        config["feature_extraction"]["dgst_t"]["four_gate_methods"] = [
            "raw_attention"
        ]
        config["feature_extraction"]["dgst_t"]["support_modes"] = ["vv"]
        config["training"]["feature_sets"]["method"] = [
            "raw_attention_risk",
            "hpre_raw_logit_gauss_risk",
        ]
        commands = build_training_commands(
            config=config,
            model="qwen3_vl_8b",
            config_path="config.yaml",
            output_dir="output",
            device="cpu",
        )
        self.assertEqual(len(commands), 4)
        for command in commands[:3]:
            self.assertIn("raw_attention_risk", command)
            self.assertNotIn("ads", command)
            self.assertFalse(
                any("hpre_raw_logit_gauss" in argument for argument in command)
            )
        self.assertEqual(
            commands[-1][1], "scripts/summarize_torch_probe_seed_runs.py"
        )

    def test_baseline_only_builds_no_root_probe(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "baseline_only"
        commands = build_training_commands(
            config=config,
            model="qwen3_vl_8b",
            config_path="config.yaml",
            output_dir="output",
            device="cpu",
        )
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][1], "scripts/train_baselines.py")

    def test_single_torch_seed_uses_legacy_result_directory(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "all"
        config["training"]["torch_probe"]["seeds"] = [42]
        commands = build_training_commands(
            config=config,
            model="qwen3_vl_8b",
            config_path="config.yaml",
            output_dir="output",
            device="cpu",
        )
        self.assertEqual(len(commands), 2)
        root, baseline = commands
        self.assertEqual(root[1], "scripts/train_torch_probe_feature_sets.py")
        self.assertNotIn("--run-name", root)
        self.assertEqual(root[root.index("--seed") + 1], "42")
        self.assertEqual(baseline[1], "scripts/train_baselines.py")

    def test_torch_probe_drop_last_flag_follows_yaml_boolean(self) -> None:
        probe = copy.deepcopy(self.config["training"]["torch_probe"])
        probe["drop_last"] = True
        self.assertIn("--drop-last", _torch_probe_cli_args(probe))

        probe["drop_last"] = False
        self.assertIn("--no-drop-last", _torch_probe_cli_args(probe))

        probe["drop_last"] = "true"
        with self.assertRaisesRegex(ValueError, "drop_last must be a boolean"):
            _torch_probe_cli_args(probe)

    def test_feature_override_and_run_name_isolate_seed_outputs(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "method_only"
        config["feature_extraction"]["dgst_t"]["four_gate_methods"] = [
            "raw_attention"
        ]
        config["feature_extraction"]["dgst_t"]["support_modes"] = ["vv"]
        feature_set = (
            "raw_attention_source_target_js+"
            "raw_attention_ev_target_dist_mass_x_cosine"
        )
        commands = build_training_commands(
            config=config,
            model="qwen3_vl_8b",
            config_path="config.yaml",
            output_dir="output",
            device="cpu",
            feature_sets_override=[feature_set],
            run_name="js_ev",
        )
        self.assertEqual(len(commands), 4)
        for seed, command in zip((43, 44, 45), commands[:3]):
            self.assertEqual(
                command[command.index("--run-name") + 1], f"js_ev_seed{seed}"
            )
            self.assertIn(feature_set, command)
        summary = commands[-1]
        self.assertIn("js_ev_seed{seed}", summary[summary.index("--run-template") + 1])
        self.assertTrue(
            summary[summary.index("--output-prefix") + 1].endswith(
                "qwen3_vl_8b_js_ev_3seed_summary"
            )
        )


if __name__ == "__main__":
    unittest.main()
