from __future__ import annotations

import copy
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.train_and_eval import build_training_commands
from utils.config_utils import load_config


class UnifiedTrainStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(
            os.path.join(ROOT, "configs/model_configs_unified.yaml")
        )

    def test_all_mode_builds_root_and_baseline_training_commands(self) -> None:
        commands = build_training_commands(
            config=copy.deepcopy(self.config),
            model="qwen3_vl_8b",
            config_path="configs/model_configs_unified.yaml",
            output_dir="outputs/qwen3_vl_8b/COCO4000-all",
            device="cuda:0",
        )
        self.assertEqual(len(commands), 5)
        roots, summary, baseline = commands[:3], commands[3], commands[4]
        self.assertEqual(
            [command[command.index("--seed") + 1] for command in roots],
            ["42", "43", "44"],
        )
        self.assertEqual(
            [command[command.index("--run-name") + 1] for command in roots],
            ["seed42", "seed43", "seed44"],
        )
        for root in roots:
            self.assertEqual(root[1], "scripts/train_torch_probe_feature_sets.py")
            self.assertIn("raw_attention_risk", root)
            self.assertIn("ads+cgc", root)
            self.assertEqual(root[root.index("--positive-class") + 1], "hallucination")
            self.assertEqual(root[root.index("--batch-size") + 1], "256")
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
        self.assertEqual(baseline_training["seeds"], [42, 43, 44])
        self.assertNotIn("halloc", baseline_training["methods"])

    def test_method_mode_and_branch_switch_filter_training(self) -> None:
        config = copy.deepcopy(self.config)
        config["run"]["extraction_mode"] = "method_only"
        config["feature_extraction"]["dgst_t"]["four_gate_methods"] = [
            "raw_attention"
        ]
        for method in config["feature_extraction"]["dgst_t"]["branches"]:
            config["feature_extraction"]["dgst_t"]["branches"][method] = (
                method == "raw_attention"
            )
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


if __name__ == "__main__":
    unittest.main()
