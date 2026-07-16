from __future__ import annotations

import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.run_pipeline import _enabled_method_feature_sets
from train_feature_sets import build_selected_matrix, parse_feature_set
from utils.config_utils import load_config


class RawAttentionIntegrationTests(unittest.TestCase):
    def test_training_aliases_use_the_standard_dgst_output_keys(self) -> None:
        feature_set = (
            "raw_attention_risk+raw_attention_target_cosine+raw_attention_ev"
        )
        blocks = parse_feature_set(feature_set)
        self.assertEqual(
            blocks,
            [
                "raw_attention_risk",
                "raw_attention_target_cosine",
                "raw_attention_ev",
            ],
        )

        record = {
            "label": 1,
            "dgst_t_raw_attention_risk_sqrt_hpre_per_layer": [0.1, 0.2],
            "dgst_t_raw_attention_target_cosine_topk32_hpre_per_layer": [
                0.3,
                0.4,
            ],
            "dgst_t_raw_attention_ev_topk32_hpre_per_layer": [0.5, 0.6],
        }
        matrix, labels = build_selected_matrix([record], blocks)
        np.testing.assert_allclose(
            matrix,
            np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        )
        self.assertEqual(labels.tolist(), [1])

    def test_runtime_branch_filter_handles_raw_attention(self) -> None:
        values = [
            "hpre_raw_logit_gauss_risk",
            "raw_attention_risk",
            "raw_attention_risk+raw_attention_ev",
        ]
        config = {
            "feature_extraction": {
                "dgst_t": {
                    "branches": {
                        "hpre_raw_logit_gauss": True,
                        "raw_attention": False,
                    }
                }
            }
        }
        self.assertEqual(
            _enabled_method_feature_sets(config, values),
            ["hpre_raw_logit_gauss_risk"],
        )

        config["feature_extraction"]["dgst_t"].update(
            {
                "four_gate_methods": ["raw_attention"],
                "branches": {"raw_attention": True},
            }
        )
        self.assertEqual(
            _enabled_method_feature_sets(config, values),
            ["raw_attention_risk", "raw_attention_risk+raw_attention_ev"],
        )

    def test_active_yaml_enables_raw_attention_control(self) -> None:
        config = load_config(os.path.join(ROOT, "configs/model_configs_unified.yaml"))
        self.assertEqual(config["run"]["prompt"], "Describe this image.")
        self.assertEqual(config["run"]["extraction_mode"], "all")
        dgst = config["feature_extraction"]["dgst_t"]
        self.assertIn("raw_attention", dgst["four_gate_methods"])
        self.assertTrue(dgst["branches"]["raw_attention"])
        self.assertIn("hpre_softmax_prob_direct", dgst["four_gate_methods"])
        self.assertTrue(dgst["branches"]["hpre_softmax_prob_direct"])
        method_sets = config["training"]["feature_sets"]["method"]
        self.assertIn("raw_attention_risk", method_sets)
        self.assertIn("hpre_softmax_prob_direct_risk", method_sets)
        self.assertIn(
            "raw_attention_risk+raw_attention_target_cosine+raw_attention_ev",
            method_sets,
        )
        self.assertEqual(config["training"]["trainer"], "torch_mlp")
        self.assertEqual(config["training"]["positive_class"], "hallucination")


if __name__ == "__main__":
    unittest.main()
