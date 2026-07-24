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
            "raw_attention_risk+raw_attention_target_cosine+"
            "raw_attention_ev_target_dist_mass_x_cosine"
        )
        blocks = parse_feature_set(feature_set)
        self.assertEqual(
            blocks,
            [
                "raw_attention_risk",
                "raw_attention_target_cosine",
                "raw_attention_ev_target_dist_mass_x_cosine",
            ],
        )

        record = {
            "label": 1,
            "dgst_t_target_region_top_k": 16,
            "dgst_t_raw_attention_risk_sqrt_hpre_per_layer": [0.1, 0.2],
            "dgst_t_raw_attention_target_cosine_topk16_hpre_per_layer": [
                0.3,
                0.4,
            ],
            "dgst_t_raw_attention_ev_target_dist_mass_x_cosine_"
            "topk16_hpre_per_layer": [0.5, 0.6],
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
            "raw_attention_risk+raw_attention_ev_target_dist_mass_x_cosine",
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
            [
                "raw_attention_risk",
                "raw_attention_risk+raw_attention_ev_target_dist_mass_x_cosine",
            ],
        )

    def test_active_yaml_uses_authoritative_vpend_method_list(self) -> None:
        config = load_config(os.path.join(ROOT, "configs/model_configs_unified.yaml"))
        self.assertEqual(config["run"]["prompt"], "Describe this image.")
        self.assertIn(
            config["run"]["extraction_mode"],
            {"all", "method_only", "ads_cgc_only", "baseline_only"},
        )
        dgst = config["feature_extraction"]["dgst_t"]
        self.assertEqual(
            dgst["four_gate_methods"],
            ["hpre_raw_logit_gauss", "hpre_softmax_prob_gauss"],
        )
        self.assertNotIn("branches", dgst)
        self.assertNotIn("target_modes", dgst)
        self.assertEqual(dgst["support_modes"], ["vpend"])
        method_sets = config["training"]["feature_sets"]["method"]
        self.assertIn(
            "vpend_hpre_softmax_prob_gauss_risk+"
            "vpend_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine",
            method_sets,
        )
        self.assertEqual(config["training"]["trainer"], "torch_mlp")
        self.assertEqual(config["training"]["positive_class"], "real")


if __name__ == "__main__":
    unittest.main()
