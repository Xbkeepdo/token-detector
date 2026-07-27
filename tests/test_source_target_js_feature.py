from __future__ import annotations

import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from train_feature_sets import build_selected_matrix, parse_feature_set


class SourceTargetJsFeatureTests(unittest.TestCase):
    def test_gaussian_js_concatenates_with_matching_ev(self) -> None:
        feature_set = (
            "hpre_raw_logit_gauss_source_target_js+"
            "hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine"
        )
        blocks = parse_feature_set(feature_set)
        record = {
            "label": 0,
            "dgst_t_target_region_top_k": 16,
            "dgst_t_source_dist_per_layer": [[0.5, 0.5], [0.9, 0.1]],
            "dgst_t_attention_support_per_layer": [[0.5, 0.5], [0.5, 0.5]],
            "dgst_t_hpre_raw_logit_gauss_gate_per_layer": [
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
            "topk16_hpre_per_layer": [0.2, 0.4],
        }

        matrix, labels = build_selected_matrix([record], blocks)

        self.assertEqual(matrix.shape, (1, 4))
        self.assertEqual(labels.tolist(), [0])
        self.assertAlmostEqual(float(matrix[0, 0]), 0.0, places=7)
        self.assertGreater(float(matrix[0, 1]), 0.0)
        np.testing.assert_allclose(matrix[0, 2:], [0.2, 0.4])

    def test_direct_and_raw_attention_targets_are_supported(self) -> None:
        base = {
            "label": 1,
            "dgst_t_source_dist_per_layer": [[0.75, 0.25]],
            "dgst_t_attention_support_per_layer": [[0.25, 0.75]],
            "dgst_t_hpre_softmax_prob_direct_target_dist_per_layer": [[0.5, 0.5]],
        }
        for block in (
            "hpre_softmax_prob_direct_source_target_js",
            "raw_attention_source_target_js",
        ):
            matrix, labels = build_selected_matrix([dict(base)], parse_feature_set(block))
            self.assertEqual(matrix.shape, (1, 1))
            self.assertEqual(labels.tolist(), [1])
            self.assertGreater(float(matrix[0, 0]), 0.0)

    def test_one_minus_target_mass_times_cosine_uses_configured_top_k(self) -> None:
        feature_set = (
            "hpre_raw_logit_gauss_risk+"
            "hpre_raw_logit_gauss_one_minus_target_dist_mass_x_cosine"
        )
        record = {
            "label": 0,
            "dgst_t_target_region_top_k": 1,
            "dgst_t_attention_support_per_layer": [[0.2, 0.8], [0.6, 0.4]],
            "dgst_t_hpre_raw_logit_gauss_gate_per_layer": [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_per_layer": [0.1, 0.2],
            "dgst_t_hpre_raw_logit_gauss_target_cosine_topk1_hpre_per_layer": [
                0.5,
                0.25,
            ],
        }

        matrix, labels = build_selected_matrix(
            [record], parse_feature_set(feature_set)
        )

        self.assertEqual(labels.tolist(), [0])
        np.testing.assert_allclose(
            matrix,
            [[0.1, 0.2, (1.0 - 0.8) * 0.5, (1.0 - 0.6) * 0.25]],
            atol=1.0e-7,
        )

    def test_union_topk_js_uses_half_budget_from_each_distribution(self) -> None:
        record = {
            "label": 1,
            "dgst_t_transport_top_k": 2,
            "dgst_t_source_dist_per_layer": [[0.4, 0.3, 0.2, 0.1]],
            "dgst_t_attention_support_per_layer": [[0.1, 0.2, 0.3, 0.4]],
        }
        block = "raw_attention_source_target_union_topk_js"

        matrix, labels = build_selected_matrix(
            [record], parse_feature_set(block)
        )

        source = np.asarray([0.8, 0.2], dtype=np.float64)
        target = np.asarray([0.2, 0.8], dtype=np.float64)
        midpoint = 0.5 * (source + target)
        expected = 0.5 * np.sum(source * np.log(source / midpoint))
        expected += 0.5 * np.sum(target * np.log(target / midpoint))
        self.assertEqual(labels.tolist(), [1])
        np.testing.assert_allclose(matrix, [[expected]], atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
