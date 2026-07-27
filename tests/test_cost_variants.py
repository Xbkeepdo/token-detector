from __future__ import annotations

import math
import os
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from features.dgst_t import (
    COST_VARIANT_RISK_KEYS,
    GAUSSIAN_MAD_SCALE,
    _compute_gate_comparison_from_parts,
    _compute_cost_variant_risks,
    _cosine_distance_matrix,
    _relative_vll_evidence_signal,
    _sinkhorn_linear_cost_batch,
    _solve_exact_emd_problem_series,
    _solve_transport_problem,
    _topk_union_indices,
    _transport_risk_on_support,
)
from features.extractor import (
    _build_cost_variant_feature_record,
    _build_gate_comparison_feature_record,
)
from train_feature_sets import build_selected_matrix, parse_feature_set


class CostVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
        )

    def test_cosine_and_sqrt_costs(self) -> None:
        distance = _cosine_distance_matrix(self.states)
        expected = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
        )
        self.assertTrue(torch.allclose(distance, expected, atol=1e-6))
        sqrt_distance = torch.sqrt((distance / 2.0).clamp_min(0.0))
        self.assertTrue(torch.allclose(sqrt_distance, sqrt_distance.T))
        self.assertTrue(torch.allclose(torch.diag(sqrt_distance), torch.zeros(3)))
        self.assertTrue(torch.isfinite(sqrt_distance).all())

    def test_gaussian_mad_gate_and_zero_mad_are_finite(self) -> None:
        kwargs = dict(
            attention_signal=torch.tensor([0.2, 0.3, 0.5]),
            support_positions=[0, 1, 2],
            visual_start=0,
            visual_end=3,
            candidate_scope="visual",
            epsilon=1e-6,
            barrier_margin=0.5,
            barrier_max=3.0,
        )
        evidence, gate, _, stats = _relative_vll_evidence_signal(
            target_logits=torch.tensor([0.0, 1.0, 4.0]),
            stat_prefix="gauss",
            mad_scale=GAUSSIAN_MAD_SCALE,
            **kwargs,
        )
        expected_z = torch.tensor([-1.0, 0.0, 3.0]) / (
            GAUSSIAN_MAD_SCALE + 1e-6
        )
        self.assertTrue(torch.allclose(gate, torch.sigmoid(expected_z), atol=1e-6))
        self.assertTrue(torch.allclose(evidence, kwargs["attention_signal"] * gate))
        self.assertEqual(stats["gauss_mad_scale"], GAUSSIAN_MAD_SCALE)

        evidence, gate, _, _ = _relative_vll_evidence_signal(
            target_logits=torch.ones(3),
            stat_prefix="zero_mad",
            mad_scale=GAUSSIAN_MAD_SCALE,
            **kwargs,
        )
        self.assertTrue(torch.isfinite(evidence).all())
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(torch.allclose(gate, torch.full((3,), 0.5)))

    def test_softmax_relative_vll_applies_mad_to_target_probabilities(self) -> None:
        # These are p(w* | h_i) values obtained after a full-vocabulary
        # softmax at each visual position, not a softmax over positions.
        probabilities = torch.tensor([0.01, 0.04, 0.25])
        evidence, gate, _, stats = _relative_vll_evidence_signal(
            attention_signal=torch.ones(3),
            target_logits=probabilities,
            support_positions=[0, 1, 2],
            visual_start=0,
            visual_end=3,
            candidate_scope="visual",
            stat_prefix="softmax_relative_vll",
            epsilon=1e-6,
            barrier_margin=0.5,
            barrier_max=3.0,
            mad_scale=GAUSSIAN_MAD_SCALE,
        )
        median = probabilities.median()
        mad = torch.abs(probabilities - median).median()
        expected = torch.sigmoid(
            (probabilities - median) / (GAUSSIAN_MAD_SCALE * mad + 1e-6)
        )
        self.assertTrue(torch.allclose(gate, expected, atol=1e-6))
        self.assertTrue(torch.allclose(evidence, expected, atol=1e-6))
        self.assertEqual(stats["softmax_relative_vll_mad_scale"], GAUSSIAN_MAD_SCALE)

    def test_gate_comparison_emits_matched_hpre_features(self) -> None:
        result = _compute_gate_comparison_from_parts(
            source_ffn_states=[torch.tensor([0.4, 0.2])],
            source_attn_states=[torch.tensor([0.1, -0.1])],
            prediction_hidden_states=[torch.tensor([0.9, 0.5])],
            support_h_prev_states=[self.states],
            support_h_mid_states=[self.states + 0.1],
            support_attentions=[torch.tensor([0.2, 0.3, 0.5])],
            semantic_probs=[torch.tensor([0.02, 0.10, 0.04])],
            relative_vll_logits=[torch.tensor([0.0, 1.0, 4.0])],
            support_positions=[0, 1, 2],
            visual_start=0,
            visual_end=3,
            tau=0.07,
            source_distribution_mode="softmax",
            transport_top_k=3,
            ot_solver="linprog",
            atarget_visual_top_k=2,
            relative_vll_mad_epsilon=1e-6,
            relative_barrier_margin=0.5,
            relative_barrier_max=3.0,
        )
        series_keys = [
            key
            for key in result
            if key.endswith("_per_layer")
        ]
        self.assertEqual(len(series_keys), 6)
        for key in series_keys:
            self.assertEqual(tuple(result[key].shape), (1,))
            self.assertTrue(torch.isfinite(result[key]).all(), key)
        self.assertEqual(result["dgst_t_gate_comparison_cost"], "sqrt_cosine_hpre")

    def test_all_risks_are_finite_and_raw_attention_is_direct(self) -> None:
        source = torch.tensor([0.7, 0.2, 0.1])
        relative = torch.tensor([0.1, 0.3, 0.6])
        gauss = torch.tensor([0.2, 0.4, 0.4])
        raw_attention = torch.tensor([2.0, 3.0, 5.0])
        risks = _compute_cost_variant_risks(
            source_dist=source,
            relative_target=relative,
            gauss_target=gauss,
            raw_attention_target=raw_attention,
            hmid_states=self.states,
            hpre_states=self.states.roll(1, dims=0),
            transport_top_k=3,
            ot_solver="linprog",
        )
        self.assertEqual(set(risks), set(COST_VARIANT_RISK_KEYS))
        self.assertTrue(all(math.isfinite(value) for value in risks.values()))
        support = _topk_union_indices(source, relative, 3)
        legacy_geo = _transport_risk_on_support(
            source_dist=source,
            target_dist=relative,
            support_states=self.states,
            semantic_probs=torch.ones(3),
            support=support,
            cost_mode="geo",
            lambda_d=1.0,
            lambda_s=1.0,
            lambda_t=1.0,
            lambda_int=1.0,
            ot_solver="linprog",
        )
        self.assertAlmostEqual(risks["risk_geo"], legacy_geo, places=6)

        scaled = _compute_cost_variant_risks(
            source_dist=source,
            relative_target=relative,
            gauss_target=gauss,
            raw_attention_target=raw_attention * 17.0,
            hmid_states=self.states,
            hpre_states=self.states.roll(1, dims=0),
            transport_top_k=3,
            ot_solver="linprog",
        )
        self.assertAlmostEqual(
            risks["risk_raw_attention_hmid"],
            scaled["risk_raw_attention_hmid"],
            places=6,
        )

    def test_parallel_emd_matches_serial_exactly(self) -> None:
        kwargs = dict(
            source_dist=torch.tensor([0.7, 0.2, 0.1]),
            relative_target=torch.tensor([0.1, 0.3, 0.6]),
            gauss_target=torch.tensor([0.2, 0.4, 0.4]),
            raw_attention_target=torch.tensor([2.0, 3.0, 5.0]),
            hmid_states=self.states,
            hpre_states=self.states.roll(1, dims=0),
            transport_top_k=3,
            ot_solver="emd",
        )
        with patch.dict(os.environ, {"DGST_COST_VARIANT_EMD_WORKERS": "1"}):
            serial = _compute_cost_variant_risks(**kwargs)
        with patch.dict(os.environ, {"DGST_COST_VARIANT_EMD_WORKERS": "10"}):
            parallel = _compute_cost_variant_risks(**kwargs)
        self.assertEqual(list(serial), list(parallel))
        for key in serial:
            self.assertAlmostEqual(serial[key], parallel[key], places=12)

    def test_exact_emd_parallelism_flattens_individual_problems(self) -> None:
        class ImmediateFuture:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class RecordingExecutor:
            def __init__(self):
                self.calls = []

            def submit(self, function, *args):
                self.calls.append((function, args))
                return ImmediateFuture(function(*args))

        executor = RecordingExecutor()
        problems = {
            "risk_a": [3.0, None, 1.0],
            "risk_b": [2.0, 4.0],
        }
        with patch.dict(os.environ, {"DGST_COST_VARIANT_EMD_WORKERS": "16"}), patch(
            "features.dgst_t._get_emd_thread_pool",
            return_value=executor,
        ) as get_pool, patch(
            "features.dgst_t._solve_transport_problem",
            side_effect=lambda problem, solver: float(problem),
        ):
            result = _solve_exact_emd_problem_series(problems)

        get_pool.assert_called_once_with(16)
        self.assertEqual(len(executor.calls), 4)
        self.assertEqual(result, {
            "risk_a": [3.0, 0.0, 1.0],
            "risk_b": [2.0, 4.0],
        })

    def test_numpy_emd_fast_path_matches_tensor_solver(self) -> None:
        source = np.asarray([0.6, 0.3, 0.1], dtype=np.float32)
        target = np.asarray([0.1, 0.2, 0.7], dtype=np.float32)
        cost = np.asarray(
            [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]],
            dtype=np.float32,
        )
        numpy_risk = _solve_transport_problem((source, target, cost), "emd")
        tensor_risk = _solve_transport_problem(
            (
                torch.from_numpy(source),
                torch.from_numpy(target),
                torch.from_numpy(cost),
            ),
            "emd",
        )
        self.assertAlmostEqual(numpy_risk, tensor_risk, places=12)

    def test_sinkhorn_returns_valid_but_regularized_transport_cost(self) -> None:
        source = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[0.5, 0.5]])
        cost = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
        risk, marginal_error = _sinkhorn_linear_cost_batch(
            source,
            target,
            cost,
            reg=0.5,
            max_iter=500,
            tol=1e-7,
        )
        self.assertLessEqual(marginal_error, 1e-6)
        self.assertGreater(float(risk[0]), 0.0)
        self.assertLess(float(risk[0]), 0.5)

    def test_training_aliases_cover_21_feature_sets(self) -> None:
        risk_names = [name.replace("_", "-") for name in COST_VARIANT_RISK_KEYS]
        risk_names[4] = "risk-rawAttention-hmid"
        risk_names[5] = "risk-rawAttention-hpre"
        feature_sets = ["hprecosine", *risk_names]
        feature_sets.extend(f"{name}+hprecosine" for name in risk_names)
        self.assertEqual(len(feature_sets), 21)
        for feature_set in feature_sets:
            self.assertTrue(parse_feature_set(feature_set))

    def test_risk_hprecosine_product_is_layerwise(self) -> None:
        row = {
            "label": 1,
            "dgst_t_risk_geo_per_layer": [0.2, 0.5, 0.8],
            "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer": [0.3, 0.4, 0.9],
        }
        blocks = parse_feature_set("risk-geo*hprecosine")
        matrix, labels = build_selected_matrix([row], blocks)
        self.assertEqual(matrix.shape, (1, 3))
        self.assertTrue(
            np.allclose(matrix[0], np.asarray([0.06, 0.20, 0.72], dtype=np.float32))
        )
        self.assertTrue(np.array_equal(labels, np.asarray([1], dtype=np.int32)))
        self.assertEqual(
            parse_feature_set("ffn_fad*risk_geo_raw"),
            ["ffn_fad_x_risk_geo_raw"],
        )

    def test_compact_feature_record_has_exact_profile(self) -> None:
        layers, support = 2, 3
        dgst_t = {
            f"dgst_t_{name}_per_layer": torch.ones(layers)
            for name in COST_VARIANT_RISK_KEYS
        }
        dgst_t.update(
            {
                "dgst_t_target_visual_hpre_cosine_relative_vll_per_layer": torch.ones(layers),
                "dgst_t_vv_support_attention_per_layer": torch.ones(layers, support),
                "dgst_t_vv_source_dist_per_layer": torch.ones(layers, support),
                "dgst_t_vv_semantic_gate_per_layer": torch.ones(layers, support),
                "dgst_t_vv_gauss_semantic_gate_per_layer": torch.ones(layers, support),
                "dgst_t_vv_support_positions": [10, 11, 12],
                "dgst_t_cost_variant_mad_scale": GAUSSIAN_MAD_SCALE,
                "dgst_t_cost_variant_transport_top_k": 64,
                "dgst_t_cost_variant_hprecosine_top_k": 32,
                "dgst_t_relative_vll_logit_source": "h_mid",
                "dgst_t_source_distribution_mode": "softmax",
            }
        )
        record = _build_cost_variant_feature_record(
            image_id=1,
            span={"word": "car", "label": 1},
            response_index=2,
            target_token_id=3,
            model_out=SimpleNamespace(token_id=3),
            dgst_t=dgst_t,
        )
        self.assertEqual(len(record), 27)
        self.assertNotIn("dgst_t_layer_stats", record)
        self.assertEqual(tuple(record["dgst_t_vv_source_dist_per_layer"].shape), (2, 3))

    def test_gate_comparison_compact_record(self) -> None:
        tensor_keys = (
            "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer",
            "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer",
            "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer",
            "dgst_t_relative_vll_gauss_target_visual_hpre_cosine_per_layer",
            "dgst_t_softmax_relative_vll_gauss_target_visual_hpre_cosine_per_layer",
            "dgst_t_legacy_prob_target_visual_hpre_cosine_per_layer",
        )
        dgst_t = {key: torch.ones(2) for key in tensor_keys}
        dgst_t.update(
            {
                "dgst_t_gate_comparison_methods": [
                    "relative_vll",
                    "softmax_relative_vll",
                    "legacy_prob",
                ],
                "dgst_t_gate_comparison_cost": "sqrt_cosine_hpre",
                "dgst_t_gate_comparison_target_cosine_state": "hpre",
                "dgst_t_gate_comparison_softmax_axis": "vocabulary",
                "dgst_t_gate_comparison_mad_axis": "visual_tokens",
                "dgst_t_gate_comparison_mad_scale": GAUSSIAN_MAD_SCALE,
                "dgst_t_gate_comparison_transport_top_k": 64,
                "dgst_t_gate_comparison_target_cosine_top_k": 32,
                "dgst_t_relative_vll_logit_source": "h_mid",
                "dgst_t_source_distribution_mode": "softmax",
            }
        )
        record = _build_gate_comparison_feature_record(
            image_id=1,
            span={"word": "car", "label": 0},
            response_index=2,
            target_token_id=3,
            model_out=SimpleNamespace(token_id=3),
            dgst_t=dgst_t,
        )
        self.assertEqual(record["label"], 0)
        self.assertEqual(record["dgst_t_gate_comparison_cost"], "sqrt_cosine_hpre")
        self.assertEqual(record["dgst_t_gate_comparison_softmax_axis"], "vocabulary")
        self.assertEqual(record["dgst_t_gate_comparison_mad_axis"], "visual_tokens")
        for key in tensor_keys:
            self.assertEqual(record[key], [1.0, 1.0])

    def test_gate_comparison_probe_aliases_build_matched_blocks(self) -> None:
        row = {
            "label": 0,
            "dgst_t_relative_vll_gauss_risk_sqrt_hpre_per_layer": [0.1, 0.2],
            "dgst_t_relative_vll_gauss_target_visual_hpre_cosine_per_layer": [
                0.3,
                0.4,
            ],
            "dgst_t_softmax_relative_vll_gauss_risk_sqrt_hpre_per_layer": [
                0.5,
                0.6,
            ],
            "dgst_t_softmax_relative_vll_gauss_target_visual_hpre_cosine_per_layer": [
                0.7,
                0.8,
            ],
            "dgst_t_legacy_prob_risk_sqrt_hpre_per_layer": [0.9, 1.0],
            "dgst_t_legacy_prob_target_visual_hpre_cosine_per_layer": [1.1, 1.2],
        }
        expected = {
            "gate-relative-vll-risk": [0.1, 0.2],
            "gate-relative-vll-hprecosine": [0.3, 0.4],
            "gate-softmax-relative-vll-risk": [0.5, 0.6],
            "gate-softmax-relative-vll-hprecosine": [0.7, 0.8],
            "gate-legacy-prob-risk": [0.9, 1.0],
            "gate-legacy-prob-hprecosine": [1.1, 1.2],
        }
        for feature_set, values in expected.items():
            matrix, labels = build_selected_matrix([row], parse_feature_set(feature_set))
            self.assertTrue(np.allclose(matrix, np.asarray([values], dtype=np.float32)))
            self.assertTrue(np.array_equal(labels, np.asarray([0], dtype=np.int32)))

        matrix, _labels = build_selected_matrix(
            [row],
            parse_feature_set(
                "gate-softmax-relative-vll-risk+"
                "gate-softmax-relative-vll-hprecosine"
            ),
        )
        self.assertEqual(matrix.shape, (1, 4))
        self.assertTrue(
            np.allclose(matrix[0], np.asarray([0.5, 0.6, 0.7, 0.8], dtype=np.float32))
        )


if __name__ == "__main__":
    unittest.main()
