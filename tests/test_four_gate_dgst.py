from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from features.dgst_t import (
    DIRECT_HPRE_SOFTMAX_METHOD,
    FOUR_GATE_CAPTURE_FIELDS,
    FOUR_GATE_METHODS,
    RAW_ATTENTION_METHOD,
    build_compact_four_gate_layer_capture,
    compute_four_gate_dgst_batch_from_captures,
    _gaussian_mad_gate,
    _prepare_transport_problem_for_state_cost,
    _solve_transport_problem,
    _stable_topk_indices,
    _topk_union_indices,
)
from features.extractor import _build_four_gate_feature_record
from models.dgst_capture import (
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    target_logits_and_probabilities_multi,
)
from models.base_wrapper import configure_image_processor_limits
from train_feature_sets import build_selected_matrix, parse_feature_set


class FourGateDGSTTests(unittest.TestCase):
    def test_dynamic_image_pixel_limit_updates_fast_and_slow_fields(self):
        image_processor = SimpleNamespace(
            max_pixels=3_240_000,
            min_pixels=3_136,
            size={"shortest_edge": 3_136, "longest_edge": 3_240_000},
        )
        processor = SimpleNamespace(image_processor=image_processor)
        configure_image_processor_limits(
            processor,
            {"max_pixels": 112_896, "min_pixels": 3_136},
        )
        self.assertEqual(image_processor.max_pixels, 112_896)
        self.assertEqual(image_processor.min_pixels, 3_136)
        self.assertEqual(image_processor.size["longest_edge"], 112_896)
        self.assertEqual(image_processor.size["shortest_edge"], 3_136)

    def _output_layer(self) -> torch.nn.Linear:
        layer = torch.nn.Linear(2, 4, bias=True)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 0.5],
                    ]
                )
            )
            layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        return layer

    def test_joint_projection_matches_full_vocabulary_softmax(self) -> None:
        layer = self._output_layer()
        states = torch.tensor(
            [[1.0, 2.0], [-1.0, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        raw, probabilities = target_logits_and_probabilities_multi(
            output_layer=layer,
            states=states,
            target_token_ids=[1, 3],
            chunk_size=1,
        )
        vocab_logits = torch.nn.functional.linear(states, layer.weight, layer.bias)
        expected_raw = torch.nn.functional.linear(
            states,
            layer.weight.index_select(0, torch.tensor([1, 3])),
            bias=None,
        )
        expected_probabilities = torch.softmax(vocab_logits, dim=-1).index_select(
            1, torch.tensor([1, 3])
        )
        self.assertTrue(torch.allclose(raw, expected_raw, atol=1e-6))
        self.assertTrue(
            torch.allclose(probabilities, expected_probabilities, atol=1e-6)
        )
        # Feature extraction is inference-only.  In particular the vocabulary
        # softmax must not retain an LM-head autograd graph across layers.
        for value in (raw, probabilities):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)

    def test_projection_work_is_conditioned_on_enabled_methods(self) -> None:
        layer = self._output_layer()
        patches, chunk_size = 5, 2
        h_prev = torch.randn(1, patches + 1, 2)
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + 0.01,
            "o_attn": torch.full_like(h_prev, 0.01),
            "o_ffn": torch.randn_like(h_prev) * 0.01,
            "attn_weights": torch.ones(1, 1, patches + 1, patches + 1),
        }
        expected_chunks = (patches + chunk_size - 1) // chunk_size

        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            raw_attention = build_compact_four_gate_layer_capture(
                output_layer=layer,
                capture=capture,
                visual_start=0,
                visual_end=patches,
                target_token_ids=[0],
                prediction_positions=[patches],
                semantic_chunk_size=chunk_size,
                enabled_methods=[RAW_ATTENTION_METHOD],
            )
        self.assertEqual(linear.call_count, 0)
        for key in FOUR_GATE_CAPTURE_FIELDS[4:]:
            self.assertIsNone(raw_attention[key])

        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            hpre_raw = build_compact_four_gate_layer_capture(
                output_layer=layer,
                capture=capture,
                visual_start=0,
                visual_end=patches,
                target_token_ids=[0],
                prediction_positions=[patches],
                semantic_chunk_size=chunk_size,
                enabled_methods=["hpre_raw_logit_gauss"],
            )
        self.assertEqual(linear.call_count, expected_chunks)
        self.assertIsNotNone(hpre_raw["hpre_raw_target_logits"])
        self.assertIsNone(hpre_raw["hpre_softmax_target_probs"])
        self.assertIsNone(hpre_raw["hmid_raw_target_logits"])
        self.assertIsNone(hpre_raw["hmid_softmax_target_probs"])

        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            direct_softmax = build_compact_four_gate_layer_capture(
                output_layer=layer,
                capture=capture,
                visual_start=0,
                visual_end=patches,
                target_token_ids=[0],
                prediction_positions=[patches],
                semantic_chunk_size=chunk_size,
                enabled_methods=[DIRECT_HPRE_SOFTMAX_METHOD],
            )
        self.assertEqual(linear.call_count, expected_chunks)
        self.assertIsNone(direct_softmax["hpre_raw_target_logits"])
        self.assertIsNotNone(direct_softmax["hpre_softmax_target_probs"])
        self.assertIsNone(direct_softmax["hmid_raw_target_logits"])
        self.assertIsNone(direct_softmax["hmid_softmax_target_probs"])

    def test_compact_capture_projects_each_state_chunk_once(self) -> None:
        layer = self._output_layer()
        patches, chunk_size = 5, 2
        h_prev = torch.randn(1, patches + 1, 2)
        o_attn = torch.randn_like(h_prev) * 0.1
        attention = torch.ones(1, 1, patches + 1, patches + 1)
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + o_attn,
            "o_attn": o_attn,
            "o_ffn": torch.randn_like(h_prev) * 0.1,
            "attn_weights": attention,
        }
        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            compact = build_compact_four_gate_layer_capture(
                output_layer=layer,
                capture=capture,
                visual_start=0,
                visual_end=patches,
                target_token_ids=[0, 1],
                prediction_positions=[patches, patches],
                semantic_chunk_size=chunk_size,
            )
        # ceil(P/chunk) calls for hpre plus the same number for hmid. Raw and
        # softmax columns share those calls rather than projecting twice.
        self.assertEqual(linear.call_count, 2 * ((patches + chunk_size - 1) // chunk_size))
        self.assertEqual(tuple(compact), FOUR_GATE_CAPTURE_FIELDS)
        self.assertFalse(
            any(
                value is not None and value.shape[-1] == layer.out_features
                for value in compact.values()
            )
        )

    def test_manual_gaussian_mad_and_known_exact_emd(self) -> None:
        values = torch.tensor([-3.0, -1.0, 2.0, 8.0])
        gate = _gaussian_mad_gate(values, epsilon=1e-6)
        median = values.median()
        mad = torch.abs(values - median).median()
        expected = torch.sigmoid((values - median) / (1.4826 * mad + 1e-6))
        self.assertTrue(torch.allclose(gate, expected, atol=1e-6))

        source = torch.tensor([1.0, 0.0])
        target = torch.tensor([0.0, 1.0])
        states = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        problem = _prepare_transport_problem_for_state_cost(
            source_dist=source,
            target_dist=target,
            states=states,
            support=torch.tensor([0, 1]),
            sqrt_cosine=True,
        )
        self.assertAlmostEqual(
            _solve_transport_problem(problem, "emd"),
            2.0 ** -0.5,
            places=6,
        )

    def test_p_over_32_uses_each_branch_own_target_region(self) -> None:
        patches = 40
        layer = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 3.0], [-1.0, 0.0], [0.0, -2.0]])
            )
        x = torch.linspace(-2.0, 2.0, patches)
        y = torch.sin(torch.linspace(0.0, 4.0 * torch.pi, patches)) * 2.0
        visual = torch.stack((x, y), dim=1)
        h_prev = torch.cat((visual, torch.tensor([[1.0, 0.0]])), dim=0).unsqueeze(0)
        o_attn = torch.zeros_like(h_prev)
        o_attn[0, :patches, 0] = torch.flip(x, dims=(0,)) - x
        o_attn[0, :patches, 1] = torch.linspace(-1.5, 1.5, patches)
        o_ffn = torch.zeros_like(h_prev)
        o_ffn[0, patches] = torch.tensor([0.3, -0.2])
        attention = torch.zeros(1, 2, patches + 1, patches + 1)
        attention[0, :, patches, :patches] = 1.0 / patches
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + o_attn,
            "o_attn": o_attn,
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        compact = build_compact_four_gate_layer_capture(
            output_layer=layer,
            capture=capture,
            visual_start=0,
            visual_end=patches,
            target_token_ids=[0],
            prediction_positions=[patches],
        )
        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=patches,
            target_token_ids=[0],
            prediction_positions=[patches],
            target_region_top_k=32,
        )[0]
        cosine_map = F.cosine_similarity(
            compact["prediction_hpre"][0].unsqueeze(0),
            compact["visual_hpre"],
            dim=-1,
        )
        hmid_cosine_map = F.cosine_similarity(
            capture["h_mid"][0, patches].unsqueeze(0),
            capture["h_mid"][0, :patches],
            dim=-1,
        )
        input_keys = {
            "hpre_raw_logit_gauss": "hpre_raw_target_logits",
            "hpre_softmax_prob_gauss": "hpre_softmax_target_probs",
            "hmid_raw_logit_gauss": "hmid_raw_target_logits",
            "hmid_softmax_prob_gauss": "hmid_softmax_target_probs",
        }
        regions = []
        for method, input_key in input_keys.items():
            gate = _gaussian_mad_gate(compact[input_key][0], epsilon=1e-6)
            target_dist = compact["attention_support"][0] * gate
            target_dist = target_dist / target_dist.sum()
            region = _stable_topk_indices(target_dist, 32)
            regions.append(tuple(region.tolist()))
            state_name = "hmid" if method.startswith("hmid_") else "hpre"
            branch_cosine = (
                hmid_cosine_map if state_name == "hmid" else cosine_map
            )
            expected_cosine = branch_cosine.index_select(0, region).mean()
            key = (
                f"dgst_t_{method}_target_cosine_topk32_"
                f"{state_name}_per_layer"
            )
            self.assertAlmostEqual(float(result[key][0]), float(expected_cosine), places=6)
            cost_states = (
                capture["h_mid"][0, :patches]
                if state_name == "hmid"
                else capture["h_prev"][0, :patches]
            )
            support = _topk_union_indices(
                compact["source_dist"][0],
                target_dist,
                64,
            )
            expected_problem = _prepare_transport_problem_for_state_cost(
                source_dist=compact["source_dist"][0],
                target_dist=target_dist,
                states=cost_states,
                support=support,
                sqrt_cosine=True,
            )
            expected_risk = _solve_transport_problem(expected_problem, "emd")
            risk_key = (
                f"dgst_t_{method}_risk_sqrt_{state_name}_per_layer"
            )
            self.assertAlmostEqual(
                float(result[risk_key][0]),
                expected_risk,
                places=6,
            )
        self.assertGreater(len(set(regions)), 1)

    def test_four_methods_emit_distinct_named_matrices_and_curves(self) -> None:
        layer = self._output_layer()
        model = SimpleNamespace(get_output_embeddings=lambda: layer)
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        o_attn = torch.tensor(
            [[[0.0, 0.2], [0.1, 0.0], [0.2, -0.1], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        o_ffn = torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.3, -0.1]]],
            dtype=torch.float32,
        )
        attention = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        attention[0, 0, 3, :3] = torch.tensor([0.6, 0.3, 0.1])
        attention[0, 1, 3, :3] = torch.tensor([0.2, 0.3, 0.5])
        captures = [
            {
                "h_prev": h_prev,
                "h_mid": h_prev + o_attn,
                "o_attn": o_attn,
                "o_ffn": o_ffn,
                "attn_weights": attention,
            }
        ]
        compact = build_compact_four_gate_layer_capture(
            output_layer=layer,
            capture=captures[0],
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            semantic_chunk_size=2,
        )
        self.assertEqual(tuple(compact), FOUR_GATE_CAPTURE_FIELDS)
        self.assertEqual(tuple(compact["prediction_hpre"].shape), (1, 2))
        self.assertEqual(tuple(compact["visual_hpre"].shape), (3, 2))
        for key in FOUR_GATE_CAPTURE_FIELDS[2:]:
            if compact[key] is not None:
                self.assertEqual(tuple(compact[key].shape), (1, 3))
        self.assertFalse(any("vocab" in key for key in compact))

        results = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=captures,
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            semantic_chunk_size=2,
            tau=0.07,
            transport_top_k=3,
            target_region_top_k=32,
        )
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(any("logits" in key or "probs" in key for key in result))
        self.assertEqual(result["dgst_t_four_gate_methods"], list(FOUR_GATE_METHODS))
        self.assertEqual(
            result["dgst_t_attention_support_per_layer"].dtype, torch.float32
        )
        self.assertEqual(result["dgst_t_source_dist_per_layer"].dtype, torch.float32)
        self.assertEqual(
            tuple(result["dgst_t_attention_support_per_layer"].shape), (1, 3)
        )

        normalized_attention = attention[0, :, 3, :3].mean(dim=0)
        normalized_attention = normalized_attention / normalized_attention.sum()

        for method in FOUR_GATE_METHODS:
            state_name = "hmid" if method.startswith("hmid_") else "hpre"
            branch_states = (
                captures[0]["h_mid"][0]
                if state_name == "hmid"
                else captures[0]["h_prev"][0]
            )
            cosine = torch.nn.functional.cosine_similarity(
                branch_states[3].unsqueeze(0),
                branch_states[:3],
                dim=-1,
            )
            expected_ev = float(
                (
                    normalized_attention
                    * ((1.0 + cosine) / 2.0)
                ).sum().item()
            )
            expected_cosine = float(cosine.mean().item())
            gate_key = f"dgst_t_{method}_gate_per_layer"
            risk_key = (
                f"dgst_t_{method}_risk_sqrt_{state_name}_per_layer"
            )
            cosine_key = (
                f"dgst_t_{method}_target_cosine_topk32_"
                f"{state_name}_per_layer"
            )
            ev_key = (
                f"dgst_t_{method}_ev_topk32_{state_name}_per_layer"
            )
            self.assertEqual(tuple(result[gate_key].shape), (1, 3))
            self.assertEqual(result[gate_key].dtype, torch.float32)
            self.assertEqual(tuple(result[risk_key].shape), (1,))
            self.assertEqual(result[risk_key].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result[risk_key]).all())
            # K=32 covers all three visual tokens in this toy example, so all
            # methods share the hand-computable region aggregate.
            self.assertAlmostEqual(float(result[cosine_key][0]), expected_cosine, places=6)
            self.assertAlmostEqual(float(result[ev_key][0]), expected_ev, places=6)

        record = _build_four_gate_feature_record(
            image_id=7,
            span={"word": "car", "label": 0},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        self.assertEqual(record["feature_schema_version"], "dgst-four-gate-v1")
        self.assertEqual(str(record["dgst_t_attention_support_per_layer"].dtype), "float32")
        self.assertEqual(
            str(record["dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_per_layer"].dtype),
            "float32",
        )
        matrix, labels = build_selected_matrix(
            [record],
            parse_feature_set(
                "hpre_raw_logit_gauss_risk+"
                "hpre_raw_logit_gauss_target_cosine+"
                "hpre_raw_logit_gauss_ev"
            ),
        )
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(labels.tolist(), [0])

    def test_raw_attention_is_post_softmax_visual_support_without_gate(self) -> None:
        layer = self._output_layer()
        model = SimpleNamespace(get_output_embeddings=lambda: layer)
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        o_ffn = torch.zeros_like(h_prev)
        o_ffn[0, 3] = torch.tensor([0.2, -0.1])
        attention = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        # These are model-returned attention weights (already post-softmax).
        # Averaging heads yields [0.4, 0.3, 0.3], already normalized here.
        attention[0, 0, 3, :3] = torch.tensor([0.7, 0.2, 0.1])
        attention[0, 1, 3, :3] = torch.tensor([0.1, 0.4, 0.5])
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev,
            "o_attn": torch.zeros_like(h_prev),
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        with mock.patch("torch.nn.functional.linear", wraps=F.linear) as linear:
            result = compute_four_gate_dgst_batch_from_captures(
                model=model,
                captures=[capture],
                visual_start=0,
                visual_end=3,
                target_token_ids=[1],
                prediction_positions=[3],
                transport_top_k=3,
                target_region_top_k=32,
                enabled_methods=[RAW_ATTENTION_METHOD],
            )[0]
        self.assertEqual(linear.call_count, 0)
        self.assertEqual(result["dgst_t_four_gate_methods"], [RAW_ATTENTION_METHOD])
        self.assertEqual(result["dgst_t_profile"], "target_comparison_v2")
        self.assertEqual(
            result["dgst_t_raw_attention_definition"],
            "post_softmax_head_mean_visual_support_renormalized",
        )
        expected_attention = torch.tensor([0.4, 0.3, 0.3])
        self.assertTrue(
            torch.allclose(
                result["dgst_t_attention_support_per_layer"][0].float(),
                expected_attention,
                atol=5e-4,
            )
        )
        self.assertTrue(
            torch.equal(
                result["dgst_t_raw_attention_gate_per_layer"],
                torch.ones(1, 3, dtype=torch.float32),
            )
        )
        cosine = F.cosine_similarity(h_prev[0, 3].unsqueeze(0), h_prev[0, :3], dim=-1)
        expected_cosine = float(cosine.mean())
        expected_ev = float((expected_attention * ((1.0 + cosine) / 2.0)).sum())
        self.assertAlmostEqual(
            float(result["dgst_t_raw_attention_target_cosine_topk32_hpre_per_layer"][0]),
            expected_cosine,
            places=6,
        )
        self.assertAlmostEqual(
            float(result["dgst_t_raw_attention_ev_topk32_hpre_per_layer"][0]),
            expected_ev,
            places=6,
        )

        record = _build_four_gate_feature_record(
            image_id=8,
            span={"word": "bus", "label": 1},
            response_index=3,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        self.assertEqual(
            record["feature_schema_version"],
            "dgst-target-comparison-v2",
        )
        self.assertEqual(
            record["dgst_t_raw_attention_definition"],
            "post_softmax_head_mean_visual_support_renormalized",
        )

    def test_direct_hpre_softmax_uses_target_probability_without_gate(self) -> None:
        layer = self._output_layer()
        model = SimpleNamespace(get_output_embeddings=lambda: layer)
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        o_ffn = torch.zeros_like(h_prev)
        o_ffn[0, 3] = torch.tensor([0.2, -0.1])
        attention = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        attention[0, 0, 3, :3] = torch.tensor([0.7, 0.2, 0.1])
        attention[0, 1, 3, :3] = torch.tensor([0.1, 0.4, 0.5])
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev,
            "o_attn": torch.zeros_like(h_prev),
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        result = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=[capture],
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            transport_top_k=3,
            target_region_top_k=32,
            enabled_methods=[DIRECT_HPRE_SOFTMAX_METHOD],
        )[0]

        vocab_logits = F.linear(h_prev[0, :3], layer.weight, layer.bias)
        expected_probs = torch.softmax(vocab_logits, dim=-1)[:, 1]
        expected_target_dist = expected_probs / expected_probs.sum()

        self.assertEqual(result["dgst_t_profile"], "target_comparison_v3")
        self.assertEqual(
            result["dgst_t_four_gate_methods"],
            [DIRECT_HPRE_SOFTMAX_METHOD],
        )
        self.assertNotIn(
            f"dgst_t_{DIRECT_HPRE_SOFTMAX_METHOD}_gate_per_layer",
            result,
        )
        self.assertTrue(
            torch.allclose(
                result[
                    "dgst_t_hpre_softmax_prob_direct_"
                    "target_prob_matrix_per_layer"
                ][0].float(),
                expected_probs,
                atol=5e-4,
            )
        )
        self.assertEqual(
            result[
                "dgst_t_hpre_softmax_prob_direct_target_prob_matrix_per_layer"
            ].dtype,
            torch.float32,
        )
        self.assertTrue(
            torch.allclose(
                result[
                    "dgst_t_hpre_softmax_prob_direct_target_dist_per_layer"
                ][0].float(),
                expected_target_dist,
                atol=5e-4,
            )
        )
        risk_key = (
            "dgst_t_hpre_softmax_prob_direct_risk_sqrt_hpre_per_layer"
        )
        self.assertTrue(torch.isfinite(result[risk_key]).all())

        record = _build_four_gate_feature_record(
            image_id=9,
            span={"word": "phone", "label": 0},
            response_index=3,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        self.assertEqual(
            record["feature_schema_version"],
            "dgst-target-comparison-v3",
        )
        self.assertNotIn(
            f"dgst_t_{DIRECT_HPRE_SOFTMAX_METHOD}_gate_per_layer",
            record,
        )
        matrix, labels = build_selected_matrix(
            [record],
            parse_feature_set(
                "hpre_softmax_prob_direct_risk+"
                "hpre_softmax_prob_direct_target_cosine+"
                "hpre_softmax_prob_direct_ev"
            ),
        )
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(labels.tolist(), [0])

    def test_method_only_can_release_full_hook_captures_layerwise(self) -> None:
        layer = self._output_layer()
        h_prev = torch.randn(1, 5, 2)
        o_attn = torch.randn_like(h_prev) * 0.01
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + o_attn,
            "o_attn": o_attn,
            "o_ffn": torch.randn_like(h_prev) * 0.01,
            "attn_weights": torch.ones(1, 1, 5, 5),
        }
        compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=4,
            target_token_ids=[1],
            prediction_positions=[4],
            release_layer_captures=True,
        )
        for key in ("h_prev", "h_mid", "o_attn", "o_ffn", "attn_weights"):
            self.assertIsNone(capture[key])

    def test_hook_and_model_response_hidden_use_same_final_norm(self) -> None:
        norm = torch.nn.LayerNorm(3)
        model = SimpleNamespace(model=SimpleNamespace(norm=norm))
        raw = torch.tensor(
            [[[1.0, 2.0, 4.0], [2.0, -1.0, 0.5], [0.1, 0.2, 0.3]]]
        )
        capture = {
            "h_mid": raw * 0.75,
            "o_ffn": raw * 0.25,
        }
        expected = norm(raw)[0, 1:3]
        from_capture = final_normalized_hidden_slice(
            model=model,
            out=SimpleNamespace(hidden_states=None),
            dgst_captures=[capture],
            start=1,
            end=3,
        )
        from_layer_output = final_normalized_hidden_slice(
            model=model,
            out=SimpleNamespace(hidden_states=None),
            layer_outputs=[raw],
            start=1,
            end=3,
        )
        from_model_output = final_normalized_hidden_slice(
            model=model,
            out=SimpleNamespace(hidden_states=(torch.zeros_like(raw), norm(raw))),
            start=1,
            end=3,
        )
        self.assertTrue(torch.allclose(from_capture, expected, atol=1e-6))
        self.assertTrue(torch.allclose(from_layer_output, expected, atol=1e-6))
        self.assertTrue(torch.allclose(from_model_output, expected, atol=1e-6))

        token, patches = hidden_states_from_layer_outputs(
            [raw, raw + 1.0],
            token_position=2,
            visual_start=0,
            visual_end=2,
        )
        self.assertEqual(tuple(token.shape), (2, 3))
        self.assertEqual(tuple(patches.shape), (2, 2, 3))

    def test_layer_features_include_post_block_injection(self) -> None:
        raw_first = torch.zeros(1, 3, 2)
        injected_first = raw_first.clone()
        injected_first[0, :2] = 7.0
        raw_second = torch.full((1, 3, 2), 2.0)
        captures = [
            {
                "h_prev": torch.full((1, 3, 2), -1.0),
                "h_mid": raw_first,
                "o_ffn": torch.zeros_like(raw_first),
            },
            {
                # Qwen3 DeepStack is applied after layer 0 returns; layer 1's
                # pre-hook therefore contains the true post-injection output.
                "h_prev": injected_first,
                "h_mid": raw_second,
                "o_ffn": torch.zeros_like(raw_second),
            },
        ]
        token, patches = hidden_states_from_captures(
            captures,
            token_position=2,
            visual_start=0,
            visual_end=2,
        )
        self.assertTrue(torch.equal(patches[0], injected_first[0, :2]))
        self.assertTrue(torch.equal(patches[1], raw_second[0, :2]))
        self.assertTrue(torch.equal(token[0], injected_first[0, 2]))


if __name__ == "__main__":
    unittest.main()
