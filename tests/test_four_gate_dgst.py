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
    _cosine_distance_matrix,
    _gaussian_mad_gate,
    _prepare_transport_problem_for_state_cost,
    _solve_transport_problem,
    _stable_topk_indices,
    _topk_union_indices,
)
from features.extractor import _build_four_gate_feature_record
from models.dgst_capture import (
    attention_row_from_capture,
    chunked_eager_attention_forward,
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    run_forward_with_dgst_captures,
    target_logits_and_probabilities_multi,
)
from models.base_wrapper import configure_image_processor_limits
from train_feature_sets import build_selected_matrix, parse_feature_set


class FourGateDGSTTests(unittest.TestCase):
    def test_chunked_eager_attention_matches_full_formula(self) -> None:
        torch.manual_seed(7)
        query = torch.randn(1, 4, 5, 3, dtype=torch.float32)
        key = torch.randn(1, 2, 5, 3, dtype=torch.float32)
        value = torch.randn(1, 2, 5, 3, dtype=torch.float32)
        mask = torch.full((1, 1, 5, 5), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        module = SimpleNamespace(
            num_key_value_groups=2,
            training=False,
            _dgst_attention_query_chunk_size=2,
            _dgst_attention_query_positions=(1, -1),
        )
        actual_output, actual_rows = chunked_eager_attention_forward(
            module,
            query,
            key,
            value,
            mask,
            scaling=3.0 ** -0.5,
        )

        repeated_key = key.repeat_interleave(2, dim=1)
        repeated_value = value.repeat_interleave(2, dim=1)
        expected_weights = torch.softmax(
            torch.matmul(query, repeated_key.transpose(2, 3)) * (3.0 ** -0.5)
            + mask,
            dim=-1,
            dtype=torch.float32,
        )
        expected_output = torch.matmul(expected_weights, repeated_value).transpose(1, 2)
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-6))
        self.assertTrue(
            torch.allclose(actual_rows, expected_weights[:, :, [1, 4], :], atol=1e-6)
        )

    def test_forward_capture_retains_only_requested_attention_rows(self) -> None:
        class FakeAttention(torch.nn.Module):
            def forward(self, hidden):
                batch, sequence, _hidden = hidden.shape
                values = torch.arange(
                    sequence * sequence,
                    dtype=hidden.dtype,
                    device=hidden.device,
                ).reshape(1, 1, sequence, sequence)
                return torch.zeros_like(hidden), values.expand(batch, 2, -1, -1)

        class FakeLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()
                self.mlp = torch.nn.Identity()

            def forward(self, hidden):
                attention_update, attention = self.self_attn(hidden)
                h_mid = hidden + attention_update
                return h_mid + self.mlp(h_mid), attention

        class FakeBody(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeBody()

            def forward(self, input_ids, **_kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
                attentions = []
                for layer in self.model.layers:
                    hidden, attention = layer(hidden)
                    attentions.append(attention)
                return SimpleNamespace(
                    logits=torch.zeros(*input_ids.shape, 3),
                    attentions=tuple(attentions),
                    hidden_states=None,
                )

        outputs, captures = run_forward_with_dgst_captures(
            FakeModel(),
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            output_hidden_states=False,
            attention_query_positions=[1, -1],
        )
        self.assertEqual(outputs.attentions[0].shape, (1, 2, 2, 4))
        self.assertEqual(captures[0]["attention_query_positions"], (1, 3))
        self.assertTrue(
            torch.equal(
                attention_row_from_capture(captures[0], 3),
                torch.tensor([[12.0, 13.0, 14.0, 15.0]]).expand(2, -1),
            )
        )

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

    def test_configured_top16_uses_each_branch_own_target_region(self) -> None:
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
            target_region_top_k=16,
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
            region = _stable_topk_indices(target_dist, 16)
            regions.append(tuple(region.tolist()))
            state_name = "hmid" if method.startswith("hmid_") else "hpre"
            branch_cosine = (
                hmid_cosine_map if state_name == "hmid" else cosine_map
            )
            expected_cosine = branch_cosine.index_select(0, region).mean()
            expected_ev = (
                target_dist.index_select(0, region).sum()
                * expected_cosine
            )
            key = (
                f"dgst_t_{method}_target_cosine_topk16_"
                f"{state_name}_per_layer"
            )
            self.assertAlmostEqual(float(result[key][0]), float(expected_cosine), places=6)
            ev_key = (
                f"dgst_t_{method}_ev_target_dist_mass_x_cosine_"
                f"topk16_{state_name}_per_layer"
            )
            self.assertAlmostEqual(float(result[ev_key][0]), float(expected_ev), places=6)
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

    def test_geo_stateupd_lu1_uses_hmid_and_ffn_update_distances(self) -> None:
        layer = self._output_layer()
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        o_attn = torch.tensor(
            [[[0.0, 0.2], [0.1, 0.0], [0.2, -0.1], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        o_ffn = torch.tensor(
            [[[0.2, 0.0], [0.0, 0.3], [-0.1, 0.2], [0.3, -0.1]]],
            dtype=torch.float32,
        )
        attention = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        attention[0, 0, 3, :3] = torch.tensor([0.6, 0.3, 0.1])
        attention[0, 1, 3, :3] = torch.tensor([0.2, 0.3, 0.5])
        h_mid = h_prev + o_attn
        capture = {
            "h_prev": h_prev,
            "h_mid": h_mid,
            "o_attn": o_attn,
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            transport_top_k=3,
            cost_mode="geo_stateupd_lu1",
            enabled_methods=[RAW_ATTENTION_METHOD],
        )[0]

        source = result["dgst_t_source_dist_per_layer"][0]
        target = result["dgst_t_attention_support_per_layer"][0]
        support = _topk_union_indices(source, target, 3)
        expected_problem = _prepare_transport_problem_for_state_cost(
            source_dist=source,
            target_dist=target,
            states=h_mid[0, :3],
            output_states=(h_mid + o_ffn)[0, :3],
            support=support,
            sqrt_cosine=False,
            cost_state_mode="state_update",
            update_lambda=1.0,
        )
        expected_risk = _solve_transport_problem(expected_problem, "emd")
        risk_key = "dgst_t_raw_attention_risk_geo_stateupd_lu1_per_layer"
        self.assertEqual(result["dgst_t_cost"], "geo_stateupd_lu1")
        self.assertAlmostEqual(float(result[risk_key][0]), expected_risk, places=6)

        record = _build_four_gate_feature_record(
            image_id=11,
            span={"word": "bus", "label": 1},
            response_index=3,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        matrix, labels = build_selected_matrix(
            [record], parse_feature_set("raw_attention_risk")
        )
        self.assertEqual(matrix.shape, (1, 1))
        self.assertAlmostEqual(float(matrix[0, 0]), expected_risk, places=6)
        self.assertEqual(labels.tolist(), [1])

    def test_hpre_cosine_cost_compares_relative_vll_and_gaussian_targets(self) -> None:
        layer = self._output_layer()
        h_prev = torch.tensor(
            [[
                [1.0, -2.0],
                [0.0, -0.5],
                [-1.0, 0.7],
                [0.5, 3.0],
                [1.0, 1.0],
            ]],
            dtype=torch.float32,
        )
        o_attn = torch.zeros_like(h_prev)
        o_attn[0, :4] = torch.tensor(
            [[0.0, 0.1], [0.1, 0.0], [0.0, -0.1], [-0.1, 0.0]]
        )
        o_ffn = torch.zeros_like(h_prev)
        o_ffn[0, :4] = torch.tensor(
            [[0.1, 0.0], [0.0, 0.2], [-0.1, 0.1], [0.2, 0.1]]
        )
        attention = torch.zeros(1, 2, 5, 5, dtype=torch.float32)
        attention[0, 0, 4, :4] = torch.tensor([0.4, 0.2, 0.1, 0.3])
        attention[0, 1, 4, :4] = torch.tensor([0.1, 0.3, 0.2, 0.4])
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + o_attn,
            "o_attn": o_attn,
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        methods = [
            "hpre_raw_logit_gauss",
            "hpre_raw_logit_relative_vll",
        ]
        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=4,
            target_token_ids=[1],
            prediction_positions=[4],
            transport_top_k=4,
            target_region_top_k=4,
            cost_mode="sqrt_matched_state",
            cost_modes=["sqrt_matched_state", "cosine_matched_state"],
            enabled_methods=methods,
        )[0]

        self.assertEqual(
            result["dgst_t_cost_modes"],
            ["sqrt_cosine_matched_state", "cosine_matched_state"],
        )
        self.assertEqual(
            result["dgst_t_mad_scale_by_method"],
            {
                "hpre_raw_logit_gauss": 1.4826,
                "hpre_raw_logit_relative_vll": 1.0,
            },
        )
        relative_gate = result[
            "dgst_t_hpre_raw_logit_relative_vll_gate_per_layer"
        ][0]
        gaussian_gate = result["dgst_t_hpre_raw_logit_gauss_gate_per_layer"][0]
        self.assertFalse(torch.allclose(relative_gate, gaussian_gate))
        self.assertGreater(
            float(relative_gate.max() - relative_gate.min()),
            float(gaussian_gate.max() - gaussian_gate.min()),
        )

        source = result["dgst_t_source_dist_per_layer"][0]
        attention_dist = result["dgst_t_attention_support_per_layer"][0]
        for method, gate in (
            ("hpre_raw_logit_gauss", gaussian_gate),
            ("hpre_raw_logit_relative_vll", relative_gate),
        ):
            target = attention_dist * gate
            target = target / target.sum()
            support = _topk_union_indices(source, target, 4)
            expected_problem = _prepare_transport_problem_for_state_cost(
                source_dist=source,
                target_dist=target,
                states=h_prev[0, :4],
                support=support,
                sqrt_cosine=False,
            )
            expected_risk = _solve_transport_problem(expected_problem, "emd")
            risk_key = f"dgst_t_{method}_risk_cosine_hpre_per_layer"
            self.assertAlmostEqual(
                float(result[risk_key][0]), expected_risk, places=6
            )

        record = _build_four_gate_feature_record(
            image_id=13,
            span={"word": "chair", "label": 1},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        matrix, labels = build_selected_matrix(
            [record],
            parse_feature_set(
                "hpre_raw_logit_gauss_risk_cosine_matched_state+"
                "hpre_raw_logit_relative_vll_risk_cosine_matched_state"
            ),
        )
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(labels.tolist(), [1])

    def test_sqrt_stateupd_alpha05_matches_requested_mixture(self) -> None:
        layer = self._output_layer()
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]]],
            dtype=torch.float32,
        )
        o_attn = torch.tensor(
            [[[0.0, 0.2], [0.1, 0.0], [0.2, -0.1], [0.0, 0.0]]],
            dtype=torch.float32,
        )
        o_ffn = torch.tensor(
            [[[0.2, 0.0], [0.0, 0.3], [-0.1, 0.2], [0.3, -0.1]]],
            dtype=torch.float32,
        )
        attention = torch.zeros(1, 2, 4, 4, dtype=torch.float32)
        attention[0, 0, 3, :3] = torch.tensor([0.6, 0.3, 0.1])
        attention[0, 1, 3, :3] = torch.tensor([0.2, 0.3, 0.5])
        h_mid = h_prev + o_attn
        capture = {
            "h_prev": h_prev,
            "h_mid": h_mid,
            "o_attn": o_attn,
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }
        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            transport_top_k=3,
            cost_mode="sqrt_stateupd_alpha05",
            cost_modes=[
                "sqrt_stateupd_alpha05",
                "sqrt_matched_state",
                "geo_stateupd_lu1",
            ],
            enabled_methods=[RAW_ATTENTION_METHOD],
        )[0]

        source = result["dgst_t_source_dist_per_layer"][0]
        target = result["dgst_t_attention_support_per_layer"][0]
        support = _topk_union_indices(source, target, 3)
        local_source = source.index_select(0, support)
        local_source = local_source / local_source.sum()
        local_target = target.index_select(0, support)
        local_target = local_target / local_target.sum()
        local_state = h_prev[0, :3].index_select(0, support)
        local_update = o_ffn[0, :3].index_select(0, support)
        state_distance = torch.sqrt(
            (_cosine_distance_matrix(local_state) / 2.0).clamp_min(0.0)
        )
        update_distance = torch.sqrt(
            (_cosine_distance_matrix(local_update) / 2.0).clamp_min(0.0)
        )
        expected_cost = 0.5 * state_distance + 0.5 * update_distance
        expected_risk = _solve_transport_problem(
            (local_source, local_target, expected_cost), "emd"
        )
        risk_key = "dgst_t_raw_attention_risk_sqrt_stateupd_alpha05_per_layer"
        self.assertEqual(result["dgst_t_cost"], "sqrt_stateupd_alpha05")
        self.assertEqual(
            result["dgst_t_cost_modes"],
            [
                "sqrt_stateupd_alpha05",
                "sqrt_cosine_matched_state",
                "geo_stateupd_lu1",
            ],
        )
        self.assertEqual(result["dgst_t_cost_alpha"], 0.5)
        self.assertAlmostEqual(float(result[risk_key][0]), expected_risk, places=6)
        self.assertIn("dgst_t_raw_attention_risk_sqrt_hpre_per_layer", result)
        self.assertIn("dgst_t_raw_attention_risk_geo_stateupd_lu1_per_layer", result)

        record = _build_four_gate_feature_record(
            image_id=12,
            span={"word": "car", "label": 0},
            response_index=3,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        matrix, labels = build_selected_matrix(
            [record],
            parse_feature_set(
                "raw_attention_risk_sqrt_stateupd_alpha05+"
                "raw_attention_risk_sqrt_matched_state+"
                "raw_attention_risk_geo_stateupd_lu1"
            ),
        )
        self.assertEqual(matrix.shape, (1, 3))
        self.assertAlmostEqual(float(matrix[0, 0]), expected_risk, places=6)
        self.assertEqual(record["dgst_t_cost_alpha"], 0.5)
        self.assertEqual(labels.tolist(), [0])

        hmid_result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: layer),
            captures=[capture],
            visual_start=0,
            visual_end=3,
            target_token_ids=[1],
            prediction_positions=[3],
            transport_top_k=3,
            cost_mode="sqrt_stateupd_alpha05",
            enabled_methods=["hmid_raw_logit_gauss"],
        )[0]
        hmid_source = hmid_result["dgst_t_source_dist_per_layer"][0]
        hmid_attention = hmid_result["dgst_t_attention_support_per_layer"][0]
        hmid_gate = hmid_result[
            "dgst_t_hmid_raw_logit_gauss_gate_per_layer"
        ][0]
        hmid_target = hmid_attention * hmid_gate
        hmid_target = hmid_target / hmid_target.sum()
        hmid_support = _topk_union_indices(hmid_source, hmid_target, 3)
        hmid_local_source = hmid_source.index_select(0, hmid_support)
        hmid_local_source = hmid_local_source / hmid_local_source.sum()
        hmid_local_target = hmid_target.index_select(0, hmid_support)
        hmid_local_target = hmid_local_target / hmid_local_target.sum()
        hmid_local_state = h_mid[0, :3].index_select(0, hmid_support)
        hmid_local_update = o_ffn[0, :3].index_select(0, hmid_support)
        hmid_state_distance = torch.sqrt(
            (_cosine_distance_matrix(hmid_local_state) / 2.0).clamp_min(0.0)
        )
        hmid_update_distance = torch.sqrt(
            (_cosine_distance_matrix(hmid_local_update) / 2.0).clamp_min(0.0)
        )
        hmid_expected_risk = _solve_transport_problem(
            (
                hmid_local_source,
                hmid_local_target,
                0.5 * hmid_state_distance + 0.5 * hmid_update_distance,
            ),
            "emd",
        )
        self.assertAlmostEqual(
            float(
                hmid_result[
                    "dgst_t_hmid_raw_logit_gauss_"
                    "risk_sqrt_stateupd_alpha05_per_layer"
                ][0]
            ),
            hmid_expected_risk,
            places=6,
        )

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
            expected_cosine = float(cosine.mean().item())
            # K=32 covers all three visual tokens, so the normalized target
            # distribution contributes mass 1.0 for every method.
            expected_ev = expected_cosine
            gate_key = f"dgst_t_{method}_gate_per_layer"
            risk_key = (
                f"dgst_t_{method}_risk_sqrt_{state_name}_per_layer"
            )
            cosine_key = (
                f"dgst_t_{method}_target_cosine_topk32_"
                f"{state_name}_per_layer"
            )
            ev_key = (
                f"dgst_t_{method}_ev_target_dist_mass_x_cosine_"
                f"topk32_{state_name}_per_layer"
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
                "hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine"
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
        # raw_attention uses attention itself as target-dist.  K=32 covers all
        # tokens here, hence target-region mass is one.
        expected_ev = expected_cosine
        self.assertAlmostEqual(
            float(result["dgst_t_raw_attention_target_cosine_topk32_hpre_per_layer"][0]),
            expected_cosine,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                result[
                    "dgst_t_raw_attention_ev_target_dist_mass_x_cosine_"
                    "topk32_hpre_per_layer"
                ][0]
            ),
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
                "hpre_softmax_prob_direct_ev_target_dist_mass_x_cosine"
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

    def test_dual_scope_hpre_raw_and_ffn_fad_are_serialized_and_trainable(self) -> None:
        layer = self._output_layer()
        model = SimpleNamespace(get_output_embeddings=lambda: layer)
        h_prev = torch.tensor(
            [[
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
            ]],
            dtype=torch.float32,
        )
        o_attn = torch.zeros_like(h_prev)
        o_attn[0, 4] = torch.tensor([0.3, 0.4])
        o_ffn = torch.zeros_like(h_prev)
        o_ffn[0, :4] = torch.tensor(
            [[0.1, 0.0], [0.0, 0.2], [-0.1, 0.1], [0.2, 0.1]]
        )
        o_ffn[0, 4] = torch.tensor([0.6, 0.8])
        attention = torch.zeros(1, 2, 5, 5, dtype=torch.float32)
        attention[0, 0, 4, :4] = torch.tensor([0.4, 0.2, 0.1, 0.3])
        attention[0, 1, 4, :4] = torch.tensor([0.1, 0.3, 0.2, 0.4])
        capture = {
            "h_prev": h_prev,
            "h_mid": h_prev + o_attn,
            "o_attn": o_attn,
            "o_ffn": o_ffn,
            "attn_weights": attention,
        }

        result = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=[capture],
            visual_start=0,
            visual_end=3,
            prompt_positions=[3],
            target_token_ids=[1],
            prediction_positions=[4],
            transport_top_k=4,
            target_region_top_k=4,
            cost_mode="sqrt_matched_state",
            cost_modes=["sqrt_matched_state", "sqrt_stateupd_alpha05"],
            enabled_methods=["hpre_raw_logit_gauss"],
            support_modes=["vv", "vp"],
            compute_ffn_injection_features=True,
        )[0]

        self.assertEqual(
            result["dgst_t_four_gate_support_scopes"],
            ["visual", "visual_prompt"],
        )
        self.assertEqual(
            tuple(result["dgst_t_attention_support_per_layer"].shape), (1, 3)
        )
        self.assertEqual(
            tuple(result["dgst_t_vp_attention_support_per_layer"].shape), (1, 4)
        )
        expected_fad = torch.log(torch.tensor(1.0 / 0.5))
        self.assertAlmostEqual(
            float(result["dgst_t_ffn_attn_dominance_per_layer"][0]),
            float(expected_fad),
            places=6,
        )

        record = _build_four_gate_feature_record(
            image_id=12,
            span={"word": "chair", "label": 1},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        matrix, labels = build_selected_matrix(
            [record],
            parse_feature_set(
                "ffn_fad+vp_hpre_raw_logit_gauss_risk_sqrt_matched_state"
            ),
        )
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(labels.tolist(), [1])

        fad = record["dgst_t_ffn_attn_dominance_per_layer"]
        risk = record[
            "dgst_t_hpre_raw_logit_gauss_"
            "risk_sqrt_hpre_per_layer"
        ]
        ev = record[
            "dgst_t_hpre_raw_logit_gauss_"
            "ev_target_dist_mass_x_cosine_topk4_hpre_per_layer"
        ]
        product_name = (
            "ffn_fad*"
            "hpre_raw_logit_gauss_risk_sqrt_matched_state"
        )
        expected_product = fad * risk
        vp_risk = record[
            "dgst_t_vp_hpre_raw_logit_gauss_"
            "risk_sqrt_hpre_per_layer"
        ]
        for feature_set, expected in (
            ("ffn_fad", fad),
            (product_name, expected_product),
            (
                product_name
                + "+hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine",
                torch.cat(
                    (
                        torch.as_tensor(expected_product),
                        torch.as_tensor(ev),
                    )
                ).numpy(),
            ),
            (
                "ffn_fad*"
                "vp_hpre_raw_logit_gauss_risk_sqrt_matched_state",
                fad * vp_risk,
            ),
        ):
            selected, _ = build_selected_matrix(
                [record], parse_feature_set(feature_set)
            )
            self.assertEqual(selected.shape, (1, len(expected)))
            self.assertTrue(
                torch.allclose(
                    torch.as_tensor(selected[0]),
                    torch.as_tensor(expected),
                    atol=1e-6,
                )
            )

        vv_only = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=[capture],
            visual_start=0,
            visual_end=3,
            prompt_positions=[3],
            target_token_ids=[1],
            prediction_positions=[4],
            enabled_methods=["hpre_raw_logit_gauss"],
            support_modes=["vv"],
        )[0]
        self.assertIn("dgst_t_hpre_raw_logit_gauss_gate_per_layer", vv_only)
        self.assertNotIn("dgst_t_vp_hpre_raw_logit_gauss_gate_per_layer", vv_only)

        vp_only = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=[capture],
            visual_start=0,
            visual_end=3,
            prompt_positions=[3],
            target_token_ids=[1],
            prediction_positions=[4],
            enabled_methods=["hpre_raw_logit_gauss"],
            support_modes=["vp"],
        )[0]
        self.assertNotIn("dgst_t_hpre_raw_logit_gauss_gate_per_layer", vp_only)
        self.assertIn("dgst_t_vp_hpre_raw_logit_gauss_gate_per_layer", vp_only)
        vp_record = _build_four_gate_feature_record(
            image_id=12,
            span={"word": "chair", "label": 1},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=vp_only,
        )
        vp_matrix, _ = build_selected_matrix(
            [vp_record],
            parse_feature_set("vp_hpre_raw_logit_gauss_risk"),
        )
        self.assertEqual(vp_matrix.shape, (1, 1))

        # VPend must follow the same positional definition as prompt CAFE:
        # keep visual tokens and prompt positions >= visual_end, while
        # excluding the image-prefix prompt token at position 0.
        vpend_only = compute_four_gate_dgst_batch_from_captures(
            model=model,
            captures=[capture],
            visual_start=1,
            visual_end=3,
            prompt_positions=[0, 3],
            target_token_ids=[1],
            prediction_positions=[4],
            enabled_methods=["hpre_raw_logit_gauss"],
            support_modes=["vpend"],
        )[0]
        self.assertEqual(
            vpend_only["dgst_t_four_gate_support_scopes"],
            ["visual_prompt_end"],
        )
        self.assertEqual(vpend_only["dgst_t_vpend_support_positions"], [1, 2, 3])
        self.assertEqual(vpend_only["dgst_t_vpend_support_size"], 3)
        self.assertEqual(
            tuple(vpend_only["dgst_t_vpend_attention_support_per_layer"].shape),
            (1, 3),
        )
        self.assertNotIn("dgst_t_vp_attention_support_per_layer", vpend_only)
        vpend_record = _build_four_gate_feature_record(
            image_id=12,
            span={"word": "chair", "label": 1},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=vpend_only,
        )
        vpend_matrix, _ = build_selected_matrix(
            [vpend_record],
            parse_feature_set(
                "vpend_hpre_raw_logit_gauss_risk+"
                "vpend_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine"
            ),
        )
        self.assertEqual(vpend_matrix.shape, (1, 2))

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
