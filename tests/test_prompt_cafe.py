from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from features.dgst_t import compute_four_gate_dgst_batch_from_captures
from features.extractor import _build_four_gate_feature_record
from models.dgst_capture import target_probabilities_multi
from train_feature_sets import build_selected_matrix, parse_feature_set


class PromptCafeTests(unittest.TestCase):
    def test_temperature_scaled_target_probabilities_match_full_softmax(self) -> None:
        output_layer = torch.nn.Linear(2, 3, bias=True)
        with torch.no_grad():
            output_layer.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5]])
            )
            output_layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        states = torch.tensor([[1.0, 2.0], [-0.5, 0.25]])
        actual = target_probabilities_multi(
            output_layer=output_layer,
            states=states,
            target_token_ids=[0, 2],
            temperature=10.0,
        )
        expected = F.softmax(output_layer(states).float() / 10.0, dim=-1)[:, [0, 2]]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        for invalid in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                target_probabilities_multi(
                    output_layer=output_layer,
                    states=states,
                    target_token_ids=[0],
                    temperature=invalid,
                )

    def test_four_gate_cafe_is_prompt_position_max_and_serializes_both_views(self) -> None:
        output_layer = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            output_layer.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [-0.5, 0.25]])
            )
        layer_states = (
            torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]
            ),
            torch.tensor(
                [[[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, 3.0], [3.0, 0.0]]]
            ),
        )
        captures = []
        for states in layer_states:
            attention = torch.zeros(1, 1, 5, 5)
            attention[0, 0, 4, 1:3] = torch.tensor([0.4, 0.6])
            captures.append(
                {
                    "h_prev": states,
                    "h_mid": states.clone(),
                    "o_attn": torch.zeros_like(states),
                    "o_ffn": torch.zeros_like(states),
                    "attn_weights": attention,
                }
            )

        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: output_layer),
            captures=captures,
            visual_start=1,
            visual_end=3,
            prompt_positions=[0, 3, 4],
            target_token_ids=[1],
            prediction_positions=[4],
            transport_top_k=2,
            target_region_top_k=2,
            enabled_methods=["hpre_raw_logit_gauss"],
            support_modes=["vv"],
            compute_prompt_cafe=True,
            prompt_cafe_temperature=10.0,
            prompt_cafe_layer=1,
        )[0]

        expected = []
        for states in layer_states:
            prompt_logits = output_layer(states[0, [3, 4]]).float() / 10.0
            expected.append(
                float(F.softmax(prompt_logits, dim=-1)[:, 1].max().detach())
            )
        self.assertTrue(
            torch.allclose(
                result["dgst_t_prompt_cafe_per_layer"],
                torch.tensor(expected),
                atol=1e-7,
            )
        )
        self.assertAlmostEqual(result["dgst_t_prompt_cafe"], expected[1], places=7)
        self.assertEqual(result["dgst_t_prompt_cafe_layer"], 1)
        self.assertEqual(result["dgst_t_prompt_cafe_prompt_size"], 2)
        self.assertEqual(result["dgst_t_prompt_cafe_temperature"], 10.0)
        self.assertEqual(
            result["dgst_t_prompt_cafe_position_scope"],
            "post_visual_instruction_tokens",
        )

        record = _build_four_gate_feature_record(
            image_id=7,
            span={"word": "chair", "label": 1},
            response_index=0,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        scalar, labels = build_selected_matrix([record], parse_feature_set("cafe"))
        curve, _ = build_selected_matrix(
            [record], parse_feature_set("prompt_cafe_per_layer")
        )
        self.assertEqual(scalar.shape, (1, 1))
        self.assertAlmostEqual(float(scalar[0, 0]), expected[1], places=7)
        self.assertEqual(curve.shape, (1, 2))
        self.assertEqual(labels.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
