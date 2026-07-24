from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from features.dgst_t import (
    _cosine_distance_matrix,
    _four_gate_risk_suffix,
    _four_gate_stateupd_alpha,
    _normalize_four_gate_cost_mode,
    _prepare_four_gate_cost_problem,
    compute_four_gate_dgst_batch_from_captures,
)
from features.extractor import _build_four_gate_feature_record
from scripts.run_pipeline import _enabled_method_feature_sets
from scripts.train_and_eval import build_training_commands
from train_feature_sets import build_selected_matrix, parse_feature_set
from utils.config_utils import load_config, resolve_dgst_four_gate_methods


ALPHA_MODES = [f"sqrt_stateupd_alpha0{value}" for value in range(1, 10)]
METHOD = "hpre_raw_logit_gauss"
SOFTMAX_METHOD = "hpre_softmax_prob_gauss"
SOFTMAX_RISK = f"{SOFTMAX_METHOD}_risk"
SOFTMAX_EV = f"{SOFTMAX_METHOD}_ev_target_dist_mass_x_cosine"
VP_SOFTMAX_METHOD = f"vp_{SOFTMAX_METHOD}"
VP_SOFTMAX_RISK = f"{VP_SOFTMAX_METHOD}_risk"
VP_SOFTMAX_EV = f"{VP_SOFTMAX_METHOD}_ev_target_dist_mass_x_cosine"
EV = f"{METHOD}_ev_target_dist_mass_x_cosine"
VP_METHOD = f"vp_{METHOD}"
VP_EV = f"{VP_METHOD}_ev_target_dist_mass_x_cosine"


def _expected_feature_sets() -> list[str]:
    values: list[str] = []
    for method, ev in ((METHOD, EV), (VP_METHOD, VP_EV)):
        costs = ["sqrt_matched_state", *ALPHA_MODES]
        for cost in costs:
            risk = f"{method}_risk_{cost}"
            values.extend((risk, f"{risk}+{ev}"))
    return values


def _expected_active_feature_sets() -> list[str]:
    values = list(_expected_feature_sets())
    for method, risk, ev in (
        (SOFTMAX_METHOD, SOFTMAX_RISK, SOFTMAX_EV),
        (VP_SOFTMAX_METHOD, VP_SOFTMAX_RISK, VP_SOFTMAX_EV),
    ):
        values.extend((risk, ev, f"{risk}+{ev}"))
        for cost in ALPHA_MODES:
            alpha_risk = f"{method}_risk_{cost}"
            values.extend((alpha_risk, f"{alpha_risk}+{ev}"))
    values.append("prompt_cafe")
    return values


class StateUpdateAlphaSweepTests(unittest.TestCase):
    def test_all_alpha_modes_normalize_and_encode_the_requested_weight(self) -> None:
        for value, mode in enumerate(ALPHA_MODES, start=1):
            self.assertEqual(_normalize_four_gate_cost_mode(mode), mode)
            self.assertAlmostEqual(_four_gate_stateupd_alpha(mode), value / 10.0)
            self.assertEqual(
                _four_gate_risk_suffix(mode, "hpre"),
                f"risk_{mode}",
            )
        with self.assertRaises(ValueError):
            _normalize_four_gate_cost_mode("sqrt_stateupd_alpha00")
        with self.assertRaises(ValueError):
            _normalize_four_gate_cost_mode("sqrt_stateupd_alpha10")

    def test_alpha_cost_is_the_exact_state_update_convex_mixture(self) -> None:
        source = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32)
        target = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        states = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        )
        updates = torch.tensor(
            [[0.2, 0.0], [0.0, 0.3], [-0.1, 0.2]],
            dtype=torch.float32,
        )
        support = torch.arange(3, dtype=torch.long)
        state_distance = torch.sqrt(
            (_cosine_distance_matrix(states) / 2.0).clamp_min(0.0)
        )
        update_distance = torch.sqrt(
            (_cosine_distance_matrix(updates) / 2.0).clamp_min(0.0)
        )
        for value in (1, 5, 9):
            alpha = value / 10.0
            _local_source, _local_target, actual = _prepare_four_gate_cost_problem(
                cost_mode=f"sqrt_stateupd_alpha0{value}",
                source_dist=source,
                target_dist=target,
                matched_states=states,
                hmid_states=states,
                hout_states=states + updates,
                update_states=updates,
                support=support,
            )
            expected = (1.0 - alpha) * state_distance + alpha * update_distance
            self.assertTrue(
                torch.allclose(torch.from_numpy(actual), expected, atol=1e-7)
            )

    def test_vv_vp_results_and_serialized_record_contain_every_alpha(self) -> None:
        output_layer = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            output_layer.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
                    dtype=torch.float32,
                )
            )
        h_prev = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        o_attn = torch.zeros_like(h_prev)
        o_ffn = torch.tensor(
            [[[0.2, 0.0], [0.0, 0.3], [-0.1, 0.2], [0.3, -0.1], [0.1, 0.2]]],
            dtype=torch.float32,
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
        result = compute_four_gate_dgst_batch_from_captures(
            model=SimpleNamespace(get_output_embeddings=lambda: output_layer),
            captures=[capture],
            visual_start=0,
            visual_end=3,
            prompt_positions=[3],
            target_token_ids=[1],
            prediction_positions=[4],
            transport_top_k=4,
            target_region_top_k=3,
            cost_mode="sqrt_matched_state",
            cost_modes=["sqrt_matched_state", *ALPHA_MODES],
            enabled_methods=[METHOD, SOFTMAX_METHOD],
            support_modes=["vv", "vp"],
            compute_prompt_cafe=True,
            prompt_cafe_temperature=10.0,
            prompt_cafe_layer=0,
        )[0]

        self.assertEqual(result["dgst_t_cost"], "sqrt_cosine_matched_state")
        self.assertEqual(
            result["dgst_t_cost_modes"],
            ["sqrt_cosine_matched_state", *ALPHA_MODES],
        )
        self.assertNotIn("dgst_t_cost_alpha", result)
        self.assertEqual(
            result["dgst_t_cost_alphas"],
            {mode: value / 10.0 for value, mode in enumerate(ALPHA_MODES, start=1)},
        )
        for scope_prefix in ("", "vp_"):
            for mode in ALPHA_MODES:
                key = f"dgst_t_{scope_prefix}{METHOD}_risk_{mode}_per_layer"
                self.assertEqual(tuple(result[key].shape), (1,))
                self.assertTrue(math.isfinite(float(result[key][0])))
        self.assertEqual(
            tuple(result[
                "dgst_t_hpre_softmax_prob_gauss_risk_sqrt_hpre_per_layer"
            ].shape),
            (1,),
        )
        for scope_prefix in ("", "vp_"):
            for mode in ALPHA_MODES:
                key = (
                    f"dgst_t_{scope_prefix}{SOFTMAX_METHOD}_"
                    f"risk_{mode}_per_layer"
                )
                self.assertEqual(tuple(result[key].shape), (1,))
                self.assertTrue(math.isfinite(float(result[key][0])))

        record = _build_four_gate_feature_record(
            image_id=12,
            span={"word": "chair", "label": 1},
            response_index=4,
            target_token_id=1,
            model_out=SimpleNamespace(token_id=1),
            dgst_t=result,
        )
        self.assertEqual(record["dgst_t_cost_alphas"], result["dgst_t_cost_alphas"])
        for feature_set in _expected_active_feature_sets():
            matrix, labels = build_selected_matrix(
                [record], parse_feature_set(feature_set)
            )
            expected_width = 2 if "+" in feature_set else 1
            self.assertEqual(matrix.shape, (1, expected_width))
            self.assertEqual(labels.tolist(), [1])

    def test_active_yamls_use_authoritative_methods_and_vpend_features(self) -> None:
        cases = {
            "configs/model_configs_unified.yaml": {
                "support_modes": ["vpend"],
                "cost_modes": ["sqrt_matched_state"],
                "features": [
                    "vpend_hpre_raw_logit_gauss_risk_sqrt_matched_state",
                    "vpend_hpre_raw_logit_gauss_risk_sqrt_matched_state+"
                    "vpend_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine",
                    "vpend_hpre_softmax_prob_gauss_risk",
                    "vpend_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine",
                    "vpend_hpre_softmax_prob_gauss_risk+"
                    "vpend_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine",
                    "prompt_cafe",
                ],
            },
            "configs/model_configs_server_fj01.yaml": {
                "support_modes": ["vv", "vpend"],
                "cost_modes": ["sqrt_matched_state", *ALPHA_MODES],
                "features": [
                    "vpend_hpre_softmax_prob_gauss_risk",
                    "vpend_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine",
                    "vpend_hpre_softmax_prob_gauss_risk+"
                    "vpend_hpre_softmax_prob_gauss_ev_target_dist_mass_x_cosine",
                ],
            },
        }
        retired_dgst_keys = {
            "branches",
            "target_modes",
            "save_raw_capture",
            "source_distribution_mode",
            "support_scope",
            "ot_solver",
            "target_attention_epsilon",
            "matrix_dtype",
            "curve_dtype",
        }
        retired_dataset_keys = {
            "split_strategy",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "validation",
        }
        for relative_path, expected in cases.items():
            config = load_config(os.path.join(ROOT, relative_path))
            dgst = config["feature_extraction"]["dgst_t"]
            self.assertEqual(
                resolve_dgst_four_gate_methods(dgst),
                [METHOD, SOFTMAX_METHOD],
            )
            self.assertTrue(retired_dgst_keys.isdisjoint(dgst))
            self.assertTrue(retired_dataset_keys.isdisjoint(config["dataset"]))
            self.assertNotIn("pope", config)
            self.assertNotIn(
                "clevr_object_coverage_policy", config["qa_benchmarks"]
            )
            self.assertNotIn(
                "shard_dtype",
                config["feature_extraction"]["baseline"]["dhcp"],
            )
            self.assertEqual(dgst["support_modes"], expected["support_modes"])
            self.assertEqual(dgst["cost_mode"], "sqrt_matched_state")
            self.assertEqual(dgst["cost_modes"], expected["cost_modes"])
            self.assertFalse(dgst["compute_ffn_injection_features"])
            self.assertTrue(dgst["compute_prompt_cafe"])
            self.assertEqual(float(dgst["prompt_cafe_temperature"]), 10.0)
            self.assertEqual(int(dgst["prompt_cafe_layer"]), 22)
            configured = config["training"]["feature_sets"]["method"]
            self.assertEqual(configured, expected["features"])
            self.assertEqual(
                _enabled_method_feature_sets(config, configured),
                expected["features"],
            )

            commands = build_training_commands(
                config=config,
                model="llava_1_5_7b",
                config_path=relative_path,
                output_dir="outputs/test-alpha-sweep",
                device="cpu",
            )
            root_commands = [
                command
                for command in commands
                if command[1] == "scripts/train_torch_probe_feature_sets.py"
            ]
            self.assertEqual(len(root_commands), 3)
            for command in root_commands:
                start = command.index("--feature-sets") + 1
                end = command.index("--device")
                expected_command_features = list(expected["features"])
                if config["run"]["extraction_mode"] == "all":
                    expected_command_features.extend(("ads", "cgc", "ads+cgc"))
                self.assertEqual(command[start:end], expected_command_features)


if __name__ == "__main__":
    unittest.main()
