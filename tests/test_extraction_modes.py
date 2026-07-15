from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from features.dgst_t import FOUR_GATE_METHODS
from features.extractor import (
    _build_enabled_feature_record,
    _feature_family_flags,
    _resolve_active_dgst_config,
    build_extraction_requirements,
    extract_features_for_dataset,
)
from models.base_wrapper import AttentionRequirement, ExtractionRequirements
from utils.io_utils import load_pkl
from detection.train import select_decision_threshold
from train_torch_probe_feature_sets import _select_validation_threshold
from train_feature_sets import (
    _require_strict_binary_splits,
    build_selected_matrix,
    parse_feature_set,
)


def _dgst_result(layers: int = 2, patches: int = 6) -> dict:
    value = {
        "dgst_t_profile": "four_gate_vv_v1",
        "dgst_t_four_gate_methods": list(FOUR_GATE_METHODS),
        "dgst_t_mad_axis": "visual_tokens",
        "dgst_t_mad_scale": 1.4826,
        "dgst_t_softmax_axis": "vocabulary",
        "dgst_t_source_distribution_mode": "softmax",
        "dgst_t_transport_top_k": 64,
        "dgst_t_target_region_top_k": 32,
        "dgst_t_cost": "sqrt_cosine_hpre",
        "dgst_t_ot_solver": "emd",
        "dgst_t_attention_support_per_layer": torch.full(
            (layers, patches), 1.0 / patches
        ),
        "dgst_t_source_dist_per_layer": torch.full(
            (layers, patches), 1.0 / patches
        ),
    }
    for offset, method in enumerate(FOUR_GATE_METHODS):
        value[f"dgst_t_{method}_gate_per_layer"] = torch.full(
            (layers, patches), 0.25 + 0.1 * offset
        )
        value[f"dgst_t_{method}_risk_sqrt_hpre_per_layer"] = torch.arange(
            layers, dtype=torch.float32
        ) + offset
        value[f"dgst_t_{method}_target_cosine_topk32_hpre_per_layer"] = torch.full(
            (layers,), 0.1 * offset
        )
        value[f"dgst_t_{method}_ev_topk32_hpre_per_layer"] = torch.full(
            (layers,), 0.2 * offset
        )
    return value


def _model_output() -> SimpleNamespace:
    layers, heads, patches, hidden = 2, 2, 6, 4
    return SimpleNamespace(
        token_id=9,
        text_to_patch_attn=torch.arange(
            layers * heads * patches, dtype=torch.float32
        ).reshape(layers, heads, patches) + 1,
        text_to_text_attn=torch.empty(0),
        token_hidden_states=torch.ones(layers, hidden),
        patch_hidden_states=torch.arange(
            layers * patches * hidden, dtype=torch.float32
        ).reshape(layers, patches, hidden) + 1,
        token_logits=None,
        dgst_t_raw=None,
        dgst_t_result=_dgst_result(layers, patches),
        visual_grid=(2, 3),
    )


class ExtractionModeTests(unittest.TestCase):
    def test_requirements_are_merged_by_enabled_family(self) -> None:
        method = build_extraction_requirements(
            method=True, ads_cgc=False, baseline=False
        )
        self.assertTrue(method.dgst_capture)
        self.assertEqual(method.attention, AttentionRequirement.NONE)
        self.assertFalse(method.logits)

        all_methods = build_extraction_requirements(
            method=True, ads_cgc=True, baseline=True
        )
        self.assertTrue(all_methods.dgst_capture)
        self.assertEqual(all_methods.attention, AttentionRequirement.PER_HEAD)
        self.assertFalse(all_methods.logits)
        self.assertTrue(all_methods.token_hidden_states)  # required by CGC
        self.assertTrue(all_methods.patch_hidden_states)
        self.assertTrue(all_methods.response_hidden_states)

        baseline_only = build_extraction_requirements(
            method=False, ads_cgc=False, baseline=True
        )
        self.assertFalse(baseline_only.logits)
        self.assertFalse(baseline_only.token_hidden_states)
        self.assertTrue(baseline_only.patch_hidden_states)

    def test_four_branch_boolean_switches_filter_method_list(self) -> None:
        cfg = {
            "enabled": True,
            "four_gate_methods": list(FOUR_GATE_METHODS),
            "branches": {
                method: method.endswith("raw_logit_gauss")
                for method in FOUR_GATE_METHODS
            },
        }
        resolved = _resolve_active_dgst_config(cfg)
        self.assertEqual(
            resolved["four_gate_methods"],
            ["hpre_raw_logit_gauss", "hmid_raw_logit_gauss"],
        )
        self.assertEqual(len(cfg["four_gate_methods"]), 4)

    def test_method_and_ads_cgc_can_share_one_record(self) -> None:
        feature_cfg = {
            "method": {"enabled": True},
            "ads_cgc": {"enabled": True},
            "baseline": {"enabled": False},
            "ads": {"top_patch_pct": 0.2},
            "cgc": {"top_k_pct": 0.5},
        }
        record = _build_enabled_feature_record(
            image_id=1,
            span={"word": "chair", "label": 0},
            response_index=3,
            target_token_id=9,
            model_out=_model_output(),
            cfg_dgst_t={"feature_output_profile": "four_gate_vv"},
            cfg_feature_extraction=feature_cfg,
            family_flags=_feature_family_flags(feature_cfg),
        )
        self.assertIn("dgst_t_hpre_raw_logit_gauss_gate_per_layer", record)
        self.assertEqual(record["ads_per_layer"].shape, (2,))
        self.assertEqual(record["cgc_per_layer"].shape, (2,))
        matrix, labels = build_selected_matrix(
            [record], parse_feature_set("ads+cgc")
        )
        self.assertEqual(matrix.shape, (1, 4))
        self.assertEqual(labels.tolist(), [0])

    def test_ads_cgc_only_does_not_require_dgst(self) -> None:
        feature_cfg = {
            "method": {"enabled": False},
            "ads_cgc": {"enabled": True},
            "baseline": {"enabled": False},
        }
        output = _model_output()
        output.dgst_t_result = None
        record = _build_enabled_feature_record(
            image_id=2,
            span={"word": "table", "label": 1},
            response_index=4,
            target_token_id=9,
            model_out=output,
            cfg_dgst_t=None,
            cfg_feature_extraction=feature_cfg,
            family_flags=_feature_family_flags(feature_cfg),
        )
        self.assertIn("ads_per_layer", record)
        self.assertFalse(any(key.startswith("dgst_t_") for key in record))

    def test_strict_trainer_never_substitutes_a_split(self) -> None:
        good_x = np.ones((2, 1), dtype=np.float32)
        good_y = np.array([0, 1], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "Splits are never substituted"):
            _require_strict_binary_splits(
                feature_set="ads",
                train=(good_x, good_y),
                val=(good_x, np.ones(2, dtype=np.int32)),
                test=(good_x, good_y),
            )

    def test_probe_thresholds_are_selected_from_validation_scores(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.int32)
        scores = np.array([0.40, 0.30, 0.35, 0.20], dtype=np.float32)
        sklearn_threshold = select_decision_threshold(labels, scores)
        torch_threshold = _select_validation_threshold(labels, scores)
        self.assertAlmostEqual(sklearn_threshold, 0.35, places=6)
        self.assertAlmostEqual(torch_threshold, 0.35, places=6)
        self.assertNotEqual(sklearn_threshold, 0.5)

    def test_joint_extraction_resumes_and_keeps_baseline_isolated(self) -> None:
        class Tokenizer:
            def encode(self, _text, add_special_tokens=False):
                return [9]

            def decode(self, _ids, skip_special_tokens=False):
                return "chair"

        class Wrapper:
            tokenizer = Tokenizer()

            def __init__(self):
                self.calls = 0

            def extract_token_features_batch(self, **_kwargs):
                self.calls += 1
                return [_model_output()]

        class Runtime:
            requirements = ExtractionRequirements()

            def build_image_records(self, *, image_id, spans, **_kwargs):
                return [
                    {
                        "feature_schema_version": "baseline-v1",
                        "image_id": int(image_id),
                        "response_token_idx": 0,
                        "token_str": spans[0]["word"],
                        "label": int(spans[0]["label"]),
                        "baselines": {"dummy": {"value": 1.0}},
                    }
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (4, 4), color="white").save(image_path)
            root_path = root / "features.pkl"
            baseline_path = root / "baseline" / "features.pkl"
            wrapper = Wrapper()
            feature_cfg = {
                "method": {"enabled": False},
                "ads_cgc": {"enabled": True},
                "baseline": {"enabled": True},
            }
            kwargs = {
                "model_wrapper": wrapper,
                "coco_samples": [{"image_id": 1, "image_path": str(image_path)}],
                "labeling_results": {
                    1: {
                        "generated_text": "chair",
                        "object_token_spans": [
                            {"word": "chair", "label": 0, "token_indices": [0]}
                        ],
                    }
                },
                "cfg_dgst_t": {},
                "output_path": str(root_path),
                "resume": True,
                "prompt": "Describe this image.",
                "cfg_feature_extraction": feature_cfg,
                "baseline_runtime": Runtime(),
                "baseline_output_path": str(baseline_path),
            }
            extract_features_for_dataset(**kwargs)
            extract_features_for_dataset(**kwargs)

            self.assertEqual(wrapper.calls, 1)
            root_records = load_pkl(str(root_path))
            baseline_records = load_pkl(str(baseline_path))
            self.assertEqual(len(root_records), 1)
            self.assertEqual(len(baseline_records), 1)
            self.assertIn("ads_per_layer", root_records[0])
            self.assertNotIn("baselines", root_records[0])
            self.assertIn("baselines", baseline_records[0])


if __name__ == "__main__":
    unittest.main()
