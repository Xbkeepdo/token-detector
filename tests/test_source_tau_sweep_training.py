from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_source_tau_transport_topk_sweep import (
    DEFAULT_RUN_NAMES,
    build_matrices,
    configured_capped_topmass_alphas,
    configured_training_methods,
    configured_training_risk_modes,
    configured_training_scopes,
    main,
    parse_args,
)
from utils.config_utils import load_config


class SourceTauSweepTrainingTests(unittest.TestCase):
    def test_cli_defaults_to_the_full_risk_curve(self) -> None:
        argv = [
            "train_source_tau_transport_topk_sweep.py",
            "--output-dir",
            "output",
            "--config",
            "config.yaml",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.layer_start, 0)
        self.assertIsNone(args.layer_end)
        self.assertIsNone(args.feature_method)
        self.assertIn("full", DEFAULT_RUN_NAMES["hpre_raw_logit_gauss"])
        self.assertIn("full", DEFAULT_RUN_NAMES["hpre_softmax_prob_gauss"])

    def test_full_layer_matrix_uses_every_risk_and_ev_layer(self) -> None:
        variant_slug = "tau0p1_topk64"
        rows = []
        for image_id, label in ((1, 0), (2, 1)):
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "dgst_t_target_region_top_k": 64,
                    "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_cosine_"
                    "topk64_hpre_per_layer": [4.0, 5.0, 6.0],
                    "dgst_t_hparam_sweep": {
                        variant_slug: {
                            "source_tau": 0.1,
                            "transport_top_k": 64,
                            "vv": {
                                "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_"
                                "per_layer": [1.0, 2.0, 3.0]
                            },
                        }
                    },
                }
            )

        matrices, labels, image_ids, audit = build_matrices(
            rows=rows,
            expected_variants={variant_slug: (0.1, 64)},
            layer_start=0,
            layer_end=None,
            scopes=("vv",),
        )

        matrix = matrices[f"vv_{variant_slug}_hpre_risk_plus_ev"]
        self.assertEqual(matrix.shape, (2, 6))
        np.testing.assert_allclose(matrix[0], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(image_ids, [1, 2])
        self.assertEqual(audit["risk_layer_slice"], [0, 3])
        self.assertEqual(audit["selected_risk_dimension"], 3)
        self.assertEqual(audit["scopes"], ["vv"])

    def test_yaml_switch_selects_methods_and_available_scopes(self) -> None:
        config = load_config(
            str(ROOT / "configs" / "model_configs_unified.yaml")
        )
        disabled = copy.deepcopy(config)
        disabled["training"]["source_tau_transport_topk_sweep"][
            "enabled"
        ] = False
        self.assertEqual(configured_training_methods(disabled), [])
        self.assertEqual(configured_training_scopes(config), ("vv",))
        self.assertEqual(
            configured_training_risk_modes(config),
            ("fixed_topk", "capped_topmass_alpha_sweep"),
        )
        self.assertEqual(
            configured_capped_topmass_alphas(config),
            (0.7, 0.75, 0.8, 0.85, 0.9),
        )

        enabled = copy.deepcopy(config)
        enabled["training"]["source_tau_transport_topk_sweep"][
            "enabled"
        ] = True
        self.assertEqual(
            configured_training_methods(enabled),
            ["hpre_raw_logit_gauss", "hpre_softmax_prob_gauss"],
        )

    def test_capped_mode_uses_capped_ev_and_deduplicates_fixed_topk(self) -> None:
        variants = {
            "tau0p1_topk16": (0.1, 16),
            "tau0p1_topk64": (0.1, 64),
        }
        capped_key = (
            "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_"
            "capped_topmass_085_per_layer"
        )
        rows = []
        for image_id, label in ((1, 0), (2, 1)):
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_"
                    "cosine_capped_topmass_085_hpre_per_layer": [7, 8, 9],
                    "dgst_t_hparam_sweep": {
                        slug: {
                            "source_tau": tau,
                            "transport_top_k": top_k,
                            "vv": {capped_key: [1, 2, 3]},
                        }
                        for slug, (tau, top_k) in variants.items()
                    },
                }
            )

        matrices, _labels, _image_ids, audit = build_matrices(
            rows=rows,
            expected_variants=variants,
            layer_start=0,
            layer_end=None,
            scopes=("vv",),
            risk_modes=("capped_topmass_085",),
        )

        self.assertEqual(list(matrices), [
            "vv_tau0p1_capped_topmass_085_hpre_risk_plus_ev"
        ])
        np.testing.assert_allclose(
            matrices["vv_tau0p1_capped_topmass_085_hpre_risk_plus_ev"][0],
            [1, 2, 3, 7, 8, 9],
        )
        self.assertEqual(audit["risk_modes"], ["capped_topmass_085"])

    def test_capped_alpha_sweep_builds_one_matrix_per_tau_and_alpha(self) -> None:
        variants = {
            "tau0p1_topk16": (0.1, 16),
            "tau0p1_topk64": (0.1, 64),
        }
        alpha_specs = {
            "capped_topmass_070": ([1, 2, 3], [7, 8, 9]),
            "capped_topmass_085": ([4, 5, 6], [10, 11, 12]),
        }
        rows = []
        for image_id, label in ((1, 0), (2, 1)):
            row = {"image_id": image_id, "label": label}
            for alpha_slug, (_risk, ev) in alpha_specs.items():
                row[
                    "dgst_t_hpre_raw_logit_gauss_ev_target_dist_mass_x_"
                    f"cosine_{alpha_slug}_hpre_per_layer"
                ] = ev
            row["dgst_t_hparam_sweep"] = {
                variant_slug: {
                    "source_tau": tau,
                    "transport_top_k": top_k,
                    "vv": {
                        "dgst_t_hpre_raw_logit_gauss_risk_sqrt_hpre_"
                        f"{alpha_slug}_per_layer": risk
                        for alpha_slug, (risk, _ev) in alpha_specs.items()
                    },
                }
                for variant_slug, (tau, top_k) in variants.items()
            }
            rows.append(row)

        matrices, _labels, _image_ids, audit = build_matrices(
            rows=rows,
            expected_variants=variants,
            layer_start=0,
            layer_end=None,
            scopes=("vv",),
            risk_modes=("capped_topmass_alpha_sweep",),
            capped_topmass_alphas=(0.7, 0.85),
        )

        self.assertEqual(
            list(matrices),
            [
                "vv_tau0p1_capped_topmass_070_hpre_risk_plus_ev",
                "vv_tau0p1_capped_topmass_085_hpre_risk_plus_ev",
            ],
        )
        np.testing.assert_allclose(
            matrices["vv_tau0p1_capped_topmass_070_hpre_risk_plus_ev"][0],
            [1, 2, 3, 7, 8, 9],
        )
        np.testing.assert_allclose(
            matrices["vv_tau0p1_capped_topmass_085_hpre_risk_plus_ev"][0],
            [4, 5, 6, 10, 11, 12],
        )
        self.assertEqual(
            audit["capped_topmass_alpha_by_slug"],
            {"capped_topmass_070": 0.7, "capped_topmass_085": 0.85},
        )

    def test_run_sh_invokes_sweep_after_main_training(self) -> None:
        text = (ROOT / "run.sh").read_text(encoding="utf-8")
        main_stage = text.index("scripts/train_and_eval.py")
        sweep_stage = text.index(
            "scripts/train_source_tau_transport_topk_sweep.py"
        )
        self.assertGreater(sweep_stage, main_stage)
        self.assertIn("--if-enabled", text[sweep_stage:])

    def test_enabled_switch_rejects_a_missing_grid_before_feature_load(self) -> None:
        config = {
            "training": {
                "source_tau_transport_topk_sweep": {
                    "enabled": True,
                    "feature_methods": ["hpre_raw_logit_gauss"],
                }
            },
            "feature_extraction": {
                "dgst_t": {
                    "support_modes": ["vv"],
                    "source_tau_values": None,
                    "transport_top_k_values": None,
                }
            },
        }
        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "train_source_tau_transport_topk_sweep.py",
                "--output-dir",
                output_dir,
                "--config",
                "config.yaml",
                "--if-enabled",
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.train_source_tau_transport_topk_sweep.load_config",
                return_value=config,
            ):
                with self.assertRaisesRegex(ValueError, "source_tau_values"):
                    main()


if __name__ == "__main__":
    unittest.main()
