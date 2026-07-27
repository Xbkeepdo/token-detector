from __future__ import annotations

import unittest

from utils.config_utils import get_labeling_cfg, manifest_validation_enabled


class LabelingConfigTests(unittest.TestCase):
    def test_schema_v2_defaults(self) -> None:
        config = get_labeling_cfg({})
        self.assertEqual(config["schema_version"], 2)
        self.assertEqual(config["sample_unit"], "first_canonical_mention")
        self.assertEqual(config["primary_locator"], "exact_response_offsets")
        self.assertTrue(config["save_all_mentions"])
        self.assertTrue(config["save_svar_official_samples"])
        self.assertEqual(config["alignment_failure_policy"], "error")

    def test_rejects_unknown_or_unsafe_protocols(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown labeling options"):
            get_labeling_cfg({"labeling": {"token_index": "word_idx"}})
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            get_labeling_cfg({"labeling": {"schema_version": 1}})
        with self.assertRaisesRegex(ValueError, "alignment_failure_policy"):
            get_labeling_cfg(
                {"labeling": {"alignment_failure_policy": "nearest_token"}}
            )
        with self.assertRaisesRegex(ValueError, "save_all_mentions must remain true"):
            get_labeling_cfg({"labeling": {"save_all_mentions": False}})

    def test_all_mentions_and_disabled_manifests_are_configurable(self) -> None:
        config = {
            "run": {"validate_manifests": False},
            "labeling": {"sample_unit": "all_mentions"},
        }
        self.assertEqual(get_labeling_cfg(config)["sample_unit"], "all_mentions")
        self.assertFalse(manifest_validation_enabled(config))


if __name__ == "__main__":
    unittest.main()
