from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.extract_features import _stable_sha256
from scripts.training_provenance import (
    expected_feature_provenance,
    load_validated_training_features,
)
from utils.config_utils import get_model_cfg
from utils.generation_provenance import build_generation_manifest
from utils.io_utils import save_json, save_pkl


class TrainingProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        config = {
            "run": {
                "prompt": "Describe this image.",
                "extraction_mode": "all",
            },
            "labeling": {
                "schema_version": 2,
                "sample_unit": "first_canonical_mention",
                "primary_locator": "exact_response_offsets",
            },
            "models": {
                "model": {
                    "hf_name": "unused",
                    "num_layers": 2,
                    "max_new_tokens": 8,
                }
            },
            "feature_extraction": {
                "method": {"enabled": True},
                "ads_cgc": {"enabled": True},
                "baseline": {
                    "enabled": True,
                    "methods": ["svar"],
                    "svar": {
                        "protocols": ["controlled", "official"],
                        "layer_start": 0,
                        "layer_end": 2,
                    },
                },
                "ads": {"top_k_layers": 1},
                "cgc": {"top_k_pct": 0.1},
                "dgst_t": {"enabled": True},
            },
            "dataset": {"num_images": 10},
        }
        splits = {
            "train": list(range(1, 9)),
            "val": [],
            "test": [9, 10],
        }
        labeling = {}
        generations = {}
        for image_id in range(1, 11):
            word = f"object{image_id}"
            label = image_id % 2
            token_id = 100 + image_id
            official = {
                "word": word,
                "query": word,
                "search_term": word,
                "search_source": "canonical",
                "label": label,
                "token_indices": [0] if image_id <= 3 else [],
                "status": "found" if image_id <= 3 else "not_found",
                "token_location": {
                    "status": "found" if image_id <= 3 else "not_found",
                    "query": word,
                    "matched_query": word if image_id <= 3 else None,
                    "query_token_id": token_id,
                    "matched_token_id": token_id,
                    "token_indices": [0] if image_id <= 3 else [],
                    "used_plural_fallback": False,
                },
            }
            labeling[str(image_id)] = {
                "schema_version": 2,
                "image_id": image_id,
                "generated_text": word,
                "object_token_spans": [
                    {
                        "word": word,
                        "canonical_object": word,
                        "surface": word,
                        "token_indices": [0],
                        "label": label,
                    }
                ],
                "official_svar_samples": [official],
            }
            generations[str(image_id)] = {
                "generated_text": word,
                "response_token_ids": [token_id],
            }
        save_json(labeling, str(root / "labeling.json"))
        save_json(generations, str(root / "generations.json"))
        model_cfg = get_model_cfg(copy.deepcopy(config), "model")
        save_json(
            build_generation_manifest(
                model="model",
                model_cfg=model_cfg,
                prompt="Describe this image.",
                generations=generations,
                expected_image_ids=set(range(1, 11)),
            ),
            str(root / "generation_manifest.json"),
        )
        save_json(
            {
                "label_schema_version": 2,
                "primary_locator": "exact_response_offsets",
                "sample_unit": "first_canonical_mention",
                "labeling_sha256": _stable_sha256(labeling),
                "generation_sha256": _stable_sha256(generations),
            },
            str(root / "labeling_manifest.json"),
        )
        save_json(splits, str(root / "image_splits.json"))
        return config, splits, labeling

    def _records(self, labeling):
        return [
            {
                "image_id": int(image_id),
                "response_token_idx": 0,
                "token_str": row["object_token_spans"][0]["word"],
                "label": row["object_token_spans"][0]["label"],
            }
            for image_id, row in labeling.items()
        ]

    def _write_artifact(
        self,
        root: Path,
        *,
        family: str,
        config,
        records,
    ) -> Path:
        if family == "root":
            directory = root
        elif family == "baseline_controlled":
            directory = root / "baseline"
        else:
            directory = root / "baseline" / "svar_official"
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / "features.pkl"
        save_pkl(records, str(feature_path))
        save_json(
            expected_feature_provenance(
                artifact_family=family,
                model_key="model",
                config=config,
                output_dir=root,
            ),
            str(directory / "features_manifest.json"),
        )
        return feature_path

    def test_root_accepts_complete_matching_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, splits, labeling = self._fixture(root)
            path = self._write_artifact(
                root,
                family="root",
                config=config,
                records=self._records(labeling),
            )
            loaded = load_validated_training_features(
                feature_path=path,
                artifact_family="root",
                model_key="model",
                config=config,
                output_dir=root,
                image_splits=splits,
            )
            self.assertEqual(len(loaded), 10)

    def test_root_rejects_stale_prompt_and_partial_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, splits, labeling = self._fixture(root)
            records = self._records(labeling)
            path = self._write_artifact(
                root,
                family="root",
                config=config,
                records=records[:-1],
            )
            with self.assertRaisesRegex(RuntimeError, "partial or incompatible"):
                load_validated_training_features(
                    feature_path=path,
                    artifact_family="root",
                    model_key="model",
                    config=config,
                    output_dir=root,
                    image_splits=splits,
                )

            save_pkl(records, str(path))
            changed = copy.deepcopy(config)
            changed["run"]["prompt"] = "A different prompt."
            with self.assertRaisesRegex(
                RuntimeError, "generation_manifest.json|provenance mismatch"
            ):
                load_validated_training_features(
                    feature_path=path,
                    artifact_family="root",
                    model_key="model",
                    config=changed,
                    output_dir=root,
                    image_splits=splits,
                )

    def test_controlled_and_official_use_their_exact_label_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, splits, labeling = self._fixture(root)
            controlled = self._write_artifact(
                root,
                family="baseline_controlled",
                config=config,
                records=self._records(labeling),
            )
            controlled_records = load_validated_training_features(
                feature_path=controlled,
                artifact_family="baseline_controlled",
                model_key="model",
                config=config,
                output_dir=root,
                image_splits=splits,
            )
            self.assertEqual(len(controlled_records), 10)

            official_records = []
            for image_id in range(1, 4):
                row = labeling[str(image_id)]
                sample = row["official_svar_samples"][0]
                official_records.append(
                    {
                        "image_id": image_id,
                        "response_token_idx": 0,
                        "token_str": sample["search_term"],
                        "label": sample["label"],
                        "metadata": {
                            "svar_protocol": "official",
                            "svar_official": {
                                "search_source": sample["search_source"],
                            },
                        },
                    }
                )
            official = self._write_artifact(
                root,
                family="baseline_svar_official",
                config=config,
                records=official_records,
            )
            loaded = load_validated_training_features(
                feature_path=official,
                artifact_family="baseline_svar_official",
                model_key="model",
                config=config,
                output_dir=root,
                image_splits=splits,
            )
            self.assertEqual(len(loaded), 3)

            save_pkl(official_records[:-1], str(official))
            with self.assertRaisesRegex(RuntimeError, "expected=3, actual=2"):
                load_validated_training_features(
                    feature_path=official,
                    artifact_family="baseline_svar_official",
                    model_key="model",
                    config=config,
                    output_dir=root,
                    image_splits=splits,
                )

    def test_split_must_cover_the_labeling_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, splits, labeling = self._fixture(root)
            path = self._write_artifact(
                root,
                family="root",
                config=config,
                records=self._records(labeling),
            )
            bad_splits = copy.deepcopy(splits)
            bad_splits["test"] = [9, 999]
            with self.assertRaisesRegex(ValueError, "coverage"):
                load_validated_training_features(
                    feature_path=path,
                    artifact_family="root",
                    model_key="model",
                    config=config,
                    output_dir=root,
                    image_splits=bad_splits,
                )


if __name__ == "__main__":
    unittest.main()
