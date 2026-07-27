from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from PIL import Image

from features.extractor import (
    _extract_token_features_fallback,
    _resolve_causal_target,
    extract_features_for_dataset,
)
from models.base_wrapper import BaseLVLMWrapper, ExtractionRequirements
from utils.io_utils import load_pkl, save_pkl


class CausalTokenPositionTests(unittest.TestCase):
    def test_object_index_is_the_target_and_prefix_stops_before_it(self) -> None:
        response_ids = [101, 102, 103]
        index, target, prefix = _resolve_causal_target(
            response_token_ids=response_ids,
            span={"word": "phone", "token_indices": [2]},
            image_id=7,
        )
        self.assertEqual(index, 2)
        self.assertEqual(target, 103)
        self.assertEqual(prefix, [101, 102])

    def test_first_generated_token_has_an_empty_response_prefix(self) -> None:
        index, target, prefix = _resolve_causal_target(
            response_token_ids=[11, 12],
            span={"word": "chair", "token_indices": [0]},
        )
        self.assertEqual((index, target, prefix), (0, 11, []))

    def test_negative_and_out_of_range_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            _resolve_causal_target(
                response_token_ids=[1, 2],
                span={"token_indices": [-1]},
                image_id=3,
            )
        with self.assertRaisesRegex(ValueError, "outside response length"):
            _resolve_causal_target(
                response_token_ids=[1, 2],
                span={"token_indices": [2]},
                image_id=3,
            )

    def test_wrapper_rejects_target_that_is_not_saved_response_token(self) -> None:
        response_ids, indices, targets = (
            BaseLVLMWrapper.validate_causal_batch_request(
                response_token_ids=[101, 102, 103],
                response_token_indices=[0, 2],
                target_token_ids=[101, 103],
            )
        )
        self.assertEqual(response_ids, [101, 102, 103])
        self.assertEqual(indices, [0, 2])
        self.assertEqual(targets, [101, 103])
        with self.assertRaisesRegex(ValueError, "actual saved response token"):
            BaseLVLMWrapper.validate_causal_batch_request(
                response_token_ids=[101, 102, 103],
                response_token_indices=[2],
                target_token_ids=[102],
            )
        with self.assertRaisesRegex(ValueError, "outside response length"):
            BaseLVLMWrapper.validate_causal_batch_request(
                response_token_ids=[101],
                response_token_indices=[1],
            )

    def test_fallback_forward_receives_only_the_predictor_prefix(self) -> None:
        class Wrapper:
            def __init__(self) -> None:
                self.received = None

            def extract_token_features(self, **kwargs):
                self.received = kwargs
                return object()

        wrapper = Wrapper()
        outputs = _extract_token_features_fallback(
            model_wrapper=wrapper,
            image=object(),
            image_id=9,
            spans=[{"word": "phone", "token_indices": [2]}],
            response_token_ids=[21, 22, 23],
            response_indices=[2],
            target_token_ids=[23],
        )
        self.assertEqual(len(outputs), 1)
        self.assertEqual(wrapper.received["prefix_token_ids"], [21, 22])
        self.assertEqual(wrapper.received["response_token_idx"], 2)
        self.assertEqual(wrapper.received["target_token_id"], 23)

    def test_controlled_and_official_positions_share_one_batch_forward(self) -> None:
        class Tokenizer:
            def encode(self, _text, add_special_tokens=False):
                return [31, 32]

            def decode(self, ids, skip_special_tokens=False):
                return str(ids[0])

        class Wrapper:
            tokenizer = Tokenizer()

            def __init__(self) -> None:
                self.calls = []

            def extract_token_features_batch(self, **kwargs):
                self.calls.append(
                    (
                        list(kwargs["response_token_indices"]),
                        list(kwargs["target_token_ids"]),
                    )
                )
                return [
                    SimpleNamespace(token_id=target)
                    for target in kwargs["target_token_ids"]
                ]

        class Runtime:
            methods = ("svar",)
            official_svar_enabled = True
            requirements = ExtractionRequirements()

            def prepare_official_svar_spans(self, _label, _response_ids):
                return [
                    {"word": "shared", "label": 1, "token_indices": [1]},
                    {"word": "first", "label": 0, "token_indices": [0]},
                ]

            def build_image_records(self, *, image_id, spans, **_kwargs):
                return [
                    {
                        "image_id": image_id,
                        "token_str": span["word"],
                        "response_token_idx": span["token_indices"][0],
                    }
                    for span in spans
                ]

            def build_official_svar_records(
                self, *, image_id, spans, **_kwargs
            ):
                return [
                    {
                        "image_id": image_id,
                        "token_str": span["word"],
                        "response_token_idx": span["token_indices"][0],
                    }
                    for span in spans
                ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (2, 2), color="white").save(image_path)
            (root / "generations.json").write_text(
                json.dumps({"1": {"generated_text": "first shared", "response_token_ids": [31, 32]}}),
                encoding="utf-8",
            )
            root_output = root / "features.pkl"
            baseline_output = root / "baseline" / "features.pkl"
            official_output = (
                root / "baseline" / "svar_official" / "features.pkl"
            )
            # Root ADS/CGC is already complete; this invocation only fills the
            # two baseline protocols from one shared wrapper call.
            save_pkl([{"image_id": 1}], str(root_output))
            wrapper = Wrapper()
            kwargs = {
                "model_wrapper": wrapper,
                "coco_samples": [
                    {"image_id": 1, "image_path": str(image_path)}
                ],
                "labeling_results": {
                    1: {
                        "generated_text": "first shared",
                        "object_token_spans": [
                            {
                                "word": "shared",
                                "label": 1,
                                "token_indices": [1],
                            }
                        ],
                    }
                },
                "cfg_dgst_t": {},
                "output_path": str(root_output),
                "resume": True,
                "cfg_feature_extraction": {
                    "method": {"enabled": False},
                    "ads_cgc": {"enabled": True},
                    "baseline": {"enabled": True},
                },
                "baseline_runtime": Runtime(),
                "baseline_output_path": str(baseline_output),
                "baseline_official_output_path": str(official_output),
            }
            extract_features_for_dataset(**kwargs)
            extract_features_for_dataset(**kwargs)

            self.assertEqual(wrapper.calls, [([1, 0], [32, 31])])
            self.assertEqual(len(load_pkl(str(baseline_output))), 1)
            self.assertEqual(len(load_pkl(str(official_output))), 2)


if __name__ == "__main__":
    unittest.main()
