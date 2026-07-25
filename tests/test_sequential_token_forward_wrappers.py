"""InternVL and LLaVA must match Qwen's per-target prefix protocol."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from models.base_wrapper import ExtractionRequirements
from models.internvl_wrapper import InternVLWrapper
from models.llava_wrapper import LLaVAWrapper


class SequentialTokenForwardWrapperTests(TestCase):
    def _wrapper(self, wrapper_type):
        wrapper = object.__new__(wrapper_type)
        wrapper.validate_causal_batch_request = lambda **_kwargs: (
            [10, 11, 12, 13],
            [0, 2, 3],
            [100, 102, 103],
        )
        return wrapper

    def test_each_target_uses_its_own_causal_prefix(self) -> None:
        for wrapper_type in (InternVLWrapper, LLaVAWrapper):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = self._wrapper(wrapper_type)
                calls = []

                def extract_token_features(**kwargs):
                    calls.append(kwargs)
                    return SimpleNamespace(
                        response_hidden_states=None,
                        baseline_capture={"target": kwargs["target_token_id"]},
                    )

                wrapper.extract_token_features = extract_token_features
                with patch("torch.cuda.is_available", return_value=False):
                    outputs = wrapper.extract_token_features_batch(
                        image="image",
                        response_token_ids=[10, 11, 12, 13],
                        response_token_indices=[0, 2, 3],
                        target_token_ids=[100, 102, 103],
                        cfg_dgst_t={"target_gate_mode": "four_gate"},
                        prompt="Describe this image.",
                        requirements=ExtractionRequirements(
                            response_hidden_states=False
                        ),
                    )

                self.assertEqual(len(outputs), 3)
                self.assertEqual(
                    [call["prefix_token_ids"] for call in calls],
                    [[], [10, 11], [10, 11, 12]],
                )
                self.assertEqual(
                    [call["response_token_idx"] for call in calls],
                    [0, 2, 3],
                )
                self.assertEqual(
                    [call["target_token_id"] for call in calls],
                    [100, 102, 103],
                )

    def test_full_response_baseline_capture_is_shared_not_recomputed_per_target(
        self,
    ) -> None:
        for wrapper_type in (InternVLWrapper, LLaVAWrapper):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = self._wrapper(wrapper_type)
                seen_requirements = []
                shared_calls = []

                def extract_token_features(**kwargs):
                    seen_requirements.append(kwargs["requirements"])
                    return SimpleNamespace(
                        response_hidden_states=None,
                        baseline_capture={"per_target": kwargs["target_token_id"]},
                    )

                def full_response_capture(**kwargs):
                    shared_calls.append(kwargs)
                    return {
                        "response_hidden_states": "shared-hidden",
                        "statistics": {"shared_stat": 7},
                    }

                wrapper.extract_token_features = extract_token_features
                wrapper._extract_full_response_baseline_capture = full_response_capture
                requirements = ExtractionRequirements(response_hidden_states=True)
                with patch("torch.cuda.is_available", return_value=False):
                    outputs = wrapper.extract_token_features_batch(
                        image="image",
                        response_token_ids=[10, 11, 12, 13],
                        response_token_indices=[0, 2, 3],
                        target_token_ids=[100, 102, 103],
                        prompt="Describe this image.",
                        requirements=requirements,
                    )

                self.assertTrue(
                    all(not item.response_hidden_states for item in seen_requirements)
                )
                self.assertEqual(len(shared_calls), 1)
                self.assertEqual(shared_calls[0]["response_token_ids"], [10, 11, 12, 13])
                for output, target in zip(outputs, [100, 102, 103]):
                    self.assertEqual(output.response_hidden_states, "shared-hidden")
                    self.assertEqual(
                        output.baseline_capture,
                        {"per_target": target, "shared_stat": 7},
                    )
