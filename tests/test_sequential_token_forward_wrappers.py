"""InternVL and classic LLaVA use one teacher-forced caption forward."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from models.internvl_wrapper import InternVLWrapper
from models.llava_wrapper import LLaVAWrapper


class FullResponseBatchForwardWrapperTests(TestCase):
    def test_public_batch_api_delegates_once_to_full_response_forward(self) -> None:
        for wrapper_type in (InternVLWrapper, LLaVAWrapper):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = object.__new__(wrapper_type)
                calls = []

                def full_response(**kwargs):
                    calls.append(kwargs)
                    return [SimpleNamespace(response_token_idx=index) for index in (0, 2, 3)]

                wrapper._extract_token_features_batch_full_response = full_response
                marker_requirements = object()
                outputs = wrapper.extract_token_features_batch(
                    image="image",
                    response_token_ids=[10, 11, 12, 13],
                    response_token_indices=[0, 2, 3],
                    target_token_ids=[100, 102, 103],
                    cfg_dgst_t={"target_gate_mode": "four_gate"},
                    prompt="Describe this image.",
                    requirements=marker_requirements,
                )

                self.assertEqual(len(outputs), 3)
                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertEqual(call["response_token_ids"], [10, 11, 12, 13])
                self.assertEqual(call["response_token_indices"], [0, 2, 3])
                self.assertEqual(call["target_token_ids"], [100, 102, 103])
                self.assertIs(call["requirements"], marker_requirements)

    def test_public_batch_api_never_calls_single_target_forward(self) -> None:
        for wrapper_type in (InternVLWrapper, LLaVAWrapper):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = object.__new__(wrapper_type)
                wrapper.extract_token_features = lambda **_kwargs: self.fail(
                    "single-target forward must not run from the batch API"
                )
                wrapper._extract_token_features_batch_full_response = (
                    lambda **_kwargs: []
                )
                self.assertEqual(
                    wrapper.extract_token_features_batch(
                        image="image",
                        response_token_ids=[],
                        response_token_indices=[],
                    ),
                    [],
                )
