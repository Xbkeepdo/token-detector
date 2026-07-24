from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from models.internvl_wrapper import (
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    InternVLWrapper,
)
from models.llava_onevision_wrapper import LLaVAOneVisionWrapper
from models.llava_wrapper import LLaVAWrapper
from models.qwen3_vl_wrapper import Qwen3VLWrapper
from models.qwen_wrapper import QwenVLWrapper


class _InternTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == IMG_START_TOKEN:
            return [101]
        if text == IMG_END_TOKEN:
            return [102]
        if text.startswith("<|im_start|>system"):
            return [11, 12, 13]
        return [21, 22]


class VisualSupportRangeTests(unittest.TestCase):
    def test_llava_single_placeholder_expands_to_visual_only_half_open_range(self) -> None:
        wrapper = SimpleNamespace(num_visual_tokens=576)
        inputs = {"input_ids": torch.tensor([[1, 2, 99, 3, 4]])}
        start, end = LLaVAWrapper._find_visual_token_range(wrapper, inputs, 99)
        self.assertEqual((start, end), (2, 578))
        self.assertEqual(len(range(start, end)), 576)

    def test_dynamic_image_pad_wrappers_exclude_vision_delimiters(self) -> None:
        input_ids = torch.tensor([7, 8, 99, 99, 99, 9, 10])
        for wrapper_type in (
            QwenVLWrapper,
            Qwen3VLWrapper,
            LLaVAOneVisionWrapper,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = SimpleNamespace(
                    _image_token_id=99,
                    _last_num_visual_tokens=None,
                )
                start, end = wrapper_type._find_vision_token_range(
                    wrapper, input_ids
                )
                self.assertEqual((start, end), (2, 5))
                self.assertEqual(list(range(start, end)), [2, 3, 4])
                self.assertEqual(wrapper._last_num_visual_tokens, 3)

    def test_dynamic_image_pad_wrappers_reject_noncontiguous_support(self) -> None:
        input_ids = torch.tensor([7, 99, 8, 99, 10])
        for wrapper_type in (
            QwenVLWrapper,
            Qwen3VLWrapper,
            LLaVAOneVisionWrapper,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = SimpleNamespace(
                    _image_token_id=99,
                    _last_num_visual_tokens=None,
                )
                with self.assertRaises(ValueError):
                    wrapper_type._find_vision_token_range(wrapper, input_ids)

    def test_internvl_support_is_exactly_img_context_tokens(self) -> None:
        wrapper = SimpleNamespace(
            tokenizer=_InternTokenizer(),
            cfg={"num_visual_tokens": 4},
            _img_ctx_id=77,
        )
        pixel_values = torch.zeros((2, 3, 2, 2))
        input_ids, start, end = InternVLWrapper._build_input_ids_with_image(
            wrapper,
            pixel_values,
            prefix_token_ids=[],
            user_prompt="Describe this image.",
        )
        values = input_ids[0].tolist()
        self.assertEqual((start, end), (4, 12))
        self.assertEqual(values[start:end], [77] * 8)
        self.assertEqual(values[start - 1], 101)
        self.assertEqual(values[end], 102)


if __name__ == "__main__":
    unittest.main()
