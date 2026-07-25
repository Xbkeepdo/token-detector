"""Prompt support positions must use merged decoder coordinates."""

from __future__ import annotations

import unittest

from models.llava_wrapper import LLaVAWrapper
from models.prompt_support import resolve_prompt_support_positions


class PromptSupportPositionTests(unittest.TestCase):
    def test_full_prompt_maps_single_image_placeholder_to_merged_positions(self) -> None:
        image_token_id = -200
        input_ids = [1, image_token_id, 2, 3]
        expected = [0, 4, 5]

        shared = resolve_prompt_support_positions(
            tokenizer=None,
            full_input_ids=input_ids,
            prompt_tokenized_length=len(input_ids),
            image_token_id=image_token_id,
            visual_start=1,
            visual_end=4,
            cfg_dgst_t={},
            model_name="test",
        )
        self.assertEqual(shared, expected)

        llava = object.__new__(LLaVAWrapper)
        model_specific = llava._resolve_dgst_prompt_support_positions(
            full_input_ids=input_ids,
            prompt_tokenized_length=len(input_ids),
            image_token_id=image_token_id,
            visual_start=1,
            visual_end=4,
            cfg_dgst_t={},
        )
        self.assertEqual(model_specific, expected)
        self.assertTrue(all(position >= 4 for position in expected[-2:]))


if __name__ == "__main__":
    unittest.main()
