from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.base_wrapper import (
    AttentionRequirement,
    BaseLVLMWrapper,
    ExtractionRequirements,
    PromptTargetAlignment,
    PromptTargetRequest,
)
from models.prompt_target import (
    extract_prompt_target_from_inputs,
    resolve_prompt_target_alignment,
    truncate_multimodal_inputs,
)


class _ContextTokenizer:
    """Small decoder whose standalone object encoding intentionally differs."""

    all_special_ids = [10]
    is_fast = False
    pieces = {
        10: "",
        11: "Is there a",
        21: " cell",
        22: " phone",
        30: "?",
        31: " Answer yes or no.",
    }

    def encode(self, text, add_special_tokens=False):
        if text == "cell phone":
            return [777]
        raise AssertionError("prompt-target alignment must not re-encode a fragment")

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        values = []
        for token_id in token_ids:
            value = int(token_id)
            if value == 99:
                if not skip_special_tokens:
                    values.append("<image>")
                continue
            values.append(self.pieces.get(value, f"<{value}>"))
        return "".join(values)

    def batch_decode(
        self,
        sequences,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return [
            self.decode(
                values,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for values in sequences
        ]


class PromptTargetAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = _ContextTokenizer()
        self.prompt = "Is there a cell phone? Answer yes or no."
        self.start = self.prompt.index("cell phone")

    def _request(self, expected=21) -> PromptTargetRequest:
        return PromptTargetRequest(
            prompt=self.prompt,
            target_text="cell phone",
            target_char_start=self.start,
            target_char_end=self.start + len("cell phone"),
            expected_target_token_id=expected,
        )

    def test_actual_contextual_id_and_full_multi_token_span(self) -> None:
        # The real sentence uses [21, 22], whereas standalone encode() would
        # return [777].  The API must use first actual contextual subtoken 21.
        alignment = resolve_prompt_target_alignment(
            tokenizer=self.tokenizer,
            full_input_ids=[10, 99, 11, 21, 22, 30, 31],
            request=self._request(),
            image_token_id=99,
            visual_token_count=4,
        )
        self.assertEqual(alignment.target_token_id, 21)
        self.assertEqual(alignment.target_tokenized_position, 3)
        self.assertEqual(alignment.tokenized_span, (3, 4))
        self.assertEqual(alignment.target_expanded_position, 6)
        self.assertEqual(alignment.prediction_position, 5)

    def test_expected_id_is_checked_against_actual_prompt_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "context-tokenized prompt ID"):
            resolve_prompt_target_alignment(
                tokenizer=self.tokenizer,
                full_input_ids=[10, 99, 11, 21, 22, 30, 31],
                request=self._request(expected=777),
                image_token_id=99,
                visual_token_count=4,
            )

    def test_target_immediately_after_image_uses_last_expanded_patch_row(self) -> None:
        request = PromptTargetRequest(
            prompt="cell phone?",
            target_text="cell phone",
            target_char_start=0,
            target_char_end=len("cell phone"),
            expected_target_token_id=21,
        )
        alignment = resolve_prompt_target_alignment(
            tokenizer=self.tokenizer,
            full_input_ids=[10, 99, 21, 22, 30],
            request=request,
            image_token_id=99,
            visual_token_count=4,
        )
        self.assertEqual(alignment.target_tokenized_position, 2)
        self.assertEqual(alignment.target_expanded_position, 5)
        self.assertEqual(alignment.prediction_position, 4)

    def test_whitespace_normalization_keeps_the_exact_repeated_object_span(self) -> None:
        from models.prompt_target import _resolve_rendered_target_span

        prompt = (
            "Some cylinders are beside a cube.\n"
            "Are there any  cylinders behind the cube?"
        )
        target = "cylinders"
        target_start = prompt.rindex(target)
        request = PromptTargetRequest(
            prompt=prompt,
            target_text=target,
            target_char_start=target_start,
            target_char_end=target_start + len(target),
        )
        rendered = (
            "SYSTEM: answer briefly. USER: Some cylinders are beside a cube. "
            "Are there any cylinders behind the cube? ASSISTANT:"
        )
        start, end = _resolve_rendered_target_span(
            rendered_text=rendered,
            request=request,
        )
        self.assertEqual(rendered[start:end], target)
        self.assertEqual(start, rendered.rindex(target))

    def test_exact_span_mapping_refuses_global_first_occurrence_fallback(self) -> None:
        from models.prompt_target import _resolve_rendered_target_span

        prompt = "Is there a cylinder in the image?"
        target = "cylinder"
        target_start = prompt.index(target)
        request = PromptTargetRequest(
            prompt=prompt,
            target_text=target,
            target_char_start=target_start,
            target_char_end=target_start + len(target),
        )
        with self.assertRaisesRegex(ValueError, "refusing first-occurrence fallback"):
            _resolve_rendered_target_span(
                rendered_text="SYSTEM: cylinder. USER: unrelated text.",
                request=request,
            )

    def test_old_generated_response_validator_remains_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "actual saved response token"):
            BaseLVLMWrapper.validate_causal_batch_request(
                response_token_ids=[1, 2],
                response_token_indices=[1],
                target_token_ids=[21],
            )

    def test_truncation_excludes_target_and_future_prompt_tokens(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[10, 99, 11, 21, 22, 30]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "pixel_values": torch.ones((1, 3, 2, 2)),
            "position_ids": torch.arange(6).unsqueeze(0),
        }
        prefix = truncate_multimodal_inputs(inputs, tokenized_end=3)
        self.assertEqual(prefix["input_ids"].tolist(), [[10, 99, 11]])
        self.assertEqual(prefix["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(tuple(prefix["pixel_values"].shape), (1, 3, 2, 2))
        self.assertNotIn("position_ids", prefix)


class PromptTargetForwardTests(unittest.TestCase):
    def test_attention_hidden_logits_and_dgst_share_prediction_row(self) -> None:
        tokenizer = _ContextTokenizer()
        alignment = PromptTargetAlignment(
            target_tokenized_position=3,
            target_expanded_position=6,
            prediction_position=5,
            target_token_id=21,
            tokenized_span=(3, 4),
            rendered_char_start=11,
            rendered_char_end=21,
        )
        sequence_length = 6
        vocabulary = 9
        logits = torch.zeros((1, sequence_length, vocabulary))
        logits[0, 5, 7] = 10.0
        attention = torch.zeros((1, 2, sequence_length, sequence_length))
        for query in range(sequence_length):
            for key in range(sequence_length):
                attention[0, :, query, key] = query * 100 + key
        out = SimpleNamespace(
            logits=logits,
            attentions=(attention,),
            hidden_states=None,
        )
        seen = {}

        def fake_forward(_model, **kwargs):
            seen["prefix_ids"] = kwargs["input_ids"].tolist()
            return out, [object()]

        def fake_hidden(_captures, *, token_position, visual_start, visual_end):
            seen["hidden_row"] = int(token_position)
            return (
                torch.full((1, 3), float(token_position)),
                torch.full((1, visual_end - visual_start, 3), float(token_position)),
            )

        def fake_dgst(**kwargs):
            seen["dgst_rows"] = list(kwargs["prediction_positions"])
            seen["dgst_targets"] = list(kwargs["target_token_ids"])
            return [{"row": int(kwargs["prediction_positions"][0])}]

        class Wrapper:
            model = object()
            cfg = {}

            @staticmethod
            def resolve_extraction_requirements(requirements, *, dgst_enabled):
                return requirements

        Wrapper.tokenizer = tokenizer
        requirements = ExtractionRequirements(
            attention=AttentionRequirement.PER_HEAD,
            logits=True,
            token_hidden_states=True,
            patch_hidden_states=True,
            visual_layout=True,
            dgst_capture=True,
        )
        with patch(
            "models.prompt_target.run_forward_with_dgst_captures",
            side_effect=fake_forward,
        ), patch(
            "models.prompt_target.hidden_states_from_captures",
            side_effect=fake_hidden,
        ), patch(
            "models.prompt_target.compute_dgst_t_batch_from_captures",
            side_effect=fake_dgst,
        ):
            result = extract_prompt_target_from_inputs(
                wrapper=Wrapper(),
                inputs={
                    "input_ids": torch.tensor([[10, 99, 11, 21, 22, 30]]),
                    "attention_mask": torch.ones((1, 6), dtype=torch.long),
                },
                full_input_ids=[10, 99, 11, 21, 22, 30],
                alignment=alignment,
                visual_start=1,
                visual_end=5,
                image_token_id=99,
                visual_grid=(2, 2),
                cfg_dgst_t={"target_gate_mode": "four_gate"},
                requirements=requirements,
                model_name="mock",
                support_scope="visual",
            )

        self.assertEqual(seen["prefix_ids"], [[10, 99, 11]])
        self.assertEqual(seen["hidden_row"], 5)
        self.assertEqual(seen["dgst_rows"], [5])
        self.assertEqual(seen["dgst_targets"], [21])
        self.assertEqual(result.token_id, 7)
        self.assertEqual(int(result.token_logits.argmax().item()), 7)
        self.assertTrue(torch.equal(result.token_hidden_states, torch.full((1, 3), 5.0)))
        self.assertEqual(result.dgst_t_result, {"row": 5})
        self.assertTrue(
            torch.equal(
                result.text_to_patch_attn[0, 0],
                torch.tensor([501.0, 502.0, 503.0, 504.0]),
            )
        )
        metadata = result.baseline_capture["prompt_target_alignment"]
        self.assertEqual(metadata["prediction_position"], 5)
        self.assertEqual(metadata["target_token_id"], 21)
        self.assertEqual(metadata["tokenized_span"], [3, 4])

    def test_all_six_wrappers_implement_prompt_target_api(self) -> None:
        from models.internvl_wrapper import InternVLWrapper
        from models.llava_next_wrapper import LLaVANextWrapper
        from models.llava_onevision_wrapper import LLaVAOneVisionWrapper
        from models.llava_wrapper import LLaVAWrapper
        from models.qwen3_vl_wrapper import Qwen3VLWrapper
        from models.qwen_wrapper import QwenVLWrapper

        for wrapper_type in (
            LLaVAWrapper,
            LLaVANextWrapper,
            InternVLWrapper,
            QwenVLWrapper,
            Qwen3VLWrapper,
            LLaVAOneVisionWrapper,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                self.assertTrue(callable(wrapper_type.extract_prompt_target_features))


if __name__ == "__main__":
    unittest.main()
