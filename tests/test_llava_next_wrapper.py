import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models import build_model
from models.llava_next_wrapper import (
    LLAVA_LLAMA3_SYSTEM_PROMPT,
    LLaVANextWrapper,
    _convert_original_llava_config,
    _is_original_llava_checkpoint,
    _validate_local_checkpoint,
)


class _FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered-llama3-prompt"


class LLaVANextWrapperTests(unittest.TestCase):
    def test_detect_and_convert_original_llava_checkpoint(self):
        raw = {
            "architectures": ["LlavaLlamaForCausalLM"],
            "model_type": "llava",
            "mm_projector_type": "mlp2x_gelu",
            "mm_hidden_size": 1024,
            "mm_vision_select_layer": -2,
            "image_token_index": 128256,
            "vocab_size": 128256,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 8192,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "image_grid_pinpoints": [[336, 672], [672, 336]],
        }
        self.assertTrue(_is_original_llava_checkpoint(raw))
        config, tokenizer_image_id, model_image_id = (
            _convert_original_llava_config(raw)
        )
        self.assertEqual(config.model_type, "llava_next")
        self.assertEqual(config.text_config.model_type, "llama")
        self.assertEqual(config.text_config.num_hidden_layers, 32)
        self.assertEqual(config.vision_config.hidden_size, 1024)
        self.assertEqual(config.vision_config.image_size, 336)
        self.assertEqual(config.image_grid_pinpoints, raw["image_grid_pinpoints"])
        self.assertEqual(tokenizer_image_id, 128256)
        self.assertEqual(model_image_id, 128255)
        self.assertEqual(config.image_token_index, 128255)

    def test_llama3_prompt_uses_system_user_and_one_image(self):
        wrapper = object.__new__(LLaVANextWrapper)
        wrapper.cfg = {}
        wrapper.tokenizer = _FakeTokenizer()
        rendered = wrapper._format_prompt("Describe this image.")
        self.assertEqual(rendered, "rendered-llama3-prompt")
        self.assertEqual(
            wrapper.tokenizer.messages,
            [
                {"role": "system", "content": LLAVA_LLAMA3_SYSTEM_PROMPT},
                {"role": "user", "content": "<image>\nDescribe this image."},
            ],
        )
        self.assertEqual(
            wrapper.tokenizer.kwargs,
            {"tokenize": False, "add_generation_prompt": True},
        )

    def test_dynamic_visual_span_is_contiguous_and_updates_count(self):
        wrapper = object.__new__(LLaVANextWrapper)
        wrapper.cfg = {"num_visual_tokens": None}
        wrapper._last_num_visual_tokens = None
        inputs = {"input_ids": torch.tensor([[1, 9, 9, 9, 9, 2]])}
        self.assertEqual(wrapper._find_visual_token_range(inputs, 9), (1, 5))
        self.assertEqual(wrapper.num_visual_tokens, 4)
        self.assertIsNone(wrapper._visual_grid_for_output(inputs, 1, 5))

        with self.assertRaisesRegex(ValueError, "one contiguous image span"):
            wrapper._find_visual_token_range(
                {"input_ids": torch.tensor([[1, 9, 2, 9, 3]])},
                9,
            )

    def test_configures_extraction_anyres_grid_on_processor_and_model(self):
        wrapper = object.__new__(LLaVANextWrapper)
        wrapper.cfg = {"image_grid_pinpoints": [[336, 336]]}
        wrapper.processor = SimpleNamespace(
            image_processor=SimpleNamespace(image_grid_pinpoints=[[672, 672]])
        )
        wrapper.model = SimpleNamespace(
            config=SimpleNamespace(image_grid_pinpoints=[[672, 672]]),
            model=SimpleNamespace(
                config=SimpleNamespace(image_grid_pinpoints=[[672, 672]])
            ),
        )
        wrapper._configure_anyres_grid_pinpoints()
        expected = [[336, 336]]
        self.assertEqual(wrapper.processor.image_processor.image_grid_pinpoints, expected)
        self.assertEqual(wrapper.model.config.image_grid_pinpoints, expected)
        self.assertEqual(wrapper.model.model.config.image_grid_pinpoints, expected)

    def test_incomplete_local_checkpoint_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            index = {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index),
                encoding="utf-8",
            )
            (model_dir / "model-00002-of-00002.safetensors").touch()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "model-00001-of-00002.safetensors",
            ):
                _validate_local_checkpoint(str(model_dir))

    def test_factory_registers_both_llava_next_names(self):
        # Patch loading so the registry can be checked without allocating 8B.
        original = LLaVANextWrapper._load_model
        try:
            LLaVANextWrapper._load_model = lambda self: None
            for key in ("llava_next_8b", "llava_next_llama3_8b"):
                wrapper = build_model(key, {"hf_name": "unused"}, device="cpu")
                self.assertIsInstance(wrapper, LLaVANextWrapper)
        finally:
            LLaVANextWrapper._load_model = original


if __name__ == "__main__":
    unittest.main()
