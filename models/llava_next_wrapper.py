"""LLaVA-NeXT Llama-3 8B wrapper with AnyRes visual-token support.

The public ``lmms-lab/llama3-llava-next-8b`` checkpoint uses the original
LLaVA-NeXT flat config and state-dict layout rather than the nested Hugging
Face ``LlavaNextConfig`` layout.  This module converts the config and weight
names at load time; the checkpoint on disk remains untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    CLIPVisionConfig,
    LlamaConfig,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
)

from models.llava_wrapper import LLaVAWrapper


LLAVA_LLAMA3_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. You are able to "
    "understand the visual content that the user provides, and assist the "
    "user with a variety of tasks using natural language."
)


_LEGACY_WEIGHT_KEY_MAPPING = {
    r"^model\.embed_tokens\.": "model.language_model.embed_tokens.",
    r"^model\.layers\.": "model.language_model.layers.",
    r"^model\.norm\.": "model.language_model.norm.",
    r"^model\.vision_tower\.vision_tower\.": "model.vision_tower.",
    r"^model\.mm_projector\.0\.": "model.multi_modal_projector.linear_1.",
    r"^model\.mm_projector\.2\.": "model.multi_modal_projector.linear_2.",
}


class _RemappedImageTokenLlavaNextProcessor(LlavaNextProcessor):
    """Map the original out-of-vocabulary image ID to a reserved in-vocab ID.

    The released Llama-3 checkpoint has 128256 embedding rows but assigns
    ``<image>`` ID 128256.  Original LLaVA replaces that placeholder before
    embedding lookup, while native Transformers embeds IDs before masked
    scatter.  Mapping only the placeholder to reserved ID 128255 is therefore
    numerically neutral and avoids an out-of-range embedding lookup.
    """

    model_image_token_id: int
    tokenizer_image_token_id: int

    def __call__(self, *args, **kwargs):
        values = super().__call__(*args, **kwargs)
        input_ids = values.get("input_ids")
        source_id = int(self.tokenizer_image_token_id)
        target_id = int(self.model_image_token_id)
        if source_id == target_id or input_ids is None:
            return values
        if torch.is_tensor(input_ids):
            values["input_ids"] = input_ids.masked_fill(
                input_ids == source_id,
                target_id,
            )
        else:
            values["input_ids"] = [
                [target_id if int(token_id) == source_id else int(token_id) for token_id in row]
                for row in input_ids
            ]
        return values


class LLaVANextWrapper(LLaVAWrapper):
    """Wrapper for ``lmms-lab/llama3-llava-next-8b``.

    Generation, batch extraction, prompt-target extraction and every DGST /
    baseline capture path are inherited from :class:`LLaVAWrapper`.  The
    overrides here supply the Llama-3 chat template and dynamic AnyRes visual
    span instead of LLaVA-1.5's fixed 24x24 patch grid.
    """

    def _load_model(self) -> None:
        hf_name = str(self.cfg["hf_name"])
        _validate_local_checkpoint(hf_name)
        print(f"[LLaVANextWrapper] Loading model from {hf_name} ...")

        raw_config = _read_raw_config(hf_name)
        legacy_checkpoint = _is_original_llava_checkpoint(raw_config)
        if legacy_checkpoint:
            config, tokenizer_image_id, model_image_id = (
                _convert_original_llava_config(raw_config)
            )
            self.processor = _build_original_checkpoint_processor(
                hf_name,
                raw_config=raw_config,
                tokenizer_image_id=tokenizer_image_id,
                model_image_id=model_image_id,
            )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                hf_name,
                config=config,
                key_mapping=_LEGACY_WEIGHT_KEY_MAPPING,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        else:
            self.processor = LlavaNextProcessor.from_pretrained(hf_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                hf_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            if self.processor.patch_size is None:
                self.processor.patch_size = int(
                    self.model.config.vision_config.patch_size
                )
            if self.processor.vision_feature_select_strategy is None:
                self.processor.vision_feature_select_strategy = str(
                    self.model.config.vision_feature_select_strategy
                )

        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self._last_num_visual_tokens: Optional[int] = None
        self._configure_anyres_grid_pinpoints()
        print(
            "[LLaVANextWrapper] Model loaded with eager attention and "
            "dynamic AnyRes image tokens."
        )

    @property
    def model_label(self) -> str:
        return "LLaVA-NeXT-8B"

    @property
    def dgst_capture_device(self) -> Optional[str]:
        # AnyRes sequences are long enough that retaining three full residual
        # stacks on a 24-GiB card leaves no room for eager attention's
        # transient FP32 softmax.  CPU capture preserves the exact tensors and
        # keeps GPU usage bounded by the current decoder layer.
        value = self.cfg.get("dgst_capture_device", "cpu")
        return str(value) if value is not None else None

    @property
    def dgst_attention_query_chunk_size(self) -> Optional[int]:
        value = self.cfg.get("dgst_attention_query_chunk_size", 512)
        return int(value) if value is not None else None

    def _configure_anyres_grid_pinpoints(self) -> None:
        configured = self.cfg.get("image_grid_pinpoints")
        if configured is None:
            return
        if not isinstance(configured, (list, tuple)) or not configured:
            raise ValueError("image_grid_pinpoints must be a non-empty list.")
        points: list[list[int]] = []
        for value in configured:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(
                    "Each image_grid_pinpoints entry must be [height, width]."
                )
            height, width = (int(value[0]), int(value[1]))
            if height <= 0 or width <= 0 or height % 336 or width % 336:
                raise ValueError(
                    "LLaVA-NeXT image_grid_pinpoints dimensions must be "
                    "positive multiples of 336."
                )
            points.append([height, width])

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("LLaVA-NeXT processor has no image_processor.")
        image_processor.image_grid_pinpoints = points
        for model_part in (self.model, getattr(self.model, "model", None)):
            config = getattr(model_part, "config", None)
            if config is not None:
                config.image_grid_pinpoints = points
        print(f"[LLaVANextWrapper] AnyRes grid pinpoints limited to {points}.")

    @property
    def num_visual_tokens(self) -> int:
        configured = self.cfg.get("num_visual_tokens")
        if configured is not None:
            return int(configured)
        if self._last_num_visual_tokens is not None:
            return int(self._last_num_visual_tokens)
        raise RuntimeError(
            "LLaVA-NeXT uses a dynamic AnyRes visual-token count. Process an "
            "image first or read the visual span from ModelOutput.baseline_capture."
        )

    def _format_prompt(self, raw_prompt: str) -> str:
        prompt = str(raw_prompt).strip()
        if "<|start_header_id|>" in prompt and "<image>" in prompt:
            return prompt
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        messages = [
            {
                "role": "system",
                "content": str(
                    self.cfg.get(
                        "system_prompt",
                        LLAVA_LLAMA3_SYSTEM_PROMPT,
                    )
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _find_visual_token_range(
        self,
        inputs: dict[str, Any],
        image_token_id: int,
    ) -> Tuple[int, int]:
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("LLaVA-NeXT inputs require rank-2 input_ids")
        positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=True)[0]
        count = int(positions.numel())
        if count == 0:
            raise ValueError(
                "LLaVA-NeXT prompt contains no image tokens after processing; "
                "ensure the rendered prompt contains exactly one <image>."
            )
        if count == 1:
            raise RuntimeError(
                "LLaVA-NeXT processor left one legacy <image> placeholder. "
                "Dynamic AnyRes extraction requires processor-expanded image tokens."
            )
        start = int(positions[0].item())
        expected = torch.arange(
            start,
            start + count,
            dtype=positions.dtype,
            device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise ValueError(
                "LLaVA-NeXT wrapper supports exactly one contiguous image span; "
                f"found positions={positions.detach().cpu().tolist()[:16]}"
            )
        self._last_num_visual_tokens = count
        return start, start + count

    def _visual_grid_for_output(
        self,
        inputs: dict[str, Any],
        visual_start: int,
        visual_end: int,
    ) -> None:
        # AnyRes concatenates a 24x24 base view, unpadded local views and row
        # newline embeddings.  That sequence cannot be represented faithfully
        # by one rectangular (height, width) grid.
        del inputs, visual_start, visual_end
        return None


def _read_raw_config(hf_name: str) -> dict[str, Any]:
    config_path = Path(hf_name) / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _is_original_llava_checkpoint(config: dict[str, Any]) -> bool:
    architectures = {str(value) for value in config.get("architectures", [])}
    return bool(
        "LlavaLlamaForCausalLM" in architectures
        or (
            config.get("model_type") == "llava"
            and "text_config" not in config
            and "mm_projector_type" in config
        )
    )


def _convert_original_llava_config(
    raw: dict[str, Any],
) -> tuple[LlavaNextConfig, int, int]:
    text_keys = {
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pretraining_tp",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    }
    text_values = {key: raw[key] for key in text_keys if key in raw}
    text_values["vocab_size"] = int(raw.get("vocab_size", 128256))
    text_config = LlamaConfig(**text_values)
    vision_config = CLIPVisionConfig(
        hidden_size=int(raw.get("mm_hidden_size", 1024)),
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=int(raw.get("image_size", 336)),
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
    )

    tokenizer_image_id = int(raw.get("image_token_index", text_config.vocab_size))
    model_image_id = tokenizer_image_id
    if model_image_id >= int(text_config.vocab_size):
        model_image_id = int(text_config.vocab_size) - 1
    if model_image_id < 0:
        raise ValueError("LLaVA-NeXT text vocabulary must be non-empty")

    config = LlavaNextConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=model_image_id,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=int(raw.get("mm_vision_select_layer", -2)),
        image_grid_pinpoints=raw.get("image_grid_pinpoints"),
        image_seq_length=576,
        multimodal_projector_bias=True,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        bos_token_id=raw.get("bos_token_id"),
        eos_token_id=raw.get("eos_token_id"),
        torch_dtype=torch.float16,
    )
    return config, tokenizer_image_id, model_image_id


def _build_original_checkpoint_processor(
    hf_name: str,
    *,
    raw_config: dict[str, Any],
    tokenizer_image_id: int,
    model_image_id: int,
) -> _RemappedImageTokenLlavaNextProcessor:
    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
    image_processor = LlavaNextImageProcessor.from_pretrained(
        hf_name,
        image_grid_pinpoints=raw_config.get("image_grid_pinpoints"),
    )
    processor = _RemappedImageTokenLlavaNextProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=14,
        vision_feature_select_strategy="default",
        chat_template=getattr(tokenizer, "chat_template", None),
        image_token="<image>",
        num_additional_image_tokens=1,
    )
    processor.tokenizer_image_token_id = int(tokenizer_image_id)
    processor.model_image_token_id = int(model_image_id)
    return processor


def _validate_local_checkpoint(hf_name: str) -> None:
    """Fail before allocating an 8B model when a local shard is incomplete."""

    model_dir = Path(hf_name)
    if not model_dir.is_dir():
        return
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = sorted({str(value) for value in index.get("weight_map", {}).values()})
    missing = [name for name in expected if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete LLaVA-NeXT checkpoint; resume the Hugging Face download. "
            "Missing weight shard(s): " + ", ".join(missing)
        )


__all__ = ["LLaVANextWrapper"]
