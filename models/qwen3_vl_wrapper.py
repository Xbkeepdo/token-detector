"""Qwen3-VL wrapper for COCO caption generation."""

from __future__ import annotations

from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput


class Qwen3VLWrapper(BaseLVLMWrapper):
    """Caption-generation wrapper for Qwen3-VL-8B-Instruct."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[Qwen3VLWrapper] Loading model from {hf_name} ...")
        self.processor = AutoProcessor.from_pretrained(
            hf_name,
            trust_remote_code=True,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_name,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print("[Qwen3VLWrapper] Loaded.")

    @property
    def num_layers(self) -> int:
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None:
            return int(text_config.num_hidden_layers)
        return int(self.cfg.get("num_layers", 36))

    @property
    def num_visual_tokens(self) -> int:
        return int(self.cfg.get("num_visual_tokens") or 256)

    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        prompt = prompt or "Describe this image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self.generation_max_new_tokens,
            )

        response_ids = output_ids[0, prompt_len:].tolist()
        generated_text = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        response_tokens = [
            self.tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in response_ids
        ]
        return GenerationOutput(
            image_id=-1,
            generated_text=generated_text,
            response_token_ids=response_ids,
            response_tokens=response_tokens,
        )

    def extract_token_features(
        self,
        image: Image.Image,
        prefix_token_ids: List[int],
        response_token_idx: int,
        target_token_id: Optional[int] = None,
        cfg_dgst_t: Optional[dict] = None,
    ) -> ModelOutput:
        raise NotImplementedError(
            "Qwen3VLWrapper currently supports caption generation only; "
            "Qwen3-VL feature extraction has not been implemented."
        )
