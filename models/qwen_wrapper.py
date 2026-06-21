"""Qwen2.5-VL wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput
from models.dgst_capture import (
    build_dgst_t_raw,
    build_dgst_t_raw_batch,
    hidden_states_from_captures,
    pre_token_prediction_positions,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
)
from features.dgst_t import compute_dgst_t_batch_from_captures


class QwenVLWrapper(BaseLVLMWrapper):
    """Wrapper for Qwen2.5-VL-7B-Instruct."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[QwenVLWrapper] Loading model from {hf_name} …")
        self.processor = AutoProcessor.from_pretrained(
            hf_name, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        self._vision_start_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_start_token", "<|vision_start|>")
        )
        self._vision_end_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_end_token", "<|vision_end|>")
        )
        print(
            f"[QwenVLWrapper] Loaded. "
            f"vision_start={self._vision_start_id}, "
            f"vision_end={self._vision_end_id}"
        )

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.get("num_visual_tokens") or 256


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        if prompt is None:
            prompt = "Describe this image."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=256,
            )

        response_ids = output_ids[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response_tokens = [
            self.tokenizer.decode([tid], skip_special_tokens=False)
            for tid in response_ids
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
    ) -> ModelOutput:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])
        partial_response = self.tokenizer.decode(
            prefix_token_ids, skip_special_tokens=True
        )
        full_text = text + partial_response

        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        input_ids = inputs["input_ids"][0]
        img_start, img_end = self._find_vision_token_range(input_ids)

        out, captures = run_forward_with_dgst_captures(self.model, **inputs)

        expanded_seq_len = out.attentions[0].shape[-1]
        text_to_patch_attn, text_to_text_attn = _extract_attention_features(
            out.attentions, img_start, img_end, expanded_seq_len
        )
        token_hidden_states, patch_hidden_states = _extract_hidden_states(
            out.hidden_states, img_start, img_end
        )

        pred_token_id = out.logits[0, -1].argmax().item()
        pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
        last_logits = out.logits[0, -1].float().cpu()
        dgst_target_id = int(target_token_id) if target_token_id is not None else int(pred_token_id)
        dgst_t_raw = build_dgst_t_raw(
            model=self.model,
            full_input_ids=input_ids.tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            captures=captures,
            visual_start=img_start,
            visual_end=img_end,
            image_token_id=int(self.model.config.image_token_id),
            target_token_id=dgst_target_id,
            prediction_position=expanded_seq_len - 1,
            support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
        )

        return ModelOutput(
            token_id=pred_token_id,
            token_str=pred_token_str,
            text_to_patch_attn=text_to_patch_attn.cpu(),
            text_to_text_attn=text_to_text_attn.cpu(),
            token_hidden_states=token_hidden_states.cpu(),
            patch_hidden_states=patch_hidden_states.cpu(),
            response_token_idx=response_token_idx,
            token_logits=last_logits,
            dgst_t_raw=dgst_t_raw,
        )

    def extract_token_features_batch(
        self,
        image: Image.Image,
        response_token_ids: Sequence[int],
        response_token_indices: Sequence[int],
        target_token_ids: Optional[Sequence[int]] = None,
        cfg_dgst_t: Optional[dict] = None,
    ) -> List[ModelOutput]:
        requested_indices = [int(index) for index in response_token_indices]
        if not requested_indices:
            return []
        response_ids = [int(token_id) for token_id in response_token_ids]
        targets = (
            [int(token_id) for token_id in target_token_ids]
            if target_token_ids is not None
            else [response_ids[index] for index in requested_indices]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prefix_inputs["input_ids"].shape[1])
        answer_ids = torch.tensor(
            response_ids,
            dtype=prefix_inputs["input_ids"].dtype,
        ).unsqueeze(0)
        full_inputs = dict(prefix_inputs)
        full_inputs["input_ids"] = torch.cat([prefix_inputs["input_ids"], answer_ids], dim=1)
        if "attention_mask" in prefix_inputs:
            answer_mask = torch.ones_like(answer_ids)
            full_inputs["attention_mask"] = torch.cat([prefix_inputs["attention_mask"], answer_mask], dim=1)
        full_inputs = _to_device_dtype(full_inputs, self.device)

        input_ids = full_inputs["input_ids"][0]
        img_start, img_end = self._find_vision_token_range(input_ids)

        out, captures = run_forward_with_dgst_captures(
            self.model,
            output_hidden_states=False,
            **full_inputs,
        )
        if out.attentions is None or len(out.attentions) == 0 or out.attentions[0] is None:
            raise RuntimeError("DGST-T batch extraction requires attention weights; use eager attention.")

        expanded_seq_len = int(out.attentions[0].shape[-1])
        image_token_id = int(self.model.config.image_token_id)
        prompt_positions = resolve_prompt_positions(
            full_input_ids=input_ids.tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            image_token_id=image_token_id,
            visual_start=img_start,
            visual_end=img_end,
        )
        prediction_positions = pre_token_prediction_positions(
            full_input_ids=input_ids.tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            response_token_indices=requested_indices,
            image_token_id=image_token_id,
            visual_token_count=int(img_end - img_start),
            prompt_positions=prompt_positions,
        )
        dgst_results = None
        dgst_raws = None
        if cfg_dgst_t is not None:
            dgst_results = compute_dgst_t_batch_from_captures(
                model=self.model,
                full_input_ids=input_ids.tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                captures=captures,
                visual_start=img_start,
                visual_end=img_end,
                image_token_id=image_token_id,
                target_token_ids=targets,
                prediction_positions=prediction_positions,
                support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
                tau=cfg_dgst_t.get("tau", 0.07),
                transport_top_k=cfg_dgst_t.get("transport_top_k", 64),
                cost_mode=cfg_dgst_t.get("cost_mode", "direct"),
                lambda_d=cfg_dgst_t.get("lambda_d", 1.0),
                lambda_s=cfg_dgst_t.get("lambda_s", 1.0),
                lambda_t=cfg_dgst_t.get("lambda_t", 1.0),
                lambda_int=cfg_dgst_t.get("lambda_int", 1.0),
                baseline_layers=cfg_dgst_t.get("baseline_layers", 10),
                risk_start_layer=cfg_dgst_t.get("risk_start_layer", 15),
                alpha=cfg_dgst_t.get("alpha", 2.0),
                ot_solver=cfg_dgst_t.get("ot_solver", "linprog"),
                atarget_visual_top_k=cfg_dgst_t.get("atarget_visual_top_k", 32),
            )
        else:
            dgst_raws = build_dgst_t_raw_batch(
                model=self.model,
                full_input_ids=input_ids.tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                captures=captures,
                visual_start=img_start,
                visual_end=img_end,
                image_token_id=image_token_id,
                target_token_ids=targets,
                prediction_positions=prediction_positions,
                support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
            )

        outputs: List[ModelOutput] = []
        for offset, (response_index, prediction_position) in enumerate(zip(
            requested_indices,
            prediction_positions,
        )):
            text_to_patch_attn, text_to_text_attn = _extract_attention_features_at_position(
                out.attentions,
                img_start,
                img_end,
                expanded_seq_len,
                int(prediction_position),
            )
            token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                captures,
                token_position=int(prediction_position),
                visual_start=img_start,
                visual_end=img_end,
            )
            logits = out.logits[0, int(prediction_position)].float().cpu()
            pred_token_id = int(logits.argmax().item())
            pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
            outputs.append(
                ModelOutput(
                    token_id=pred_token_id,
                    token_str=pred_token_str,
                    text_to_patch_attn=text_to_patch_attn.cpu(),
                    text_to_text_attn=text_to_text_attn.cpu(),
                    token_hidden_states=token_hidden_states.cpu(),
                    patch_hidden_states=patch_hidden_states.cpu(),
                    response_token_idx=int(response_index),
                    token_logits=logits,
                    dgst_t_raw=dgst_raws[offset] if dgst_raws is not None else None,
                    dgst_t_result=dgst_results[offset] if dgst_results is not None else None,
                )
            )
        return outputs


    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Locate <|vision_start|> and <|vision_end|> in input_ids and"""
        ids = input_ids.tolist()
        try:
            vs_pos = ids.index(self._vision_start_id)
            ve_pos = ids.index(self._vision_end_id)
            return vs_pos + 1, ve_pos
        except ValueError:
            return 5, 5 + 256


def _extract_text_to_patch_attn(
    attentions: tuple, img_start: int, img_end: int
) -> torch.Tensor:
    layers = []
    for attn in attentions:
        patch_attn = attn[0, :, -1, img_start:img_end]
        layers.append(patch_attn)
    return torch.stack(layers, dim=0)


def _extract_hidden_states(
    hidden_states: tuple, img_start: int, img_end: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, -1, :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)


def _extract_attention_features(attentions, img_start, img_end, seq_len):
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [i for i in range(seq_len) if i not in visual_set and i != last_pos]
    text_idx_tensor = __import__('torch').tensor(text_indices, dtype=__import__('torch').long)
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])
    import torch
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _extract_attention_features(attentions, img_start, img_end, seq_len):
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [i for i in range(seq_len) if i not in visual_set and i != last_pos]
    text_idx_tensor = __import__('torch').tensor(text_indices, dtype=__import__('torch').long)
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])
    import torch
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _extract_attention_features_at_position(attentions, img_start, img_end, seq_len, token_position):
    visual_set = set(range(img_start, img_end))
    last_pos = int(token_position)
    text_indices = [i for i in range(seq_len) if i not in visual_set and i != last_pos]
    text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        idx = text_idx_tensor.to(layer_attn.device)
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, idx])
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _to_device_dtype(inputs: dict, device: str) -> dict:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.to(device=device)
        else:
            result[key] = value
    return result
