"""LLaVA-1.5 wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

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

IMAGE_TOKEN_INDEX = -200
NUM_VISUAL_TOKENS = 576


class LLaVAWrapper(BaseLVLMWrapper):
    """Wrapper for LLaVA-1.5-7B (HF transformers implementation)."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[LLaVAWrapper] Loading model from {hf_name} …")
        self.processor = AutoProcessor.from_pretrained(hf_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print("[LLaVAWrapper] Model loaded.")


    @property
    def num_layers(self) -> int:
        return self.model.language_model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return NUM_VISUAL_TOKENS


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        if prompt is None:
            prompt = self.cfg["prompt_template"]

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=256,
            )

        prompt_len = inputs["input_ids"].shape[1]
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
        """Runs a forward pass with prompt+partial_response and returns"""
        prompt_text = self.cfg["prompt_template"]
        partial_text = self.tokenizer.decode(prefix_token_ids, skip_special_tokens=True)
        full_prompt = prompt_text + partial_text

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])

        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        input_ids = inputs["input_ids"]

        image_token_id = int(getattr(self.model.config, "image_token_index", IMAGE_TOKEN_INDEX))
        img_placeholder_mask = (input_ids[0] == image_token_id)
        if img_placeholder_mask.any():
            img_placeholder_pos = img_placeholder_mask.nonzero(as_tuple=True)[0][0].item()
            img_start = img_placeholder_pos
            img_end = img_start + NUM_VISUAL_TOKENS
        else:
            img_start, img_end = self._find_img_range_from_embeds(inputs)

        out, captures = run_forward_with_dgst_captures(self.model, **inputs)

        if (
            out.attentions is None
            or len(out.attentions) == 0
            or out.attentions[0] is None
        ):
            raise RuntimeError(
                "out.attentions is empty or None. This usually means flash attention "
                "is active and suppressing attention output. "
                "Fix: load the model with attn_implementation='eager':\n"
                "  LlavaForConditionalGeneration.from_pretrained(..., "
                "attn_implementation='eager')"
            )

        expanded_seq_len = out.attentions[0].shape[-1]
        text_to_patch_attn, text_to_text_attn = self._extract_attention_features(
            out.attentions, img_start, img_end, expanded_seq_len
        )

        token_hidden_states, patch_hidden_states = self._extract_hidden_states(
            out.hidden_states, img_start, img_end
        )

        pred_token_id = out.logits[0, -1].argmax().item()
        pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
        last_logits = out.logits[0, -1].float().cpu()
        dgst_target_id = int(target_token_id) if target_token_id is not None else int(pred_token_id)
        dgst_t_raw = build_dgst_t_raw(
            model=self.model,
            full_input_ids=input_ids[0].tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            captures=captures,
            visual_start=img_start,
            visual_end=img_end,
            image_token_id=image_token_id,
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

        prompt_text = self.cfg["prompt_template"]
        prefix_inputs = self.processor(
            text=prompt_text,
            images=image,
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
        full_inputs = _to_device_dtype(full_inputs, self.device, torch.float16)
        input_ids = full_inputs["input_ids"]

        image_token_id = int(getattr(self.model.config, "image_token_index", IMAGE_TOKEN_INDEX))
        img_placeholder_mask = input_ids[0] == image_token_id
        if img_placeholder_mask.any():
            img_placeholder_pos = img_placeholder_mask.nonzero(as_tuple=True)[0][0].item()
            img_start = img_placeholder_pos
            img_end = img_start + NUM_VISUAL_TOKENS
        else:
            img_start, img_end = self._find_img_range_from_embeds(full_inputs)

        out, captures = run_forward_with_dgst_captures(
            self.model,
            output_hidden_states=False,
            **full_inputs,
        )
        if out.attentions is None or len(out.attentions) == 0 or out.attentions[0] is None:
            raise RuntimeError("DGST-T batch extraction requires attention weights; use eager attention.")

        expanded_seq_len = int(out.attentions[0].shape[-1])
        visual_token_count = int(img_end - img_start)
        prompt_positions = resolve_prompt_positions(
            full_input_ids=input_ids[0].tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            image_token_id=image_token_id,
            visual_start=img_start,
            visual_end=img_end,
        )
        prediction_positions = pre_token_prediction_positions(
            full_input_ids=input_ids[0].tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            response_token_indices=requested_indices,
            image_token_id=image_token_id,
            visual_token_count=visual_token_count,
            prompt_positions=prompt_positions,
        )
        dgst_results = None
        dgst_raws = None
        if cfg_dgst_t is not None:
            dgst_results = compute_dgst_t_batch_from_captures(
                model=self.model,
                full_input_ids=input_ids[0].tolist(),
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
                topmass_alpha=cfg_dgst_t.get("topmass_085_alpha", 0.85),
                capped_topmass_alpha=cfg_dgst_t.get("capped_topmass_085_alpha", 0.85),
                capped_topmass_min_k=cfg_dgst_t.get("capped_topmass_085_min_k", 32),
                capped_topmass_max_k=cfg_dgst_t.get("capped_topmass_085_max_k", 64),
                compute_topmass_085=cfg_dgst_t.get("compute_topmass_085", True),
                compute_capped_topmass_085=cfg_dgst_t.get("compute_capped_topmass_085", True),
                target_gate_mode=cfg_dgst_t.get("target_gate_mode", "legacy_prob"),
                relative_vll_mad_epsilon=cfg_dgst_t.get("relative_vll_mad_epsilon", 1e-6),
            )
        else:
            dgst_raws = build_dgst_t_raw_batch(
                model=self.model,
                full_input_ids=input_ids[0].tolist(),
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
            text_to_patch_attn, text_to_text_attn = self._extract_attention_features_at_position(
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


    def _find_img_range_from_embeds(self, inputs: dict) -> Tuple[int, int]:
        """Fallback: estimate img_start by counting non-image prompt tokens."""
        return 4, 4 + NUM_VISUAL_TOKENS

    @staticmethod
    def _extract_attention_features(
        attentions: tuple,
        img_start: int,
        img_end: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract two attention tensors from the last token across all layers:"""
        visual_set = set(range(img_start, img_end))
        last_pos   = seq_len - 1
        text_indices = [
            i for i in range(seq_len)
            if i not in visual_set and i != last_pos
        ]
        text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

        patch_layers = []
        text_layers  = []
        for layer_attn in attentions:
            text_idx_tensor = text_idx_tensor.to(layer_attn.device)
            row = layer_attn[0, :, last_pos, :]
            patch_layers.append(row[:, img_start:img_end])
            text_layers.append(row[:, text_idx_tensor])

        return (
            torch.stack(patch_layers, dim=0),
            torch.stack(text_layers,  dim=0),
        )

    @staticmethod
    def _extract_attention_features_at_position(
        attentions: tuple,
        img_start: int,
        img_end: int,
        seq_len: int,
        token_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract attention features for a specific prediction position."""
        visual_set = set(range(img_start, img_end))
        last_pos = int(token_position)
        text_indices = [
            i for i in range(seq_len)
            if i not in visual_set and i != last_pos
        ]
        text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

        patch_layers = []
        text_layers = []
        for layer_attn in attentions:
            row = layer_attn[0, :, last_pos, :]
            patch_layers.append(row[:, img_start:img_end])
            text_layers.append(row[:, text_idx_tensor])

        return (
            torch.stack(patch_layers, dim=0),
            torch.stack(text_layers, dim=0),
        )

    @staticmethod
    def _extract_hidden_states(
        hidden_states: tuple,
        img_start: int,
        img_end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: tuple of [1, seq_len, hidden_dim], length = num_layers+1"""
        token_list, patch_list = [], []
        for hs in hidden_states[1:]:
            token_list.append(hs[0, -1, :])
            patch_list.append(hs[0, img_start:img_end, :])
        token_hs = torch.stack(token_list, dim=0)
        patch_hs = torch.stack(patch_list, dim=0)
        return token_hs, patch_hs


from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def _to_device_dtype(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                result[key] = value.to(device=device, dtype=dtype)
            else:
                result[key] = value.to(device=device)
        else:
            result[key] = value
    return result

class LLaVANextWrapper(LLaVAWrapper):
    """Wrapper for LLaVA-Next (1.6) — dynamic resolution variant of LLaVA."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[LLaVANextWrapper] Loading model from {hf_name} …")
        self.processor = LlavaNextProcessor.from_pretrained(hf_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        print("[LLaVANextWrapper] Model loaded.")
