"""LLaVA-OneVision 1.5 wrapper for generation and DGST-T feature extraction."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from features.dgst_t import compute_dgst_t
from models.base_wrapper import BaseLVLMWrapper, GenerationOutput, ModelOutput
from models.dgst_capture import build_dgst_t_raw, run_forward_with_dgst_captures
from models.prompt_support import resolve_prompt_support_positions


class LLaVAOneVisionWrapper(BaseLVLMWrapper):
    """Wrapper for local LLaVA-OneVision-1.5-8B-Instruct checkpoints.

    The released OneVision 1.5 checkpoint uses a Qwen2.5-VL style processor and
    chat template, but exposes its model through trust_remote_code.  This class
    keeps the token-detector interface aligned with the existing Qwen wrapper
    while locating the expanded image-pad token span dynamically.
    """

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[LLaVAOneVisionWrapper] Loading model from {hf_name} ...")

        try:
            self.processor = AutoProcessor.from_pretrained(
                hf_name,
                trust_remote_code=True,
                fix_mistral_regex=True,
                use_fast=False,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                hf_name,
                trust_remote_code=True,
                use_fast=False,
            )

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device,
            "attn_implementation": "eager",
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)

        self.model.eval()
        self.tokenizer = self.processor.tokenizer

        self._vision_start_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_start_token", "<|vision_start|>")
        )
        self._vision_end_id = self.tokenizer.convert_tokens_to_ids(
            self.cfg.get("vision_end_token", "<|vision_end|>")
        )
        config_image_token_id = getattr(self.model.config, "image_token_id", None)
        self._image_token_id = int(
            config_image_token_id
            if config_image_token_id is not None
            else self.tokenizer.convert_tokens_to_ids(
                self.cfg.get("image_token", "<|image_pad|>")
            )
        )
        print(
            "[LLaVAOneVisionWrapper] Loaded. "
            f"vision_start={self._vision_start_id}, "
            f"vision_end={self._vision_end_id}, "
            f"image_token={self._image_token_id}"
        )

    @property
    def num_layers(self) -> int:
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            return int(text_config.num_hidden_layers)
        language_model = getattr(self.model, "language_model", None)
        lm_config = getattr(language_model, "config", None)
        if lm_config is not None and hasattr(lm_config, "num_hidden_layers"):
            return int(lm_config.num_hidden_layers)
        return int(getattr(self.model.config, "num_hidden_layers"))

    @property
    def num_visual_tokens(self) -> int:
        return int(self.cfg.get("num_visual_tokens") or 144)

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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=self.generation_max_new_tokens,
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
        cfg_dgst_t: Optional[dict] = None,
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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])
        partial_response = self.tokenizer.decode(
            prefix_token_ids,
            skip_special_tokens=True,
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

        expanded_seq_len = int(out.attentions[0].shape[-1])
        text_to_patch_attn, text_to_text_attn = _extract_attention_features(
            out.attentions,
            img_start,
            img_end,
            expanded_seq_len,
        )
        token_hidden_states, patch_hidden_states = _extract_hidden_states(
            out.hidden_states,
            img_start,
            img_end,
        )

        pred_token_id = int(out.logits[0, -1].argmax().item())
        pred_token_str = self.tokenizer.decode(
            [pred_token_id],
            skip_special_tokens=False,
        )
        last_logits = out.logits[0, -1].float().cpu()
        dgst_target_id = (
            int(target_token_id) if target_token_id is not None else int(pred_token_id)
        )
        prompt_positions_override = resolve_prompt_support_positions(
            tokenizer=self.tokenizer,
            full_input_ids=input_ids.tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            image_token_id=self._image_token_id,
            visual_start=img_start,
            visual_end=img_end,
            cfg_dgst_t=cfg_dgst_t,
            model_name="LLaVA-OneVision-1.5",
        )
        dgst_t_raw = build_dgst_t_raw(
            model=self.model,
            full_input_ids=input_ids.tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            captures=captures,
            visual_start=img_start,
            visual_end=img_end,
            image_token_id=self._image_token_id,
            target_token_id=dgst_target_id,
            prediction_position=expanded_seq_len - 1,
            support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
            relative_vll_logit_source=(
                cfg_dgst_t.get("relative_vll_logit_source", "h_mid")
                if cfg_dgst_t is not None
                else "h_mid"
            ),
            prompt_positions_override=prompt_positions_override,
            keep_on_device=cfg_dgst_t is not None,
        )
        dgst_t_result = None
        if cfg_dgst_t is not None:
            dgst_t_result = compute_dgst_t(
                dgst_t_raw,
                tau=cfg_dgst_t.get("tau", 0.07),
                source_distribution_mode=cfg_dgst_t.get(
                    "source_distribution_mode", "softmax"
                ),
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
                capped_topmass_alpha=cfg_dgst_t.get(
                    "capped_topmass_085_alpha", 0.85
                ),
                capped_topmass_min_k=cfg_dgst_t.get("capped_topmass_085_min_k", 32),
                capped_topmass_max_k=cfg_dgst_t.get("capped_topmass_085_max_k", 64),
                compute_topmass_085=cfg_dgst_t.get("compute_topmass_085", True),
                compute_capped_topmass_085=cfg_dgst_t.get(
                    "compute_capped_topmass_085", True
                ),
                target_gate_mode=cfg_dgst_t.get("target_gate_mode", "legacy_prob"),
                relative_vll_mad_epsilon=cfg_dgst_t.get(
                    "relative_vll_mad_epsilon", 1e-6
                ),
                relative_cost_mode=cfg_dgst_t.get("relative_cost_mode"),
                relative_cost_modes=cfg_dgst_t.get("relative_cost_modes"),
                relative_cost_state_modes=cfg_dgst_t.get(
                    "relative_cost_state_modes"
                ),
                relative_cost_update_lambdas=cfg_dgst_t.get(
                    "relative_cost_update_lambdas"
                ),
                relative_barrier_lambda=cfg_dgst_t.get(
                    "relative_barrier_lambda", 1.0
                ),
                relative_barrier_margin=cfg_dgst_t.get(
                    "relative_barrier_margin", 0.5
                ),
                relative_barrier_max=cfg_dgst_t.get("relative_barrier_max", 3.0),
                source_modes=cfg_dgst_t.get("source_modes"),
                target_attention_gammas=cfg_dgst_t.get("target_attention_gammas"),
                target_attention_epsilon=cfg_dgst_t.get(
                    "target_attention_epsilon", 1e-12
                ),
                compute_ffn_injection_features=cfg_dgst_t.get(
                    "compute_ffn_injection_features", True
                ),
                ffn_injection_evidence_top_k=cfg_dgst_t.get(
                    "ffn_injection_evidence_top_k", 32
                ),
                ffn_injection_evidence_rank=cfg_dgst_t.get(
                    "ffn_injection_evidence_rank", 8
                ),
                ffn_injection_eps=cfg_dgst_t.get("ffn_injection_eps", 1e-12),
                compute_dual_scope=cfg_dgst_t.get(
                    "dgst_t_dual_scope",
                    cfg_dgst_t.get("compute_dual_scope", False),
                ),
            )
            dgst_t_raw = None

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
            dgst_t_result=dgst_t_result,
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

        outputs: List[ModelOutput] = []
        for response_index, target_token_id in zip(requested_indices, targets):
            outputs.append(
                self.extract_token_features(
                    image=image,
                    prefix_token_ids=response_ids[:response_index],
                    response_token_idx=int(response_index),
                    target_token_id=int(target_token_id),
                    cfg_dgst_t=cfg_dgst_t,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = [int(token_id) for token_id in input_ids.tolist()]
        image_positions = [
            index for index, token_id in enumerate(ids) if token_id == self._image_token_id
        ]
        if image_positions:
            return int(min(image_positions)), int(max(image_positions) + 1)

        try:
            vs_pos = ids.index(int(self._vision_start_id))
            ve_pos = ids.index(int(self._vision_end_id))
            return int(vs_pos + 1), int(ve_pos)
        except ValueError:
            fallback_tokens = self.num_visual_tokens
            return 5, 5 + fallback_tokens


def _extract_hidden_states(
    hidden_states: tuple,
    img_start: int,
    img_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, -1, :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)


def _extract_attention_features(
    attentions: tuple,
    img_start: int,
    img_end: int,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    visual_set = set(range(int(img_start), int(img_end)))
    last_pos = int(seq_len) - 1
    text_indices = [
        i for i in range(int(seq_len)) if i not in visual_set and i != last_pos
    ]
    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        text_idx_tensor = torch.tensor(
            text_indices,
            dtype=torch.long,
            device=row.device,
        )
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)
