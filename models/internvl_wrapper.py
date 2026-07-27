"""InternVL-2.5 wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from models.base_wrapper import (
    AttentionRequirement,
    BaseLVLMWrapper,
    ExtractionRequirements,
    GenerationOutput,
    ModelOutput,
    PromptTargetRequest,
    compact_response_logit_statistics,
)
from models.dgst_capture import (
    build_dgst_t_raw,
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    pre_token_prediction_positions,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
    run_forward_with_layer_hidden_captures,
)
from models.prompt_support import resolve_prompt_support_positions
from models.prompt_target import (
    extract_prompt_target_from_inputs,
    resolve_prompt_target_alignment,
)
from features.dgst_t import (
    compute_dgst_t_batch_from_captures,
    compute_four_gate_dgst_batch_from_captures,
)

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


class InternVLWrapper(BaseLVLMWrapper):
    """Wrapper for InternVL2.5-8B."""

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[InternVLWrapper] Loading model from {hf_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self._img_ctx_id

        from transformers import GenerationMixin, GenerationConfig
        lm = self.model.language_model
        lm_cls = type(lm)
        if GenerationMixin not in lm_cls.__mro__:
            lm_cls.__bases__ = (GenerationMixin,) + lm_cls.__bases__
        if lm.generation_config is None:
            lm.generation_config = GenerationConfig(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        _original_prepare = lm_cls.prepare_inputs_for_generation
        def _safe_prepare(self_lm, input_ids, past_key_values=None, **kwargs):
            if past_key_values is None:
                kwargs.pop("past_key_values", None)
                return {
                    "input_ids": input_ids,
                    "inputs_embeds": kwargs.get("inputs_embeds"),
                    "attention_mask": kwargs.get("attention_mask"),
                    "past_key_values": None,
                    "use_cache": kwargs.get("use_cache", True),
                }
            return _original_prepare(self_lm, input_ids, past_key_values=past_key_values, **kwargs)
        lm_cls.prepare_inputs_for_generation = _safe_prepare

        print(f"[InternVLWrapper] Loaded. IMG_CONTEXT token id = {self._img_ctx_id}")

    @property
    def num_layers(self) -> int:
        return self.model.language_model.config.num_hidden_layers

    @property
    def num_visual_tokens(self) -> int:
        return self.cfg.get("num_visual_tokens", 256)


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        prompt = self.resolve_prompt(prompt)

        pixel_values = self._preprocess_image(image)
        input_ids, img_start, img_end = self._build_input_ids_with_image(
            pixel_values, prefix_token_ids=[], user_prompt=prompt
        )
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)

            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            img_mask = (input_ids == self._img_ctx_id).squeeze(0)
            input_embeds[0][img_mask] = vit_embeds.reshape(-1, vit_embeds.shape[-1])

            response_ids = []
            past_key_values = None
            cur_embeds = input_embeds
            stop_token_ids = set(_as_token_id_list(self.tokenizer.eos_token_id))
            im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
            if (
                isinstance(im_end_token_id, int)
                and im_end_token_id >= 0
                and im_end_token_id != unk_token_id
            ):
                stop_token_ids.add(int(im_end_token_id))

            for _ in range(self.generation_max_new_tokens):
                out = self.model.language_model(
                    inputs_embeds=cur_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                next_token_id = int(out.logits[0, -1].argmax())
                response_ids.append(next_token_id)
                if next_token_id in stop_token_ids:
                    break
                cur_embeds = self.model.language_model.get_input_embeddings()(
                    torch.tensor([[next_token_id]], device=self.device)
                )

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
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        """Build InternVL's native chat input and retain requested tensors."""
        user_prompt = self.resolve_prompt(prompt)
        requirements_were_explicit = requirements is not None
        requirements = self.resolve_extraction_requirements(
            requirements,
            dgst_enabled=cfg_dgst_t is not None,
        )
        pixel_values = self._preprocess_image(image)

        prompt_input_ids, _, _ = self._build_input_ids_with_image(
            pixel_values, prefix_token_ids=[], user_prompt=user_prompt
        )
        prompt_tokenized_length = int(prompt_input_ids.shape[1])

        input_ids, img_start, img_end = self._build_input_ids_with_image(
            pixel_values, prefix_token_ids, user_prompt=user_prompt
        )

        attention_mask = torch.ones_like(input_ids)

        image_flags = torch.ones(
            pixel_values.shape[0], dtype=torch.long, device=self.device
        )

        forward_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "image_flags": image_flags,
        }
        use_dgst = bool(requirements.dgst_capture and cfg_dgst_t is not None)
        layer_outputs = None
        if use_dgst:
            out, captures = run_forward_with_dgst_captures(
                self.model,
                output_hidden_states=False,
                retain_attention_updates=(
                    not _is_four_gate_mode(cfg_dgst_t)
                    or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
                ),
                **forward_inputs,
            )
        elif requirements.needs_hidden_states:
            captures = None
            out, layer_outputs = run_forward_with_layer_hidden_captures(
                self.model,
                output_attentions=requirements.needs_attention_weights,
                capture_all_layers=(
                    requirements.token_hidden_states
                    or requirements.patch_hidden_states
                ),
                **forward_inputs,
            )
        else:
            captures = None
            with torch.no_grad():
                out = self.model(
                    **forward_inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        seq_len = int(out.logits.shape[1])
        compact_profile = _is_compact_profile(cfg_dgst_t)
        keep_attention = (
            (not compact_profile or requirements_were_explicit)
            and requirements.attention is not AttentionRequirement.NONE
        )
        keep_hidden = (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        )
        if keep_attention:
            _require_attentions(out, model_name="InternVL")
            text_to_patch_attn, text_to_text_attn = _extract_attention_features(
                out.attentions, img_start, img_end, seq_len
            )
            if requirements.attention is AttentionRequirement.HEAD_MEAN:
                text_to_patch_attn = text_to_patch_attn.mean(dim=1, keepdim=True)
                text_to_text_attn = text_to_text_attn.mean(dim=1, keepdim=True)
        else:
            text_to_patch_attn = torch.empty(0)
            text_to_text_attn = torch.empty(0)
        text_to_patch_attn = text_to_patch_attn.cpu()
        text_to_text_attn = text_to_text_attn.cpu()
        out.attentions = None

        if keep_hidden:
            if captures is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                    captures,
                    token_position=seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                )
            elif layer_outputs is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                    layer_outputs,
                    token_position=seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                )
            elif out.hidden_states is not None:
                token_hidden_states, patch_hidden_states = _extract_hidden_states(
                    out.hidden_states, img_start, img_end
                )
            else:
                raise RuntimeError("InternVL hidden states were requested but not returned.")
            if not requirements.token_hidden_states:
                token_hidden_states = torch.empty(0, device=patch_hidden_states.device)
            if not requirements.patch_hidden_states:
                patch_hidden_states = torch.empty(0, device=token_hidden_states.device)
        else:
            token_hidden_states = torch.empty(0)
            patch_hidden_states = torch.empty(0)
        token_hidden_states = token_hidden_states.cpu()
        patch_hidden_states = patch_hidden_states.cpu()

        response_hidden = None
        if requirements.response_hidden_states:
            response_hidden = final_normalized_hidden_slice(
                model=self.model,
                out=out,
                dgst_captures=captures,
                layer_outputs=layer_outputs,
                start=prompt_tokenized_length,
                end=prompt_tokenized_length + len(prefix_token_ids),
            ).cpu()

        pred_token_id = int(out.logits[0, -1].argmax().item())
        pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
        last_logits = out.logits[0, -1].float().cpu() if requirements.logits else None
        out.logits = None
        dgst_target_id = int(target_token_id) if target_token_id is not None else int(pred_token_id)
        dgst_t_raw = None
        dgst_t_result = None
        if use_dgst:
            prompt_positions_override = resolve_prompt_support_positions(
                tokenizer=self.tokenizer,
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=int(self._img_ctx_id),
                visual_start=img_start,
                visual_end=img_end,
                cfg_dgst_t=cfg_dgst_t,
                model_name="InternVL",
            )
            if _is_four_gate_mode(cfg_dgst_t):
                dgst_t_result = compute_four_gate_dgst_batch_from_captures(
                    model=self.model,
                    captures=captures,
                    visual_start=img_start,
                    visual_end=img_end,
                    target_token_ids=[dgst_target_id],
                    prediction_positions=[seq_len - 1],
                    prompt_positions=prompt_positions_override,
                    semantic_chunk_size=int(cfg_dgst_t.get("semantic_chunk_size", 64)),
                    tau=float(cfg_dgst_t.get("tau", 0.07)),
                    transport_top_k=int(cfg_dgst_t.get("transport_top_k", 64)),
                    target_region_top_k=int(cfg_dgst_t.get("atarget_visual_top_k", 32)),
                    mad_epsilon=float(cfg_dgst_t.get("relative_vll_mad_epsilon", 1e-6)),
                    cost_mode=cfg_dgst_t.get("cost_mode", "sqrt_matched_state"),
                    cost_modes=cfg_dgst_t.get("cost_modes"),
                    enabled_methods=cfg_dgst_t.get("four_gate_methods"),
                    compute_dual_scope=bool(
                        cfg_dgst_t.get(
                            "dgst_t_dual_scope",
                            cfg_dgst_t.get("compute_dual_scope", False),
                        )
                    ),
                    support_modes=cfg_dgst_t.get("support_modes"),
                    source_tau_values=cfg_dgst_t.get("source_tau_values"),
                    transport_top_k_values=cfg_dgst_t.get(
                        "transport_top_k_values"
                    ),
                    compute_capped_topmass_085=bool(
                        cfg_dgst_t.get("compute_capped_topmass_085", False)
                    ),
                    capped_topmass_alphas=cfg_dgst_t.get(
                        "capped_topmass_alphas"
                    ),
                    capped_topmass_alpha=float(
                        cfg_dgst_t.get("capped_topmass_085_alpha", 0.85)
                    ),
                    capped_topmass_min_k=int(
                        cfg_dgst_t.get("capped_topmass_085_min_k", 32)
                    ),
                    capped_topmass_max_k=int(
                        cfg_dgst_t.get("capped_topmass_085_max_k", 64)
                    ),
                    compute_prompt_cafe=bool(
                        cfg_dgst_t.get("compute_prompt_cafe", False)
                    ),
                    prompt_cafe_temperature=float(
                        cfg_dgst_t.get("prompt_cafe_temperature", 10.0)
                    ),
                    prompt_cafe_layer=int(
                        cfg_dgst_t.get("prompt_cafe_layer", 22)
                    ),
                    compute_ffn_injection_features=bool(
                        cfg_dgst_t.get("compute_ffn_injection_features", False)
                    ),
                    ffn_injection_eps=float(
                        cfg_dgst_t.get("ffn_injection_eps", 1e-12)
                    ),
                    release_layer_captures=True,
                )[0]
            else:
                dgst_t_raw = build_dgst_t_raw(
                    model=self.model,
                    full_input_ids=input_ids[0].tolist(),
                    prompt_tokenized_length=prompt_tokenized_length,
                    captures=captures,
                    visual_start=img_start,
                    visual_end=img_end,
                    image_token_id=int(self._img_ctx_id),
                    target_token_id=dgst_target_id,
                    prediction_position=seq_len - 1,
                    support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
                    semantic_chunk_size=int(cfg_dgst_t.get("semantic_chunk_size", 64)),
                    relative_vll_logit_source=cfg_dgst_t.get(
                        "relative_vll_logit_source", "h_mid"
                    ),
                    prompt_positions_override=prompt_positions_override,
                )

        return ModelOutput(
            token_id=pred_token_id,
            token_str=pred_token_str,
            text_to_patch_attn=text_to_patch_attn,
            text_to_text_attn=text_to_text_attn,
            token_hidden_states=token_hidden_states,
            patch_hidden_states=patch_hidden_states,
            response_token_idx=response_token_idx,
            token_logits=last_logits,
            dgst_t_raw=dgst_t_raw,
            dgst_t_result=dgst_t_result,
            visual_grid=(
                _square_visual_grid(img_end - img_start)
                if requirements.visual_layout
                else None
            ),
            response_hidden_states=response_hidden,
            baseline_capture={
                "prediction_position": seq_len - 1,
                "visual_start": int(img_start),
                "visual_end": int(img_end),
                "attention_requirement": requirements.attention.value,
            },
        )

    def extract_token_features_batch(
        self,
        image: Image.Image,
        response_token_ids: Sequence[int],
        response_token_indices: Sequence[int],
        target_token_ids: Optional[Sequence[int]] = None,
        cfg_dgst_t: Optional[dict] = None,
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> List[ModelOutput]:
        """Extract every requested causal row in one full-caption forward."""
        return self._extract_token_features_batch_full_response(
            image=image,
            response_token_ids=response_token_ids,
            response_token_indices=response_token_indices,
            target_token_ids=target_token_ids,
            cfg_dgst_t=cfg_dgst_t,
            prompt=prompt,
            requirements=requirements,
        )

    def _extract_token_features_batch_full_response(
        self,
        image: Image.Image,
        response_token_ids: Sequence[int],
        response_token_indices: Sequence[int],
        target_token_ids: Optional[Sequence[int]] = None,
        cfg_dgst_t: Optional[dict] = None,
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> List[ModelOutput]:
        """Use causal-mask rows from one teacher-forced full response."""
        response_ids, requested_indices, targets = self.validate_causal_batch_request(
            response_token_ids=response_token_ids,
            response_token_indices=response_token_indices,
            target_token_ids=target_token_ids,
        )
        if not requested_indices:
            return []
        requirements_were_explicit = requirements is not None
        requirements = self.resolve_extraction_requirements(
            requirements,
            dgst_enabled=cfg_dgst_t is not None,
        )
        user_prompt = self.resolve_prompt(prompt)

        pixel_values = self._preprocess_image(image)
        prompt_input_ids, _, _ = self._build_input_ids_with_image(
            pixel_values,
            prefix_token_ids=[],
            user_prompt=user_prompt,
        )
        prompt_tokenized_length = int(prompt_input_ids.shape[1])
        input_ids, img_start, img_end = self._build_input_ids_with_image(
            pixel_values,
            prefix_token_ids=response_ids,
            user_prompt=user_prompt,
        )
        attention_mask = torch.ones_like(input_ids)
        image_flags = torch.ones(
            pixel_values.shape[0],
            dtype=torch.long,
            device=self.device,
        )

        forward_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "image_flags": image_flags,
        }
        use_dgst = bool(requirements.dgst_capture and cfg_dgst_t is not None)
        layer_outputs = None
        if use_dgst:
            out, captures = run_forward_with_dgst_captures(
                self.model,
                output_hidden_states=False,
                retain_attention_updates=(
                    not _is_four_gate_mode(cfg_dgst_t)
                    or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
                ),
                **forward_inputs,
            )
        elif requirements.needs_hidden_states:
            captures = None
            out, layer_outputs = run_forward_with_layer_hidden_captures(
                self.model,
                output_attentions=requirements.needs_attention_weights,
                capture_all_layers=(
                    requirements.token_hidden_states
                    or requirements.patch_hidden_states
                ),
                **forward_inputs,
            )
        else:
            captures = None
            with torch.no_grad():
                out = self.model(
                    **forward_inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        seq_len = int(out.logits.shape[1])
        full_prompt_positions = resolve_prompt_positions(
            full_input_ids=input_ids[0].tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            image_token_id=int(self._img_ctx_id),
            visual_start=img_start,
            visual_end=img_end,
        )
        prediction_positions = pre_token_prediction_positions(
            full_input_ids=input_ids[0].tolist(),
            prompt_tokenized_length=prompt_tokenized_length,
            response_token_indices=requested_indices,
            image_token_id=int(self._img_ctx_id),
            visual_token_count=int(img_end - img_start),
            prompt_positions=full_prompt_positions,
        )
        compact_profile = _is_compact_profile(cfg_dgst_t)
        keep_attention = (
            (not compact_profile or requirements_were_explicit)
            and requirements.attention is not AttentionRequirement.NONE
        )
        keep_hidden = (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        )
        if keep_attention:
            _require_attentions(out, model_name="InternVL")
        if requirements.logits:
            position_logits: list[Optional[torch.Tensor]] = [
                out.logits[0, int(position)].float().cpu()
                for position in prediction_positions
            ]
            position_pred_ids = [int(logits.argmax().item()) for logits in position_logits]
        else:
            position_logits = [None] * len(prediction_positions)
            position_pred_ids = [
                int(out.logits[0, int(position)].argmax().item())
                for position in prediction_positions
            ]

        shared_response_hidden = None
        shared_baseline_capture: dict[str, Any] = {
            "visual_start": int(img_start),
            "visual_end": int(img_end),
            "attention_requirement": requirements.attention.value,
        }
        if requirements.response_hidden_states:
            shared_response_hidden = final_normalized_hidden_slice(
                model=self.model,
                out=out,
                dgst_captures=captures,
                layer_outputs=layer_outputs,
                start=prompt_tokenized_length,
                end=prompt_tokenized_length + len(response_ids),
            ).cpu()
            response_positions = torch.arange(
                prompt_tokenized_length - 1,
                prompt_tokenized_length - 1 + len(response_ids),
                dtype=torch.long,
                device=out.logits.device,
            )
            response_logits = (
                out.logits[0].index_select(0, response_positions)
                if response_ids
                else torch.empty((0, int(out.logits.shape[-1])), device=out.logits.device)
            )
            shared_baseline_capture.update(
                compact_response_logit_statistics(
                    response_logits,
                    response_token_ids=response_ids,
                )
            )
            del response_logits
        out.logits = None
        if not keep_attention:
            out.attentions = None

        dgst_results = None
        if use_dgst:
            support_prompt_positions = resolve_prompt_support_positions(
                tokenizer=self.tokenizer,
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=int(self._img_ctx_id),
                visual_start=img_start,
                visual_end=img_end,
                cfg_dgst_t=cfg_dgst_t,
                model_name="InternVL",
            )
            dgst_results = compute_dgst_t_batch_from_captures(
                model=self.model,
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                captures=captures,
                visual_start=img_start,
                visual_end=img_end,
                image_token_id=int(self._img_ctx_id),
                target_token_ids=targets,
                prediction_positions=prediction_positions,
                support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
                semantic_chunk_size=int(cfg_dgst_t.get("semantic_chunk_size", 64)),
                prompt_positions_override=support_prompt_positions,
                tau=cfg_dgst_t.get("tau", 0.07),
                source_distribution_mode=cfg_dgst_t.get("source_distribution_mode", "softmax"),
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
                four_gate_compute_capped_topmass_085=cfg_dgst_t.get(
                    "compute_capped_topmass_085", False
                ),
                four_gate_capped_topmass_alphas=cfg_dgst_t.get(
                    "capped_topmass_alphas"
                ),
                target_gate_mode=cfg_dgst_t.get("target_gate_mode", "legacy_prob"),
                relative_vll_mad_epsilon=cfg_dgst_t.get("relative_vll_mad_epsilon", 1e-6),
                relative_vll_logit_source=cfg_dgst_t.get("relative_vll_logit_source", "h_mid"),
                relative_cost_mode=cfg_dgst_t.get("relative_cost_mode"),
                relative_cost_modes=cfg_dgst_t.get("relative_cost_modes"),
                relative_cost_state_modes=cfg_dgst_t.get("relative_cost_state_modes"),
                relative_cost_update_lambdas=cfg_dgst_t.get("relative_cost_update_lambdas"),
                relative_barrier_lambda=cfg_dgst_t.get("relative_barrier_lambda", 1.0),
                relative_barrier_margin=cfg_dgst_t.get("relative_barrier_margin", 0.5),
                relative_barrier_max=cfg_dgst_t.get("relative_barrier_max", 3.0),
                source_modes=cfg_dgst_t.get("source_modes"),
                target_attention_gammas=cfg_dgst_t.get("target_attention_gammas"),
                target_attention_epsilon=cfg_dgst_t.get("target_attention_epsilon", 1e-12),
                compute_ffn_injection_features=cfg_dgst_t.get("compute_ffn_injection_features", True),
                ffn_injection_evidence_top_k=cfg_dgst_t.get("ffn_injection_evidence_top_k", 32),
                ffn_injection_evidence_rank=cfg_dgst_t.get("ffn_injection_evidence_rank", 8),
                ffn_injection_eps=cfg_dgst_t.get("ffn_injection_eps", 1e-12),
                compute_dual_scope=cfg_dgst_t.get(
                    "dgst_t_dual_scope",
                    cfg_dgst_t.get("compute_dual_scope", False),
                ),
                four_gate_methods=cfg_dgst_t.get("four_gate_methods"),
                four_gate_cost_modes=cfg_dgst_t.get("cost_modes"),
                four_gate_support_modes=cfg_dgst_t.get("support_modes"),
                four_gate_source_tau_values=cfg_dgst_t.get(
                    "source_tau_values"
                ),
                four_gate_transport_top_k_values=cfg_dgst_t.get(
                    "transport_top_k_values"
                ),
                compute_prompt_cafe=bool(
                    cfg_dgst_t.get("compute_prompt_cafe", False)
                ),
                prompt_cafe_temperature=float(
                    cfg_dgst_t.get("prompt_cafe_temperature", 10.0)
                ),
                prompt_cafe_layer=int(cfg_dgst_t.get("prompt_cafe_layer", 22)),
                release_layer_captures=(not keep_attention and not keep_hidden),
            )

        outputs: List[ModelOutput] = []
        for offset, (response_index, prediction_position) in enumerate(zip(
            requested_indices,
            prediction_positions,
        )):
            if keep_attention:
                text_to_patch_attn, text_to_text_attn = _extract_attention_features_at_position(
                    out.attentions,
                    img_start,
                    img_end,
                    seq_len,
                    int(prediction_position),
                )
                if requirements.attention is AttentionRequirement.HEAD_MEAN:
                    text_to_patch_attn = text_to_patch_attn.mean(dim=1, keepdim=True)
                    text_to_text_attn = text_to_text_attn.mean(dim=1, keepdim=True)
            else:
                text_to_patch_attn = torch.empty(0)
                text_to_text_attn = torch.empty(0)
            if keep_hidden:
                if captures is not None:
                    token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                        captures,
                        token_position=int(prediction_position),
                        visual_start=img_start,
                        visual_end=img_end,
                    )
                elif layer_outputs is not None:
                    token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                        layer_outputs,
                        token_position=int(prediction_position),
                        visual_start=img_start,
                        visual_end=img_end,
                    )
                elif out.hidden_states is not None:
                    token_hidden_states, patch_hidden_states = _extract_hidden_states_at_position(
                        out.hidden_states,
                        img_start,
                        img_end,
                        int(prediction_position),
                    )
                else:
                    raise RuntimeError("InternVL hidden states were requested but not returned.")
                if not requirements.token_hidden_states:
                    token_hidden_states = torch.empty(0, device=patch_hidden_states.device)
                if not requirements.patch_hidden_states:
                    patch_hidden_states = torch.empty(0, device=token_hidden_states.device)
            else:
                token_hidden_states = torch.empty(0)
                patch_hidden_states = torch.empty(0)
            logits = position_logits[offset]
            pred_token_id = position_pred_ids[offset]
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
                    dgst_t_raw=None,
                    dgst_t_result=dgst_results[offset] if dgst_results is not None else None,
                    visual_grid=(
                        _square_visual_grid(img_end - img_start)
                        if requirements.visual_layout
                        else None
                    ),
                    response_hidden_states=shared_response_hidden,
                    baseline_capture={
                        **shared_baseline_capture,
                        "prediction_position": int(prediction_position),
                    },
                )
            )
        out.attentions = None
        if compact_profile and not requirements_were_explicit:
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return outputs

    def _extract_full_response_baseline_capture(
        self,
        *,
        image: Image.Image,
        response_token_ids: Sequence[int],
        prompt: Optional[str],
    ) -> dict[str, Any]:
        """One lightweight full-caption pass for HalLoc/MetaToken inputs."""
        user_prompt = self.resolve_prompt(prompt)
        pixel_values = self._preprocess_image(image)
        prompt_input_ids, _, _ = self._build_input_ids_with_image(
            pixel_values,
            prefix_token_ids=[],
            user_prompt=user_prompt,
        )
        prompt_length = int(prompt_input_ids.shape[1])
        input_ids, _, _ = self._build_input_ids_with_image(
            pixel_values,
            prefix_token_ids=[int(token_id) for token_id in response_token_ids],
            user_prompt=user_prompt,
        )
        response_count = len(response_token_ids)
        if response_count == 0:
            return {
                "response_hidden_states": torch.empty((0, 0)),
                "statistics": compact_response_logit_statistics(
                    torch.empty((0, int(self.model.config.vocab_size))),
                    response_token_ids=[],
                ),
            }

        attention_mask = torch.ones_like(input_ids)
        image_flags = torch.ones(
            pixel_values.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                pixel_values=pixel_values.to(self.device),
                image_flags=image_flags,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        response_hidden = out.hidden_states[-1][
            0,
            prompt_length : prompt_length + response_count,
            :,
        ].detach().cpu()
        prediction_positions = torch.arange(
            prompt_length - 1,
            prompt_length + response_count - 1,
            dtype=torch.long,
            device=out.logits.device,
        )
        teacher_logits = out.logits[0].index_select(0, prediction_positions)
        statistics = compact_response_logit_statistics(
            teacher_logits,
            response_token_ids=response_token_ids,
        )
        del teacher_logits, out
        return {
            "response_hidden_states": response_hidden,
            "statistics": statistics,
        }

    def extract_prompt_target_features(
        self,
        image: Image.Image,
        request: PromptTargetRequest,
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        pixel_values = self._preprocess_image(image)
        input_ids, visual_start, visual_end = self._build_input_ids_with_image(
            pixel_values,
            prefix_token_ids=[],
            user_prompt=request.prompt,
        )
        attention_mask = torch.ones_like(input_ids)
        image_flags = torch.ones(
            pixel_values.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": pixel_values.to(self.device),
            "image_flags": image_flags,
        }
        alignment = resolve_prompt_target_alignment(
            tokenizer=self.tokenizer,
            full_input_ids=input_ids[0].tolist(),
            request=request,
            image_token_id=self._img_ctx_id,
            visual_token_count=visual_end - visual_start,
        )
        return extract_prompt_target_from_inputs(
            wrapper=self,
            inputs=inputs,
            full_input_ids=input_ids[0].tolist(),
            alignment=alignment,
            visual_start=visual_start,
            visual_end=visual_end,
            image_token_id=self._img_ctx_id,
            visual_grid=_square_visual_grid(visual_end - visual_start),
            cfg_dgst_t=cfg_dgst_t,
            requirements=requirements,
            model_name="InternVL",
            support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
        )


    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Use the legacy single-tile InternVL preprocessing."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(
                (self.cfg["image_size"], self.cfg["image_size"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transform(image.convert("RGB")).unsqueeze(0).to(torch.bfloat16).to(self.device)

    def _build_input_ids_with_image(
        self,
        pixel_values: torch.Tensor,
        prefix_token_ids: List[int],
        user_prompt: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int, int]:
        """Assemble the legacy token sequence that InternVL's LM backbone sees."""
        num_tiles = pixel_values.shape[0]
        tokens_per_tile = self.cfg.get("num_visual_tokens", 256)
        num_img_tokens = num_tiles * tokens_per_tile

        if user_prompt is None:
            user_prompt = "Describe this image."

        sys_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        sys_ids = self.tokenizer.encode(sys_prompt, add_special_tokens=False)

        img_start_ids = self.tokenizer.encode(IMG_START_TOKEN, add_special_tokens=False)
        img_ctx_ids = [self._img_ctx_id] * num_img_tokens
        img_end_ids = self.tokenizer.encode(IMG_END_TOKEN, add_special_tokens=False)

        user_suffix_ids = self.tokenizer.encode(
            f"\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )

        full_ids = (
            sys_ids
            + img_start_ids
            + img_ctx_ids
            + img_end_ids
            + user_suffix_ids
            + prefix_token_ids
        )

        img_start = len(sys_ids) + len(img_start_ids)
        img_end = img_start + num_img_tokens

        input_ids = torch.tensor([full_ids], dtype=torch.long)
        return input_ids, img_start, img_end


def _extract_attention_features(
    attentions: tuple,
    img_start: int,
    img_end: int,
    seq_len: int,
) -> tuple:
    """Extract text_to_patch_attn [L, n_heads, n_patches] and"""
    visual_set = set(range(img_start, img_end))
    last_pos = seq_len - 1
    text_indices = [
        i for i in range(seq_len)
        if i not in visual_set and i != last_pos
    ]
    text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, text_idx_tensor])

    return (
        torch.stack(patch_layers, dim=0),
        torch.stack(text_layers,  dim=0),
    )


def _extract_hidden_states(
    hidden_states: tuple,
    img_start: int,
    img_end: int,
) -> tuple:
    """Returns token_hs [L, hidden_dim] and patch_hs [L, n_patches, hidden_dim]."""
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, -1, :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)


def _extract_hidden_states_at_position(
    hidden_states: tuple,
    img_start: int,
    img_end: int,
    token_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_list, patch_list = [], []
    for hs in hidden_states[1:]:
        token_list.append(hs[0, int(token_position), :])
        patch_list.append(hs[0, img_start:img_end, :])
    return torch.stack(token_list, 0), torch.stack(patch_list, 0)


def _as_token_id_list(token_id_or_ids) -> list[int]:
    if token_id_or_ids is None:
        return []
    if isinstance(token_id_or_ids, int):
        return [int(token_id_or_ids)]
    return [int(token_id) for token_id in token_id_or_ids]


def _extract_attention_features_at_position(
    attentions: tuple,
    img_start: int,
    img_end: int,
    seq_len: int,
    token_position: int,
) -> tuple:
    visual_set = set(range(img_start, img_end))
    last_pos = int(token_position)
    text_indices = [
        i for i in range(seq_len)
        if i not in visual_set and i != last_pos
    ]
    text_idx_tensor = torch.tensor(text_indices, dtype=torch.long)

    patch_layers, text_layers = [], []
    for layer_attn in attentions:
        row = layer_attn[0, :, last_pos, :]
        idx = text_idx_tensor.to(layer_attn.device)
        patch_layers.append(row[:, img_start:img_end])
        text_layers.append(row[:, idx])

    return (
        torch.stack(patch_layers, dim=0),
        torch.stack(text_layers, dim=0),
    )


def _is_compact_profile(cfg_dgst_t: Optional[dict]) -> bool:
    return bool(
        cfg_dgst_t is not None
        and cfg_dgst_t.get("feature_output_profile")
        in {"costvariant_vv", "gate_comparison_vv", "four_gate_vv"}
    )


def _is_four_gate_mode(cfg_dgst_t: Optional[dict]) -> bool:
    return bool(
        cfg_dgst_t is not None
        and str(cfg_dgst_t.get("target_gate_mode", "")).strip().lower()
        in {"four_gate", "four_gates", "four_gate_vv", "four_branch"}
    )


def _require_attentions(out: Any, *, model_name: str) -> None:
    attentions = getattr(out, "attentions", None)
    if attentions is None or len(attentions) == 0 or attentions[0] is None:
        raise RuntimeError(
            f"{model_name} attention features were requested but no attention "
            "weights were returned; load the model with eager attention."
        )


def _square_visual_grid(visual_token_count: int) -> Optional[Tuple[int, int]]:
    side = int(round(int(visual_token_count) ** 0.5))
    if side * side != int(visual_token_count):
        return None
    return side, side
