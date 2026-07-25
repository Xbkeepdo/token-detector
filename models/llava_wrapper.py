"""LLaVA-1.5 wrapper for generation and feature extraction."""

from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

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
    attention_row_index,
    build_dgst_t_raw,
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    merged_position_for_tokenized_position,
    pre_token_prediction_positions,
    resolve_prompt_positions,
    run_forward_with_dgst_captures,
    run_forward_with_layer_hidden_captures,
)
from models.prompt_target import (
    extract_prompt_target_from_inputs,
    resolve_prompt_target_alignment,
)
from features.dgst_t import (
    compute_dgst_t_batch_from_captures,
    compute_four_gate_dgst_batch_from_captures,
)

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
        return int(self.cfg.get("num_visual_tokens", NUM_VISUAL_TOKENS))

    @property
    def model_label(self) -> str:
        return "LLaVA-1.5"

    @property
    def dgst_capture_device(self) -> Optional[str]:
        value = self.cfg.get("dgst_capture_device")
        return str(value) if value is not None else None

    @property
    def dgst_attention_query_chunk_size(self) -> Optional[int]:
        value = self.cfg.get("dgst_attention_query_chunk_size")
        return int(value) if value is not None else None

    def _format_prompt(self, raw_prompt: str) -> str:
        return _format_llava_prompt(raw_prompt)

    def _visual_grid_for_output(
        self,
        inputs: dict[str, Any],
        visual_start: int,
        visual_end: int,
    ) -> Optional[Tuple[int, int]]:
        del inputs, visual_start, visual_end
        return (24, 24)


    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        prompt = self._format_prompt(self.resolve_prompt(prompt))

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_new_tokens=self.generation_max_new_tokens,
                pad_token_id=pad_token_id,
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
        cfg_dgst_t: Optional[dict] = None,
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        """Extract one prefix position while retaining only requested tensors."""
        prompt_text = self._format_prompt(self.resolve_prompt(prompt))
        requirements_were_explicit = requirements is not None
        requirements = self.resolve_extraction_requirements(
            requirements,
            dgst_enabled=cfg_dgst_t is not None,
        )
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=prefix_token_ids,
            device=self.device,
            dtype=torch.float16,
        )
        input_ids = inputs["input_ids"]
        image_token_id = self._image_token_id()
        img_start, img_end = self._find_visual_token_range(inputs, image_token_id)

        use_dgst = bool(requirements.dgst_capture and cfg_dgst_t is not None)
        layer_outputs = None
        attention_query_positions = None
        if use_dgst:
            out, captures = run_forward_with_dgst_captures(
                self.model,
                output_hidden_states=False,
                retain_attention_updates=(
                    not _is_four_gate_mode(cfg_dgst_t)
                    or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
                ),
                attention_query_positions=[-1],
                capture_device=self.dgst_capture_device,
                attention_query_chunk_size=self.dgst_attention_query_chunk_size,
                # LlamaAttention always returns its native eager weights to
                # hooks.  Disable Transformers' second, full-matrix recorder
                # so only the requested rows remain after each native layer.
                record_model_attentions=False,
                **inputs,
            )
            attention_query_positions = captures[0]["attention_query_positions"]
        elif requirements.needs_hidden_states:
            captures = None
            out, layer_outputs = run_forward_with_layer_hidden_captures(
                self.model,
                output_attentions=requirements.needs_attention_weights,
                capture_all_layers=(
                    requirements.token_hidden_states
                    or requirements.patch_hidden_states
                ),
                **inputs,
            )
        else:
            captures = None
            with torch.no_grad():
                out = self.model(
                    **inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        expanded_seq_len = int(out.logits.shape[1])
        compact_profile = _is_compact_profile(cfg_dgst_t)
        keep_attention = (
            (not compact_profile or requirements_were_explicit)
            and requirements.attention is not AttentionRequirement.NONE
        )
        keep_hidden = (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        )
        if keep_attention:
            _require_attentions(out, model_name=self.model_label)
            text_to_patch_attn, text_to_text_attn = self._extract_attention_features(
                out.attentions,
                img_start,
                img_end,
                expanded_seq_len,
                attention_query_positions=attention_query_positions,
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
                    token_position=expanded_seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                    prompt_positions=prompt_positions_override,
                )
            elif layer_outputs is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                    layer_outputs,
                    token_position=expanded_seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                )
            elif out.hidden_states is not None:
                token_hidden_states, patch_hidden_states = self._extract_hidden_states(
                    out.hidden_states, img_start, img_end
                )
            else:
                raise RuntimeError("LLaVA hidden states were requested but not returned.")
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
        baseline_capture: dict[str, Any] = {
            "prediction_position": expanded_seq_len - 1,
            "visual_start": int(img_start),
            "visual_end": int(img_end),
            "attention_requirement": requirements.attention.value,
        }
        if requirements.response_hidden_states:
            # Use the actual merged decoder length.  Recent HF processors
            # already expand <image> to all visual IDs; older ones expose one
            # placeholder which the model expands internally.
            response_start = expanded_seq_len - len(prefix_token_ids)
            response_end = response_start + len(prefix_token_ids)
            response_hidden = final_normalized_hidden_slice(
                model=self.model,
                out=out,
                dgst_captures=captures,
                layer_outputs=layer_outputs,
                start=response_start,
                end=response_end,
            ).cpu()

        pred_token_id = int(out.logits[0, -1].argmax().item())
        pred_token_str = self.tokenizer.decode([pred_token_id], skip_special_tokens=False)
        last_logits = out.logits[0, -1].float().cpu() if requirements.logits else None
        out.logits = None
        dgst_target_id = int(target_token_id) if target_token_id is not None else int(pred_token_id)
        dgst_t_raw = None
        dgst_t_result = None
        if use_dgst:
            prompt_positions_override = self._resolve_dgst_prompt_support_positions(
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=image_token_id,
                visual_start=img_start,
                visual_end=img_end,
                cfg_dgst_t=cfg_dgst_t,
            )
            if _is_four_gate_mode(cfg_dgst_t):
                dgst_t_result = compute_four_gate_dgst_batch_from_captures(
                    model=self.model,
                    captures=captures,
                    visual_start=img_start,
                    visual_end=img_end,
                    target_token_ids=[dgst_target_id],
                    prediction_positions=[expanded_seq_len - 1],
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
                    image_token_id=image_token_id,
                    target_token_id=dgst_target_id,
                    prediction_position=expanded_seq_len - 1,
                    support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
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
                self._visual_grid_for_output(inputs, img_start, img_end)
                if requirements.visual_layout
                else None
            ),
            response_hidden_states=response_hidden,
            baseline_capture=baseline_capture,
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
        prompt_text = self._format_prompt(self.resolve_prompt(prompt))
        prefix_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prefix_inputs["input_ids"].shape[1])
        full_inputs = _append_prefix_token_ids(
            prefix_inputs,
            prefix_token_ids=response_ids,
            device=self.device,
            dtype=torch.float16,
        )
        input_ids = full_inputs["input_ids"]

        image_token_id = self._image_token_id()
        img_start, img_end = self._find_visual_token_range(
            full_inputs,
            image_token_id,
        )

        visual_token_count = int(img_end - img_start)
        full_prompt_positions = resolve_prompt_positions(
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
            prompt_positions=full_prompt_positions,
        )

        use_dgst = bool(requirements.dgst_capture and cfg_dgst_t is not None)
        layer_outputs = None
        attention_query_positions = None
        if use_dgst:
            out, captures = run_forward_with_dgst_captures(
                self.model,
                output_hidden_states=False,
                retain_attention_updates=(
                    not _is_four_gate_mode(cfg_dgst_t)
                    or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
                ),
                attention_query_positions=prediction_positions,
                capture_device=self.dgst_capture_device,
                attention_query_chunk_size=self.dgst_attention_query_chunk_size,
                record_model_attentions=False,
                **full_inputs,
            )
            attention_query_positions = captures[0]["attention_query_positions"]
        elif requirements.needs_hidden_states:
            captures = None
            out, layer_outputs = run_forward_with_layer_hidden_captures(
                self.model,
                output_attentions=requirements.needs_attention_weights,
                capture_all_layers=(
                    requirements.token_hidden_states
                    or requirements.patch_hidden_states
                ),
                **full_inputs,
            )
        else:
            captures = None
            with torch.no_grad():
                out = self.model(
                    **full_inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        expanded_seq_len = int(out.logits.shape[1])
        compact_profile = _is_compact_profile(cfg_dgst_t)
        keep_attention = (
            (not compact_profile or requirements_were_explicit)
            and requirements.attention is not AttentionRequirement.NONE
        )
        keep_hidden = (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        )
        if keep_attention:
            _require_attentions(out, model_name=self.model_label)
        if requirements.logits:
            position_logits: list[Optional[torch.Tensor]] = [
                out.logits[0, int(position)].float().cpu()
                for position in prediction_positions
            ]
            position_pred_ids = [int(logits.argmax().item()) for logits in position_logits]
        else:
            # Detection baselines only need the decoded prediction.  Keep the
            # vocabulary row on device and retain a single integer rather than
            # copying one full vocabulary vector per labelled object to CPU.
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
            response_start = expanded_seq_len - len(response_ids)
            shared_response_hidden = final_normalized_hidden_slice(
                model=self.model,
                out=out,
                dgst_captures=captures,
                layer_outputs=layer_outputs,
                start=response_start,
                end=response_start + len(response_ids),
            ).cpu()
            all_prediction_positions = list(
                range(response_start - 1, response_start - 1 + len(response_ids))
            )
            response_logits = (
                out.logits[0].index_select(
                    0,
                    torch.tensor(
                        all_prediction_positions,
                        dtype=torch.long,
                        device=out.logits.device,
                    ),
                )
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
        # All vocabulary rows have now either been reduced to prediction IDs,
        # requested CPU rows, or compact MetaToken statistics.
        out.logits = None
        if not keep_attention:
            # DGST captures own the only remaining attention references and
            # will release them one layer at a time below.
            out.attentions = None

        dgst_results = None
        if use_dgst:
            support_prompt_positions = self._resolve_dgst_prompt_support_positions(
                full_input_ids=input_ids[0].tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=image_token_id,
                visual_start=img_start,
                visual_end=img_end,
                cfg_dgst_t=cfg_dgst_t,
            )
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
                text_to_patch_attn, text_to_text_attn = self._extract_attention_features_at_position(
                    out.attentions,
                    img_start,
                    img_end,
                    expanded_seq_len,
                    int(prediction_position),
                    attention_query_positions=attention_query_positions,
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
                    token_hidden_states, patch_hidden_states = (
                        self._extract_hidden_states_at_position(
                            out.hidden_states,
                            img_start,
                            img_end,
                            int(prediction_position),
                        )
                    )
                else:
                    raise RuntimeError("LLaVA hidden states were requested but not returned.")
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
                        self._visual_grid_for_output(full_inputs, img_start, img_end)
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

    def extract_prompt_target_features(
        self,
        image: Image.Image,
        request: PromptTargetRequest,
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        prompt_text = self._format_prompt(request.prompt)
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=[],
            device=self.device,
            dtype=torch.float16,
        )
        input_ids = inputs["input_ids"][0]
        image_token_id = self._image_token_id()
        visual_start, visual_end = self._find_visual_token_range(
            inputs,
            image_token_id,
        )
        alignment = resolve_prompt_target_alignment(
            tokenizer=self.tokenizer,
            full_input_ids=input_ids.tolist(),
            request=request,
            image_token_id=image_token_id,
            visual_token_count=visual_end - visual_start,
        )
        return extract_prompt_target_from_inputs(
            wrapper=self,
            inputs=inputs,
            full_input_ids=input_ids.tolist(),
            alignment=alignment,
            visual_start=visual_start,
            visual_end=visual_end,
            image_token_id=image_token_id,
            visual_grid=self._visual_grid_for_output(
                inputs,
                visual_start,
                visual_end,
            ),
            cfg_dgst_t=cfg_dgst_t,
            requirements=requirements,
            model_name=self.model_label,
            support_scope=self.cfg.get("dgst_t_support_scope", "visual_prompt"),
        )


    def _find_img_range_from_embeds(self, inputs: dict) -> Tuple[int, int]:
        """Fallback: estimate img_start by counting non-image prompt tokens."""
        del inputs
        return 4, 4 + self.num_visual_tokens

    def _image_token_id(self) -> int:
        return int(
            getattr(self.model.config, "image_token_index", IMAGE_TOKEN_INDEX)
        )

    def _find_visual_token_range(
        self,
        inputs: dict[str, Any],
        image_token_id: int,
    ) -> Tuple[int, int]:
        """Locate one contiguous visual span in processor-expanded input IDs.

        Recent Transformers processors expand ``<image>`` to the exact number
        of decoder-side visual embeddings.  Older LLaVA-1.5 processors leave a
        single placeholder which the model expands internally, so retain the
        configured fixed-size fallback for that case.
        """

        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
            raise ValueError("LLaVA inputs require rank-2 input_ids")
        image_positions = (input_ids[0] == int(image_token_id)).nonzero(
            as_tuple=True
        )[0]
        count = int(image_positions.numel())
        if count == 0:
            return self._find_img_range_from_embeds(inputs)
        start = int(image_positions[0].item())
        if count == 1:
            return start, start + self.num_visual_tokens
        expected = torch.arange(
            start,
            start + count,
            dtype=image_positions.dtype,
            device=image_positions.device,
        )
        if not torch.equal(image_positions, expected):
            raise ValueError(
                "LLaVA wrapper supports one contiguous image-token span; "
                f"found positions={image_positions.detach().cpu().tolist()[:16]}"
            )
        return start, start + count

    def _resolve_dgst_prompt_support_positions(
        self,
        *,
        full_input_ids: Sequence[int],
        prompt_tokenized_length: int,
        image_token_id: int,
        visual_start: int,
        visual_end: int,
        cfg_dgst_t: Optional[dict],
    ) -> list[int] | None:
        if cfg_dgst_t is None:
            return None
        mode = str(
            cfg_dgst_t.get(
                "dgst_t_prompt_support_mode",
                cfg_dgst_t.get("prompt_support_mode", "full"),
            )
        ).strip().lower()
        if mode in {"full", "all", "template"}:
            return None
        if mode not in {"user_text", "user", "semantic"}:
            raise ValueError(
                "dgst_t_prompt_support_mode must be 'full' or 'user_text', "
                f"got {mode!r}."
            )

        user_text = str(
            cfg_dgst_t.get(
                "dgst_t_user_prompt_text",
                cfg_dgst_t.get("user_prompt_text", "Describe this image."),
            )
        )
        prompt_ids = [int(token_id) for token_id in full_input_ids[: int(prompt_tokenized_length)]]
        span = self._find_user_text_token_span(
            prompt_ids=prompt_ids,
            user_text=user_text,
            image_token_id=int(image_token_id),
        )
        visual_count = int(visual_end) - int(visual_start)
        return [
            int(
                merged_position_for_tokenized_position(
                    full_input_ids=full_input_ids,
                    tokenized_position=tokenized_position,
                    image_token_id=int(image_token_id),
                    visual_token_count=visual_count,
                )
            )
            for tokenized_position in range(span[0], span[1])
        ]

    def _find_user_text_token_span(
        self,
        *,
        prompt_ids: Sequence[int],
        user_text: str,
        image_token_id: int,
    ) -> tuple[int, int]:
        target = _normalize_prompt_text(user_text)
        if not target:
            raise ValueError("dgst_t_user_prompt_text must be non-empty in user_text mode.")
        matches: list[tuple[int, int]] = []
        max_span = min(32, int(len(prompt_ids)))
        for start in range(int(len(prompt_ids))):
            for end in range(start + 1, min(int(len(prompt_ids)), start + max_span) + 1):
                span_ids = [int(token_id) for token_id in prompt_ids[start:end]]
                if int(image_token_id) in span_ids:
                    continue
                decoded = self.tokenizer.decode(span_ids, skip_special_tokens=False)
                if _normalize_prompt_text(decoded) == target:
                    matches.append((start, end))
        if not matches:
            decoded_prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            raise ValueError(
                "Could not locate dgst_t_user_prompt_text in LLaVA prompt tokens: "
                f"{user_text!r}. Decoded prompt={decoded_prompt!r}"
            )
        return min(matches, key=lambda item: (item[1] - item[0], item[0]))

    @staticmethod
    def _extract_attention_features(
        attentions: tuple,
        img_start: int,
        img_end: int,
        seq_len: int,
        attention_query_positions: Optional[Sequence[int]] = None,
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
        row_index = attention_row_index(
            prediction_position=last_pos,
            attention_query_positions=attention_query_positions,
        )
        for layer_attn in attentions:
            text_idx_tensor = text_idx_tensor.to(layer_attn.device)
            row = layer_attn[0, :, row_index, :]
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
        attention_query_positions: Optional[Sequence[int]] = None,
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
        row_index = attention_row_index(
            prediction_position=last_pos,
            attention_query_positions=attention_query_positions,
        )
        for layer_attn in attentions:
            row = layer_attn[0, :, row_index, :]
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

    @staticmethod
    def _extract_hidden_states_at_position(
        hidden_states: tuple,
        img_start: int,
        img_end: int,
        token_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_list, patch_list = [], []
        for hs in hidden_states[1:]:
            token_list.append(hs[0, int(token_position), :])
            patch_list.append(hs[0, img_start:img_end, :])
        return torch.stack(token_list, dim=0), torch.stack(patch_list, dim=0)


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


def _append_prefix_token_ids(
    prompt_inputs: Any,
    *,
    prefix_token_ids: Sequence[int],
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs = _to_device_dtype(dict(prompt_inputs), device, dtype)
    prefix = torch.tensor(
        [int(token_id) for token_id in prefix_token_ids],
        dtype=inputs["input_ids"].dtype,
        device=inputs["input_ids"].device,
    ).unsqueeze(0)
    if prefix.numel() == 0:
        return inputs
    inputs["input_ids"] = torch.cat((inputs["input_ids"], prefix), dim=1)
    if "attention_mask" in inputs:
        suffix_mask = torch.ones(
            (inputs["attention_mask"].shape[0], prefix.shape[1]),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        inputs["attention_mask"] = torch.cat(
            (inputs["attention_mask"], suffix_mask), dim=1
        )
    for stale_key in ("position_ids", "cache_position"):
        inputs.pop(stale_key, None)
    return inputs


def _format_llava_prompt(raw_prompt: str) -> str:
    """Embed a raw instruction in the LLaVA-1.5 conversation template."""
    prompt = str(raw_prompt).strip()
    if "<image>" in prompt and "ASSISTANT:" in prompt:
        return prompt
    return f"USER: <image>\n{prompt}\nASSISTANT:"


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


def _normalize_prompt_text(text: str) -> str:
    return " ".join(str(text).strip().split())
