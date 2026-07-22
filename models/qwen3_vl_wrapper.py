"""Independent Qwen3-VL wrapper for generation and feature extraction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from features.dgst_t import compute_dgst_t_batch_from_captures
from models.base_wrapper import (
    AttentionRequirement,
    BaseLVLMWrapper,
    ExtractionRequirements,
    GenerationOutput,
    ModelOutput,
    PromptTargetRequest,
    compact_response_logit_statistics,
    configure_image_processor_limits,
)
from models.dgst_capture import (
    final_normalized_hidden_slice,
    hidden_states_from_captures,
    hidden_states_from_layer_outputs,
    run_forward_with_dgst_captures,
    run_forward_with_layer_hidden_captures,
)
from models.prompt_support import resolve_prompt_support_positions
from models.prompt_target import (
    extract_prompt_target_from_inputs,
    resolve_prompt_target_alignment,
)


class Qwen3VLWrapper(BaseLVLMWrapper):
    """Wrapper for Qwen3-VL-8B-Instruct.

    This implementation deliberately remains independent of ``QwenVLWrapper``.
    Qwen3 has its own model class and chat-template path, while exposing the
    same token-detector interface.
    """

    def _load_model(self) -> None:
        hf_name = self.cfg["hf_name"]
        print(f"[Qwen3VLWrapper] Loading model from {hf_name} ...")
        self.processor = AutoProcessor.from_pretrained(
            hf_name,
            trust_remote_code=True,
        )
        configure_image_processor_limits(self.processor, self.cfg)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="eager",
        )
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self._image_token_id = int(
            getattr(self.model.config, "image_token_id", None)
            or self.tokenizer.convert_tokens_to_ids(
                self.cfg.get("image_token", "<|image_pad|>")
            )
        )
        self._vision_start_id = int(
            getattr(self.model.config, "vision_start_token_id", None)
            or self.tokenizer.convert_tokens_to_ids(
                self.cfg.get("vision_start_token", "<|vision_start|>")
            )
        )
        self._vision_end_id = int(
            getattr(self.model.config, "vision_end_token_id", None)
            or self.tokenizer.convert_tokens_to_ids(
                self.cfg.get("vision_end_token", "<|vision_end|>")
            )
        )
        self._last_num_visual_tokens: Optional[int] = None
        print(
            "[Qwen3VLWrapper] Loaded. "
            f"image_token={self._image_token_id}, "
            f"vision_start={self._vision_start_id}, "
            f"vision_end={self._vision_end_id}"
        )

    @property
    def num_layers(self) -> int:
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            return int(text_config.num_hidden_layers)
        return int(self.cfg.get("num_layers", 36))

    @property
    def num_visual_tokens(self) -> int:
        configured = self.cfg.get("num_visual_tokens")
        if configured is not None:
            return int(configured)
        if self._last_num_visual_tokens is not None:
            return int(self._last_num_visual_tokens)
        raise RuntimeError(
            "Qwen3-VL uses a dynamic visual-token count. Process an image first "
            "or read ModelOutput.visual_grid instead of assuming 256 tokens."
        )

    def _prompt_inputs(self, image: Image.Image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        prompt = self.resolve_prompt(prompt)
        inputs = self._prompt_inputs(image, prompt).to(self.device)
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
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        prompt = self.resolve_prompt(prompt)
        requirements_were_explicit = requirements is not None
        requirements = self.resolve_extraction_requirements(
            requirements,
            dgst_enabled=cfg_dgst_t is not None,
        )
        prompt_inputs = self._prompt_inputs(image, prompt)
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=prefix_token_ids,
            device=self.device,
        )
        input_ids = inputs["input_ids"][0]
        visual_start, visual_end = self._find_vision_token_range(input_ids)
        visual_grid = self._resolve_visual_grid(inputs, visual_end - visual_start)

        use_dgst = bool(requirements.dgst_capture and cfg_dgst_t is not None)
        layer_outputs = None
        if use_dgst:
            out, captures = run_forward_with_dgst_captures(
                self.model,
                output_hidden_states=False,
                retain_attention_updates=(
                    str(cfg_dgst_t.get("target_gate_mode", "four_gate"))
                    .strip()
                    .lower()
                    != "four_gate"
                    or bool(cfg_dgst_t.get("compute_ffn_injection_features", False))
                ),
                **inputs,
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
                **inputs,
            )
        else:
            captures = None
            with torch.inference_mode():
                out = self.model(
                    **inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        seq_len = int(input_ids.shape[0])
        if out.attentions is not None and len(out.attentions) > 0:
            seq_len = int(out.attentions[0].shape[-1])
        compact_profile = bool(
            cfg_dgst_t is not None
            and cfg_dgst_t.get("feature_output_profile")
            in {"costvariant_vv", "gate_comparison_vv", "four_gate_vv"}
        )

        if (
            (not compact_profile or requirements_were_explicit)
            and requirements.attention is not AttentionRequirement.NONE
        ):
            if out.attentions is None or not out.attentions:
                raise RuntimeError(
                    "Qwen3-VL attention features were requested but no attention "
                    "weights were returned; eager attention is required."
                )
            text_to_patch_attn, text_to_text_attn = _extract_attention_features(
                out.attentions,
                visual_start,
                visual_end,
                seq_len,
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

        if (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        ):
            if captures is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                    captures,
                    token_position=seq_len - 1,
                    visual_start=visual_start,
                    visual_end=visual_end,
                )
            elif layer_outputs is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                    layer_outputs,
                    token_position=seq_len - 1,
                    visual_start=visual_start,
                    visual_end=visual_end,
                )
            elif out.hidden_states is not None:
                token_hidden_states, patch_hidden_states = _extract_hidden_states(
                    out.hidden_states,
                    visual_start,
                    visual_end,
                )
            else:
                raise RuntimeError("Qwen3-VL hidden states were requested but not returned.")
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
                end=int(input_ids.shape[0]),
            ).cpu()

        pred_token_id = int(out.logits[0, -1].argmax().item())
        pred_token_str = self.tokenizer.decode(
            [pred_token_id],
            skip_special_tokens=False,
        )
        token_logits = out.logits[0, -1].float().cpu() if requirements.logits else None
        out.logits = None
        dgst_target_id = (
            int(target_token_id) if target_token_id is not None else pred_token_id
        )
        dgst_result = None
        if use_dgst:
            prompt_positions = resolve_prompt_support_positions(
                tokenizer=self.tokenizer,
                full_input_ids=input_ids.tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                image_token_id=self._image_token_id,
                visual_start=visual_start,
                visual_end=visual_end,
                cfg_dgst_t=cfg_dgst_t,
                model_name="Qwen3-VL",
            )
            results = compute_dgst_t_batch_from_captures(
                model=self.model,
                full_input_ids=input_ids.tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                captures=captures,
                visual_start=visual_start,
                visual_end=visual_end,
                image_token_id=self._image_token_id,
                target_token_ids=[dgst_target_id],
                prediction_positions=[seq_len - 1],
                support_scope=self.cfg.get("dgst_t_support_scope", "visual"),
                semantic_chunk_size=int(cfg_dgst_t.get("semantic_chunk_size", 64)),
                prompt_positions_override=prompt_positions,
                tau=float(cfg_dgst_t.get("tau", 0.07)),
                source_distribution_mode=cfg_dgst_t.get("source_distribution_mode", "softmax"),
                transport_top_k=int(cfg_dgst_t.get("transport_top_k", 64)),
                cost_mode=cfg_dgst_t.get("cost_mode", "direct"),
                lambda_d=float(cfg_dgst_t.get("lambda_d", 1.0)),
                lambda_s=float(cfg_dgst_t.get("lambda_s", 1.0)),
                lambda_t=float(cfg_dgst_t.get("lambda_t", 1.0)),
                lambda_int=float(cfg_dgst_t.get("lambda_int", 1.0)),
                baseline_layers=int(cfg_dgst_t.get("baseline_layers", 10)),
                risk_start_layer=int(cfg_dgst_t.get("risk_start_layer", 15)),
                alpha=float(cfg_dgst_t.get("alpha", 2.0)),
                ot_solver=cfg_dgst_t.get("ot_solver", "linprog"),
                atarget_visual_top_k=int(cfg_dgst_t.get("atarget_visual_top_k", 32)),
                topmass_alpha=float(cfg_dgst_t.get("topmass_085_alpha", 0.85)),
                capped_topmass_alpha=float(cfg_dgst_t.get("capped_topmass_085_alpha", 0.85)),
                capped_topmass_min_k=int(cfg_dgst_t.get("capped_topmass_085_min_k", 32)),
                capped_topmass_max_k=int(cfg_dgst_t.get("capped_topmass_085_max_k", 64)),
                compute_topmass_085=bool(cfg_dgst_t.get("compute_topmass_085", True)),
                compute_capped_topmass_085=bool(cfg_dgst_t.get("compute_capped_topmass_085", True)),
                target_gate_mode=cfg_dgst_t.get("target_gate_mode", "four_gate"),
                relative_vll_mad_epsilon=float(cfg_dgst_t.get("relative_vll_mad_epsilon", 1e-6)),
                relative_vll_logit_source=cfg_dgst_t.get("relative_vll_logit_source", "h_mid"),
                relative_cost_mode=cfg_dgst_t.get("relative_cost_mode"),
                relative_cost_modes=cfg_dgst_t.get("relative_cost_modes"),
                relative_cost_state_modes=cfg_dgst_t.get("relative_cost_state_modes"),
                relative_cost_update_lambdas=cfg_dgst_t.get("relative_cost_update_lambdas"),
                relative_barrier_lambda=float(cfg_dgst_t.get("relative_barrier_lambda", 1.0)),
                relative_barrier_margin=float(cfg_dgst_t.get("relative_barrier_margin", 0.5)),
                relative_barrier_max=float(cfg_dgst_t.get("relative_barrier_max", 3.0)),
                source_modes=cfg_dgst_t.get("source_modes"),
                target_attention_gammas=cfg_dgst_t.get("target_attention_gammas"),
                target_attention_epsilon=float(cfg_dgst_t.get("target_attention_epsilon", 1e-12)),
                compute_ffn_injection_features=bool(cfg_dgst_t.get("compute_ffn_injection_features", False)),
                ffn_injection_evidence_top_k=int(cfg_dgst_t.get("ffn_injection_evidence_top_k", 32)),
                ffn_injection_evidence_rank=int(cfg_dgst_t.get("ffn_injection_evidence_rank", 8)),
                ffn_injection_eps=float(cfg_dgst_t.get("ffn_injection_eps", 1e-12)),
                compute_dual_scope=bool(
                    cfg_dgst_t.get(
                        "dgst_t_dual_scope",
                        cfg_dgst_t.get("compute_dual_scope", False),
                    )
                ),
                four_gate_methods=cfg_dgst_t.get("four_gate_methods"),
                four_gate_cost_modes=cfg_dgst_t.get("cost_modes"),
                four_gate_support_modes=cfg_dgst_t.get("support_modes"),
                compute_prompt_cafe=bool(
                    cfg_dgst_t.get("compute_prompt_cafe", False)
                ),
                prompt_cafe_temperature=float(
                    cfg_dgst_t.get("prompt_cafe_temperature", 10.0)
                ),
                prompt_cafe_layer=int(cfg_dgst_t.get("prompt_cafe_layer", 22)),
                release_layer_captures=True,
            )
            if len(results) != 1:
                raise RuntimeError(f"Expected one DGST result, received {len(results)}.")
            dgst_result = results[0]

        return ModelOutput(
            token_id=pred_token_id,
            token_str=pred_token_str,
            text_to_patch_attn=text_to_patch_attn,
            text_to_text_attn=text_to_text_attn,
            token_hidden_states=token_hidden_states,
            patch_hidden_states=patch_hidden_states,
            response_token_idx=int(response_token_idx),
            token_logits=token_logits,
            dgst_t_raw=None,
            dgst_t_result=dgst_result,
            visual_grid=visual_grid if requirements.visual_layout else None,
            response_hidden_states=response_hidden,
            baseline_capture={
                "prediction_position": int(seq_len - 1),
                "visual_start": int(visual_start),
                "visual_end": int(visual_end),
                "attention_requirement": requirements.attention.value,
            },
        )

    def extract_token_features_batch(
        self,
        image: Image.Image,
        response_token_ids: Sequence[int],
        response_token_indices: Sequence[int],
        target_token_ids: Optional[Sequence[int]] = None,
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> List[ModelOutput]:
        """Sequential-prefix extraction keeps full attentions within GPU limits."""
        response_ids, requested_indices, targets = self.validate_causal_batch_request(
            response_token_ids=response_token_ids,
            response_token_indices=response_token_indices,
            target_token_ids=target_token_ids,
        )
        if not requested_indices:
            return []

        per_prefix_requirements = requirements
        if requirements is not None and requirements.response_hidden_states:
            per_prefix_requirements = replace(
                requirements,
                response_hidden_states=False,
            )

        outputs = []
        for response_index, target_token in zip(requested_indices, targets):
            outputs.append(
                self.extract_token_features(
                    image=image,
                    prefix_token_ids=response_ids[:response_index],
                    response_token_idx=response_index,
                    target_token_id=target_token,
                    cfg_dgst_t=cfg_dgst_t,
                    prompt=prompt,
                    requirements=per_prefix_requirements,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if requirements is not None and requirements.response_hidden_states:
            shared_capture = self._extract_full_response_baseline_capture(
                image=image,
                response_token_ids=response_ids,
                prompt=prompt,
            )
            for output in outputs:
                output.response_hidden_states = shared_capture[
                    "response_hidden_states"
                ]
                output.baseline_capture = {
                    **(output.baseline_capture or {}),
                    **shared_capture["statistics"],
                }
        return outputs

    def extract_prompt_target_features(
        self,
        image: Image.Image,
        request: PromptTargetRequest,
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        inputs = self._prompt_inputs(image, request.prompt).to(self.device)
        input_ids = inputs["input_ids"][0]
        visual_start, visual_end = self._find_vision_token_range(input_ids)
        alignment = resolve_prompt_target_alignment(
            tokenizer=self.tokenizer,
            full_input_ids=input_ids.tolist(),
            request=request,
            image_token_id=self._image_token_id,
            visual_token_count=visual_end - visual_start,
        )
        return extract_prompt_target_from_inputs(
            wrapper=self,
            inputs=inputs,
            full_input_ids=input_ids.tolist(),
            alignment=alignment,
            visual_start=visual_start,
            visual_end=visual_end,
            image_token_id=self._image_token_id,
            visual_grid=self._resolve_visual_grid(inputs, visual_end - visual_start),
            cfg_dgst_t=cfg_dgst_t,
            requirements=requirements,
            model_name="Qwen3-VL",
            support_scope=self.cfg.get("dgst_t_support_scope", "visual"),
        )

    def _extract_full_response_baseline_capture(
        self,
        *,
        image: Image.Image,
        response_token_ids: Sequence[int],
        prompt: Optional[str],
    ) -> dict[str, Any]:
        prompt_inputs = self._prompt_inputs(image, self.resolve_prompt(prompt))
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=response_token_ids,
            device=self.device,
        )
        response_count = len(response_token_ids)
        if response_count == 0:
            return {
                "response_hidden_states": torch.empty((0, 0)),
                "statistics": compact_response_logit_statistics(
                    torch.empty((0, 0)),
                    response_token_ids=[],
                ),
            }
        logits_positions = torch.arange(
            prompt_length - 1,
            prompt_length + response_count - 1,
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        with torch.inference_mode():
            out = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
                logits_to_keep=logits_positions,
            )
        response_hidden = out.hidden_states[-1][
            0,
            prompt_length:prompt_length + response_count,
            :,
        ].detach().cpu()
        teacher_logits = out.logits[0]
        if int(teacher_logits.shape[0]) != response_count:
            teacher_logits = teacher_logits.index_select(
                0,
                logits_positions.to(teacher_logits.device),
            )
        statistics = compact_response_logit_statistics(
            teacher_logits,
            response_token_ids=response_token_ids,
        )
        del teacher_logits, out
        return {
            "response_hidden_states": response_hidden,
            "statistics": statistics,
        }

    def _find_vision_token_range(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        ids = [int(token_id) for token_id in input_ids.tolist()]
        positions = [
            index for index, token_id in enumerate(ids)
            if token_id == self._image_token_id
        ]
        if not positions:
            raise ValueError(
                "Qwen3-VL processor output contains no <|image_pad|> tokens; "
                "cannot align visual attention safely."
            )
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("Qwen3-VL image-pad positions are not contiguous.")
        self._last_num_visual_tokens = len(positions)
        return int(positions[0]), int(positions[-1] + 1)

    def _resolve_visual_grid(
        self,
        inputs: dict[str, Any],
        visual_token_count: int,
    ) -> Optional[Tuple[int, int]]:
        grid = inputs.get("image_grid_thw")
        if grid is None or int(grid.shape[0]) != 1:
            return None
        merge_size = int(getattr(self.processor.image_processor, "merge_size", 1))
        height = int(grid[0, 1].item()) // merge_size
        width = int(grid[0, 2].item()) // merge_size
        if height * width != int(visual_token_count):
            raise ValueError(
                "Qwen3-VL visual grid does not match image-pad token count: "
                f"grid={height}x{width}, tokens={visual_token_count}."
            )
        return height, width


def _append_prefix_token_ids(
    prompt_inputs: Any,
    *,
    prefix_token_ids: Sequence[int],
    device: str,
) -> dict[str, Any]:
    inputs = {
        key: value.to(device=device) if torch.is_tensor(value) else value
        for key, value in dict(prompt_inputs).items()
    }
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
            (inputs["attention_mask"], suffix_mask),
            dim=1,
        )
    for stale_key in ("position_ids", "cache_position", "rope_deltas"):
        inputs.pop(stale_key, None)
    return inputs


def _extract_attention_features(
    attentions: tuple,
    visual_start: int,
    visual_end: int,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    visual_set = set(range(int(visual_start), int(visual_end)))
    last_position = int(seq_len) - 1
    text_positions = [
        position
        for position in range(int(seq_len))
        if position not in visual_set and position != last_position
    ]
    patch_layers, text_layers = [], []
    for layer_attention in attentions:
        row = layer_attention[0, :, last_position, :]
        text_index = torch.tensor(
            text_positions,
            dtype=torch.long,
            device=row.device,
        )
        patch_layers.append(row[:, visual_start:visual_end])
        text_layers.append(row[:, text_index])
    return torch.stack(patch_layers, dim=0), torch.stack(text_layers, dim=0)


def _extract_hidden_states(
    hidden_states: tuple,
    visual_start: int,
    visual_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_states, patch_states = [], []
    for hidden in hidden_states[1:]:
        token_states.append(hidden[0, -1, :])
        patch_states.append(hidden[0, visual_start:visual_end, :])
    return torch.stack(token_states, dim=0), torch.stack(patch_states, dim=0)
