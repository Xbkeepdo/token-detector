"""LLaVA-OneVision 1.5 wrapper for generation and DGST-T feature extraction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
        configure_image_processor_limits(self.processor, self.cfg)

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
        self._last_num_visual_tokens: Optional[int] = None
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
        configured = self.cfg.get("num_visual_tokens")
        if configured is not None:
            return int(configured)
        if self._last_num_visual_tokens is not None:
            return int(self._last_num_visual_tokens)
        raise RuntimeError(
            "LLaVA-OneVision uses a dynamic visual-token count. Process an image "
            "first or read ModelOutput.visual_grid instead of assuming 144 tokens."
        )

    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        prompt = self.resolve_prompt(prompt)

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
                temperature=self.cfg.get("temperature", 0.1),
                top_p=self.cfg.get("top_p", 0.5),
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
        prompt: Optional[str] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        prompt = self.resolve_prompt(prompt)
        requirements_were_explicit = requirements is not None
        requirements = self.resolve_extraction_requirements(
            requirements,
            dgst_enabled=cfg_dgst_t is not None,
        )
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
        prompt_inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        prompt_tokenized_length = int(prompt_inputs["input_ids"].shape[1])
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=prefix_token_ids,
            device=self.device,
        )

        input_ids = inputs["input_ids"][0]
        img_start, img_end = self._find_vision_token_range(input_ids)
        visual_grid = self._resolve_visual_grid(inputs, img_end - img_start)

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
            with torch.no_grad():
                out = self.model(
                    **inputs,
                    output_attentions=requirements.needs_attention_weights,
                    output_hidden_states=requirements.needs_hidden_states,
                    return_dict=True,
                    use_cache=False,
                )

        expanded_seq_len = int(input_ids.shape[0])
        if out.attentions is not None and len(out.attentions) > 0:
            expanded_seq_len = int(out.attentions[0].shape[-1])
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
                    "OneVision attention features were requested but no attention "
                    "weights were returned; eager attention is required."
                )
            text_to_patch_attn, text_to_text_attn = _extract_attention_features(
                out.attentions,
                img_start,
                img_end,
                expanded_seq_len,
            )
            if requirements.attention is AttentionRequirement.HEAD_MEAN:
                text_to_patch_attn = text_to_patch_attn.mean(dim=1, keepdim=True)
                text_to_text_attn = text_to_text_attn.mean(dim=1, keepdim=True)
        else:
            text_to_patch_attn = torch.empty(0)
            text_to_text_attn = torch.empty(0)
        # The model output keeps the complete per-layer attention tuple.  The
        # compact matrices above are the only downstream attention consumer,
        # so move them off device and drop the duplicate ModelOutput reference
        # before the layer-wise DGST reduction starts.
        text_to_patch_attn = text_to_patch_attn.cpu()
        text_to_text_attn = text_to_text_attn.cpu()
        out.attentions = None

        if (not compact_profile or requirements_were_explicit) and (
            requirements.token_hidden_states or requirements.patch_hidden_states
        ):
            if captures is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_captures(
                    captures,
                    token_position=expanded_seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                )
            elif layer_outputs is not None:
                token_hidden_states, patch_hidden_states = hidden_states_from_layer_outputs(
                    layer_outputs,
                    token_position=expanded_seq_len - 1,
                    visual_start=img_start,
                    visual_end=img_end,
                )
            elif out.hidden_states is not None:
                token_hidden_states, patch_hidden_states = _extract_hidden_states(
                    out.hidden_states,
                    img_start,
                    img_end,
                )
            else:
                raise RuntimeError("OneVision hidden states were requested but not returned.")
            if not requirements.token_hidden_states:
                token_hidden_states = torch.empty(0, device=patch_hidden_states.device)
            if not requirements.patch_hidden_states:
                patch_hidden_states = torch.empty(0, device=token_hidden_states.device)
        else:
            token_hidden_states = torch.empty(0)
            patch_hidden_states = torch.empty(0)
        # In all mode this stack is roughly 100 MiB for a normal COCO image.
        # It has already been fully materialized for ADS/ProjectAway, so keep
        # the reusable copy on CPU while DGST consumes and releases captures.
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
        last_logits = out.logits[0, -1].float().cpu() if requirements.logits else None
        # No active method consumes the model's full [sequence,vocabulary]
        # output after the prediction row has been compacted.
        out.logits = None
        dgst_target_id = (
            int(target_token_id) if target_token_id is not None else int(pred_token_id)
        )
        dgst_t_result = None
        if use_dgst:
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
            dgst_t_result = _compute_dgst_result_from_captures(
                model=self.model,
                input_ids=input_ids.tolist(),
                prompt_tokenized_length=prompt_tokenized_length,
                captures=captures,
                visual_start=img_start,
                visual_end=img_end,
                image_token_id=self._image_token_id,
                target_token_id=dgst_target_id,
                prediction_position=expanded_seq_len - 1,
                support_scope=self.cfg.get("dgst_t_support_scope", "visual"),
                prompt_positions_override=prompt_positions_override,
                cfg={
                    **cfg_dgst_t,
                    "semantic_chunk_size": int(
                        cfg_dgst_t.get(
                            "semantic_chunk_size",
                            self.cfg.get("semantic_chunk_size", 8),
                        )
                    ),
                },
                release_layer_captures=True,
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
            dgst_t_raw=None,
            dgst_t_result=dgst_t_result,
            visual_grid=visual_grid if requirements.visual_layout else None,
            response_hidden_states=response_hidden,
            baseline_capture={
                "prediction_position": int(expanded_seq_len - 1),
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
        response_ids, requested_indices, targets = self.validate_causal_batch_request(
            response_token_ids=response_token_ids,
            response_token_indices=response_token_indices,
            target_token_ids=target_token_ids,
        )
        if not requested_indices:
            return []

        # Sequential prefix passes do not need a response-caption stack: one
        # lightweight full-caption pass below supplies the shared MetaToken /
        # HalLoc capture.  Avoiding it here also allows each DGST capture to be
        # released layer-by-layer.
        per_prefix_requirements = requirements
        if requirements is not None and requirements.response_hidden_states:
            per_prefix_requirements = replace(
                requirements,
                response_hidden_states=False,
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
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": request.prompt},
            ],
        }]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_inputs = self.processor(
            text=[rendered],
            images=[image],
            return_tensors="pt",
        )
        inputs = _append_prefix_token_ids(
            prompt_inputs,
            prefix_token_ids=[],
            device=self.device,
        )
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
            model_name="LLaVA-OneVision-1.5",
            support_scope=self.cfg.get("dgst_t_support_scope", "visual"),
            semantic_chunk_size_default=int(self.cfg.get("semantic_chunk_size", 8)),
        )

    def _extract_full_response_baseline_capture(
        self,
        *,
        image: Image.Image,
        response_token_ids: Sequence[int],
        prompt: Optional[str],
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.resolve_prompt(prompt)},
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
        with torch.no_grad():
            out = self.model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        response_hidden = out.hidden_states[-1][
            0,
            prompt_length:prompt_length + response_count,
            :,
        ].detach().cpu()
        # The local OneVision conditional forward does not accept Qwen's
        # ``logits_to_keep`` argument.  Slice the full output immediately and
        # retain only the response prediction rows.
        teacher_logits = out.logits[0].index_select(
            0,
            logits_positions.to(out.logits.device),
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
        image_positions = [
            index for index, token_id in enumerate(ids) if token_id == self._image_token_id
        ]
        if image_positions:
            if image_positions != list(range(image_positions[0], image_positions[-1] + 1)):
                raise ValueError("OneVision image-pad positions are not contiguous.")
            self._last_num_visual_tokens = len(image_positions)
            return int(image_positions[0]), int(image_positions[-1] + 1)
        raise ValueError(
            "OneVision processor output contains no <|image_pad|> tokens; "
            "cannot align visual attention safely."
        )

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
                "OneVision visual grid does not match image-pad token count: "
                f"grid={height}x{width}, tokens={visual_token_count}."
            )
        return height, width


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


def _compute_dgst_result_from_captures(
    *,
    model: Any,
    input_ids: Sequence[int],
    prompt_tokenized_length: int,
    captures: Sequence[dict[str, Any]],
    visual_start: int,
    visual_end: int,
    image_token_id: int,
    target_token_id: int,
    prediction_position: int,
    support_scope: str,
    prompt_positions_override: Optional[Sequence[int]],
    cfg: dict[str, Any],
    release_layer_captures: bool = False,
) -> dict[str, Any]:
    results = compute_dgst_t_batch_from_captures(
        model=model,
        full_input_ids=input_ids,
        prompt_tokenized_length=int(prompt_tokenized_length),
        captures=captures,
        visual_start=int(visual_start),
        visual_end=int(visual_end),
        image_token_id=int(image_token_id),
        target_token_ids=[int(target_token_id)],
        prediction_positions=[int(prediction_position)],
        support_scope=str(support_scope),
        # This checkpoint leaves very little headroom on a 32 GiB card after
        # eager-attention capture; eight rows keeps the transient [rows,V]
        # projection safely bounded.
        semantic_chunk_size=int(cfg.get("semantic_chunk_size", 8)),
        prompt_positions_override=prompt_positions_override,
        tau=float(cfg.get("tau", 0.07)),
        source_distribution_mode=cfg.get("source_distribution_mode", "softmax"),
        transport_top_k=int(cfg.get("transport_top_k", 64)),
        cost_mode=cfg.get("cost_mode", "direct"),
        lambda_d=float(cfg.get("lambda_d", 1.0)),
        lambda_s=float(cfg.get("lambda_s", 1.0)),
        lambda_t=float(cfg.get("lambda_t", 1.0)),
        lambda_int=float(cfg.get("lambda_int", 1.0)),
        baseline_layers=int(cfg.get("baseline_layers", 10)),
        risk_start_layer=int(cfg.get("risk_start_layer", 15)),
        alpha=float(cfg.get("alpha", 2.0)),
        ot_solver=cfg.get("ot_solver", "linprog"),
        atarget_visual_top_k=int(cfg.get("atarget_visual_top_k", 32)),
        topmass_alpha=float(cfg.get("topmass_085_alpha", 0.85)),
        capped_topmass_alpha=float(cfg.get("capped_topmass_085_alpha", 0.85)),
        capped_topmass_min_k=int(cfg.get("capped_topmass_085_min_k", 32)),
        capped_topmass_max_k=int(cfg.get("capped_topmass_085_max_k", 64)),
        compute_topmass_085=bool(cfg.get("compute_topmass_085", True)),
        compute_capped_topmass_085=bool(cfg.get("compute_capped_topmass_085", True)),
        four_gate_compute_capped_topmass_085=bool(
            cfg.get("compute_capped_topmass_085", False)
        ),
        four_gate_capped_topmass_alphas=cfg.get("capped_topmass_alphas"),
        target_gate_mode=cfg.get("target_gate_mode", "four_gate"),
        relative_vll_mad_epsilon=float(cfg.get("relative_vll_mad_epsilon", 1e-6)),
        relative_vll_logit_source=cfg.get("relative_vll_logit_source", "h_mid"),
        relative_cost_mode=cfg.get("relative_cost_mode"),
        relative_cost_modes=cfg.get("relative_cost_modes"),
        relative_cost_state_modes=cfg.get("relative_cost_state_modes"),
        relative_cost_update_lambdas=cfg.get("relative_cost_update_lambdas"),
        relative_barrier_lambda=float(cfg.get("relative_barrier_lambda", 1.0)),
        relative_barrier_margin=float(cfg.get("relative_barrier_margin", 0.5)),
        relative_barrier_max=float(cfg.get("relative_barrier_max", 3.0)),
        source_modes=cfg.get("source_modes"),
        target_attention_gammas=cfg.get("target_attention_gammas"),
        target_attention_epsilon=float(cfg.get("target_attention_epsilon", 1e-12)),
        compute_ffn_injection_features=bool(cfg.get("compute_ffn_injection_features", False)),
        ffn_injection_evidence_top_k=int(cfg.get("ffn_injection_evidence_top_k", 32)),
        ffn_injection_evidence_rank=int(cfg.get("ffn_injection_evidence_rank", 8)),
        ffn_injection_eps=float(cfg.get("ffn_injection_eps", 1e-12)),
        compute_dual_scope=bool(
            cfg.get("dgst_t_dual_scope", cfg.get("compute_dual_scope", False))
        ),
        four_gate_methods=cfg.get("four_gate_methods"),
        four_gate_cost_modes=cfg.get("cost_modes"),
        four_gate_support_modes=cfg.get("support_modes"),
        four_gate_source_tau_values=cfg.get("source_tau_values"),
        four_gate_transport_top_k_values=cfg.get("transport_top_k_values"),
        compute_prompt_cafe=bool(cfg.get("compute_prompt_cafe", False)),
        prompt_cafe_temperature=float(cfg.get("prompt_cafe_temperature", 10.0)),
        prompt_cafe_layer=int(cfg.get("prompt_cafe_layer", 22)),
        release_layer_captures=bool(release_layer_captures),
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one DGST result, received {len(results)}.")
    return results[0]
