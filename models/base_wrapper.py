"""Base class for LVLM wrappers. Defines the interface for generation and feature extraction."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image


def configure_image_processor_limits(
    processor: Any,
    config: Mapping[str, Any],
) -> None:
    """Apply one YAML pixel budget to both slow and fast HF processors.

    Qwen-family processors have used both ``max_pixels`` and
    ``size.longest_edge`` across transformers releases.  Setting both keeps
    generation and every extraction mode on the same bounded visual grid.
    """

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    max_pixels = config.get("max_pixels")
    min_pixels = config.get("min_pixels")
    if max_pixels is not None:
        maximum = int(max_pixels)
        if maximum <= 0:
            raise ValueError("model max_pixels must be positive")
        image_processor.max_pixels = maximum
        size = dict(getattr(image_processor, "size", None) or {})
        size["longest_edge"] = maximum
        image_processor.size = size
    if min_pixels is not None:
        minimum = int(min_pixels)
        if minimum <= 0:
            raise ValueError("model min_pixels must be positive")
        if max_pixels is not None and minimum > int(max_pixels):
            raise ValueError("model min_pixels cannot exceed max_pixels")
        image_processor.min_pixels = minimum
        size = dict(getattr(image_processor, "size", None) or {})
        size["shortest_edge"] = minimum
        image_processor.size = size


@dataclass
class ModelOutput:
    """Structured output of a single prefix forward pass."""

    token_id: int
    token_str: str

    text_to_patch_attn: torch.Tensor

    text_to_text_attn: torch.Tensor

    token_hidden_states: torch.Tensor

    patch_hidden_states: torch.Tensor

    response_token_idx: int

    token_logits: Optional[torch.Tensor] = None

    dgst_t_raw: Optional[dict[str, Any]] = None

    dgst_t_result: Optional[dict[str, Any]] = None

    # Dynamic-layout models (Qwen-VL/OneVision) cannot expose a meaningful
    # class-wide ``num_visual_tokens`` constant.  Keep the layout alongside
    # the forward result instead.  These fields are optional so older
    # wrappers and serialized call sites remain source compatible.
    visual_grid: Optional[Tuple[int, int]] = None

    tile_offsets: Optional[List[Tuple[int, int]]] = None

    response_hidden_states: Optional[torch.Tensor] = None

    baseline_capture: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class PromptTargetRequest:
    """One token target that occurs inside the *actual* user prompt.

    ``target_char_start`` / ``target_char_end`` are offsets in ``prompt`` and
    are the preferred way to disambiguate repeated object words.  When they
    are omitted, ``occurrence`` selects an exact surface occurrence in the
    rendered prompt.  ``expected_target_token_id`` is an optional assertion;
    wrappers always obtain the operative ID from their real multimodal input
    IDs, never by encoding ``target_text`` in isolation.
    """

    prompt: str
    target_text: str
    target_char_start: Optional[int] = None
    target_char_end: Optional[int] = None
    occurrence: int = 0
    expected_target_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        prompt = str(self.prompt)
        target = str(self.target_text)
        if not prompt:
            raise ValueError("prompt-target prompt must be non-empty")
        if not target:
            raise ValueError("prompt-target target_text must be non-empty")
        if int(self.occurrence) < 0:
            raise ValueError("prompt-target occurrence must be non-negative")
        has_start = self.target_char_start is not None
        has_end = self.target_char_end is not None
        if has_start != has_end:
            raise ValueError(
                "target_char_start and target_char_end must be provided together"
            )
        if has_start:
            start = int(self.target_char_start)
            end = int(self.target_char_end)
            if start < 0 or end <= start or end > len(prompt):
                raise ValueError(
                    f"invalid prompt target character span [{start}, {end})"
                )
            if prompt[start:end] != target:
                raise ValueError(
                    "prompt target character span does not equal target_text: "
                    f"{prompt[start:end]!r} != {target!r}"
                )


@dataclass(frozen=True)
class PromptTargetAlignment:
    """Resolved prompt target in tokenized and expanded decoder coordinates."""

    target_tokenized_position: int
    target_expanded_position: int
    prediction_position: int
    target_token_id: int
    tokenized_span: Tuple[int, ...]
    rendered_char_start: int
    rendered_char_end: int


class AttentionRequirement(str, Enum):
    """Amount of decoder attention requested from a wrapper forward."""

    NONE = "none"
    HEAD_MEAN = "head_mean"
    PER_HEAD = "per_head"

    @classmethod
    def normalize(cls, value: "AttentionRequirement | str") -> "AttentionRequirement":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "mean": cls.HEAD_MEAN,
            "headmean": cls.HEAD_MEAN,
            "full": cls.PER_HEAD,
            "all_heads": cls.PER_HEAD,
        }
        try:
            return aliases[normalized] if normalized in aliases else cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown attention requirement '{value}'. Expected one of: {choices}."
            ) from exc


@dataclass(frozen=True)
class ExtractionRequirements:
    """Tensors an extraction consumer needs from one model forward.

    ``None`` at the wrapper API boundary keeps the historical behaviour.  The
    unified pipeline can pass an explicit instance so baseline-only runs do
    not install DGST hooks and method-only runs do not retain large per-head
    tensors unnecessarily.
    """

    attention: AttentionRequirement = AttentionRequirement.PER_HEAD
    logits: bool = True
    token_hidden_states: bool = True
    patch_hidden_states: bool = True
    response_hidden_states: bool = False
    visual_layout: bool = True
    dgst_capture: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", AttentionRequirement.normalize(self.attention))

    @property
    def needs_attention_weights(self) -> bool:
        return self.attention is not AttentionRequirement.NONE or self.dgst_capture

    @property
    def needs_hidden_states(self) -> bool:
        # DGST hooks already capture h_prev/o_attn/o_ffn and can reconstruct
        # every active hpre/hmid/output state.  Requesting model-level
        # hidden_states as well would duplicate the largest layerwise tensor.
        return bool(
            self.token_hidden_states
            or self.patch_hidden_states
            or self.response_hidden_states
        )

    @classmethod
    def legacy(cls, *, dgst_enabled: bool = True) -> "ExtractionRequirements":
        """Requirements matching the pre-requirements wrapper contract."""
        return cls(dgst_capture=bool(dgst_enabled))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExtractionRequirements":
        allowed = {
            "attention",
            "logits",
            "token_hidden_states",
            "patch_hidden_states",
            "response_hidden_states",
            "visual_layout",
            "dgst_capture",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "Unknown extraction requirement field(s): " + ", ".join(unknown)
            )
        return cls(**{key: value[key] for key in allowed if key in value})

    def merged(self, *others: "ExtractionRequirements") -> "ExtractionRequirements":
        """Return the least upper bound needed by all requested methods."""
        attention_rank = {
            AttentionRequirement.NONE: 0,
            AttentionRequirement.HEAD_MEAN: 1,
            AttentionRequirement.PER_HEAD: 2,
        }
        requirements = (self, *others)
        attention = max(requirements, key=lambda item: attention_rank[item.attention]).attention
        return ExtractionRequirements(
            attention=attention,
            logits=any(item.logits for item in requirements),
            token_hidden_states=any(item.token_hidden_states for item in requirements),
            patch_hidden_states=any(item.patch_hidden_states for item in requirements),
            response_hidden_states=any(item.response_hidden_states for item in requirements),
            visual_layout=any(item.visual_layout for item in requirements),
            dgst_capture=any(item.dgst_capture for item in requirements),
        )


@dataclass
class GenerationOutput:
    """Full generation result for one image."""
    image_id: int
    generated_text: str
    response_token_ids: List[int]
    response_tokens: List[str]


def compact_response_logit_statistics(
    logits: torch.Tensor,
    *,
    response_token_ids: Sequence[int],
    row_chunk_size: int = 8,
) -> dict[str, Any]:
    """Compress teacher-forced ``[T,V]`` logits without retaining them.

    The returned statistics are the complete inputs needed by MetaToken.  Row
    chunking bounds the temporary fp32 probability/log-probability tensors.
    """
    token_ids = [int(token_id) for token_id in response_token_ids]
    if int(logits.shape[0]) != len(token_ids):
        raise ValueError(
            "Teacher-forced logits and response_token_ids differ in length: "
            f"{int(logits.shape[0])} != {len(token_ids)}."
        )
    names = (
        "response_target_logprobs",
        "response_target_probs",
        "response_logprob_variances",
        "response_normalized_entropies",
        "response_top1_probs",
        "response_top2_probs",
    )
    values: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    vocab_size = int(logits.shape[-1])
    normalizer = torch.log(
        torch.tensor(float(max(2, vocab_size)), dtype=torch.float32)
    )
    step = max(1, int(row_chunk_size))
    for start in range(0, len(token_ids), step):
        chunk = logits[start:start + step].float()
        log_probs = torch.log_softmax(chunk, dim=-1)
        probs = torch.exp(log_probs)
        chunk_ids = torch.tensor(
            token_ids[start:start + int(chunk.shape[0])],
            dtype=torch.long,
            device=chunk.device,
        )
        target_logprobs = log_probs.gather(1, chunk_ids.unsqueeze(1)).squeeze(1)
        top_two = torch.topk(probs, k=min(2, vocab_size), dim=-1).values
        if int(top_two.shape[1]) == 1:
            top_two = torch.cat((top_two, torch.zeros_like(top_two)), dim=1)
        values["response_target_logprobs"].append(target_logprobs.cpu())
        values["response_target_probs"].append(torch.exp(target_logprobs).cpu())
        values["response_logprob_variances"].append(
            torch.var(log_probs, dim=-1, unbiased=False).cpu()
        )
        values["response_normalized_entropies"].append(
            (-(probs * log_probs).sum(dim=-1) / normalizer.to(probs.device)).cpu()
        )
        values["response_top1_probs"].append(top_two[:, 0].cpu())
        values["response_top2_probs"].append(top_two[:, 1].cpu())
        del chunk, log_probs, probs, target_logprobs, top_two
    result: dict[str, Any] = {
        name: (
            torch.cat(chunks, dim=0).float()
            if chunks
            else torch.empty(0, dtype=torch.float32)
        )
        for name, chunks in values.items()
    }
    result["response_token_ids"] = token_ids
    return result


class BaseLVLMWrapper(ABC):
    """Abstract wrapper for Large Vision-Language Models."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()


    @abstractmethod
    def _load_model(self) -> None:
        """Load model, tokenizer/processor onto self.device."""
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> GenerationOutput:
        """Generate a description for `image` using greedy decoding"""
        ...

    @abstractmethod
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
        """Run ONE forward pass using `prefix_token_ids` as input and"""
        ...

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
        """Fallback batch API; wrappers can override to reuse one image forward."""
        response_ids, requested_indices, targets = self.validate_causal_batch_request(
            response_token_ids=response_token_ids,
            response_token_indices=response_token_indices,
            target_token_ids=target_token_ids,
        )
        outputs: List[ModelOutput] = []
        for response_index, target_token_id in zip(requested_indices, targets):
            index = int(response_index)
            outputs.append(
                self.extract_token_features(
                    image=image,
                    prefix_token_ids=response_ids[:index],
                    response_token_idx=index,
                    target_token_id=int(target_token_id),
                    cfg_dgst_t=cfg_dgst_t,
                    prompt=prompt,
                    requirements=requirements,
                )
            )
        return outputs

    def extract_prompt_target_features(
        self,
        image: Image.Image,
        request: PromptTargetRequest,
        cfg_dgst_t: Optional[dict[str, Any]] = None,
        requirements: Optional[ExtractionRequirements] = None,
    ) -> ModelOutput:
        """Extract the causal row immediately before a target in the prompt.

        This is deliberately separate from ``extract_token_features_batch``:
        the latter validates saved generated-response IDs and must never be
        relaxed to accept an unrelated question/object token.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not implement prompt-target extraction"
        )

    @staticmethod
    def validate_causal_batch_request(
        *,
        response_token_ids: Sequence[int],
        response_token_indices: Sequence[int],
        target_token_ids: Optional[Sequence[int]] = None,
    ) -> tuple[List[int], List[int], List[int]]:
        """Validate that each requested state predicts the saved target token."""

        response_ids = [int(value) for value in response_token_ids]
        requested_indices = [int(value) for value in response_token_indices]
        for index in requested_indices:
            if index < 0 or index >= len(response_ids):
                raise ValueError(
                    f"response token index {index} is outside response length "
                    f"{len(response_ids)}"
                )
        expected_targets = [response_ids[index] for index in requested_indices]
        if target_token_ids is None:
            targets = expected_targets
        else:
            targets = [int(value) for value in target_token_ids]
            if len(targets) != len(requested_indices):
                raise ValueError(
                    "target_token_ids and response_token_indices must have "
                    "the same length."
                )
            mismatches = [
                (offset, requested_indices[offset], targets[offset], expected)
                for offset, expected in enumerate(expected_targets)
                if targets[offset] != expected
            ]
            if mismatches:
                raise ValueError(
                    "target_token_ids must equal the actual saved response token "
                    f"at each requested index; mismatches={mismatches[:5]}"
                )
        return response_ids, requested_indices, targets

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers in the LM backbone."""
        ...

    @property
    @abstractmethod
    def num_visual_tokens(self) -> int:
        """Number of visual patch tokens in the LM sequence."""
        ...


    def _ids_to_str(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def resolve_prompt(self, prompt: Optional[str] = None) -> str:
        """Resolve a raw user instruction with explicit arguments taking priority."""
        if prompt is not None and str(prompt).strip():
            return str(prompt)
        for key in ("prompt", "generation_prompt"):
            value = self.cfg.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return "Describe this image."

    @staticmethod
    def resolve_extraction_requirements(
        requirements: Optional[ExtractionRequirements | Mapping[str, Any]],
        *,
        dgst_enabled: bool,
    ) -> ExtractionRequirements:
        if isinstance(requirements, Mapping):
            return ExtractionRequirements.from_mapping(requirements)
        if requirements is not None:
            return requirements
        return ExtractionRequirements.legacy(dgst_enabled=dgst_enabled)

    @property
    def generation_max_new_tokens(self) -> int:
        """Default caption generation budget, overridable by model config."""
        return int(self.cfg.get("max_new_tokens", 512))

    @torch.no_grad()
    def _safe_forward(self, **kwargs) -> dict:
        """Wrapper around model(**kwargs) that always disables gradients"""
        try:
            return self.model(**kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                "GPU OOM during forward pass. Try reducing image resolution "
                "or processing fewer object tokens at once."
            ) from e
