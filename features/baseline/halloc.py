"""Paper-compatible HalLoc object detector with lazy pretrained loading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
from torch import nn


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_VISUALBERT_MODEL = "uclanlp/visualbert-vqa-coco-pre"


class HalLocObjectDetector(nn.Module):
    """CLIP-ViT-B/32 + VisualBERT + one object-hallucination head.

    Passing ``clip_encoder`` and ``visualbert`` enables dependency-injected
    unit tests and offline construction.  If omitted, transformers is imported
    and pretrained weights are loaded only when this class is instantiated;
    importing the baseline package never downloads a model.

    Class logits use detector targets ``0=real, 1=hallucination``.  This is an
    explicit training target derived from the project's stored raw labels
    ``0=hallucination, 1=real``.
    """

    def __init__(
        self,
        *,
        lvlm_hidden_size: int,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        visualbert_model_name: str = DEFAULT_VISUALBERT_MODEL,
        freeze_clip: bool = True,
        clip_encoder: Optional[nn.Module] = None,
        visualbert: Optional[nn.Module] = None,
        clip_hidden_size: Optional[int] = None,
        visualbert_hidden_size: Optional[int] = None,
        visual_embedding_dim: Optional[int] = None,
        load_clip: bool = True,
    ) -> None:
        super().__init__()
        if int(lvlm_hidden_size) <= 0:
            raise ValueError("lvlm_hidden_size must be positive")

        if visualbert is None or (clip_encoder is None and load_clip):
            pretrained_clip, pretrained_visualbert = _load_pretrained_backbones(
                clip_model_name=clip_model_name,
                visualbert_model_name=visualbert_model_name,
                load_clip=load_clip and clip_encoder is None,
                load_visualbert=visualbert is None,
            )
            clip_encoder = clip_encoder or pretrained_clip
            visualbert = visualbert or pretrained_visualbert
        if visualbert is None:
            raise ValueError("visualbert is required")
        if load_clip and clip_encoder is None:
            raise ValueError("clip_encoder is required when load_clip=True")

        self.clip_encoder = clip_encoder
        self.visualbert = visualbert
        self.freeze_clip = bool(freeze_clip)
        self.clip_model_name = str(clip_model_name)
        self.visualbert_model_name = str(visualbert_model_name)

        clip_dim = clip_hidden_size or _config_value(
            clip_encoder,
            ("hidden_size", "config.hidden_size", "vision_config.hidden_size"),
        )
        bert_hidden = visualbert_hidden_size or _config_value(
            visualbert, ("hidden_size", "config.hidden_size")
        )
        visual_dim = visual_embedding_dim or _config_value(
            visualbert, ("visual_embedding_dim", "config.visual_embedding_dim")
        )
        if clip_dim is None:
            raise ValueError(
                "clip_hidden_size is required when it cannot be inferred from CLIP"
            )
        if bert_hidden is None or visual_dim is None:
            raise ValueError(
                "VisualBERT hidden_size/visual_embedding_dim could not be inferred"
            )

        self.text_projection = nn.Linear(int(lvlm_hidden_size), int(bert_hidden))
        self.vision_projection = nn.Linear(int(clip_dim), int(visual_dim))
        self.object_head = nn.Linear(int(bert_hidden), 2)
        if self.freeze_clip and self.clip_encoder is not None:
            self.clip_encoder.requires_grad_(False)
            self.clip_encoder.eval()

    def train(self, mode: bool = True) -> "HalLocObjectDetector":
        super().train(mode)
        if self.freeze_clip and self.clip_encoder is not None:
            self.clip_encoder.eval()
        return self

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.clip_encoder is None:
            raise RuntimeError(
                "This HalLoc instance was created for precomputed CLIP features"
            )
        if self.freeze_clip:
            with torch.no_grad():
                output = self.clip_encoder(pixel_values=pixel_values)
        else:
            output = self.clip_encoder(pixel_values=pixel_values)
        hidden = _last_hidden_state(output)
        return hidden

    def forward(
        self,
        *,
        lvlm_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        clip_visual_features: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        object_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lvlm_embeddings.ndim != 3:
            raise ValueError("lvlm_embeddings must have shape [B,T,D]")
        batch, text_length, _ = lvlm_embeddings.shape
        if clip_visual_features is None:
            if pixel_values is None:
                raise ValueError("Provide pixel_values or clip_visual_features")
            clip_visual_features = self.encode_images(pixel_values)
        if clip_visual_features.ndim != 3 or clip_visual_features.shape[0] != batch:
            raise ValueError("clip_visual_features must have shape [B,V,D_clip]")

        text_embeds = self.text_projection(lvlm_embeddings.float())
        visual_embeds = self.vision_projection(clip_visual_features.float())
        device = text_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, text_length), device=device, dtype=torch.long
            )
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)
        if visual_attention_mask is None:
            visual_attention_mask = torch.ones(
                visual_embeds.shape[:2], device=device, dtype=torch.long
            )
        else:
            visual_attention_mask = visual_attention_mask.to(
                device=device, dtype=torch.long
            )
        visual_token_type_ids = torch.ones_like(visual_attention_mask)

        output = self.visualbert(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            visual_embeds=visual_embeds,
            visual_token_type_ids=visual_token_type_ids,
            visual_attention_mask=visual_attention_mask,
        )
        fused_text = _last_hidden_state(output)[:, :text_length]
        logits = self.object_head(fused_text)
        if object_indices is None:
            return logits
        indices = object_indices.to(device=device, dtype=torch.long).reshape(-1)
        if indices.numel() != batch:
            raise ValueError("object_indices must contain one index per batch item")
        if int(indices.min()) < 0 or int(indices.max()) >= text_length:
            raise IndexError("object_indices contain a position outside the text sequence")
        return logits[torch.arange(batch, device=device), indices]

    def paper_metadata(self) -> dict[str, Any]:
        return {
            "clip_model": self.clip_model_name,
            "visualbert_model": self.visualbert_model_name,
            "freeze_clip": self.freeze_clip,
            "heads": ["object"],
            "stored_label_semantics": {"0": "hallucination", "1": "real"},
            "detector_target_semantics": {"0": "real", "1": "hallucination"},
        }


class HalLocCLIPFeatureExtractor:
    """Frozen CLIP-ViT image feature cache used before HalLoc training."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        *,
        device: str = "cuda",
        processor: Optional[Any] = None,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        if processor is None or encoder is None:
            try:
                from transformers import CLIPImageProcessor, CLIPVisionModel
            except ImportError as exc:
                raise ImportError("HalLoc CLIP extraction requires transformers") from exc
            processor = processor or CLIPImageProcessor.from_pretrained(model_name)
            encoder = encoder or CLIPVisionModel.from_pretrained(model_name)
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.processor = processor
        self.encoder = encoder.to(self.device).eval()
        self.encoder.requires_grad_(False)

    @torch.no_grad()
    def encode(self, image: Any) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        output = self.encoder(pixel_values=pixel_values)
        return _last_hidden_state(output)[0].detach().cpu().float()


def halloc_optimizer_config() -> dict[str, Any]:
    """Hyperparameters stated in *Beyond the Global Scores* supplement."""

    return {
        "optimizer": "AdamW",
        "learning_rate": 1e-6,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-2,
        "batch_size": 16,
    }


def _load_pretrained_backbones(
    *,
    clip_model_name: str,
    visualbert_model_name: str,
    load_clip: bool,
    load_visualbert: bool,
) -> tuple[Optional[nn.Module], Optional[nn.Module]]:
    try:
        from transformers import CLIPVisionModel, VisualBertModel
    except ImportError as exc:
        raise ImportError(
            "HalLoc requires transformers with CLIPVisionModel and VisualBertModel"
        ) from exc
    clip = CLIPVisionModel.from_pretrained(clip_model_name) if load_clip else None
    visualbert = (
        VisualBertModel.from_pretrained(visualbert_model_name)
        if load_visualbert
        else None
    )
    return clip, visualbert


def _config_value(module: Optional[nn.Module], paths: Sequence[str]) -> Optional[int]:
    if module is None:
        return None
    for path in paths:
        value: Any = module
        for component in path.split("."):
            if not hasattr(value, component):
                value = None
                break
            value = getattr(value, component)
        if value is not None:
            return int(value)
    return None


def _last_hidden_state(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("Backbone output does not expose last_hidden_state")


class SyntheticBackboneOutput(SimpleNamespace):
    """Tiny public helper useful for downstream dependency-injected smoke tests."""
