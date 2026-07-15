"""Reusable resources for joint and baseline-only feature extraction.

``BaselineRuntime`` owns the expensive image/model-level resources needed by
the paper baselines.  It intentionally does not run an LVLM forward: callers
can feed the same ``ModelOutput`` objects to DGST/ADS/CGC and this runtime.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from models.base_wrapper import ExtractionRequirements, ModelOutput
from models.dgst_capture import resolve_output_embedding_layer

from .dhcp import DHCPShardWriter
from .extractor import (
    BaselineExtractionContext,
    baseline_extraction_requirements,
    compute_baseline_record,
)
from .halloc import HalLocCLIPFeatureExtractor
from .projectaway import compute_projectaway_probability_cache
from .schema import SUPPORTED_BASELINES


def normalize_baseline_methods(methods: Sequence[str] | str) -> tuple[str, ...]:
    """Normalize config input while retaining one deterministic method order."""

    if isinstance(methods, str):
        values = [methods]
    else:
        values = list(methods)
    normalized = [
        str(value).strip().lower().replace("-", "_") for value in values
    ]
    aliases = {"project_away": "projectaway", "meta_token": "metatoken"}
    normalized = [aliases.get(value, value) for value in normalized]
    if "all" in normalized:
        normalized = list(sorted(SUPPORTED_BASELINES))
    unknown = set(normalized) - SUPPORTED_BASELINES
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    # Dict order provides stable deduplication without imposing an arbitrary
    # alphabetic order on an explicit YAML list.
    return tuple(dict.fromkeys(normalized))


def baseline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge legacy top-level and unified nested baseline configuration."""

    top_level = config.get("baselines") or {}
    nested = (config.get("feature_extraction") or {}).get("baseline") or {}
    return _deep_merge(top_level, nested)


class BaselineRuntime:
    """Image-level baseline resource manager shared by extraction modes.

    Parameters are deliberately dependency-injectable.  Unit tests and joint
    extraction can provide an already-created CLIP extractor, output layer, or
    final norm; baseline-only extraction can let the runtime resolve them once
    from ``wrapper``.
    """

    def __init__(
        self,
        *,
        wrapper: Any,
        methods: Sequence[str] | str,
        baseline_dir: str | os.PathLike[str],
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        worker_id: int = 0,
        parallel: bool = False,
        resume: bool = True,
        dhcp_writer: Optional[DHCPShardWriter] = None,
        clip_extractor: Optional[HalLocCLIPFeatureExtractor] = None,
        output_layer: Optional[Any] = None,
        final_norm: Optional[Any] = None,
    ) -> None:
        self.wrapper = wrapper
        self.methods = normalize_baseline_methods(methods)
        if not self.methods:
            raise ValueError("At least one baseline method must be selected")
        self.config = deepcopy(dict(config or {}))
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.device = str(device or getattr(wrapper, "device", "cuda"))
        self.worker_id = int(worker_id)
        self.parallel = bool(parallel)
        self.requirements: ExtractionRequirements = baseline_extraction_requirements(
            self.methods
        )
        self._closed = False

        self._owns_dhcp_writer = False
        self.dhcp_writer = dhcp_writer
        if "dhcp" in self.methods and self.dhcp_writer is None:
            shard_prefix = f"worker_{self.worker_id}" if self.parallel else None
            shard_root = self.baseline_dir / "dhcp" / "shards"
            writer_root = shard_root / shard_prefix if shard_prefix else shard_root
            dhcp_cfg = dict(self.config.get("dhcp") or {})
            self.dhcp_writer = DHCPShardWriter(
                writer_root,
                shard_size=int(dhcp_cfg.get("shard_size", 256)),
                resume=resume,
                reference_prefix=shard_prefix,
            )
            self._owns_dhcp_writer = True

        self.clip_extractor = clip_extractor
        if "halloc" in self.methods and self.clip_extractor is None:
            halloc_cfg = dict(self.config.get("halloc") or {})
            self.clip_extractor = HalLocCLIPFeatureExtractor(
                str(halloc_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
                device=self.device,
            )

        self.output_layer = output_layer
        self.final_norm = final_norm
        if "projectaway" in self.methods:
            model = getattr(wrapper, "model", None)
            if self.output_layer is None:
                self.output_layer = resolve_output_embedding_layer(model)

    def build_image_records(
        self,
        *,
        image: Image.Image,
        image_id: int,
        response_token_ids: Sequence[int],
        spans: Sequence[Mapping[str, Any]],
        model_outputs: Sequence[ModelOutput],
    ) -> list[dict[str, Any]]:
        """Build all object records from one shared set of wrapper outputs."""

        if self._closed:
            raise RuntimeError("BaselineRuntime is already closed")
        if len(spans) != len(model_outputs):
            raise ValueError(
                f"Received {len(spans)} spans but {len(model_outputs)} model outputs"
            )
        if not spans:
            return []
        response_ids = [int(value) for value in response_token_ids]
        normalized_spans = [dict(span) for span in spans]
        for span in normalized_spans:
            indices = [int(value) for value in span.get("token_indices", [])]
            if not indices or any(index < 0 or index >= len(response_ids) for index in indices):
                raise ValueError(
                    f"Invalid object token indices {indices} for response length "
                    f"{len(response_ids)}"
                )

        projectaway_cache = None
        if "projectaway" in self.methods:
            union_ids = tuple(
                dict.fromkeys(
                    response_ids[index]
                    for span in normalized_spans
                    for index in (int(value) for value in span["token_indices"])
                )
            )
            patch_hidden = model_outputs[0].patch_hidden_states
            if patch_hidden is None:
                raise RuntimeError(
                    "ProjectAway requested but wrapper returned no patch_hidden_states"
                )
            projectaway_cfg = dict(self.config.get("projectaway") or {})
            projectaway_cache = compute_projectaway_probability_cache(
                patch_hidden,
                self.output_layer.weight,
                union_ids,
                unembedding_bias=getattr(self.output_layer, "bias", None),
                # ProjectAway applies the LM head directly to intermediate
                # visual hidden states; decoder final-norm here would define a
                # different detector.
                normalizer=None,
                vocab_chunk_size=int(
                    projectaway_cfg.get("vocab_chunk_size", 8192)
                ),
                row_chunk_size=int(projectaway_cfg.get("row_chunk_size", 256)),
            )

        halloc_cache_file = None
        if "halloc" in self.methods:
            response_hidden = model_outputs[0].response_hidden_states
            if response_hidden is None:
                raise RuntimeError(
                    "HalLoc requested but wrapper returned no response_hidden_states"
                )
            clip_features = self.clip_extractor.encode(image)
            relative = Path("halloc") / "cache"
            if self.parallel:
                relative /= f"worker_{self.worker_id}"
            relative /= f"{int(image_id)}.npz"
            _atomic_npz(
                self.baseline_dir / relative,
                lvlm_embeddings=response_hidden.detach()
                .cpu()
                .numpy()
                .astype(np.float16),
                clip_visual_features=clip_features.detach()
                .cpu()
                .numpy()
                .astype(np.float16),
            )
            halloc_cache_file = str(relative)

        counts = _occurrence_counts(normalized_spans)
        # Caption statistics are image-level and wrappers may attach the shared
        # object to only the first result to avoid redundant references.
        shared_metatoken_stats = next(
            (
                output.baseline_capture
                for output in model_outputs
                if output.baseline_capture is not None
            ),
            None,
        )
        records = []
        for span, model_out in zip(normalized_spans, model_outputs):
            token_indices = [int(value) for value in span["token_indices"]]
            context = BaselineExtractionContext(
                response_token_ids=response_ids,
                image_id=int(image_id),
                metatoken_step_stats=(
                    model_out.baseline_capture or shared_metatoken_stats
                ),
                occurrence_count=counts[_normalize_word(span.get("word", ""))],
                target_token_ids=[response_ids[index] for index in token_indices],
                projectaway_probability_cache=projectaway_cache,
                halloc_cache_file=halloc_cache_file,
                halloc_object_index=token_indices[0],
            )
            records.append(
                compute_baseline_record(
                    model_out=model_out,
                    span=span,
                    context=context,
                    methods=self.methods,
                    dhcp_writer=self.dhcp_writer,
                    config=self.config,
                )
            )
        # One image is the resume transaction boundary.  The caller appends
        # ``records`` immediately after this method returns, so every referenced
        # DHCP shard must already exist on disk before the pickle commit.
        if self.dhcp_writer is not None:
            self.dhcp_writer.flush()
        return records

    def close(self) -> None:
        """Flush resources owned by this runtime exactly once."""

        if self._closed:
            return
        if self._owns_dhcp_writer and self.dhcp_writer is not None:
            self.dhcp_writer.close()
        self._closed = True

    def __enter__(self) -> "BaselineRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Flush prior, already-published records even if a later image fails;
        # append-only resume files may already refer to entries in the buffer.
        self.close()


def _occurrence_counts(spans: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        word = _normalize_word(span.get("word", ""))
        counts[word] = counts.get(word, 0) + 1
    return counts


def _normalize_word(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
