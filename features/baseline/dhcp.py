"""Cross-modal attention features and mmap-friendly shards for DHCP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F


_SHARD_PATTERN = re.compile(r"^shard_(\d{6})\.npy$")


@dataclass(frozen=True)
class DHCPShardReference:
    shard: str
    index: int
    shape: tuple[int, ...]
    dtype: str = "float16"

    def as_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "DHCPShardReference":
        return cls(
            shard=str(value["shard"]),
            index=int(value["index"]),
            shape=tuple(int(x) for x in value["shape"]),
            dtype=str(value.get("dtype", "float16")),
        )


def resize_attention_preserve_mass(
    visual_attention: Union[torch.Tensor, np.ndarray],
    *,
    source_grid: Optional[Sequence[int]] = None,
    target_grid: Sequence[int] = (12, 12),
) -> torch.Tensor:
    """Resize every attention map and restore its original total mass.

    Inputs can be ``[...,P]`` (with an explicit or inferred source grid) or
    ``[...,H,W]`` when ``source_grid`` is omitted.  The result always has shape
    ``[...,target_h,target_w]``.  Bilinear interpolation alone changes the sum
    when the number of patches changes; the final rescaling is what makes this
    suitable for DHCP rather than a generic image resize.
    """

    attention = torch.as_tensor(visual_attention).detach().float()
    if not torch.isfinite(attention).all():
        raise ValueError("visual_attention contains non-finite values")
    if attention.numel() == 0:
        raise ValueError("visual_attention cannot be empty")
    if float(attention.min()) < -1e-7:
        raise ValueError("attention weights must be non-negative")
    attention = attention.clamp_min(0)

    target_h, target_w = _validated_grid(target_grid, name="target_grid")
    if source_grid is not None:
        source_h, source_w = _validated_grid(source_grid, name="source_grid")
        if attention.shape[-1] != source_h * source_w:
            raise ValueError(
                f"source_grid {source_h}x{source_w} does not match P={attention.shape[-1]}"
            )
        prefix = attention.shape[:-1]
        maps = attention.reshape(-1, 1, source_h, source_w)
    else:
        if attention.ndim >= 4:
            source_h, source_w = int(attention.shape[-2]), int(attention.shape[-1])
            prefix = attention.shape[:-2]
            maps = attention.reshape(-1, 1, source_h, source_w)
        else:
            source_h, source_w = infer_patch_grid(int(attention.shape[-1]))
            prefix = attention.shape[:-1]
            maps = attention.reshape(-1, 1, source_h, source_w)

    source_mass = maps.sum(dim=(-2, -1), keepdim=True)
    resized = F.interpolate(
        maps,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).clamp_min(0)
    resized_mass = resized.sum(dim=(-2, -1), keepdim=True)
    scale = torch.where(
        source_mass > 0,
        source_mass / resized_mass.clamp_min(torch.finfo(resized.dtype).tiny),
        torch.zeros_like(source_mass),
    )
    resized = resized * scale
    return resized.reshape(*prefix, target_h, target_w)


def infer_patch_grid(num_patches: int) -> tuple[int, int]:
    """Infer the closest-to-square exact factorization of ``num_patches``."""

    patches = int(num_patches)
    if patches <= 0:
        raise ValueError("num_patches must be positive")
    for height in range(int(patches**0.5), 0, -1):
        if patches % height == 0:
            return height, patches // height
    raise AssertionError("Every positive integer has at least the factorization 1 x P")


class DHCPShardWriter:
    """Append fixed-shape DHCP tensors to atomic float16 ``.npy`` shards."""

    def __init__(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        shard_size: int = 256,
        resume: bool = True,
        reference_prefix: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.reference_prefix = (
            None
            if reference_prefix is None
            else str(reference_prefix).strip("/ ")
        )
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        existing = self._existing_indices()
        if existing and not resume:
            raise FileExistsError(
                f"DHCP shard directory is not empty: {self.root}"
            )
        self._shard_index = (max(existing) + 1) if existing else 0
        self._buffer: list[np.ndarray] = []
        self._buffer_shape: Optional[tuple[int, ...]] = None
        self._manifest_entries: list[dict[str, Any]] = []

    def add(
        self,
        tensor: Union[torch.Tensor, np.ndarray],
    ) -> DHCPShardReference:
        array = _as_float16_array(tensor)
        shape = tuple(int(x) for x in array.shape)
        if self._buffer and shape != self._buffer_shape:
            self.flush()
        if self._buffer_shape is None:
            self._buffer_shape = shape
        shard_name = self._shard_name(self._shard_index)
        reference_name = (
            f"{self.reference_prefix}/{shard_name}"
            if self.reference_prefix
            else shard_name
        )
        reference = DHCPShardReference(
            shard=reference_name,
            index=len(self._buffer),
            shape=shape,
        )
        self._buffer.append(array)
        if len(self._buffer) >= self.shard_size:
            self.flush()
        return reference

    def flush(self) -> None:
        if not self._buffer:
            return
        stacked = np.stack(self._buffer, axis=0).astype(np.float16, copy=False)
        shard_name = self._shard_name(self._shard_index)
        destination = self.root / shard_name
        _atomic_numpy_save(destination, stacked)
        self._manifest_entries.append(
            {
                "shard": shard_name,
                "count": int(stacked.shape[0]),
                "item_shape": list(stacked.shape[1:]),
                "dtype": "float16",
            }
        )
        self._shard_index += 1
        self._buffer.clear()
        self._buffer_shape = None
        self._write_manifest()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "DHCPShardWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()

    def _existing_indices(self) -> list[int]:
        indices = []
        for path in self.root.glob("shard_*.npy"):
            match = _SHARD_PATTERN.match(path.name)
            if match:
                indices.append(int(match.group(1)))
        return indices

    @staticmethod
    def _shard_name(index: int) -> str:
        return f"shard_{int(index):06d}.npy"

    def _write_manifest(self) -> None:
        entries = []
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                current = json.load(handle)
            entries.extend(current.get("shards", []))
        entries.extend(self._manifest_entries)
        payload = {
            "format": "dhcp-float16-npy-shards-v1",
            "shards": entries,
        }
        _atomic_json_save(manifest_path, payload)
        self._manifest_entries.clear()


class DHCPShardReader:
    """Read individual DHCP items without loading an entire shard into RAM."""

    def __init__(self, root: Union[str, os.PathLike[str]]) -> None:
        self.root = Path(root).resolve()

    def load(
        self,
        reference: Union[DHCPShardReference, Mapping[str, Any]],
        *,
        mmap_mode: Optional[str] = "r",
        as_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        ref = (
            reference
            if isinstance(reference, DHCPShardReference)
            else DHCPShardReference.from_payload(reference)
        )
        path = (self.root / ref.shard).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"Shard reference escapes root: {ref.shard!r}")
        array = np.load(path, mmap_mode=mmap_mode)
        if array.dtype != np.float16:
            raise ValueError(f"Expected float16 DHCP shard, got {array.dtype}")
        if not 0 <= ref.index < len(array):
            raise IndexError(f"DHCP index {ref.index} outside shard length {len(array)}")
        item = np.asarray(array[ref.index])
        if tuple(item.shape) != ref.shape:
            raise ValueError(
                f"DHCP reference shape {ref.shape} != stored shape {tuple(item.shape)}"
            )
        return torch.from_numpy(item.copy()) if as_tensor else item


def _validated_grid(grid: Sequence[int], *, name: str) -> tuple[int, int]:
    if len(grid) != 2:
        raise ValueError(f"{name} must contain [height, width]")
    height, width = int(grid[0]), int(grid[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} entries must be positive")
    return height, width


def _as_float16_array(value: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("DHCP tensor is empty or contains non-finite values")
    return array.astype(np.float16, copy=False)


def _atomic_numpy_save(path: Path, value: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
