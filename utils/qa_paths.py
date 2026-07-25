"""Canonical paths for one named QA output.

The output directory owns reusable model generations, while each benchmark
subdirectory owns labels, features, and results derived from those generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Mapping


DEFAULT_QA_OUTPUT = "default"


def resolve_qa_output_name(
    cli_value: str | None,
    qa_config: Mapping[str, object] | None = None,
) -> str:
    """Resolve and validate the single directory name used for an output."""

    configured = (qa_config or {}).get("output")
    raw = cli_value if cli_value is not None else configured
    value = str(raw if raw is not None else DEFAULT_QA_OUTPUT).strip()
    if not value:
        raise ValueError("QA output name cannot be empty")
    if value in {".", ".."} or PurePath(value).name != value:
        raise ValueError(
            "QA output must be one directory name without path separators: "
            f"{value!r}"
        )
    if "/" in value or "\\" in value:
        raise ValueError(
            "QA output must be one directory name without path separators: "
            f"{value!r}"
        )
    return value


@dataclass(frozen=True)
class QAOutputPaths:
    """All durable paths for one model/output/benchmark tuple."""

    output_root: Path
    model: str
    output: str
    dataset: str

    @property
    def model_dir(self) -> Path:
        return self.output_root / self.model

    @property
    def output_dir(self) -> Path:
        return self.model_dir / self.output

    @property
    def benchmark_dir(self) -> Path:
        return self.output_dir / self.dataset

    @property
    def generations_path(self) -> Path:
        return self.output_dir / f"{self.dataset}_generations.jsonl"

    @property
    def generation_failures_path(self) -> Path:
        return self.output_dir / f"{self.dataset}_generation_failures.jsonl"


def resolve_qa_paths(
    output_root: str | Path,
    model: str,
    output: str,
    dataset: str,
) -> QAOutputPaths:
    """Build canonical QA paths after validating the output component."""

    return QAOutputPaths(
        output_root=Path(output_root),
        model=str(model),
        output=resolve_qa_output_name(output),
        dataset=str(dataset),
    )


def generations_path_for_benchmark_dir(
    benchmark_dir: str | Path,
) -> Path:
    """Return the canonical sibling generation file for a benchmark directory."""

    benchmark = Path(benchmark_dir)
    return benchmark.parent / f"{benchmark.name}_generations.jsonl"


def locate_qa_generations(
    benchmark_dir: str | Path,
    explicit_path: str | Path | None = None,
) -> Path:
    """Locate new-layout generations, with read-only legacy compatibility."""

    if explicit_path is not None:
        return Path(explicit_path)
    benchmark = Path(benchmark_dir)
    canonical = generations_path_for_benchmark_dir(benchmark)
    legacy = benchmark / "generations.jsonl"
    if canonical.is_file() or not legacy.is_file():
        return canonical
    return legacy
