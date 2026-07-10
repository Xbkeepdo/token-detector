#!/usr/bin/env python3
"""Compatibility entry point for COCO CHAIR generation + labeling.

This script used to run GPT-4o labeling. It now forwards to
``coco-labeling/label_coco.py`` so older commands keep working without any
OpenAI/GitHub Models API key.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


LEGACY_OPENAI_FLAGS = {
    "--openai-key",
    "--openai-base-url",
    "--openai-model",
    "--openai-proxy",
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    chair_script = repo_root / "coco-labeling" / "label_coco.py"
    sys.argv = [str(chair_script), *_strip_legacy_openai_args(sys.argv[1:])]
    runpy.run_path(str(chair_script), run_name="__main__")


def _strip_legacy_openai_args(argv: list[str]) -> list[str]:
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if any(arg.startswith(f"{flag}=") for flag in LEGACY_OPENAI_FLAGS):
            continue
        if arg in LEGACY_OPENAI_FLAGS:
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


if __name__ == "__main__":
    main()
