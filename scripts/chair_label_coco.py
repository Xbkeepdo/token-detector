#!/usr/bin/env python3
"""Compatibility wrapper for the local COCO/CHAIR labeling pipeline."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "coco-labeling" / "label_coco.py"
    os.chdir(repo_root)
    runpy.run_path(str(script), run_name="__main__")
