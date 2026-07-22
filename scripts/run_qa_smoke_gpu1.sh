#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi
OUT=outputs/qa_benchmarks_smoke
export CUDA_VISIBLE_DEVICES=1
"$PYTHON_BIN" scripts/qa_pipeline.py --model internvl_2_5_8b --dataset pope --output-root "$OUT" --limit 10 --stage all --feature-shard-size 2
"$PYTHON_BIN" scripts/qa_pipeline.py --model internvl_2_5_8b --dataset clevr_exist_9k --output-root "$OUT" --limit 10 --stage all --feature-shard-size 2
