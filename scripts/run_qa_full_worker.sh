#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL DATASET GPU" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$3"
# Each layer has exactly ten independent cost-variant EMD problems.  POT's
# network-simplex solver is CPU-bound, so solve those ten problems concurrently
# while keeping BLAS/OpenMP at one thread per problem to avoid oversubscription.
export DGST_COST_VARIANT_EMD_WORKERS="${DGST_COST_VARIANT_EMD_WORKERS:-10}"
export DGST_COST_VARIANT_EMD_BACKEND="${DGST_COST_VARIANT_EMD_BACKEND:-process}"
exec "$PYTHON_BIN" scripts/qa_pipeline.py \
  --model "$1" \
  --dataset "$2" \
  --config configs/model_configs_server_fj01.yaml \
  --output-root outputs/qa_benchmarks \
  --stage all \
  --checkpoint-every 10 \
  --feature-shard-size 25
