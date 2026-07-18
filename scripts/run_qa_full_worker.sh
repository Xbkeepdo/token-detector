#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL DATASET GPU" >&2
  exit 2
fi
cd /root/rivermind-data/project/token-detector
export CUDA_VISIBLE_DEVICES="$3"
# Each layer has exactly ten independent cost-variant EMD problems.  POT's
# network-simplex solver is CPU-bound, so solve those ten problems concurrently
# while keeping BLAS/OpenMP at one thread per problem to avoid oversubscription.
export DGST_COST_VARIANT_EMD_WORKERS="${DGST_COST_VARIANT_EMD_WORKERS:-10}"
export DGST_COST_VARIANT_EMD_BACKEND="${DGST_COST_VARIANT_EMD_BACKEND:-process}"
exec /opt/conda/envs/td/bin/python scripts/qa_pipeline.py \
  --model "$1" \
  --dataset "$2" \
  --config configs/model_configs_server_fj01.yaml \
  --output-root outputs/qa_benchmarks \
  --stage all \
  --checkpoint-every 10 \
  --feature-shard-size 25
