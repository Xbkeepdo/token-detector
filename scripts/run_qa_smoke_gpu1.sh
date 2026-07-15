#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/project/token-detector
PY=/opt/conda/envs/td/bin/python
OUT=outputs/qa_benchmarks_smoke
export CUDA_VISIBLE_DEVICES=1
"$PY" scripts/qa_pipeline.py --model internvl_2_5_8b --dataset pope --output-root "$OUT" --limit 10 --stage all --feature-shard-size 2
"$PY" scripts/qa_pipeline.py --model internvl_2_5_8b --dataset clevr_exist_5k --output-root "$OUT" --limit 10 --stage all --feature-shard-size 2
