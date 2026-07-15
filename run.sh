#!/bin/bash
# Full generation -> extraction -> training pipeline for one model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="${MODEL:-qwen3_vl_8b}"
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO4000-512}"
CONFIG="${CONFIG:-configs/model_configs_unified.yaml}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
GENERATION_DEVICES="${GENERATION_DEVICES:-cuda:0 cuda:1}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
CHAIR_CACHE="${CHAIR_CACHE:-outputs/chair_cache/coco_val2014_chair.pkl}"
NLTK_DATA="${NLTK_DATA:-/home/apulis-dev/userdata/nltk_data}"

DEFAULT_PYTHON="/opt/conda/private/envs/vicr/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES NLTK_DATA

# Step 1: Generate descriptions + CHAIR-style COCO labeling
"$PYTHON_BIN" scripts/generate_and_label.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --chair-cache "$CHAIR_CACHE" \
    --device "$DEVICE" \
    --generation-devices $GENERATION_DEVICES \
    --resume

# Step 2: Extract method + ADS/CGC + baseline features selected by YAML
"$PYTHON_BIN" scripts/extract_features.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE" \
    --feature-devices $FEATURE_DEVICES \
    --resume

# Step 3: Train configured probes/baselines + evaluate
"$PYTHON_BIN" scripts/train_and_eval.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE"
