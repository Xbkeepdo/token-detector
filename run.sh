#!/bin/bash
# Full pipeline for one model. Replace MODEL and paths as needed.
set -euo pipefail

MODEL="${MODEL:-internvl_2_5_8b}"          # or: internvl_2_5_8b, qwen2_5_vl_7b, llava_1_5_7b
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO4000}"
CONFIG="${CONFIG:-configs/model_configs.yaml}"
DEVICE="${DEVICE:-cuda}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[run.sh] OPENAI_API_KEY is not set. Export it before GPT-4o labeling:"
    echo 'export OPENAI_API_KEY="your-openai-api-key"'
    exit 1
fi

# Step 1: Generate descriptions + GPT-4o labeling
python scripts/generate_and_label.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT \
    --device $DEVICE \
    --resume \
    --openai-proxy http://127.0.0.1:12596

# Step 2: Extract DGST-T features
python scripts/extract_features.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT \
    --device $DEVICE \
    --resume

# Step 3: Train classifiers + evaluate
python scripts/train_and_eval.py \
    --model $MODEL \
    --config $CONFIG \
    --output-dir $OUTPUT
