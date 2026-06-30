#!/bin/bash
# Full pipeline for one model. Replace MODEL and paths as needed.
set -euo pipefail

MODEL="${MODEL:-internvl_2_5_8b}"          # or: internvl_2_5_8b, qwen2_5_vl_7b, llava_1_5_7b
CONFIG="${CONFIG:-configs/model_configs.yaml}"
LABELER="${LABELER:-gpt4o}"
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO500-visualonly}"       # gpt4o, chair, or coco-output
# if [ "$LABELER" = "coco-output" ]; then
#     LABELER="chair"
# fi

# if [ "$LABELER" = "chair" ]; then
#     OUTPUT="${OUTPUT:-coco-output/${MODEL}/4000COCO}"
# else
#     OUTPUT="${OUTPUT:-coco-output/${MODEL}/4000COCO}"
# fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
GENERATION_DEVICES="${GENERATION_DEVICES:-cuda:0 cuda:1}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
OPENAI_PROXY="${OPENAI_PROXY:-http://127.0.0.1:12598}"
export CUDA_VISIBLE_DEVICES

if [ "$LABELER" = "gpt4o" ] \
   && [ -z "${OPENAI_API_KEY:-}" ] \
   && [ -z "${GITHUB_MODELS_TOKEN:-}" ] \
   && [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "[run.sh] GPT-4o labeling needs OPENAI_API_KEY, GITHUB_MODELS_TOKEN, or GITHUB_TOKEN."
    echo '  export OPENAI_API_KEY="your-openai-api-key"'
    echo '  export GITHUB_MODELS_TOKEN="your-github-pat-with-models-scope"'
    exit 1
fi

echo "[run.sh] LABELER=${LABELER}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, DEVICE=${DEVICE}, GENERATION_DEVICES=${GENERATION_DEVICES}, FEATURE_DEVICES=${FEATURE_DEVICES}"
echo "[run.sh] OUTPUT=${OUTPUT}"

if [ "$LABELER" = "gpt4o" ]; then
    # Step 1: Generate descriptions + GPT-4o labeling
    cmd=(
        python scripts/generate_and_label.py
        --model "$MODEL"
        --config "$CONFIG"
        --output-dir "$OUTPUT"
        --device "$DEVICE"
        --generation-devices $GENERATION_DEVICES
        --resume
    )
    if [ -n "$OPENAI_PROXY" ]; then
        cmd+=(--openai-proxy "$OPENAI_PROXY")
    fi
    "${cmd[@]}"
elif [ "$LABELER" = "chair" ]; then
    # Step 1: Generate/reuse descriptions + CHAIR-style COCO labeling
    python coco-labeling/label_coco.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT" \
        --device "$DEVICE" \
        --generation-devices $GENERATION_DEVICES \
        --resume
else
    echo "[run.sh] Unsupported LABELER=${LABELER}. Use LABELER=gpt4o or LABELER=chair."
    exit 1
fi

# Step 2: Extract DGST-T features
python scripts/extract_features.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE" \
    --feature-devices $FEATURE_DEVICES \
    --resume

# Step 3: Train classifiers + evaluate
python scripts/train_and_eval.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT"
