#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Activate the intended environment or set PYTHON_BIN=/path/to/python." >&2
    exit 2
fi
MODEL="${MODEL:-qwen3_vl_8b}"
DATASET="${DATASET:-pope}"                 # pope | clevr_exist_9k | amber_discriminative
CONFIG="${CONFIG:-configs/model_configs_unified.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
GENERATION_DEVICES="${GENERATION_DEVICES:-cuda:0 cuda:1}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
BASELINE_LABEL_PROTOCOLS="${BASELINE_LABEL_PROTOCOLS:-}"
export CUDA_VISIBLE_DEVICES
# Required by PyTorch deterministic CUDA matrix multiplication on CUDA >= 10.2.
# Keep this set before any Python process imports torch.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ -z "$BASELINE_LABEL_PROTOCOLS" ]]; then
    BASELINE_LABEL_PROTOCOLS="$("$PYTHON_BIN" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(" ".join(c.get("qa_benchmarks",{}).get("baseline_label_protocols",["object_hallucination_yes_only"])))' "$CONFIG")"
fi

# Step 1: Prepare one immutable outer image-level 8:2 question split.
if [[ "$DATASET" == "pope" ]]; then
    PREPARE_DATASET="pope"
elif [[ "$DATASET" == "clevr_exist_5k" || "$DATASET" == "clevr_exist_9k" ]]; then
    PREPARE_DATASET="clevr"
elif [[ "$DATASET" == "amber_discriminative" ]]; then
    PREPARE_DATASET="amber"
else
    echo "DATASET must be pope, clevr_exist_9k, clevr_exist_5k, or amber_discriminative" >&2
    exit 2
fi
"$PYTHON_BIN" scripts/prepare_qa_benchmarks.py \
    --dataset "$PREPARE_DATASET" \
    --config "$CONFIG"

# Step 2: Generate, label, and jointly extract every YAML-selected family.
"$PYTHON_BIN" scripts/qa_pipeline.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --generation-devices $GENERATION_DEVICES \
    --feature-devices $FEATURE_DEVICES \
    --baseline-label-protocols $BASELINE_LABEL_PROTOCOLS \
    --stage all \
    --resume

# Step 3: Train DGST and ADS+CGC probes for both labels at the selected position.
"$PYTHON_BIN" scripts/train_qa_probes.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --device "$DEVICE"

# Step 4: Train the baseline features already produced by the shared forward.
for protocol in $BASELINE_LABEL_PROTOCOLS; do
    "$PYTHON_BIN" scripts/train_qa_baselines.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --label-protocol "$protocol"
done

# Step 5: Write one comparison table for each trained label protocol.
if [[ -f scripts/summarize_qa_comparison.py ]]; then
    for protocol in $BASELINE_LABEL_PROTOCOLS; do
        "$PYTHON_BIN" scripts/summarize_qa_comparison.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --label-protocol "$protocol" \
            --config "$CONFIG"
    done
fi
