#!/bin/bash
# Full generation -> extraction -> training pipeline for one model.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="${MODEL:-qwen2_5_vl_7b}"  # llava_1_5_7b  qwen2_5_vl_7b
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO4000-512-CAPPEDSWEEP}"
CONFIG="${CONFIG:-configs/model_configs_server_fj01.yaml}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
GENERATION_DEVICES="${GENERATION_DEVICES:-cuda:0 cuda:1}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
export DGST_COST_VARIANT_EMD_WORKERS="${DGST_COST_VARIANT_EMD_WORKERS:-16}"
export DGST_FOUR_GATE_PREP_CACHE="${DGST_FOUR_GATE_PREP_CACHE:-1}"
export DGST_FOUR_GATE_EMD_DEDUP="${DGST_FOUR_GATE_EMD_DEDUP:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CHAIR_CACHE="${CHAIR_CACHE:-outputs/chair_cache/coco_val2014_chair.pkl}"
NLTK_DATA="${NLTK_DATA:-$HOME/userdata/nltk_data}"
REUSE_GENERATIONS_FROM="${REUSE_GENERATIONS_FROM:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Activate the intended environment or set PYTHON_BIN=/path/to/python." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES NLTK_DATA
# Required by deterministic CUDA matrix multiplication for reproducible seeds.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

LABEL_ARGS=()
[[ -n "$REUSE_GENERATIONS_FROM" ]] && \
    LABEL_ARGS+=(--reuse-generations-from "$REUSE_GENERATIONS_FROM")

# Step 1: Generate descriptions + CHAIR-style COCO labeling
"$PYTHON_BIN" scripts/generate_and_label.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --chair-cache "$CHAIR_CACHE" \
    --device "$DEVICE" \
    --generation-devices $GENERATION_DEVICES \
    "${LABEL_ARGS[@]}" \
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

# Step 4: Optionally train every YAML-configured source-tau x Top-K sweep.
# The Python entrypoint exits immediately when the YAML switch is disabled.
"$PYTHON_BIN" scripts/train_source_tau_transport_topk_sweep.py \
    --output-dir "$OUTPUT" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --if-enabled
