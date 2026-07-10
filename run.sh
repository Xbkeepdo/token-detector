#!/bin/bash
# Full pipeline for one model. Replace MODEL and paths as needed.
set -euo pipefail

MODEL="${MODEL:-internvl_2_5_8b}"          # or: internvl_2_5_8b, qwen2_5_vl_7b, llava_1_5_7b，llava_onevision_1_5_8b
CONFIG="${CONFIG:-configs/model_configs_unified.yaml}"
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO500-vp}"
CHAIR_CACHE="${CHAIR_CACHE:-outputs/chair_cache/coco_val2014_chair.pkl}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
GENERATION_DEVICES="${GENERATION_DEVICES:-cuda:0 cuda:1}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
TRAINER="${TRAINER:-torch_mlp}"          # torch_mlp by default; use TRAINER=sklearn for old XGB/RF training
TORCH_PROBE_DEVICE="${TORCH_PROBE_DEVICE:-$DEVICE}"
TORCH_PROBE_ARGS="${TORCH_PROBE_ARGS:-}"
DEFAULT_PYTHON="/opt/conda/private/envs/vicr/bin/python"
if [ -x "$DEFAULT_PYTHON" ]; then
    PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
export CUDA_VISIBLE_DEVICES

echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, DEVICE=${DEVICE}, GENERATION_DEVICES=${GENERATION_DEVICES}, FEATURE_DEVICES=${FEATURE_DEVICES}"
echo "[run.sh] OUTPUT=${OUTPUT}"
echo "[run.sh] CHAIR_CACHE=${CHAIR_CACHE}"
echo "[run.sh] PYTHON_BIN=${PYTHON_BIN}"
echo "[run.sh] TRAINER=${TRAINER}, TORCH_PROBE_DEVICE=${TORCH_PROBE_DEVICE}"

FEATURE_SETS="${FEATURE_SETS:-}"
if [ -z "$FEATURE_SETS" ]; then
    FEATURE_SETS="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r") as handle:
    config = yaml.safe_load(handle) or {}

experiment = config.get("experiment") or {}
mode = str(experiment.get("mode", "")).strip().lower()
feature_sets = experiment.get("feature_sets")

values = None
if isinstance(feature_sets, dict):
    values = feature_sets.get(mode) or feature_sets.get("default")
elif isinstance(feature_sets, list):
    values = feature_sets

if not values:
    values = ["risk", "target_cosine", "risk+target_cosine"]

print(" ".join(str(item) for item in values))
PY
)"
fi
echo "[run.sh] FEATURE_SETS=${FEATURE_SETS}"

# Step 1: Generate/reuse descriptions + CHAIR-style COCO labeling
"$PYTHON_BIN" coco-labeling/label_coco.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --chair-cache "$CHAIR_CACHE" \
    --device "$DEVICE" \
    --generation-devices $GENERATION_DEVICES \
    --resume

# Step 2: Extract DGST-T features
"$PYTHON_BIN" scripts/extract_features.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE" \
    --feature-devices $FEATURE_DEVICES \
    --resume

# Step 3: Train risk / target-cosine classifiers + evaluate
if [ "$TRAINER" = "torch_mlp" ] || [ "$TRAINER" = "torch_probe" ]; then
    "$PYTHON_BIN" scripts/train_torch_probe_feature_sets.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT" \
        --feature-sets $FEATURE_SETS \
        --device "$TORCH_PROBE_DEVICE" \
        --positive-class real \
        $TORCH_PROBE_ARGS
elif [ "$TRAINER" = "sklearn" ] || [ "$TRAINER" = "xgb_rf" ]; then
    "$PYTHON_BIN" scripts/train_feature_sets.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output-dir "$OUTPUT" \
        --feature-sets $FEATURE_SETS
else
    echo "[run.sh] Unknown TRAINER=${TRAINER}; expected torch_mlp, torch_probe, sklearn, or xgb_rf." >&2
    exit 2
fi
