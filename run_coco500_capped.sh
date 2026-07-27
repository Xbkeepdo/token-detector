#!/bin/bash
# Extract and train the retained COCO500 capped-sweep subset without regenerating captions.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

MODEL="${MODEL:-qwen2_5_vl_7b}"
OUTPUT="${OUTPUT:-outputs/${MODEL}/COCO500-capped}"
CONFIG="${CONFIG:-configs/model_configs_server_fj01_coco500_capped.yaml}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
FEATURE_DEVICES="${FEATURE_DEVICES:-cuda:0 cuda:1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES
export DGST_COST_VARIANT_EMD_WORKERS="${DGST_COST_VARIANT_EMD_WORKERS:-16}"
export DGST_FOUR_GATE_PREP_CACHE="${DGST_FOUR_GATE_PREP_CACHE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
generations = json.load((output / "generations.json").open(encoding="utf-8"))
labeling = json.load((output / "labeling.json").open(encoding="utf-8"))
splits = json.load((output / "image_splits.json").open(encoding="utf-8"))
generation_ids = {int(value) for value in generations}
label_ids = {int(value) for value in labeling}
train_ids = {int(value) for value in splits.get("train", [])}
val_ids = {int(value) for value in splits.get("val", [])}
test_ids = {int(value) for value in splits.get("test", [])}
if len(generation_ids) != 500 or generation_ids != label_ids:
    raise RuntimeError("COCO500-capped must contain 500 aligned generation/label IDs.")
if len(train_ids) != 400 or val_ids or len(test_ids) != 100:
    raise RuntimeError("COCO500-capped must retain the strict 400/0/100 split.")
if train_ids & test_ids or generation_ids != train_ids | test_ids:
    raise RuntimeError("COCO500-capped split IDs are overlapping or incomplete.")
print("[COCO500-capped] Validated 500 captions and strict 400/100 split.")
PY

"$PYTHON_BIN" scripts/extract_features.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE" \
    --feature-devices $FEATURE_DEVICES \
    --resume

"$PYTHON_BIN" scripts/train_and_eval.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    --device "$DEVICE"

"$PYTHON_BIN" scripts/train_source_tau_transport_topk_sweep.py \
    --output-dir "$OUTPUT" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --if-enabled
