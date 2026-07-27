#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL [DEVICE]" >&2
  exit 2
fi

MODEL="$1"
DEVICE="${2:-cuda:0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

case "$MODEL" in
  llava_1_5_7b) MODEL_LABEL="LLaVA-1.5-7B" ;;
  qwen2_5_vl_7b) MODEL_LABEL="Qwen2.5-VL-7B" ;;
  internvl_2_5_8b) MODEL_LABEL="InternVL2.5-8B" ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

EXPERIMENT_DIR="outputs/${MODEL}/COCO4000-all/torchmlp-main-seed3-811"
if [[ ! -f "$EXPERIMENT_DIR/image_splits.json" ]]; then
  echo "Missing strict split: $EXPERIMENT_DIR/image_splits.json" >&2
  exit 1
fi

FEATURE_SETS=(
  "risk_geo_raw"
  "risk_geo_cap085"
  "risk_relative_vll_cost_qmatch"
  "risk_relative_vll_cost_qmatch_capped_topmass_085"
  "risk_relative_vll_cost_geo_stateupd_lu1"
  "risk_visual_prompt_relative_vll_cost_geo"
  "risk_visual_prompt_relative_vll_cost_geo_capped_topmass_085"
  "risk_visual_prompt_relative_vll_cost_qmatch"
  "risk_visual_prompt_relative_vll_cost_qmatch_capped_topmass_085"
  "risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1"
  "visualcosine_raw"
  "cosine16"
  "visualcosine_cap085"
  "hprecosine_raw"
  "hprecosine16"
  "hprecosine_cap085"
  "vp_target_cosine"
  "vp_cosine16"
  "target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
  "vp_hprecosine"
  "vp_hprecosine16"
  "vp_hprecosine_cap085"
  "risk_geo_raw+visualcosine_raw"
  "risk_geo_raw+hprecosine_raw"
  "risk_geo_cap085+visualcosine_cap085"
  "risk_geo_cap085+hprecosine_cap085"
  "risk_relative_vll_cost_qmatch+visualcosine_raw"
  "risk_relative_vll_cost_qmatch+hprecosine_raw"
  "risk_relative_vll_cost_qmatch_capped_topmass_085+visualcosine_cap085"
  "risk_relative_vll_cost_qmatch_capped_topmass_085+hprecosine_cap085"
  "risk_relative_vll_cost_geo_stateupd_lu1+visualcosine_raw"
  "risk_relative_vll_cost_geo_stateupd_lu1+hprecosine_raw"
  "risk_visual_prompt_relative_vll_cost_geo+vp_target_cosine"
  "risk_visual_prompt_relative_vll_cost_geo+vp_hprecosine"
  "risk_visual_prompt_relative_vll_cost_geo_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
  "risk_visual_prompt_relative_vll_cost_geo_capped_topmass_085+vp_hprecosine_cap085"
  "risk_visual_prompt_relative_vll_cost_qmatch+vp_target_cosine"
  "risk_visual_prompt_relative_vll_cost_qmatch+vp_hprecosine"
  "risk_visual_prompt_relative_vll_cost_qmatch_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085"
  "risk_visual_prompt_relative_vll_cost_qmatch_capped_topmass_085+vp_hprecosine_cap085"
  "risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1+vp_target_cosine"
  "risk_visual_prompt_relative_vll_cost_geo_stateupd_lu1+vp_hprecosine"
)

for SEED in 42 43 44; do
  RUN_DIR="${EXPERIMENT_DIR}/seed${SEED}"
  if [[ ! -e "$RUN_DIR/features.pkl" || ! -e "$RUN_DIR/image_splits.json" ]]; then
    echo "Missing prepared inputs in $RUN_DIR" >&2
    exit 1
  fi
  echo "[Main811] model=$MODEL seed=$SEED device=$DEVICE"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    scripts/train_torch_probe_feature_sets.py \
    --model "$MODEL" \
    --config configs/model_configs_coco4000_all.yaml \
    --output-dir "$RUN_DIR" \
    --feature-sets "${FEATURE_SETS[@]}" \
    --batch-size 256 \
    --seed "$SEED" \
    --positive-class real \
    --device "$DEVICE"
done

"$PYTHON_BIN" scripts/summarize_torch_probe_seed_runs.py \
  --models "$MODEL" \
  --model-labels "$MODEL_LABEL" \
  --seeds 42 43 44 \
  --run-template "outputs/{model}/COCO4000-all/torchmlp-main-seed3-811/seed{seed}/results/{model}_selected_feature_sets.json" \
  --output-prefix "${EXPERIMENT_DIR}/${MODEL}_torchmlp_main_3seeds_811_summary" \
  --title "${MODEL_LABEL} COCO4000-all torch MLP main experiment (8:1:1)"
