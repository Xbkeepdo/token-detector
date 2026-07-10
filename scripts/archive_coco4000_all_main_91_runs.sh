#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MODEL" >&2
  exit 2
fi

MODEL="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$MODEL" in
  llava_1_5_7b) MODEL_LABEL="LLaVA-1.5-7B" ;;
  qwen2_5_vl_7b) MODEL_LABEL="Qwen2.5-VL-7B" ;;
  internvl_2_5_8b) MODEL_LABEL="InternVL2.5-8B" ;;
  *)
    echo "Unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

BASE="outputs/${MODEL}/COCO4000-all"
ARCHIVE="${BASE}/torchmlp-main-seed3-91"

for SEED in 42 43 44; do
  SOURCE="${BASE}/torchmlp-main-seed${SEED}"
  RESULT="${SOURCE}/results/${MODEL}_selected_feature_sets.json"
  if [[ ! -f "$RESULT" ]]; then
    echo "Seed $SEED is incomplete: missing $RESULT" >&2
    exit 1
  fi
  /opt/conda/private/envs/vicr/bin/python - "$RESULT" "$SEED" <<'PY'
import json
import sys

path, expected_seed = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    results = json.load(handle)
complete = []
for feature_set, classifiers in results.items():
    metrics = classifiers.get("torch_probe") if isinstance(classifiers, dict) else None
    if not isinstance(metrics, dict):
        continue
    actual_seed = metrics.get("best_params", {}).get("seed")
    if actual_seed is not None and int(actual_seed) != expected_seed:
        raise SystemExit(
            f"{path}: {feature_set} has seed {actual_seed}, expected {expected_seed}"
        )
    complete.append(feature_set)
if len(complete) != 42:
    raise SystemExit(f"{path}: expected 42 completed feature sets, found {len(complete)}")
PY
  if [[ -e "${ARCHIVE}/seed${SEED}" ]]; then
    echo "Archive destination already exists: ${ARCHIVE}/seed${SEED}" >&2
    exit 1
  fi
done

mkdir -p "$ARCHIVE"
for SEED in 42 43 44; do
  mv "${BASE}/torchmlp-main-seed${SEED}" "${ARCHIVE}/seed${SEED}"
  ln -sfn ../../features.pkl "${ARCHIVE}/seed${SEED}/features.pkl"
  ln -sfn ../../image_splits.json "${ARCHIVE}/seed${SEED}/image_splits.json"
done

/opt/conda/private/envs/vicr/bin/python scripts/summarize_torch_probe_seed_runs.py \
  --models "$MODEL" \
  --model-labels "$MODEL_LABEL" \
  --seeds 42 43 44 \
  --run-template "outputs/{model}/COCO4000-all/torchmlp-main-seed3-91/seed{seed}/results/{model}_selected_feature_sets.json" \
  --output-prefix "${ARCHIVE}/${MODEL}_torchmlp_main_3seeds_91_summary" \
  --title "${MODEL_LABEL} COCO4000-all torch MLP main experiment (9:1, val=test)"

echo "[Archive91] Saved completed runs under $ARCHIVE"
