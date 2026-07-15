#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/project/token-detector

PY=/opt/conda/envs/td/bin/python
ROOT=outputs/qa_benchmarks
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

wait_for_completion() {
  local model="$1"
  local dataset="$2"
  local log="$3"
  local marker="[qa_pipeline] Complete: ${model}/${dataset}/all"
  until grep -Fq "$marker" "$log" 2>/dev/null; do
    sleep 30
  done
}

validate_run() {
  local model="$1"
  local dataset="$2"
  local expected="$3"
  local extra=()
  if [[ "$dataset" == "pope" ]]; then
    extra+=(--require-object-cgc)
  fi
  "$PY" scripts/validate_qa_artifacts.py \
    --run-dir "$ROOT/$model/$dataset" \
    --expected "$expected" \
    "${extra[@]}"
}

run_pipeline() {
  local model="$1"
  local dataset="$2"
  local gpu="$3"
  local log="$4"
  bash scripts/run_qa_full_worker.sh "$model" "$dataset" "$gpu" > "$log" 2>&1
}

echo "[coordinator] waiting for initial LLaVA/Qwen POPE jobs"
wait_for_completion llava_1_5_7b pope "$LOGS/llava_pope.log"
wait_for_completion qwen2_5_vl_7b pope "$LOGS/qwen_pope.log"
validate_run llava_1_5_7b pope 9000
validate_run qwen2_5_vl_7b pope 9000

echo "[coordinator] running InternVL POPE"
run_pipeline internvl_2_5_8b pope 0 "$LOGS/internvl_pope.log"
validate_run internvl_2_5_8b pope 9000

echo "[coordinator] all POPE extraction complete; starting CLEVR"
run_pipeline llava_1_5_7b clevr_exist_5k 0 "$LOGS/llava_clevr.log" &
pid_llava=$!
run_pipeline qwen2_5_vl_7b clevr_exist_5k 1 "$LOGS/qwen_clevr.log" &
pid_qwen=$!
wait "$pid_llava"
wait "$pid_qwen"
validate_run llava_1_5_7b clevr_exist_5k 5000
validate_run qwen2_5_vl_7b clevr_exist_5k 5000

run_pipeline internvl_2_5_8b clevr_exist_5k 0 "$LOGS/internvl_clevr.log"
validate_run internvl_2_5_8b clevr_exist_5k 5000

echo "[coordinator] summarizing original-model results"
for model in llava_1_5_7b qwen2_5_vl_7b internvl_2_5_8b; do
  for dataset in pope clevr_exist_5k; do
    "$PY" scripts/summarize_qa_generations.py --run-dir "$ROOT/$model/$dataset"
  done
done

echo "[coordinator] starting strict 8:1:1 probes"
(
  export CUDA_VISIBLE_DEVICES=0
  "$PY" scripts/train_qa_probes.py --model llava_1_5_7b --dataset pope --output-root "$ROOT"
  "$PY" scripts/train_qa_probes.py --model llava_1_5_7b --dataset clevr_exist_5k --output-root "$ROOT"
  "$PY" scripts/train_qa_probes.py --model internvl_2_5_8b --dataset pope --output-root "$ROOT"
  "$PY" scripts/train_qa_probes.py --model internvl_2_5_8b --dataset clevr_exist_5k --output-root "$ROOT"
) > "$LOGS/probes_gpu0.log" 2>&1 &
pid_probe0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  "$PY" scripts/train_qa_probes.py --model qwen2_5_vl_7b --dataset pope --output-root "$ROOT"
  "$PY" scripts/train_qa_probes.py --model qwen2_5_vl_7b --dataset clevr_exist_5k --output-root "$ROOT"
) > "$LOGS/probes_gpu1.log" 2>&1 &
pid_probe1=$!
wait "$pid_probe0"
wait "$pid_probe1"

echo "[coordinator] running paper-compatible LLaVA POPE ADS+CGC five-fold"
"$PY" scripts/train_pope_paper_protocol.py \
  --run-dir "$ROOT/llava_1_5_7b/pope" \
  > "$LOGS/pope_paper_protocol.log" 2>&1

echo "[coordinator] ALL COMPLETE"
