#!/usr/bin/env bash
# LLaVA-1.5 GQA / POPE / MME with a trained VLA-ACL scorer.
# TOKEN must match the K stored in SCORER. K=64 and K=32 are not trained here.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/root/Test/envs/vla-acl/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/scripts/vla_acl${PYTHONPATH:+:$PYTHONPATH}"
export VLA_ACL=1

LLAVA="$ROOT/ThirdParty/LLaVA"
EVAL_ROOT="$LLAVA/playground/data/eval"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/vla-acl-v1.5}"
TOKEN="${TOKEN:-128}"
SCORER="${SCORER:-$RUN_ROOT/train/k${TOKEN}/patch_scorer.pt}"
SNAP="$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
LINK="$ROOT/weights/llava-v1.5-7b"
TASK="${1:-gqa}"

if [[ ! -s "$SNAP/config.json" || ! -s "$SNAP/mm_projector.bin" ]]; then
  echo "FAIL: missing LLaVA-v1.5 snapshot at $SNAP" >&2
  exit 1
fi
if [[ ! -s "$SCORER" ]]; then
  echo "FAIL: no scorer at $SCORER" >&2
  echo "K=128 is the trained budget. K=64 and K=32 are not trained in this migration." >&2
  exit 1
fi
ln -sfn "$SNAP" "$LINK"
MODEL="$LINK"
export VLA_ACL_SCORER="$SCORER"
export VLA_ACL_TOKEN="$TOKEN"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS="${#GPULIST[@]}"

write_manifest() {
  local out="$1"
  local task_name="$2"
  mkdir -p "$out"
  {
    echo "date: $(date -Is)"
    echo "method: vla-acl"
    echo "model: $MODEL"
    echo "scorer: $SCORER"
    echo "visual_token_num: $TOKEN"
    echo "gpus: $GPU_LIST"
    echo "task: $task_name"
  } > "$out/command.txt"
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "FAIL: missing $1" >&2
    exit 1
  fi
}

merge_chunks() {
  local out="$1"
  local dest="$2"
  : > "$dest"
  local idx
  for idx in $(seq 0 $((CHUNKS - 1))); do
    cat "$out/${CHUNKS}_${idx}.jsonl" >> "$dest"
  done
}

launch_chunks() {
  local pids=()
  local idx
  for idx in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" "$PYTHON" -m llava.eval.model_vqa_loader \
      --model-path "$MODEL" \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      "$@" \
      --answers-file "$OUT/${CHUNKS}_${idx}.jsonl" \
      --num-chunks "$CHUNKS" \
      --chunk-idx "$idx" &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

score_task() {
  local task_name="$1"
  mkdir -p "$OUT/answers/$task_name"
  cp "$OUT/merge.jsonl" "$OUT/answers/$task_name/merge.jsonl"
  "$PYTHON" "$ROOT/scripts/repro/score_visionzip.py" \
    --run-dir "$OUT" --eval-root "$EVAL_ROOT" --tasks "$task_name" \
    | tee "$OUT/evaluator.log"
}

run_gqa() {
  local gqa="$EVAL_ROOT/gqa"
  require_file "$gqa/llava_gqa_testdev_balanced.jsonl"
  require_file "$gqa/data/images"
  OUT="$RUN_ROOT/gqa/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" gqa
  launch_chunks \
    --question-file "$gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "$gqa/data/images"
  merge_chunks "$OUT" "$OUT/merge.jsonl"
  score_task gqa
}

run_pope() {
  local pope="$EVAL_ROOT/pope"
  require_file "$pope/llava_pope_test.jsonl"
  require_file "$pope/val2014"
  OUT="$RUN_ROOT/pope/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" pope
  launch_chunks \
    --question-file "$pope/llava_pope_test.jsonl" \
    --image-folder "$pope/val2014"
  merge_chunks "$OUT" "$OUT/merge.jsonl"
  score_task pope
}

run_mme() {
  local mme="$EVAL_ROOT/MME"
  require_file "$mme/llava_mme.jsonl"
  require_file "$mme/MME_Benchmark_release_version"
  OUT="$RUN_ROOT/mme/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" mme
  launch_chunks \
    --question-file "$mme/llava_mme.jsonl" \
    --image-folder "$mme/MME_Benchmark_release_version"
  merge_chunks "$OUT" "$OUT/merge.jsonl"
  score_task mme
}

cd "$LLAVA"
case "$TASK" in
  gqa) run_gqa ;;
  pope) run_pope ;;
  mme) run_mme ;;
  all) run_gqa; run_pope; run_mme ;;
  *)
    echo "FAIL: unknown task '$TASK'" >&2
    exit 1
    ;;
esac
echo "OK: $TASK token=$TOKEN → $RUN_ROOT"
