#!/usr/bin/env bash
# LLaVA-v1.5-7B CDPruner image eval on the local GQA / POPE / MME sets.
# See Docs/Implementation/05-cdpruner-reproduction.md.
#
# One process sees one GPU. Answers are scored with scripts/repro/score_visionzip.py
# (GQA Accuracy does not use the official eval.py scene-graph path).
#
# Usage:
#   bash scripts/cdpruner/run_llava15.sh gqa
#   CUDA_VISIBLE_DEVICES=0,1 TOKEN=64 bash scripts/cdpruner/run_llava15.sh pope
#   bash scripts/cdpruner/run_llava15.sh all
#
# Tasks: gqa pope mme all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"

export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CD="$ROOT/ThirdParty/CDPruner"
EVAL_ROOT="$ROOT/ThirdParty/LLaVA/playground/data/eval"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/cdpruner-v1.5}"
TOKEN="${TOKEN:-128}"
SNAP="$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
LINK="$ROOT/weights/llava-v1.5-7b"
TASK="${1:-gqa}"

if [[ ! -s "$SNAP/config.json" || ! -s "$SNAP/mm_projector.bin" ]]; then
  echo "FAIL: missing LLaVA-v1.5 snapshot at $SNAP" >&2
  exit 1
fi
ln -sfn "$SNAP" "$LINK"
MODEL="$LINK"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cdpruner
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"

cd "$CD"
mkdir -p "$RUN_ROOT"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS="${#GPULIST[@]}"
if (( CHUNKS < 1 )); then
  echo "FAIL: CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi

write_manifest() {
  local out="$1"
  local task_name="$2"
  mkdir -p "$out"
  {
    echo "date: $(date -Is)"
    echo "method: cdpruner"
    echo "model: $MODEL"
    echo "snapshot: $SNAP"
    echo "visual_token_num: $TOKEN"
    echo "gpus: $GPU_LIST"
    echo "chunks: $CHUNKS"
    echo "task: $task_name"
    git -C "$CD" rev-parse HEAD
    python - <<'PY'
import torch, transformers, torchvision, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
PY
  } > "$out/command.txt"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "FAIL: missing $path" >&2
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
  local idx pid
  local pids=()
  for idx in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" python -m llava.eval.model_vqa_loader \
      --model-path "$MODEL" \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      --visual_token_num "$TOKEN" \
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
  python "$ROOT/scripts/repro/score_visionzip.py" \
    --run-dir "$OUT" --eval-root "$EVAL_ROOT" --tasks "$task_name" \
    | tee "$OUT/evaluator.log"
}

run_gqa() {
  local gqa="$EVAL_ROOT/gqa"
  require_file "$gqa/llava_gqa_testdev_balanced.jsonl"
  require_file "$gqa/data/images"
  require_file "$gqa/data/testdev_balanced_questions.json"
  OUT="$RUN_ROOT/gqa/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" "gqa"
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
  require_file "$pope/coco"
  OUT="$RUN_ROOT/pope/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" "pope"
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
  require_file "$mme/convert_answer_to_mme.py"
  OUT="$RUN_ROOT/mme/vtn_${TOKEN}"
  mkdir -p "$OUT"
  write_manifest "$OUT" "mme"
  launch_chunks \
    --question-file "$mme/llava_mme.jsonl" \
    --image-folder "$mme/MME_Benchmark_release_version"
  merge_chunks "$OUT" "$OUT/merge.jsonl"
  score_task mme
}

case "$TASK" in
  gqa) run_gqa ;;
  pope) run_pope ;;
  mme) run_mme ;;
  all)
    run_gqa
    run_pope
    run_mme
    ;;
  *)
    echo "FAIL: unknown task '$TASK'" >&2
    echo "tasks: gqa pope mme all" >&2
    exit 1
    ;;
esac

echo "OK: $TASK token=$TOKEN → $RUN_ROOT"
