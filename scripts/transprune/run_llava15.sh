#!/usr/bin/env bash
# LLaVA-v1.5-7B TransPrune image-benchmark eval.
# See Docs/Implementation/04-transprune-reproduction.md.
#
# One process sees one GPU. load_pretrained_model uses device_map=auto, and
# TransPrune calls the next layer's q_proj directly, so a sharded 7B crashes
# at the first prune.
#
# Usage:
#   bash scripts/transprune/run_llava15.sh gqa
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/transprune/run_llava15.sh gqa
#   METHOD=llava bash scripts/transprune/run_llava15.sh gqa
#   bash scripts/transprune/run_llava15.sh all
#
# Tasks: gqa mmbench mme pope textvqa scienceqa vqav2 seed all
# GQA is the only benchmark with local images and a local Accuracy scorer.
# MMBench / VQAv2 / SEED write submission files; they do not print paper scores.
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

TP="$ROOT/ThirdParty/TransPrune"
DATA="$ROOT/weights"
EVAL_ROOT="$ROOT/ThirdParty/LLaVA/playground/data/eval"
EVAL_EXTRA="${EVAL_EXTRA:-$DATA/eval}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/transprune-v1.5}"
CKPT="${CKPT:-llava-v1.5-7b}"
METHOD="${METHOD:-transprune}"
SNAP="$DATA/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
# Loader only inspects the last path component. A shell that still exports
# MODEL=.../haotian-from-hf must not be passed through.
LINK="$DATA/llava-v1.5-7b"

TASK="${1:-gqa}"

if [[ ! -s "$SNAP/config.json" || ! -s "$SNAP/mm_projector.bin" ]]; then
  echo "FAIL: missing LLaVA-v1.5 snapshot at $SNAP" >&2
  exit 1
fi
ln -sfn "$SNAP" "$LINK"
case "$(basename "${MODEL:-$LINK}")" in
  *llava*|*LLaVA*|*Llava*) MODEL="${MODEL:-$LINK}" ;;
  *)
    echo "NOTE: MODEL=${MODEL:-unset} last component is not a LLaVA name; using $LINK" >&2
    MODEL="$LINK"
    ;;
esac

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate transprune
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"

cd "$TP"
mkdir -p "$RUN_ROOT"

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS="${#GPULIST[@]}"
if (( CHUNKS < 1 )); then
  echo "FAIL: CUDA_VISIBLE_DEVICES is empty" >&2
  exit 1
fi

COMMON_ARGS=(
  --model-path "$MODEL"
  --temperature 0
  --method "$METHOD"
  --accmulating_layer_list '[7,8,9,10,11,12]'
  --pruning_layer_list '[7,9,12]'
  --image_token_ratio_list '[0.875,0.625,0.125]'
  --conv-mode vicuna_v1
)

write_manifest() {
  local out="$1"
  local task_name="$2"
  mkdir -p "$out"
  {
    echo "date: $(date -Is)"
    echo "method: $METHOD"
    echo "model: $MODEL"
    echo "snapshot: $SNAP"
    echo "gpus: $GPU_LIST"
    echo "chunks: $CHUNKS"
    echo "task: $task_name"
    git -C "$TP" rev-parse HEAD
    python - <<'PY'
import torch, transformers, torchvision, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("cuda", torch.cuda.is_available())
PY
  } > "$out/command.txt"
  python -m pip freeze > "$out/env.pip-freeze.txt"
  nvidia-smi > "$out/nvidia-smi.txt" || true
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

# module extra-args...  -- answers land in $OUT/${CHUNKS}_${IDX}.jsonl
launch_chunks() {
  local module="$1"
  shift
  local idx pid
  local pids=()
  for idx in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$idx]}" python -m "$module" \
      "${COMMON_ARGS[@]}" \
      "$@" \
      --answers-file "$OUT/${CHUNKS}_${idx}.jsonl" \
      --num-chunks "$CHUNKS" \
      --chunk-idx "$idx" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_gqa() {
  local gqa="$EVAL_ROOT/gqa"
  require_file "$gqa/llava_gqa_testdev_balanced.jsonl"
  require_file "$gqa/data/images"
  require_file "$gqa/data/testdev_balanced_questions.json"
  OUT="$RUN_ROOT/gqa/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$gqa/llava_gqa_testdev_balanced.jsonl" \
    --image-folder "$gqa/data/images"
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  mkdir -p "$OUT/answers/gqa"
  cp "$OUT/$CKPT.jsonl" "$OUT/answers/gqa/merge.jsonl"
  python "$ROOT/scripts/repro/score_visionzip.py" \
    --run-dir "$OUT" --eval-root "$EVAL_ROOT" --tasks gqa \
    | tee "$OUT/evaluator.log"
}

run_mmbench() {
  local q="$EVAL_EXTRA/mmbench/mmbench_dev_20230712.tsv"
  require_file "$q"
  OUT="$RUN_ROOT/mmbench/$METHOD"
  mkdir -p "$OUT/upload"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_mmbench \
    --question-file "$q" \
    --single-pred-prompt
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  python scripts/convert_mmbench_for_submission.py \
    --annotation-file "$q" \
    --result-dir "$OUT" \
    --upload-dir "$OUT/upload" \
    --experiment "$CKPT"
  echo "MMBench upload: $OUT/upload/$CKPT.xlsx (official server scores this; no local accuracy)"
}

run_mme() {
  local mme="$EVAL_EXTRA/MME"
  require_file "$mme/llava_mme.jsonl"
  require_file "$mme/MME_Benchmark_release_version"
  OUT="$RUN_ROOT/mme/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$mme/llava_mme.jsonl" \
    --image-folder "$mme/MME_Benchmark_release_version"
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  if [[ ! -f "$mme/convert_answer_to_mme.py" || ! -f "$mme/eval_tool/calculation.py" ]]; then
    echo "MME answers written to $OUT/$CKPT.jsonl"
    echo "SKIP score: place official convert_answer_to_mme.py and eval_tool/ under $mme"
    return 0
  fi
  mkdir -p "$mme/answers"
  cp "$OUT/$CKPT.jsonl" "$mme/answers/$CKPT.jsonl"
  (
    cd "$mme"
    python convert_answer_to_mme.py --experiment "$CKPT"
    cd eval_tool
    python calculation.py --results_dir "answers/$CKPT"
  ) | tee "$OUT/evaluator.log"
}

run_pope() {
  local pope="$EVAL_EXTRA/pope"
  require_file "$pope/llava_pope_test.jsonl"
  require_file "$pope/val2014"
  require_file "$pope/coco"
  OUT="$RUN_ROOT/pope/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$pope/llava_pope_test.jsonl" \
    --image-folder "$pope/val2014"
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  python llava/eval/eval_pope.py \
    --annotation-dir "$pope/coco" \
    --question-file "$pope/llava_pope_test.jsonl" \
    --result-file "$OUT/$CKPT.jsonl" \
    | tee "$OUT/evaluator.log"
}

run_textvqa() {
  local tv="$EVAL_EXTRA/textvqa"
  require_file "$tv/llava_textvqa_val_v051_ocr.jsonl"
  require_file "$tv/train_images"
  require_file "$tv/TextVQA_0.5.1_val.json"
  OUT="$RUN_ROOT/textvqa/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$tv/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "$tv/train_images"
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  python -m llava.eval.eval_textvqa \
    --annotation-file "$tv/TextVQA_0.5.1_val.json" \
    --result-file "$OUT/$CKPT.jsonl" \
    | tee "$OUT/evaluator.log"
}

run_scienceqa() {
  local sqa="$EVAL_EXTRA/scienceqa"
  require_file "$sqa/llava_test_CQM-A.json"
  require_file "$sqa/images/test"
  require_file "$sqa/pid_splits.json"
  require_file "$sqa/problems.json"
  OUT="$RUN_ROOT/scienceqa/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_science \
    --question-file "$sqa/llava_test_CQM-A.json" \
    --image-folder "$sqa/images/test" \
    --single-pred-prompt
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  python llava/eval/eval_science_qa.py \
    --base-dir "$sqa" \
    --result-file "$OUT/$CKPT.jsonl" \
    --output-file "$OUT/${CKPT}_output.json" \
    --output-result "$OUT/${CKPT}_result.json" \
    | tee "$OUT/evaluator.log"
}

run_vqav2() {
  local vqa="$EVAL_EXTRA/vqav2"
  local split="llava_vqav2_mscoco_test-dev2015"
  require_file "$vqa/$split.jsonl"
  require_file "$vqa/test2015"
  require_file "$vqa/llava_vqav2_mscoco_test2015.jsonl"
  OUT="$RUN_ROOT/vqav2/$METHOD"
  mkdir -p "$OUT" "$vqa/answers/$split/$CKPT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$vqa/$split.jsonl" \
    --image-folder "$vqa/test2015"
  merge_chunks "$OUT" "$vqa/answers/$split/$CKPT/merge.jsonl"
  cp "$vqa/answers/$split/$CKPT/merge.jsonl" "$OUT/$CKPT.jsonl"
  python scripts/convert_vqav2_for_submission.py \
    --split "$split" --ckpt "$CKPT" --dir "$vqa"
  echo "VQAv2 submission: $vqa/answers_upload/$split/$CKPT.json (eval server; no local accuracy)"
}

run_seed() {
  local seed="$EVAL_EXTRA/seed_bench"
  require_file "$seed/llava-seed-bench.jsonl"
  require_file "$seed/SEED-Bench.json"
  OUT="$RUN_ROOT/seed/$METHOD"
  mkdir -p "$OUT"
  write_manifest "$OUT" "${OUT#"$RUN_ROOT/"}"
  launch_chunks llava.eval.model_vqa_loader \
    --question-file "$seed/llava-seed-bench.jsonl" \
    --image-folder "$seed"
  merge_chunks "$OUT" "$OUT/$CKPT.jsonl"
  python scripts/convert_seed_for_submission.py \
    --annotation-file "$seed/SEED-Bench.json" \
    --result-file "$OUT/$CKPT.jsonl" \
    --result-upload-file "$OUT/${CKPT}_upload.jsonl"
  echo "SEED upload: $OUT/${CKPT}_upload.jsonl (official eval; no local SEED-I accuracy)"
}

case "$TASK" in
  gqa) run_gqa ;;
  mmbench) run_mmbench ;;
  mme) run_mme ;;
  pope) run_pope ;;
  textvqa) run_textvqa ;;
  scienceqa) run_scienceqa ;;
  vqav2) run_vqav2 ;;
  seed) run_seed ;;
  all)
    run_gqa
    run_mmbench
    run_mme
    run_pope
    run_textvqa
    run_scienceqa
    run_vqav2
    run_seed
    ;;
  *)
    echo "FAIL: unknown task '$TASK'" >&2
    echo "tasks: gqa mmbench mme pope textvqa scienceqa vqav2 seed all" >&2
    exit 1
    ;;
esac

echo "OK: $TASK ($METHOD) → $RUN_ROOT"
