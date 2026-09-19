#!/usr/bin/env bash
# Full VisionZip × LLaVA-1.5-7B retain-64 on GQA / MME / POPE (8-way chunked)
# plus official metric scoring into $RUN_DIR/metrics/.
#
# Usage:
#   bash scripts/repro/full_visionzip_llava15.sh
#   TASKS=pope,gqa bash scripts/repro/full_visionzip_llava15.sh
#   SCORE_ONLY=1 bash scripts/repro/full_visionzip_llava15.sh   # score existing merges
#   SKIP_SCORE=1 bash ...                                       # inference only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}
if [[ "$CHUNKS" -ne "$NUM_GPUS" ]]; then
  echo "WARN: NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES has $CHUNKS entries; using $CHUNKS" >&2
  NUM_GPUS="$CHUNKS"
fi

RUN_DIR="${1:-$ROOT/runs/visionzip/full_llava15_retain64}"
EVAL_ROOT="${EVAL_ROOT:-$ROOT/ThirdParty/LLaVA/playground/data/eval}"
TASKS="${TASKS:-gqa,mme,pope}"
SCORE_ONLY="${SCORE_ONLY:-0}"
SKIP_SCORE="${SKIP_SCORE:-0}"
mkdir -p "$RUN_DIR/answers"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate llava
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
# Drop VidCom2/FastVID leftovers from the parent shell — they shadow haotian LLaVA.
unset PYTHONPATH
export PYTHONPATH="$ROOT/ThirdParty/VisionZip:$ROOT/ThirdParty/LLaVA"
if [[ "${REPRO_PROBE:-}" == "1" ]]; then
  export REPRO_PROBE_METHOD=visionzip
  export REPRO_PROBE_DIR="${REPRO_PROBE_DIR:-$ROOT/runs/visionzip/full_llava15_probe}"
  mkdir -p "$REPRO_PROBE_DIR"
  export PYTHONPATH="$ROOT/scripts/repro:$PYTHONPATH"
fi

ok=0
fail=0

if [[ "$SCORE_ONLY" != "1" ]]; then
  python "$ROOT/scripts/repro/stage0.py" --method visionzip --run-dir "$RUN_DIR"
  PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
  # Prefer converted haotian dump when present
  HAOTIAN="$HF_HOME/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf"
  if [[ -f "$HAOTIAN/mm_projector.bin" && -f "$HAOTIAN/config.json" ]]; then
    PRETRAINED="$HAOTIAN"
    echo "$PRETRAINED" > "$RUN_DIR/pretrained_path.txt"
  fi
  echo "PRETRAINED=$PRETRAINED"

  run_chunked() {
    local task="$1"
    local question_file="$2"
    local image_folder="$3"
    local out_dir="$RUN_DIR/answers/$task"
    mkdir -p "$out_dir"
    if [[ ! -f "$question_file" ]]; then
      echo "SKIP $task: missing question file $question_file" >&2
      return 1
    fi
    if [[ ! -d "$image_folder" ]]; then
      echo "SKIP $task: missing image folder $image_folder" >&2
      return 1
    fi
    echo "=== $task ($NUM_GPUS chunks) ==="
    local pids=()
    local IDX
    for IDX in $(seq 0 $((NUM_GPUS - 1))); do
      CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python "$ROOT/scripts/repro/visionzip_vqa_loader.py" \
        --model-path "$PRETRAINED" \
        --model-name llava-v1.5-7b \
        --question-file "$question_file" \
        --image-folder "$image_folder" \
        --answers-file "$out_dir/${NUM_GPUS}_${IDX}.jsonl" \
        --num-chunks "$NUM_GPUS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --dominant 54 \
        --contextual 10 \
        >"$out_dir/chunk_${IDX}.log" 2>&1 &
      pids+=($!)
    done
    local rc=0
    for pid in "${pids[@]}"; do
      wait "$pid" || rc=1
    done
    if [[ "$rc" -ne 0 ]]; then
      echo "FAIL $task: one or more chunks failed; see $out_dir/chunk_*.log" >&2
      return 1
    fi
    local merge="$out_dir/merge.jsonl"
    : >"$merge"
    for IDX in $(seq 0 $((NUM_GPUS - 1))); do
      cat "$out_dir/${NUM_GPUS}_${IDX}.jsonl" >>"$merge"
    done
    echo "OK $task → $merge ($(wc -l <"$merge") lines)"
    return 0
  }

  IFS=',' read -ra TASK_ARR <<< "$TASKS"
  for task in "${TASK_ARR[@]}"; do
    case "$task" in
      gqa)
        if run_chunked gqa \
          "$EVAL_ROOT/gqa/llava_gqa_testdev_balanced.jsonl" \
          "$EVAL_ROOT/gqa/data/images"; then
          ok=$((ok + 1))
        else
          fail=$((fail + 1))
        fi
        ;;
      mme)
        if run_chunked mme \
          "$EVAL_ROOT/MME/llava_mme.jsonl" \
          "$EVAL_ROOT/MME/MME_Benchmark_release_version"; then
          ok=$((ok + 1))
        else
          fail=$((fail + 1))
        fi
        ;;
      pope)
        if run_chunked pope \
          "$EVAL_ROOT/pope/llava_pope_test.jsonl" \
          "$EVAL_ROOT/pope/val2014"; then
          ok=$((ok + 1))
        else
          fail=$((fail + 1))
        fi
        ;;
      *)
        echo "unknown task: $task" >&2
        fail=$((fail + 1))
        ;;
    esac
  done

  echo "VisionZip inference: ok=$ok fail=$fail → $RUN_DIR"
  if [[ "$ok" -eq 0 ]]; then
    echo "FAIL: no tasks completed. Check chunk_*.log under $RUN_DIR/answers/*/." >&2
    exit 1
  fi
else
  echo "SCORE_ONLY=1 — skipping inference; scoring merges under $RUN_DIR/answers/"
fi

if [[ "$SKIP_SCORE" == "1" ]]; then
  echo "SKIP_SCORE=1 — not running metric scripts"
  exit 0
fi

echo "=== scoring GQA / MME / POPE ==="
python "$ROOT/scripts/repro/score_visionzip.py" \
  --run-dir "$RUN_DIR" \
  --eval-root "$EVAL_ROOT" \
  --tasks "$TASKS"
echo "VisionZip metrics → $RUN_DIR/metrics/"
