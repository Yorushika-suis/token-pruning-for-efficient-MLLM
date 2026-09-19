#!/usr/bin/env bash
# Full FastVID × LLaVA-Video × Video-MME (optional REPRO_PROBE=1).
# Usage (from anywhere):
#   bash scripts/repro/full_fastvid_llava_vid.sh
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/repro/full_fastvid_llava_vid.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12345}"

RUN_DIR="${1:-$ROOT/runs/fastvid/full_videomme}"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastvid
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
export PYTHONPATH="$ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/ThirdParty/FastVID/fastvid_llavavideo"
MODEL_ARGS_PROBE="mm_spatial_pool_mode=average,attn_implementation=sdpa"
export EVAL_ARGS="$MODEL_ARGS_PROBE"
python "$ROOT/scripts/repro/stage0.py" --method fastvid --run-dir "$RUN_DIR"
PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
FASTVID_RETENTION_RATIO="${FASTVID_RETENTION_RATIO:-0.10}"
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,force_sample=True,mm_spatial_pool_mode=average,attn_implementation=sdpa,fastvid_retention_ratio=${FASTVID_RETENTION_RATIO},fastvid_DySeg_c=8,fastvid_DySeg_tau=0.9,fastvid_STPrune_d=0.4,fastvid_DTM_p=4,fastvid_DTM_beta=0.6"
export EVAL_ARGS="$MODEL_ARGS"

if [[ "${REPRO_PROBE:-}" == "1" ]]; then
  export REPRO_PROBE_METHOD=fastvid
  export REPRO_PROBE_DIR="${REPRO_PROBE_DIR:-$ROOT/runs/fastvid/full_videomme_probe}"
  mkdir -p "$REPRO_PROBE_DIR"
fi

{
  accelerate launch --num_processes="$NUM_GPUS" --main_process_port "$MAIN_PROCESS_PORT" -m lmms_eval \
    --model llava_vid \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_full \
    --output_path "$RUN_DIR/logs/"
} 2>&1 | tee "$RUN_DIR/full_launch.log"

log="$RUN_DIR/full_launch.log"
if [[ -f "$log" ]] && grep -q "Traceback (most recent call last)" "$log"; then
  echo "FAIL: traceback in $log" >&2
  exit 1
fi
if [[ -f "$log" ]] && ! grep -q 'videomme_perception_score' "$log"; then
  echo "FAIL: missing videomme_perception_score in $log" >&2
  exit 1
fi
echo "OK: FastVID full run finished → $RUN_DIR"
