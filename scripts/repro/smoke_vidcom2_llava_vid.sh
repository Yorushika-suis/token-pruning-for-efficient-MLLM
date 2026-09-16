#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset CUDA_PATH || true
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COMPRESSOR=vidcom2
export R_RATIO="${R_RATIO:-0.25}"

RUN_DIR="${1:-$ROOT/runs/vidcom2/smoke}"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate VidCom2
export PYTHONPATH="$ROOT/scripts/repro:$ROOT/ThirdParty/VidCom2${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/ThirdParty/VidCom2"
export EVAL_ARGS="mm_spatial_pool_mode=average,attn_implementation=sdpa"
python "$ROOT/scripts/repro/stage0.py" --method vidcom2 --run-dir "$RUN_DIR"
PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=sdpa"
export EVAL_ARGS="$MODEL_ARGS"

if [[ "${REPRO_PROBE:-}" == "1" ]]; then
  export REPRO_PROBE_METHOD=vidcom2
  export REPRO_PROBE_DIR="${REPRO_PROBE_DIR:-$ROOT/runs/vidcom2/smoke_probe}"
  mkdir -p "$REPRO_PROBE_DIR"
fi

{
  accelerate launch --num_processes=1 --main_process_port "${MAIN_PROCESS_PORT:-12346}" -m lmms_eval \
    --model llava_vid \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --limit 4 \
    --log_samples \
    --log_samples_suffix llava_vid_smoke \
    --output_path "$RUN_DIR/logs/"
} 2>&1 | tee "$RUN_DIR/smoke_launch.log"


# accelerate can return 0 after lmms_eval swallows an exception
log="$RUN_DIR/smoke_launch.log"
if [[ -f "$log" ]] && grep -q "Traceback (most recent call last)" "$log"; then
  echo "FAIL: traceback in $log" >&2
  exit 1
fi
if [[ -f "$log" ]] && ! grep -q 'Successfully integrated VidCom2' "$log"; then
  echo "FAIL: missing Successfully integrated VidCom2 in $log" >&2
  exit 1
fi
if [[ -f "$log" ]] && ! grep -q 'videomme_perception_score' "$log"; then
  echo "FAIL: missing videomme_perception_score in $log" >&2
  exit 1
fi
