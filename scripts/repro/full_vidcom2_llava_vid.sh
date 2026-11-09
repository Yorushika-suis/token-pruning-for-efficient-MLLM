#!/usr/bin/env bash
# Full VidCom2 × LLaVA-Video × Video-MME (optional REPRO_PROBE=1).
# Resume: REPRO_PROBE=1 skips samples with an existing mask NPZ (REPRO_RESUME=0 to disable).
# Usage:
#   bash scripts/repro/full_vidcom2_llava_vid.sh
#   NUM_GPUS=8 bash scripts/repro/full_vidcom2_llava_vid.sh
#
# Important: do NOT export PYTHONPATH before `conda activate` in an interactive
# shell if you are still in base — sitecustomize is hardened, but activate hooks
# are cleaner with PYTHONPATH set only after activate (this script does that).
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
export COMPRESSOR=vidcom2
export R_RATIO="${R_RATIO:-0.25}"

NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"

RUN_DIR="${1:-$ROOT/runs/vidcom2/full_videomme}"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate VidCom2
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
export PYTHONPATH="$ROOT/scripts/repro:$ROOT/ThirdParty/VidCom2${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT/ThirdParty/VidCom2"
export EVAL_ARGS="mm_spatial_pool_mode=average,attn_implementation=sdpa"
python "$ROOT/scripts/repro/stage0.py" --method vidcom2 --run-dir "$RUN_DIR"
PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,attn_implementation=sdpa"
export EVAL_ARGS="$MODEL_ARGS"

if [[ "${REPRO_PROBE:-}" == "1" ]]; then
  export REPRO_PROBE_METHOD=vidcom2
  export REPRO_PROBE_DIR="${REPRO_PROBE_DIR:-$ROOT/runs/vidcom2/full_videomme_probe}"
  export REPRO_RESUME="${REPRO_RESUME:-1}"
  mkdir -p "$REPRO_PROBE_DIR"
  echo "probe dir $REPRO_PROBE_DIR  resume=$REPRO_RESUME (skip existing masks)"
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
if [[ -f "$log" ]] && ! grep -q 'Successfully integrated VidCom2' "$log"; then
  echo "FAIL: missing Successfully integrated VidCom2 in $log" >&2
  exit 1
fi
if [[ -f "$log" ]] && ! grep -q 'videomme_perception_score' "$log"; then
  echo "FAIL: missing videomme_perception_score in $log" >&2
  exit 1
fi
echo "OK: VidCom2 full run finished → $RUN_DIR"
