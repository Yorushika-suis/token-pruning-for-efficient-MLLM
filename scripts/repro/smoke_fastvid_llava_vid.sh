#!/usr/bin/env bash
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

RUN_DIR="${1:-$ROOT/runs/fastvid/smoke}"
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
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,force_sample=True,mm_spatial_pool_mode=average,attn_implementation=sdpa,fastvid_retention_ratio=0.10,fastvid_DySeg_c=8,fastvid_DySeg_tau=0.9,fastvid_STPrune_d=0.4,fastvid_DTM_p=4,fastvid_DTM_beta=0.6"
export EVAL_ARGS="$MODEL_ARGS"

if [[ "${REPRO_PROBE:-}" == "1" ]]; then
  export REPRO_PROBE_METHOD=fastvid
  export REPRO_PROBE_DIR="${REPRO_PROBE_DIR:-$ROOT/runs/fastvid/smoke_probe}"
  mkdir -p "$REPRO_PROBE_DIR"
fi

{
  accelerate launch --num_processes=1 --main_process_port "${MAIN_PROCESS_PORT:-12345}" -m lmms_eval \
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
if [[ -f "$log" ]] && ! grep -q 'videomme_perception_score' "$log"; then
  echo "FAIL: missing videomme_perception_score in $log" >&2
  exit 1
fi
