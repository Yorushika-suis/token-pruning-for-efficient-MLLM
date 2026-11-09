#!/usr/bin/env bash
# CDPruner on LLaVA-Video × Video-MME.
# See Docs/Implementation/05-cdpruner-reproduction.md.
#
# Uses the fastvid env and lmms-eval. CDPRUNER=1 installs scripts/cdpruner/hook_videomme.py,
# which prunes pooled spatial tokens and skips FastVID's second prune.
#
# Usage:
#   bash scripts/cdpruner/run_videomme.sh
#   NUM_GPUS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/cdpruner/run_videomme.sh
#   CDPRUNER_RETENTION_RATIO=0.25 bash scripts/cdpruner/run_videomme.sh
#   CDPRUNER_VISUAL_TOKEN_NUM=2048 bash scripts/cdpruner/run_videomme.sh
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

export CDPRUNER=1
export CDPRUNER_RETENTION_RATIO="${CDPRUNER_RETENTION_RATIO:-0.10}"
# scripts/cdpruner must come first: Python loads only the first sitecustomize.py.
export PYTHONPATH="$ROOT/scripts/cdpruner:$ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"

RUN_DIR="${1:-$ROOT/runs/cdpruner/full_videomme}"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastvid
# shellcheck disable=SC1091
source "$ROOT/scripts/repro/cuda_env.sh"
export PYTHONPATH="$ROOT/scripts/cdpruner:$ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"
export CDPRUNER=1

cd "$ROOT/ThirdParty/FastVID/fastvid_llavavideo"
MODEL_ARGS_PROBE="mm_spatial_pool_mode=average,attn_implementation=sdpa"
export EVAL_ARGS="$MODEL_ARGS_PROBE"
python "$ROOT/scripts/repro/stage0.py" --method fastvid --run-dir "$RUN_DIR"
PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
# retention_ratio is unused: the hook sets cdpruner_skip_fastvid and skips that block.
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,force_sample=True,mm_spatial_pool_mode=average,attn_implementation=sdpa,fastvid_retention_ratio=1.0,fastvid_DySeg_c=8,fastvid_DySeg_tau=0.9,fastvid_STPrune_d=0.4,fastvid_DTM_p=4,fastvid_DTM_beta=0.6"
export EVAL_ARGS="$MODEL_ARGS"

{
  echo "date: $(date -Is)"
  echo "cdpruner_retention_ratio: ${CDPRUNER_RETENTION_RATIO}"
  echo "cdpruner_visual_token_num: ${CDPRUNER_VISUAL_TOKEN_NUM:-}"
  echo "pretrained: ${PRETRAINED}"
  git -C "$ROOT/ThirdParty/CDPruner" rev-parse HEAD
  git -C "$ROOT/ThirdParty/FastVID" rev-parse HEAD
} > "$RUN_DIR/command.txt"

{
  accelerate launch --num_processes="$NUM_GPUS" --main_process_port "$MAIN_PROCESS_PORT" -m lmms_eval \
    --model llava_vid \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_cdpruner \
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
echo "OK: CDPruner VideoMME run finished → $RUN_DIR"
