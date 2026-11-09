#!/usr/bin/env bash
# Video-MME eval with a trained per-frame scorer.
#   RETENTION=0.10 bash scripts/vla_acl/run_videomme.sh heldout
#   RETENTION=0.25 bash scripts/vla_acl/run_videomme.sh full
# heldout drops the train videos. full scores all 2700 and is not the FastVID table.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/root/Test/envs/vla-acl/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Single-process `accelerate launch` does not set these. Accelerator refuses RTX 4090 without them.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONPATH="$ROOT/scripts/vla_acl:$ROOT/scripts/repro${PYTHONPATH:+:$PYTHONPATH}"
export VLA_ACL_VIDEO=1
export VLA_ACL_FRAME_CACHE="${VLA_ACL_FRAME_CACHE:-$ROOT/decode/output}"

RETENTION="${RETENTION:-0.10}"
export VLA_ACL_RETENTION="$RETENTION"
TAG="${RETENTION/./}"
SPLIT="${SPLIT:-$ROOT/runs/vla-acl-video/split.json}"
export VLA_ACL_SPLIT="$SPLIT"
SCORER="${SCORER:-$ROOT/runs/vla-acl-video/train/r${TAG}/patch_scorer.pt}"
export VLA_ACL_SCORER="$SCORER"
SCOPE="${1:-heldout}"
if [[ "$SCOPE" != "heldout" && "$SCOPE" != "full" ]]; then
  echo "FAIL: scope must be heldout or full" >&2
  exit 1
fi
if [[ "$SCOPE" == "heldout" ]]; then
  export VLA_ACL_EVAL=heldout
else
  unset VLA_ACL_EVAL || true
fi
if [[ ! -s "$SCORER" ]]; then
  echo "FAIL: no scorer at $SCORER" >&2
  echo "Train this retention first: RETENTION=$RETENTION bash scripts/vla_acl/run_train_video.sh" >&2
  exit 1
fi
if [[ "$SCOPE" == "heldout" && ! -s "$SPLIT" ]]; then
  echo "FAIL: missing $SPLIT" >&2
  exit 1
fi

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
NUM_GPUS="${NUM_GPUS:-${#GPULIST[@]}}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/vla-acl-video/${SCOPE}_r${TAG}}"
mkdir -p "$RUN_DIR"

cd "$ROOT/ThirdParty/FastVID/fastvid_llavavideo"
# stage0 reads EVAL_ARGS before the pretrained path exists. Pass the pool mode it checks for.
export EVAL_ARGS="mm_spatial_pool_mode=average,attn_implementation=sdpa"
"$PYTHON" "$ROOT/scripts/repro/stage0.py" --method fastvid --run-dir "$RUN_DIR"
PRETRAINED="$(cat "$RUN_DIR/pretrained_path.txt")"
MODEL_ARGS="pretrained=${PRETRAINED},conv_template=qwen_1_5,max_frames_num=64,force_sample=True,mm_spatial_pool_mode=average,attn_implementation=sdpa,fastvid_retention_ratio=1.0,fastvid_DySeg_c=8,fastvid_DySeg_tau=0.9,fastvid_STPrune_d=0.4,fastvid_DTM_p=4,fastvid_DTM_beta=0.6"
export EVAL_ARGS="$MODEL_ARGS"

{
  echo "date: $(date -Is)"
  echo "method: vla-acl-video"
  echo "scope: $SCOPE"
  echo "retention: $RETENTION"
  echo "scorer: $SCORER"
  echo "pretrained: $PRETRAINED"
  echo "gpus: $GPU_LIST"
  if [[ "$SCOPE" == "full" ]]; then
    echo "note: full includes videos used to train the scorer; do not compare it with the FastVID 2700 table"
  fi
} > "$RUN_DIR/command.txt"

{
  "$PYTHON" -m accelerate.commands.launch --num_processes="$NUM_GPUS" --main_process_port "$MAIN_PROCESS_PORT" -m lmms_eval \
    --model llava_vid \
    --model_args "$MODEL_ARGS" \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "vla_acl_${SCOPE}_r${TAG}" \
    --output_path "$RUN_DIR/logs/"
} 2>&1 | tee "$RUN_DIR/full_launch.log"

log="$RUN_DIR/full_launch.log"
if grep -q "Traceback (most recent call last)" "$log"; then
  echo "FAIL: traceback in $log" >&2
  exit 1
fi
if ! grep -q 'videomme_perception_score' "$log"; then
  echo "FAIL: missing videomme_perception_score in $log" >&2
  exit 1
fi
echo "OK: VLA-ACL video $SCOPE retention=$RETENTION → $RUN_DIR"
