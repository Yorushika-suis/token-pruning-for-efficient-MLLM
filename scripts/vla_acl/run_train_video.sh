#!/usr/bin/env bash
# Train one video scorer.
# Default: Video-MME train split + decode/output frame cache.
# 178K online: set DATA_YAML, JSON_ROOT, VIDEO_ROOT, and NO_FRAME_CACHE=1.
# PYTHON must import the FastVID LLaVA-Video stack (decord, pyarrow, llava).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/root/Test/envs/vla-acl/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  echo "Set PYTHON to an env that can import FastVID's llava and pyarrow." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

RETENTION="${RETENTION:-0.10}"
TAG="${RETENTION/./}"
OUT="${OUT:-$ROOT/runs/vla-acl-video/train/r${TAG}}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${RESUME:-}" ]]; then
  LIMIT_ARGS+=(--resume)
fi

SNAP_ROOT="$ROOT/weights/hub/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots"
MODEL="${MODEL:-$(find "$SNAP_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n 1)}"
if [[ ! -s "$MODEL/config.json" ]]; then
  echo "FAIL: LLaVA-Video snapshot not found under $SNAP_ROOT" >&2
  exit 1
fi

DATA_ARGS=()
if [[ -n "${DATA_YAML:-}" ]]; then
  JSON_ROOT="${JSON_ROOT:-$ROOT/weights/datasets/llava-video-178k/raw}"
  VIDEO_ROOT="${VIDEO_ROOT:-/mnt/sdc18t/Cyl/llava-video-178k/videos}"
  if [[ ! -s "$DATA_YAML" ]]; then
    echo "FAIL: missing $DATA_YAML" >&2
    exit 1
  fi
  if [[ ! -d "$JSON_ROOT" ]]; then
    echo "FAIL: missing json root $JSON_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$VIDEO_ROOT" ]]; then
    echo "FAIL: missing video root $VIDEO_ROOT" >&2
    exit 1
  fi
  DATA_ARGS+=(--data-yaml "$DATA_YAML" --json-root "$JSON_ROOT" --video-root "$VIDEO_ROOT")
  DATA_ARGS+=(--sample-seed "${SAMPLE_SEED:-1234}")
  if [[ "${MC_ONLY:-1}" == "0" ]]; then
    DATA_ARGS+=(--no-mc-only)
  fi
else
  SPLIT="${SPLIT:-$ROOT/runs/vla-acl-video/split.json}"
  if [[ ! -s "$SPLIT" ]]; then
    echo "FAIL: missing $SPLIT. Run scripts/vla_acl/split_videomme.py first." >&2
    exit 1
  fi
  VIDEO_ROOT="${VIDEO_ROOT:-$ROOT/weights/videomme/data}"
  DATA_ARGS+=(--split "$SPLIT" --video-root "$VIDEO_ROOT")
fi

CACHE_ARGS=()
if [[ "${NO_FRAME_CACHE:-0}" == "1" ]]; then
  CACHE_ARGS+=(--no-frame-cache)
else
  CACHE_ARGS+=(--frame-cache "${FRAME_CACHE:-$ROOT/decode/output}")
fi

"$PYTHON" "$ROOT/scripts/vla_acl/train_video.py" \
  --model-path "$MODEL" \
  --output "$OUT" \
  --retention "$RETENTION" \
  --frames "${FRAMES:-64}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --vision-chunk "${VISION_CHUNK:-4}" \
  "${DATA_ARGS[@]}" \
  "${CACHE_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  "$@"
