#!/usr/bin/env bash
# Train the K=128 LLaVA-1.5 patch scorer on local GQA.
# Create the interpreter yourself and pass it as PYTHON.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-/root/Test/envs/vla-acl/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  echo "Create that env, or set PYTHON to one that can import ThirdParty/LLaVA." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/weights}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

OUT="${OUT:-$ROOT/runs/vla-acl-v1.5/train/k128}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

"$PYTHON" "$ROOT/scripts/vla_acl/train_gqa.py" \
  --model-path "$ROOT/weights/hub/models--liuhaotian--llava-v1.5-7b/snapshots/haotian-from-hf" \
  --questions "$ROOT/weights/datasets/gqa/processed/mt_gqa/mt_gqa_train_balanced_20000.json" \
  --image-root "$ROOT/weights/datasets/gqa/processed/mt_gqa/images" \
  --testdev-question-ids "$ROOT/weights/datasets/gqa/processed/mt_gqa_testdev/question_ids.txt" \
  --output "$OUT" \
  --k 128 \
  "${LIMIT_ARGS[@]}" \
  "$@"
