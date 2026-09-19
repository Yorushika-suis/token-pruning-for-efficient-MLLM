#!/usr/bin/env bash
# Download GQA / MME / POPE into ThirdParty/LLaVA/playground/data/eval/
# Usage:
#   bash scripts/repro/download_visionzip_eval_data.sh
#   TASKS=gqa,pope bash scripts/repro/download_visionzip_eval_data.sh
#   SKIP_COCO_IMAGES=1 bash scripts/repro/download_visionzip_eval_data.sh   # annotations only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_ROOT="${LLAVA_EVAL_ROOT:-$ROOT/ThirdParty/LLaVA/playground/data/eval}"
TASKS="${TASKS:-gqa,mme,pope}"
JOBS="${JOBS:-4}"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /opt/miniconda3/envs/llava/bin/python ]]; then
    PY=/opt/miniconda3/envs/llava/bin/python
  elif [[ -x /opt/miniconda3/envs/fastvid/bin/python ]]; then
    PY=/opt/miniconda3/envs/fastvid/bin/python
  else
    PY=python3
  fi
fi

# pyarrow needed to unpack ModelScope parquets
"$PY" -c 'import pyarrow' 2>/dev/null || "$PY" -m pip install -q pyarrow

extra=()
if [[ "${EVAL_ZIP:-0}" == "1" ]]; then
  extra+=(--eval-zip)
  "$PY" -c 'import gdown' 2>/dev/null || "$PY" -m pip install -q gdown
fi
if [[ "${SKIP_COCO_IMAGES:-0}" == "1" ]]; then
  extra+=(--skip-coco-images)
fi

exec "$PY" -u "$ROOT/utils/download_llava_eval_data.py" \
  --eval-root "$EVAL_ROOT" \
  --tasks "$TASKS" \
  --jobs "$JOBS" \
  "${extra[@]}"
