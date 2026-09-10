#!/bin/bash
# Resumable, detached download of LLaVA-Video-7B-Qwen2 weights via hf-mirror.
# Must survive pod recreation: re-running this script resumes from partial files.
set -u
WS=/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WS/weights"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$WS/runs" "$WS/weights"
LOG="$WS/runs/download_weights.log"
PY=/root/miniconda3/bin/python

echo "=== START $(date -Is) pod=$(hostname) ===" >> "$LOG"

if [ -f "$WS/runs/weights.DONE" ]; then
  echo "already DONE, nothing to do" >> "$LOG"
  exit 0
fi

for i in $(seq 1 40); do
  echo "--- attempt $i $(date -Is) ---" >> "$LOG"
  "$PY" "$WS/scripts/dl_weights.py" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "=== DONE $(date -Is) ===" >> "$LOG"
    touch "$WS/runs/weights.DONE"
    break
  fi
  echo "attempt $i failed rc=$rc, retry in 15s" >> "$LOG"
  sleep 15
done

echo "=== EXIT $(date -Is) ===" >> "$LOG"
