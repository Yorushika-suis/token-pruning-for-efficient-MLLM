#!/usr/bin/env bash
# Sequential Video-MME full runs at extra keep ratios (separate dirs).
#
# FastVID default in full_fastvid_llava_vid.sh is FASTVID_RETENTION_RATIO=0.10
#   (run dir runs/fastvid/full_videomme, probe full_videomme_probe).
# VidCom2 default is R_RATIO=0.25 — not FASTVID_RETENTION_RATIO.
#   This sweep sets VidCom2's R_RATIO to 0.10 then 0.06 (same keep fraction).
#
# Does not overwrite the paper defaults (FastVID 0.10 / VidCom2 0.25).
#
# Usage (repo root):
#   bash scripts/repro/sweep_retention_videomme.sh
#   bash scripts/repro/sweep_retention_videomme.sh --fastvid-only
#   bash scripts/repro/sweep_retention_videomme.sh --vidcom2-only
#   SKIP_DONE=0 bash scripts/repro/sweep_retention_videomme.sh   # redo even if 2700 npz exist
#   FASTVID_RATIOS="0.04 0.06" VIDCOM2_RATIOS="0.10" bash scripts/repro/sweep_retention_videomme.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

FASTVID_RATIOS="${FASTVID_RATIOS:-0.04 0.06 0.08}"
VIDCOM2_RATIOS="${VIDCOM2_RATIOS:-0.10 0.06}"
SKIP_DONE="${SKIP_DONE:-1}"
N_MASKS_DONE="${N_MASKS_DONE:-2700}"
# Per-sample: existing mask NPZ → skip generate; missing → infer. Override with REPRO_RESUME=0.
export REPRO_RESUME="${REPRO_RESUME:-1}"

do_fastvid=1
do_vidcom2=1
for arg in "$@"; do
  case "$arg" in
    --fastvid-only) do_vidcom2=0 ;;
    --vidcom2-only) do_fastvid=0 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

ratio_tag() {
  # awk only: do not use python (sitecustomize / OpenCLIP may leak onto stdout).
  awk -v r="$1" 'BEGIN { printf "r%03d", int(r * 100 + 0.5) }'
}

npz_count() {
  local d="$1/masks"
  if [[ ! -d "$d" ]]; then
    echo 0
    return
  fi
  find "$d" -maxdepth 1 -name '*.npz' | wc -l
}

run_done() {
  local run_dir="$1" probe_dir="$2" ok_needle="$3"
  local n
  n="$(npz_count "$probe_dir")"
  if [[ "$SKIP_DONE" != "1" ]]; then
    return 1
  fi
  if [[ "$n" -ge "$N_MASKS_DONE" ]] && [[ -f "$run_dir/full_launch.log" ]] && grep -q "$ok_needle" "$run_dir/full_launch.log"; then
    return 0
  fi
  return 1
}

echo "FastVID paper default FASTVID_RETENTION_RATIO=0.10 (not in this sweep)."
echo "VidCom2 paper default R_RATIO=0.25; this sweep uses R_RATIO for VidCom2 keep fraction."
echo "SKIP_DONE=$SKIP_DONE  REPRO_RESUME=$REPRO_RESUME (1=skip existing masks)  FASTVID_RATIOS=$FASTVID_RATIOS  VIDCOM2_RATIOS=$VIDCOM2_RATIOS"

if [[ "$do_fastvid" == "1" ]]; then
  for ratio in $FASTVID_RATIOS; do
    tag="$(ratio_tag "$ratio")"
    run_dir="$ROOT/runs/fastvid/full_videomme_${tag}"
    probe_dir="$ROOT/runs/fastvid/full_videomme_probe_${tag}"
    echo "======== FastVID FASTVID_RETENTION_RATIO=${ratio}  ${tag} ========"
    if run_done "$run_dir" "$probe_dir" "videomme_perception_score"; then
      echo "skip (already ${N_MASKS_DONE}+ npz and score in log): $probe_dir"
      continue
    fi
    # full_fastvid_llava_vid.sh: conda activate fastvid
    REPRO_PROBE=1 \
      REPRO_PROBE_DIR="$probe_dir" \
      FASTVID_RETENTION_RATIO="$ratio" \
      bash "$ROOT/scripts/repro/full_fastvid_llava_vid.sh" "$run_dir"
  done
fi

if [[ "$do_vidcom2" == "1" ]]; then
  for ratio in $VIDCOM2_RATIOS; do
    tag="$(ratio_tag "$ratio")"
    run_dir="$ROOT/runs/vidcom2/full_videomme_${tag}"
    probe_dir="$ROOT/runs/vidcom2/full_videomme_probe_${tag}"
    echo "======== VidCom2 R_RATIO=${ratio}  ${tag} ========"
    if run_done "$run_dir" "$probe_dir" "videomme_perception_score"; then
      echo "skip (already ${N_MASKS_DONE}+ npz and score in log): $probe_dir"
      continue
    fi
    # full_vidcom2_llava_vid.sh: conda activate VidCom2
    REPRO_PROBE=1 \
      REPRO_PROBE_DIR="$probe_dir" \
      R_RATIO="$ratio" \
      bash "$ROOT/scripts/repro/full_vidcom2_llava_vid.sh" "$run_dir"
  done
fi

echo "OK: retention sweep finished"
