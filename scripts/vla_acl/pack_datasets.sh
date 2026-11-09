#!/usr/bin/env bash
# Pack LLaVA-Video-178K train videos/annotations and Video-MME test set onto
# /mnt/sdc18t, then optionally serve them for download from another machine.
#
# Usage:
#   bash scripts/vla_acl/pack_datasets.sh pack
#   bash scripts/vla_acl/pack_datasets.sh serve [port]
#   bash scripts/vla_acl/pack_datasets.sh list
#   bash scripts/vla_acl/pack_datasets.sh all [port]   # pack then serve
#
# On the receiving machine (replace HOST):
#   mkdir -p llava-vla-data && cd llava-vla-data
#   wget -c http://HOST:8765/SHA256SUMS
#   wget -c http://HOST:8765/01_178k_annotations.tar.gz
#   wget -c http://HOST:8765/02_videomme_annotations.tar.gz
#   wget -c http://HOST:8765/03_videomme_videos.tar
#   wget -c http://HOST:8765/04_178k_videos.tar
#   sha256sum -c SHA256SUMS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PACK_OUT:-/mnt/sdc18t/Cyl/packs}"
VID_178K="${VID_178K:-/mnt/sdc18t/Cyl/llava-video-178k/videos}"
ANN_178K="${ANN_178K:-$ROOT/weights/datasets/llava-video-178k}"
VID_MME="${VID_MME:-$ROOT/weights/videomme/data}"nvidia
ANN_MME="${ANN_MME:-$ROOT/weights/hub/datasets--lmms-lab--Video-MME}"

need() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "FAIL: missing $path" >&2
    exit 1
  fi
}

pack_one() {
  local name="$1"
  shift
  local dest="$OUT/$name"
  if [[ -f "$dest" ]]; then
    echo "skip existing $dest" >&2
    return 0
  fi
  echo "writing $dest" >&2
  "$@"
  echo "done $dest ($(du -h "$dest" | awk '{print $1}'))" >&2
}

cmd_pack() {
  need "$VID_178K"
  need "$ANN_178K/sample/output_rnd05.yaml"
  need "$ANN_178K/raw"
  need "$VID_MME"
  need "$ANN_MME"
  mkdir -p "$OUT"
  local free
  free="$(df -BG --output=avail "$OUT" | tail -1 | tr -dc '0-9')"
  if [[ "${free:-0}" -lt 500 ]]; then
    echo "FAIL: need ~500G free under $OUT, have ${free}G" >&2
    exit 1
  fi

  # Small first: annotations. Videos are already compressed; keep them as plain tar.
  # GNU tar: --exclude must come before member names.
  pack_one "01_178k_annotations.tar.gz" bash -c '
    tar -C "$0" --exclude="raw/.cache" --exclude="__pycache__" -cf - sample raw \
      | pigz -1 >"$1"
  ' "$ANN_178K" "$OUT/01_178k_annotations.tar.gz"

  pack_one "02_videomme_annotations.tar.gz" bash -c '
    tar -C "$0" --exclude=".cache" --exclude="__pycache__" -cf - . \
      | pigz -1 >"$1"
  ' "$ANN_MME" "$OUT/02_videomme_annotations.tar.gz"

  pack_one "03_videomme_videos.tar" \
    tar -C "$(dirname "$VID_MME")" -cf "$OUT/03_videomme_videos.tar" "$(basename "$VID_MME")"

  pack_one "04_178k_videos.tar" \
    tar -C "$(dirname "$VID_178K")" -cf "$OUT/04_178k_videos.tar" "$(basename "$VID_178K")"

  (
    cd "$OUT"
    sha256sum \
      01_178k_annotations.tar.gz \
      02_videomme_annotations.tar.gz \
      03_videomme_videos.tar \
      04_178k_videos.tar \
      >SHA256SUMS
  )
  cmd_list
  cat >"$OUT/DOWNLOAD.txt" <<EOF
# From another machine (wget -c resumes):
HOST=<this-machine-ip>
PORT=${PACK_PORT:-8765}
mkdir -p llava-vla-data && cd llava-vla-data
wget -c http://\$HOST:\$PORT/SHA256SUMS
wget -c http://\$HOST:\$PORT/01_178k_annotations.tar.gz
wget -c http://\$HOST:\$PORT/02_videomme_annotations.tar.gz
wget -c http://\$HOST:\$PORT/03_videomme_videos.tar
wget -c http://\$HOST:\$PORT/04_178k_videos.tar
sha256sum -c SHA256SUMS

# Extract:
mkdir -p 178k videomme
tar -xzf 01_178k_annotations.tar.gz -C 178k
tar -xzf 02_videomme_annotations.tar.gz -C videomme/annotations
tar -xf 03_videomme_videos.tar -C videomme
tar -xf 04_178k_videos.tar -C 178k
EOF
  echo "pack complete under $OUT"
}

cmd_serve() {
  local port="${1:-${PACK_PORT:-8765}}"
  need "$OUT"
  if [[ ! -f "$OUT/SHA256SUMS" ]]; then
    echo "FAIL: run pack first; missing $OUT/SHA256SUMS" >&2
    exit 1
  fi
  echo "serving $OUT on 0.0.0.0:$port"
  echo "local IP candidates:"
  hostname -I 2>/dev/null || true
  cd "$OUT"
  exec python3 -m http.server "$port" --bind 0.0.0.0
}

cmd_list() {
  mkdir -p "$OUT"
  echo "OUT=$OUT"
  if compgen -G "$OUT/*" >/dev/null; then
    du -sh "$OUT"/* 2>/dev/null | sort
    echo "----"
    du -sh "$OUT"
  else
    echo "(empty)"
  fi
}

case "${1:-}" in
  pack) cmd_pack ;;
  serve) shift; cmd_serve "${1:-}" ;;
  list) cmd_list ;;
  all)
    shift
    cmd_pack
    cmd_serve "${1:-}"
    ;;
  *)
    echo "usage: $0 {pack|serve|list|all} [port]" >&2
    exit 2
    ;;
esac
