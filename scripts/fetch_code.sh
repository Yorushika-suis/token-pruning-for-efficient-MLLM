#!/usr/bin/env bash
# Fetch one immutable FlashVID commit into a new directory. Never replaces code/FlashVID.
set -euo pipefail

PROJECT_ROOT=/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial
FLASHVID_COMMIT=${FLASHVID_COMMIT:-983cce6e30d7a8012442bfc7557d3afa61b3572d}
[[ "$FLASHVID_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'FLASHVID_COMMIT must be a full 40-character lowercase SHA: %s\n' "$FLASHVID_COMMIT"
  exit 1
}

SHORT_SHA=${FLASHVID_COMMIT:0:12}
CODE_ROOT="$PROJECT_ROOT/code"
ARCHIVE_DIR="$CODE_ROOT/archives"
SOURCE_DIR="$CODE_ROOT/sources/FlashVID-$FLASHVID_COMMIT"
RUN_STAMP=$(date +%Y%m%dT%H%M%S%z)
RUN_ID="fetch-flashvid_${SHORT_SHA}_${RUN_STAMP}"
LOG_DIR="$PROJECT_ROOT/runs/source_identity"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"
MANIFEST_PATH="$LOG_DIR/${RUN_ID}.json"
ARCHIVE_PATH="$ARCHIVE_DIR/flashvid-$FLASHVID_COMMIT.tar.gz"
URL="https://codeload.github.com/Fanziyang-v/FlashVID/tar.gz/$FLASHVID_COMMIT"

mkdir -p "$ARCHIVE_DIR" "$CODE_ROOT/sources" "$LOG_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

[[ ! -e "$SOURCE_DIR" ]] || {
  printf 'refusing to overwrite existing source directory: %s\n' "$SOURCE_DIR"
  exit 1
}
[[ ! -e "$ARCHIVE_PATH" ]] || {
  printf 'refusing to overwrite existing archive: %s\n' "$ARCHIVE_PATH"
  exit 1
}

TEMP_DIR=$(mktemp -d "$CODE_ROOT/.flashvid-fetch-${SHORT_SHA}.XXXXXX")
cleanup() {
  rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

printf 'fetch_code START time=%s commit=%s\n' "$(date -Is)" "$FLASHVID_COMMIT"
printf 'responsibility_note=%s\n' \
  '合作者接入时间未定，来源获取暂由 Codex 代行；这是临时安排。'
curl --fail --silent --show-error --location --retry 3 --max-time 300 \
  --output "$TEMP_DIR/archive.tar.gz" "$URL"
tar -tzf "$TEMP_DIR/archive.tar.gz" >/dev/null
tar -xzf "$TEMP_DIR/archive.tar.gz" -C "$TEMP_DIR"
EXTRACTED="$TEMP_DIR/FlashVID-$FLASHVID_COMMIT"
[[ -d "$EXTRACTED/flashvid" && -d "$EXTRACTED/llava" ]] || {
  printf 'archive does not contain the expected FlashVID tree\n'
  exit 1
}

mv "$TEMP_DIR/archive.tar.gz" "$ARCHIVE_PATH"
mv "$EXTRACTED" "$SOURCE_DIR"

export FLASHVID_FETCH_MANIFEST="$MANIFEST_PATH"
export FLASHVID_FETCH_COMMIT="$FLASHVID_COMMIT"
export FLASHVID_FETCH_ARCHIVE="$ARCHIVE_PATH"
export FLASHVID_FETCH_SOURCE="$SOURCE_DIR"
python3 - <<'PYEOF'
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


source = Path(os.environ["FLASHVID_FETCH_SOURCE"])
payload = {
    "schema_version": "flashvid-source-fetch-v1",
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "repository": "https://github.com/Fanziyang-v/FlashVID",
    "full_commit_sha": os.environ["FLASHVID_FETCH_COMMIT"],
    "archive": os.environ["FLASHVID_FETCH_ARCHIVE"],
    "archive_sha256": sha256_file(os.environ["FLASHVID_FETCH_ARCHIVE"]),
    "source_directory": str(source),
    "source_file_count": sum(1 for path in source.rglob("*") if path.is_file()),
    "active_project_source_replaced": False,
    "temporary_collaborator_duty_delegation": True,
    "responsibility_note": "合作者接入时间未定，来源获取暂由 Codex 代行；这是临时安排。"
}
target = Path(os.environ["FLASHVID_FETCH_MANIFEST"])
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PYEOF

printf 'fetch_code DONE time=%s source=%s manifest=%s\n' \
  "$(date -Is)" "$SOURCE_DIR" "$MANIFEST_PATH"
trap - EXIT
cleanup
