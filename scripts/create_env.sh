#!/usr/bin/env bash
# Build a disposable GPU-container venv and leave an immutable environment inventory.
set -euo pipefail

PROJECT_ROOT=/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial
BASE_PYTHON=/root/miniconda3/bin/python
VENV_PATH=${FLASHVID_VENV_PATH:-/root/flashvid_venv}
FLASHVID_SOURCE="$PROJECT_ROOT/code/FlashVID"
EXPECTED_GPU_COUNT=${FLASHVID_EXPECTED_GPU_COUNT:-8}
QSCODE_DISPLAY_NAME=${FLASHVID_QSCODE_NAME:-qscode-REDACTED}
RUN_STAMP=$(date +%Y%m%dT%H%M%S%z)
RUN_ID="create-env_$(hostname)_${RUN_STAMP}"
LOG_DIR="$PROJECT_ROOT/runs/environment"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"
MANIFEST_PATH="$LOG_DIR/${RUN_ID}.json"
PACKAGE_INVENTORY="$PROJECT_ROOT/env/${RUN_ID}.pip-freeze.txt"
PIP_CHECK_REPORT="$PROJECT_ROOT/env/${RUN_ID}.pip-check.txt"

mkdir -p "$LOG_DIR" "$PROJECT_ROOT/env"
exec > >(tee -a "$LOG_PATH") 2>&1

on_exit() {
  status=$?
  if [[ $status -ne 0 ]]; then
    printf 'create_env FAILED status=%s time=%s\n' "$status" "$(date -Is)"
  fi
}
trap on_exit EXIT

printf 'create_env START time=%s host=%s target=%s venv=%s\n' \
  "$(date -Is)" "$(hostname)" "$QSCODE_DISPLAY_NAME" "$VENV_PATH"
printf 'responsibility_note=%s\n' \
  '合作者接入时间未定，环境重建暂由 Codex 代行；这是临时安排。'

[[ -x "$BASE_PYTHON" ]] || { printf 'missing base Python: %s\n' "$BASE_PYTHON"; exit 1; }
[[ -d "$FLASHVID_SOURCE/flashvid" && -d "$FLASHVID_SOURCE/llava" ]] || {
  printf 'missing persistent FlashVID source: %s\n' "$FLASHVID_SOURCE"
  exit 1
}
[[ ! -e "$VENV_PATH" ]] || {
  printf 'refusing to reuse existing venv: %s\n' "$VENV_PATH"
  exit 1
}

"$BASE_PYTHON" -m venv --system-site-packages "$VENV_PATH"
VENV_PYTHON="$VENV_PATH/bin/python"

"$VENV_PYTHON" -m pip install --quiet --upgrade pip
"$VENV_PYTHON" -m pip install --quiet "transformers==4.57.3"
"$VENV_PYTHON" -m pip install --quiet \
  decord shortuuid einops-exts sentencepiece protobuf

# This venv deliberately inherits the GPU image's system packages.  A global
# pip check therefore also reports conflicts among unrelated image packages
# (for example redaccel/cupy/opentelemetry).  Preserve that report as evidence,
# but gate this environment on the exact imports, versions, source locations,
# and CUDA topology required by FlashVID below.
if "$VENV_PYTHON" -m pip check > "$PIP_CHECK_REPORT" 2>&1; then
  PIP_CHECK_STATUS=0
else
  PIP_CHECK_STATUS=$?
fi
printf 'global pip check diagnostic status=%s report=%s\n' \
  "$PIP_CHECK_STATUS" "$PIP_CHECK_REPORT"

SITE_PACKAGES=$("$VENV_PYTHON" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$FLASHVID_SOURCE" > "$SITE_PACKAGES/flashvid_project_source.pth"

"$VENV_PYTHON" -m pip freeze --all > "$PACKAGE_INVENTORY"

export FLASHVID_ENV_MANIFEST="$MANIFEST_PATH"
export FLASHVID_ENV_RUN_ID="$RUN_ID"
export FLASHVID_ENV_PACKAGE_INVENTORY="$PACKAGE_INVENTORY"
export FLASHVID_ENV_PIP_CHECK_REPORT="$PIP_CHECK_REPORT"
export FLASHVID_ENV_PIP_CHECK_STATUS="$PIP_CHECK_STATUS"
export FLASHVID_ENV_SOURCE="$FLASHVID_SOURCE"
export FLASHVID_ENV_EXPECTED_GPU_COUNT="$EXPECTED_GPU_COUNT"
export FLASHVID_ENV_QSCODE_NAME="$QSCODE_DISPLAY_NAME"

"$VENV_PYTHON" - <<'PYEOF'
import hashlib
import importlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


source = Path(os.environ["FLASHVID_ENV_SOURCE"]).resolve()
expected_gpus = int(os.environ["FLASHVID_ENV_EXPECTED_GPU_COUNT"])
if transformers.__version__ != "4.57.3":
    raise RuntimeError(f"unexpected transformers version: {transformers.__version__}")
if not torch.cuda.is_available() or torch.cuda.device_count() != expected_gpus:
    raise RuntimeError(
        f"expected {expected_gpus} CUDA devices, got available={torch.cuda.is_available()} "
        f"count={torch.cuda.device_count()}"
    )

required = ("flash_attn", "decord", "accelerate", "einops", "timm", "cv2", "av",
            "flashvid", "llava")
versions = {}
locations = {}
for name in required:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, "__version__", "unknown"))
    locations[name] = str(Path(getattr(module, "__file__", "")).resolve())

for name in ("flashvid", "llava"):
    path = Path(locations[name])
    if source not in path.parents:
        raise RuntimeError(f"{name} imported from {path}, expected under {source}")

inventory = Path(os.environ["FLASHVID_ENV_PACKAGE_INVENTORY"])
pip_check_report = Path(os.environ["FLASHVID_ENV_PIP_CHECK_REPORT"])
payload = {
    "schema_version": "flashvid-created-environment-v2",
    "run_id": os.environ["FLASHVID_ENV_RUN_ID"],
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "qscode_display_name": os.environ["FLASHVID_ENV_QSCODE_NAME"],
    "hostname": platform.node(),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "persistent_source": str(source),
    "module_versions": versions,
    "module_locations": locations,
    "package_inventory": str(inventory),
    "package_inventory_sha256": sha256_file(inventory),
    "package_inventory_role": "inventory of venv plus inherited system packages; not a standalone lock",
    "global_pip_check": {
        "status": int(os.environ["FLASHVID_ENV_PIP_CHECK_STATUS"]),
        "report": str(pip_check_report),
        "report_sha256": sha256_file(pip_check_report),
        "is_acceptance_gate": False,
        "reason": "The venv inherits system packages; unrelated base-image conflicts are retained as diagnostics. Exact required imports, pinned versions, source locations, and CUDA topology are the acceptance gates."
    },
    "temporary_collaborator_duty_delegation": True,
    "responsibility_note": "合作者接入时间未定，环境重建暂由 Codex 代行；这是临时安排。",
    "throughput_benchmarked": False
}
target = Path(os.environ["FLASHVID_ENV_MANIFEST"])
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)

print(json.dumps({
    "python": payload["python"],
    "torch": payload["torch"],
    "transformers": payload["transformers"],
    "gpu_count": payload["gpu_count"],
    "flashvid": locations["flashvid"]
}, ensure_ascii=False))
PYEOF

printf 'create_env DONE time=%s manifest=%s inventory=%s\n' \
  "$(date -Is)" "$MANIFEST_PATH" "$PACKAGE_INVENTORY"
trap - EXIT
