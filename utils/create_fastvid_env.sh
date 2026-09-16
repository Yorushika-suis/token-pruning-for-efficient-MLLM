#!/usr/bin/env bash
# Build a FastVID conda env without the official create_env.sh pitfalls:
#   - do not `pip install --upgrade pip` (breaks conda pip)
#   - lmms-eval lives under each stack dir, not ThirdParty/FastVID/lmms-eval
#   - av==14.4.0 wheels were yanked from PyPI; installing it compiles from source
#     and needs FFmpeg 7. Use av==14.2.0 (binary wheel) and skip the 14.4.0 pin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STACK="${1:-llavavideo}"
ENV_NAME="${FASTVID_ENV_NAME:-fastvid}"

case "$STACK" in
  llavavideo) STACK_DIR="$ROOT/ThirdParty/FastVID/fastvid_llavavideo"; HAS_LLAVA=1 ;;
  llavaonevision) STACK_DIR="$ROOT/ThirdParty/FastVID/fastvid_llavaonevision"; HAS_LLAVA=1 ;;
  qwen25vl) STACK_DIR="$ROOT/ThirdParty/FastVID/fastvid_qwen25vl"; HAS_LLAVA=0 ;;
  *)
    printf 'usage: %s {llavavideo|llavaonevision|qwen25vl}\n' "$0" >&2
    exit 2
    ;;
esac

FLASH_WHL_NAME='flash_attn-2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
FLASH_WHL="$STACK_DIR/$FLASH_WHL_NAME"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/$FLASH_WHL_NAME"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  printf 'reusing conda env %s (set FASTVID_ENV_NAME to use another)\n' "$ENV_NAME"
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
PY="$(command -v python)"
[[ "$PY" == *"/envs/$ENV_NAME/"* ]] || {
  printf 'python is not the conda env: %s\n' "$PY" >&2
  exit 1
}

pip_install() {
  "$PY" -m pip install --no-input "$@"
}

# Never upgrade pip itself.
pip_install setuptools_scm pyarrow==20.0.0 wheel
conda install -n "$ENV_NAME" -y 'sentencepiece==0.1.99'

# Binary wheel; 14.4.0 (lmms-eval pin) has no PyPI wheels anymore.
pip_install --only-binary=:all: 'av==14.2.0'

# Remaining lmms-eval deps except the yanked av==14.4.0 pin, then editable --no-deps.
DEPS_FILE="$(mktemp)"
"$PY" - "$STACK_DIR/lmms-eval/pyproject.toml" "$DEPS_FILE" <<'PY'
import pathlib, re, sys
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines, in_deps, deps = src.read_text().splitlines(), False, []
for line in lines:
    s = line.strip()
    if s == "dependencies = [":
        in_deps = True
        continue
    if in_deps:
        if s.startswith("]"):
            break
        code = s.split("#", 1)[0]
        match = re.search(r'"([^"]*)"', code)
        if not match:
            continue
        item = match.group(1).strip()
        if not item or item.startswith("av"):
            continue
        deps.append(item)
dst.write_text("\n".join(deps) + "\n")
PY
pip_install -r "$DEPS_FILE"
rm -f "$DEPS_FILE"

cd "$STACK_DIR/lmms-eval"
pip_install -e . --no-deps

if [[ "$HAS_LLAVA" -eq 1 ]]; then
  cd "$STACK_DIR/LLaVA-NeXT"
  # No .[train]: that extra pulls a different transformers git pin and fights 4.51.3.
  pip_install -e . --no-deps
fi

pip_install accelerate==1.1.0 datasets==2.20.0 transformers==4.51.3 huggingface_hub==0.34.4

if [[ ! -f "$FLASH_WHL" ]]; then
  wget -O "$FLASH_WHL" "$FLASH_URL"
fi
pip_install "$FLASH_WHL"

printf '\ncreate_fastvid_env OK env=%s stack=%s python=%s av=%s\n' \
  "$ENV_NAME" "$STACK" "$PY" "$("$PY" -c 'import av; print(av.__version__)')"
"$PY" -c 'import lmms_eval, inspect; print("lmms_eval", inspect.getfile(lmms_eval))'
if [[ "$HAS_LLAVA" -eq 1 ]]; then
  "$PY" -c 'import llava, inspect; print("llava", inspect.getfile(llava))'
fi
