"""Download the LLaVA-Video-7B-Qwen2 snapshot from hf-mirror.

Resumable: snapshot_download reuses any already-complete blobs in HF_HOME,
so re-invocation after a pod recreation continues rather than restarts.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "lmms-lab/LLaVA-Video-7B-Qwen2"

print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT"), flush=True)
print("HF_HOME     =", os.environ.get("HF_HOME"), flush=True)

path = snapshot_download(REPO, max_workers=4)

print("SNAPSHOT_PATH", path, flush=True)

# Report what landed, so a truncated download is visible in the log.
total = 0
for root, _dirs, files in os.walk(path):
    for name in files:
        fp = os.path.join(root, name)
        try:
            total += os.path.getsize(os.path.realpath(fp))
        except OSError:
            pass
print("TOTAL_BYTES", total, flush=True)
if total < 15_000_000_000:
    print("WARNING: snapshot smaller than expected (~16 GB)", flush=True)
    sys.exit(3)
print("SIZE_OK", flush=True)
