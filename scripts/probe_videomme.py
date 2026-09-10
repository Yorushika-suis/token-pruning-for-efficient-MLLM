"""Probe the lmms-lab/Video-MME dataset layout on hf-mirror.

We must not assume the archive structure: the plan's ZIP64 range-extraction step depends
on how many volumes there are, how they are named, and which parquet carries the
short/medium/long duration label. Print the facts, download only the small metadata.
"""
import os

from huggingface_hub import HfApi, hf_hub_download

REPO = "lmms-lab/Video-MME"
ENDPOINT = "https://hf-mirror.com"

api = HfApi(endpoint=ENDPOINT)

print("=== repo files ===", flush=True)
files = api.list_repo_files(REPO, repo_type="dataset")
for f in sorted(files):
    print("  ", f, flush=True)

print("\n=== file sizes (via repo_info) ===", flush=True)
info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
total = 0
for s in sorted(info.siblings, key=lambda x: x.rfilename):
    sz = s.size or 0
    total += sz
    print(f"   {sz/1e9:9.3f} GB  {s.rfilename}", flush=True)
print(f"   {total/1e9:9.3f} GB  TOTAL", flush=True)

print("\n=== downloading parquet metadata only ===", flush=True)
parquets = [f for f in files if f.endswith(".parquet")]
print("parquet files:", parquets, flush=True)
for p in parquets:
    path = hf_hub_download(REPO, p, repo_type="dataset", endpoint=ENDPOINT)
    print("  downloaded:", path, os.path.getsize(path), "bytes", flush=True)
