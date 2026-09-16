"""Download eval checkpoints into $HF_HOME/hub (via ModelScope on this cluster)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_cli import hf_download, read_refs_main

# Requested Hub SHAs from Docs/02-reproduction-runbook.md. ModelScope often
# lacks those exact commits and falls back to master; record the printed refs/main.
MODELS: list[tuple[str, str | None]] = [
    ("lmms-lab/LLaVA-Video-7B-Qwen2", "013210b3aff822f1558b166d39c1046dd109520f"),
    ("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"),
    ("lmms-lab/llava-onevision-qwen2-7b-ov", None),
    ("Qwen/Qwen2.5-VL-7B-Instruct", None),
    ("liuhaotian/llava-v1.5-7b", None),
    # LLaVA-1.5 vision tower; offline load hangs/fails without this snapshot.
    ("openai/clip-vit-large-patch14-336", None),
]


def main() -> int:
    for repo_id, revision in MODELS:
        hf_download(repo_id, revision=revision)
        print(f"record {repo_id} refs/main={read_refs_main(repo_id)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
