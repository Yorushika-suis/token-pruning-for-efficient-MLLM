"""Download pinned Video-MME parquet into $HF_HOME (via ModelScope on this cluster)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_cli import hf_download, read_refs_main

REPO_ID = "lmms-lab/Video-MME"
FILENAME = "videomme/test-00000-of-00001.parquet"
# Hub SHA from the runbook; ModelScope may only have master and will fall back.
REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"


def main() -> int:
    hf_download(REPO_ID, FILENAME, repo_type="dataset", revision=REVISION)
    print(f"refs/main={read_refs_main(REPO_ID, repo_type='dataset')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
