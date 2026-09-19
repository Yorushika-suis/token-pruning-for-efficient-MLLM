"""Redirect lmms-eval Video-MME hub loads to the local parquet snapshot."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parquet_path() -> Path | None:
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    base = hf / "hub" / "datasets--lmms-lab--Video-MME" / "snapshots"
    if not base.exists():
        return None
    hits = list(base.glob("*/videomme/*.parquet"))
    return hits[0] if hits else None


def patch() -> None:
    parquet = parquet_path()
    if parquet is None:
        return
    try:
        import datasets
    except ImportError:
        # sitecustomize can load under base/conda hooks before the eval env is ready
        return

    original = datasets.load_dataset
    if getattr(original, "_repro_local_videomme", False):
        return

    def load_dataset(path=None, *args, **kwargs):
        if path == "lmms-lab/Video-MME":
            from datasets import DatasetDict

            local = original("parquet", data_files=str(parquet))
            table = local["train"] if "train" in local else next(iter(local.values()))
            return DatasetDict({"test": table, "train": table})
        return original(path, *args, **kwargs)

    load_dataset._repro_local_videomme = True
    datasets.load_dataset = load_dataset
