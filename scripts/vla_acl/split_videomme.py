#!/usr/bin/env python3
"""Split Video-MME by video id: 20% train, 80% held-out. Seed is fixed."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parquet_path() -> Path:
    hf = Path(os.environ.get("HF_HOME", ROOT / "weights"))
    hits = sorted((hf / "hub" / "datasets--lmms-lab--Video-MME" / "snapshots").glob("*/videomme/*.parquet"))
    if not hits:
        raise SystemExit("Video-MME parquet not found under weights/hub")
    return hits[0]


def video_ids(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["videoID"])
    return sorted(set(table.column("videoID").to_pylist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "vla-acl-video" / "split.json")
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ids = video_ids(parquet_path())
    if not ids:
        raise SystemExit("no video IDs in Video-MME")
    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * args.train_ratio)))
    train = sorted(shuffled[:n_train])
    heldout = sorted(shuffled[n_train:])
    if set(train) & set(heldout):
        raise SystemExit("train and held-out overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "n_videos": len(ids),
        "train_video_ids": train,
        "heldout_video_ids": heldout,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"videos {len(ids)} train {len(train)} heldout {len(heldout)} -> {args.output}")


if __name__ == "__main__":
    main()
