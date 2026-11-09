#!/usr/bin/env python3
"""Decode Video-MME clips to the 64 frames used at FastVID / VLA-ACL eval.

Eval calls LlavaVid.load_video with max_frames_num=64 and force_sample=True.
That path keeps np.linspace(0, total_frames - 1, 64, dtype=int) and ignores
the fps stride. Each file stores those frames as uint8 so a later reader can
skip decord.

Output is decode/output/<video_id>.npz with:
  frames       uint8 (64, H, W, 3)
  indices      int32 (64,)
  total_frames int32
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[2]


def frame_indices(total_frames: int, n: int) -> np.ndarray:
    """Same indices as LlavaVid.load_video(..., force_sample=True)."""
    if total_frames < 1:
        raise ValueError(f"video has {total_frames} frames")
    return np.linspace(0, total_frames - 1, n, dtype=int)


def video_path(root: Path, video_id: str) -> Path | None:
    for name in (f"{video_id}.mp4", f"{video_id}.MP4"):
        path = root / name
        if path.is_file():
            return path
    return None


def video_ids(video_root: Path, split: Path | None, which: str) -> list[str]:
    if which == "all":
        ids = sorted({path.stem for path in video_root.glob("*.mp4")} | {path.stem for path in video_root.glob("*.MP4")})
        return ids
    if split is None:
        raise SystemExit("--split is required unless --which all")
    payload = json.loads(split.read_text())
    key = {"train": "train_video_ids", "heldout": "heldout_video_ids"}[which]
    return list(payload[key])


def decode_one(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    total = len(reader)
    indices = frame_indices(total, n)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    if frames.shape[0] != n or frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"{path.name}: unexpected frame tensor {frames.shape} {frames.dtype}")
    return frames, indices.astype(np.int32, copy=False), total


def save_npz(path: Path, frames: np.ndarray, indices: np.ndarray, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, frames=frames, indices=indices, total_frames=np.int32(total))
    temporary.replace(path)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=ROOT / "weights" / "videomme" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "decode" / "output")
    parser.add_argument("--split", type=Path, default=ROOT / "runs" / "vla-acl-video" / "split.json")
    parser.add_argument("--which", choices=("all", "train", "heldout"), default="all")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")

    ids = video_ids(args.video_root, args.split, args.which)
    if args.limit is not None:
        ids = ids[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    missing = 0
    for index, video_id in enumerate(ids, start=1):
        destination = args.output / f"{video_id}.npz"
        if destination.is_file():
            skipped += 1
            continue
        source = video_path(args.video_root, video_id)
        if source is None:
            missing += 1
            print(f"missing {video_id}", flush=True)
            continue
        if free_bytes(args.output) < args.min_free_gb * (1024**3):
            raise SystemExit(
                f"stopping: less than {args.min_free_gb:g} GiB free on {args.output}. "
                f"wrote {done}, skipped {skipped}, missing {missing}, left {len(ids) - index + 1}"
            )
        frames, indices, total = decode_one(source, args.frames)
        save_npz(destination, frames, indices, total)
        done += 1
        print(
            f"{index}/{len(ids)} {video_id} frames {total} -> {tuple(frames.shape)} {destination.stat().st_size}",
            flush=True,
        )
    print(f"done {done} skipped {skipped} missing {missing} output {args.output}", flush=True)


if __name__ == "__main__":
    main()
