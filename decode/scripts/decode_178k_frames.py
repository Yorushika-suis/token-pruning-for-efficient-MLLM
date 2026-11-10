#!/usr/bin/env python3
"""Decode LLaVA-Video-178K MC training clips to 64-frame npz caches.

Uses the same FlexSelect recipe as scripts/vla_acl/train_video.py:
output_rnd05.yaml, random:5%, sample_seed=1234, MC-only (*_mc_* jsons).
Frame indices match force_sample=True (np.linspace over the clip).

Cache key is the path relative to --video-root without the suffix, so names
like split_5.mp4 under different parents do not collide. Training reads the
same key via frames.cache_rel_key / load_training_frames(..., video_root=...).

Keep originals on disk: dataset construction still requires the mp4 path.

Output default: decode/output_178k/<rel>.npz with
  frames       uint8 (64, H, W, 3)
  indices      int32 (64,)
  total_frames int32
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "vla_acl"))
from frames import cache_rel_key  # noqa: E402


def frame_indices(total_frames: int, n: int) -> np.ndarray:
    if total_frames < 1:
        raise ValueError(f"video has {total_frames} frames")
    return np.linspace(0, total_frames - 1, n, dtype=int)


def resolve_video(video_root: Path, data_source: str, rel: str) -> Path | None:
    rel = str(rel)
    candidates = [rel]
    if not rel.endswith((".mp4", ".MP4")):
        candidates.extend([rel + ".mp4", rel + ".MP4"])
    for name in candidates:
        path = video_root / data_source / name
        if path.is_file():
            return path
    return None


def mc_json_paths(data_yaml: Path, json_root: Path) -> list[Path]:
    paths = []
    for line in data_yaml.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- json_path:"):
            continue
        rel = stripped.split("LLaVA-Video-178K/", 1)[-1]
        if "_mc_" not in Path(rel).name:
            continue
        path = json_root / rel
        if not path.is_file():
            raise SystemExit(f"missing annotation json: {path}")
        paths.append(path)
    if not paths:
        raise SystemExit(f"no MC json paths in {data_yaml}")
    return paths


def collect_videos(
    data_yaml: Path,
    json_root: Path,
    video_root: Path,
    sample_seed: int,
) -> tuple[dict[str, Path], int]:
    """Return cache_key -> path for local MC videos; missing is reference count."""
    videos: dict[str, Path] = {}
    missing = 0
    random.seed(sample_seed)
    for path in mc_json_paths(data_yaml, json_root):
        data = json.loads(path.read_text())
        k = math.ceil(5 * len(data) / 100)
        random.shuffle(data)
        for item in data[:k]:
            video = resolve_video(video_root, item["data_source"], item["video"])
            if video is None:
                missing += 1
                continue
            key = cache_rel_key(video, video_root)
            if key in videos and videos[key] != video:
                raise SystemExit(f"cache key collision {key!r}:\n  {videos[key]}\n  {video}")
            videos[key] = video
    return videos, missing


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
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=ROOT / "weights" / "datasets" / "llava-video-178k" / "sample" / "output_rnd05.yaml",
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=ROOT / "weights" / "datasets" / "llava-video-178k" / "raw",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=ROOT / "weights" / "datasets" / "llava-video-178k" / "videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "decode" / "output_178k",
    )
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=1.0)
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")

    videos, missing = collect_videos(args.data_yaml, args.json_root, args.video_root, args.sample_seed)
    items = sorted(videos.items(), key=lambda kv: kv[0])
    if args.limit is not None:
        items = items[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"local videos {len(videos)} missing_refs {missing} "
        f"decode {len(items)} -> {args.output}",
        flush=True,
    )

    done = 0
    skipped = 0
    failed = 0
    fail_log = args.output / "decode_failures.txt"
    for index, (key, source) in enumerate(items, start=1):
        destination = args.output / f"{key}.npz"
        if destination.is_file():
            skipped += 1
            continue
        if free_bytes(args.output) < args.min_free_gb * (1024**3):
            raise SystemExit(
                f"stopping: less than {args.min_free_gb:g} GiB free on {args.output}. "
                f"wrote {done}, skipped {skipped}, failed {failed}, left {len(items) - index + 1}"
            )
        try:
            frames, indices, total = decode_one(source, args.frames)
            save_npz(destination, frames, indices, total)
        except Exception as exc:  # noqa: BLE001 — keep going on corrupt/partial mp4s
            failed += 1
            message = f"{key}\t{source}\t{type(exc).__name__}: {exc}"
            print(f"FAIL {index}/{len(items)} {message}", flush=True)
            with fail_log.open("a") as fh:
                fh.write(message + "\n")
            continue
        done += 1
        print(
            f"{index}/{len(items)} {key} frames {total} -> {tuple(frames.shape)} "
            f"{destination.stat().st_size}",
            flush=True,
        )
    print(
        f"done {done} skipped {skipped} failed {failed} output {args.output}",
        flush=True,
    )
    if failed:
        print(f"failures logged to {fail_log}", flush=True)


if __name__ == "__main__":
    main()
