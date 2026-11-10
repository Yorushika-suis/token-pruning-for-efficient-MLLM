"""Load the 64 eval frames written by decode/scripts/decode_frames.py.

Online decoding uses the same np.linspace indices as FlexSelect / FastVID
with force_sample=True.

178K caches use the path relative to --video-root (no suffix) so stems like
split_5.mp4 under different parents do not collide. Video-MME keeps flat ids.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_rel_key(video: Path, video_root: Path | None) -> str:
    """Relative path without suffix under video_root, else the file stem."""
    if video_root is not None:
        try:
            rel = video.resolve().relative_to(video_root.resolve())
            return str(rel.with_suffix(""))
        except ValueError:
            pass
    return video.stem


def load_cached_frames(cache_dir: Path, video_id: str, n_frames: int) -> np.ndarray:
    path = Path(cache_dir) / f"{video_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing cached frames for {video_id}: {path}")
    frames = np.load(path)["frames"]
    if frames.ndim != 4 or frames.shape[0] != n_frames or frames.shape[-1] != 3:
        raise RuntimeError(f"{path.name}: expected ({n_frames}, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    return frames


def decode_uniform_frames(video: Path, n_frames: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
    total = len(reader)
    if total < 1:
        raise RuntimeError(f"{video}: empty video")
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    if frames.ndim != 4 or frames.shape[0] != n_frames or frames.shape[-1] != 3:
        raise RuntimeError(f"{video.name}: expected ({n_frames}, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    return frames


def load_training_frames(
    video: Path,
    n_frames: int,
    cache_dir: Path | None,
    video_root: Path | None = None,
) -> np.ndarray:
    if cache_dir is not None:
        return load_cached_frames(cache_dir, cache_rel_key(video, video_root), n_frames)
    return decode_uniform_frames(video, n_frames)
