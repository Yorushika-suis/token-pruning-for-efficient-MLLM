"""Load v4-lite NPZ masks and map them onto spatial grids."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class KeepRecord:
    method: str
    key: str
    keep_hw: np.ndarray  # (T, H, W) bool
    n_in: int
    n_out: int
    retention: float
    spatial_kept: int
    frames: int
    grid_h: int
    grid_w: int
    notes: str
    meta: dict


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def load_meta(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return json.loads(bytes(z["meta_json"]).decode("utf-8"))


def load_keep(path: Path, method: str | None = None) -> KeepRecord:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    method = (method or meta.get("method") or "unknown").lower()
    key = path.stem
    frames = int(meta.get("frames") or 1)
    grid_h = int(meta.get("grid_h") or 0)
    grid_w = int(meta.get("grid_w") or 0)
    n_in = int(_scalar(z["n_in"]))
    n_out = int(_scalar(z["n_out"]))
    retention = float(_scalar(z["retention"]))

    if method == "vidcom2":
        mask, notes = _vidcom2_grid(z, frames, grid_h, grid_w)
    elif method == "fastvid":
        mask, notes = _fastvid_grid(z, meta, frames, grid_h, grid_w)
    elif method == "visionzip":
        mask, notes = _visionzip_grid(z, grid_h, grid_w)
        frames = 1
    else:
        raise ValueError(f"unsupported method {method!r}")

    spatial = int(mask.sum())
    return KeepRecord(
        method=method,
        key=key,
        keep_hw=mask,
        n_in=n_in,
        n_out=n_out,
        retention=retention,
        spatial_kept=spatial,
        frames=mask.shape[0],
        grid_h=mask.shape[1],
        grid_w=mask.shape[2],
        notes=notes,
        meta=meta,
    )


def _vidcom2_grid(z, frames: int, grid_h: int, grid_w: int) -> tuple[np.ndarray, str]:
    km = np.asarray(z["keep_mask"], dtype=bool)
    if km.ndim != 2:
        raise ValueError(f"vidcom2 keep_mask expected (T, HW), got {km.shape}")
    t, hw = km.shape
    if grid_h * grid_w != hw:
        raise ValueError(f"vidcom2 grid {grid_h}x{grid_w} != {hw}")
    if t != frames and frames:
        # still usable
        pass
    return km.reshape(t, grid_h, grid_w), "keep_mask (64,169) → 13×13"


def _fastvid_grid(z, meta: dict, frames: int, grid_h: int, grid_w: int) -> tuple[np.ndarray, str]:
    """Map FastVID sequence keep_indices onto 13×13 patches (drop newline cells).

    Layout after LLaVA-Video grid newlines: 13 rows × (13 patches + 1 newline) = 182 / frame.
    Prefix tokens 0..start-1 are always kept; visual window is start:start+64*182.
    """
    keep = np.sort(np.asarray(z["keep_indices"], dtype=np.int64).reshape(-1))
    tokens_per_frame = grid_h * (grid_w + 1)  # 13 * 14
    vlen = frames * tokens_per_frame
    start = _infer_prefix_len(keep)
    vis = keep[(keep >= start) & (keep < start + vlen)] - start
    if vis.size == 0:
        raise ValueError("fastvid visual keep window is empty")
    stride = tokens_per_frame
    within = vis % stride
    is_nl = (within + 1) % (grid_w + 1) == 0
    pure = vis[~is_nl]
    frm = (pure // stride).astype(np.int64)
    loc = (pure % stride).astype(np.int64)
    row = loc // (grid_w + 1)
    col = loc % (grid_w + 1)
    if np.any(col >= grid_w) or np.any(row >= grid_h):
        raise ValueError("fastvid patch col/row out of 13×13")
    out = np.zeros((frames, grid_h, grid_w), dtype=bool)
    out[frm, row, col] = True
    note = (
        f"seq keep→grid start={start} vlen={vlen} "
        f"visual={vis.size} newline={int(is_nl.sum())} patches={int(out.sum())} "
        f"(NPZ n_out={int(_scalar(z['n_out']))} subtracts all original newlines)"
    )
    return out, note


def _infer_prefix_len(keep: np.ndarray) -> int:
    if keep.size == 0 or int(keep[0]) != 0:
        return 0
    for i, v in enumerate(keep):
        if int(v) != i:
            return i
    return int(keep.size)


def _visionzip_grid(z, grid_h: int, grid_w: int) -> tuple[np.ndarray, str]:
    idx = np.asarray(z["keep_indices"], dtype=np.int64).reshape(-1)
    # CLIP: 0 = CLS, 1..576 = 24×24 patches
    patches = idx[idx > 0] - 1
    gh = grid_h or 24
    gw = grid_w or 24
    out = np.zeros((1, gh, gw), dtype=bool)
    if patches.size:
        row = patches // gw
        col = patches % gw
        valid = (row >= 0) & (row < gh) & (col >= 0) & (col < gw)
        out[0, row[valid], col[valid]] = True
    tgt = np.asarray(z["target_indices"]).reshape(-1) if "target_indices" in z.files else np.zeros((0,))
    note = (
        f"dominant keep_indices n={idx.size} (CLS dropped); "
        f"target_indices n={int(tgt.size)} "
        f"{'(empty — contextual not captured)' if tgt.size == 0 else ''}"
    )
    return out, note.strip()


def subsample_frames(keep_hw: np.ndarray, n_show: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly pick n_show indices from T frames. Returns (keep_sub, indices)."""
    t = keep_hw.shape[0]
    if t == 1:
        return keep_hw, np.array([0], dtype=np.int64)
    n_show = min(int(n_show), t)
    idx = np.unique(np.linspace(0, t - 1, n_show, dtype=np.int64))
    return keep_hw[idx], idx
