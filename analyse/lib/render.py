"""Keep overlay, layer heatmaps, and montages (numpy RGB, PNG via stdlib zlib)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def save_png(path: Path, rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def resize_nearest(img: np.ndarray, height: int, width: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    ys = (np.linspace(0, img.shape[0] - 1, height)).astype(np.int64)
    xs = (np.linspace(0, img.shape[1] - 1, width)).astype(np.int64)
    return img[ys[:, None], xs[None, :]]


def overlay_keep(frame: np.ndarray, keep_hw: np.ndarray, dim: float = 0.22) -> np.ndarray:
    """Kept patches full brightness; dropped patches dimmed."""
    fr = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = fr.shape[:2]
    gh, gw = keep_hw.shape
    ys = np.clip((np.arange(h) * gh / h).astype(np.int64), 0, gh - 1)
    xs = np.clip((np.arange(w) * gw / w).astype(np.int64), 0, gw - 1)
    mask = keep_hw[ys[:, None], xs[None, :]]
    out = fr.astype(np.float32)
    out[~mask] *= float(dim)
    # light green grid on kept cells
    tint = out.copy()
    tint[..., 1] = np.minimum(tint[..., 1] + 35, 255)
    out = np.where(mask[..., None], tint, out)
    return out.clip(0, 255).astype(np.uint8)


def upsample_mask(keep_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    gh, gw = keep_hw.shape
    ys = np.clip((np.arange(height) * gh / height).astype(np.int64), 0, gh - 1)
    xs = np.clip((np.arange(width) * gw / width).astype(np.int64), 0, gw - 1)
    return keep_hw[ys[:, None], xs[None, :]]


def overlay_keep_attn(
    frame: np.ndarray,
    keep_hw: np.ndarray,
    energy_hw: np.ndarray,
    dim: float = 0.55,
    alpha: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Attn heatmap only (no photo). Unkept patches are dimmed; kept stay full.

    `frame` sets output height/width. `alpha` is ignored (kept for callers).
    """
    del alpha
    fr = np.ascontiguousarray(frame, dtype=np.uint8)
    h, w = fr.shape[:2]
    sel = np.asarray(keep_hw, dtype=bool)
    heat = heatmap_to_rgb(np.asarray(energy_hw, dtype=np.float32))
    heat = resize_nearest(heat, h, w).astype(np.float32)
    keep_px = upsample_mask(sel, h, w)
    out = heat.copy()
    out[~keep_px] *= float(dim)
    out = out.clip(0, 255).astype(np.uint8)
    edge = np.zeros((h, w), dtype=bool)
    edge[1:, :] |= keep_px[1:, :] != keep_px[:-1, :]
    edge[:, 1:] |= keep_px[:, 1:] != keep_px[:, :-1]
    out = out.copy()
    out[edge] = np.array([255, 255, 0], dtype=np.uint8)
    stats = {
        "dim": float(dim),
        "n_keep": int(sel.sum()),
        "n_drop": int((~sel).sum()),
    }
    return out, stats


def heatmap_to_rgb(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Per-map percentile clip → turbo-ish blue-red RGB."""
    v = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros((*v.shape, 3), dtype=np.uint8)
    data = v[finite]
    if lo is None:
        lo = float(np.percentile(data, 2))
    if hi is None:
        hi = float(np.percentile(data, 98))
    if hi <= lo:
        hi = lo + 1e-6
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    t = np.where(finite, t, 0.0)
    r = np.clip(1.5 * t - 0.1, 0, 1)
    g = np.clip(1.5 - np.abs(2 * t - 1) * 1.2, 0, 1)
    b = np.clip(1.2 - 1.5 * t, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_keep_contour(rgb: np.ndarray, keep_hw: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    gh, gw = keep_hw.shape
    ys = np.clip((np.arange(h) * gh / h).astype(np.int64), 0, gh - 1)
    xs = np.clip((np.arange(w) * gw / w).astype(np.int64), 0, gw - 1)
    mask = keep_hw[ys[:, None], xs[None, :]]
    out = rgb.copy()
    # cell borders: where neighbor differs
    edge = np.zeros((h, w), dtype=bool)
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    out[edge] = np.array([255, 255, 0], dtype=np.uint8)
    return out


def _pad_to(img: np.ndarray, height: int, width: int, fill: int = 20) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    canvas[:h, :w] = img
    return canvas


def _label_bar(text: str, width: int, height: int = 28) -> np.ndarray:
    bar = np.full((height, width, 3), 18, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(bar)
        ImageDraw.Draw(im).text((6, 6), text[:120], fill=(230, 230, 230))
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return bar


def montage_grid(
    cells: list[list[np.ndarray]],
    col_titles: list[str] | None = None,
    row_titles: list[str] | None = None,
    cell_h: int = 224,
    cell_w: int = 224,
) -> np.ndarray:
    """cells[row][col] RGB images."""
    n_r = len(cells)
    n_c = max(len(r) for r in cells) if cells else 0
    if n_r == 0 or n_c == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    title_h = 28 if col_titles else 0
    row_w = 72 if row_titles else 0
    gap = 4
    out_h = title_h + n_r * (cell_h + gap) + 8
    out_w = row_w + n_c * (cell_w + gap) + 8
    canvas = np.full((out_h, out_w, 3), 12, dtype=np.uint8)
    if col_titles:
        for j, t in enumerate(col_titles):
            x = row_w + j * (cell_w + gap)
            canvas[0:title_h, x : x + cell_w] = _label_bar(t, cell_w, title_h)
    for i, row in enumerate(cells):
        y = title_h + i * (cell_h + gap)
        if row_titles and i < len(row_titles):
            canvas[y : y + cell_h, 0:row_w] = _label_bar(row_titles[i], row_w, cell_h)
        for j, cell in enumerate(row):
            x = row_w + j * (cell_w + gap)
            resized = resize_nearest(cell, cell_h, cell_w)
            canvas[y : y + cell_h, x : x + cell_w] = _pad_to(resized, cell_h, cell_w)
    return canvas
