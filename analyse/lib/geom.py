"""Geometric transforms on RGB frames and HxW keep / energy maps."""
from __future__ import annotations

import numpy as np

# rot90 = 90° counter-clockwise (numpy convention).
GEOM_NAMES = ("orig", "rot90", "rot180", "rot270", "hflip")


def apply_geom(arr: np.ndarray, name: str) -> np.ndarray:
    x = np.ascontiguousarray(arr)
    if name in ("orig", "identity", "none"):
        return x
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(x, 1, axes=(0, 1)))
    if name == "rot180":
        return np.ascontiguousarray(np.rot90(x, 2, axes=(0, 1)))
    if name == "rot270":
        return np.ascontiguousarray(np.rot90(x, 3, axes=(0, 1)))
    if name == "hflip":
        return np.ascontiguousarray(np.flip(x, axis=1))
    raise ValueError(f"unknown geom {name!r}; expected {GEOM_NAMES}")


def apply_geom_keep(keep_hw: np.ndarray, name: str) -> np.ndarray:
    """keep_hw is (T, H, W) or (H, W)."""
    k = np.asarray(keep_hw)
    if k.ndim == 2:
        return apply_geom(k, name)
    return np.stack([apply_geom(k[t], name) for t in range(k.shape[0])])


def parse_geoms(text: str | None) -> list[str]:
    if not text:
        return list(GEOM_NAMES)
    out = []
    for part in text.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in GEOM_NAMES and name not in {"identity", "none"}:
            raise SystemExit(f"unknown --geoms entry {name!r}; expected {GEOM_NAMES}")
        if name in {"identity", "none"}:
            name = "orig"
        if name not in out:
            out.append(name)
    if not out:
        raise SystemExit("need at least one --geoms entry")
    return out
