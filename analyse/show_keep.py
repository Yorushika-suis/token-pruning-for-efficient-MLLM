#!/usr/bin/env python3
"""Visualize keep maps for one question; optional method comparison montage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.masks import load_keep, subsample_frames
from analyse.media import load_show_frames
from analyse.paths import (
    N_SHOW_FRAMES,
    check_dataset_methods,
    mask_path,
    merged_mask_dirs,
    normalize_key,
    parse_mask_dirs,
    parse_methods,
)
from analyse.render import montage_grid, overlay_keep, save_png


def visualize(
    methods: list[str],
    dataset: str,
    key: str,
    out: Path,
    n_show: int = N_SHOW_FRAMES,
    cell: int = 224,
    mask_dirs: dict[str, Path] | None = None,
    json_out: Path | None = None,
    force: bool = False,
    frames: list[np.ndarray] | None = None,
    frame_idx: np.ndarray | None = None,
    keep_by_method: dict[str, np.ndarray] | None = None,
) -> Path | None:
    check_dataset_methods(dataset, methods)
    key_n = normalize_key(key)
    out = Path(out)
    if json_out is None:
        if out.parent.name == "pngs":
            json_out = out.parent.parent / "jsons" / f"{out.stem}.json"
        else:
            json_out = out.with_suffix(".json")
    json_out = Path(json_out)
    from analyse.keys import skip_existing

    if skip_existing(out, force) and skip_existing(json_out, force):
        print(f"skip {key_n} (exists {out})")
        return out
    if frames is None:
        frames, frame_idx = load_show_frames(dataset, key_n, n_show=n_show)
    assert frame_idx is not None
    cols = []
    titles = []
    sidecar = {"dataset": dataset, "key": key_n, "frame_idx": np.asarray(frame_idx).tolist(), "methods": []}
    for method in methods:
        directory = merged_mask_dirs(mask_dirs).get(method)
        rec = load_keep(mask_path(method, key_n, directory), method=method)
        if keep_by_method is not None and method in keep_by_method:
            keep_sub = keep_by_method[method]
            used = np.asarray(frame_idx).tolist()
        elif rec.keep_hw.shape[0] == 1:
            keep_sub = rec.keep_hw
            used = [0] * len(frames)
        else:
            used = [int(i) for i in frame_idx if int(i) < rec.keep_hw.shape[0]]
            keep_sub = rec.keep_hw[used]
            if len(keep_sub) != len(frames):
                keep_sub, extra = subsample_frames(rec.keep_hw, len(frames))
                used = extra.tolist()
        col = []
        for fr, km in zip(frames, keep_sub):
            col.append(overlay_keep(fr, km))
        cols.append(col)
        titles.append(
            f"{method}  spatial={rec.spatial_kept}/{rec.keep_hw.size}  "
            f"n_out/n_in={rec.n_out}/{rec.n_in}  ret={rec.retention:.3f}"
        )
        sidecar["methods"].append(
            {
                "method": method,
                "n_in": rec.n_in,
                "n_out": rec.n_out,
                "retention": rec.retention,
                "spatial_kept": rec.spatial_kept,
                "grid": [rec.grid_h, rec.grid_w, rec.frames],
                "notes": rec.notes,
                "frame_idx": used if rec.keep_hw.shape[0] > 1 else [0],
            }
        )
    rows = list(map(list, zip(*cols)))  # frame-major
    row_titles = [f"f{int(i)}" for i in frame_idx] if dataset.lower() == "videomme" else ["img"]
    grid = montage_grid(rows, col_titles=titles, row_titles=row_titles, cell_h=cell, cell_w=cell)
    save_png(out, grid)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", required=True, help="comma list: fastvid,vidcom2 or visionzip")
    p.add_argument("--dataset", required=True, choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--key", required=True, help="e.g. 1NcGHbFWBFA__573-1 or code_reasoning/0020.png")
    p.add_argument("--out", type=Path, default=ROOT / "analyse" / "out" / "keep.png")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=224)
    p.add_argument(
        "--mask-dirs",
        default=None,
        help="override probe NPZ dirs, e.g. fastvid=/path/masks,vidcom2=/path/masks",
    )
    p.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = p.parse_args(argv)
    methods = parse_methods(args.methods)
    path = visualize(
        methods,
        args.dataset,
        args.key,
        args.out,
        n_show=args.n_show,
        cell=args.cell,
        mask_dirs=parse_mask_dirs(args.mask_dirs),
        force=args.force,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
