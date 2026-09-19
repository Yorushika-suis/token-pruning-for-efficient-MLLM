#!/usr/bin/env python3
"""Keep montage plus vision-encoder layer maps (first / 3 mid / last)."""
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
    parse_methods,
)
from analyse.keys import (
    layer_json_path,
    layer_png_path,
    resolve_keys,
    skip_existing,
    write_keys_manifest,
)
from analyse.render import heatmap_to_rgb, montage_grid, overlay_keep_contour, resize_nearest, save_png
from analyse.vision_layers import encode_layers


def _layer_montage(
    method: str,
    frames: list[np.ndarray],
    frame_idx: np.ndarray,
    keep_hw: np.ndarray,
    indices: list[int],
    maps: dict[int, np.ndarray],
    cell: int,
) -> np.ndarray:
    if keep_hw.shape[0] == 1:
        keeps = [keep_hw[0]] * len(frames)
    elif keep_hw.shape[0] == len(frames):
        keeps = list(keep_hw)
    else:
        used = [int(i) for i in frame_idx if int(i) < keep_hw.shape[0]]
        if len(used) != len(frames):
            sub, extra = subsample_frames(keep_hw, len(frames))
            keeps = list(sub)
        else:
            keeps = [keep_hw[i] for i in used]
    col_titles = [f"L{i}" for i in indices]
    rows = []
    row_titles = [f"f{int(i)}" for i in frame_idx] if len(frames) > 1 else ["img"]
    for t, fr in enumerate(frames):
        row = []
        km = keeps[min(t, len(keeps) - 1)]
        for li in indices:
            energy = maps[li][min(t, maps[li].shape[0] - 1)]
            heat = heatmap_to_rgb(energy)
            heat = resize_nearest(heat, fr.shape[0], fr.shape[1])
            # blend with frame for context
            mix = (0.45 * fr.astype(np.float32) + 0.55 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
            mix = overlay_keep_contour(mix, km)
            row.append(mix)
        rows.append(row)
    return montage_grid(rows, col_titles=col_titles, row_titles=row_titles, cell_h=cell, cell_w=cell)


def run_one(
    methods: list[str],
    dataset: str,
    key: str,
    out_dir: Path,
    n_show: int,
    cell: int,
    n_layers: int,
    force: bool = False,
    frames: list[np.ndarray] | None = None,
    frame_idx: np.ndarray | None = None,
    mask_dirs: dict[str, Path] | None = None,
    keep_by_method: dict[str, np.ndarray] | None = None,
    cache_key: str | None = None,
    stem: str | None = None,
) -> None:
    key_n = normalize_key(key)
    out_stem = normalize_key(stem) if stem else key_n
    layer_pngs = [layer_png_path(out_dir, out_stem, m) for m in methods]
    summary_js = layer_json_path(out_dir, out_stem)
    all_done = all(skip_existing(p, force) for p in layer_pngs) and skip_existing(summary_js, force)
    if all_done:
        print(f"skip layers {out_stem} (exists)")
        return
    if frames is None:
        frames, frame_idx = load_show_frames(dataset, key_n, n_show=n_show)
    assert frame_idx is not None
    cache_id = cache_key if cache_key is not None else key_n
    summary = {"key": key_n, "stem": out_stem, "dataset": dataset, "methods": {}}
    for method, png in zip(methods, layer_pngs):
        if skip_existing(png, force):
            print(f"skip {method} layers {out_stem} ({png})")
            continue
        rec = load_keep(
            mask_path(method, key_n, merged_mask_dirs(mask_dirs).get(method)),
            method=method,
        )
        keep_hw = keep_by_method[method] if keep_by_method and method in keep_by_method else rec.keep_hw
        indices, maps = encode_layers(method, frames, out_dir, cache_id, n_pick=n_layers)
        grid = _layer_montage(method, frames, frame_idx, keep_hw, indices, maps, cell)
        save_png(png, grid)
        summary["methods"][method] = {
            "layer_indices": indices,
            "png": str(png),
            "notes": rec.notes,
        }
        print(f"ok {method} layers {indices} → {png}")
    summary_js.parent.mkdir(parents=True, exist_ok=True)
    if summary["methods"] or not skip_existing(summary_js, force):
        if summary["methods"]:
            summary_js.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", required=True)
    p.add_argument("--dataset", required=True, choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--key", default=None, help="single key; omit when using --n")
    p.add_argument("--n", type=int, default=None, help="sample this many keys")
    p.add_argument("--keys-file", type=Path, default=None, help="keys.txt or keys.json from pick/run script")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "analyse" / "out" / "layers")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=5, help="must include first+last; default 5")
    p.add_argument("--force", action="store_true", help="regenerate even if png/json exist")
    args = p.parse_args(argv)
    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys, seed_used = resolve_keys(
        methods=methods,
        dataset=args.dataset,
        keys_file=args.keys_file,
        key=args.key,
        n=args.n,
        seed=args.seed,
    )
    write_keys_manifest(
        args.out_dir,
        keys=keys,
        methods=methods,
        dataset=args.dataset,
        n=len(keys),
        seed=seed_used,
    )
    for key in keys:
        print(f"=== {key} ===")
        run_one(
            methods,
            args.dataset,
            key,
            args.out_dir,
            args.n_show,
            args.cell,
            args.n_layers,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
