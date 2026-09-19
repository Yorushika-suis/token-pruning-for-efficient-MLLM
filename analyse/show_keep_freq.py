#!/usr/bin/env python3
"""Spatial keep-frequency on the 13×13 grid (FastVID / VidCom2).

For each method, over all shared Video-MME keys and all frames in the NPZ,
count how often each grid cell is kept. Writes a heatmap montage plus JSON/NPY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.keys import list_keys, read_keys_file, write_keys_manifest
from analyse.masks import load_keep
from analyse.paths import (
    ROOT as REPO,
    check_dataset_methods,
    mask_path,
    merged_mask_dirs,
    parse_mask_dirs,
    parse_methods,
)
from analyse.render import heatmap_to_rgb, montage_grid, resize_nearest, save_png


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid,vidcom2")
    p.add_argument("--dataset", default="videomme", choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--max-keys", type=int, default=None, help="optional cap after listing keys")
    p.add_argument("--mask-dirs", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO / "analyse" / "out" / "videomme_pair" / "keep_freq")
    p.add_argument("--cell", type=int, default=256)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    mask_dirs = parse_mask_dirs(args.mask_dirs)
    if args.keys_file is not None:
        keys = read_keys_file(args.keys_file)
    else:
        keys = list_keys(methods, args.dataset, mask_dirs=mask_dirs)
    if args.max_keys is not None:
        keys = keys[: args.max_keys]
    if not keys:
        raise SystemExit("no shared keys")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_keys_manifest(
        args.out_dir,
        keys=keys,
        methods=methods,
        dataset=args.dataset,
        n=len(keys),
        seed=None,
    )

    counts: dict[str, np.ndarray] = {}
    n_units: dict[str, int] = {m: 0 for m in methods}
    n_ok = 0
    n_skip = 0
    grid_hw: tuple[int, int] | None = None

    for i, key in enumerate(keys):
        try:
            recs = []
            for method in methods:
                rec = load_keep(mask_path(method, key, merged_mask_dirs(mask_dirs).get(method)), method=method)
                recs.append(rec)
            hw = (int(recs[0].grid_h), int(recs[0].grid_w))
            if any((r.grid_h, r.grid_w) != hw for r in recs):
                raise ValueError("grid mismatch across methods")
            if grid_hw is None:
                grid_hw = hw
                for method in methods:
                    counts[method] = np.zeros(hw, dtype=np.float64)
            elif hw != grid_hw:
                raise ValueError(f"grid {hw} != {grid_hw}")
            for method, rec in zip(methods, recs):
                km = np.asarray(rec.keep_hw, dtype=np.float64)
                counts[method] += km.sum(axis=0)
                n_units[method] += int(km.shape[0])
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            n_skip += 1
            print(f"skip {key}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"... {i + 1}/{len(keys)} keys (ok={n_ok} skip={n_skip})")

    if grid_hw is None or n_ok == 0:
        raise SystemExit("no usable masks")

    freq = {m: counts[m] / max(n_units[m], 1) for m in methods}
    payload = {
        "dataset": args.dataset,
        "methods": methods,
        "n_keys": n_ok,
        "n_skip": n_skip,
        "grid": list(grid_hw),
        "n_frame_units": n_units,
        "mean_keep_rate": {m: float(freq[m].mean()) for m in methods},
        "freq": {m: freq[m].tolist() for m in methods},
    }
    jsons = args.out_dir / "jsons"
    jsons.mkdir(parents=True, exist_ok=True)
    (jsons / "keep_freq.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for method in methods:
        np.save(args.out_dir / f"freq_{method}.npy", freq[method])

    cells = []
    titles = []
    lo = min(float(freq[m].min()) for m in methods)
    hi = max(float(freq[m].max()) for m in methods)
    for method in methods:
        heat = heatmap_to_rgb(freq[method], lo=lo, hi=hi)
        heat = resize_nearest(heat, args.cell, args.cell)
        cells.append(heat)
        titles.append(f"{method}  mean={freq[method].mean():.3f}  n={n_units[method]}")
    grid = montage_grid([cells], col_titles=titles, row_titles=["keep freq"], cell_h=args.cell, cell_w=args.cell)
    png = args.out_dir / "pngs" / "keep_freq.png"
    save_png(png, grid)
    print(f"ok {n_ok} keys → {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
