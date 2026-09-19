#!/usr/bin/env python3
"""Redraw keep / layer / hybrid maps after rotating or flipping the 8 show frames.

Keep masks are transformed with the image so overlays stay on the same content.
Layer energy is recomputed on the transformed frames (separate cache key).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.geom import apply_geom, apply_geom_keep, parse_geoms
from analyse.keys import keep_artifact_paths, resolve_keys, write_keys_manifest
from analyse.masks import load_keep, subsample_frames
from analyse.media import load_show_frames
from analyse.paths import (
    N_SHOW_FRAMES,
    ROOT as REPO,
    check_dataset_methods,
    mask_path,
    merged_mask_dirs,
    normalize_key,
    parse_mask_dirs,
    parse_methods,
)
from analyse.show_keep import visualize
from analyse.show_keep_attn import visualize_keep_attn
from analyse.show_layers import run_one


def _keeps_aligned(keep_hw, frame_idx, n_frames):
    import numpy as np

    if keep_hw.shape[0] == 1:
        return np.repeat(keep_hw, n_frames, axis=0)
    used = [int(i) for i in frame_idx if int(i) < keep_hw.shape[0]]
    if len(used) != n_frames:
        sub, _ = subsample_frames(keep_hw, n_frames)
        return sub
    return keep_hw[used]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid,vidcom2")
    p.add_argument("--dataset", default="videomme", choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--key", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--geoms", default="rot90,rot180,rot270,hflip", help="comma list; orig also allowed")
    p.add_argument("--mask-dirs", default=None)
    p.add_argument("--out-dir", type=Path, default=REPO / "analyse" / "out" / "videomme_pair" / "geom")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=224)
    p.add_argument("--layer-cell", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--dim", type=float, default=0.55)
    p.add_argument("--skip-keep", action="store_true")
    p.add_argument("--skip-layers", action="store_true")
    p.add_argument("--skip-keep-attn", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    mask_dirs = parse_mask_dirs(args.mask_dirs)
    geoms = parse_geoms(args.geoms)
    n = args.n if args.n is not None else (None if args.keys_file or args.key else 1)
    keys, seed_used = resolve_keys(
        methods=methods,
        dataset=args.dataset,
        keys_file=args.keys_file,
        key=args.key,
        n=n,
        seed=args.seed,
        mask_dirs=mask_dirs,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_keys_manifest(
        args.out_dir,
        keys=keys,
        methods=methods,
        dataset=args.dataset,
        n=len(keys),
        seed=seed_used,
    )
    keep_dir = args.out_dir / "keep"
    layers_dir = args.out_dir / "layers"
    attn_dir = args.out_dir / "keep_attn"

    for key in keys:
        key_n = normalize_key(key)
        frames, frame_idx = load_show_frames(args.dataset, key_n, n_show=args.n_show)
        keep_src = {}
        for method in methods:
            rec = load_keep(
                mask_path(method, key_n, merged_mask_dirs(mask_dirs).get(method)),
                method=method,
            )
            keep_src[method] = _keeps_aligned(rec.keep_hw, frame_idx, len(frames))
        for geom in geoms:
            print(f"=== {key_n} {geom} ===")
            frames_g = [apply_geom(fr, geom) for fr in frames]
            keep_g = {m: apply_geom_keep(km, geom) for m, km in keep_src.items()}
            stem = f"{key_n}_{geom}"
            cache_id = f"{key_n}__{geom}"
            if not args.skip_keep:
                png, js = keep_artifact_paths(keep_dir, stem)
                visualize(
                    methods,
                    args.dataset,
                    key_n,
                    png,
                    n_show=args.n_show,
                    cell=args.cell,
                    mask_dirs=mask_dirs,
                    json_out=js,
                    force=args.force,
                    frames=frames_g,
                    frame_idx=frame_idx,
                    keep_by_method=keep_g,
                )
                print(f"keep {stem} → {png}")
            if not args.skip_layers:
                run_one(
                    methods,
                    args.dataset,
                    key_n,
                    layers_dir,
                    args.n_show,
                    args.layer_cell,
                    args.n_layers,
                    force=args.force,
                    frames=frames_g,
                    frame_idx=frame_idx,
                    mask_dirs=mask_dirs,
                    keep_by_method=keep_g,
                    cache_key=cache_id,
                    stem=stem,
                )
            if not args.skip_keep_attn:
                visualize_keep_attn(
                    methods,
                    args.dataset,
                    key_n,
                    attn_dir,
                    layers_dir,
                    n_show=args.n_show,
                    cell=args.layer_cell,
                    n_layers=args.n_layers,
                    force=args.force,
                    dim=args.dim,
                    mask_dirs=mask_dirs,
                    frames=frames_g,
                    frame_idx=frame_idx,
                    keep_by_method=keep_g,
                    cache_key=cache_id,
                    stem=stem,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
