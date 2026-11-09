#!/usr/bin/env python3
"""Transform Video-MME frames, re-run FastVID/VidCom2 keep, then redraw keep/layers/hybrid.

Keep is NOT a rotated copy of the original NPZ. Each geom is a new generate() with
probe observers, so DySeg / uniqueness see the transformed video.

Run FastVID and VidCom2 in their own conda envs (llava packages conflict):
  conda activate fastvid && python analyse/geom/show_geom.py --methods fastvid ...
  conda activate VidCom2 && python analyse/geom/show_geom.py --methods vidcom2 ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.lib.geom import apply_geom, parse_geoms
from analyse.lib.geom_infer import generate_keep, probe_mask_dir
from analyse.lib.keys import keep_artifact_paths, resolve_keys, write_keys_manifest
from analyse.lib.masks import load_keep, subsample_frames
from analyse.lib.media import load_videomme_model_frames, uniform_indices
from analyse.lib.paths import (
    N_SHOW_FRAMES,
    ROOT as REPO,
    check_dataset_methods,
    normalize_key,
    parse_methods,
)
from analyse.keep.show_keep import visualize
from analyse.keep.show_keep_attn import visualize_keep_attn
from analyse.layers.show_layers import run_one


def _keeps_aligned(keep_hw, frame_idx, n_frames):
    if keep_hw.shape[0] == 1:
        return np.repeat(keep_hw, n_frames, axis=0)
    used = [int(i) for i in frame_idx if int(i) < keep_hw.shape[0]]
    if len(used) != n_frames:
        sub, _ = subsample_frames(keep_hw, n_frames)
        return sub
    return keep_hw[used]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid")
    p.add_argument("--dataset", default="videomme", choices=["videomme"])
    p.add_argument("--key", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--geoms", default="rot90,rot180,rot270,hflip")
    p.add_argument("--out-dir", type=Path, default=REPO / "analyse" / "out" / "videomme_pair" / "geom")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=224)
    p.add_argument("--layer-cell", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--dim", type=float, default=0.55)
    p.add_argument("--fastvid-retention-ratio", type=float, default=float(os.environ.get("FASTVID_RETENTION_RATIO", "0.10")))
    p.add_argument("--r-ratio", type=float, default=float(os.environ.get("R_RATIO", "0.25")))
    p.add_argument("--skip-keep", action="store_true")
    p.add_argument("--skip-layers", action="store_true")
    p.add_argument("--skip-keep-attn", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    if len(methods) > 1:
        raise SystemExit(
            "run one method per process (FastVID and VidCom2 use different conda envs). "
            "Example: --methods fastvid   then   --methods vidcom2 with the same --out-dir"
        )
    geoms = parse_geoms(args.geoms)
    n = args.n if args.n is not None else (None if args.keys_file or args.key else 1)
    keys, seed_used = resolve_keys(
        methods=methods,
        dataset=args.dataset,
        keys_file=args.keys_file,
        key=args.key,
        n=n,
        seed=args.seed,
    )
    # load_method_model chdirs into ThirdParty; keep artefacts in the repo out-dir.
    args.out_dir = (Path.cwd() / args.out_dir).resolve()
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
    method = methods[0]

    for key in keys:
        key_n = normalize_key(key)
        frames64 = load_videomme_model_frames(key_n)
        show_idx = uniform_indices(len(frames64), args.n_show)
        for geom in geoms:
            print(f"=== infer {method} {key_n} {geom} ===")
            frames64_g = [apply_geom(fr, geom) for fr in frames64]
            frames_show = [frames64_g[int(i)] for i in show_idx]
            probe_root = probe_mask_dir(args.out_dir, method, geom).parent
            generate_keep(
                method,
                key_n,
                frames64_g,
                probe_root,
                fastvid_ratio=args.fastvid_retention_ratio,
                r_ratio=args.r_ratio,
                force=args.force,
            )
            rec = load_keep(probe_mask_dir(args.out_dir, method, geom) / f"{key_n}.npz", method=method)
            keep_g = {method: _keeps_aligned(rec.keep_hw, show_idx, len(frames_show))}
            stem = f"{key_n}_{geom}"
            cache_id = f"{key_n}__{geom}"
            mask_dirs = {method: probe_mask_dir(args.out_dir, method, geom)}
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
                    frames=frames_show,
                    frame_idx=show_idx,
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
                    frames=frames_show,
                    frame_idx=show_idx,
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
                    frames=frames_show,
                    frame_idx=show_idx,
                    keep_by_method=keep_g,
                    cache_key=cache_id,
                    stem=stem,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
