#!/usr/bin/env python3
"""Keep-style dimming on the layer-energy heatmap (no photo underlay).

Unkept patches are scaled by --dim (default 0.55); kept patches stay full.
Yellow edges mark keep-cell borders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.lib.keys import (
    ensure_pngs_jsons,
    resolve_keys,
    skip_existing,
    write_keys_manifest,
)
from analyse.lib.masks import load_keep, subsample_frames
from analyse.lib.media import load_show_frames
from analyse.lib.paths import (
    N_SHOW_FRAMES,
    ROOT as REPO,
    check_dataset_methods,
    mask_path,
    merged_mask_dirs,
    normalize_key,
    parse_mask_dirs,
    parse_methods,
)
from analyse.lib.render import montage_grid, overlay_keep_attn, save_png
from analyse.lib.vision_layers import encode_layers


def _keeps_for_frames(keep_hw: np.ndarray, frame_idx: np.ndarray, n_frames: int) -> list[np.ndarray]:
    if keep_hw.shape[0] == 1:
        return [keep_hw[0]] * n_frames
    used = [int(i) for i in frame_idx if int(i) < keep_hw.shape[0]]
    if len(used) != n_frames:
        sub, _ = subsample_frames(keep_hw, n_frames)
        return list(sub)
    return [keep_hw[i] for i in used]


def visualize_keep_attn(
    methods: list[str],
    dataset: str,
    key: str,
    out_dir: Path,
    cache_dir: Path,
    n_show: int = N_SHOW_FRAMES,
    cell: int = 192,
    n_layers: int = 5,
    force: bool = False,
    dim: float = 0.55,
    mask_dirs: dict[str, Path] | None = None,
    frames: list[np.ndarray] | None = None,
    frame_idx: np.ndarray | None = None,
    keep_by_method: dict[str, np.ndarray] | None = None,
    cache_key: str | None = None,
    stem: str | None = None,
) -> None:
    key_n = normalize_key(key)
    out_stem = normalize_key(stem) if stem else key_n
    pngs, jsons = ensure_pngs_jsons(out_dir)
    if frames is None:
        frames, frame_idx = load_show_frames(dataset, key_n, n_show=n_show)
    assert frame_idx is not None
    cache_id = cache_key if cache_key is not None else key_n
    sidecar = {
        "key": key_n,
        "stem": out_stem,
        "dataset": dataset,
        "frame_idx": np.asarray(frame_idx).tolist(),
        "methods": {},
    }

    for method in methods:
        png = pngs / f"{out_stem}_{method}_keep_attn.png"
        js = jsons / f"{out_stem}_{method}_keep_attn.json"
        if skip_existing(png, force) and skip_existing(js, force):
            print(f"skip keep-attn {method} {out_stem}")
            continue
        rec = load_keep(
            mask_path(method, key_n, merged_mask_dirs(mask_dirs).get(method)),
            method=method,
        )
        if keep_by_method is not None and method in keep_by_method:
            keeps = list(keep_by_method[method])
        else:
            keeps = _keeps_for_frames(rec.keep_hw, frame_idx, len(frames))
        indices, maps = encode_layers(method, frames, cache_dir, cache_id, n_pick=n_layers)
        rows = []
        layer_stats = {}
        for t, fr in enumerate(frames):
            km = keeps[min(t, len(keeps) - 1)]
            row = []
            for li in indices:
                energy = maps[li][min(t, maps[li].shape[0] - 1)]
                cell_img, st = overlay_keep_attn(fr, km, energy, dim=dim)
                row.append(cell_img)
                layer_stats.setdefault(str(li), []).append(st)
            rows.append(row)
        grid = montage_grid(
            rows,
            col_titles=[f"L{i}" for i in indices],
            row_titles=[f"f{int(i)}" for i in frame_idx] if len(frames) > 1 else ["img"],
            cell_h=cell,
            cell_w=cell,
        )
        save_png(png, grid)
        rec_js = {
            "method": method,
            "png": str(png),
            "layer_indices": indices,
            "stats": layer_stats,
            "notes": rec.notes,
        }
        js.write_text(json.dumps(rec_js, indent=2) + "\n", encoding="utf-8")
        sidecar["methods"][method] = rec_js
        print(f"ok keep-attn {method} {out_stem} → {png}")
    (jsons / f"{out_stem}_keep_attn.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid,vidcom2")
    p.add_argument("--dataset", default="videomme", choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--key", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=REPO / "analyse" / "out" / "videomme_pair" / "keep_attn")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO / "analyse" / "out" / "videomme_pair" / "layers",
        help="reuse SigLIP/CLIP layer dumps (cache/ under this dir)",
    )
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument(
        "--dim",
        type=float,
        default=0.55,
        help="brightness of unkept patches on the heatmap (1=same as kept)",
    )
    p.add_argument(
        "--mask-dirs",
        default=None,
        help="override probe NPZ dirs, e.g. fastvid=/path/masks,vidcom2=/path/masks",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    keys, seed_used = resolve_keys(
        methods=methods,
        dataset=args.dataset,
        keys_file=args.keys_file,
        key=args.key,
        n=args.n,
        seed=args.seed,
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
    for key in keys:
        print(f"=== {key} ===")
        visualize_keep_attn(
            methods,
            args.dataset,
            key,
            args.out_dir,
            args.cache_dir,
            n_show=args.n_show,
            cell=args.cell,
            n_layers=args.n_layers,
            force=args.force,
            dim=args.dim,
            mask_dirs=parse_mask_dirs(args.mask_dirs),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
