#!/usr/bin/env python3
"""Line up geom keep / hybrid / layer montages so transforms sit in one row.

Reads NPZ (and existing PNGs) under show_geom --out-dir. No re-inference.
Also looks under ThirdParty/*/analyse/out/... if a run chdir'd before writing.
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

from analyse.lib.geom import apply_geom, parse_geoms
from analyse.lib.geom_infer import probe_mask_dir
from analyse.lib.keys import ensure_pngs_jsons, read_keys_file, skip_existing, write_keys_manifest
from analyse.lib.masks import load_keep, subsample_frames
from analyse.lib.media import load_videomme_model_frames, uniform_indices
from analyse.lib.paths import (
    N_SHOW_FRAMES,
    ROOT as REPO,
    check_dataset_methods,
    normalize_key,
    parse_mask_dirs,
    parse_methods,
)
from analyse.lib.render import montage_grid, overlay_keep, resize_nearest, save_png, _label_bar


def _keeps_aligned(keep_hw, frame_idx, n_frames):
    if keep_hw.shape[0] == 1:
        return np.repeat(keep_hw, n_frames, axis=0)
    used = [int(i) for i in frame_idx if int(i) < keep_hw.shape[0]]
    if len(used) != n_frames:
        sub, _ = subsample_frames(keep_hw, n_frames)
        return sub
    return keep_hw[used]


def _run_roots(geom_dir: Path) -> list[Path]:
    roots = [geom_dir.resolve()]
    try:
        rel = geom_dir.resolve().relative_to(REPO)
    except ValueError:
        return roots
    for extra in (
        REPO / "ThirdParty" / "VidCom2" / rel,
        REPO / "ThirdParty" / "FastVID" / "fastvid_llavavideo" / rel,
    ):
        if extra.is_dir() and extra not in roots:
            roots.append(extra)
    return roots


def find_npz(geom_dir: Path, method: str, geom: str, key: str) -> Path | None:
    name = f"{normalize_key(key)}.npz"
    for root in _run_roots(geom_dir):
        path = probe_mask_dir(root, method, geom) / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def find_png(geom_dir: Path, rel: str) -> Path | None:
    for root in _run_roots(geom_dir):
        path = root / rel
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def hstack_labeled(images: list[np.ndarray], titles: list[str], height: int) -> np.ndarray:
    gap = 6
    title_h = 28
    scaled = []
    for im in images:
        h, w = im.shape[:2]
        nw = max(1, int(round(w * height / max(h, 1))))
        scaled.append(resize_nearest(im, height, nw))
    widths = [im.shape[1] for im in scaled]
    out_w = sum(widths) + gap * (len(scaled) + 1)
    out_h = title_h + height + gap * 2
    canvas = np.full((out_h, out_w, 3), 12, dtype=np.uint8)
    x = gap
    for im, title, w in zip(scaled, titles, widths):
        canvas[gap : gap + title_h, x : x + w] = _label_bar(title, w, title_h)
        canvas[gap + title_h : gap + title_h + height, x : x + w] = im
        x += w + gap
    return canvas


def discover_keys(geom_dir: Path, methods: list[str], geoms: list[str]) -> list[str]:
    names: set[str] | None = None
    for method in methods:
        for geom in geoms:
            found: set[str] = set()
            for root in _run_roots(geom_dir):
                d = probe_mask_dir(root, method, geom)
                if d.is_dir():
                    found.update(p.stem for p in d.glob("*.npz"))
            if not found:
                continue
            names = found if names is None else names & found
    return sorted(names or [])


def visualize_keep_row(
    *,
    methods: list[str],
    key: str,
    geoms: list[str],
    geom_dir: Path,
    frames64: list[np.ndarray],
    show_idx: np.ndarray,
    orig_mask_dirs: dict[str, Path] | None,
    out_dir: Path,
    cell: int,
    force: bool,
) -> None:
    key_n = normalize_key(key)
    pngs, jsons = ensure_pngs_jsons(out_dir)
    frames_by_geom = {g: [apply_geom(frames64[int(i)], g) for i in show_idx] for g in geoms}
    combined_cols: list[list[np.ndarray]] = []
    combined_titles: list[str] = []
    sidecar: dict = {"key": key_n, "geoms": geoms, "methods": {}}

    for method in methods:
        cols = []
        titles = []
        used_geoms = []
        for geom in geoms:
            if geom == "orig":
                directory = (orig_mask_dirs or {}).get(method)
                if directory is None:
                    continue
                npz = directory / f"{key_n}.npz"
            else:
                npz = find_npz(geom_dir, method, geom, key_n)
            if npz is None or not npz.is_file():
                print(f"skip {method} {key_n} {geom}: no NPZ")
                continue
            rec = load_keep(npz, method=method)
            keeps = _keeps_aligned(rec.keep_hw, show_idx, len(show_idx))
            frames = frames_by_geom[geom]
            cols.append([overlay_keep(fr, km) for fr, km in zip(frames, keeps)])
            titles.append(f"{geom}  ret={rec.retention:.3f}")
            used_geoms.append(geom)
            sidecar["methods"].setdefault(method, {})[geom] = {
                "npz": str(npz),
                "retention": rec.retention,
                "n_out": rec.n_out,
                "spatial_kept": rec.spatial_kept,
            }
        if not cols:
            continue
        rows = list(map(list, zip(*cols)))
        row_titles = [f"f{int(i)}" for i in show_idx]
        png = pngs / f"{key_n}_{method}_keep.png"
        js = jsons / f"{key_n}_{method}_keep.json"
        if not (skip_existing(png, force) and skip_existing(js, force)):
            grid = montage_grid(rows, col_titles=titles, row_titles=row_titles, cell_h=cell, cell_w=cell)
            save_png(png, grid)
            js.write_text(json.dumps(sidecar["methods"].get(method, {}), indent=2) + "\n", encoding="utf-8")
            print(f"keep-row {method} {key_n} → {png}")
        else:
            print(f"skip keep-row {method} {key_n}")
        for title, col in zip((f"{method} {g}" for g in used_geoms), cols):
            combined_titles.append(title)
            combined_cols.append(col)

    if len(methods) > 1 and combined_cols:
        png = pngs / f"{key_n}_keep.png"
        if skip_existing(png, force):
            print(f"skip keep-row combined {key_n}")
        else:
            rows = list(map(list, zip(*combined_cols)))
            grid = montage_grid(
                rows,
                col_titles=combined_titles,
                row_titles=[f"f{int(i)}" for i in show_idx],
                cell_h=cell,
                cell_w=cell,
            )
            save_png(png, grid)
            print(f"keep-row combined {key_n} → {png}")
    (jsons / f"{key_n}_keep.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def visualize_png_row(
    *,
    kind: str,
    methods: list[str],
    key: str,
    geoms: list[str],
    geom_dir: Path,
    out_dir: Path,
    height: int,
    force: bool,
) -> None:
    key_n = normalize_key(key)
    pngs, jsons = ensure_pngs_jsons(out_dir)
    suffix = {"keep-attn": "keep_attn", "layers": "layers"}[kind]
    src_subdir = {"keep-attn": "keep_attn", "layers": "layers"}[kind]
    for method in methods:
        imgs = []
        titles = []
        used = []
        for geom in geoms:
            if geom == "orig":
                rel = f"{src_subdir}/pngs/{key_n}_{method}_{suffix}.png"
            else:
                rel = f"{src_subdir}/pngs/{key_n}_{geom}_{method}_{suffix}.png"
            path = find_png(geom_dir, rel)
            if path is None:
                print(f"skip {kind} {method} {key_n} {geom}: no PNG {rel}")
                continue
            imgs.append(load_rgb(path))
            titles.append(geom)
            used.append(str(path))
        if not imgs:
            continue
        png = pngs / f"{key_n}_{method}_{suffix}.png"
        js = jsons / f"{key_n}_{method}_{suffix}.json"
        if skip_existing(png, force) and skip_existing(js, force):
            print(f"skip {kind}-row {method} {key_n}")
            continue
        save_png(png, hstack_labeled(imgs, titles, height=height))
        js.write_text(json.dumps({"sources": used, "geoms": titles}, indent=2) + "\n", encoding="utf-8")
        print(f"{kind}-row {method} {key_n} → {png}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid,vidcom2")
    p.add_argument("--dataset", default="videomme", choices=["videomme"])
    p.add_argument("--key", default=None)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--geoms", default="rot90,rot180,rot270,hflip")
    p.add_argument("--geom-dir", type=Path, default=REPO / "analyse" / "out" / "videomme_pair" / "geom")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--orig-mask-dirs", default=None, help="include orig column: fastvid=/masks,vidcom2=/masks")
    p.add_argument("--kinds", default="keep,keep-attn,layers")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=160)
    p.add_argument("--png-height", type=int, default=720)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    geoms = parse_geoms(args.geoms)
    orig_dirs = parse_mask_dirs(args.orig_mask_dirs)
    geom_dir = (Path.cwd() / args.geom_dir).resolve()
    out_dir = (Path.cwd() / args.out_dir).resolve() if args.out_dir else geom_dir / "row"
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = set(kinds) - {"keep", "keep-attn", "layers"}
    if unknown:
        raise SystemExit(f"unknown --kinds {sorted(unknown)}")

    keys_file = args.keys_file
    if args.key:
        keys = [args.key]
        seed_used = None
    elif keys_file is not None:
        keys = read_keys_file(keys_file)
        seed_used = None
    elif (geom_dir / "keys.txt").is_file():
        keys = read_keys_file(geom_dir / "keys.txt")
        seed_used = None
    else:
        keys = discover_keys(geom_dir, methods, [g for g in geoms if g != "orig"])
        seed_used = None
        if not keys:
            raise SystemExit(f"no keys under {geom_dir}/probe")
    keys = [normalize_key(k) for k in keys]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_keys_manifest(
        out_dir,
        keys=keys,
        methods=methods,
        dataset=args.dataset,
        n=len(keys),
        seed=seed_used,
    )

    frames64_cache: dict[str, list[np.ndarray]] = {}
    show_idx = None
    for key in keys:
        if "keep" in kinds:
            if key not in frames64_cache:
                frames64_cache[key] = load_videomme_model_frames(key)
            frames64 = frames64_cache[key]
            if show_idx is None:
                show_idx = uniform_indices(len(frames64), args.n_show)
            visualize_keep_row(
                methods=methods,
                key=key,
                geoms=geoms,
                geom_dir=geom_dir,
                frames64=frames64,
                show_idx=show_idx,
                orig_mask_dirs=orig_dirs,
                out_dir=out_dir,
                cell=args.cell,
                force=args.force,
            )
        if "keep-attn" in kinds:
            visualize_png_row(
                kind="keep-attn",
                methods=methods,
                key=key,
                geoms=geoms,
                geom_dir=geom_dir,
                out_dir=out_dir,
                height=args.png_height,
                force=args.force,
            )
        if "layers" in kinds:
            visualize_png_row(
                kind="layers",
                methods=methods,
                key=key,
                geoms=geoms,
                geom_dir=geom_dir,
                out_dir=out_dir,
                height=args.png_height,
                force=args.force,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
