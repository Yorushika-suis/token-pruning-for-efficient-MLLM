#!/usr/bin/env python3
"""Pick a shared Video-MME (or other) key list and optionally run keep + layers on it.

Writes keys.txt / keys.json (each key is the same --key you'd pass to show_keep / show_layers).
Default: sample then run both keep maps and vision-encoder layer plots.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.lib.keys import keep_artifact_paths, list_keys, resolve_keys, sample_keys, write_keys_manifest
from analyse.lib.paths import N_SHOW_FRAMES, check_dataset_methods, parse_methods
from analyse.keep.show_keep import visualize
from analyse.layers.show_layers import run_one


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="fastvid,vidcom2")
    p.add_argument("--dataset", default="videomme", choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keys-file", type=Path, default=None, help="reuse an existing keys.txt/json instead of sampling")
    p.add_argument("--key", default=None, help="single key; ignores --n")
    p.add_argument("--out-dir", type=Path, default=ROOT / "analyse" / "out" / "videomme_pair")
    p.add_argument("--keep-dir", type=Path, default=None, help="default: <out-dir>/keep")
    p.add_argument("--layers-dir", type=Path, default=None, help="default: <out-dir>/layers")
    p.add_argument("--keys-only", action="store_true", help="only write keys.txt/json, do not plot")
    p.add_argument("--skip-keep", action="store_true")
    p.add_argument("--skip-layers", action="store_true")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=224)
    p.add_argument("--layer-cell", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = p.parse_args(argv)

    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.keys_file or args.key:
        keys, seed_used = resolve_keys(
            methods=methods,
            dataset=args.dataset,
            keys_file=args.keys_file,
            key=args.key,
            n=None if args.key or args.keys_file else args.n,
            seed=args.seed,
        )
        n_rec = len(keys)
    else:
        pool = list_keys(methods, args.dataset)
        keys = sample_keys(pool, args.n, args.seed)
        seed_used = args.seed
        n_rec = args.n

    txt, js = write_keys_manifest(
        args.out_dir,
        keys=keys,
        methods=methods,
        dataset=args.dataset,
        n=n_rec,
        seed=seed_used,
    )
    print(f"keys ({len(keys)}) → {txt}")
    print(f"manifest → {js}")
    print(f"questions → {args.out_dir / 'questions.txt'}")
    from analyse.lib.paths import normalize_key
    from analyse.lib.questions import format_question_block, lookup_question

    for k in keys:
        kn = normalize_key(k)
        print(f"  --key {kn}")
        print(format_question_block(kn, lookup_question(args.dataset, kn)).rstrip())
        print()

    if args.keys_only:
        return 0

    keep_dir = args.keep_dir or (args.out_dir / "keep")
    layers_dir = args.layers_dir or (args.out_dir / "layers")
    if not args.skip_keep:
        keep_dir.mkdir(parents=True, exist_ok=True)
        write_keys_manifest(keep_dir, keys=keys, methods=methods, dataset=args.dataset, n=n_rec, seed=seed_used)
        for key in keys:
            png, js = keep_artifact_paths(keep_dir, key)
            visualize(
                methods,
                args.dataset,
                key,
                png,
                n_show=args.n_show,
                cell=args.cell,
                json_out=js,
                force=args.force,
            )
            print(f"keep {key} → {png}")
    if not args.skip_layers:
        layers_dir.mkdir(parents=True, exist_ok=True)
        write_keys_manifest(layers_dir, keys=keys, methods=methods, dataset=args.dataset, n=n_rec, seed=seed_used)
        for key in keys:
            print(f"=== layers {key} ===")
            run_one(
                methods,
                args.dataset,
                key,
                layers_dir,
                args.n_show,
                args.layer_cell,
                args.n_layers,
                force=args.force,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
