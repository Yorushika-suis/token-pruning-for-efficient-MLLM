#!/usr/bin/env python3
"""Sample N keys and reuse show_keep.visualize for keep-map montages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyse.lib.keys import keep_artifact_paths, resolve_keys, write_keys_manifest
from analyse.lib.paths import (
    N_SHOW_FRAMES,
    check_dataset_methods,
    parse_mask_dirs,
    parse_methods,
)
from analyse.keep.show_keep import visualize


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", required=True)
    p.add_argument("--dataset", required=True, choices=["videomme", "gqa", "mme", "pope"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keys-file", type=Path, default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "analyse" / "out" / "sample_keep")
    p.add_argument("--n-show", type=int, default=N_SHOW_FRAMES)
    p.add_argument("--cell", type=int, default=224)
    p.add_argument(
        "--mask-dirs",
        default=None,
        help="override probe NPZ dirs, e.g. fastvid=/path/masks,vidcom2=/path/masks",
    )
    p.add_argument("--force", action="store_true", help="regenerate even if png/json exist")
    args = p.parse_args(argv)
    methods = parse_methods(args.methods)
    check_dataset_methods(args.dataset, methods)
    mask_dirs = parse_mask_dirs(args.mask_dirs)
    n = args.n if args.n is not None else (None if args.keys_file or args.key else 8)
    keys, seed_used = resolve_keys(
        methods=methods,
        dataset=args.dataset,
        keys_file=args.keys_file,
        key=args.key,
        n=n,
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
        png, js = keep_artifact_paths(args.out_dir, key)
        visualize(
            methods,
            args.dataset,
            key,
            png,
            n_show=args.n_show,
            cell=args.cell,
            mask_dirs=mask_dirs,
            json_out=js,
            force=args.force,
        )
        print(f"ok {key} → {png}")
    print(f"keys → {args.out_dir / 'keys.txt'} ({len(keys)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
