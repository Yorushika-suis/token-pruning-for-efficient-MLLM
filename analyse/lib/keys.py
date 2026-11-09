"""Shared sample-key list: pick once, reuse for keep maps and layer plots."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyse.lib.paths import merged_mask_dirs


def list_keys(methods: list[str], dataset: str, mask_dirs: dict[str, Path] | None = None) -> list[str]:
    dirs = merged_mask_dirs(mask_dirs)
    sets = []
    for method in methods:
        directory = dirs[method]
        names = {p.stem for p in directory.glob("*.npz")}
        if dataset.lower() == "gqa":
            names = {n for n in names if n.isdigit() and len(n) >= 6}
        elif dataset.lower() == "mme":
            names = {n for n in names if "_" in n and not n.isdigit()}
        elif dataset.lower() == "pope":
            names = {n for n in names if n.isdigit() and int(n) < 10000 and len(n) <= 4}
        sets.append(names)
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def sample_keys(keys: list[str], n: int, seed: int) -> list[str]:
    if n > len(keys):
        raise SystemExit(f"requested n={n} but only {len(keys)} shared keys")
    rng = np.random.default_rng(seed)
    pick = rng.choice(np.array(keys, dtype=object), size=n, replace=False)
    return [str(x) for x in pick]


def read_keys_file(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            keys = obj
        else:
            keys = obj.get("keys") or obj.get("sample_keys")
        if not keys:
            raise SystemExit(f"no keys in {path}")
        return [str(k).strip() for k in keys if str(k).strip()]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if not out:
        raise SystemExit(f"no keys in {path}")
    return out


def write_keys_manifest(
    dest_dir: Path,
    *,
    keys: list[str],
    methods: list[str],
    dataset: str,
    n: int | None,
    seed: int | None,
) -> tuple[Path, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    txt = dest_dir / "keys.txt"
    js = dest_dir / "jsons" / "keys.json"
    from analyse.lib.questions import write_questions
    from analyse.lib.paths import normalize_key

    keys = [normalize_key(k) for k in keys]
    txt.write_text("\n".join(keys) + "\n", encoding="utf-8")
    questions: dict = {}
    try:
        _, _, questions = write_questions(dest_dir, dataset, keys)
    except Exception as exc:
        print(f"warning: could not write question text ({exc})")
    payload = {
        "methods": methods,
        "dataset": dataset,
        "n": n if n is not None else len(keys),
        "seed": seed,
        "keys": keys,
        "cli_keys": [f"--key {k}" for k in keys],
        "questions": questions,
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt, js


def ensure_pngs_jsons(root: Path) -> tuple[Path, Path]:
    pngs = Path(root) / "pngs"
    jsons = Path(root) / "jsons"
    pngs.mkdir(parents=True, exist_ok=True)
    jsons.mkdir(parents=True, exist_ok=True)
    return pngs, jsons


def skip_existing(path: Path, force: bool) -> bool:
    """True → do not regenerate."""
    if force:
        return False
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def keep_artifact_paths(root: Path, key: str) -> tuple[Path, Path]:
    from analyse.lib.paths import normalize_key

    pngs, jsons = ensure_pngs_jsons(root)
    stem = normalize_key(key)
    return pngs / f"{stem}.png", jsons / f"{stem}.json"


def layer_png_path(root: Path, key: str, method: str) -> Path:
    from analyse.lib.paths import normalize_key

    pngs, _ = ensure_pngs_jsons(root)
    return pngs / f"{normalize_key(key)}_{method}_layers.png"


def layer_json_path(root: Path, key: str) -> Path:
    from analyse.lib.paths import normalize_key

    _, jsons = ensure_pngs_jsons(root)
    return jsons / f"{normalize_key(key)}_layers.json"


def resolve_keys(
    *,
    methods: list[str],
    dataset: str,
    keys_file: Path | None,
    key: str | None,
    n: int | None,
    seed: int,
    mask_dirs: dict[str, Path] | None = None,
) -> tuple[list[str], int | None]:
    """Return (keys, seed_used). seed_used is None if the list was not RNG-sampled here."""
    if keys_file is not None:
        return read_keys_file(keys_file), None
    if key:
        return [key], None
    if n is not None:
        return sample_keys(list_keys(methods, dataset, mask_dirs=mask_dirs), n, seed), seed
    raise SystemExit("need --keys-file, --key, or --n")
