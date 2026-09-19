"""Range-extract Video-MME mp4s into $HF_HOME/videomme/data.

lmms-eval looks for `$HF_HOME/videomme/data/{videoID}.mp4`. Parquet-only
downloads are not enough. Full zip volumes are ~101 GB; this script uses the
zhr zip index and HTTP Range on ModelScope.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "ThirdParty/zhr/data/videomme_zip_index.json"
MS_ZIP = (
    "https://www.modelscope.cn/datasets/lmms-lab/Video-MME/resolve/master/"
    "videos_chunked_{chunk:02d}.zip"
)
DEFAULT_IDS = ["fFjv93ACGo8", "N1cdUjctpG8", "HIjX8OPuf-w", "HwnB8aCn8yE"]


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", str(ROOT / "weights"))).expanduser()


def out_dir() -> Path:
    return hf_home() / "videomme" / "data"


def get_range(url: str, start: int, length: int, retries: int = 6) -> bytes:
    import urllib.request

    if start < 0:
        rng = f"bytes={start}"
    else:
        rng = f"bytes={start}-{start + length - 1}"
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "token-pruning-extract/1.0", "Range": rng},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status in (200, 206):
                    data = resp.read()
                    if start >= 0 and len(data) != length:
                        last = f"short read {len(data)} != {length}"
                    else:
                        return data
                else:
                    last = f"HTTP {resp.status}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range read failed for {url} {rng}: {last}")


def fetch_one(entry: dict, dest_dir: Path) -> tuple[str, int, bool]:
    stem = Path(entry["name"]).stem
    dst = dest_dir / f"{stem}.mp4"
    if dst.exists() and dst.stat().st_size == entry["usz"]:
        return stem, entry["usz"], True
    url = MS_ZIP.format(chunk=entry["chunk"])
    hdr = get_range(url, entry["lho"], 30)
    if hdr[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{stem}: bad local header signature at {entry['lho']}")
    nlen, elen = struct.unpack_from("<HH", hdr, 26)
    data_off = entry["lho"] + 30 + nlen + elen
    raw = get_range(url, data_off, entry["csz"])
    if len(raw) != entry["csz"]:
        raise RuntimeError(f"{stem}: short read {len(raw)} != {entry['csz']}")
    if entry["method"] == 0:
        data = raw
    elif entry["method"] == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"{stem}: unsupported compression method {entry['method']}")
    if len(data) != entry["usz"]:
        raise RuntimeError(f"{stem}: size mismatch {len(data)} != {entry['usz']}")
    if zlib.crc32(data) & 0xFFFFFFFF != entry["crc"]:
        raise RuntimeError(f"{stem}: CRC mismatch")
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".mp4.tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return stem, len(data), False


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids", nargs="*", help="videoIDs (default: smoke four)")
    p.add_argument("--all", action="store_true", help="extract every ID in the zip index")
    p.add_argument("--from-file", type=Path, help="one videoID per line")
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="parallel workers (default: 8; I/O-bound Range downloads)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "weights"))
    ns = parse_args(argv)
    entries = json.loads(INDEX_PATH.read_text())
    by_stem = {Path(e["name"]).stem: e for e in entries}

    if ns.all:
        ids = list(by_stem)
    elif ns.from_file is not None:
        ids = [
            line.strip()
            for line in ns.from_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        ids = list(ns.ids) or list(DEFAULT_IDS)

    missing = [i for i in ids if i not in by_stem]
    if missing:
        raise SystemExit(f"videoIDs not in zip index: {missing}")

    dest = out_dir()
    dest.mkdir(parents=True, exist_ok=True)
    jobs = max(1, ns.jobs)
    print(f"HF_HOME={hf_home()}", flush=True)
    print(f"writing {len(ids)} mp4s to {dest} (jobs={jobs})", flush=True)

    ok = 0
    cached_n = 0
    wrote_n = 0
    errors: list[str] = []

    def _work(vid: str) -> tuple[str, int, bool]:
        return fetch_one(by_stem[vid], dest)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_work, vid): vid for vid in ids}
        for fut in as_completed(futs):
            vid = futs[fut]
            try:
                stem, n, cached = fut.result()
            except Exception as exc:  # noqa: BLE001
                msg = f"FAIL {vid}: {exc}"
                errors.append(msg)
                print(f"  {msg}", flush=True)
                continue
            tag = "cached" if cached else "wrote"
            if cached:
                cached_n += 1
            else:
                wrote_n += 1
            ok += 1
            print(f"  {tag} {stem}.mp4 ({n} bytes)  [{ok}/{len(ids)}]", flush=True)

    print(f"done ok={ok} wrote={wrote_n} cached={cached_n} fail={len(errors)}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
