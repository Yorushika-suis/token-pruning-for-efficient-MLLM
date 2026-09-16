"""Range-extract a few Video-MME mp4s into $HF_HOME/videomme/data.

lmms-eval looks for `$HF_HOME/videomme/data/{videoID}.mp4`. Parquet-only
downloads are not enough. Full zip volumes are ~101 GB; this script uses the
zhr zip index and HTTP Range on ModelScope.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import zlib
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


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "weights"))
    ids = list(argv or sys.argv[1:] or DEFAULT_IDS)
    entries = json.loads(INDEX_PATH.read_text())
    by_stem = {Path(e["name"]).stem: e for e in entries}
    missing = [i for i in ids if i not in by_stem]
    if missing:
        raise SystemExit(f"videoIDs not in zip index: {missing}")
    dest = out_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"HF_HOME={hf_home()}", flush=True)
    print(f"writing {len(ids)} mp4s to {dest}", flush=True)
    for vid in ids:
        stem, n, cached = fetch_one(by_stem[vid], dest)
        tag = "cached" if cached else "wrote"
        print(f"  {tag} {stem}.mp4 ({n} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
