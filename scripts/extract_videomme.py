"""Range-extract the Video-MME `short` subset out of 20 ZIP64 volumes.

Why this exists: the 300 short videos total ~3 GB but are scattered across all 20
`videos_chunked_NN.zip` volumes (101 GB total). Downloading the volumes would cost 101 GB
of transfer and ~200 GB of peak JuiceFS usage for a 3 GB payload. Instead we read each
volume's central directory over HTTP Range, then fetch only the byte spans we need.

The ZIP64 sentinel handling is the part that silently corrupts things if you get it wrong:
when a central-directory size/offset field reads 0xFFFFFFFF, the real 8-byte value lives in
the 0x0001 extra field, and the values appear there *in a fixed order* covering only the
fields that were sentinelled. Reading them positionally without checking which fields were
sentinelled yields plausible-looking garbage offsets.

Usage:
    python extract_videomme.py index     # build + cache the central-directory index
    python extract_videomme.py extract   # fetch the short subset (resumable)
"""
import argparse
import io
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
ENDPOINT = "https://hf-mirror.com"
REPO_PATH = "datasets/lmms-lab/Video-MME/resolve/main"
N_CHUNKS = 20
INDEX_PATH = f"{W}/data/videomme_zip_index.json"
OUT_DIR = f"{W}/data/videos_short"
MANIFEST = f"{W}/data/videomme_short_manifest.json"

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDFH_SIG = b"PK\x01\x02"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "flashvid-adts-spatial/1.0"


def url_for(chunk: int) -> str:
    return f"{ENDPOINT}/{REPO_PATH}/videos_chunked_{chunk:02d}.zip"


def get_range(url: str, start: int, length: int, retries: int = 5) -> bytes:
    """HTTP Range read with retries. `start` may be negative for a suffix read."""
    if start < 0:
        rng = f"bytes={start}"
    else:
        rng = f"bytes={start}-{start + length - 1}"
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, headers={"Range": rng}, timeout=120, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - transient network, we just retry
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range read failed for {url} {rng}: {last}")


def content_length(url: str) -> int:
    r = SESSION.head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def parse_zip64_extra(extra: bytes, need_usz: bool, need_csz: bool, need_lho: bool):
    """Return (usz, csz, lho) overrides found in the 0x0001 extra field, else Nones.

    Only the sentinelled fields are present, and only in this order.
    """
    pos = 0
    while pos + 4 <= len(extra):
        hid, dsz = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4 : pos + 4 + dsz]
        pos += 4 + dsz
        if hid != 0x0001:
            continue
        off = 0
        usz = csz = lho = None
        if need_usz and off + 8 <= len(body):
            usz = struct.unpack_from("<Q", body, off)[0]
            off += 8
        if need_csz and off + 8 <= len(body):
            csz = struct.unpack_from("<Q", body, off)[0]
            off += 8
        if need_lho and off + 8 <= len(body):
            lho = struct.unpack_from("<Q", body, off)[0]
            off += 8
        return usz, csz, lho
    return None, None, None


def read_central_directory(chunk: int):
    """Return list of dicts describing every entry in one zip volume."""
    url = url_for(chunk)
    size = content_length(url)

    tail = get_range(url, -min(262144, size), 0)
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError(f"chunk {chunk}: no EOCD in tail")
    cd_size, cd_off = struct.unpack_from("<II", tail, idx + 12)
    total_entries = struct.unpack_from("<H", tail, idx + 10)[0]

    # ZIP64 path: EOCD fields are sentinelled, real values live in the zip64 EOCD record.
    loc = tail.rfind(EOCD64_LOC_SIG, 0, idx)
    if loc >= 0 and (cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF or total_entries == 0xFFFF):
        z64_off = struct.unpack_from("<Q", tail, loc + 8)[0]
        z64 = get_range(url, z64_off, 56)
        if z64[:4] != EOCD64_SIG:
            raise RuntimeError(f"chunk {chunk}: bad zip64 EOCD signature")
        total_entries = struct.unpack_from("<Q", z64, 32)[0]
        cd_size = struct.unpack_from("<Q", z64, 40)[0]
        cd_off = struct.unpack_from("<Q", z64, 48)[0]

    cd = get_range(url, cd_off, cd_size)
    entries = []
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == CDFH_SIG:
        method = struct.unpack_from("<H", cd, pos + 10)[0]
        crc = struct.unpack_from("<I", cd, pos + 16)[0]
        csz = struct.unpack_from("<I", cd, pos + 20)[0]
        usz = struct.unpack_from("<I", cd, pos + 24)[0]
        nlen, elen, clen = struct.unpack_from("<HHH", cd, pos + 28)
        lho = struct.unpack_from("<I", cd, pos + 42)[0]
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]

        z_usz, z_csz, z_lho = parse_zip64_extra(
            extra, usz == 0xFFFFFFFF, csz == 0xFFFFFFFF, lho == 0xFFFFFFFF
        )
        if usz == 0xFFFFFFFF and z_usz is not None:
            usz = z_usz
        if csz == 0xFFFFFFFF and z_csz is not None:
            csz = z_csz
        if lho == 0xFFFFFFFF and z_lho is not None:
            lho = z_lho

        entries.append(
            {"chunk": chunk, "name": name, "method": method, "crc": crc,
             "csz": csz, "usz": usz, "lho": lho}
        )
        pos += 46 + nlen + elen + clen

    if len(entries) != total_entries:
        print(f"  WARNING chunk {chunk}: parsed {len(entries)} entries, EOCD says {total_entries}",
              flush=True)
    return entries


def cmd_index():
    all_entries = []
    for c in range(1, N_CHUNKS + 1):
        t0 = time.time()
        ents = read_central_directory(c)
        all_entries.extend(ents)
        print(f"chunk {c:02d}: {len(ents):4d} entries  ({time.time()-t0:.1f}s)", flush=True)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w") as fh:
        json.dump(all_entries, fh)
    print(f"\ntotal entries: {len(all_entries)}", flush=True)

    methods = {}
    for e in all_entries:
        methods[e["method"]] = methods.get(e["method"], 0) + 1
    print("compression methods:", methods, "(0=stored, 8=deflate)", flush=True)
    print("sample names:", [e["name"] for e in all_entries[:4]], flush=True)

    short = json.load(open(f"{W}/data/short_subset.json"))["short_video_ids"]
    by_stem = {}
    for e in all_entries:
        stem = os.path.splitext(os.path.basename(e["name"]))[0]
        by_stem.setdefault(stem, []).append(e)
    hit = [v for v in short if v in by_stem]
    miss = [v for v in short if v not in by_stem]
    tot = sum(by_stem[v][0]["usz"] for v in hit)
    print(f"\nshort videos matched: {len(hit)}/{len(short)}  missing: {len(miss)}", flush=True)
    if miss:
        print("  missing sample:", miss[:5], flush=True)
    print(f"short subset uncompressed size: {tot/1e9:.3f} GB", flush=True)
    print(f"wrote {INDEX_PATH}", flush=True)


def fetch_one(entry):
    """Range-read one member and write it out atomically. Returns (name, bytes, cached)."""
    stem = os.path.splitext(os.path.basename(entry["name"]))[0]
    dst = os.path.join(OUT_DIR, f"{stem}.mp4")
    if os.path.exists(dst) and os.path.getsize(dst) == entry["usz"]:
        return stem, entry["usz"], True

    url = url_for(entry["chunk"])
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

    tmp = dst + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dst)
    return stem, len(data), False


def cmd_extract(limit=None, workers=4):
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = json.load(open(INDEX_PATH))
    short = json.load(open(f"{W}/data/short_subset.json"))["short_video_ids"]
    by_stem = {}
    for e in entries:
        by_stem.setdefault(os.path.splitext(os.path.basename(e["name"]))[0], e)
    targets = [by_stem[v] for v in short if v in by_stem]
    if limit:
        targets = targets[:limit]
    print(f"extracting {len(targets)} videos with {workers} workers", flush=True)

    done, cached, failed, nbytes = 0, 0, [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, e): e for e in targets}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                stem, n, was_cached = fut.result()
                done += 1
                nbytes += n
                cached += int(was_cached)
                if done % 25 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(targets)}  {nbytes/1e9:.2f} GB  {el:.0f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed.append((os.path.basename(e["name"]), str(exc)))

    print(f"\ndone={done} cached={cached} failed={len(failed)} bytes={nbytes/1e9:.3f} GB "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    for name, err in failed[:10]:
        print("  FAIL", name, err, flush=True)

    with open(MANIFEST, "w") as fh:
        json.dump({"targets": targets, "failed": failed}, fh, indent=1)
    print("wrote", MANIFEST, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["index", "extract"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "index":
        cmd_index()
    else:
        sys.exit(cmd_extract(a.limit, a.workers))
