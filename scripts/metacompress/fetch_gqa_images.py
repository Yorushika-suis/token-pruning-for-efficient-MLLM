#!/usr/bin/env python3
"""Fetch only the GQA image members required by the reconstructed MT-GQA set.

The GQA image archive is a regular ZIP served with HTTP Range support.  This
script reads its central directory, requests each selected member's local ZIP
record, and writes the decompressed JPEG directly to the output directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import struct
import threading
import time
import zlib

import requests
from pathlib import Path


CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
LOCAL_HEADER = struct.Struct("<4s5H3I2H")


_thread_state = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        _thread_state.session = session
    return session


def http_range(url: str, start: int, end: int, timeout: int, retries: int = 16) -> bytes:
    expected = end - start + 1
    for attempt in range(retries):
        try:
            response = _session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(15, timeout),
            )
            response.raise_for_status()
            data = response.content
            if response.status_code != 206 or len(data) != expected:
                raise RuntimeError(
                    f"Range {start}-{end} returned HTTP {response.status_code}, {len(data)} bytes"
                )
            if len(data) != expected:
                raise RuntimeError(
                    f"Range {start}-{end} returned {len(data)} bytes, expected {expected}"
                )
            return data
        except (requests.RequestException, TimeoutError, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def parse_central(data: bytes) -> dict[str, tuple[int, int, int, int]]:
    """Return member -> (local offset, compressed size, method, filename length)."""
    members = {}
    pos = 0
    while pos + CENTRAL_HEADER.size <= len(data) and data[pos : pos + 4] == b"PK\x01\x02":
        (
            _sig,
            _made_by,
            _needed,
            _flags,
            method,
            _mtime,
            _mdate,
            _crc,
            compressed_size,
            _uncompressed_size,
            filename_size,
            extra_size,
            comment_size,
            _disk,
            _internal_attr,
            _external_attr,
            local_offset,
        ) = CENTRAL_HEADER.unpack_from(data, pos)
        name_start = pos + CENTRAL_HEADER.size
        name = data[name_start : name_start + filename_size].decode("utf-8")
        extra = data[name_start + filename_size : name_start + filename_size + extra_size]
        if 0xFFFFFFFF in (compressed_size, _uncompressed_size, local_offset):
            extra_pos = 0
            while extra_pos + 4 <= len(extra):
                field_id, field_size = struct.unpack_from("<HH", extra, extra_pos)
                field = extra[extra_pos + 4 : extra_pos + 4 + field_size]
                if field_id == 0x0001:
                    field_pos = 0
                    if _uncompressed_size == 0xFFFFFFFF:
                        _uncompressed_size = struct.unpack_from("<Q", field, field_pos)[0]
                        field_pos += 8
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack_from("<Q", field, field_pos)[0]
                        field_pos += 8
                    if local_offset == 0xFFFFFFFF:
                        local_offset = struct.unpack_from("<Q", field, field_pos)[0]
                    break
                extra_pos += 4 + field_size
        members[name] = (local_offset, compressed_size, method, filename_size)
        pos += CENTRAL_HEADER.size + filename_size + extra_size + comment_size
    if pos != len(data):
        raise RuntimeError(f"central directory parse stopped at {pos} of {len(data)} bytes")
    return members


def fetch_member(
    archive_url: str,
    member: str,
    info: tuple[int, int, int, int],
    output: Path,
    timeout: int,
) -> tuple[str, int]:
    local_offset, compressed_size, expected_method, expected_filename_size = info
    # Local extra fields are small, but include a generous header allowance so
    # the actual local header can be parsed from the same range request.
    header_end = local_offset + LOCAL_HEADER.size + expected_filename_size + 64 - 1
    payload = http_range(archive_url, local_offset, header_end + compressed_size, timeout)
    header = LOCAL_HEADER.unpack_from(payload, 0)
    if header[0] != b"PK\x03\x04":
        raise RuntimeError(f"{member}: invalid local header signature")
    _sig, _version, flags, method, _mtime, _mdate, _crc, csize, _usize, filename_size, extra_size = header
    if flags & 0x08:
        raise RuntimeError(f"{member}: data descriptors are unsupported")
    if method != expected_method or method != 8:
        raise RuntimeError(f"{member}: expected deflate method 8, got {method}")
    if csize != compressed_size:
        raise RuntimeError(f"{member}: central/local compressed sizes differ")
    name_start = LOCAL_HEADER.size
    local_name = payload[name_start : name_start + filename_size].decode("utf-8")
    if local_name != member:
        raise RuntimeError(f"{member}: local name is {local_name!r}")
    data_start = LOCAL_HEADER.size + filename_size + extra_size
    compressed = payload[data_start : data_start + compressed_size]
    if len(compressed) != compressed_size:
        raise RuntimeError(f"{member}: truncated compressed member")
    image = zlib.decompress(compressed, -15)
    if not image.startswith(b"\xff\xd8\xff"):
        raise RuntimeError(f"{member}: decompressed payload is not JPEG")
    destination = output / Path(member).name
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(image)
    temporary.replace(destination)
    return member, len(image)


def fetch_batch(
    archive_url: str,
    jobs: list[tuple[str, tuple[int, int, int, int]]],
    output: Path,
    timeout: int,
) -> list[tuple[str, int]]:
    """Fetch one physical archive span and unpack all selected members in it."""
    start = jobs[0][1][0]
    end = max(local_offset + 30 + filename_size + 128 + compressed_size
              for _member, (local_offset, compressed_size, _method, filename_size) in jobs)
    payload = http_range(archive_url, start, end - 1, timeout)
    results = []
    for member, info in jobs:
        local_offset, compressed_size, expected_method, expected_filename_size = info
        relative = local_offset - start
        header = LOCAL_HEADER.unpack_from(payload, relative)
        if header[0] != b"PK\x03\x04":
            raise RuntimeError(f"{member}: invalid local header signature in batch")
        _sig, _version, flags, method, _mtime, _mdate, _crc, csize, _usize, filename_size, extra_size = header
        if flags & 0x08 or method != expected_method or method != 8 or csize != compressed_size:
            raise RuntimeError(f"{member}: local header does not match central directory")
        name_start = relative + LOCAL_HEADER.size
        local_name = payload[name_start : name_start + filename_size].decode("utf-8")
        if local_name != member or filename_size != expected_filename_size:
            raise RuntimeError(f"{member}: local name does not match central directory")
        data_start = name_start + filename_size + extra_size
        compressed = payload[data_start : data_start + compressed_size]
        if len(compressed) != compressed_size:
            raise RuntimeError(f"{member}: truncated compressed member in batch")
        image = zlib.decompress(compressed, -15)
        if not image.startswith(b"\xff\xd8\xff"):
            raise RuntimeError(f"{member}: decompressed payload is not JPEG")
        destination = output / Path(member).name
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(image)
        temporary.replace(destination)
        results.append((member, len(image)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", required=True)
    parser.add_argument("--central-directory", type=Path, required=True)
    parser.add_argument("--image-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--merge-gap", type=int, default=2_000_000,
                        help="merge selected members whose local offsets are within this many bytes")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    central = parse_central(args.central_directory.read_bytes())
    image_ids = [line.strip() for line in args.image_ids.read_text().splitlines() if line.strip()]
    selected = {}
    missing = []
    for image_id in image_ids:
        member = f"images/{image_id}.jpg"
        if member not in central:
            missing.append(image_id)
        else:
            selected[member] = central[member]
    if missing:
        raise SystemExit(f"missing {len(missing)} image members; first IDs: {missing[:10]}")

    args.output.mkdir(parents=True, exist_ok=True)
    pending = sorted(
        ((member, info) for member, info in selected.items()
         if not (args.output / Path(member).name).is_file()),
        key=lambda item: item[1][0],
    )
    batches: list[list[tuple[str, tuple[int, int, int, int]]]] = []
    for member, info in pending:
        if not batches:
            batches.append([(member, info)])
            continue
        previous = batches[-1][-1][1]
        previous_end = previous[0] + 30 + previous[3] + 128 + previous[1]
        if info[0] - previous_end <= args.merge_gap:
            batches[-1].append((member, info))
        else:
            batches.append([(member, info)])
    print(f"pending {len(pending)} images in {len(batches)} Range batches", flush=True)
    total_bytes = 0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_batch, args.archive_url, batch, args.output, args.timeout)
                   for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            for _member, size in results:
                total_bytes += size
                completed += 1
            if completed % 250 == 0 or completed == len(pending):
                print(f"fetched {completed}/{len(pending)} images ({total_bytes / 1024**3:.2f} GiB)", flush=True)

    available = sorted(p.name for p in args.output.glob("*.jpg"))
    expected_names = sorted(Path(member).name for member in selected)
    missing_names = sorted(set(expected_names) - set(available))
    if missing_names:
        raise RuntimeError(
            f"output missing {len(missing_names)} selected files; first: {missing_names[:5]}"
        )

    manifest = json.loads(args.manifest.read_text())
    manifest["image_root_checked"] = str(args.output)
    manifest["available_image_count"] = len(available)
    manifest["missing_image_count"] = 0
    manifest["missing_image_ids"] = []
    manifest["training_ready"] = True
    manifest["image_archive"] = args.archive_url
    manifest["download_mode"] = "HTTP Range selective ZIP member download"
    manifest["selected_member_count"] = len(selected)
    manifest["downloaded_image_bytes"] = sum(p.stat().st_size for p in args.output.glob("*.jpg"))
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    (args.manifest.parent / "missing_image_ids.txt").write_text("")
    print(f"ready: {len(available)} images, {total_bytes / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
