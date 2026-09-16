"""v4-lite scientific hashing, NPZ, and journal helpers."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

HASH_SPEC = "v4-json-1"
SCHEMA_VERSION = "repro-artifact/v4-lite-1"
FORBIDDEN_SCIENCE_KEYS = {
    "run_id",
    "started_at",
    "finished_at",
    "generated_at",
    "mtime",
    "duration",
    "elapsed_seconds",
    "physical_path",
    "pid",
    "worker_pid",
    "gpu_uuid",
    "heartbeat",
    "wall_time",
    "log_path",
}


def canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_science(value) -> None:
    canonical(value)
    if isinstance(value, dict):
        forbidden = FORBIDDEN_SCIENCE_KEYS.intersection(value)
        if forbidden:
            raise ValueError("execution data in science: " + repr(sorted(forbidden)))
        for child in value.values():
            check_science(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_science(child)


def _valid_member(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "_-" for c in name)


def write_npz(path, arrays, metadata, *, replace: bool = False) -> Path:
    check_science(metadata)
    if "meta_json" in arrays:
        raise ValueError("meta_json is reserved")
    values = dict(arrays)
    values["meta_json"] = np.frombuffer(canonical(metadata), dtype=np.uint8)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() or p.is_symlink():
        if not replace:
            raise FileExistsError(str(p))
        p.unlink()
    tmp = p.parent / ("." + p.name + "." + str(uuid.uuid4()) + ".tmp")
    packed = {}
    for name in sorted(values):
        if not _valid_member(name):
            raise ValueError("invalid NPZ member name: " + name)
        arr = np.asarray(values[name])
        if arr.dtype.kind not in "biuf" or not np.isfinite(arr).all():
            raise ValueError("nonfinite or unsupported scientific array: " + name)
        packed[name] = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<"))
    with tmp.open("wb") as fh:
        np.savez(fh, **packed)
    os.replace(tmp, p)
    return p


def write_failure_npz(path, arrays, metadata) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    values = dict(arrays)
    values["meta_json"] = np.frombuffer(
        json.dumps(metadata, ensure_ascii=False, default=str).encode("utf-8"),
        dtype=np.uint8,
    )
    np.savez(p, **{k: np.asarray(v) for k, v in values.items()})
    return p


def append_journal(path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line)


def mask_filename(key: list) -> str:
    parts = [str(x).replace("/", "_") for x in key]
    return "__".join(parts) + ".npz"
