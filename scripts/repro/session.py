"""Shared probe session: output dir, restore stack, journal."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

from artifacts import append_journal, mask_filename, write_failure_npz, write_npz
from context import current_key, current_meta
from identity import identity

ROOT = Path(__file__).resolve().parents[2]


class ProbeSession:
    def __init__(self, method: str):
        self.method = method
        self.restorers = []
        self.identities = {}
        run_dir = os.environ.get("REPRO_PROBE_DIR")
        if not run_dir:
            run_dir = str(ROOT / "runs" / method / "probe")
        self.root = Path(run_dir)
        (self.root / "masks").mkdir(parents=True, exist_ok=True)
        (self.root / "sidecars" / "failures").mkdir(parents=True, exist_ok=True)
        self.journal = self.root / "sidecars" / "journal.jsonl"
        self.features = os.environ.get("REPRO_PROBE_FEATURES") == "1"
        (self.root / "predictions").mkdir(parents=True, exist_ok=True)

    def remember(self, name, original, restore):
        self.identities[name] = identity(original)
        self.restorers.append(restore)

    def restore(self) -> None:
        errors = []
        for fn in reversed(self.restorers):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        self.restorers.clear()
        if errors:
            raise RuntimeError("probe restore failed: " + "; ".join(errors))

    def write_mask(self, arrays: dict, metadata: dict, config_hash: str) -> None:
        key = current_key()
        meta = {
            "coverage": [key],
            "method": self.method,
            **current_meta(),
            **metadata,
        }
        path = self.root / "masks" / mask_filename(key)
        write_npz(path, arrays, meta, replace=True)
        append_journal(self.journal, {"key": key, "config_hash": config_hash, "ok": True})

    def write_failure(self, exc: BaseException, arrays: dict | None = None) -> None:
        key = current_key()
        dest = self.root / "sidecars" / "failures" / mask_filename(key).removesuffix(".npz")
        dest.mkdir(parents=True, exist_ok=True)
        info = {
            "key": key,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            **current_meta(),
        }
        (dest / "difference.json").write_text(
            __import__("json").dumps(info, indent=2, default=str),
            encoding="utf-8",
        )
        write_failure_npz(dest / "observation.npz", arrays or {}, info)
        append_journal(
            self.journal,
            {"key": key, "config_hash": None, "ok": False, "error": type(exc).__name__},
        )

    def mask_path_for(self, key: list | None = None) -> Path:
        from artifacts import mask_filename

        return self.root / "masks" / mask_filename(key if key is not None else current_key())

    def prediction_path_for(self, key: list | None = None) -> Path:
        from artifacts import mask_filename

        stem = mask_filename(key if key is not None else current_key()).removesuffix(".npz")
        return self.root / "predictions" / f"{stem}.txt"

    def sample_is_done(self, key: list | None = None) -> bool:
        path = self.mask_path_for(key)
        return path.is_file() and path.stat().st_size > 0

    def save_prediction(self, text: str, key: list | None = None) -> None:
        dest = self.prediction_path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text if isinstance(text, str) else str(text), encoding="utf-8")

    def load_prediction(self, key: list | None = None) -> str | None:
        dest = self.prediction_path_for(key)
        if not dest.is_file():
            return None
        return dest.read_text(encoding="utf-8")


SESSION: ProbeSession | None = None


def session() -> ProbeSession:
    if SESSION is None:
        raise RuntimeError("probe session is not installed")
    return SESSION
