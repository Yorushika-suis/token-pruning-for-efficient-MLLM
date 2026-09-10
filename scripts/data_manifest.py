"""Authoritative dataset-ID loading and coverage checks.

Every collection, feature, mask, and analysis stage must start from the same manifest. Directory
listings are useful for checking what landed, but they are never the source of sample inclusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "short_subset.json"
DEFAULT_ID_KEY = "short_video_ids"


class ManifestError(ValueError):
    """The dataset manifest or an artifact inventory violates the run contract."""


def _canonical_id(value: object, index: int) -> str:
    if isinstance(value, dict):
        if set(value) != {"videoID"}:
            raise ManifestError(
                f"ID entry {index} must contain only 'videoID'; got keys {sorted(value)}"
            )
        value = value["videoID"]
    if not isinstance(value, str) or not value:
        raise ManifestError(f"ID entry {index} must be a non-empty string")
    if value in {".", ".."} or Path(value).name != value or "\x00" in value:
        raise ManifestError(f"unsafe video ID at entry {index}: {value!r}")
    return value


def load_video_ids(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    id_key: str = DEFAULT_ID_KEY,
) -> list[str]:
    """Load the declared video IDs in manifest order and reject ambiguous schemas."""

    path = Path(manifest_path)
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read dataset manifest {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError(f"dataset manifest {path} must be a JSON object")
    if id_key not in payload:
        raise ManifestError(
            f"dataset manifest {path} has no {id_key!r}; available keys: {sorted(payload)}"
        )
    raw_ids = payload[id_key]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ManifestError(f"{id_key!r} in {path} must be a non-empty list")

    video_ids = [_canonical_id(value, index) for index, value in enumerate(raw_ids)]
    duplicates = sorted({video_id for video_id in video_ids if video_ids.count(video_id) > 1})
    if duplicates:
        raise ManifestError(f"duplicate video IDs in {path}: {duplicates[:10]}")
    return video_ids


def directory_ids(directory: str | Path, suffix: str) -> list[str]:
    """Inventory artifact IDs by exact suffix, without filtering leading underscores."""

    root = Path(directory)
    if not root.is_dir():
        raise ManifestError(f"artifact directory does not exist: {root}")
    return sorted(path.name[: -len(suffix)] for path in root.iterdir() if path.name.endswith(suffix))


def require_exact_ids(
    expected: Sequence[str],
    actual: Iterable[str],
    *,
    label: str,
) -> list[str]:
    """Hard-fail on missing, extra, duplicate, or zero artifact IDs."""

    expected_list = list(expected)
    actual_list = list(actual)
    if not expected_list:
        raise ManifestError(f"{label}: expected ID set is empty")
    duplicate_actual = sorted({item for item in actual_list if actual_list.count(item) > 1})
    if duplicate_actual:
        raise ManifestError(f"{label}: duplicate artifact IDs: {duplicate_actual[:10]}")
    expected_set, actual_set = set(expected_list), set(actual_list)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise ManifestError(
            f"{label}: ID mismatch; expected={len(expected_set)} actual={len(actual_set)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return expected_list


def require_video_files(
    expected: Sequence[str],
    video_directory: str | Path,
    *,
    suffix: str = ".mp4",
) -> list[str]:
    """Require the video directory to match the declared dataset exactly."""

    return require_exact_ids(
        expected,
        directory_ids(video_directory, suffix),
        label=f"video assets in {Path(video_directory)}",
    )


def require_video_subset(
    selected: Sequence[str],
    declared: Sequence[str],
    video_directory: str | Path,
    *,
    suffix: str = ".mp4",
) -> list[str]:
    """Validate an explicit pilot/benchmark subset without scanning it into existence.

    ``declared`` remains the authoritative parent manifest.  Extra files in the full
    video directory are expected here, so this check only permits selected IDs that
    are declared, unique, safe, and backed by regular files.
    """

    selected_ids = [_canonical_id(value, index) for index, value in enumerate(selected)]
    declared_ids = [_canonical_id(value, index) for index, value in enumerate(declared)]
    if not selected_ids:
        raise ManifestError("video subset is empty")
    duplicates = sorted(
        {video_id for video_id in selected_ids if selected_ids.count(video_id) > 1}
    )
    if duplicates:
        raise ManifestError(f"duplicate video IDs in subset: {duplicates[:10]}")
    undeclared = sorted(set(selected_ids) - set(declared_ids))
    if undeclared:
        raise ManifestError(f"video subset contains undeclared IDs: {undeclared[:10]}")

    root = Path(video_directory)
    if not root.is_dir():
        raise ManifestError(f"video directory does not exist: {root}")
    missing = [video_id for video_id in selected_ids if not (root / f"{video_id}{suffix}").is_file()]
    if missing:
        raise ManifestError(f"video subset has missing assets: {missing[:10]}")
    return selected_ids
