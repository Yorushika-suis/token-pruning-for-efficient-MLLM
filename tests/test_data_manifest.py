import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from data_manifest import (  # noqa: E402
    ManifestError,
    directory_ids,
    load_video_ids,
    require_exact_ids,
    require_video_files,
    require_video_subset,
)


def write_manifest(path, ids):
    path.write_text(json.dumps({"short_video_ids": ids}), encoding="utf-8")


def test_manifest_preserves_declared_ids_and_leading_underscore(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(path, ["b", "_legal-id", "a"])
    assert load_video_ids(path) == ["b", "_legal-id", "a"]


@pytest.mark.parametrize("bad_id", ["", ".", "..", "../escape", "a/b", "a\x00b", 3])
def test_manifest_rejects_unsafe_or_non_string_ids(tmp_path, bad_id):
    path = tmp_path / "manifest.json"
    write_manifest(path, [bad_id])
    with pytest.raises(ManifestError):
        load_video_ids(path)


def test_manifest_rejects_duplicates_and_schema_guessing(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    write_manifest(duplicate, ["a", "a"])
    with pytest.raises(ManifestError, match="duplicate"):
        load_video_ids(duplicate)

    alternate = tmp_path / "alternate.json"
    alternate.write_text(json.dumps({"videos": ["a"]}), encoding="utf-8")
    with pytest.raises(ManifestError, match="short_video_ids"):
        load_video_ids(alternate)


def test_directory_inventory_does_not_drop_leading_underscore(tmp_path):
    (tmp_path / "_valid.npz").touch()
    (tmp_path / "plain.npz").touch()
    (tmp_path / "_helper.pt").touch()
    assert directory_ids(tmp_path, ".npz") == ["_valid", "plain"]


def test_exact_coverage_reports_missing_and_extra():
    with pytest.raises(ManifestError, match=r"missing=\['b'\].*extra=\['c'\]"):
        require_exact_ids(["a", "b"], ["a", "c"], label="test artifacts")


def test_video_assets_must_match_manifest(tmp_path):
    (tmp_path / "_a.mp4").touch()
    (tmp_path / "b.mp4").touch()
    assert require_video_files(["_a", "b"], tmp_path) == ["_a", "b"]

    (tmp_path / "extra.mp4").touch()
    with pytest.raises(ManifestError, match="extra"):
        require_video_files(["_a", "b"], tmp_path)


def test_explicit_video_subset_uses_parent_manifest(tmp_path):
    for video_id in ("_a", "b", "unused"):
        (tmp_path / f"{video_id}.mp4").touch()
    assert require_video_subset(["_a", "b"], ["_a", "b", "unused"], tmp_path) == [
        "_a",
        "b",
    ]

    with pytest.raises(ManifestError, match="undeclared"):
        require_video_subset(["other"], ["_a", "b"], tmp_path)
    with pytest.raises(ManifestError, match="duplicate"):
        require_video_subset(["_a", "_a"], ["_a", "b"], tmp_path)
