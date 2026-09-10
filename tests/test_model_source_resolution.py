import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adts_harness as harness  # noqa: E402


def test_default_sources_are_exact_cached_revisions(monkeypatch):
    monkeypatch.delenv("FLASHVID_MODEL_SNAPSHOT", raising=False)
    monkeypatch.delenv("FLASHVID_VISION_TOWER_SNAPSHOT", raising=False)
    model, vision = harness.resolve_cached_model_sources()
    assert model.name == harness.MODEL_REVISION
    assert vision.name == harness.VISION_TOWER_REVISION
    assert (model / "config.json").is_file()
    assert (vision / "config.json").is_file()


def test_missing_override_fails_instead_of_using_remote_ref(monkeypatch, tmp_path):
    missing = tmp_path / "missing-snapshot"
    monkeypatch.setenv("FLASHVID_MODEL_SNAPSHOT", str(missing))
    with pytest.raises(FileNotFoundError, match="immutable local snapshot"):
        harness.resolve_cached_model_sources()
