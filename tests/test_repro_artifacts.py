from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "repro"))

from artifacts import FORBIDDEN_SCIENCE_KEYS, canonical, check_science, write_npz  # noqa: E402
from context import current_key, set_sample  # noqa: E402


def test_canonical_sorts_and_rejects_nan():
    assert canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})


def test_forbidden_science_keys():
    with pytest.raises(ValueError):
        check_science({"run_id": "x"})
    assert "gpu_uuid" in FORBIDDEN_SCIENCE_KEYS


def test_write_npz_roundtrip(tmp_path):
    path = tmp_path / "s.npz"
    write_npz(
        path,
        {
            "keep_indices": np.asarray([0, 2, 5], dtype=np.int32),
            "n_in": np.int32(10),
            "n_out": np.int32(3),
            "retention": np.float32(0.3),
        },
        {"coverage": [["vid", "q1"]], "method": "fastvid"},
    )
    with np.load(path, allow_pickle=False) as z:
        assert "meta_json" in z.files
        meta = json.loads(z["meta_json"].tobytes())
        assert meta["coverage"] == [["vid", "q1"]]
        assert z["keep_indices"].tolist() == [0, 2, 5]


def test_sample_key_uses_question_id():
    set_sample(video_id="abc", question_id="001-1")
    assert current_key() == ["abc", "001-1"]


def test_probe_restore_fastvid():
    os.environ["REPRO_PROBE_DIR"] = str(Path("/tmp/repro-restore-test"))
    import session as sessmod

    sessmod.SESSION = sessmod.ProbeSession("fastvid")
    import probe_fastvid

    calls = {}

    class Dummy:
        def forward(self, x):
            return x + 1

    mod = types.SimpleNamespace(Qwen2Model=Dummy)
    original = Dummy.forward
    probe_fastvid.install(mod)
    assert Dummy.forward is not original
    sessmod.SESSION.restore()
    assert Dummy.forward is original
    calls["ok"] = True
    assert calls["ok"]
