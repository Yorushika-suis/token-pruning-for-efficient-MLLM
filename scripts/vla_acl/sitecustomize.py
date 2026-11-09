"""Loaded when scripts/vla_acl is first on PYTHONPATH.

VLA_ACL=1 installs the image hook. VLA_ACL_VIDEO=1 installs the video hook
and the local Video-MME parquet redirect. VLA_ACL_EVAL=heldout drops train videos.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_REPRO = _ROOT / "scripts" / "repro"
if str(_REPRO) not in sys.path:
    sys.path.insert(0, str(_REPRO))


def _load_repro_sitecustomize() -> None:
    path = _REPRO / "sitecustomize.py"
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location("_repro_sitecustomize", path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _filter_heldout() -> None:
    if os.environ.get("VLA_ACL_EVAL") != "heldout":
        return
    split_path = Path(os.environ.get("VLA_ACL_SPLIT", _ROOT / "runs" / "vla-acl-video" / "split.json"))
    heldout = set(json.loads(split_path.read_text())["heldout_video_ids"])
    import datasets

    current = datasets.load_dataset

    def load_dataset(path=None, *args, **kwargs):
        loaded = current(path, *args, **kwargs)
        if path != "lmms-lab/Video-MME":
            return loaded
        from datasets import DatasetDict

        def keep(row):
            return row["videoID"] in heldout

        return DatasetDict({name: table.filter(keep) for name, table in loaded.items()})

    datasets.load_dataset = load_dataset


def _install_later() -> None:
    if os.environ.get("VLA_ACL_VIDEO") == "1":
        _load_repro_sitecustomize()
        _filter_heldout()
        import hook_video

        hook_video.install()
        return
    if os.environ.get("VLA_ACL") == "1":
        import hook

        hook.install()


_install_later()

_real_import = __import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    if os.environ.get("VLA_ACL_VIDEO") == "1" and name in (
        "llava.model.llava_arch",
        "llava.model.language_model.llava_qwen",
        "llava.model.language_model.modeling_qwen2",
    ):
        import hook_video

        hook_video.install()
    if os.environ.get("VLA_ACL_VIDEO") == "1" and name == "lmms_eval.models.llava_vid":
        import hook_video

        hook_video.install_frame_cache()
    if os.environ.get("VLA_ACL") == "1" and name in (
        "llava.model.llava_arch",
        "llava.model.language_model.llava_llama",
    ):
        import hook

        hook.install()
    return mod


import builtins

builtins.__import__ = _import
