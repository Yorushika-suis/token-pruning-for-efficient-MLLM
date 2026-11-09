"""Auto-install local Video-MME loader; probes when REPRO_PROBE=1.

FRAME_CACHE points at decode/output. When set, LlavaVid.load_video reads those
64 frames instead of decoding the mp4. FastVID keeps this directory on
PYTHONPATH and does not load the VLA-ACL pruner.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

try:
    import local_videomme

    local_videomme.patch()
except Exception:
    # Never break conda activate / unrelated python when PYTHONPATH includes this dir.
    pass

if os.environ.get("REPRO_PROBE") == "1":
    try:
        import install as _repro_install

        _repro_install.install_session()
    except Exception:
        pass


def _patch_hub_model_name() -> None:
    """HF snapshot dirs end in a SHA; LLaVA builders key off names like LLaVA-Video-7B-Qwen2."""
    try:
        import llava.mm_utils as mm_utils
    except Exception:
        return
    orig = mm_utils.get_model_name_from_path
    if getattr(orig, "_repro_hub_name", False):
        return

    def wrapped(model_path):
        path = str(model_path).replace("\\", "/").strip("/")
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part.startswith("models--") and i + 1 < len(parts) and parts[i + 1] == "snapshots":
                return part.split("--")[-1]
        return orig(model_path)

    wrapped._repro_hub_name = True
    mm_utils.get_model_name_from_path = wrapped


_patch_hub_model_name()


def _load_cached_frames():
    path = Path(__file__).resolve().parents[1] / "vla_acl" / "frames.py"
    spec = importlib.util.spec_from_file_location("_frame_cache_loader", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frame cache reader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_cached_frames


def _patch_llava_vid_cache(module, cache: str) -> None:
    cls = getattr(module, "LlavaVid", None)
    if cls is None or getattr(getattr(cls, "load_video", None), "_frame_cache", False):
        return
    load_cached_frames = _load_cached_frames()
    cache_dir = Path(cache)
    original = cls.load_video

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        if int(max_frames_num) == 0:
            return original(self, video_path, max_frames_num, fps, force_sample)
        if not force_sample:
            raise RuntimeError("frame cache matches force_sample=True only")
        if getattr(self, "add_time_instruction", False):
            raise RuntimeError("frame cache does not store timestamps; leave add_time_instruction off")
        frames = load_cached_frames(cache_dir, Path(video_path).stem, int(max_frames_num))
        return frames, "", 0.0

    load_video._frame_cache = True
    cls.load_video = load_video


def _install_frame_cache_import_hook() -> None:
    cache = os.environ.get("FRAME_CACHE")
    if not cache:
        return
    import builtins
    import sys

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        # import pkg.mod returns the top package when fromlist is empty.
        if name == "lmms_eval.models.llava_vid" or (
            name == "lmms_eval.models" and fromlist and "llava_vid" in fromlist
        ):
            target = sys.modules.get("lmms_eval.models.llava_vid")
            if target is not None:
                _patch_llava_vid_cache(target, cache)
        return module

    builtins.__import__ = _import


_install_frame_cache_import_hook()
