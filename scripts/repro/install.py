"""Install official-function observers when REPRO_PROBE=1."""
from __future__ import annotations

import builtins
import os
import sys


_REAL_IMPORT = builtins.__import__
_PATCHED = set()
_IN_PATCH = False
_RETRY = False


def _method() -> str:
    return os.environ.get("REPRO_PROBE_METHOD", "").strip().lower()


def _patch_generate_until(mod) -> None:
    from context import clear_sample, set_sample
    from session import session

    cls = getattr(mod, "LlavaVid", None)
    if cls is None:
        return
    original = cls.generate_until
    if getattr(original, "_repro_probe", False):
        return

    def wrapped(self, requests):
        out = []
        for req in requests:
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            set_sample(
                video_id=doc.get("videoID") or doc.get("video_id"),
                question_id=doc.get("question_id"),
            )
            try:
                out.extend(original(self, [req]))
            finally:
                clear_sample()
        return out

    wrapped._repro_probe = True
    cls.generate_until = wrapped
    session().remember("LlavaVid.generate_until", original, lambda: setattr(cls, "generate_until", original))


def _maybe_patch() -> None:
    global _IN_PATCH, _RETRY
    if _IN_PATCH:
        _RETRY = True
        return
    _IN_PATCH = True
    try:
        while True:
            _RETRY = False
            method = _method()
            if method in ("", "fastvid") and "llava.model.language_model.modeling_qwen2" in sys.modules:
                key = "fastvid.qwen2"
                mod = sys.modules["llava.model.language_model.modeling_qwen2"]
                if key not in _PATCHED and getattr(mod, "Qwen2Model", None) is not None:
                    import probe_fastvid as probe

                    probe.install(mod)
                    _PATCHED.add(key)
            if method in ("", "vidcom2", "fastvid") and "lmms_eval.models.llava_vid" in sys.modules:
                key = "llava_vid.generate"
                mod = sys.modules["lmms_eval.models.llava_vid"]
                if key not in _PATCHED and getattr(mod, "LlavaVid", None) is not None:
                    _patch_generate_until(mod)
                    _PATCHED.add(key)
            if method in ("", "vidcom2") and "token_compressor.vidcom2.vidcom2" in sys.modules:
                key = "vidcom2.mod"
                mod = sys.modules["token_compressor.vidcom2.vidcom2"]
                if key not in _PATCHED and getattr(mod, "compute_scales", None) is not None:
                    import probe_vidcom2 as probe

                    probe.install(mod)
                    _PATCHED.add(key)
            if method == "visionzip" and "llava.model.multimodal_encoder.clip_encoder" in sys.modules:
                key = "visionzip.clip"
                clip = sys.modules["llava.model.multimodal_encoder.clip_encoder"]
                fwd = getattr(getattr(clip, "CLIPVisionTower", None), "forward", None)
                qual = getattr(fwd, "__qualname__", "")
                if fwd is not None and "VisionZip" in qual and key not in _PATCHED:
                    import visionzip_probe as probe

                    probe.install(clip)
                    _PATCHED.add(key)
            if not _RETRY:
                break
    finally:
        _IN_PATCH = False


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _REAL_IMPORT(name, globals, locals, fromlist, level)
    try:
        _maybe_patch()
    except Exception:
        raise
    return mod


def install_session(method: str | None = None) -> None:
    import atexit

    import session as sessmod

    chosen = method or _method() or "unknown"
    if sessmod.SESSION is None:
        sessmod.SESSION = sessmod.ProbeSession(chosen)
        atexit.register(lambda: sessmod.SESSION.restore() if sessmod.SESSION else None)
    if builtins.__import__ is not _import:
        builtins.__import__ = _import
    _maybe_patch()
