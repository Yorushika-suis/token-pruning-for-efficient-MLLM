"""Observe CLIPVisionTower_VisionZip.forward after visionzip() has patched it."""
from __future__ import annotations

import sys
import types

import numpy as np
import torch

from context import current_meta
from session import session

CONFIG_HASH = "visionzip-llava15-v4-lite-1"


def _as_int32(x) -> np.ndarray:
    if torch.is_tensor(x):
        return np.ascontiguousarray(x.detach().to("cpu").numpy().astype(np.int32).reshape(-1))
    return np.ascontiguousarray(np.asarray(x, dtype=np.int32).reshape(-1))


def wrap_forward(original):
    def wrapped(self, images):
        captured = {"target_indices": None, "all_indices": None}
        target_path = original.__code__.co_filename

        def tracer(frame, event, arg):
            if event != "line":
                return tracer
            if frame.f_code.co_filename != target_path:
                return None
            loc = frame.f_locals
            tgt = loc.get("target_indices")
            if torch.is_tensor(tgt) and tgt.ndim == 1:
                captured["target_indices"] = tgt.detach().clone()
            return tracer

        prev = sys.gettrace()
        sys.settrace(tracer)
        try:
            out = original(self, images)
        except Exception as exc:
            sys.settrace(prev)
            session().write_failure(exc)
            raise
        sys.settrace(prev)
        try:
            hidden, all_indices = out
            n_out = int(hidden.shape[1])
            n_in = 577
            arrays = {
                "keep_indices": _as_int32(all_indices),
                "n_in": np.int32(n_in),
                "n_out": np.int32(n_out),
                "retention": np.float32(n_out / n_in),
                "topk_indices": _as_int32(all_indices),
                "target_indices": _as_int32(
                    captured["target_indices"] if captured["target_indices"] is not None else torch.zeros(0, dtype=torch.int64)
                ),
            }
            info = getattr(getattr(self, "vision_tower", self), "_info", {})
            meta = {
                "grid_h": 24,
                "grid_w": 24,
                "frames": 1,
                "newline_policy": "none",
                "api_dominant": int(info.get("dominant", 0)) + 1 if info else None,
                "internal_dominant": info.get("dominant"),
                "contextual": info.get("contextual"),
                **current_meta(),
            }
            session().write_mask(arrays, meta, CONFIG_HASH)
        except Exception as exc:  # noqa: BLE001
            session().write_failure(exc)
            raise
        return out

    wrapped._repro_probe = True
    return wrapped


def install(clip_mod) -> None:
    sess = session()
    cls = getattr(clip_mod, "CLIPVisionTower", None)
    if cls is None:
        return
    original = cls.forward
    if getattr(original, "_repro_probe", False):
        return
    wrapped = wrap_forward(original)
    cls.forward = wrapped

    def restore():
        cls.forward = original

    sess.remember("CLIPVisionTower.forward", original, restore)
