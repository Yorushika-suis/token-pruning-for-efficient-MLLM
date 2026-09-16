"""Observe FastVID keep_indexs inside official Qwen2Model.forward."""
from __future__ import annotations

import sys
import types

import numpy as np
import torch

from context import current_meta
from session import session

CONFIG_HASH = "fastvid-llava-vid-v4-lite-1"


def _as_int32(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.int32)
    if torch.is_tensor(x):
        x = x.detach().to("cpu")
        return np.ascontiguousarray(x.numpy().astype(np.int32).reshape(-1))
    return np.ascontiguousarray(np.asarray(x, dtype=np.int32).reshape(-1))


def _observe_forward(original):
    def wrapped(self, *args, **kwargs):
        captured = {
            "segment_sizes": None,
            "keep_indexs": None,
            "salient": [],
            "context": [],
            "seen_salient": set(),
            "seen_context": set(),
        }
        target_file = getattr(original, "__code__", None)
        target_path = original.__code__.co_filename if target_file else ""

        def tracer(frame, event, arg):
            if event != "line":
                return tracer
            if frame.f_code.co_filename != target_path:
                return None
            loc = frame.f_locals
            sizes = loc.get("segment_sizes")
            if isinstance(sizes, list) and sizes:
                captured["segment_sizes"] = list(sizes)
            top = loc.get("frm_top_indexes")
            if torch.is_tensor(top) and id(top) not in captured["seen_salient"]:
                captured["seen_salient"].add(id(top))
                captured["salient"].append(top.detach().to("cpu"))
            ctx = loc.get("frm_context_indexes")
            if torch.is_tensor(ctx) and id(ctx) not in captured["seen_context"]:
                captured["seen_context"].add(id(ctx))
                captured["context"].append(ctx.detach().to("cpu"))
            keep = loc.get("keep_indexs")
            hs = loc.get("hidden_states")
            mask = loc.get("newline_mask")
            if torch.is_tensor(mask):
                captured["newline_n"] = int((~mask).sum().item())
            if (
                torch.is_tensor(keep)
                and keep.ndim == 1
                and torch.is_tensor(hs)
                and hs.shape[1] == keep.shape[0]
            ):
                captured["keep_indexs"] = keep.detach().clone()
                cpu = keep.detach().to("cpu")
                if torch.equal(cpu, cpu.sort().values):
                    sys.settrace(None)
            return tracer

        prev = sys.gettrace()
        sys.settrace(tracer)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            sys.settrace(prev)
            session().write_failure(exc)
            raise
        sys.settrace(prev)
        try:
            keep = captured["keep_indexs"]
            if keep is None:
                # Decode steps skip FastVID (seq_length == 1); only prefill prunes.
                return result
            n_in = 64 * 13 * 13
            start = int(getattr(self, "video_start_idx", 0))
            vlen = int(getattr(self, "video_token_len", 0))
            vis = keep[(keep >= start) & (keep < start + vlen)]
            # visual tokens including newline; pure visual is vis minus newline rows
            n_out_with_nl = int(vis.numel())
            newline_n = captured.get("newline_n") or 0
            n_out = max(n_out_with_nl - int(newline_n), 0)
            retention = np.float32(n_out / n_in if n_in else 0.0)
            arrays = {
                "keep_indices": _as_int32(keep.sort().values if keep.ndim == 1 else keep),
                "n_in": np.int32(n_in),
                "n_out": np.int32(n_out),
                "retention": retention,
                "segment_sizes": _as_int32(captured["segment_sizes"] or []),
                "frm_top_indexes": _as_int32(
                    torch.cat(captured["salient"]) if captured["salient"] else torch.zeros(0, dtype=torch.int64)
                ),
                "frm_context_indexes": _as_int32(
                    torch.cat(captured["context"]) if captured["context"] else torch.zeros(0, dtype=torch.int64)
                ),
                "newline_n": np.int32(newline_n),
            }
            meta = {
                "grid_h": 13,
                "grid_w": 13,
                "frames": int(getattr(self, "frame_num", 64) or 64),
                "newline_policy": "drop_before_prune_then_restore",
                "n_out_with_newline": n_out_with_nl,
                **current_meta(),
            }
            session().write_mask(arrays, meta, CONFIG_HASH)
        except Exception as exc:  # noqa: BLE001
            session().write_failure(exc)
            raise
        return result

    return wrapped


def install(mod) -> None:
    sess = session()
    cls = getattr(mod, "Qwen2Model", None)
    if cls is None:
        return
    original = cls.forward
    if getattr(original, "_repro_probe", False):
        return
    wrapped = _observe_forward(original)
    wrapped._repro_probe = True
    cls.forward = wrapped

    def restore():
        cls.forward = original

    sess.remember("Qwen2Model.forward", original, restore)
