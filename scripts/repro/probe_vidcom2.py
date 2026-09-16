"""Observe official vidcom2_compression without replacing prepare_inputs."""
from __future__ import annotations

import functools

import numpy as np
import torch

from context import current_meta
from session import session

CONFIG_HASH = "vidcom2-llava-vid-v4-lite-1"


def _as_int32(x) -> np.ndarray:
    if torch.is_tensor(x):
        return np.ascontiguousarray(x.detach().to("cpu").numpy().astype(np.int32).reshape(-1))
    return np.ascontiguousarray(np.asarray(x, dtype=np.int32).reshape(-1))


def install(mod) -> None:
    sess = session()
    buf = {
        "scales": None,
        "indices": None,
        "ks": None,
        "global_idx": None,
        "vid_score": None,
        "frame_score": None,
        "tpf": None,
        "model": None,
    }

    orig_scales = mod.compute_scales
    orig_select = mod.select_outlier_indices
    orig_map = mod.map_features
    orig_comp = mod.vidcom2_compression

    if getattr(orig_comp, "_repro_probe", False):
        return

    @functools.wraps(orig_scales)
    def scales_wrapped(scores, base, temp=0.01):
        out = orig_scales(scores, base, temp)
        buf["scales"] = out.detach().clone()
        return out

    @functools.wraps(orig_select)
    def select_wrapped(scores, scales, tpf):
        out = orig_select(scores, scales, tpf)
        buf["indices"] = [idx.detach().clone() for idx in out]
        buf["ks"] = [int(idx.numel()) for idx in out]
        buf["tpf"] = int(tpf)
        return out

    @functools.wraps(orig_map)
    def map_wrapped(indices, flat, img, spec):
        if spec.get("mapper") == "linear":
            stride = flat.shape[0] // len(indices)
            buf["global_idx"] = mod._map_linear_offset(indices, stride).detach().clone()
        elif spec.get("mapper") == "grid_vid":
            buf["global_idx"] = mod._map_grid_vid(indices, spec["grid"]).detach().clone()
        return orig_map(indices, flat, img, spec)

    @functools.wraps(orig_comp)
    def comp_wrapped(flattened_feat, model="llava_ov", base_scale=0.25, frame_token_len=None, img_feat=None):
        buf["model"] = model
        try:
            result = orig_comp(
                flattened_feat,
                model=model,
                base_scale=base_scale,
                frame_token_len=frame_token_len,
                img_feat=img_feat,
            )
        except Exception as exc:
            session().write_failure(exc)
            raise
        try:
            spec = mod.MODEL_SPECS[model]
            tpf = buf["tpf"] or spec.get("tpf") or 169
            frames = len(buf["indices"] or [])
            n_in = int(frames * tpf)
            local = buf["indices"] or []
            n_out = int(sum(int(x.numel()) for x in local))
            keep_mask = np.zeros((frames, tpf), dtype=np.bool_)
            for i, idx in enumerate(local):
                keep_mask[i, idx.detach().to("cpu").numpy().astype(np.int64)] = True
            gidx = buf["global_idx"]
            arrays = {
                "keep_indices": _as_int32(gidx if gidx is not None else torch.zeros(0, dtype=torch.int64)),
                "n_in": np.int32(n_in),
                "n_out": np.int32(n_out),
                "retention": np.float32(n_out / n_in if n_in else 0.0),
                "scales": np.ascontiguousarray(
                    (buf["scales"].detach().to("cpu").numpy() if buf["scales"] is not None else np.zeros((0,), dtype=np.float32)).astype(
                        np.float32
                    )
                ),
                "ks": _as_int32(buf["ks"] or []),
                "keep_mask": keep_mask,
            }
            if sess.features and buf["vid_score"] is not None:
                arrays["vid_score"] = buf["vid_score"]
            meta = {
                "grid_h": int(spec.get("grid", 0) or (14 if tpf == 196 else 13)),
                "grid_w": int(spec.get("grid", 0) or (14 if tpf == 196 else 13)),
                "frames": frames,
                "newline_policy": "grid_vid_appends_row_newlines" if spec.get("mapper") == "grid_vid" else "none",
                "model": model,
                "notes": "code uses L2-normalized multi-scale Gaussian uniqueness, not paper cosine",
                **current_meta(),
            }
            session().write_mask(arrays, meta, CONFIG_HASH)
        except Exception as exc:  # noqa: BLE001
            session().write_failure(exc)
            raise
        return result

    orig_gauss = mod.compute_gaussian_scores

    @functools.wraps(orig_gauss)
    def gauss_wrapped(x, tpf):
        vid, frame = orig_gauss(x, tpf)
        if sess.features:
            buf["vid_score"] = vid.detach().clone().to("cpu").numpy().astype(np.float32)
            buf["frame_score"] = frame.detach().clone().to("cpu").numpy().astype(np.float32)
        return vid, frame

    for name, wrapped, original in (
        ("compute_scales", scales_wrapped, orig_scales),
        ("select_outlier_indices", select_wrapped, orig_select),
        ("map_features", map_wrapped, orig_map),
        ("compute_gaussian_scores", gauss_wrapped, orig_gauss),
        ("vidcom2_compression", comp_wrapped, orig_comp),
    ):
        setattr(mod, name, wrapped)
        wrapped._repro_probe = True

        def restore(n=name, o=original):
            setattr(mod, n, o)

        sess.remember(name, original, restore)
