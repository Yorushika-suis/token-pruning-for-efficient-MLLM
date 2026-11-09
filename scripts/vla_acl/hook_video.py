"""Prune LLaVA-Video spatial tokens with a trained per-frame scorer.

VLA_ACL_VIDEO=1. Newline tokens stay. FastVID's second prune is skipped.
Kept patches stay in their shortened order and get contiguous positions. This
SDPA build rejects the original grid positions.
"""
from __future__ import annotations

import inspect
import math
import os
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

from frames import load_cached_frames
from scorer import PatchScorer, hard_top_k_per_frame, patches_per_frame_for

_INSTALLED = False
_SCORER: PatchScorer | None = None
SPATIAL = 169


def _enabled() -> bool:
    return os.environ.get("VLA_ACL_VIDEO") == "1"


def _retention() -> float:
    return float(os.environ.get("VLA_ACL_RETENTION", "0.10"))


def _load_scorer(device: torch.device) -> PatchScorer:
    global _SCORER
    if _SCORER is not None:
        return _SCORER
    blob = torch.load(Path(os.environ["VLA_ACL_SCORER"]), map_location="cpu")
    expected = patches_per_frame_for(_retention())
    if int(blob["k"]) != expected:
        raise RuntimeError(f"scorer k={blob['k']} does not match retention {_retention()} (k={expected})")
    scorer = PatchScorer(
        vision_dim=int(blob["vision_dim"]),
        hidden_dim=int(blob["hidden_dim"]),
        num_layers=int(blob["num_layers"]),
        num_heads=int(blob["num_heads"]),
    )
    scorer.load_state_dict(blob["state_dict"])
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)
    _SCORER = scorer.to(device=device, dtype=torch.float32)
    return _SCORER


def pool_siglip(raw: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """(T, 729, C) -> (T, 169, C) with the same average pool as get_2dPool."""
    frames, tokens, dim = raw.shape
    side = int(math.sqrt(tokens))
    if side * side != tokens:
        raise RuntimeError(f"expected square SigLIP grid, got {tokens}")
    pooled = F.avg_pool2d(raw.view(frames, side, side, dim).permute(0, 3, 1, 2), stride)
    return pooled.permute(0, 2, 3, 1).reshape(frames, -1, dim)


def _skip_fastvid(module) -> None:
    language = getattr(module, "model", None)
    if language is not None:
        language.vla_acl_skip_fastvid = True
        inner = getattr(language, "model", None)
        if inner is not None:
            inner.vla_acl_skip_fastvid = True


def _patch_fastvid(qwen_cls) -> None:
    if getattr(qwen_cls.forward, "_vla_acl_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(qwen_cls.forward))
    old = "if seq_length > 1: ### FastVID"
    new = "if seq_length > 1 and not getattr(self, 'vla_acl_skip_fastvid', False): ### FastVID"
    if old not in src:
        raise RuntimeError("FastVID guard not found in Qwen2Model.forward")
    src = src.replace(old, new, 1)
    ns: dict = {}
    exec(src, qwen_cls.forward.__globals__, ns)  # noqa: S102
    fn = ns["forward"]
    fn._vla_acl_patched = True
    qwen_cls.forward = fn


_CACHE_INSTALLING = False


def install_frame_cache() -> None:
    """Replace LlavaVid.load_video with the decoded 64-frame cache."""
    global _CACHE_INSTALLING
    cache = os.environ.get("VLA_ACL_FRAME_CACHE")
    if _CACHE_INSTALLING or not cache or "lmms_eval.models.llava_vid" not in sys.modules:
        return
    _CACHE_INSTALLING = True
    try:
        from lmms_eval.models.llava_vid import LlavaVid
    finally:
        _CACHE_INSTALLING = False
    if getattr(LlavaVid.load_video, "_vla_acl_cache", False):
        return
    cache_dir = Path(cache)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        if getattr(self, "add_time_instruction", False):
            raise RuntimeError("frame cache does not store timestamps; leave add_time_instruction off")
        video_id = Path(video_path).stem
        frames = load_cached_frames(cache_dir, video_id, int(max_frames_num))
        return frames, "", 0.0

    load_video._vla_acl_cache = True
    LlavaVid.load_video = load_video


def install() -> None:
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return
    import llava.model.llava_arch as arch

    def _wrap_tower(model) -> None:
        tower = model.get_vision_tower()
        if getattr(tower.forward, "_vla_acl", False):
            return
        orig = tower.forward

        def forward(images):
            out = orig(images)
            tower._vla_acl_raw = out.detach()
            return out

        forward._vla_acl = True
        tower.forward = forward

    orig_encode = arch.LlavaMetaForCausalLM.encode_images

    def encode_images(self, images):
        _wrap_tower(self)
        _skip_fastvid(self)
        return orig_encode(self, images)

    orig_grid = arch.LlavaMetaForCausalLM.add_token_per_grid

    def add_token_per_grid(self, image_feature):
        out = orig_grid(self, image_feature)
        if not _enabled() or image_feature.ndim != 3:
            return out
        if getattr(self.config, "mm_newline_position", "one_token") != "grid":
            return out
        side = int(math.sqrt(image_feature.shape[1]))
        if side * side != image_feature.shape[1]:
            return out
        period = side + 1
        if out.shape[0] % period != 0:
            return out
        raw = getattr(self.get_vision_tower(), "_vla_acl_raw", None)
        if raw is None:
            return out
        pooled = pool_siglip(raw)
        if pooled.shape[0] * pooled.shape[1] != image_feature.shape[0] * image_feature.shape[1]:
            pooled = pooled[: image_feature.shape[0]]
        scorer = _load_scorer(pooled.device)
        with torch.no_grad():
            logits = scorer(pooled.float())
            chosen = hard_top_k_per_frame(logits, patches_per_frame_for(_retention()))
        idx = torch.arange(out.shape[0], device=out.device)
        is_nl = (idx + 1) % period == 0
        spatial_pos = idx[~is_nl]
        if chosen.numel() > spatial_pos.numel():
            return out
        mask = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
        mask[spatial_pos[chosen.to(spatial_pos.device)]] = True
        mask[is_nl] = True
        kept_idx = idx[mask]
        _skip_fastvid(self)
        return out[kept_idx]

    arch.LlavaMetaForCausalLM.encode_images = encode_images
    arch.LlavaMetaForCausalLM.add_token_per_grid = add_token_per_grid

    try:
        from llava.model.language_model.modeling_qwen2 import Qwen2Model
    except ImportError:
        Qwen2Model = None
    if Qwen2Model is not None:
        _patch_fastvid(Qwen2Model)
    install_frame_cache()
    _INSTALLED = True
