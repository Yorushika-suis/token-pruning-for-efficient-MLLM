"""Install CDPruner on LLaVA-Video without editing ThirdParty.

CDPRUNER=1:
- After grid newlines are inserted, DPP-select spatial tokens and keep every newline.
- For one_token / no_token, select on the pooled sequence inside get_2dPool.
- Skip the FastVID block in Qwen2Model.forward (it assumes an unpruned 14-token grid).
- Read the current question from llava_vid.generate_until contexts.
"""
from __future__ import annotations

import contextvars
import inspect
import math
import os
import textwrap
from pathlib import Path

import torch

from dpp import select_indices

_QUESTION: contextvars.ContextVar[str] = contextvars.ContextVar("cdpruner_question", default="")
_LLAVA_PATCHED = False
_VID_PATCHED = False


def set_question(text: str) -> None:
    _QUESTION.set(text or "")


def current_question() -> str:
    return _QUESTION.get()


def _enabled() -> bool:
    return os.environ.get("CDPRUNER") == "1"


def _keep_count(n_spatial: int) -> int:
    explicit = os.environ.get("CDPRUNER_VISUAL_TOKEN_NUM", "").strip()
    if explicit:
        keep = int(explicit)
    else:
        ratio = float(os.environ.get("CDPRUNER_RETENTION_RATIO", "0.10"))
        keep = int(n_spatial * ratio)
    return max(1, min(n_spatial, keep))


def _siglip_snapshot() -> Path:
    root = Path(__file__).resolve().parents[2]
    hf = Path(os.environ.get("HF_HOME", root / "weights"))
    snaps = hf / "hub" / "models--google--siglip-so400m-patch14-384" / "snapshots"
    for cand in sorted(snaps.iterdir()) if snaps.exists() else []:
        if (cand / "config.json").exists():
            return cand
    raise FileNotFoundError(f"SigLIP snapshot missing under {snaps}")


class _SiglipText:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def _load(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.model is not None:
            return
        from transformers import SiglipModel, SiglipTokenizer

        path = str(_siglip_snapshot())
        self.tokenizer = SiglipTokenizer.from_pretrained(path, local_files_only=True)
        self.model = SiglipModel.from_pretrained(path, local_files_only=True)
        self.model.eval()
        self.model.to(device=device, dtype=dtype)
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def embed(self, text: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self._load(device, dtype)
        max_len = int(self.model.config.text_config.max_position_embeddings)
        # Keep the instruction, not only the first 64 tokens.
        raw = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
        ids = raw["input_ids"][0]
        if ids.numel() == 0:
            ids = self.tokenizer("image", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        chunks = []
        step = max(1, max_len - 2)
        for start in range(0, ids.numel(), step):
            piece = ids[start : start + step]
            enc = self.tokenizer.prepare_for_model(
                piece.tolist(),
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(device) for k, v in enc.items() if torch.is_tensor(v)}
            chunks.append(self.model.get_text_features(**enc))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def project_image(self, hidden: torch.Tensor) -> torch.Tensor:
        self._load(hidden.device, hidden.dtype)
        proj = self.model.visual_projection
        if hidden.shape[-1] != proj.in_features:
            return hidden
        flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=proj.weight.dtype)
        out = proj(flat)
        return out.reshape(*hidden.shape[:-1], out.shape[-1])


_TEXT = _SiglipText()


def _text_embeds(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    text = current_question().strip() or "video"
    return _TEXT.embed(text, device, dtype)


def _mark_skip_fastvid(module) -> None:
    inner = module.get_model() if hasattr(module, "get_model") else module
    inner.cdpruner_skip_fastvid = True


def _pool_raw(module, raw: torch.Tensor, n_frames: int) -> torch.Tensor | None:
    used = int(getattr(module, "_cdpruner_raw_used", 0))
    if raw is None or raw.shape[0] < used + n_frames:
        return None
    chunk = raw[used : used + n_frames]
    module._cdpruner_raw_used = used + n_frames
    orig = getattr(module.get_2dPool, "_cdpruner_orig", module.get_2dPool)
    return orig(module, chunk)


def _chosen_indices(spatial: torch.Tensor, image_embeds: torch.Tensor | None) -> torch.Tensor:
    n = spatial.shape[0]
    if image_embeds is None or image_embeds.shape[0] != n:
        embeds = spatial
    else:
        embeds = _TEXT.project_image(image_embeds)
    text = _text_embeds(spatial.device, spatial.dtype)
    return select_indices(spatial, embeds, text, _keep_count(n))


_INSTALLING = False


def install() -> None:
    """Patch LLaVA-Video pieces as each module becomes importable.

    llava_arch is imported before lmms_eval.models.llava_vid, so one flag would
    skip the question-text hook.
    """
    global _LLAVA_PATCHED, _VID_PATCHED, _INSTALLING
    if _INSTALLING or not _enabled():
        return
    if _LLAVA_PATCHED and _VID_PATCHED:
        return
    _INSTALLING = True
    try:
        if not _LLAVA_PATCHED:
            import llava.model.llava_arch as arch
            from llava.model.language_model.modeling_qwen2 import Qwen2Model

            _patch_encode(arch.LlavaMetaForCausalLM)
            _patch_grid(arch.LlavaMetaForCausalLM)
            _patch_pool(arch.LlavaMetaForCausalLM)
            _patch_fastvid(Qwen2Model)
            _LLAVA_PATCHED = True
        if not _VID_PATCHED:
            _VID_PATCHED = _patch_generate_until()
    except ImportError:
        return
    finally:
        _INSTALLING = False


def _patch_encode(cls) -> None:
    if getattr(cls.encode_images, "_cdpruner", False):
        return
    orig = cls.encode_images

    def encode_images(self, images):
        tower = self.get_model().get_vision_tower()
        raw_box: dict = {}
        orig_fwd = tower.forward

        def forward(images_):
            out = orig_fwd(images_)
            raw_box["raw"] = out.detach()
            return out

        tower.forward = forward
        try:
            result = orig(self, images)
        finally:
            tower.forward = orig_fwd
        self._cdpruner_raw = raw_box.get("raw")
        self._cdpruner_raw_used = 0
        _mark_skip_fastvid(self)
        return result

    encode_images._cdpruner = True
    cls.encode_images = encode_images


def _patch_grid(cls) -> None:
    if getattr(cls.add_token_per_grid, "_cdpruner", False):
        return
    orig = cls.add_token_per_grid

    def add_token_per_grid(self, image_feature):
        out = orig(self, image_feature)
        if not _enabled() or image_feature.ndim != 3:
            return out
        if getattr(self.config, "mm_newline_position", "one_token") != "grid":
            return out
        if getattr(self.config, "add_faster_video", False):
            return out
        side = int(math.sqrt(image_feature.shape[1]))
        if side * side != image_feature.shape[1]:
            return out
        period = side + 1
        if out.shape[0] % period != 0:
            return out
        idx = torch.arange(out.shape[0], device=out.device)
        is_nl = (idx + 1) % period == 0
        spatial = out[~is_nl]
        raw = getattr(self, "_cdpruner_raw", None)
        pooled = _pool_raw(self, raw, image_feature.shape[0]) if raw is not None else None
        image_embeds = None if pooled is None else pooled.reshape(-1, pooled.shape[-1])
        if image_embeds is not None and image_embeds.shape[0] != spatial.shape[0]:
            image_embeds = None
        chosen = _chosen_indices(spatial, image_embeds)
        spatial_pos = idx[~is_nl]
        mask = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
        mask[spatial_pos[chosen]] = True
        mask[is_nl] = True
        return out[idx[mask]]

    add_token_per_grid._cdpruner = True
    cls.add_token_per_grid = add_token_per_grid


def _patch_pool(cls) -> None:
    if getattr(cls.get_2dPool, "_cdpruner", False):
        return
    orig = cls.get_2dPool
    orig._cdpruner_orig = orig

    def get_2dPool(self, image_feature, stride=2):
        pooled = orig(self, image_feature, stride)
        pos = getattr(self.config, "mm_newline_position", "one_token")
        if not _enabled() or pos in ("grid", "frame") or image_feature.ndim != 3:
            return pooled
        flat = pooled.reshape(-1, pooled.shape[-1])
        raw = getattr(self, "_cdpruner_raw", None)
        image_embeds = None
        if raw is not None:
            pooled_raw = _pool_raw(self, raw, pooled.shape[0] if pos != "one_token" else image_feature.shape[0])
            if pooled_raw is not None:
                image_embeds = pooled_raw.reshape(-1, pooled_raw.shape[-1])
                if image_embeds.shape[0] != flat.shape[0]:
                    image_embeds = None
        chosen = _chosen_indices(flat, image_embeds)
        kept = flat[chosen]
        return kept.view(1, kept.shape[0], kept.shape[-1])

    get_2dPool._cdpruner = True
    get_2dPool._cdpruner_orig = orig
    cls.get_2dPool = get_2dPool


def _patch_fastvid(qwen_cls) -> None:
    if getattr(qwen_cls.forward, "_cdpruner_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(qwen_cls.forward))
    old = "if seq_length > 1: ### FastVID"
    new = "if seq_length > 1 and not getattr(self, 'cdpruner_skip_fastvid', False): ### FastVID"
    if old not in src:
        raise RuntimeError(f"FastVID guard not found in {qwen_cls.__module__}.forward")
    src = src.replace(old, new, 1)
    ns: dict = {}
    exec(src, qwen_cls.forward.__globals__, ns)  # noqa: S102
    fn = ns["forward"]
    fn._cdpruner_patched = True
    qwen_cls.forward = fn


def _patch_generate_until() -> bool:
    try:
        from lmms_eval.models.llava_vid import LlavaVid
    except ImportError:
        return False
    if getattr(LlavaVid.generate_until, "_cdpruner", False):
        return True
    src = textwrap.dedent(inspect.getsource(LlavaVid.generate_until))
    needle = "qs = contexts\n"
    if src.count(needle) != 1:
        raise RuntimeError("expected exactly one `qs = contexts` in generate_until")
    # inspect.getsource keeps the class indent; dedent removes those 4 spaces,
    # so the for-loop body that was 12 spaces is 8 spaces here.
    insert = (
        "qs = contexts\n"
        "        from hook_videomme import set_question\n"
        "        set_question(contexts)\n"
        "        _inner = getattr(self.model, 'model', None)\n"
        "        if _inner is not None:\n"
        "            _inner.cdpruner_skip_fastvid = True\n"
    )
    src = src.replace(needle, insert, 1)
    ns: dict = {}
    exec(src, LlavaVid.generate_until.__globals__, ns)  # noqa: S102
    fn = ns["generate_until"]
    fn._cdpruner = True
    LlavaVid.generate_until = fn
    return True
