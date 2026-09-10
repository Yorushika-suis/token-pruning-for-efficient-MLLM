"""Shared harness for the FlashVID ADTS spatial-position study.

Two responsibilities:

1. `build_model()` — load LLaVA-Video-7B-Qwen2 the way `scripts/llava_vid.sh` does, but with
   the language model pinned to CPU. H1 only needs vision_tower -> mm_projector ->
   get_2dPool -> flashvid_compression, none of which touch LLM weights. Keeping the LLM off
   the GPU frees ~15 GB of HBM per process and, together with `install_llm_tripwire()`,
   makes "the LLM cannot influence these results" a machine-checked fact rather than a
   claim in the write-up.

   The checkpoint's own config says mm_spatial_pool_mode=bilinear / mm_newline_position=grid,
   which yields a 14x14 grid. The eval script overrides both. Loading without the override
   silently produces a different token grid and therefore a different coordinate system, so
   the override is asserted, not assumed.

2. `AdtsRecorder` — records, for every ADTS invocation, which token indices were selected.
   Two hooks are needed and neither can be a torch forward hook, because the selection
   functions are plain module-level functions held in a dict, not nn.Modules:

     a) `flashvid.utils.ALL_TOKEN_SELECTION_METHOD[...]` is looked up at call time, so
        rebinding the dict entry works.
     b) `flashvid.llava_arch.flashvid_compression` must be patched on `llava_arch`, not on
        `utils`: llava_arch does `from .utils import flashvid_compression`, so the function
        object is already bound into that module's globals.

   The recorder deliberately keeps duplicate indices. ADTS v2 lacks the anti-reselection
   guard that v1 has (token_selection.py:70), and its second calibration term is an
   unnormalised dot product that can go negative, so the greedy loop can re-pick an
   already-selected index. `mask.scatter_` is idempotent and would hide that; the raw
   index list is the only place the duplication is visible.
"""
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch

# scripts/llava_vid.sh, verbatim.
EVAL_OVERWRITE_CONFIG = {
    "mm_spatial_pool_mode": "average",
    "mm_newline_position": "frame",
}
EVAL_MAX_FRAMES_NUM = 64
EVAL_CONV_TEMPLATE = "qwen_1_5"
EVAL_FORCE_SAMPLE = True

# flashvid() kwargs from scripts/llava_vid.sh. retention_ratio is swept separately.
FLASHVID_KWARGS = dict(
    do_segment=True,
    segment_threshold=0.9,
    min_segment_num=8,
    complementary_segment=True,
    alpha=0.70,
    token_selection_method="attn_div_v2",
    temporal_threshold=0.8,
    expansion=1.25,
    pruning_layer=20,
    llm_retention_ratio=0.3,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = "lmms-lab/LLaVA-Video-7B-Qwen2"
MODEL_REVISION = "013210b3aff822f1558b166d39c1046dd109520f"
VISION_TOWER_PATH = "google/siglip-so400m-patch14-384"
VISION_TOWER_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
EXPECTED_PATCHES_PER_SIDE = 27
EXPECTED_NV = 169  # floor(27 / 2) ** 2 under average pooling
GRID = 13


def _cached_snapshot(repo_id, revision, override_env):
    """Resolve one exact local HF snapshot; never silently fall back to a moving ref."""
    override = os.environ.get(override_env)
    if override:
        path = Path(override).resolve()
    else:
        repo_dir = "models--" + repo_id.replace("/", "--")
        path = (PROJECT_ROOT / "weights" / "hub" / repo_dir / "snapshots" / revision).resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"missing exact cached snapshot for {repo_id}@{revision}: {path}; "
            f"set {override_env} only to an immutable local snapshot directory"
        )
    if not override:
        ref_path = path.parents[1] / "refs" / "main"
        if not ref_path.is_file() or ref_path.read_text(encoding="utf-8").strip() != revision:
            raise RuntimeError(
                f"cached main ref for {repo_id} is not pinned to {revision}: {ref_path}"
            )
    return path


def resolve_cached_model_sources():
    """Return exact local main-model and vision-tower snapshot directories."""
    return (
        _cached_snapshot(MODEL_PATH, MODEL_REVISION, "FLASHVID_MODEL_SNAPSHOT"),
        _cached_snapshot(
            VISION_TOWER_PATH,
            VISION_TOWER_REVISION,
            "FLASHVID_VISION_TOWER_SNAPSHOT",
        ),
    )


class LlmInvoked(RuntimeError):
    """Raised if the language model runs during a vision-only stage."""


def install_llm_tripwire(model):
    """Make any language-model forward an immediate, loud failure.

    Guarantees that the LLM's device, dtype and attn_implementation cannot silently
    influence Stage 1 results.
    """
    lm = model.get_model()
    handles = []

    def _trip(mod, args, kwargs=None):
        raise LlmInvoked(
            "language model was invoked during a vision-only stage; "
            "H1 must not depend on LLM weights"
        )

    for layer in lm.layers:
        handles.append(layer.register_forward_pre_hook(_trip, with_kwargs=True))
    return handles


def build_model(device="cuda:0", dtype=torch.float16, attn_implementation="sdpa",
                retention_ratio=0.10, tripwire=True, verbose=True):
    """Load the model, place vision on GPU / LLM on CPU, and apply FlashVID."""
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_source, vision_source = resolve_cached_model_sources()
    model_name = get_model_name_from_path(MODEL_PATH)
    overwrite_config = dict(EVAL_OVERWRITE_CONFIG)
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        str(model_source),
        None,
        model_name,
        device_map={"": "cpu"},
        attn_implementation=attn_implementation,
        overwrite_config=overwrite_config,
        multimodal=True,
    )
    model.eval()

    # The pooling mode decides the entire coordinate system. Never infer it.
    assert model.config.mm_spatial_pool_mode == "average", (
        f"mm_spatial_pool_mode is {model.config.mm_spatial_pool_mode!r}; the checkpoint "
        "default (bilinear) yields a 14x14 grid and silently invalidates every coordinate"
    )
    assert model.config.mm_newline_position == "frame", model.config.mm_newline_position
    # Keep the repository-style SigLIP name here.  The upstream vision builder checks
    # ``os.path.exists`` before checking ``"siglip"`` and would misroute an absolute
    # local snapshot to CLIPVisionTower.  Offline mode plus the validated refs/main
    # above resolves this ID to the exact cached revision without network access.
    assert model.config.mm_vision_tower == VISION_TOWER_PATH, (
        model.config.mm_vision_tower,
        VISION_TOWER_PATH,
    )

    vt = model.get_vision_tower()
    if hasattr(vt, "is_loaded") and not vt.is_loaded:
        vt.load_model()
    assert vt.num_patches_per_side == EXPECTED_PATCHES_PER_SIDE, vt.num_patches_per_side

    vt.to(device=device, dtype=dtype)
    model.get_model().mm_projector.to(device=device, dtype=dtype)

    from flashvid import flashvid

    model = flashvid(model, retention_ratio=retention_ratio, **FLASHVID_KWARGS)

    # flashvid() sets is_last_layer on the final SigLIP block; without it the tower returns
    # attentions[-1] = None and the [CLS] calibration term silently becomes garbage.
    last = vt.vision_tower.vision_model.encoder.layers[-1].self_attn
    assert getattr(last, "is_last_layer", False), "is_last_layer was not set by flashvid()"

    if tripwire:
        install_llm_tripwire(model)

    if verbose:
        print(f"[build_model] vision on {device} {dtype}, LLM on cpu, "
              f"pool={model.config.mm_spatial_pool_mode}, "
              f"newline={model.config.mm_newline_position}, "
              f"patches_per_side={vt.num_patches_per_side}, "
              f"retention_ratio={retention_ratio}, "
              f"model_revision={MODEL_REVISION}, "
              f"vision_revision={VISION_TOWER_REVISION}", flush=True)
    return tokenizer, model, image_processor, max_length


def encode_to_pooled(model, pixel_values):
    """frames -> (pooled_features, pooled_cls_attention) exactly as llava_arch.py:77-78."""
    image_features, cls_attentions = model.encode_images(pixel_values)
    pooled = model.get_2dPool(image_features)
    pooled_attn = model.get_2dPool(cls_attentions.unsqueeze(-1)).squeeze(-1)
    return pooled, pooled_attn


class AdtsRecorder:
    """Capture ADTS selections (and optionally ban positions) via two monkeypatches."""

    def __init__(self, banned_positions=None):
        self.records = []
        self.calls = []
        self.banned = None
        if banned_positions is not None and len(banned_positions):
            self.banned = torch.as_tensor(sorted(set(int(p) for p in banned_positions)),
                                          dtype=torch.long)
        self._orig_sel = None
        self._orig_comp = None

    # --- selection-level hook -------------------------------------------------
    def _wrap_selection(self, orig, is_v2):
        import flashvid.token_selection as ts

        def wrapper(features, num_retained_tokens, cls_attention=None, **kw):
            ties = margins = None
            if is_v2:
                # Instrumented reimplementation of attn_div_v2. Used for every v2 call so
                # that argmax tie multiplicity can be recorded; verified against the
                # shipped function by assert_v2_equivalence() before collection starts.
                feats, keep, ties, margins = self._v2_selection(
                    ts, features, cls_attention, num_retained_tokens, self.banned
                )
            elif cls_attention is None:
                feats, keep = orig(features=features,
                                   num_retained_tokens=num_retained_tokens, **kw)
            else:
                feats, keep = orig(features=features, cls_attention=cls_attention,
                                   num_retained_tokens=num_retained_tokens, **kw)
            k = keep.shape[1]
            distinct = np.array([torch.unique(keep[b]).numel() for b in range(keep.shape[0])],
                                dtype=np.int32)
            self.calls.append({
                "n_frames": int(keep.shape[0]),
                "k": int(k),
                "n_visual_tokens": int(features.shape[1]),
                "keep_frame_local": keep.detach().to("cpu", torch.int16).numpy(),
                "distinct_per_frame": distinct,
                "n_duplicate_slots": int((k - distinct).sum()),
                "ties": None if ties is None else ties.to("cpu", torch.int16).numpy(),
                "margins": None if margins is None else margins.to("cpu", torch.float32).numpy(),
            })
            return feats, keep

        return wrapper

    def _v2_selection(self, ts, features, cls_attention, num_retained_tokens, banned):
        """attn_div_v2 (token_selection.py:79-130), instrumented, optionally masked.

        Two additions over the shipped function, neither of which changes the result:
          * per-step argmax tie multiplicity and top1-top2 margin are recorded. torch.argmax
            returns the SMALLEST index on a tie, and in row-major order the smallest index is
            the top-left of the frame -- so ties produce a top-left selection bias that looks
            exactly like the spatial prior this study is trying to detect, moves with the
            index rather than with the content, and is therefore invisible to the hflip and
            shift control arms. It has to be measured directly or it cannot be ruled out.
          * an optional ban mask, applied AFTER the calibration terms are multiplied in
            (:107). Applying it to the raw cosine distances instead would be multiplied by
            calibration_term2, an unnormalised dot product that can be negative -- turning a
            large negative into a large positive, i.e. "never pick this" into "pick this
            first", silently. A large finite negative avoids NaN from later multiplies.
        """
        original_features = features
        feats = features.float()
        pooled_features = feats.mean(1)
        global_cls = cls_attention.float() * 1e6
        bsz, n_tok, dim = feats.shape
        dist = ts.pairwise_cosine_distances(feats)
        term1 = global_cls.unsqueeze(1)
        local_cls = torch.einsum("b n d, c d -> b c n", feats, pooled_features).mean(1)
        term2 = local_cls.unsqueeze(1)
        dist = dist * term1 * term2

        NEG = -1e30
        if banned is not None:
            b = banned.to(dist.device)
            assert n_tok - b.numel() >= num_retained_tokens, (
                f"only {n_tok - b.numel()} candidates survive the ban but "
                f"{num_retained_tokens} must be selected")
            dist = dist.clone()
            dist[:, :, b] = NEG
        else:
            b = None

        keep = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=feats.device)
        ties = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=feats.device)
        margins = torch.zeros(bsz, num_retained_tokens, dtype=torch.float32, device=feats.device)

        def _pick(md, i):
            top = torch.argmax(md, dim=-1)
            best = md.gather(1, top.unsqueeze(1))
            ties[:, i] = (md == best).sum(-1)
            two = md.topk(2, dim=-1).values
            margins[:, i] = (two[:, 0] - two[:, 1]).float()
            return top

        min_dist = torch.topk(dist, k=2, dim=1, largest=False).values[:, 1, :]
        keep[:, 0] = _pick(min_dist, 0)
        for i in range(1, num_retained_tokens):
            sub = torch.gather(dist, dim=1,
                               index=keep[:, :i].unsqueeze(-1).expand(-1, -1, n_tok))
            md = torch.min(sub, dim=1).values
            if b is not None:
                md[:, b] = NEG
            keep[:, i] = _pick(md, i)
        keep = keep.sort().values
        if b is not None:
            assert not torch.isin(keep, b).any(), "a banned position was selected"
        sel = torch.gather(original_features, dim=1,
                           index=keep.unsqueeze(-1).expand(-1, -1, dim))
        return sel, keep, ties, margins

    def _banned_selection(self, ts, features, cls_attention, num_retained_tokens):
        return self._v2_selection(ts, features, cls_attention, num_retained_tokens,
                                  self.banned)[:2]

    # --- compression-level hook ----------------------------------------------
    def _wrap_compression(self, orig):
        def wrapper(video_features, cls_attention, flashvid_config):
            start = len(self.calls)
            try:
                out = orig(video_features=video_features, cls_attention=cls_attention,
                           flashvid_config=flashvid_config)
                tokens, final_idx = out
                nf, nv = video_features.shape[0], video_features.shape[1]
                calls = self.calls[start:]
                # Recover per-segment frame counts from the per-call batch sizes.
                seg_lengths = [c["n_frames"] for c in calls]
                keep_global = []
                off = 0
                for c in calls:
                    kl = torch.as_tensor(c["keep_frame_local"], dtype=torch.long)
                    frames = torch.arange(off, off + c["n_frames"]).unsqueeze(1)
                    keep_global.append((frames * nv + kl).reshape(-1))
                    off += c["n_frames"]
                keep_global = (torch.cat(keep_global) if keep_global
                               else torch.zeros(0, dtype=torch.long))
                # Per-frame distinct count. The exact permutation null is "each frame draws k
                # of 169 uniformly", so k must be known per frame, not per video: if the v2
                # re-selection bug fires on some frames their effective k is smaller, and a
                # video-level k would bias every position's expected rate the same way.
                distinct_pf = (np.concatenate([c["distinct_per_frame"] for c in calls])
                               if calls else np.zeros(0, dtype=np.int32))
                have_ties = calls and calls[0]["ties"] is not None
                ties = (np.concatenate([c["ties"] for c in calls]) if have_ties else None)
                margins = (np.concatenate([c["margins"] for c in calls]) if have_ties else None)
                fin = final_idx.detach().cpu()
                self.records.append({
                    "num_frames": int(nf),
                    "num_visual_tokens": int(nv),
                    "num_attn_div_tokens": int(flashvid_config.num_attn_div_tokens),
                    "num_sttm_tokens": int(flashvid_config.num_sttm_tokens),
                    "retention_ratio": float(flashvid_config.retention_ratio),
                    "token_selection_method": str(flashvid_config.token_selection_method),
                    "segment_lengths": seg_lengths,
                    "n_segments": len(seg_lengths),
                    "n_single_frame_segments": sum(1 for s in seg_lengths if s == 1),
                    "adts_keep_global": keep_global.numpy(),
                    "distinct_per_frame": distinct_pf,
                    "ties": ties,
                    "margins": margins,
                    "n_duplicate_slots": int(sum(c["n_duplicate_slots"] for c in calls)),
                    "n_adts_slots": int(keep_global.numel()),
                    "n_adts_distinct": int(torch.unique(keep_global).numel()),
                    "final_global_indices": fin.numpy(),
                    "n_final": int(fin.numel()),
                    "n_final_distinct": int(torch.unique(fin).numel()),
                })
                return out
            finally:
                # Must run even on failure, or a raised exception leaves the per-segment
                # calls from a partial compression to be misattributed to the next one.
                self.calls = self.calls[:start]

        return wrapper

    @contextlib.contextmanager
    def active(self):
        import flashvid.llava_arch as fa
        import flashvid.utils as fu
        from flashvid.token_selection import TokenSelectionMethod

        # Wrap every selection method, not just ADTS v2: the attribution ablation switches
        # flashvid_config.token_selection_method to attn / div / attn_div, and if those are
        # left unwrapped the compression hook still fires but self.calls stays empty, so the
        # recorded mask silently comes out as a zero-length array instead of erroring.
        self._orig_sel = dict(fu.ALL_TOKEN_SELECTION_METHOD)
        self._orig_comp = fa.flashvid_compression
        v2key = TokenSelectionMethod.ADTS_v2
        for key, fn in self._orig_sel.items():
            fu.ALL_TOKEN_SELECTION_METHOD[key] = self._wrap_selection(fn, key == v2key)
        fa.flashvid_compression = self._wrap_compression(self._orig_comp)
        assert fu.ALL_TOKEN_SELECTION_METHOD["attn_div_v2"] is \
            fu.ALL_TOKEN_SELECTION_METHOD[v2key], "enum/str key mismatch in the method dict"
        try:
            yield self
        finally:
            fu.ALL_TOKEN_SELECTION_METHOD.clear()
            fu.ALL_TOKEN_SELECTION_METHOD.update(self._orig_sel)
            fa.flashvid_compression = self._orig_comp


def assert_v2_equivalence(pooled, pooled_attn, k, atol_frac=0.0):
    """Confirm the instrumented v2 reproduces the shipped attn_div_v2 bit-for-bit.

    AdtsRecorder substitutes its own copy of the v2 greedy loop so it can record argmax tie
    multiplicity. That copy is only trustworthy if it selects exactly what the released code
    selects, so check it on real pooled features before collecting anything.
    """
    import flashvid.token_selection as ts

    with torch.no_grad():
        _, keep_ref = ts.attn_div_v2_based_token_selection(
            features=pooled, cls_attention=pooled_attn, num_retained_tokens=k)
        rec = AdtsRecorder()
        _, keep_ins, ties, margins = rec._v2_selection(ts, pooled, pooled_attn, k, None)
    n_diff = int((keep_ref != keep_ins).sum())
    return {
        "identical": n_diff == 0,
        "n_differing_slots": n_diff,
        "total_slots": int(keep_ref.numel()),
        "frac_steps_with_ties": float((ties > 1).float().mean()),
        "max_tie_multiplicity": int(ties.max()),
        "median_margin": float(margins.median()),
    }


def load_video_frames(path, num_frames=EVAL_MAX_FRAMES_NUM):
    """Uniform frame sampling, identical to playground/llava_vid_infer.py:46-57."""
    import numpy as np
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, num_frames, dtype=int).tolist()
    return vr.get_batch(idx).asnumpy()


def index_to_rc(idx, grid=GRID):
    """Flat within-frame index -> (row, col). Row-major; see llava_arch.py:171-190."""
    return idx // grid, idx % grid


def rc_to_index(r, c, grid=GRID):
    return r * grid + c


def global_to_frc(g, nv=EXPECTED_NV, grid=GRID):
    """Global index -> (frame, row, col). See utils.py:43,60 and llava_arch.py:132-137."""
    frame, within = divmod(g, nv)
    return frame, within // grid, within % grid
