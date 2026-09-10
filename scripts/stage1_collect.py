"""Stage 1 collection: ADTS selection masks over Video-MME short, with controls.

Design points that matter for correctness:

* Geometric arms are applied to the [0:364, 0:364] sub-region of the 384x384 pixel tensor,
  NOT to the whole frame and NOT to the source video:
    - The 13x13 pooled grid is built from SigLIP patches 0..25, i.e. pixels [0, 364).
      (27 patches * 14 px = 378 px, of which avg_pool2d(2) drops patch 26; pixels
      [378, 384) are never patched at all.) Flipping [0:364) maps pooled cell c to
      original cell 12-c exactly; flipping the full 384 would be off by 20 px and would not
      map cells to cells at all.
    - Transforming the source video and re-encoding would put an x264 generation between the
      arms. Its rate allocation and deblocking are themselves position-dependent, and
      re-encoding changes the GOP layout so uniform sampling would draw *different source
      frames* per arm -- destroying the one comparison the arms exist to make.

* The [364:384) border strip is blanked to neutral grey for every arm including native.
  It is outside the pooled grid, but 26 layers of global self-attention still mix it into
  all 169 cells, and it does not move with the content under hflip or shift. Leaving it
  as-is would inject an arm-dependent perturbation concentrated on row/col 12 -- precisely
  where the hypothesised edge effect lives.

* Pooled features are computed once per (video, arm) and reused across the retention-ratio
  sweep and the method ablation, but each flashvid_compression call gets a CLONE:
  utils.py:58 hands segment_compression a view of the input and spatiotemporal_compression
  writes back into it (utils.py:376, :387), so an un-cloned reuse would feed every later
  iteration a TAM-corrupted feature map with no error raised.

* Resume is per work item, atomic, and discovered through an append-only journal rather
  than by globbing the output directory: the mount is JuiceFS, where file creation costs
  ~37 ms and a directory walk of thousands of entries is slower than the work itself.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
FLASHVID_SRC = os.path.join(ROOT, "code", "FlashVID")
if FLASHVID_SRC not in sys.path:
    sys.path.insert(0, FLASHVID_SRC)

from adts_harness import (  # noqa: E402
    EVAL_OVERWRITE_CONFIG,
    EXPECTED_NV,
    FLASHVID_KWARGS,
    GRID,
    MODEL_PATH,
    MODEL_REVISION,
    VISION_TOWER_PATH,
    VISION_TOWER_REVISION,
    AdtsRecorder,
    assert_v2_equivalence,
    build_model,
    encode_to_pooled,
    load_video_frames,
)
from data_manifest import load_video_ids, require_video_files  # noqa: E402

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
VIDEO_DIR = f"{W}/data/videos_short"
OUT_DIR = f"{W}/runs/stage1"
CELL = 28                      # pixels per pooled cell
ACTIVE = GRID * CELL           # 364: the region the pooled grid actually sees
N_FRAMES = 64

RETENTION_RATIOS = [0.10, 0.15, 0.20, 0.25]   # the sweep reported in the paper
METHODS_NATIVE_ABLATION = ["attn", "div", "attn_div"]   # v2 is the main arm
SHIFT_CELLS = (3, 3)           # diagonal roll, in pooled cells
N_SYNTHETIC = 30
BORDER_VALUE = 0.0             # normalised mid-grey (SigLIP mean=std=0.5)
DTYPE = torch.float16
ATTN_IMPL = "sdpa"             # LLM only, and the LLM never runs (tripwire in build_model)


# --------------------------------------------------------------------------- arms
def blank_border(px):
    """Neutralise the [364:384) strip so it is a true constant across arms."""
    px[..., ACTIVE:, :] = BORDER_VALUE
    px[..., :, ACTIVE:] = BORDER_VALUE
    return px


def arm_native(px):
    return px.clone()


def arm_hflip(px):
    """Horizontal flip of the active region -> pooled cell c holds original cell 12-c."""
    out = px.clone()
    out[..., :ACTIVE, :ACTIVE] = torch.flip(px[..., :ACTIVE, :ACTIVE], dims=[-1])
    return out


def arm_shift(px, dr=SHIFT_CELLS[0], dc=SHIFT_CELLS[1]):
    """Circular roll of the active region by whole pooled cells.

    Unlike a crop or pad this is a bijection on the grid (cell (r,c) holds original cell
    (r-dr, c-dc) mod 13) and, being a whole-patch translation, it permutes patch embeddings
    exactly. The only thing that does NOT move with it is the tower's absolute position
    embedding -- which is exactly what makes it a clean content-vs-position discriminator.
    """
    out = px.clone()
    out[..., :ACTIVE, :ACTIVE] = torch.roll(
        px[..., :ACTIVE, :ACTIVE], shifts=(dr * CELL, dc * CELL), dims=(-2, -1)
    )
    return out


ARMS = {"native": arm_native, "hflip": arm_hflip, "shift": arm_shift}
TRANSLATION_ARMS = {"native", "shift"}


def expected_permutation(arm):
    """Index permutation the arm induces on the 13x13 grid: new[j] holds old[perm[j]]."""
    perm = np.zeros(GRID * GRID, dtype=np.int64)
    for r in range(GRID):
        for c in range(GRID):
            if arm == "native":
                sr, sc = r, c
            elif arm == "hflip":
                sr, sc = r, GRID - 1 - c
            elif arm == "shift":
                sr = (r - SHIFT_CELLS[0]) % GRID
                sc = (c - SHIFT_CELLS[1]) % GRID
            else:
                raise ValueError(arm)
            perm[r * GRID + c] = sr * GRID + sc
    return perm


# ------------------------------------------------------------------ synthetic controls
def synthetic_video(kind, seed, n=N_FRAMES, size=384):
    """Content whose statistics are position-independent by construction.

    Generated as arrays, never encoded to mp4: x264 deblocking and quantisation would smooth
    away exactly the spatial homogeneity these controls exist to provide.
    """
    rng = np.random.default_rng(seed)
    if kind == "noise":
        return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    if kind == "static_texture":
        base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        return np.repeat(base[None], n, axis=0)
    if kind == "grating":
        y, x = np.mgrid[0:size, 0:size]
        out = np.empty((n, size, size, 3), dtype=np.uint8)
        theta = rng.uniform(0, np.pi)
        period = rng.uniform(18, 46)
        for t in range(n):
            phase = 2 * np.pi * t / n
            v = np.sin((x * np.cos(theta) + y * np.sin(theta)) / period * 2 * np.pi + phase)
            out[t] = np.repeat((((v + 1) * 127.5).astype(np.uint8))[:, :, None], 3, axis=2)
        return out
    raise ValueError(kind)


SYNTH_KINDS = ["noise", "static_texture", "grating"]


def synthetic_items():
    """Synthetics run through every arm too.

    Real footage carries centre-framing and horizon priors, so "does the heatmap follow the
    content or stay on the grid" is confounded there. On position-independent content it is
    not, which makes this the cleanest version of the study's central test.
    """
    items = []
    for i in range(N_SYNTHETIC):
        kind = SYNTH_KINDS[i % len(SYNTH_KINDS)]
        for a in ARMS:
            items.append({"video_id": f"__synth_{kind}_{i:02d}", "arm": a,
                          "synthetic": kind, "seed": 1000 + i})
    return items


# --------------------------------------------------------------------------- covariates
COVARIATE_NAMES = ["luma_var", "edge_energy", "temporal_diff"]


def cell_covariates(px):
    """Per-cell low-level content stats on the same 13x13 grid as the ADTS mask.

    Computed from the resized tensor, not the source frames: the anisotropic squash is itself
    position-dependent, so covariates measured pre-resize would not describe what the encoder
    saw. Accumulated in fp32 -- fp16 quantisation error is largest in low-texture cells, and
    would bias the residual in the same direction as the effect under test.
    """
    g = px.float().mean(1)                                   # (F, 384, 384) grey
    f = g.shape[0]
    # Gradients are taken on [0:ACTIVE+1] so the last cell row/col gets its full complement;
    # truncating at ACTIVE would leave the grid edge ~3.5% low, collinear with the effect.
    gy = (g[:, 1:ACTIVE + 1, :ACTIVE] - g[:, :ACTIVE, :ACTIVE]).abs()
    gx = (g[:, :ACTIVE, 1:ACTIVE + 1] - g[:, :ACTIVE, :ACTIVE]).abs()
    a = g[:, :ACTIVE, :ACTIVE]

    def to_cells(t):
        return (t.reshape(f, GRID, CELL, GRID, CELL)
                 .permute(0, 1, 3, 2, 4)
                 .reshape(f, GRID * GRID, CELL * CELL))

    cells = to_cells(a)
    var = cells.var(dim=-1)
    edge = to_cells(gy + gx).mean(-1)
    tdiff = torch.full_like(var, float("nan"))               # frame 0 is undefined, not zero
    if f > 1:
        tdiff[1:] = (cells[1:] - cells[:-1]).abs().mean(-1)
    return torch.stack([var, edge, tdiff], dim=-1)           # (F, 169, 3)


# --------------------------------------------------------------------------- worklist
def build_worklist():
    vids = load_video_ids()
    require_video_files(vids, VIDEO_DIR)
    items = [{"video_id": v, "arm": a, "synthetic": None, "seed": None}
             for v in vids for a in ARMS]
    return items + synthetic_items()


def item_key(it):
    return f"{it['video_id']}__{it['arm']}"


def stable_shard(key, n):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


# Everything that can change the recorded numbers belongs in the hash. alpha and expansion
# decide k (16 at R=0.10); the segment thresholds decide the segmentation the v2 event-
# relevance term is computed over; EVAL_OVERWRITE_CONFIG decides whether the grid is 13x13
# or 14x14, i.e. the entire coordinate system. Omitting any of them lets a stale shard be
# silently merged with a fresh one.
_SRC = "".join(inspect.getsource(f) for f in
               (cell_covariates, synthetic_video, arm_native, arm_hflip, arm_shift,
                blank_border, expected_permutation))
CONFIG = {
    "retention_ratios": RETENTION_RATIOS,
    "methods": METHODS_NATIVE_ABLATION + ["attn_div_v2"],
    "shift_cells": list(SHIFT_CELLS), "n_frames": N_FRAMES, "grid": GRID, "active": ACTIVE,
    "border_value": BORDER_VALUE, "dtype": str(DTYPE), "attn_impl": ATTN_IMPL,
    "flashvid_kwargs": FLASHVID_KWARGS, "model": MODEL_PATH,
    "model_revision": MODEL_REVISION,
    "vision_tower": VISION_TOWER_PATH,
    "vision_tower_revision": VISION_TOWER_REVISION,
    "overwrite_config": EVAL_OVERWRITE_CONFIG,
    "n_synthetic": N_SYNTHETIC, "synth_kinds": SYNTH_KINDS, "synth_seed_base": 1000,
    "src_md5": hashlib.md5(_SRC.encode()).hexdigest(),
    "version": 6,
}
CONFIG_HASH = hashlib.md5(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- preflight
def preflight(model, image_processor, device):
    """Verify the arms before collecting.

    Translations move whole patches, so their patch embedding is an exact permutation of the
    native one. Reflections are not: torch.flip also reverses pixel order inside each 14x14
    patch, and Conv2d of a mirrored input is not the Conv2d of the original. Demanding a
    bitwise feature permutation from hflip would be wrong; what it must guarantee is the
    pixel-level cell correspondence.

    Also records how far the FULL tower is from equivariant under shift. ADTS consumes tower
    output, so that number is the scale against which any measured positional effect has to
    be judged -- it is reported, never asserted.
    """
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(2, 384, 384, 3), dtype=np.uint8)
    px = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    px = blank_border(px.to(DTYPE).to(device))
    pe = model.get_vision_tower().vision_tower.vision_model.embeddings.patch_embedding

    def pooled_patchemb(t):
        with torch.inference_mode():
            z = pe(t).flatten(2).transpose(1, 2)
            return model.get_2dPool(z).float().cpu()

    base_feat = pooled_patchemb(px)
    report = {}
    for arm, fn in ARMS.items():
        out = blank_border(fn(px))
        perm = expected_permutation(arm)

        pix_err = 0.0
        for r in range(GRID):
            for c in range(GRID):
                src = perm[r * GRID + c]
                got = out[..., r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                want = px[..., (src // GRID) * CELL:(src // GRID + 1) * CELL,
                          (src % GRID) * CELL:(src % GRID + 1) * CELL]
                if arm == "hflip":
                    want = torch.flip(want, dims=[-1])
                pix_err = max(pix_err, float((got.float() - want.float()).abs().max()))

        feat_err = None
        if arm in TRANSLATION_ARMS:
            feat_err = float((pooled_patchemb(out) - base_feat[:, perm, :]).abs().max())

        report[arm] = {"pixel_cell_err": pix_err, "feature_perm_err": feat_err,
                       "ok": pix_err < 1e-3 and (feat_err is None or feat_err < 1e-2)}

    with torch.inference_mode():
        p_base, a_base = encode_to_pooled(model, px)
        p_shift, _ = encode_to_pooled(model, blank_border(arm_shift(px)))
    perm = expected_permutation("shift")
    report["shift"]["tower_nonequivariance_rel"] = float(
        (p_shift - p_base[:, perm, :]).abs().max() / p_base.abs().max())

    k = int(np.ceil(np.ceil(EXPECTED_NV * RETENTION_RATIOS[0] * FLASHVID_KWARGS["expansion"])
                    * FLASHVID_KWARGS["alpha"]))
    report["v2_equivalence"] = assert_v2_equivalence(p_base, a_base, k)
    return report


# --------------------------------------------------------------------------- main loop
def run(shard, nshards, device, limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    journal = f"{OUT_DIR}/progress_shard{shard}.jsonl"
    done = set()
    if os.path.exists(journal):
        for line in open(journal):
            try:
                rec = json.loads(line)
                # Only successes count. Marking failures done would drop a video from the
                # sample permanently, and the loss would never surface anywhere.
                if rec.get("config_hash") == CONFIG_HASH and rec.get("ok"):
                    done.add(rec["key"])
            except Exception:  # noqa: BLE001 - a torn final line is expected after a kill
                continue

    items = [it for it in build_worklist() if stable_shard(item_key(it), nshards) == shard]
    todo = [it for it in items if item_key(it) not in done]
    if limit:
        todo = todo[:limit]
    print(f"shard {shard}/{nshards} device={device} config={CONFIG_HASH} "
          f"total={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    _, model, image_processor, _ = build_model(device=device, dtype=DTYPE,
                                               attn_implementation=ATTN_IMPL,
                                               retention_ratio=RETENTION_RATIOS[0])

    pf = preflight(model, image_processor, device)
    print("preflight:", json.dumps(pf), flush=True)
    assert all(pf[a]["ok"] for a in ARMS), f"arm is not an exact grid permutation: {pf}"
    assert pf["v2_equivalence"]["identical"], \
        f"instrumented v2 diverges from the shipped one: {pf['v2_equivalence']}"

    import flashvid.llava_arch as fa

    t0 = time.time()
    jf = open(journal, "a")
    for n, it in enumerate(todo, 1):
        key = item_key(it)
        try:
            if it["synthetic"]:
                frames = synthetic_video(it["synthetic"], it["seed"])
            else:
                frames = load_video_frames(f"{VIDEO_DIR}/{it['video_id']}.mp4", N_FRAMES)
            px = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            px = blank_border(px.to(DTYPE).to(device))
            px = blank_border(ARMS[it["arm"]](px))

            cov = cell_covariates(px).cpu().numpy().astype(np.float16)
            with torch.inference_mode():
                pooled, pooled_attn = encode_to_pooled(model, px)

            payload = {
                "video_id": it["video_id"], "arm": it["arm"],
                "synthetic": it["synthetic"] or "", "config_hash": CONFIG_HASH,
                "config_json": json.dumps(CONFIG, sort_keys=True),
                "covariates": cov,
                "covariate_names": np.array(COVARIATE_NAMES),
                # Full per-frame CLS attention, not a summary: it IS calibration_term1 and
                # the entire criterion of the attn baseline, so it is the strongest content
                # covariate available. Collapsing it to a mean would leave the content
                # regression under-corrected and push the residual onto the position term.
                "cls_attn": pooled_attn.float().cpu().numpy().astype(np.float16),
                "perm": expected_permutation(it["arm"]).astype(np.int16),
                "shift_cells": np.array(SHIFT_CELLS, dtype=np.int16),
                "preflight_pixel_err": np.float32(pf[it["arm"]]["pixel_cell_err"]),
                "tower_nonequivariance_rel": np.float32(
                    pf["shift"].get("tower_nonequivariance_rel", np.nan)),
            }

            cfg = model.flashvid_config
            for R in RETENTION_RATIOS:
                methods = ["attn_div_v2"]
                if it["arm"] == "native" and R == RETENTION_RATIOS[0]:
                    methods = methods + METHODS_NATIVE_ABLATION
                for meth in methods:
                    cfg.retention_ratio = R
                    cfg.token_selection_method = meth
                    rec = AdtsRecorder()
                    with rec.active(), torch.inference_mode():
                        fa.flashvid_compression(video_features=pooled.clone(),
                                                cls_attention=pooled_attn.clone(),
                                                flashvid_config=cfg)
                    tag = f"R{int(R*100):03d}_{meth}"
                    assert len(rec.records) == 1, (tag, len(rec.records))
                    r = rec.records[0]
                    # Cheap invariants. These are the only probe for "the hook silently did
                    # not fire" or "segment bookkeeping drifted"; both would otherwise write
                    # a well-formed but wrong mask.
                    assert r["num_visual_tokens"] == EXPECTED_NV, (tag, r["num_visual_tokens"])
                    assert sum(r["segment_lengths"]) == r["num_frames"], \
                        (tag, r["segment_lengths"])
                    assert r["n_adts_slots"] == r["num_frames"] * r["num_attn_div_tokens"], \
                        (tag, r["n_adts_slots"])
                    payload[f"keep_{tag}"] = r["adts_keep_global"].astype(np.int32)
                    payload[f"final_{tag}"] = r["final_global_indices"].astype(np.int32)
                    payload[f"kpf_{tag}"] = r["distinct_per_frame"].astype(np.int16)
                    payload[f"seglen_{tag}"] = np.array(r["segment_lengths"], dtype=np.int32)
                    if r["ties"] is not None:
                        payload[f"ties_{tag}"] = r["ties"].astype(np.int16)
                        payload[f"margin_{tag}"] = r["margins"].astype(np.float32)
                    payload[f"meta_{tag}"] = np.array([
                        r["num_frames"], r["num_visual_tokens"], r["num_attn_div_tokens"],
                        r["num_sttm_tokens"], r["n_segments"], r["n_single_frame_segments"],
                        r["n_adts_slots"], r["n_adts_distinct"], r["n_duplicate_slots"],
                        r["n_final"], r["n_final_distinct"],
                    ], dtype=np.int32)
            cfg.token_selection_method = "attn_div_v2"
            cfg.retention_ratio = RETENTION_RATIOS[0]

            out_path = f"{OUT_DIR}/{key}.npz"
            # numpy appends .npz when the name lacks it, so the temp name must already end in
            # .npz or os.replace would look for a file that was never written.
            tmp = f"{OUT_DIR}/.{key}.tmp{shard}.{os.getpid()}.npz"
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, out_path)
            assert os.path.getsize(out_path) > 0, out_path
            jf.write(json.dumps({"key": key, "config_hash": CONFIG_HASH, "ok": True}) + "\n")
            jf.flush()
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el:.0f}s  {el/n:.2f}s/item", flush=True)
        except (torch.cuda.OutOfMemoryError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 - one bad video must not kill the shard
            traceback.print_exc()
            jf.write(json.dumps({"key": key, "config_hash": CONFIG_HASH,
                                 "ok": False, "error": str(exc)[:300]}) + "\n")
            jf.flush()
    jf.close()

    # Fail loud. Without this, "finished N items" prints identically whether 930 files landed
    # or zero did. Arms are separate work items, so a partial failure also silently turns the
    # paired native/hflip/shift comparison into an unpaired subset -- and failures correlate
    # with video length, so the missingness would not be random.
    missing = [item_key(it) for it in items
               if not os.path.exists(f"{OUT_DIR}/{item_key(it)}.npz")]
    by_vid = {}
    for it in items:
        by_vid.setdefault(it["video_id"], []).append(
            os.path.exists(f"{OUT_DIR}/{item_key(it)}.npz"))
    unbalanced = [v for v, oks in by_vid.items() if any(oks) and not all(oks)]
    print(f"shard {shard} finished {len(todo)} items in {time.time()-t0:.0f}s "
          f"MISSING={len(missing)} UNBALANCED_VIDEOS={len(unbalanced)}", flush=True)
    if missing:
        print("MISSING KEYS:", missing[:50], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sys.exit(run(a.shard, a.nshards, a.device, a.limit))
