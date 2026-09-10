"""Stage 0 gate: assertions A0-A7. Nothing downstream runs until all of these pass.

Each assertion targets a specific way this experiment could produce confident, wrong
numbers. The two that matter most are A3 (if index->coordinate mapping is wrong, every
result is wrong but nothing errors) and A4 (if ADTS v2 re-selects indices, the per-frame
budget is not constant and the exact null distribution is invalid).
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
FLASHVID_SOURCE = os.path.join(PROJECT_ROOT, "code", "FlashVID")
if FLASHVID_SOURCE not in sys.path:
    sys.path.insert(0, FLASHVID_SOURCE)

from adts_harness import (  # noqa: E402
    EXPECTED_NV,
    GRID,
    LlmInvoked,
    AdtsRecorder,
    build_model,
    encode_to_pooled,
    load_video_frames,
)

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
VIDEO_DIR = f"{W}/data/videos_short"
REPORT = f"{W}/runs/stage0_assertions.json"
DEV = "cuda:0"

GRAY = 110
BRIGHT = 245
results = {}


def record(name, passed, **detail):
    results[name] = {"pass": bool(passed), **detail}
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: " + json.dumps(detail, default=str)[:400], flush=True)
    return passed


def synth_frames(h, w, patches, n=2):
    """n identical frames of mid-gray with bright squares at normalised (u, v) centres."""
    img = np.full((h, w, 3), GRAY, dtype=np.uint8)
    side = max(4, int(round(min(h, w) * 0.045)))
    for u, v in patches:
        cx, cy = int(u * w), int(v * h)
        y0, y1 = max(0, cy - side // 2), min(h, cy + side // 2 + 1)
        x0, x1 = max(0, cx - side // 2), min(w, cx + side // 2 + 1)
        img[y0:y1, x0:x1] = BRIGHT
    return np.repeat(img[None], n, axis=0)


def pooled_of(model, image_processor, frames):
    px = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    px = px.half().to(DEV)
    with torch.inference_mode():
        pooled, pooled_attn = encode_to_pooled(model, px)
    return px, pooled.float().cpu(), pooled_attn.float().cpu()


def main():
    t0 = time.time()
    tokenizer, model, image_processor, _ = build_model(device=DEV, retention_ratio=0.10)

    # ---------------- A0: preprocessing contract ----------------
    # llava ships its own SigLipImageProcessor (note the capitalisation: not transformers'
    # SiglipImageProcessor). preprocessor_config.json does NOT drive it -- the constructor
    # defaults do. Its transform chain is convert_to_rgb -> to_numpy_array -> resize(384,384)
    # -> rescale -> normalize, with no crop and no pad step; crop_size is stored but never
    # read by preprocess(). So assert on the live object plus the observed output shape.
    import inspect

    ipd = {k: str(v) for k, v in vars(image_processor).items()}
    src = inspect.getsource(type(image_processor).preprocess)
    no_crop_in_chain = ("center_crop" not in src) and ("pad" not in src)

    vids = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    real = load_video_frames(os.path.join(VIDEO_DIR, vids[0]), 64)
    px = image_processor.preprocess(real, return_tensors="pt")["pixel_values"]
    record(
        "A0_preprocess_contract",
        tuple(px.shape) == (64, 3, 384, 384)
        and tuple(image_processor.size) == (384, 384)
        and no_crop_in_chain,
        processor_class=type(image_processor).__name__,
        pixel_values_shape=list(px.shape),
        source_frame_shape=list(real.shape),
        size=str(image_processor.size),
        crop_size=str(getattr(image_processor, "crop_size", None)),
        crop_or_pad_in_preprocess_source=not no_crop_in_chain,
        attrs=ipd,
        n_videos_available=len(vids),
        note="crop_size is stored but unused; the chain is a bare anisotropic resize",
    )

    # ---------------- A1: pooling mode / grid ----------------
    vt = model.get_vision_tower()
    dummy = torch.zeros(1, 729, model.config.hidden_size, device=DEV, dtype=torch.float16)
    with torch.inference_mode():
        pooled_dummy = model.get_2dPool(dummy)
    record(
        "A1_grid_169",
        pooled_dummy.shape[1] == EXPECTED_NV
        and model.config.mm_spatial_pool_mode == "average"
        and model.config.mm_newline_position == "frame"
        and vt.num_patches_per_side == 27,
        pooled_shape=list(pooled_dummy.shape),
        mm_spatial_pool_mode=model.config.mm_spatial_pool_mode,
        mm_newline_position=model.config.mm_newline_position,
        patches_per_side=vt.num_patches_per_side,
        note="checkpoint default is bilinear->196; eval-script override to average->169",
    )

    # ---------------- A2: vision tower provenance ----------------
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snap = snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2", local_files_only=True)
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    vkeys = [k for k in idx if "vision_tower" in k and k.endswith("q_proj.weight")]
    key = sorted(vkeys)[0]
    with safe_open(os.path.join(snap, idx[key]), framework="pt") as fh:
        w_ckpt = fh.get_tensor(key).float()
    sub = model
    for part in key.replace(".weight", "").split("."):
        sub = getattr(sub, part) if not part.isdigit() else sub[int(part)]
    w_live = sub.weight.detach().float().cpu()
    same = torch.allclose(w_ckpt, w_live, atol=1e-3, rtol=1e-3)
    record(
        "A2_vision_tower_from_checkpoint", same,
        key=key, shard=idx[key],
        n_vision_tensors=sum(1 for k in idx if "vision_tower" in k),
        max_abs_diff=float((w_ckpt - w_live).abs().max()),
        note="tower was fine-tuned (mm_vision_tower_lr=2e-06); stock google/siglip differs",
    )

    # ---------------- A3a: source -> 384 mapping policy ----------------
    a3a = {}
    ok_a3a = True
    for label, (h, w) in {"16:9": (720, 1280), "1:1": (720, 720), "9:16": (1280, 720)}.items():
        base = synth_frames(h, w, [])
        _, p_base, _ = pooled_of(model, image_processor, base)
        cols = {}
        for u in (0.04, 0.50, 0.96):
            fr = synth_frames(h, w, [(u, 0.5)])
            _, p, _ = pooled_of(model, image_processor, fr)
            d = (p[0] - p_base[0]).norm(dim=-1)
            top = int(d.argmax())
            cols[u] = {"index": top, "row": top // GRID, "col": top % GRID,
                       "delta": float(d.max()), "delta_median": float(d.median())}
        # Anisotropic resize keeps all three visible at cols ~0 / ~6 / ~12.
        # Shortest-edge resize + centre crop would clip u=0.04 and u=0.96 for 16:9.
        pred = {0.04: 0, 0.50: 6, 0.96: 12}
        hit = all(abs(cols[u]["col"] - pred[u]) <= 1 for u in pred)
        monotone = cols[0.04]["col"] < cols[0.50]["col"] < cols[0.96]["col"]
        a3a[label] = {"cols": cols, "hit": hit, "monotone": monotone}
        ok_a3a = ok_a3a and hit and monotone
    record("A3a_resize_is_anisotropic_no_crop", ok_a3a, per_aspect=a3a,
           note="edge patches at u=0.04/0.96 must be visible; centre-crop would clip them")

    # ---------------- A3b: 384 -> index bijection (all 169 cells) ----------------
    # Probe the patch embedding directly, not the tower output. The question A3b asks --
    # "does flat index j mean (j//13, j%13) in 384-pixel space?" -- is decided by the
    # Conv2d(k=14, s=14) + flatten(2).transpose(1,2) + avg_pool2d(2) chain alone. Attention
    # is irrelevant to it and actively harmful as a probe: a handful of high-norm
    # sink/register tokens dominate ||delta|| in the tower output no matter where the
    # stimulus is, so an argmax over tower features measures sink identity, not geometry.
    # Because the convolution is non-overlapping, a stimulus confined to one pooled cell
    # must move that cell and leave every other cell at exactly zero.
    cell = 384 // 27 * 2  # 28 px: one pooled cell = 2 SigLIP patches
    stack = [np.full((384, 384, 3), GRAY, dtype=np.uint8)]
    for r in range(GRID):
        for c in range(GRID):
            img = np.full((384, 384, 3), GRAY, dtype=np.uint8)
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = BRIGHT
            stack.append(img)
    frames = np.stack(stack)
    pxp = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    pxp = pxp.half().to(DEV)

    pe = vt.vision_tower.vision_model.embeddings.patch_embedding
    with torch.inference_mode():
        z = pe(pxp)                       # (B, d, 27, 27)
        z = z.flatten(2).transpose(1, 2)  # (B, 729, d)  -- siglip_encoder.py:169-174
        zp = model.get_2dPool(z).float().cpu()

    z_base = zp[0]
    hits, misses, leak = 0, [], 0.0
    for j in range(GRID * GRID):
        d = (zp[j + 1] - z_base).norm(dim=-1)
        top = int(d.argmax())
        second = float(d.clone().scatter(0, torch.tensor([top]), -1).max())
        leak = max(leak, second / max(float(d[top]), 1e-9))
        if top == j:
            hits += 1
        elif len(misses) < 20:
            misses.append({"expected_rc": [j // GRID, j % GRID],
                           "got_rc": [top // GRID, top % GRID]})
    record("A3b_index_bijection_row_major", hits == GRID * GRID and leak < 1e-3,
           hits=hits, total=GRID * GRID, misses=misses,
           max_offtarget_ratio=leak,
           note="patch-embedding probe; cell (r,c) covers px [28r,28r+28)x[28c,28c+28), "
                "index = r*13+c; non-overlapping conv => off-target delta must be 0")

    # ---------------- A3c: where does content sensitivity actually land? -------------
    # Same stimuli through the full tower. Not a gate -- a measurement. Positions that win
    # the argmax regardless of stimulus location are sink/register candidates, which is a
    # mechanism H1 must be able to distinguish from a genuine spatial prior.
    with torch.inference_mode():
        pooled_full = encode_to_pooled(model, pxp)[0].float().cpu()
    base_full = pooled_full[0]
    winners = {}
    correct = 0
    for j in range(GRID * GRID):
        d = (pooled_full[j + 1] - base_full).norm(dim=-1)
        top = int(d.argmax())
        winners[top] = winners.get(top, 0) + 1
        correct += int(top == j)
    sticky = sorted(winners.items(), key=lambda kv: -kv[1])[:8]
    record("A3c_tower_content_sensitivity", True,
           end_to_end_argmax_correct=correct, total=GRID * GRID,
           sticky_positions=[{"index": i, "rc": [i // GRID, i % GRID], "wins": n}
                             for i, n in sticky],
           note="measurement only; high-norm sink tokens capture the argmax and are a "
                "rival explanation for any apparent spatial prior in H1")

    # ---------------- A4 / A5: run real videos through ADTS ----------------
    rec = AdtsRecorder()
    per_video = []
    with rec.active():
        for name in vids[:8]:
            fr = load_video_frames(os.path.join(VIDEO_DIR, name), 64)
            pxv = image_processor.preprocess(fr, return_tensors="pt")["pixel_values"]
            pxv = pxv.half().to(DEV)
            with torch.inference_mode():
                pooled, pooled_attn = encode_to_pooled(model, pxv)
                import flashvid.llava_arch as fa
                fa.flashvid_compression(video_features=pooled, cls_attention=pooled_attn,
                                        flashvid_config=model.flashvid_config)
            per_video.append(name)

    recs = rec.records
    k = recs[0]["num_attn_div_tokens"]
    dup_slots = sum(r["n_adts_slots"] - r["n_adts_distinct"] for r in recs)
    expected_slots = sum(r["num_frames"] * r["num_attn_div_tokens"] for r in recs)
    record(
        "A4_no_duplicate_adts_selection", dup_slots == 0,
        n_videos=len(recs), num_attn_div_tokens=k,
        num_sttm_tokens=recs[0]["num_sttm_tokens"],
        num_visual_tokens=recs[0]["num_visual_tokens"],
        total_adts_slots=sum(r["n_adts_slots"] for r in recs),
        expected_slots=expected_slots,
        duplicate_slots=dup_slots,
        note="ADTS v2 lacks v1's anti-reselection guard; duplicates break the exact null",
    )

    single = sum(r["n_single_frame_segments"] for r in recs)
    fin_dup = sum(r["n_final"] - r["n_final_distinct"] for r in recs)
    record(
        "A5_segment_and_final_index_health", True,  # measurement, not a gate
        total_segments=sum(r["n_segments"] for r in recs),
        single_frame_segments=single,
        seg_length_examples=[r["segment_lengths"] for r in recs[:3]],
        final_tokens_total=sum(r["n_final"] for r in recs),
        final_duplicate_indices=fin_dup,
        note="single-frame segments bypass token_mask (utils.py:129-132) -> ADTS/TSTM "
             "index collisions; quantified, not gated",
    )

    # ---------------- A6: determinism ----------------
    fr = load_video_frames(os.path.join(VIDEO_DIR, vids[0]), 64)
    pxv = image_processor.preprocess(fr, return_tensors="pt")["pixel_values"].half().to(DEV)
    runs = []
    for _ in range(2):
        r2 = AdtsRecorder()
        with r2.active(), torch.inference_mode():
            pooled, pooled_attn = encode_to_pooled(model, pxv)
            import flashvid.llava_arch as fa
            fa.flashvid_compression(video_features=pooled, cls_attention=pooled_attn,
                                    flashvid_config=model.flashvid_config)
        runs.append(r2.records[0]["adts_keep_global"])
    same_dev = bool(np.array_equal(runs[0], runs[1]))

    cross = None
    if torch.cuda.device_count() > 1:
        model.get_vision_tower().to("cuda:1")
        model.get_model().mm_projector.to("cuda:1")
        r3 = AdtsRecorder()
        with r3.active(), torch.inference_mode():
            pooled, pooled_attn = encode_to_pooled(model, pxv.to("cuda:1"))
            import flashvid.llava_arch as fa
            fa.flashvid_compression(video_features=pooled, cls_attention=pooled_attn,
                                    flashvid_config=model.flashvid_config)
        other = r3.records[0]["adts_keep_global"]
        cross = bool(np.array_equal(runs[0], other))
        n_diff = int((runs[0] != other).sum()) if runs[0].shape == other.shape else -1
        model.get_vision_tower().to(DEV)
        model.get_model().mm_projector.to(DEV)
    else:
        n_diff = None
    record("A6_determinism", same_dev, same_device_bitwise=same_dev,
           cross_device_bitwise=cross, cross_device_differing_slots=n_diff,
           note="same-device identity is the gate; cross-device is measured, not gated")

    # ---------------- A7: LLM tripwire ----------------
    tripped = False
    try:
        with torch.inference_mode():
            model.get_model().layers[0](
                torch.zeros(1, 4, model.config.hidden_size, dtype=torch.float32)
            )
    except LlmInvoked:
        tripped = True
    except Exception as exc:  # noqa: BLE001 - any other error means the hook did not fire
        tripped = False
        results.setdefault("_a7_note", str(exc)[:200])
    record("A7_llm_tripwire", tripped,
           note="guarantees LLM device/dtype/attn_impl cannot influence Stage 1")

    gates = [k for k in results if k.startswith("A") and k not in ("A5_segment_and_final_index_health", "A3c_tower_content_sensitivity")]
    all_pass = all(results[g]["pass"] for g in gates)
    results["_summary"] = {
        "all_gates_pass": all_pass,
        "elapsed_s": round(time.time() - t0, 1),
        "gates": {g: results[g]["pass"] for g in gates},
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results["_summary"], indent=1), flush=True)
    print("wrote", REPORT, flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
