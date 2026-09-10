"""Stage-2 第一步：把视觉侧特征一次性落盘，之后所有算法对比都在离线特征上跑。

为什么要这一步
--------------
Stage-1 只存了选择结果（keep 索引）和 cls_attn，**没存 pooled 特征本身**，所以每换一个
选择算法就要重跑一遍视觉前向（990 item / 21 分钟 / 2xH20，其中约一半是 decord 解码）。
把特征落盘之后，加一个新算法的边际成本从「21 分钟 GPU」降到「一遍 numpy」，
新算法的设计-验证循环可以做几十上百轮。这是本阶段最重要的基础设施决策。

存什么（每视频一个 npz，约 185 MB）
--------------------------------
  pooled169  (64,169,3584) fp16   统一 token universe U，ADTS 就在这个网格上跑
  attn169    (64,169)      fp16   = FlashVID 的 calibration_term1 本体
  sig729     (64,729,1152) fp16   pre-projector 的 SigLIP 输出，VisionZip/VisPruner/
                                  PruMerge/HiRED 这批方法原生就在这个网格上跑
  attn729    (64,729)      fp16   末层 attention 对 head 与全 query 位置取均值
post-projector 的 (64,729,3584) **不存**——它 = mm_projector(sig729)，projector 是逐 token
MLP，权重单独存一份（约 30M 参数）离线重放即可，省 334 MB/视频。

为什么直接用 build_model（会调 flashvid()）而不是干净的 hook 路线
--------------------------------------------------------------
本进程只负责「产出特征」，不做任何选择实验，所以 F4（flashvid 改全局 class）无害；
反过来，走 flashvid 的官方通路能保证 pooled169/attn169 与 Stage-1 的 990 个 npz
**逐位一致**，这是现成的、极强的正确性锚点（见 --preflight）。
选择算法全部在另一个离线进程里跑，不受污染。

用法
----
  # 先做门禁：与 Stage-1 的 cls_attn 逐位比对
  python3 stage2_dump_features.py --preflight
  # 双卡分片后台跑
  for s in 0 1; do setsid nohup python3 stage2_dump_features.py --shard $s --nshard 2 \
      --device cuda:$s > ../runs/dump_shard$s.log 2>&1 & done
  python3 stage2_dump_features.py --status
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_manifest import load_video_ids, require_video_files  # noqa: E402

ROOT = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
FLASHVID_SRC = os.path.join(ROOT, "code", "FlashVID")
VIDEO_DIR = os.path.join(ROOT, "data", "videos_short")
STAGE1_DIR = os.path.join(ROOT, "runs", "stage1")
OUT_DIR = os.path.join(ROOT, "runs", "stage2_features")
VERSION = 1

NV, GRID, NP729, SIDE = 169, 13, 729, 27
NUM_FRAMES = 64
# 13x13 池化网格只看得到像素 [0,364)：27 个 patch 覆盖 378px，avg_pool 丢掉 patch 26，
# [378,384) 从不成 patch。Stage-1 对所有 arm 统一把 [364:384) 边条涂成中性灰
# （SigLIP mean=std=0.5，归一化后就是 0.0），因为全局注意力会把它混进全部 169 格
# 而它又不随几何对照臂移动。这里必须照做，否则 attn169 与 Stage-1 对不上。
ACTIVE, BORDER_VALUE = GRID * 28, 0.0


def blank_border(px):
    px[..., ACTIVE:, :] = BORDER_VALUE
    px[..., :, ACTIVE:] = BORDER_VALUE
    return px


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def video_ids():
    ids = load_video_ids()
    return require_video_files(ids, VIDEO_DIR)


def _stable_shard(key, n):
    import hashlib
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


# ---------------------------------------------------------------- 模型

def build(device, dtype=torch.float16):
    """装配模型并把视觉侧搬上 GPU。LLM 留在 CPU（本步骤不碰 LLM）。"""
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "weights"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if FLASHVID_SRC not in sys.path:
        sys.path.insert(0, FLASHVID_SRC)
    import adts_harness as H

    # tripwire=True 让任何 LLM decoder 前向立刻抛异常——本步骤不该碰 LLM。
    _, model, image_processor, _ = H.build_model(
        device=device, dtype=dtype, retention_ratio=0.10, tripwire=True, verbose=True)
    return H, model, image_processor


@torch.no_grad()
def encode_all(H, model, image_processor, frames, device, dtype):
    """源帧 -> 四个张量。与 llava_arch.py:62,77-78 同一条通路。"""
    px = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    px = blank_border(px.to(device=device, dtype=dtype))
    assert px.shape[0] == NUM_FRAMES and px.shape[-1] == 384, px.shape

    # flashvid patch 过的 encode_images 返回 (post-projector 729 特征, 729 注意力)
    feat729_proj, attn729 = model.encode_images(px)
    assert feat729_proj.shape[:2] == (NUM_FRAMES, NP729), feat729_proj.shape
    assert attn729.shape == (NUM_FRAMES, NP729), attn729.shape

    pooled = model.get_2dPool(feat729_proj)
    attn169 = model.get_2dPool(attn729.unsqueeze(-1)).squeeze(-1)
    assert pooled.shape[:2] == (NUM_FRAMES, NV), pooled.shape

    # pre-projector 的 SigLIP 输出：VisionZip / VisPruner / PruMerge / HiRED 原生网格。
    # 注意 flashvid() 把 SigLipVisionTower.forward 也 patch 成返回 (features, attentions)，
    # 原版只返回 features——两种都要能吃。
    vt = model.get_vision_tower()
    out = vt(px)
    sig729 = out[0] if isinstance(out, (tuple, list)) else out
    assert sig729.shape[:2] == (NUM_FRAMES, NP729), sig729.shape
    assert sig729.shape[-1] == 1152, f"期望 SigLIP hidden 1152，实得 {sig729.shape[-1]}"

    return (pooled.to(torch.float16).cpu().numpy(),
            attn169.to(torch.float16).cpu().numpy(),
            sig729.to(torch.float16).cpu().numpy(),
            attn729.to(torch.float16).cpu().numpy())


def dump_projector(model, path):
    if os.path.exists(path):
        return
    sd = {k: v.to(torch.float16).cpu() for k, v in
          model.get_model().mm_projector.state_dict().items()}
    torch.save(sd, path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"projector saved -> {path} ({sum(v.numel() for v in sd.values()) / 1e6:.1f}M params)")


# ---------------------------------------------------------------- 门禁

def preflight(device, n=3):
    """与 Stage-1 的 990 个 npz 逐位比对 attn169，确认通路完全一致。"""
    H, model, ip = build(device)
    dtype = torch.float16
    ok, ncmp = True, 0
    for vid in video_ids()[:n]:
        ref_p = os.path.join(STAGE1_DIR, f"{vid}__native.npz")
        if not os.path.exists(ref_p):
            log(f"  {vid}: 无 Stage-1 参考，跳过")
            continue
        frames = H.load_video_frames(os.path.join(VIDEO_DIR, f"{vid}.mp4"))
        pooled, attn169, sig729, attn729 = encode_all(H, model, ip, frames, device, dtype)
        ref = np.load(ref_p)["cls_attn"]
        same = np.array_equal(attn169, ref)
        mx = float(np.abs(attn169.astype(np.float64) - ref.astype(np.float64)).max())
        log(f"  {vid}: attn169 bitwise={same} maxabs={mx:.3e} | "
            f"pooled{pooled.shape} sig729{sig729.shape} attn729{attn729.shape}")
        ok &= same
        ncmp += 1
    if ncmp == 0:
        log("PREFLIGHT FAIL — 一个视频都没比对上，这是空真通过，不是通过")
        return False
    log(f"PREFLIGHT {'PASS' if ok else 'FAIL — 通路与 Stage-1 不一致，先查明再采集'} "
        f"({ncmp} 个视频逐位比对)")
    return ok


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ids = video_ids()

    if a.status:
        done = [f for f in os.listdir(OUT_DIR) if f.endswith(".npz")]
        sz = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in done)
        print(f"videos {len(ids)} | done {len(done)} | remaining {len(ids) - len(done)} "
              f"| {sz / 2**30:.1f} GiB")
        return
    if a.preflight:
        sys.exit(0 if preflight(a.device) else 1)

    mine = [v for v in ids if _stable_shard(v, a.nshard) == a.shard]
    todo = [v for v in mine if not os.path.exists(os.path.join(OUT_DIR, f"{v}.npz"))]
    if a.limit:
        todo = todo[:a.limit]
    log(f"shard {a.shard}/{a.nshard} on {a.device}: mine={len(mine)} todo={len(todo)}")
    if not todo:
        return

    H, model, ip = build(a.device)
    dump_projector(model, os.path.join(OUT_DIR, "_mm_projector_fp16.pt"))
    dtype = torch.float16
    t0 = time.time()
    for i, vid in enumerate(todo, 1):
        out = os.path.join(OUT_DIR, f"{vid}.npz")
        if os.path.exists(out):
            continue
        try:
            frames = H.load_video_frames(os.path.join(VIDEO_DIR, f"{vid}.mp4"))
            pooled, attn169, sig729, attn729 = encode_all(H, model, ip, frames, a.device, dtype)
        except Exception as e:
            log(f"  [{i}/{len(todo)}] FAIL {vid}: {type(e).__name__}: {e}")
            continue
        tmp = os.path.join(OUT_DIR, f".{vid}.tmp{a.shard}.{os.getpid()}.npz")
        # 不用 savez_compressed：高熵 fp16 激活压不动，只白烧 CPU
        np.savez(tmp, pooled169=pooled, attn169=attn169, sig729=sig729, attn729=attn729,
                 meta=np.array([VERSION, NUM_FRAMES, NV, NP729, 3584, 1152], dtype=np.int32))
        os.replace(tmp, out)
        if i % 10 == 0 or i == len(todo):
            el = time.time() - t0
            log(f"  [{i}/{len(todo)}] {vid} {os.path.getsize(out) / 2**20:.0f}MB "
                f"| {el / i:.2f}s/video | ETA {(len(todo) - i) * el / i / 60:.1f}min")
    log(f"shard {a.shard} DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
