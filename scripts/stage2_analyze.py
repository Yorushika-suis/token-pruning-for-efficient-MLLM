"""Stage-2 横向对比分析：从 mask 算出全部统计量。

对应本阶段的三个目标：
  目标2 探索现有算法边界 -> 两两一致度（Jaccard + 机会校正 AO）、空间热力图、家族聚类
  目标3 量化 token 冗余   -> 三种读法都算：
        (a) 位置 (r,c) 跨视频的选中率方差   —— 测「算法的位置先验有多强」，H1 的延续
        (b) token 实例跨算法的选择方差       —— 测「这个 token 到底重不重要」，四象限
        (c) 内容侧冗余（留给 stage2_content.py）
  目标4 提出新算法        -> 共识核心 / 家族共识 / 分歧前沿 / 全部忽略 的四分划

机会校正（关键）
--------------
低保留率下交集天然就小，不校正会把「预算小」误读成「算法更一致」。
  E_random = Ka*Kb/N
  AO = (|Sa∩Sb| - E_random) / (min(Ka,Kb) - E_random)
AO=1 表示完全一致，AO=0 表示与随机无异，AO<0 表示比随机还互斥。
逐帧算再对帧与视频取平均——不要先把 mask 拍平再算，那会把不同帧的预算混在一起。
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_manifest import directory_ids, load_video_ids, require_exact_ids  # noqa: E402

ROOT = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
MASKS = os.path.join(ROOT, "runs", "stage2_masks")
OUTD = os.path.join(ROOT, "results")
NV, GRID, NF = 169, 13, 64

# H1 实测的 register / sink 位置（(row,col) 0-indexed，按 R=0.10 选中率降序）。
# report/H1_report.md §6：(10,1) 在 19200 帧中选中率恒为 1.000，前 10 名吞掉 37.7% 预算。
H1_REGISTER_RC = [(10, 1), (0, 11), (12, 1), (2, 0), (12, 0), (0, 1),
                  (1, 0), (11, 0), (0, 0), (12, 12)]
H1_REGISTER_IDX = np.array([r * GRID + c for r, c in H1_REGISTER_RC], dtype=np.int64)

CONTROLS = {"random", "grid_uniform", "spatial_stratified"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_masks(vid):
    """-> {tag: (64,169) bool}"""
    with np.load(os.path.join(MASKS, f"{vid}.npz")) as z:
        return {k[2:]: np.unpackbits(z[k], axis=-1, count=NV).astype(bool)
                for k in z.files if k.startswith("m_")}


def ring_of(idx, grid=GRID):
    """到画面边框的环距：0=最外圈，6=中心。"""
    r, c = idx // grid, idx % grid
    return np.minimum(np.minimum(r, grid - 1 - r), np.minimum(c, grid - 1 - c))


RING = ring_of(np.arange(NV))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUTD, "stage2_analysis.json"))
    ap.add_argument("--npz", default=os.path.join(OUTD, "stage2_arrays.npz"))
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)

    vids = require_exact_ids(
        load_video_ids(),
        directory_ids(MASKS, ".npz"),
        label=f"mask artifacts in {MASKS}",
    )
    tags = sorted(load_masks(vids[0]))
    algos = sorted({t.split("__")[0] for t in tags})
    budgets = sorted({t.split("__R")[1] for t in tags})
    real = [x for x in algos if x not in CONTROLS]
    log(f"videos={len(vids)} algos={algos} budgets={budgets} 参与共识的真实算法={real}")

    # ---- 累加器 ----
    # pos_rate[R][algo] : (n_video, 169)  每视频每位置的选中率（分母 = 64 帧）
    pos_rate = {R: {g: np.zeros((len(vids), NV), np.float32) for g in algos} for R in budgets}
    # inter[R][(a,b)] : 逐帧 |Sa ∩ Sb| 的累加
    pairs = list(itertools.combinations(algos, 2))
    inter = {R: {p: 0.0 for p in pairs} for R in budgets}
    ksum = {R: {g: 0.0 for g in algos} for R in budgets}
    nframe = {R: 0 for R in budgets}
    # freq[R] : (n_video, 64, 169) uint8，某 token 被几个真实算法选中
    freq = {R: np.zeros((len(vids), NF, NV), np.uint8) for R in budgets}

    for vi, vid in enumerate(vids):
        m = load_masks(vid)
        for R in budgets:
            mats = {g: m[f"{g}__R{R}"] for g in algos}
            for g, mm in mats.items():
                pos_rate[R][g][vi] = mm.mean(axis=0)
                ksum[R][g] += mm.sum()
            for x, y in pairs:
                inter[R][(x, y)] += float((mats[x] & mats[y]).sum())
            nframe[R] += mats[algos[0]].shape[0]
            freq[R][vi] = sum(mats[g].astype(np.uint8) for g in real)
        if (vi + 1) % 50 == 0:
            log(f"  loaded {vi + 1}/{len(vids)}")

    res = {"n_videos": len(vids), "algos": algos, "budgets": budgets,
           "real_algos": real, "controls": sorted(CONTROLS & set(algos))}
    arrays = {}

    for R in budgets:
        nf = nframe[R]
        kbar = {g: ksum[R][g] / nf for g in algos}          # 每帧平均实际选中数
        blk = res.setdefault(f"R{R}", {})
        blk["k_per_frame"] = {g: round(kbar[g], 4) for g in algos}

        # ---- 1. 空间：每位置跨视频平均选中率 + 环形剖面 + register 占用 ----
        spatial, ringprof, regshare, xvstd = {}, {}, {}, {}
        for g in algos:
            pr = pos_rate[R][g]                              # (n_video,169)
            mean = pr.mean(0)
            spatial[g] = mean
            ringprof[g] = [float(mean[RING == r].mean()) for r in range(7)]
            # register 位置吞掉的预算份额（公平值 = 10/169 = 5.9%）
            regshare[g] = float(mean[H1_REGISTER_IDX].sum() / mean.sum())
            # 目标3(a)：同一位置在不同视频间的选中率标准差
            xvstd[g] = pr.std(0)
        blk["ring_profile"] = {g: [round(x, 5) for x in v] for g, v in ringprof.items()}
        blk["h1_register_budget_share"] = {g: round(v, 4) for g, v in regshare.items()}
        blk["h1_register_share_fair"] = round(len(H1_REGISTER_IDX) / NV, 4)
        blk["cross_video_std_mean"] = {g: round(float(xvstd[g].mean()), 5) for g in algos}
        for g in algos:
            arrays[f"spatial_R{R}_{g}"] = spatial[g].astype(np.float32)
            arrays[f"xvstd_R{R}_{g}"] = xvstd[g].astype(np.float32)

        # ---- 2. 两两一致度：Jaccard 与机会校正 AO ----
        jac, ao = {}, {}
        for x, y in pairs:
            I = inter[R][(x, y)] / nf                        # 每帧平均交集
            Ka, Kb = kbar[x], kbar[y]
            U = Ka + Kb - I
            e = Ka * Kb / NV
            jac[f"{x}|{y}"] = round(I / U, 4) if U > 0 else None
            den = min(Ka, Kb) - e
            ao[f"{x}|{y}"] = round((I - e) / den, 4) if den > 1e-9 else None
        blk["jaccard"] = jac
        blk["chance_corrected_overlap"] = ao

        # ---- 3. 目标3(b) + 目标4：跨算法选择频率的四象限 ----
        A = len(real)
        fr = freq[R]                                          # (V,64,169) 0..A
        hist = np.bincount(fr.reshape(-1), minlength=A + 1).astype(np.int64)
        blk["consensus_hist"] = {str(i): int(hist[i]) for i in range(A + 1)}
        tot = int(hist.sum())
        blk["quadrants"] = {
            "all_ignored_frac": round(float(hist[0] / tot), 4),
            "frontier_frac": round(float(hist[1:A].sum() / tot), 4),
            "consensus_core_frac": round(float(hist[A] / tot), 4),
        }
        # 每个 token 实例被选中的跨算法标准差（伯努利，A 个算法）
        p = fr.astype(np.float32) / A
        blk["cross_algo_std_mean"] = round(float(np.sqrt(p * (1 - p)).mean()), 5)
        # 共识核心与全被忽略的 token 在空间上落在哪
        arrays[f"core_spatial_R{R}"] = (fr == A).mean(axis=(0, 1)).astype(np.float32)
        arrays[f"ignored_spatial_R{R}"] = (fr == 0).mean(axis=(0, 1)).astype(np.float32)
        arrays[f"freq_hist_R{R}"] = hist

        log(f"R{R}: core={blk['quadrants']['consensus_core_frac']:.3f} "
            f"frontier={blk['quadrants']['frontier_frac']:.3f} "
            f"ignored={blk['quadrants']['all_ignored_frac']:.3f}")

    arrays["h1_register_idx"] = H1_REGISTER_IDX
    arrays["ring"] = RING
    np.savez_compressed(a.npz, **arrays)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"wrote {a.out} 与 {a.npz}")


if __name__ == "__main__":
    sys.exit(main())
