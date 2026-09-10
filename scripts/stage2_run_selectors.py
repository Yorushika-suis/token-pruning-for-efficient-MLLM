"""在离线特征上跑全部选择算法，把 mask 落成一份紧凑的 npz。

产物：runs/stage2_masks/{video_id}.npz，每个 (算法, 预算) 一个 packbits 后的 (64,169) bool。
体积：64x169 bit = 1352 B/组合，7 算法 x 4 预算 x 300 视频 ≈ 11 MB —— 相比特征的 53 GB
可以忽略，所以后续所有统计分析都只读 mask，不碰特征。

先跑 --gate：用 Stage-1 的 990 个 npz 校验重实现。attn / div / attn_div 三个是严格帧内
算法，必须逐位一致；attn_div_v2 依赖 DySeg 分段，单独报告一致率。
"""
import argparse
import json
import os
import sys
import time

# 必须在 import numpy/torch 之前设：torch 默认按核数（本机 64）开 BLAS 线程，
# 6 个 worker 就是 384 线程，会把内存受限的容器打成 BrokenProcessPool。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import token_selectors as S  # noqa: E402
from data_manifest import directory_ids, load_video_ids, require_exact_ids  # noqa: E402

ROOT = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
FEAT = os.path.join(ROOT, "runs", "stage2_features")
STAGE1 = os.path.join(ROOT, "runs", "stage1")
OUT = os.path.join(ROOT, "runs", "stage2_masks")
NV, NF = 169, 64


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def feat_ids():
    expected = load_video_ids()
    actual = directory_ids(FEAT, ".npz")
    return require_exact_ids(expected, actual, label=f"feature artifacts in {FEAT}")


def idx_to_mask(idx):
    """(F,k) 帧内索引 -> (F,169) bool。重复索引会被折叠，这是有意的。"""
    m = np.zeros((idx.shape[0], NV), dtype=bool)
    np.put_along_axis(m, idx.astype(np.int64), True, axis=1)
    return m


def run_one(vid, algos, budgets, existing=None):
    """跑指定算法；existing 里已有的 tag 原样保留，只补缺的（增量模式）。"""
    keep = dict(existing or {})
    want = [(n, R, k) for n in algos for R, k in budgets.items()
            if f"{n}__R{int(R * 100):03d}" not in keep]
    if not want:
        return None, None
    with np.load(os.path.join(FEAT, f"{vid}.npz")) as z:
        ctx = S.Ctx(z, vid=vid)
        out, meta = {}, {}
        for name, R, k in want:
            t0 = time.time()
            idx = S.REGISTRY[name](ctx, k)
            assert idx.shape == (ctx.F, k), (name, R, idx.shape)
            assert idx.min() >= 0 and idx.max() < NV, (name, R)
            tag = f"{name}__R{int(R * 100):03d}"
            out[tag] = np.packbits(idx_to_mask(idx), axis=-1)
            # 每帧 distinct 数：v2 的重选缺陷会让它 < k，是必须记录的诊断量
            meta[tag] = np.array(
                [k, int(np.median([len(set(r)) for r in idx])),
                 int(round((time.time() - t0) * 1000))], dtype=np.int32)
    return out, meta


def gate(n=5):
    """回归门禁。"""
    ids = [v for v in feat_ids()
           if os.path.exists(os.path.join(STAGE1, f"{v}__native.npz"))][:n]
    assert ids, "没有既有特征又有 Stage-1 参考的视频"
    agg = {}
    for vid in ids:
        with np.load(os.path.join(FEAT, f"{vid}.npz")) as z, \
             np.load(os.path.join(STAGE1, f"{vid}__native.npz")) as s1:
            rep = S.verify_against_stage1(S.Ctx(z, vid=vid), s1)
        for t, r in rep.items():
            agg.setdefault(t, []).append(r)
        log(f"  {vid}: " + "  ".join(
            f"{t.split('_', 1)[1]}={r['frame_exact_rate']:.3f}/{r['token_overlap']:.3f}"
            for t, r in rep.items()))
    log("--- 汇总（frame_exact_rate / token_overlap，n=%d）---" % len(ids))
    ok = True
    for t, rs in sorted(agg.items()):
        fe = float(np.mean([r["frame_exact_rate"] for r in rs]))
        to = float(np.mean([r["token_overlap"] for r in rs]))
        strict = "attn_div_v2" not in t          # v2 依赖分段，不做硬性要求
        verdict = "PASS" if (fe == 1.0 or not strict) else "FAIL"
        ok &= (fe == 1.0) or not strict
        log(f"  {t:24s} frame_exact={fe:.4f} token_overlap={to:.4f}  {verdict}"
            + ("  (v2 依赖 DySeg 分段，仅供参考)" if not strict else ""))
    log(f"GATE {'PASS' if ok else 'FAIL'}")
    return ok


def _work(args):
    """单个视频：补齐缺失的 (算法, 预算) 组合并原子落盘。已有的 tag 不重算。"""
    vid, algos, drop = args
    p = os.path.join(OUT, f"{vid}.npz")
    old_m, old_i = {}, {}
    if os.path.exists(p):
        try:
            with np.load(p) as z:
                old_m = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
                old_i = {k[2:]: z[k] for k in z.files if k.startswith("i_")}
        except Exception:
            old_m, old_i = {}, {}          # 半截文件，重算
    for t in [t for t in old_m if t.split("__")[0] in drop]:
        old_m.pop(t, None)
        old_i.pop(t, None)
    try:
        new_m, new_i = run_one(vid, algos, S.BUDGETS, existing=old_m)
    except Exception as e:
        return vid, f"{type(e).__name__}: {e}"
    if new_m is None:
        return vid, None                    # 已齐，无需重写
    old_m.update(new_m)
    old_i.update(new_i)
    tmp = p + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **{f"m_{k}": v for k, v in old_m.items()},
                        **{f"i_{k}": v for k, v in old_i.items()})
    os.replace(tmp, p)
    return vid, None


def _stable_shard(key, n):
    import hashlib
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--algos", default=",".join(S.REGISTRY))
    ap.add_argument("--recompute", default="",
                    help="逗号分隔的算法名，强制重算（实现改过时用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if a.status:
        ms = [f for f in os.listdir(OUT) if f.endswith(".npz")]
        tags = set()
        if ms:
            with np.load(os.path.join(OUT, ms[0])) as z:
                tags = {k[2:] for k in z.files if k.startswith("m_")}
        print(f"features {len(feat_ids())} | masks {len(ms)} | tags {len(tags)}")
        for t in sorted(tags):
            print("   ", t)
        return
    if a.gate:
        sys.exit(0 if gate() else 1)

    algos = [x for x in a.algos.split(",") if x]
    drop = {x for x in a.recompute.split(",") if x}
    # 不用 ProcessPoolExecutor：在这个内存受限的容器里 fork 与 spawn 都会
    # BrokenProcessPool（单进程跑同样的活完全正常，峰值 RSS 才 1.1 GB）。
    # 改成互不通信的独立分片进程 —— 与 stage2_dump_features.py 同一套做法，已验证可靠。
    todo = [v for v in feat_ids() if _stable_shard(v, a.nshard) == a.shard]
    if a.limit:
        todo = todo[:a.limit]
    log(f"shard {a.shard}/{a.nshard} algos={algos} recompute={sorted(drop)} "
        f"budgets={S.BUDGETS} candidates={len(todo)}")

    t0 = time.time()
    for done, vid in enumerate(todo, 1):
        _, err = _work((vid, algos, drop))
        if err:
            log(f"  FAIL {vid}: {err}")
        elif done % 20 == 0 or done == len(todo):
            el = time.time() - t0
            log(f"  [{done}/{len(todo)}] {el / done:.2f}s/video "
                f"ETA {(len(todo) - done) * el / done / 60:.1f}min")
    log(f"shard {a.shard} DONE in {(time.time() - t0) / 60:.1f} min")
    if a.shard == 0:
        with open(os.path.join(OUT, "_manifest.json"), "w") as f:
            json.dump({"algos": algos, "budgets": S.BUDGETS, "nv": NV, "nf": NF}, f, indent=1)


if __name__ == "__main__":
    main()
