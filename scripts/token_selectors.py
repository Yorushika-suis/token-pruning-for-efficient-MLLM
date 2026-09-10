"""统一 token universe 上的 token selection 算法库。

设计约定
--------
所有算法共享同一份离线特征（stage2_dump_features.py 产出），签名统一为

    fn(ctx: Ctx, k: int) -> np.ndarray, shape (F, k), int32, 帧内局部索引

即「每帧固定 k 个」的 P1 预算协议。需要跨帧自由分配预算的算法（P2 协议）另有
`GLOBAL_SELECTORS`，签名为 fn(ctx, K) -> (K,) int32 全局索引。

为什么固定 k 而不是固定 retention ratio R：FlashVID 的 k = ceil(ceil(169·R·expansion)·alpha)，
expansion=1.25 让 vision 侧实际保留率是 R×1.25，alpha=0.7 又让 ADTS 只拿 70% 预算。
不同算法若各自从自己的超参推 k，对比就不是等预算的。见 utils.py:46-47。

正确性锚点
----------
adts_v2 / adts_v1 / attn / div 四个是 FlashVID 官方实现的逐行重写，
用 Stage-1 的 990 个 npz 里的 keep_* 字段做回归门禁（见 verify_against_stage1）。
实测事实（来自源码侦察）：attn / div / attn_div 三个是严格帧内算法，整段跑与按段跑
bitwise 相同；只有 attn_div_v2 的 calibration_term2 是跨帧项，对 DySeg 分段敏感。
"""
import numpy as np
import torch

NV, GRID = 169, 13
NP729, SIDE = 729, 27


class Ctx:
    """一个视频的全部离线特征。惰性上采样到 float32（各算法内部都用 fp32）。"""

    __slots__ = ("pooled", "attn169", "_sig729", "_attn729", "vid", "_d169", "_d729")

    def __init__(self, npz, vid=""):
        self.vid = vid
        self.pooled = torch.from_numpy(np.ascontiguousarray(npz["pooled169"])).float()
        self.attn169 = torch.from_numpy(np.ascontiguousarray(npz["attn169"])).float()
        self._sig729 = npz["sig729"]
        self._attn729 = npz["attn729"]
        self._d169 = self._d729 = None

    @property
    def sig729(self):
        if not torch.is_tensor(self._sig729):
            self._sig729 = torch.from_numpy(np.ascontiguousarray(self._sig729)).float()
        return self._sig729

    @property
    def attn729(self):
        if not torch.is_tensor(self._attn729):
            self._attn729 = torch.from_numpy(np.ascontiguousarray(self._attn729)).float()
        return self._attn729

    @property
    def F(self):
        return self.pooled.shape[0]

    def dist169(self):
        """(F,169,169) 帧内余弦距离，缓存复用——多数算法都要它。"""
        if self._d169 is None:
            self._d169 = pairwise_cosine(self.pooled)
        return self._d169

    def dist729(self):
        if self._d729 is None:
            self._d729 = pairwise_cosine(self.sig729)
        return self._d729


def pairwise_cosine(x):
    """(B,N,D) -> (B,N,N) 余弦距离 1-cos，对角为 0。照 token_selection.py:14-25。"""
    xn = torch.nn.functional.normalize(x, p=2, dim=-1)
    d = 1.0 - torch.bmm(xn, xn.transpose(1, 2))
    d.diagonal(dim1=1, dim2=2).zero_()
    return d


# ============================================================ FlashVID 家族

def _fps_greedy(M, k, anti_reselect=True):
    """在列加权距离矩阵 M (B,N,N) 上做 max-min 贪心，返回 (B,k) 升序索引。

    逐行对照 token_selection.py:171-185（div）/ :113-125（v2）。

    有一处极易写错：`topk(M,2,dim=1,largest=False).values[:,1,:]`（= min_{i≠j} M[i,j]）
    **只用来选种子**。进入循环后官方实现每步都从头重算
    `min_dist = min over 已选行 s 的 M[s,j]`，并不把种子那个量继续递推。
    误把它递推下去会得到完全不同的选择——实测 token_overlap 只有 0.42。
    这里用等价的增量形式：run = min_{s∈S} M[s,·]，初值取 M[s0,·] 而非 topk 那个量。

    anti_reselect 对应 div/v1 的 :184 那句 scatter_(-1)；v2 缺这一句（缺陷 F1）。
    v2 的列权重 W_j 可为负，使 min_dist 整体可为负、0 不再是最小值，
    argmax 因而可能落回已选 token。此处忠实复现，不代为修补。
    """
    B, N, _ = M.shape
    keep = torch.zeros(B, k, dtype=torch.long)
    seed = M.topk(2, dim=1, largest=False).values[:, 1, :]       # 仅用于选种子
    keep[:, 0] = seed.argmax(dim=-1)
    if k == 1:
        return keep
    run = M.gather(1, keep[:, 0:1, None].expand(B, 1, N))[:, 0, :]   # min over {s0}
    for i in range(1, k):
        cur = run.scatter(1, keep[:, :i], -1.0) if anti_reselect else run
        keep[:, i] = cur.argmax(dim=-1)
        if i + 1 < k:
            nxt = M.gather(1, keep[:, i:i + 1, None].expand(B, 1, N))[:, 0, :]
            run = torch.minimum(run, nxt)
    return keep.sort(dim=1).values


def sel_div(ctx, k):
    """FlashVID 的逐帧 ``div`` 消融。

    它复用 max-min diversity/FPS 原语，但在 P1、169-token 帧内协议上运行，不是论文
    DivPrune 的完整预算与视频级执行协议。
    """
    return _fps_greedy(ctx.dist169(), k, anti_reselect=True).numpy().astype(np.int32)


def sel_attn(ctx, k):
    """纯注意力 top-k（FlashVID 的 `attn`；也是 ATS / VisionZip dominant 的 169 版）。"""
    return ctx.attn169.topk(k, dim=-1).indices.sort(dim=1).values.numpy().astype(np.int32)


def sel_adts_v1(ctx, k):
    """attn_div：距离矩阵按列乘以注意力，再做 FPS。token_selection.py:28-76。"""
    a = ctx.attn169 * 1e6                                        # :46 防 fp32 下溢，不改 argmax
    M = ctx.dist169() * a[:, None, :]                            # :51 权重挂在列 j 上
    return _fps_greedy(M, k, anti_reselect=True).numpy().astype(np.int32)


def dyseg(pooled, segment_threshold=0.9, min_segment_num=8):
    """DySeg：按相邻帧 mean-pooled 特征的余弦相似度切段。utils.py:211-245。

    v2 的 calibration_term2 是段级量，不复现分段就复现不了 v2。
    """
    f = torch.nn.functional.normalize(pooled.mean(dim=1), p=2, dim=-1)   # (F,D)
    sim = (f[:-1] * f[1:]).sum(-1)                                        # (F-1,)
    F = pooled.shape[0]
    cuts = (sim < segment_threshold).nonzero().flatten() + 1
    if len(cuts) < min_segment_num - 1:
        # 相似度最低的若干处强制切开，凑够 min_segment_num 段
        need = min_segment_num - 1
        cuts = (torch.topk(sim, min(need, len(sim)), largest=False).indices + 1)
        cuts = cuts.sort().values
    bounds = [0] + sorted(set(int(c) for c in cuts)) + [F]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def sel_adts_v2(ctx, k, segments=None):
    """attn_div_v2：距离 x term1(注意力) x term2(事件相关性)，再做 FPS。

    token_selection.py:79-130。实测恒等式：term2_{b,j} = <x_{b,j}, mu_seg>，
    mu_seg 是**该段内所有帧所有 token** 的平均向量，所以可以写成 feats @ feats.mean((0,1))，
    省掉论文实现里那个 (B,C,N) 的 einsum 中间量。
    term2 未归一化、可为负（实测约 30%）——W_j<0 时贪心从 min 静默翻转成 max，
    且这类 token 永远当不上种子。这是比已知 F1 更严重的缺陷，此处忠实复现。
    """
    if segments is None:
        segments = dyseg(ctx.pooled)
    out = np.zeros((ctx.F, k), dtype=np.int32)
    D = ctx.dist169()
    for s, e in segments:
        feats = ctx.pooled[s:e]                                   # (C,169,D)
        mu = feats.mean(dim=(0, 1))                               # 段级平均向量
        term1 = ctx.attn169[s:e] * 1e6                            # (C,169)
        term2 = feats @ mu                                        # (C,169) 可为负
        M = D[s:e] * term1[:, None, :] * term2[:, None, :]
        out[s:e] = _fps_greedy(M, k, anti_reselect=False).numpy()
    return out


# ============================================================ 对照基线
# 「朴素空间均匀采样常胜过复杂设计」是 arXiv 2502.11501 的核心论断，
# 不放这几条对照，任何「某算法更好」的结论都站不住。

def sel_random(ctx, k, seed=0):
    rng = np.random.default_rng(seed + hash(ctx.vid) % 2**31)
    return np.sort(np.stack([rng.choice(NV, k, replace=False) for _ in range(ctx.F)]),
                   axis=1).astype(np.int32)


def sel_grid_uniform(ctx, k):
    """空间上尽量均匀的固定格点。逐帧相同，是「纯位置、零内容」的极端对照。

    注意不能用 `linspace(0, 168, k)` 取 flat index —— 那在 13x13 上是光栅条纹，
    实测 ring5/ring6 选中率恒为 0，根本不是空间均匀。必须在二维上布点。
    「朴素空间均匀采样常胜过复杂设计」是 arXiv 2502.11501 的核心论断，
    这条对照写错就等于没做这个检验。
    """
    nr = int(np.floor(np.sqrt(k)))
    while nr > 1 and k // nr > GRID:
        nr -= 1
    nc = int(np.ceil(k / nr))
    rows = np.unique(np.round(np.linspace(0, GRID - 1, nr)).astype(int))
    cols = np.unique(np.round(np.linspace(0, GRID - 1, nc)).astype(int))
    pts = [r * GRID + c for r in rows for c in cols]
    if len(pts) > k:                                  # 行列乘积超了就在格点里等距抽
        pts = [pts[i] for i in np.round(np.linspace(0, len(pts) - 1, k)).astype(int)]
    pts = sorted(set(pts))
    if len(pts) < k:                                  # 不足则用离已选点最远的格补齐
        rc = np.stack([np.arange(NV) // GRID, np.arange(NV) % GRID], 1).astype(float)
        while len(pts) < k:
            d = np.linalg.norm(rc[:, None, :] - rc[pts][None, :, :], axis=-1).min(1)
            d[pts] = -1
            pts = sorted(pts + [int(d.argmax())])
    return np.tile(np.array(pts[:k], dtype=np.int32)[None, :], (ctx.F, 1))


def sel_spatial_stratified(ctx, k, seed=0):
    """把 13x13 切成 k 个近似等面积块，每块随机取一个——空间分层随机。"""
    rng = np.random.default_rng(seed + hash(ctx.vid) % 2**31)
    nb = int(np.ceil(np.sqrt(k)))
    rows = np.array_split(np.arange(GRID), nb)
    cols = np.array_split(np.arange(GRID), nb)
    cells = [[r * GRID + c for r in rr for c in cc] for rr in rows for cc in cols]
    out = np.zeros((ctx.F, k), dtype=np.int32)
    for f in range(ctx.F):
        order = rng.permutation(len(cells))
        picked = [int(rng.choice(cells[b])) for b in order[:k]]
        while len(picked) < k:                            # 块数不足时从剩余位置补
            rest = np.setdiff1d(np.arange(NV), picked)
            picked.append(int(rng.choice(rest)))
        out[f] = np.sort(picked)
    return out


REGISTRY = {
    # FlashVID 家族（正确性锚点）
    "adts_v2": sel_adts_v2,
    "adts_v1": sel_adts_v1,
    "attn": sel_attn,
    "FlashVID-div-ablation@P1U169": sel_div,
    # 对照
    "random": sel_random,
    "grid_uniform": sel_grid_uniform,
    "spatial_stratified": sel_spatial_stratified,
}

# k = ceil(ceil(169 * R * expansion) * alpha)，expansion=1.25 alpha=0.7
BUDGETS = {0.10: 16, 0.15: 23, 0.20: 31, 0.25: 38}


def verify_against_stage1(ctx, stage1_npz, tags=("R010_attn_div_v2", "R010_attn",
                                                 "R010_div", "R010_attn_div")):
    """回归门禁：重实现的选择结果 vs Stage-1 落盘的 keep_*。

    attn / div / attn_div 应当逐位一致；attn_div_v2 依赖 DySeg 分段，
    分段一旦不同就会偏，所以单独报告一致率而不硬性要求 100%。
    """
    fn_of = {"attn_div_v2": "adts_v2", "attn": "attn",
             "div": "FlashVID-div-ablation@P1U169",
             "attn_div": "adts_v1"}
    rep = {}
    for tag in tags:
        if f"keep_{tag}" not in stage1_npz:
            continue
        meth = tag.split("_", 1)[1]
        k = int(stage1_npz[f"meta_{tag}"][2])
        ref = stage1_npz[f"keep_{tag}"].reshape(ctx.F, k) % NV
        got = REGISTRY[fn_of[meth]](ctx, k)
        agree = float((np.sort(ref, 1) == got).all(axis=1).mean())
        rep[tag] = {"k": k, "frame_exact_rate": agree,
                    "token_overlap": float(np.mean([
                        len(set(ref[f]) & set(got[f])) / k for f in range(ctx.F)]))}
    return rep
