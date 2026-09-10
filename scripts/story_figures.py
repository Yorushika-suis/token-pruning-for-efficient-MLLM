"""Figures for story.md — redrawn for a reader with no machine-learning background.

Different goals from the technical figures: no parameter names on the axes, no log-odds, no
ring indices. Every panel has to make one point that survives a five-second glance, and the
labels are in plain Chinese. Where the technical version showed a quantity, this version
shows the comparison that quantity was there to support.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
IN_DIR = f"{W}/runs/stage1"
OUT = f"{W}/results/figures/story"
VIDEO_DIR = f"{W}/data/videos_short"
G, CELL, ACTIVE, NV = 13, 28, 364, 169
BUDGETS = [(0.10, 16), (0.15, 23), (0.20, 31), (0.25, 38)]
DPI = 130

KEEP = "#00B140"     # 留下
DROP = "#C0C4CC"     # 丢掉
HOT = "#E8422F"      # 强调
COOL = "#2E6FD9"

# 「留下得多」用暖色，「留下得少」用冷色，中间过渡到浅灰，方便一眼看出偏离
CMAP = LinearSegmentedColormap.from_list(
    "keep", ["#F2F4F7", "#FFE082", "#FF8A3D", "#E8422F", "#8B0000"])


def tag(r):
    return f"R{int(r * 100):03d}_attn_div_v2"


def load_rates(t, arm="native", synthetic=False):
    """每个视频、每个格子被留下的比例。"""
    out = []
    for fn in sorted(os.listdir(IN_DIR)):
        if not fn.endswith(".npz") or fn.startswith("."):
            continue
        z = np.load(os.path.join(IN_DIR, fn))
        if str(z["arm"]) != arm or (str(z["synthetic"]) != "") != synthetic:
            continue
        if f"keep_{t}" not in z.files:
            continue
        nf = int(z[f"meta_{t}"][0])
        c = np.bincount(z[f"keep_{t}"].astype(np.int64) % NV, minlength=NV)
        out.append(c / nf)
    return np.array(out)


def grid(v):
    return np.asarray(v).reshape(G, G)


def frame_of(video_id, idx):
    sys.path.insert(0, "/root/FlashVID")
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from decord import VideoReader, cpu

    vr = VideoReader(f"{VIDEO_DIR}/{video_id}.mp4", ctx=cpu(0))
    ids = np.linspace(0, len(vr) - 1, 64, dtype=int).tolist()
    fr = vr.get_batch(ids).asnumpy()
    px = SigLipImageProcessor().preprocess(fr, return_tensors="pt")["pixel_values"].numpy()
    img = np.clip(px[idx] * 0.5 + 0.5, 0, 1)
    return np.transpose(img, (1, 2, 0))[:ACTIVE, :ACTIVE]


def kept_cells(video_id, t, idx):
    z = np.load(f"{IN_DIR}/{video_id}__native.npz")
    nf, k = int(z[f"meta_{t}"][0]), int(z[f"meta_{t}"][2])
    return np.unique(z[f"keep_{t}"].astype(np.int64).reshape(nf, k)[idx] % NV)


def paint(ax, img, sel=None, gridlines=True):
    show = img.copy()
    if sel is not None:
        mask = np.ones((G, G), bool)
        mask.ravel()[sel] = False
        for r in range(G):
            for c in range(G):
                if mask[r, c]:
                    b = show[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                    g = b.mean(axis=2, keepdims=True)
                    # 留住大部分亮度和一部分颜色：画面要认得出来，绿框要跳出来
                    show[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = \
                        0.78 * (0.5 * b + 0.5 * g) + 0.13
    ax.imshow(show, interpolation="nearest")
    if gridlines:
        for i in range(G + 1):
            ax.axhline(i * CELL - .5, color="w", lw=.6, alpha=.5)
            ax.axvline(i * CELL - .5, color="w", lw=.6, alpha=.5)
    if sel is not None:
        for j in sel:
            ax.add_patch(Rectangle(((j % G) * CELL - .5, (j // G) * CELL - .5), CELL, CELL,
                                   fill=False, edgecolor=KEEP, lw=2.4))
    ax.set_xlim(-.5, ACTIVE - .5)
    ax.set_ylim(ACTIVE - .5, -.5)
    ax.set_xticks([])
    ax.set_yticks([])


def ticks(ax):
    """给 13x13 图加行列刻度：正文按“第几行第几列”点名，图上必须数得出来。"""
    pos = [0, 4, 8, 12]
    ax.set_xticks(pos); ax.set_xticklabels([str(p + 1) for p in pos], fontsize=10)
    ax.set_yticks(pos); ax.set_yticklabels([str(p + 1) for p in pos], fontsize=10)
    ax.set_xlabel("第几列", fontsize=10)
    ax.set_ylabel("第几行", fontsize=10)


def finish(fig, title, path, top=0.84, y=0.97, fontsize=19):
    """统一收尾：先排版再放大标题，避免主标题压到子标题上。"""
    fig.tight_layout()
    fig.subplots_adjust(top=top)
    fig.suptitle(title, fontsize=fontsize, y=y)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- 1 切块与取舍
def f01(vid, idx):
    img = frame_of(vid, idx)
    sel = kept_cells(vid, tag(0.10), idx)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 6.4))
    paint(ax[0], img)
    ax[0].set_title("AI 眼里的一帧画面\n被切成 13×13 = 169 个小方块", fontsize=13.5, pad=10)
    paint(ax[1], img, sel)
    ax[1].set_title(f"但它只来得及看清 {len(sel)} 块（绿框）\n其余 {169-len(sel)} 块被打包压缩",
                    fontsize=13.5, pad=10)
    finish(fig, "视频太大，AI 必须做取舍", f"{OUT}/01_切块与取舍.png", top=0.80)


# ----------------------------------------------------------------- 2 公平 vs 实际
def f02():
    rates = load_rates(tag(0.10)).mean(0)
    fair = np.full(NV, 16 / 169)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6))
    vmax = 1.0
    for a, m, ttl in [(ax[0], fair, "假如 AI 不看位置只看内容\n每块被留下的机会都该是 9.5%"),
                      (ax[1], rates, "实际测出来的样子\n有几块几乎每次都被留下")]:
        im = a.imshow(grid(m), cmap=CMAP, vmin=0, vmax=vmax, interpolation="nearest")
        a.set_title(ttl, fontsize=13.5, pad=10)
        ticks(a)
        for sp in a.spines.values():
            sp.set_visible(False)
    ax[1].annotate("第11行第2列\n19200 帧一次没漏", xy=(1, 10), xytext=(5.2, 8.0),
                   fontsize=11, color="k", weight="bold",
                   bbox=dict(fc="w", ec="k", alpha=.85, boxstyle="round,pad=.35"),
                   arrowprops=dict(arrowstyle="->", lw=2, color="k"))
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("这一块被留下的比例", fontsize=12)
    cb.set_ticks([0, .25, .5, .75, 1.0])
    cb.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.subplots_adjust(top=0.80)
    fig.suptitle("同样一张“座位表”，公平的样子和真实的样子差得很远", fontsize=18, y=0.96)
    fig.savefig(f"{OUT}/02_公平与实际.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- 3 常客排行榜
def f03():
    rates = load_rates(tag(0.10)).mean(0)
    order = np.argsort(rates)[::-1][:12]
    labels = [f"第{j//G+1}行 第{j%G+1}列" for j in order]
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1])
    a0 = fig.add_subplot(gs[0])
    y = np.arange(len(order))[::-1]
    a0.barh(y, rates[order] * 100, color=[HOT if v > .5 else "#F0A868" for v in rates[order]])
    a0.axvline(9.5, color=COOL, ls="--", lw=2, label="公平情况下应该是 9.5%")
    for yy, v in zip(y, rates[order] * 100):
        a0.text(v + 1.2, yy, f"{v:.0f}%", va="center", fontsize=11)
    a0.set_yticks(y)
    a0.set_yticklabels(labels, fontsize=11)
    a0.set_xlabel("这一块被留下的比例", fontsize=12)
    a0.set_xlim(0, 112)
    a0.xaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    a0.legend(fontsize=11, loc="lower right")
    a0.set_title("最受宠的 12 个位置", fontsize=14)
    for s in ("top", "right"):
        a0.spines[s].set_visible(False)

    a1 = fig.add_subplot(gs[1])
    m = np.zeros(NV)
    m[order[:6]] = 1
    a1.imshow(grid(m), cmap="Reds", vmin=0, vmax=1.4, interpolation="nearest")
    a1.set_aspect("equal")
    for j in order[:6]:
        a1.add_patch(Rectangle((j % G - .5, j // G - .5), 1, 1,
                               fill=False, edgecolor=HOT, lw=2.5))
    for i in range(G + 1):
        a1.axhline(i - .5, color="w", lw=.7)
        a1.axvline(i - .5, color="w", lw=.7)
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("它们在画面上的位置\n（几乎全挤在左边缘和角落）", fontsize=13)
    for s in a1.spines.values():
        s.set_visible(False)
    a0.text(.98, .06, "红＝超过一半的画面都留它    橙＝超过公平线但不到一半",
            transform=a0.transAxes, ha="right", fontsize=9.5, color="#555")
    finish(fig, "169 个位置里，有 3 个几乎场场不缺席，还有一批远超公平线",
           f"{OUT}/03_常客排行榜.png", top=0.82, fontsize=17)


# ----------------------------------------------------------------- 4 镜像实验
def f04():
    nat = load_rates(tag(0.10), "native").mean(0)
    flip = load_rates(tag(0.10), "hflip").mean(0)
    mirrored = grid(nat)[:, ::-1].ravel()
    r_stay = np.corrcoef(flip, nat)[0, 1]
    r_follow = np.corrcoef(flip, mirrored)[0, 1]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.6))
    for a, m, ttl in [
            (ax[0], nat, "① 原始画面\n测出来的偏好"),
            (ax[1], flip, "② 把画面左右翻转后\n重新测的偏好"),
            (ax[2], mirrored, "③ 假如偏好跟着画面走\n② “本应” 长成的样子")]:
        im = a.imshow(grid(m), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        a.set_title(ttl, fontsize=13.5, pad=10)
        ticks(a)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78, bottom=0.16)
    fig.suptitle("最关键的一次实验：把画面翻过来，偏好却纹丝不动", fontsize=19, y=0.96)
    fig.text(.5, .065, f"② 和 ① 有多像：{r_stay:.2f}  —— 几乎一模一样",
             ha="center", fontsize=14, color=HOT, weight="bold")
    fig.text(.5, .015, f"② 和 ③ 有多像：{r_follow:.2f}  —— 基本不像",
             ha="center", fontsize=14, color=COOL)
    fig.savefig(f"{OUT}/04_镜像实验.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- 5 雪花实验
def f05():
    rng = np.random.default_rng(1000)
    noise = rng.integers(0, 256, (ACTIVE, ACTIVE, 3)).astype(np.float32) / 255
    real = load_rates(tag(0.10), "native").mean(0)
    syn = load_rates(tag(0.10), "native", synthetic=True).mean(0)
    r = np.corrcoef(real, syn)[0, 1]
    top_r = set(np.argsort(real)[::-1][:6].tolist())
    top_s = set(np.argsort(syn)[::-1][:6].tolist())
    same = len(top_r & top_s)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 6.4))
    im = ax[0].imshow(grid(real), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax[0].set_title("① 看 300 个真实视频时\nAI 的偏好（就是前面那张）", fontsize=13)
    ticks(ax[0])
    ax[1].imshow(noise, interpolation="nearest")
    for i in range(G + 1):
        ax[1].axhline(i * CELL - .5, color="w", lw=.6, alpha=.5)
        ax[1].axvline(i * CELL - .5, color="w", lw=.6, alpha=.5)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("② 我们喂进去的画面\n纯随机雪花，哪块都不特别", fontsize=13)
    ax[2].imshow(grid(syn), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title("③ 看雪花时\nAI 的偏好", fontsize=13)
    ticks(ax[2])
    for a in (ax[0], ax[2]):
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78, bottom=0.14)
    fig.suptitle("画面里什么都没有，偏好依然在", fontsize=19, y=0.95)
    fig.text(.5, .055, f"③ 和 ① 有多像：{r:.2f}", ha="center", fontsize=14,
             color=HOT, weight="bold")
    fig.text(.5, .005, f"两边最受宠的前 6 名，有 {same} 个是同样的格子",
             ha="center", fontsize=13, color=COOL)
    fig.savefig(f"{OUT}/05_雪花实验.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- 6 猜想反了
def f06():
    rings = np.array([min(r, c, G - 1 - r, G - 1 - c) for r in range(G) for c in range(G)])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    names = ["最外圈\n(画面边框)", "第2圈", "第3圈", "第4圈", "第5圈", "第6圈", "正中心"]
    for (r, k), col in zip(BUDGETS, ["#8B0000", HOT, "#F0A868", "#F5C77E"]):
        m = load_rates(tag(r)).mean(0)
        prof = [100 * (1 - m[rings == i].mean()) for i in range(7)]
        ax.plot(range(7), prof, marker="o", lw=2.6, color=col, label=f"只留 {k} 块")
    ax.set_xticks(range(7))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("这一圈被丢掉的比例", fontsize=12)
    ax.yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    ax.legend(fontsize=11)
    ax.grid(alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate("最容易被丢的其实是这里，\n不是最外圈", xy=(2, 0), xycoords=("data", "axes fraction"),
                xytext=(3.4, .18), textcoords=("data", "axes fraction"), fontsize=12,
                color=COOL, arrowprops=dict(arrowstyle="->", color=COOL, lw=1.8))
    finish(fig, "我们原本猜错了方向：画面最边上并不是最容易被丢的",
           f"{OUT}/06_猜想反了.png", top=0.88, fontsize=17)


# ----------------------------------------------------------------- 7 注意力偏心
def f07():
    rings = np.array([min(r, c, G - 1 - r, G - 1 - c) for r in range(G) for c in range(G)])
    acc = []
    for fn in sorted(os.listdir(IN_DIR)):
        if not fn.endswith(".npz") or fn.startswith("."):
            continue
        z = np.load(os.path.join(IN_DIR, fn))
        if str(z["arm"]) != "native" or str(z["synthetic"]) != "":
            continue
        acc.append(z["cls_attn"].astype(np.float64).mean(0))
    m = np.array(acc).mean(0)
    prof = np.array([m[rings == i].mean() for i in range(7)])
    prof = prof / prof[3]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    names = ["最外圈\n(画面边框)", "第2圈", "第3圈", "第4圈", "第5圈", "第6圈", "正中心"]
    cols = [HOT] + ["#9FB3C8"] * 6
    ax.bar(range(7), prof, color=cols)
    for i, v in enumerate(prof):
        ax.text(i, v + .04, f"{v:.2f}×", ha="center", fontsize=11.5)
    ax.axhline(1.0, color=COOL, ls="--", lw=1.8)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("关注度（以第 4 圈为 1 倍）", fontsize=12)
    ax.set_ylim(0, prof.max() * 1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "追根溯源：还没开始挑选，AI 对画面边框的关注就已经高出近 3 倍",
           f"{OUT}/07_注意力偏心.png", top=0.88, fontsize=16.5)


# ----------------------------------------------------------------- 8 预算变大
def f08():
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 5.2))
    for a, (r, k) in zip(ax, BUDGETS):
        m = load_rates(tag(r)).mean(0)
        im = a.imshow(grid(m), cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"每帧只留 {k} 块", fontsize=14)
        ticks(a)
        for sp in a.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.02, pad=.01)
    cb.set_ticks([0, .5, 1.0]); cb.set_ticklabels(["0%", "50%", "100%"])
    cb.set_label("被留下的比例", fontsize=11)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("给的名额越多，中间区域越吃香 —— 但左边缘那几块从头到尾都在",
                 fontsize=17, y=0.95)
    fig.savefig(f"{OUT}/08_预算变大.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- 9 同帧跨预算
def f09(vid, idx):
    img = frame_of(vid, idx)
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 5.4))
    for a, (r, k) in zip(ax, BUDGETS):
        sel = kept_cells(vid, tag(r), idx)
        paint(a, img, sel)
        a.set_title(f"只留 {len(sel)} 块", fontsize=14)
    finish(fig, "同一帧画面，名额从 16 块放宽到 38 块：新增的绿框往中间长，"
                "左边缘那一竖列始终占着", f"{OUT}/09_同帧跨预算.png", top=0.80, fontsize=16.5)


# ----------------------------------------------------------------- 10 名额被瓜分
def f10():
    ks, pct, blocks = [], [], []
    for r, k in BUDGETS:
        m = load_rates(tag(r)).mean(0)
        top = np.argsort(m)[::-1][:10]
        ks.append(k); blocks.append(m[top].sum()); pct.append(100 * m[top].sum() / k)
    x = np.arange(len(ks))
    labels = [f"每帧\n留 {k} 块" for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax[0].bar(x, pct, color=[HOT, "#F0A868", "#F5C77E", "#FAE0B5"])
    for i, v in enumerate(pct):
        ax[0].text(i, v + .8, f"{v:.0f}%", ha="center", fontsize=13, weight="bold")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=11)
    ax[0].set_ylabel("常客占掉的名额比例", fontsize=12)
    ax[0].yaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
    ax[0].set_ylim(0, max(pct) * 1.3)
    ax[0].set_title("按比例看：好像在下降", fontsize=13.5)

    ax[1].bar(x, blocks, color=COOL)
    for i, v in enumerate(blocks):
        ax[1].text(i, v + .12, f"{v:.1f} 块", ha="center", fontsize=13, weight="bold")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=11)
    ax[1].set_ylabel("常客实际拿走的块数", fontsize=12)
    ax[1].set_ylim(0, max(blocks) * 1.3)
    ax[1].set_title("按块数看：其实一路在涨", fontsize=13.5)

    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    finish(fig, "同一批常客，比例在降，拿走的块数却在涨 —— 降的只是分母",
           f"{OUT}/10_名额被瓜分.png", top=0.82, fontsize=17)


# ----------------------------------------------------------------- 0 座位表核对
def f00():
    # 来自 Stage 0 的 A3b / A3c：第一次用整套流程测只对上 144，
    # 失手的信号被少数几个位置吸走；换成只看切块那一步后 169 全中。
    sticky = [((1, 2), 7), ((1, 9), 3), ((11, 2), 3), ((2, 9), 2),
              ((3, 1), 2), ((3, 2), 2), ((4, 9), 2), ((6, 5), 2)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4))

    m = np.zeros((G, G))
    for (r, c), w in sticky:
        m[r, c] = w
    im = ax[0].imshow(m, cmap="Reds", vmin=0, vmax=8, interpolation="nearest")
    for (r, c), w in sticky:
        ax[0].add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                  edgecolor=HOT, lw=2.2))
        ax[0].text(c, r, str(w), ha="center", va="center", fontsize=10,
                   weight="bold", color="w" if w > 4 else "k")
    ax[0].set_title("第一次测：169 个位置只对上 144 个\n红格里的数字＝它抢走了几个位置的信号",
                    fontsize=12.5, pad=10)
    ticks(ax[0])

    ax[1].imshow(np.ones((G, G)), cmap=LinearSegmentedColormap.from_list(
        "g", ["#7BD389", "#7BD389"]), interpolation="nearest")
    ax[1].set_title("换个笨办法再测：169 个全对上\n没放亮块的位置，反应严格是 0",
                    fontsize=12.5, pad=10)
    ticks(ax[1])
    for a in ax:
        for i in range(G + 1):
            a.axhline(i - .5, color="w", lw=.7)
            a.axvline(i - .5, color="w", lw=.7)
        for sp in a.spines.values():
            sp.set_visible(False)
    finish(fig, "动手之前：先把“座位表”核对清楚", f"{OUT}/00_座位表核对.png",
           top=0.80, fontsize=18)


# ----------------------------------------------------------------- 11 偏心来自哪一项
def f11():
    variants = [("只看关注度", "attn"), ("完整版去掉修正项", "attn_div"),
                ("完整版（论文方法）", "attn_div_v2"), ("只看差异大不大", "div")]
    names, vals = [], []
    for name, meth in variants:
        m = load_rates(f"R010_{meth}").mean(0)
        names.append(name)
        vals.append(np.sort(m)[::-1][:10].sum())
    fig, ax = plt.subplots(figsize=(10, 5.4))
    cols = [HOT, "#F0A868", COOL, "#7BD389"]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + .12, f"{v:.1f} 块", ha="center", fontsize=13, weight="bold")
    ax.axhline(16 * 10 / 169, color="#555", ls="--", lw=1.6,
               label="完全公平的话只该占 0.9 块")
    ax.set_ylabel("每帧 16 个名额里，前 10 个常客占掉几块", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.legend(fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    finish(fig, "把挑选规则拆开：偏心主要来自“看关注度”这一项",
           f"{OUT}/11_偏心来自哪一项.png", top=0.86, fontsize=17)


def main():
    os.makedirs(OUT, exist_ok=True)
    picks = json.load(open(f"{W}/results/figures/example_videos.json"))
    vid, idx = picks["videos"][0], picks["frame"]
    vid2 = picks["videos"][2]
    print("using", vid, vid2, "frame", idx, flush=True)
    f00(); print("00 ok", flush=True)
    f01(vid2, idx); print("01 ok", flush=True)
    f02(); print("02 ok", flush=True)
    f03(); print("03 ok", flush=True)
    f04(); print("04 ok", flush=True)
    f05(); print("05 ok", flush=True)
    f06(); print("06 ok", flush=True)
    f07(); print("07 ok", flush=True)
    f08(); print("08 ok", flush=True)
    f09(vid, idx); print("09 ok", flush=True)
    f10(); print("10 ok", flush=True)
    f11(); print("11 ok", flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
