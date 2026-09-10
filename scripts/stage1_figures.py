"""Stage 1 figures.

Everything is drawn in the coordinate system that was verified in Stage 0:
pooled cell (r, c) covers pixels [28r, 28r+28) x [28c, 28c+28) of the 384x384 tensor the
encoder actually sees, and flat index j = 13r + c. That tensor is an anisotropic resize of
the source frame with no crop or pad, so a cell also corresponds to a fixed fractional
region of the original frame.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, "/root/FlashVID")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
IN_DIR = f"{W}/runs/stage1"
FIG_DIR = f"{W}/results/figures"
VIDEO_DIR = f"{W}/data/videos_short"
GRID, CELL, ACTIVE, NV = 13, 28, 364, 169
RATIOS = [0.10, 0.15, 0.20, 0.25]
TAGS = {r: f"R{int(r*100):03d}_attn_div_v2" for r in RATIOS}
RINGS = np.array([min(r, c, GRID - 1 - r, GRID - 1 - c)
                  for r in range(GRID) for c in range(GRID)])
DPI = 130


def tag_of(r):
    return TAGS[r]


def load_native(tag, synthetic=False):
    """Per-video (169,) non-selection rate for one tag, native arm."""
    rates, vids, kbars = [], [], []
    for fn in sorted(os.listdir(IN_DIR)):
        if not fn.endswith(".npz") or fn.startswith("."):
            continue
        z = np.load(os.path.join(IN_DIR, fn), allow_pickle=False)
        if str(z["arm"]) != "native":
            continue
        if (str(z["synthetic"]) != "") != synthetic:
            continue
        if f"keep_{tag}" not in z.files:
            continue
        meta = z[f"meta_{tag}"]
        nf, k = int(meta[0]), int(meta[2])
        pos = z[f"keep_{tag}"].astype(np.int64) % NV
        cnt = np.bincount(pos, minlength=NV).astype(np.float64)
        rates.append(1.0 - cnt / nf)
        kbars.append(float(z[f"kpf_{tag}"].mean()))
        vids.append(str(z["video_id"]))
    return np.array(vids), np.array(rates), np.array(kbars)


def load_arm_map(tag, arm, synthetic=False):
    acc = []
    for fn in sorted(os.listdir(IN_DIR)):
        if not fn.endswith(".npz") or fn.startswith("."):
            continue
        z = np.load(os.path.join(IN_DIR, fn), allow_pickle=False)
        if str(z["arm"]) != arm or (str(z["synthetic"]) != "") != synthetic:
            continue
        if f"keep_{tag}" not in z.files:
            continue
        meta = z[f"meta_{tag}"]
        nf = int(meta[0])
        cnt = np.bincount(z[f"keep_{tag}"].astype(np.int64) % NV, minlength=NV)
        acc.append(1.0 - cnt / nf)
    return np.array(acc).mean(0) if acc else None


def as_grid(v):
    return np.asarray(v).reshape(GRID, GRID)


# ---------------------------------------------------------------- figure 1: histogram
def fig_histogram(out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for ax, r in zip(axes.ravel(), RATIOS):
        _, rates, kbars = load_native(tag_of(r))
        m = rates.mean(0)
        base = 1.0 - kbars.mean() / NV
        colors = plt.cm.viridis(RINGS / RINGS.max())
        ax.bar(np.arange(NV), m, color=colors, width=1.0)
        ax.axhline(base, color="crimson", lw=1.6, ls="--",
                   label=f"uniform baseline 1-k/169 = {base:.4f}")
        ax.set_xlim(-1, NV)
        lo, hi = m.min(), m.max()
        pad = (hi - lo) * 0.15 + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"retention_ratio = {r:.2f}   (k = {kbars.mean():.1f} of 169 kept/frame)")
        ax.set_xlabel("flat token index  j = 13·row + col")
        ax.set_ylabel("non-selection frequency")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.15, axis="y")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(0, RINGS.max()))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("ring index (0 = frame border, 6 = centre)")
    fig.suptitle("ADTS non-selection frequency by token index — Video-MME short, "
                 "300 videos x 64 frames, bar colour = distance from frame border",
                 fontsize=12)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_rate_distribution(out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in RATIOS:
        _, rates, kbars = load_native(tag_of(r))
        m = rates.mean(0)
        ax.hist(m, bins=40, histtype="step", lw=1.8,
                label=f"R={r:.2f}  (baseline {1 - kbars.mean()/NV:.3f})")
    ax.set_xlabel("per-position non-selection frequency")
    ax.set_ylabel("number of the 169 positions")
    ax.set_title("Spread of non-selection frequency across positions\n"
                 "(a position-blind algorithm would give a single spike)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: heatmaps
def fig_heatmaps(out):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for i, r in enumerate(RATIOS):
        _, rates, kbars = load_native(tag_of(r))
        m = rates.mean(0)
        base = 1.0 - kbars.mean() / NV

        ax = axes[0, i]
        im = ax.imshow(as_grid(m), cmap="magma", interpolation="nearest")
        ax.set_title(f"R = {r:.2f}\nnon-selection rate (baseline {base:.3f})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, i]
        # Relative log-odds against the uniform baseline. Absolute rates all sit near 0.9, so
        # the raw map has almost no dynamic range; the log-odds view is what makes the spatial
        # structure legible. The colour scale is set from the 2nd-98th percentile because a
        # handful of always-selected register cells reach -7 to -11 and would otherwise
        # flatten every other cell to white.
        p = np.clip(m, 1e-4, 1 - 1e-4)
        rel = np.log(p / (1 - p)) - np.log(base / (1 - base))
        v = max(np.percentile(np.abs(rel), 98), 1e-3)
        im = ax.imshow(as_grid(rel), cmap="RdBu_r", vmin=-v, vmax=v,
                       interpolation="nearest")
        g = as_grid(rel)
        for rr in range(GRID):
            for cc in range(GRID):
                if abs(g[rr, cc]) > v:
                    ax.text(cc, rr, f"{g[rr, cc]:.0f}", ha="center", va="center",
                            fontsize=6, color="k",
                            bbox=dict(fc="w", ec="none", alpha=.75, pad=.6))
        ax.set_title(f"relative log-odds (scale clipped to ±{v:.2f})\n"
                     "red = dropped more than chance; boxed = off-scale", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.ravel():
        ax.set_xticks(range(0, GRID, 2))
        ax.set_yticks(range(0, GRID, 2))
        ax.set_xlabel("col")
        ax.set_ylabel("row")
    fig.suptitle("ADTS non-selection heat maps in frame space — 13x13 pooled grid, "
                 "cell (r,c) = pixels [28r,28r+28) x [28c,28c+28) of the 384x384 input",
                 fontsize=12)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_arms(out, tag):
    panels = [("native (real)", load_arm_map(tag, "native", False)),
              ("hflip (real)", load_arm_map(tag, "hflip", False)),
              ("shift +3,+3 (real)", load_arm_map(tag, "shift", False)),
              ("native (synthetic)", load_arm_map(tag, "native", True)),
              ("hflip (synthetic)", load_arm_map(tag, "hflip", True)),
              ("shift (synthetic)", load_arm_map(tag, "shift", True))]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (name, m) in zip(axes.ravel(), panels):
        if m is None:
            ax.axis("off")
            continue
        base = m.mean()
        p = np.clip(m, 1e-4, 1 - 1e-4)
        rel = np.log(p / (1 - p)) - np.log(base / (1 - base))
        v = np.abs(rel).max()
        im = ax.imshow(as_grid(rel), cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(0, GRID, 3))
        ax.set_yticks(range(0, GRID, 3))
    fig.suptitle("Control arms — if the map is a property of CONTENT it should move with the\n"
                 "transform (mirror / roll by 3 cells); if of POSITION it should stay put",
                 fontsize=12)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_attribution(out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for meth, label in [("attn", "attention only (ATS)"),
                        ("div", "diversity only (DivPrune)"),
                        ("attn_div", "ADTS v1 (CLS attn calibration)"),
                        ("attn_div_v2", "ADTS v2 (paper's full method)")]:
        tag = f"R010_{meth}"
        try:
            _, rates, kbars = load_native(tag)
        except Exception:  # noqa: BLE001
            continue
        if not len(rates):
            continue
        m = rates.mean(0)
        base = 1.0 - kbars.mean() / NV
        p = np.clip(m, 1e-4, 1 - 1e-4)
        rel = np.log(p / (1 - p)) - np.log(base / (1 - base))
        prof = [rel[RINGS == k].mean() for k in range(7)]
        ax.plot(range(7), prof, marker="o", lw=2, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("ring index (0 = frame border  →  6 = centre)")
    ax.set_ylabel("mean relative log-odds of non-selection")
    ax.set_title("Where does the positional structure come from?\n"
                 "ADTS ingredients ablated at R = 0.10")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- figure 3: per-patch frame overlays
def get_display_frames(video_id, idxs):
    """Return the exact 384x384 tensors the encoder saw, as displayable RGB."""
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    from adts_harness import load_video_frames

    ip = SigLipImageProcessor()
    frames = load_video_frames(f"{VIDEO_DIR}/{video_id}.mp4", 64)
    px = ip.preprocess(frames, return_tensors="pt")["pixel_values"].numpy()
    px[..., ACTIVE:, :] = 0.0          # same border blanking the collector applied
    px[..., :, ACTIVE:] = 0.0
    img = np.clip(px * 0.5 + 0.5, 0, 1)   # SigLIP mean = std = 0.5
    return {i: np.transpose(img[i], (1, 2, 0)) for i in idxs}


def selected_cells(video_id, tag, frame_idx):
    z = np.load(f"{IN_DIR}/{video_id}__native.npz", allow_pickle=False)
    meta = z[f"meta_{tag}"]
    nf, k = int(meta[0]), int(meta[2])
    keep = z[f"keep_{tag}"].astype(np.int64).reshape(nf, k)
    return np.unique(keep[frame_idx] % NV)


def draw_overlay(ax, img, sel, title):
    """Original frame, kept cells at full brightness, dropped cells veiled and hatched."""
    shown = img.copy()
    veil = np.ones((GRID, GRID), bool)
    veil.ravel()[sel] = False
    for r in range(GRID):
        for c in range(GRID):
            if veil[r, c]:
                blk = shown[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]
                grey = blk.mean(axis=2, keepdims=True)
                # Keep most of the luminance so the scene stays readable; pull two thirds of
                # the way to greyscale so the kept cells are the only saturated regions.
                shown[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = \
                    0.62 * (0.35 * blk + 0.65 * grey) + 0.10
    ax.imshow(shown, interpolation="nearest")
    for r in range(GRID):
        for c in range(GRID):
            if veil[r, c]:
                ax.add_patch(Rectangle((c * CELL - .5, r * CELL - .5), CELL, CELL,
                                       fill=False, hatch="///", edgecolor=(1, 1, 1, .18),
                                       lw=0))
            else:
                ax.add_patch(Rectangle((c * CELL - .5, r * CELL - .5), CELL, CELL,
                                       fill=False, edgecolor="#39FF14", lw=1.6))
    ax.set_xlim(-.5, ACTIVE - .5)
    ax.set_ylim(ACTIVE - .5, -.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def fig_frames_examples(out, video_ids, frame_idx, r):
    tag = tag_of(r)
    fig, axes = plt.subplots(1, len(video_ids), figsize=(4.1 * len(video_ids), 4.7))
    for ax, vid in zip(np.atleast_1d(axes), video_ids):
        img = get_display_frames(vid, [frame_idx])[frame_idx]
        sel = selected_cells(vid, tag, frame_idx)
        draw_overlay(ax, img, sel, f"{vid}\nframe {frame_idx} — {len(sel)}/169 kept")
    fig.suptitle(f"Per-patch ADTS decision at retention_ratio = {r:.2f}   "
                 "(green outline = kept verbatim by ADTS; dimmed + hatched = handed to TSTM)",
                 fontsize=12)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_frame_across_ratios(out, video_id, frame_idx):
    img = get_display_frames(video_id, [frame_idx])[frame_idx]
    fig, axes = plt.subplots(1, len(RATIOS), figsize=(4.1 * len(RATIOS), 4.9))
    for ax, r in zip(axes, RATIOS):
        sel = selected_cells(video_id, tag_of(r), frame_idx)
        draw_overlay(ax, img, sel, f"R = {r:.2f} — {len(sel)}/169 kept")
    fig.suptitle(f"Same frame, four retention ratios — {video_id}, frame {frame_idx}\n"
                 "green outline = kept verbatim by ADTS; dimmed + hatched = handed to TSTM",
                 fontsize=12)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_register_cells(out):
    """The handful of cells ADTS keeps in almost every frame of almost every video.

    These dominate the log-odds map and are the clearest signature of high-norm
    register/sink tokens: a token whose embedding is an outlier is far from everything else
    in cosine distance, so a max-min diversity rule selects it unconditionally.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _, rates, kbars = load_native(tag_of(0.10))
    m = rates.mean(0)
    order = np.argsort(m)
    top = order[:14]
    labels = [f"({j//GRID},{j%GRID})" for j in top]
    axes[0].bar(range(len(top)), 1.0 - m[top], color="#2c7fb8")
    axes[0].axhline(kbars.mean() / NV, color="crimson", ls="--", lw=1.5,
                    label=f"chance {kbars.mean()/NV:.3f}")
    axes[0].set_xticks(range(len(top)))
    axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("selection rate (fraction of frames kept)")
    axes[0].set_xlabel("pooled cell (row, col)")
    axes[0].set_title("14 most-selected cells at R = 0.10\n"
                      "(only 16 of 169 cells can be kept per frame)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2, axis="y")

    for r in RATIOS:
        _, rr, kb = load_native(tag_of(r))
        s = np.sort(1.0 - rr.mean(0))[::-1]
        axes[1].plot(s, lw=1.8, label=f"R={r:.2f} (chance {kb.mean()/NV:.3f})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cells, sorted by selection rate")
    axes[1].set_ylabel("selection rate (log scale)")
    axes[1].set_title("Selection rate is extremely unevenly distributed\n"
                      "across the 169 positions")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2, which="both")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=32)
    ap.add_argument("--n-examples", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    vids, _, _ = load_native(tag_of(0.10))
    # Content-blind, deterministic pick: evenly spaced in the sorted id list.
    picks = [vids[int(round(i * (len(vids) - 1) / (a.n_examples - 1)))]
             for i in range(a.n_examples)]
    print("example videos:", list(picks), flush=True)

    jobs = [
        ("fig1_histogram_by_index.png", lambda p: fig_histogram(p)),
        ("fig2_rate_distribution.png", lambda p: fig_rate_distribution(p)),
        ("fig3_heatmaps_by_ratio.png", lambda p: fig_heatmaps(p)),
        ("fig4_control_arms.png", lambda p: fig_arms(p, tag_of(0.10))),
        ("fig5_attribution_rings.png", lambda p: fig_attribution(p)),
        ("fig10_register_cells.png", lambda p: fig_register_cells(p)),
        ("fig6_frames_R010.png", lambda p: fig_frames_examples(p, picks, a.frame, 0.10)),
        ("fig7_frames_R025.png", lambda p: fig_frames_examples(p, picks, a.frame, 0.25)),
        ("fig8_frame_across_ratios.png",
         lambda p: fig_frame_across_ratios(p, picks[0], a.frame)),
        ("fig9_frame_across_ratios_b.png",
         lambda p: fig_frame_across_ratios(p, picks[2], a.frame)),
    ]
    for name, fn in jobs:
        path = os.path.join(FIG_DIR, name)
        try:
            fn(path)
            print("wrote", path, flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", name, exc, flush=True)

    with open(f"{FIG_DIR}/example_videos.json", "w") as fh:
        json.dump({"videos": list(picks), "frame": a.frame}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
