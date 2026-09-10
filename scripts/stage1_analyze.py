"""Stage 1 analysis: does ADTS non-selection frequency depend on spatial position?

Pre-registered before looking at any collected map:
  * unit of analysis  = video (frames within a DySeg segment are near-duplicates by
    construction -- DySeg cuts where cosine similarity drops below 0.9 -- so treating frames
    as independent would inflate the effective n by roughly F/S ~ 8x);
  * primary contrast  = ring 0 (border) minus rings 3-6 (interior), one-sided, ring 0 higher;
  * null              = within-video spatial label permutation. Permuting the 169 position
    labels inside a video, with the same permutation applied to all its frames, preserves
    every per-frame marginal exactly (each frame still selects its own k positions) and the
    within-video temporal dependence, and breaks only the association between position
    identity and selection. The "each frame draws k of 169 uniformly" null that suggests
    itself first is wrong for H1: it tests whether selection is non-uniform, which is
    trivially true for any content-driven algorithm, and says nothing about position.
  * effect size       = relative log-odds of NON-selection against logit(1 - kbar/169).
    Note the reference is the non-selection rate, not k/169; using the selection rate as the
    reference would offset the whole map by 2*logit(k/169) ~ -4.5.
  * multiplicity      = ring family first (gatekeeping), then BH-FDR over 169 positions.
"""
import argparse
import json
import os
import sys

import numpy as np

W = "/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial"
IN_DIR = f"{W}/runs/stage1"
OUT_DIR = f"{W}/results"
GRID = 13
NV = GRID * GRID
MAIN_TAG = "R010_attn_div_v2"
ARMS = ["native", "hflip", "shift"]
RINGS = np.array([min(r, c, GRID - 1 - r, GRID - 1 - c)
                  for r in range(GRID) for c in range(GRID)])
BORDER = RINGS == 0
INTERIOR = RINGS >= 3
RNG = np.random.default_rng(20260828)
N_PERM = 2000


def logit(p):
    return np.log(p / (1.0 - p))


def load_all():
    """Return {(video_id, arm): npz-dict-lite}. Only the fields analysis needs."""
    out = {}
    for fn in sorted(os.listdir(IN_DIR)):
        if not fn.endswith(".npz") or fn.startswith("."):
            continue
        z = np.load(os.path.join(IN_DIR, fn), allow_pickle=False)
        vid = str(z["video_id"])
        arm = str(z["arm"])
        rec = {"synthetic": str(z["synthetic"]), "keys": set(z.files)}
        for k in z.files:
            if k.startswith(("keep_", "kpf_", "meta_", "ties_", "margin_")) or \
               k in ("cls_attn", "covariates", "perm"):
                rec[k] = z[k]
        out[(vid, arm)] = rec
    return out


def per_video_counts(rec, tag):
    """(sel_count[169], n_frames, kbar) for one video/arm/tag.

    keep_* is frame-major: segment_compression is called on contiguous frame ranges in order,
    and each call contributes exactly k entries per frame.
    """
    keep = rec[f"keep_{tag}"].astype(np.int64)
    meta = rec[f"meta_{tag}"]
    nf, nv, k = int(meta[0]), int(meta[1]), int(meta[2])
    assert nv == NV, (nv, NV)
    assert keep.size == nf * k, (keep.size, nf, k)
    pos = keep % NV
    cnt = np.bincount(pos, minlength=NV).astype(np.float64)
    kpf = rec[f"kpf_{tag}"].astype(np.float64)   # distinct positions actually taken per frame
    return cnt, nf, float(kpf.mean())


def rel_logodds(cnt, nf, kbar):
    """Haldane-Anscombe corrected relative log-odds of NON-selection.

    Without the correction every cell a video never selects has rate exactly 1 and logit
    diverges -- and those are precisely the cells H1 is about.
    """
    nonsel = nf - cnt
    p = (nonsel + 0.5) / (nf + 1.0)
    ref = 1.0 - kbar / NV
    return logit(p) - logit(ref)


def build_matrix(data, tag, arm, synthetic=None):
    vids, mats, cnts, nfs, kbars = [], [], [], [], []
    for (vid, a), rec in sorted(data.items()):
        if a != arm:
            continue
        is_syn = rec["synthetic"] != ""
        if synthetic is True and not is_syn:
            continue
        if synthetic is False and is_syn:
            continue
        if f"keep_{tag}" not in rec:
            continue
        cnt, nf, kbar = per_video_counts(rec, tag)
        vids.append(vid)
        cnts.append(cnt)
        nfs.append(nf)
        kbars.append(kbar)
        mats.append(rel_logodds(cnt, nf, kbar))
    return (np.array(vids), np.array(mats), np.array(cnts),
            np.array(nfs), np.array(kbars))


def ring_contrast(mat):
    """Primary statistic: mean border effect minus mean interior effect, averaged over videos."""
    return float(mat[:, BORDER].mean() - mat[:, INTERIOR].mean())


def _rel_logodds_mat(cnts, nfs, kbars):
    """Vectorised rel_logodds over a stack of videos: (n_vid, NV) -> (n_vid, NV)."""
    p = (nfs[:, None] - cnts + 0.5) / (nfs[:, None] + 1.0)
    ref = 1.0 - kbars[:, None] / NV
    return logit(p) - logit(ref)


def permutation_analysis(cnts, nfs, kbars, mat, n_perm=N_PERM):
    """One set of within-video relabelling draws, three statistics read off it.

    Sharing the draws is both faster and cleaner than drawing separately per statistic: the
    ring contrast, the per-position map and the omnibus are then calibrated against exactly
    the same null realisations.
    """
    n_vid = cnts.shape[0]
    base_idx = np.tile(np.arange(NV), (n_vid, 1))
    obs_contrast = ring_contrast(mat)
    obs_pos = mat.mean(0)
    obs_om = float(np.sum((obs_pos - obs_pos.mean()) ** 2))

    null_contrast = np.empty(n_perm)
    null_pos = np.empty((n_perm, NV))
    null_om = np.empty(n_perm)
    for i in range(n_perm):
        idx = RNG.permuted(base_idx, axis=1)
        acc = _rel_logodds_mat(np.take_along_axis(cnts, idx, axis=1), nfs, kbars)
        null_contrast[i] = acc[:, BORDER].mean() - acc[:, INTERIOR].mean()
        m = acc.mean(0)
        null_pos[i] = m
        null_om[i] = float(np.sum((m - m.mean()) ** 2))

    centre = null_pos.mean(0)
    p_pos = ((np.abs(null_pos - centre) >= np.abs(obs_pos - centre)[None]).sum(0) + 1) \
        / (n_perm + 1)
    order = np.argsort(p_pos)
    q = p_pos[order] * NV / (np.arange(NV) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_pos = np.empty(NV)
    q_pos[order] = np.clip(q, 0, 1)

    return {
        "obs_contrast": obs_contrast,
        "p_contrast_onesided": float(((null_contrast >= obs_contrast).sum() + 1) / (n_perm + 1)),
        "null_contrast_mean": float(null_contrast.mean()),
        "null_contrast_sd": float(null_contrast.std()),
        "obs_pos": obs_pos,
        "q_pos": q_pos,
        "obs_omnibus": obs_om,
        "p_omnibus": float(((null_om >= obs_om).sum() + 1) / (n_perm + 1)),
    }


def arm_following_test(data, tag, synthetic=False, n_boot=2000):
    """Does the heat map move with the content, or stay on the grid?

    For arm A with induced permutation perm (cell j holds native cell perm[j]):
      content-following => map_A ~= map_native[perm]
      position-locked   => map_A ~= map_native
    Reported as a paired bootstrap over videos of the Fisher-z difference.
    """
    out = {}
    v_n, m_n, _, _, _ = build_matrix(data, tag, "native", synthetic=synthetic)
    idx_n = {v: i for i, v in enumerate(v_n)}
    for arm in ARMS:
        if arm == "native":
            continue
        v_a, m_a, _, _, _ = build_matrix(data, tag, arm, synthetic=synthetic)
        shared = [v for v in v_a if v in idx_n]
        if not shared:
            continue
        ia = {v: i for i, v in enumerate(v_a)}
        A = np.array([m_a[ia[v]] for v in shared])
        N = np.array([m_n[idx_n[v]] for v in shared])
        perm = data[(shared[0], arm)]["perm"].astype(np.int64)

        def stats(sel):
            a, n = A[sel].mean(0), N[sel].mean(0)
            r_stay = np.corrcoef(a, n)[0, 1]
            r_follow = np.corrcoef(a, n[perm])[0, 1]
            z = np.arctanh(np.clip([r_stay, r_follow], -0.999999, 0.999999))
            return r_stay, r_follow, z[0] - z[1]

        r_stay, r_follow, dz = stats(np.arange(len(shared)))
        boot = np.array([stats(RNG.integers(0, len(shared), len(shared)))[2]
                         for _ in range(n_boot)])
        out[arm] = {
            "n_videos": len(shared),
            "r_stay": float(r_stay), "r_follow": float(r_follow),
            "delta_z": float(dz),
            "delta_z_ci95": [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
            "verdict": ("position-locked" if np.percentile(boot, 2.5) > 0 else
                        "content-following" if np.percentile(boot, 97.5) < 0 else
                        "inconclusive"),
        }
    return out


def tie_report(data, tag):
    """How often does argmax break a tie, and does that bias toward low (top-left) indices?"""
    tot = ties = 0
    tied_first = 0
    for (vid, arm), rec in data.items():
        key = f"ties_{tag}"
        if key not in rec:
            continue
        t = rec[key].astype(np.int64)
        tot += t.size
        ties += int((t > 1).sum())
        tied_first += int(t[t > 1].sum() - (t > 1).sum())
    return {"steps": int(tot), "steps_with_ties": int(ties),
            "tie_rate": float(ties / tot) if tot else 0.0,
            "excess_tied_candidates": int(tied_first)}


def cls_attention_baseline(data, arm="native", synthetic=False):
    """ADTS-free positional baseline: the raw pooled [CLS] attention map.

    This is calibration_term1 itself. If it already has the spatial structure we find in the
    selection map, the prior lives in the encoder, not in the selection rule.
    """
    maps = []
    for (vid, a), rec in data.items():
        if a != arm or "cls_attn" not in rec:
            continue
        if synthetic != (rec["synthetic"] != ""):
            continue
        maps.append(rec["cls_attn"].astype(np.float64).mean(0))
    if not maps:
        return None
    M = np.array(maps)
    m = M.mean(0)
    return {"map": m.tolist(),
            "ring_profile": [float(m[RINGS == r].mean()) for r in range(7)],
            "n_videos": int(M.shape[0])}


def main(tags):
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_all()
    print(f"loaded {len(data)} (video, arm) records", flush=True)
    res = {"n_records": len(data), "rings": RINGS.tolist()}

    for tag in tags:
        vids, mat, cnts, nfs, kbars = build_matrix(data, tag, "native", synthetic=False)
        if not len(vids):
            print(f"  {tag}: no data, skipping", flush=True)
            continue
        pa = permutation_analysis(cnts, nfs, kbars, mat)
        pos_mean, q_pos = pa["obs_pos"], pa["q_pos"]

        entry = {
            "n_videos": int(len(vids)),
            "kbar": float(kbars.mean()),
            "n_frames": int(nfs[0]),
            "mean_nonselection_rate": float(1.0 - kbars.mean() / NV),
            "primary_ring_contrast": pa["obs_contrast"],
            "primary_p_onesided": pa["p_contrast_onesided"],
            "primary_null_mean": pa["null_contrast_mean"],
            "primary_null_sd": pa["null_contrast_sd"],
            "omnibus_stat": pa["obs_omnibus"], "omnibus_p": pa["p_omnibus"],
            "position_mean_rel_logodds": pos_mean.tolist(),
            "position_q": q_pos.tolist(),
            "n_positions_fdr_sig": int((q_pos < 0.05).sum()),
            "ring_profile": [float(pos_mean[RINGS == r].mean()) for r in range(7)],
            "ties": tie_report(data, tag),
        }
        res[tag] = entry
        print(f"  {tag}: contrast={pa['obs_contrast']:+.4f} "
              f"p={pa['p_contrast_onesided']:.4g} omnibus_p={pa['p_omnibus']:.4g} "
              f"fdr_sig={entry['n_positions_fdr_sig']}/169 "
              f"rings={[round(x, 3) for x in entry['ring_profile']]}", flush=True)

    res["arm_following_real"] = arm_following_test(data, MAIN_TAG, synthetic=False)
    res["arm_following_synth"] = arm_following_test(data, MAIN_TAG, synthetic=True)
    res["cls_attn_baseline_real"] = cls_attention_baseline(data, synthetic=False)
    res["cls_attn_baseline_synth"] = cls_attention_baseline(data, synthetic=True)

    # Synthetic controls: content is position-independent by construction, so any residual
    # structure here is the algorithm's own prior rather than a property of the footage.
    sv, sm, sc, sn, sk = build_matrix(data, MAIN_TAG, "native", synthetic=True)
    if len(sv):
        spa = permutation_analysis(sc, sn, sk, sm, n_perm=1000)
        res["synthetic_native"] = {
            "n_videos": int(len(sv)),
            "ring_contrast": spa["obs_contrast"],
            "p_onesided": spa["p_contrast_onesided"],
            "omnibus_p": spa["p_omnibus"],
            "ring_profile": [float(sm.mean(0)[RINGS == r].mean()) for r in range(7)],
        }
        print(f"  synthetic: contrast={spa['obs_contrast']:+.4f} "
              f"p={spa['p_contrast_onesided']:.4g} "
              f"omnibus_p={spa['p_omnibus']:.4g}", flush=True)

    with open(f"{OUT_DIR}/stage1_analysis.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", f"{OUT_DIR}/stage1_analysis.json", flush=True)

    print("\n=== arm following (real) ===", flush=True)
    print(json.dumps(res["arm_following_real"], indent=1), flush=True)
    print("\n=== arm following (synthetic) ===", flush=True)
    print(json.dumps(res["arm_following_synth"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=[
        MAIN_TAG, "R015_attn_div_v2", "R020_attn_div_v2", "R025_attn_div_v2",
        "R010_attn", "R010_div", "R010_attn_div"])
    a = ap.parse_args()
    sys.exit(main(a.tags))
