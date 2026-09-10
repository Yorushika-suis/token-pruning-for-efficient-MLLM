"""Pixel interventions, coordinate maps and set-valued geometric responses."""
import numpy as np
from .spec import ACTIVE, ARMS, GRID, NV, PROTOCOLS, SIDE


def transform(px, protocol, arm):
    if protocol not in PROTOCOLS or arm not in ARMS or tuple(px.shape[-2:]) != (SIDE, SIDE):
        raise ValueError('invalid pixel protocol, arm or shape')
    out = px.clone() if hasattr(px, 'clone') else np.array(px, copy=True)
    if protocol == 'controlled-gray':
        out[..., ACTIVE:, :] = 0.0
        out[..., :, ACTIVE:] = 0.0
    region = out[..., :ACTIVE, :ACTIVE]
    if isinstance(out, np.ndarray):
        if arm == ARMS[1]:
            region[...] = np.flip(region, axis=-1).copy()
        elif arm == ARMS[2]:
            region[...] = np.flip(region, axis=-2).copy()
        elif arm == ARMS[3]:
            region[...] = np.roll(region, (84, 140), axis=(-2, -1))
    else:
        import torch
        if arm == ARMS[1]:
            region.copy_(region.flip(-1))
        elif arm == ARMS[2]:
            region.copy_(region.flip(-2))
        elif arm == ARMS[3]:
            region.copy_(torch.roll(region, (84, 140), (-2, -1)))
    return out


def permutation(arm):
    r, c = np.divmod(np.arange(NV), GRID)
    if arm == ARMS[1]:
        c = GRID - 1 - c
    elif arm == ARMS[2]:
        r = GRID - 1 - r
    elif arm == ARMS[3]:
        r, c = (r + 3) % GRID, (c + 5) % GRID
    elif arm != ARMS[0]:
        raise ValueError('unknown arm')
    return r * GRID + c


def binary_mask(indices, nominal_k=None, allow_duplicates=False):
    x = np.asarray(indices)
    if x.ndim != 2 or x.dtype.kind not in 'iu' or x.shape[0] != 64 or x.shape[1] == 0:
        raise ValueError('indices must be a nonempty 64 x k integer array')
    if np.any((x < 0) | (x >= NV)) or (nominal_k is not None and x.shape[1] != nominal_k):
        raise ValueError('index or nominal budget outside contract')
    m = np.zeros((64, NV), dtype=bool)
    np.put_along_axis(m, x.astype(np.int64), True, axis=1)
    if not allow_duplicates and np.any(m.sum(-1) != x.shape[-1]):
        raise ValueError('strict subset selector returned duplicate slots')
    return m


def responses(original, transformed, arm, removed=()):
    o, t = np.asarray(original), np.asarray(transformed)
    if o.dtype != bool or t.dtype != bool or o.shape != t.shape or o.shape[-1] != NV:
        raise ValueError('geometric response requires matching binary masks')
    if np.any(t.sum(-1) == 0):
        raise ValueError('empty transformed selection is an input/algorithm error')
    o = o.copy()
    if removed:
        o[..., list(removed)] = False
    n = o.sum(-1)
    valid = n > 0
    if not removed and not valid.all():
        raise ValueError('empty primary original set')
    mapped = np.zeros_like(o)
    mapped[..., permutation(arm)] = o
    # Internal NaN is never serialized: callers must produce typed NA with counts.
    stay = np.divide((o & t).sum(-1), n, out=np.full(n.shape, np.nan), where=valid)
    follow = np.divide((mapped & t).sum(-1), n, out=np.full(n.shape, np.nan), where=valid)
    union = (o | t).sum(-1)
    if np.any(union == 0):
        raise ValueError('empty mask union')
    jaccard = np.divide((o & t).sum(-1), union, out=np.full(n.shape, np.nan), where=union > 0)
    return {'stay': stay, 'follow': follow, 'delta': stay-follow, 'valid': valid,
            'original_unique_k': n, 'transformed_unique_k': t.sum(-1),
            'hamming_per_slot': (o ^ t).mean(-1), 'jaccard': jaccard}


def coverage(mask):
    m = np.asarray(mask)
    if m.dtype != bool or m.shape[-1] != NV or np.any(m.sum(-1) == 0):
        raise ValueError('coverage requires nonempty binary masks')
    r, c = np.divmod(np.arange(NV), GRID)
    dist = np.maximum(np.abs(r[:, None]-r), np.abs(c[:, None]-c))
    flat = m.reshape(-1, NV)
    answer = [int(dist[:, selected].min(axis=1).max()) for selected in flat]
    return np.asarray(answer).reshape(m.shape[:-1])
