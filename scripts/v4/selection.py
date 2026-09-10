"""Independent candidate segmentation/selectors plus unmodified official reference.

The candidate retains the original precision and reduction order. It does not
use the old mean-vector simplification or complement-all-topk approximation.
"""
import importlib
import inspect
import hashlib

import numpy as np
import torch

from .spec import BUDGETS, GRID, NV, INTERNAL, RANDOM, METHODS, stream
from .geometry import binary_mask


def additional_from_similarities(sim, threshold=0.9, minimum=8, complementary=True):
    x = sim.clone()
    cuts = torch.where(x < threshold)[0]
    missing = minimum - (cuts.numel()+1)
    if complementary and missing > 0:
        x[x < threshold] = 1.0
        extra = torch.topk(x, k=min(missing, x.shape[0]), largest=False).indices
        cuts = torch.cat((cuts, extra)).sort().values
    points = torch.nn.functional.pad(cuts, (1,1), value=0)
    points[0], points[-1] = -1, sim.shape[0]
    return torch.diff(points)


def candidate_segments(pooled, threshold=0.9, minimum=8, complementary=True):
    if pooled.ndim != 3 or pooled.shape[:2] != (64,169) or not torch.isfinite(pooled).all():
        raise ValueError('segmentation needs full finite 64-frame pooled input')
    features = pooled.mean(1)
    norm = features.norm(p=2, dim=-1, keepdim=True)
    if (norm == 0).any():
        raise ValueError('official segmentation undefined for zero frame mean')
    normalized = features / norm
    sim = torch.sum(normalized[:-1] * normalized[1:], dim=-1)
    return additional_from_similarities(sim, threshold, minimum, complementary)


def _candidate_segment(features, attention, k, method):
    if method == 'attn':
        return torch.topk(attention, k, dim=-1).indices.sort().values
    x = features.float()
    norm = x.norm(p=2, dim=-1, keepdim=True)
    if (norm == 0).any():
        raise ValueError('official cosine distance undefined for zero token')
    unit = x / norm
    d = 1.0 - torch.bmm(unit, unit.transpose(-1,-2))
    if method == 'adts_v1':
        d = d * (attention.float()*1e6).unsqueeze(1)
    elif method == 'adts_v2':
        frame_mean = x.mean(1)
        relevance = torch.einsum('b n d, c d -> b c n', x, frame_mean).mean(1)
        d = d * (attention.float()*1e6).unsqueeze(1) * relevance.unsqueeze(1)
    elif method != 'FlashVID-div-ablation@P1U169':
        raise ValueError('unknown internal method')
    if not torch.isfinite(d).all():
        raise ValueError('nonfinite calibrated distance')
    out = torch.zeros((x.shape[0],k), dtype=torch.long, device=x.device)
    seed_score = torch.topk(d, k=2, dim=1, largest=False).values[:,1,:]
    out[:,0] = seed_score.argmax(-1)
    for i in range(1,k):
        rows = out[:,:i].unsqueeze(-1).expand(-1,-1,NV)
        score = d.gather(1,rows).min(1).values
        if method != 'adts_v2':
            score.scatter_(1,out[:,:i],-1)
        out[:,i] = score.argmax(-1)
    return out.sort().values


def candidate(pooled, attention, method, budget):
    if method not in INTERNAL or budget not in BUDGETS or attention.shape != (64,169) or pooled.dtype != attention.dtype:
        raise ValueError('invalid selector configuration or input dtype')
    if not torch.isfinite(attention).all():
        raise ValueError('nonfinite attention')
    lengths = candidate_segments(pooled)
    selections = []
    start = 0
    for size in lengths.tolist():
        if size <= 0:
            raise ValueError('empty official-compatible segment')
        end = start + size
        selections.append(_candidate_segment(pooled[start:end].clone(), attention[start:end].clone(), BUDGETS[budget], method))
        start = end
    if start != 64:
        raise ValueError('segmentation does not cover all frames')
    return lengths, torch.cat(selections,0)


def original_registry():
    ts = importlib.import_module('flashvid.token_selection')
    u = importlib.import_module('flashvid.utils')
    originals = {'adts_v2':ts.attn_div_v2_based_token_selection, 'adts_v1':ts.attn_div_based_token_selection,
                 'attn':ts.attn_based_token_selection, 'FlashVID-div-ablation@P1U169':ts.div_based_token_selection}
    names = {'adts_v2':'attn_div_v2','adts_v1':'attn_div','attn':'attn','FlashVID-div-ablation@P1U169':'div'}
    for name, function in originals.items():
        if u.ALL_TOKEN_SELECTION_METHOD[names[name]] is not function or function.__module__ != 'flashvid.token_selection':
            raise ValueError('official selector registry has been replaced')
    return u, originals


def official_reference(pooled, attention, method, budget):
    """Independently computes full-video boundaries, without candidate boundaries."""
    utils, originals = original_registry()
    x, a = pooled.clone(), attention.clone()
    lengths = utils.segment(video_features=x.mean(1), segment_threshold=0.9, min_segment_num=8, complementary_segment=True)
    out, start = [], 0
    for size in lengths.tolist():
        end = start + size
        kw = {'features':x[start:end].clone(), 'num_retained_tokens': BUDGETS[budget]}
        if method != 'FlashVID-div-ablation@P1U169':
            kw['cls_attention'] = a[start:end].clone()
        _, keep = originals[method](**kw)
        out.append(keep.clone())
        start = end
    if start != 64 or (lengths <= 0).any():
        raise ValueError('invalid official segmentation')
    return lengths.clone(), torch.cat(out,0)


class AcceptanceMismatch(ValueError):
    def __init__(self, message, bounds, selected, ref_bounds, ref_selected):
        super().__init__(message)
        self.arrays = {name:value.detach().cpu().clone().numpy() for name,value in (
            ('candidate_lengths',bounds),('candidate_selected',selected),
            ('reference_lengths',ref_bounds),('reference_selected',ref_selected))}


def acceptance(pooled, attention, method, budget):
    original_x, original_a = pooled.clone(), attention.clone()
    bounds, selected = candidate(pooled.clone(), attention.clone(), method, budget)
    ref_bounds, ref_selected = official_reference(pooled.clone(), attention.clone(), method, budget)
    if not torch.equal(pooled,original_x) or not torch.equal(attention,original_a):
        raise ValueError('selection mutated source snapshots')
    if not torch.equal(bounds,ref_bounds):
        raise AcceptanceMismatch('DySeg exact mismatch: '+repr((bounds.tolist(),ref_bounds.tolist())),
                                 bounds,selected,ref_bounds,ref_selected)
    if not torch.equal(selected,ref_selected):
        mismatch = torch.nonzero(selected != ref_selected)[0].tolist()
        raise AcceptanceMismatch('ordered selector index mismatch at '+repr(mismatch),
                                 bounds,selected,ref_bounds,ref_selected)
    binary_mask(selected.cpu().numpy(), BUDGETS[budget], allow_duplicates=method=='adts_v2')
    return bounds, selected


def grid_points(k):
    nr = int(np.floor(np.sqrt(k)))
    while nr > 1 and k//nr > GRID:
        nr -= 1
    nc = int(np.ceil(k/nr))
    rows = np.unique(np.round(np.linspace(0,GRID-1,nr)).astype(int))
    cols = np.unique(np.round(np.linspace(0,GRID-1,nc)).astype(int))
    points = [r*GRID+c for r in rows for c in cols]
    if len(points) > k:
        points = [points[i] for i in np.round(np.linspace(0,len(points)-1,k)).astype(int)]
    points = sorted(set(points))
    coordinates = np.stack(np.divmod(np.arange(NV),GRID),1).astype(float)
    while len(points) < k:
        dist = np.linalg.norm(coordinates[:,None,:]-coordinates[points][None,:,:],axis=-1).min(1)
        dist[points] = -1
        points = sorted(points+[int(dist.argmax())])
    return np.array(points[:k],dtype=np.int32)


def stratified_cells(k):
    nb = int(np.ceil(np.sqrt(k)))
    return [np.array([r*GRID+c for r in rr for c in cc])
            for rr in np.array_split(np.arange(GRID),nb)
            for cc in np.array_split(np.arange(GRID),nb)]


def control(video_id, method, budget, seed, purpose, arm):
    k = BUDGETS[budget]
    if method == 'grid_uniform':
        if seed != 'deterministic':
            raise ValueError('deterministic grid has one effective seed')
        return np.tile(grid_points(k),(64,1)), None
    if method not in RANDOM:
        raise ValueError('unknown stochastic control')
    record = stream(seed, video_id, method, purpose, budget, arm)
    rng = np.random.Generator(np.random.PCG64(record['seed128']))
    out = []
    cells = stratified_cells(k)
    points = grid_points(k)
    for frame in range(64):
        if method == 'random_uniform_without_replacement':
            selected = rng.choice(NV,k,replace=False)
        elif method == 'spatial_stratified_legacy_rule':
            order = rng.permutation(len(cells))
            selected = [int(rng.choice(cells[b])) for b in order[:k]]
            while len(selected) < k:
                rest = np.setdiff1d(np.arange(NV), selected)
                selected.append(int(rng.choice(rest)))
        else:
            dr, dc = rng.integers(0,GRID,size=2)
            r,c = np.divmod(points,GRID)
            selected = ((r+dr)%GRID)*GRID+(c+dc)%GRID
        out.append(np.sort(selected))
    answer = np.asarray(out,dtype=np.int32)
    binary_mask(answer,k)
    return answer, record


def control_marginals(method, budget):
    k = BUDGETS[budget]
    if method in ('random_uniform_without_replacement','translated_fixed_grid_torus'):
        return np.full(NV,k/NV)
    if method == 'grid_uniform':
        p = np.zeros(NV);p[grid_points(k)] = 1
        return p
    if method == 'spatial_stratified_legacy_rule':
        cells = stratified_cells(k)
        p = np.zeros(NV)
        for cell in cells:
            p[cell] = k / len(cells) / len(cell)
        return p
    raise ValueError('no registered marginal formula')
