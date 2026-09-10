"""Approved descriptive estimands; V,S,F,J axis order is explicit throughout."""
import numpy as np
from .spec import FRAMES, NV


def na(reason, layer, valid, invalid, **extra):
    return {'value': None, 'reason': reason, 'layer': layer,
            'valid_count': int(valid), 'invalid_count': int(invalid), **extra}


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    if not np.all((p > 0) & (p < 1)):
        raise ValueError('nonfinite reference logit')
    return np.log(p) - np.log1p(-p)


def validate_masks(m):
    m = np.asarray(m)
    if m.dtype != bool or m.ndim != 4 or m.shape[2:] != (FRAMES, NV) or m.shape[0] < 2 or m.shape[1] not in (1, 5):
        raise ValueError('expected binary V x S x 64 x 169, V>=2, S=1 or 5')
    return m


def position_summary(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[-1] != NV or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('invalid video position rates')
    mu = p.mean(0)
    s = p.std(0, ddof=1)
    centered = p - p.mean(1, keepdims=True)
    cs = centered.std(0, ddof=1)
    return {'mean_rate': mu, 'P': float(mu.std(ddof=0)), 'S': s,
            'Q': float(np.sqrt(np.mean(s*s))), 'budget_centered_S': cs,
            'budget_centered_Q': float(np.sqrt(np.mean(cs*cs)))}


def b10_from_counts(c, u):
    """c: V x S x 169; u: V x S mean unique frame budgets."""
    c, u = np.asarray(c, dtype=np.float64), np.asarray(u, dtype=np.float64)
    if c.ndim != 3 or c.shape[2] != NV or c.shape[0] < 2 or c.shape[1] not in (1,5) or u.shape != c.shape[:2] or not np.isfinite(c).all() or not np.isfinite(u).all():
        raise ValueError('invalid B10 shape/finiteness')
    if np.any((c < 0) | (c > FRAMES)) or np.any(c != np.floor(c)) or np.any((u < 0) | (u > NV)):
        raise ValueError('B10 counts or budgets outside domain')
    if not np.array_equal(c.sum(-1)/FRAMES,u):
        raise ValueError('B10 reference budgets do not derive from the same binary counts')
    v, s, _ = c.shape
    valid = (u > 0) & (u < NV)
    out = {}
    if valid.all():
        per_video_seed = logit((FRAMES-c+0.5)/(FRAMES+1)) - logit(1-u/NV)[..., None]
        out['video_level'] = {'value': per_video_seed.mean((0, 1)),
                              'per_seed': per_video_seed.mean(0),
                              'per_video': per_video_seed.mean(1),
                              'valid_count': v*s, 'invalid_count': 0, 'layer': 'method_protocol_budget_video_B10'}
    else:
        out['video_level'] = na('at_least_one_video_seed_reference_is_zero_or_one',
                                'method_protocol_budget_video_B10', valid.sum(), (~valid).sum(),
                                invalid_video_seed_indices=np.argwhere(~valid))
    pooled_u = u.mean(0)
    valid_pool = (pooled_u > 0) & (pooled_u < NV)
    if valid_pool.all():
        g = logit(((FRAMES-c).sum(0)+0.5)/(FRAMES*v+1)) - logit(1-pooled_u/NV)[:, None]
        out['pooled_marginal_rel_logodds'] = {'value': g.mean(0), 'per_seed': g,
                                           'valid_count': s, 'invalid_count': 0,
                                           'layer': 'method_protocol_budget_pooled_B10'}
    else:
        out['pooled_marginal_rel_logodds'] = na('at_least_one_pooled_seed_reference_is_zero_or_one',
                                              'method_protocol_budget_pooled_B10', valid_pool.sum(), (~valid_pool).sum())
    return out


def summarize(m):
    m = validate_masks(m)
    c = m.sum(2, dtype=np.int64)
    u = m.sum(-1).mean(2)
    ps = c/FRAMES
    p = ps.mean(1)
    per_seed = [position_summary(ps[:, s]) for s in range(m.shape[1])]
    seed_spread = {}
    for metric in ('P', 'Q', 'budget_centered_Q'):
        values = np.array([x[metric] for x in per_seed])
        seed_spread[metric] = {'values': values, 'range': [values.min(), values.max()],
                               'sample_SD': float(values.std(ddof=1)) if len(values)>1 else
                               na('deterministic_method_has_one_effective_seed', 'seed_spread', 1, 0)}
    return {**position_summary(p), 'per_video_rate': p, 'per_seed': per_seed,
            'seed_spread': seed_spread, 'b10': b10_from_counts(c, u),
            'mean_unique_k_per_video_seed': u}


def bootstrap_indices(v=300):
    if v != 300:
        raise ValueError('formal bootstrap fixes the 300-video census')
    return np.random.Generator(np.random.PCG64(20260905)).integers(0, v, size=(2000, v), dtype=np.int64)


def interval(values, layer, reason='at_least_one_bootstrap_replicate_is_undefined'):
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] != 2000:
        raise ValueError('exactly 2000 planned bootstrap replicates required')
    good = np.isfinite(x).reshape(2000, -1).all(1)
    if not good.all():
        return na(reason, layer, good.sum(), (~good).sum(),
                  invalid_replicate_indices=np.flatnonzero(~good), planned_replicates=2000)
    return {'value': np.quantile(x, [0.025, 0.975], axis=0, method='linear'),
            'layer': layer, 'valid_count': 2000, 'invalid_count': 0, 'quantile_method': 'linear'}


def paired_bootstrap(gray, unblanked, indices):
    a, b = validate_masks(gray), validate_masks(unblanked)
    ix = np.asarray(indices)
    if a.shape != b.shape or a.shape[0] != 300 or ix.shape != (2000, 300) or ix.dtype.kind not in 'iu' or np.any((ix < 0) | (ix >= 300)):
        raise ValueError('paired bootstrap coverage or indices do not match')
    # A single externally fixed resampling table is used by every call.
    traces = {}
    for name, m in (('controlled-gray', a), ('unblanked-native', b)):
        c = m.sum(2, dtype=np.int64)
        u = m.sum(-1).mean(2)
        p = (c/FRAMES).mean(1)
        rows = {k: [] for k in ('P', 'Q', 'budget_centered_Q', 'video_B10', 'pooled_B10')}
        for draw in ix:
            summary = position_summary(p[draw])
            for k in ('P', 'Q', 'budget_centered_Q'):
                rows[k].append(summary[k])
            terms = b10_from_counts(c[draw], u[draw])
            for k, term in (('video_B10', 'video_level'), ('pooled_B10', 'pooled_marginal_rel_logodds')):
                value = terms[term]['value']
                rows[k].append(np.full(NV, np.nan) if value is None else value)
        traces[name] = {k: np.asarray(v) for k, v in rows.items()}
    result = {name: {k: interval(v, 'video_cluster_bootstrap_'+k) for k, v in values.items()}
              for name, values in traces.items()}
    result['protocol_difference'] = {k: interval(traces['unblanked-native'][k]-traces['controlled-gray'][k],
                                                'paired_protocol_difference_'+k)
                                     for k in ('P', 'Q', 'budget_centered_Q')}
    result['assumptions'] = ['fixed_census_is_descriptive', 'external_population_interpretation_requires_exchangeable_representative_videos',
                             'does_not_cover_dataset_selection_bias', 'conditional_on_registered_seeds', 'no_equivalence_or_algorithm_superiority_test']
    return result, traces


def restricted_geometry_summary(values, ids, indices):
    """V x S x F; undefined frames remain explicit, never replaced by zero."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] != len(ids) or x.shape[-1] != FRAMES or x.shape[1] not in (1,5) or len(ids)<2 or len(set(ids))!=len(ids):
        raise ValueError('expected V x S x 64 geometry values')
    if np.isinf(x).any():
        raise ValueError('infinite geometry value is an error, not an empty-set NA')
    indices=np.asarray(indices)
    if indices.shape!=(2000,len(ids)) or indices.dtype.kind not in 'iu' or np.any((indices<0)|(indices>=len(ids))):
        raise ValueError('invalid geometry bootstrap indices')
    finite = np.isfinite(x)
    counts = finite.sum(-1)
    seed_mean = np.divide(np.where(finite, x, 0).sum(-1), counts,
                          out=np.full(counts.shape, np.nan), where=counts>0)
    good_seed = np.isfinite(seed_mean)
    per_video = np.divide(np.where(good_seed, seed_mean, 0).sum(-1), good_seed.sum(-1),
                         out=np.full(len(ids), np.nan), where=good_seed.sum(-1)>0)
    valid = np.isfinite(per_video)
    draws = []
    for draw in indices:
        values_draw = per_video[draw]
        eligible = np.isfinite(values_draw)
        draws.append(values_draw[eligible].mean() if eligible.any() else np.nan)
    result = {'value': float(per_video[valid].mean()) if valid.any() else None,
              'reason': None if valid.any() else 'no_video_has_an_eligible_frame',
              'layer': 'restricted_geometry_video_mean', 'valid_count': int(valid.sum()),
              'invalid_count': int((~valid).sum()), 'eligible_video_ids': [v for v, ok in zip(ids, valid) if ok],
              'per_video': [{'video_id': v, 'value': float(per_video[i]) if valid[i] else None,
                             'reason':None if valid[i] else 'no_seed_has_an_eligible_frame',
                             'layer':'restricted_geometry_per_video_seed_mean',
                             'valid_count':int(good_seed[i].sum()),'invalid_count':int((~good_seed[i]).sum()),
                             'valid_frames_per_seed': counts[i], 'invalid_frames_per_seed': FRAMES-counts[i],
                             'valid_seeds': int(good_seed[i].sum()), 'invalid_seeds': int((~good_seed[i]).sum())}
                            for i, v in enumerate(ids)],
              'interval': interval(draws, 'restricted_geometry_cluster_bootstrap'),
              'population_note': 'mean_over_eligible_videos_not_necessarily_all_300'}
    return result


def jsonable(x):
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        raise ValueError('untyped nonfinite value cannot be serialized')
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x
