"""Per-element and per-vector comparisons at registered capture boundaries."""
import numpy as np

CAPTURE_AXES = {'pixels': 'exact', 'sig729': -1, 'projected729': -1,
                'pooled169': -1, 'attn729': -1, 'attn169': -1}


def _array(x):
    if hasattr(x, 'detach'):
        return x.detach().double().cpu().numpy(), str(x.dtype)
    a = np.asarray(x)
    return a.astype(np.float64), str(a.dtype)


def compare(actual, reference, capture):
    if capture not in CAPTURE_AXES:
        raise ValueError('unregistered capture point')
    a, ad = _array(actual)
    b, bd = _array(reference)
    if ad != bd or a.shape != b.shape:
        return {'passed': False, 'reason': 'original_dtype_or_shape_mismatch',
                'actual_dtype': ad, 'reference_dtype': bd, 'actual_shape': list(a.shape), 'reference_shape': list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed': False, 'reason': 'nonfinite_capture'}
    exact = np.array_equal(a,b)
    if capture == 'pixels':
        return {'passed': exact, 'exact': exact, 'dtype': ad, 'shape': list(a.shape), 'axis': 'all_elements_exact'}
    if ad in ('torch.float32','float32'):
        atol,rtol,minimum_cosine = 1e-5,1e-4,0.99999
    elif ad in ('torch.float16','float16','torch.bfloat16','bfloat16'):
        atol,rtol,minimum_cosine = 1e-3,1e-2,0.9999
    else:
        raise ValueError('unregistered numeric precision at capture: '+ad)
    error = np.abs(a-b)
    bound = atol+rtol*np.abs(b)
    ratios = error/bound
    coord = np.unravel_index(int(ratios.argmax()),a.shape)
    zero_a,zero_b = np.all(a==0,axis=-1),np.all(b==0,axis=-1)
    zero_mismatch = zero_a != zero_b
    nonzero = ~zero_a & ~zero_b
    cosine = np.ones(a.shape[:-1],dtype=np.float64)
    cosine[nonzero] = ((a*b).sum(-1)[nonzero]/
                       (np.linalg.norm(a,axis=-1)[nonzero]*np.linalg.norm(b,axis=-1)[nonzero]))
    worst = np.unravel_index(int(cosine.argmin()),cosine.shape)
    passed = not zero_mismatch.any() and bool((ratios<=1).all()) and bool((cosine[nonzero]>=minimum_cosine).all())
    return {'passed': passed, 'exact': exact, 'dtype':ad, 'shape':list(a.shape), 'vector_axis':-1,
            'atol':atol,'rtol':rtol,'required_minimum_cosine':minimum_cosine,
            'max_absolute_error':float(error.max()),'worst_scaled_error':float(ratios[coord]),
            'worst_element_coordinate':list(coord),'minimum_cosine':float(cosine.min()),
            'worst_vector_coordinate':list(worst),'zero_classification_mismatches':int(zero_mismatch.sum()),
            'zero_vectors':int(zero_b.sum()),'nonzero_vectors':int(nonzero.sum())}
