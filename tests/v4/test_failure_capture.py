"""A real injected selector mismatch must retain both sides before failing."""
import numpy as np
import pytest
import torch

from v4 import artifacts as A, selection as L, worker as W


def test_exact_mismatch_preserves_original_arrays(tmp_path,monkeypatch):
    rng=torch.Generator().manual_seed(71)
    pooled=torch.randn((64,169,8),generator=rng)
    attn=torch.rand((64,169),generator=rng)
    original=L.official_reference
    reference=[]
    def different(*args):
        bounds,indices=original(*args)
        indices=indices.clone();indices[0,0]=(indices[0,0]+1)%169
        reference.append(indices.numpy().copy())
        return bounds,indices
    monkeypatch.setattr(L,'official_reference',different)
    with pytest.raises(L.AcceptanceMismatch):
        W.accept_or_save(pooled,attn,'attn','R010','injected_counterexample',tmp_path/'first_difference')
    arrays=np.load(tmp_path/'first_difference/selector_capture.npz',allow_pickle=False)
    np.testing.assert_array_equal(arrays['input_pooled169'],pooled.numpy())
    np.testing.assert_array_equal(arrays['input_attn169'],attn.numpy())
    np.testing.assert_array_equal(arrays['reference_selected'],reference[0])
    assert not np.array_equal(arrays['candidate_selected'],arrays['reference_selected'])
    assert (tmp_path/'first_difference/difference.json').is_file()
    with pytest.raises(ValueError):A.parse_npz((tmp_path/'first_difference/selector_capture.npz').read_bytes())


def test_nonfinite_first_difference_is_preserved_but_not_scientific(tmp_path):
    from v4.worker import compare_or_save
    from v4.numerics import CAPTURE_AXES
    reference={'tensors':{name:torch.ones((1,1,2)) for name in CAPTURE_AXES}}
    actual={'tensors':{name:value.clone() for name,value in reference['tensors'].items()}}
    actual['tensors']['pooled169']=torch.tensor([[[float('nan'),float('inf')]]])
    with pytest.raises(ValueError,match='first numerical'):
        compare_or_save(actual,reference,'nonfinite_fixture',tmp_path/'first_difference')
    raw=tmp_path/'first_difference/actual.npz'
    with np.load(raw,allow_pickle=False) as z:
        assert np.isnan(z['pooled169'][0,0,0]) and np.isinf(z['pooled169'][0,0,1])
    with pytest.raises(ValueError):A.parse_npz(raw.read_bytes())


def test_observation_mismatch_keeps_both_sides_through_exception_boundary(tmp_path):
    from v4 import runtime as R
    actual=np.arange(64)+1;reference=np.arange(64)
    def official_style_wrapper():
        try:
            raise R.ObservationMismatch('index mismatch',actual_indices=actual,reference_indices=reference)
        except Exception:
            return 'incorrectly swallowed'
    with pytest.raises(R.ObservationMismatch):
        W.call_with_evidence(official_style_wrapper,(),'injected_sampling',tmp_path/'first_difference',{})
    with np.load(tmp_path/'first_difference/observation_capture.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['actual_indices'],actual)
        np.testing.assert_array_equal(z['reference_indices'],reference)


def test_comparison_exception_keeps_available_arrays_and_first_evidence(tmp_path):
    import json
    actual={'tensors':{'pooled169':torch.arange(4,dtype=torch.float32).reshape(1,1,4)}}
    reference={'tensors':{}}
    root=tmp_path/'first_difference'
    with pytest.raises(ValueError,match='six registered'):W.compare_or_save(actual,reference,'missing_capture',root)
    detail=json.loads((root/'difference.json').read_text())
    assert detail['exception']['type']=='ValueError' and 'Traceback' in detail['exception']['traceback']
    with np.load(root/'actual.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pooled169'],actual['tensors']['pooled169'].numpy())
    with np.load(root/'reference.npz',allow_pickle=False) as z:assert z.files==['meta_json']
    hashes={p.name:A.file_hash(p) for p in root.iterdir()}
    actual['tensors']['pooled169'].add_(10)
    with pytest.raises(ValueError,match='six registered'):W.compare_or_save(actual,reference,'second_failure',root)
    assert hashes=={p.name:A.file_hash(p) for p in root.iterdir()}
