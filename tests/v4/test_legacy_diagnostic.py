"""Seven recorded legacy tags remain separate from new-source exact acceptance."""
from pathlib import Path
import numpy as np
import pytest
import torch

from v4 import runtime as R, selection as L, spec, worker as W


def test_seven_tag_coverage_and_preserved_legacy_difference(tmp_path,monkeypatch):
    video='synthetic_legacy'
    monkeypatch.setattr(R,'PROJECT',tmp_path)
    original_to=torch.Tensor.to
    def cpu_for_fixture(self,*args,**kwargs):
        if args and args[0]=='cuda:0':return self
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_for_fixture)
    rng=torch.Generator().manual_seed(888)
    pooled=torch.randn((64,169,8),generator=rng);attention=torch.rand((64,169),generator=rng)
    fp=tmp_path/'runs/stage2_features'/f'{video}.npz';mp=tmp_path/'runs/stage1'/f'{video}__native.npz'
    fp.parent.mkdir(parents=True);mp.parent.mkdir(parents=True)
    np.savez(fp,pooled169=pooled.numpy(),attn169=attention.numpy())
    names={'adts_v2':'attn_div_v2','adts_v1':'attn_div','attn':'attn','FlashVID-div-ablation@P1U169':'div'}
    old={'cls_attn':attention.numpy()}
    for method,budget in [(m,'R010') for m in spec.INTERNAL]+[('adts_v2',b) for b in ('R015','R020','R025')]:
        bounds,keep=L.official_reference(pooled,attention,method,budget)
        tag=budget+'_'+names[method]
        old['keep_'+tag]=(keep.numpy()+169*np.arange(64)[:,None]).reshape(-1).astype('<i4')
        old['seglen_'+tag]=bounds.numpy().astype('<i4')
    np.savez(mp,**old)
    result=W.legacy_seven_tag_regression(video,tmp_path/'exact')
    assert result['status']=='EXACT_ON_SAVED_INPUTS' and len(result['seven_present_tags'])==7 and len(result['absent_not_tested'])==9
    old['keep_R010_attn_div_v2'][0]=(old['keep_R010_attn_div_v2'][0]+1)%169
    np.savez(mp,**old)
    result=W.legacy_seven_tag_regression(video,tmp_path/'different')
    assert result['status']=='DIFFERENCES_REQUIRE_PARENT_ASSESSMENT'
    assert all(r['current_candidate_vs_independent_official_exact'] for r in result['seven_present_tags'])
    assert any(not r['historical_ordered_indices_exact'] for r in result['seven_present_tags'])
    assert (tmp_path/'different/legacy_diagnostics'/f'{video}_difference.npz').is_file()
    old.pop('keep_R015_attn_div_v2');np.savez(mp,**old)
    with pytest.raises(ValueError,match='exactly the seven'):W.legacy_seven_tag_regression(video,tmp_path/'missing')
