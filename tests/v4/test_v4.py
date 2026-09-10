"""D2 counterexamples and independent computations for the approved contracts."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(2)

from v4 import artifacts as A, geometry as G, statistics as S, selection as L, spec
from v4.numerics import compare


def test_legacy_seed_vectors():
    vectors = [(0,'_KU4X06VNiE','random',3974431837),(0,'fFjv93ACGo8','random',167265200),
               (1,'_KU4X06VNiE','random',525023160),(0,'_KU4X06VNiE','spatial_stratified',4077748110),
               (20260904,'fFjv93ACGo8','spatial_stratified',1617129638)]
    for seed,vid,method,expected in vectors:
        fields = [b'flashvid-adts/v3',seed.to_bytes(8,'big'),vid.encode(),method.encode()]
        raw = b''.join(len(b).to_bytes(4,'big')+b for b in fields)
        assert int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest()[:4],'big')==expected


NEW_VECTORS = [(0,'_KU4X06VNiE',spec.RANDOM[0],'protocol_pair','R010','identity'),
               (4,'fFjv93ACGo8',spec.RANDOM[1],'geometry_arm','R025',spec.ARMS[3]),
               (1,'-ApL8d6tX5U',spec.RANDOM[2],'geometry_arm','R015',spec.ARMS[1])]


def test_new_vectors_independent_encoding():
    for seed,vid,method,purpose,budget,arm in NEW_VECTORS:
        # Independent encoder uses int.to_bytes, not production struct/material.
        fields = [b'flashvid-adts/v4-stream-1',seed.to_bytes(8,'big'),vid.encode('utf-8'),
                  (method+'/v4-1').encode('ascii'),purpose.encode(),budget.encode(),arm.encode()]
        raw = b''.join(len(x).to_bytes(4,'big')+x for x in fields)
        got = spec.stream(seed,vid,method,purpose,budget,arm)
        assert got['material_hex']==raw.hex()
        assert got['seed128']==int(hashlib.blake2b(raw,digest_size=16).hexdigest(),16)
    with pytest.raises(ValueError):spec.stream(True,'x',spec.RANDOM[0],'protocol_pair','R010','identity')
    with pytest.raises(ValueError):spec.stream(0,'x',spec.RANDOM[0],'protocol_pair','R010',spec.ARMS[1])


def test_stream_retries_order_and_purpose():
    keys = [(s,'_KU4X06VNiE',m,p,b,a) for m in spec.RANDOM for s in spec.SEEDS for p,a in spec.PURPOSE_ARMS for b in spec.BUDGETS]
    first = {}
    for s,v,m,p,b,a in keys:
        first[(s,v,m,p,b,a)] = L.control(v,m,b,s,p,a)[0]
    for s,v,m,p,b,a in reversed(keys):
        np.testing.assert_array_equal(first[(s,v,m,p,b,a)],L.control(v,m,b,s,p,a)[0])
    for m in spec.RANDOM:
        pp = L.control('_KU4X06VNiE',m,'R010',0,'protocol_pair','identity')[0]
        geo = L.control('_KU4X06VNiE',m,'R010',0,'geometry_arm','identity')[0]
        assert not np.array_equal(pp,geo)
    assert spec.check_streams(['_KU4X06VNiE','fFjv93ACGo8'])['distinct_keys']==600


def test_all_pixel_sources_and_inverse():
    # Unique label per frame/channel/pixel, no video codec or interpolation involved.
    labels = np.arange(2*3*384*384,dtype=np.int64).reshape(2,3,384,384)
    source = labels.copy()
    for protocol in spec.PROTOCOLS:
        base = labels.copy()
        if protocol=='controlled-gray':base[...,364:,:]=0;base[...,:,364:]=0
        for arm in spec.ARMS:
            out = G.transform(labels,protocol,arm)
            mapping = G.permutation(arm)
            assert sorted(mapping.tolist())==list(range(169))
            inverse = np.argsort(mapping)
            np.testing.assert_array_equal(inverse[mapping],np.arange(169))
            np.testing.assert_array_equal(out[...,364:,:],base[...,364:,:])
            np.testing.assert_array_equal(out[...,:,364:],base[...,:,364:])
            for r in range(13):
                for c in range(13):
                    src = base[...,28*r:28*(r+1),28*c:28*(c+1)]
                    if arm==spec.ARMS[1]:rr,cc=r,12-c;src=src[...,::-1]
                    elif arm==spec.ARMS[2]:rr,cc=12-r,c;src=src[...,::-1,:]
                    elif arm==spec.ARMS[3]:rr,cc=(r+3)%13,(c+5)%13
                    else:rr,cc=r,c
                    assert mapping[13*r+c]==13*rr+cc
                    np.testing.assert_array_equal(out[...,28*rr:28*(rr+1),28*cc:28*(cc+1)],src)
            np.testing.assert_array_equal(G.transform(torch.from_numpy(labels),protocol,arm).numpy(),out)
    np.testing.assert_array_equal(labels,source)


def test_geometry_direction_fixed_follow_symmetric_and_empty():
    o = np.zeros((64,169),bool);o[:,[0,1,14]]=True
    m = G.permutation(spec.ARMS[1]);moving = np.zeros_like(o);moving[:,m[[0,1,14]]]=True
    assert np.all(G.responses(o,o,spec.ARMS[1])['delta']==1)
    assert np.all(G.responses(o,moving,spec.ARMS[1])['delta']==-1)
    symmetric = np.zeros_like(o);symmetric[:,[0,12]]=True
    assert np.all(G.responses(symmetric,symmetric,spec.ARMS[1])['delta']==0)
    removed = G.responses(o,o,spec.ARMS[1],removed=(0,1,14))
    assert not removed['valid'].any() and np.isnan(removed['delta']).all()
    with pytest.raises(ValueError):G.responses(np.zeros_like(o),o,spec.ARMS[1])
    with pytest.raises(ValueError,match='empty transformed'):
        G.responses(o,np.zeros_like(o),spec.ARMS[1])
    with pytest.raises(ValueError,match='empty transformed'):
        G.responses(o,np.zeros_like(o),spec.ARMS[1],removed=(0,1,14))


def test_plane_coverage_and_control_theory():
    mask = np.zeros((64,169),bool);mask[:,0]=True
    assert np.all(G.coverage(mask)==12)
    mask[:]=False;mask[:,84]=True;assert np.all(G.coverage(mask)==6)
    for budget,k in spec.BUDGETS.items():
        for name in spec.CONTROLS:
            p = L.control_marginals(name,budget)
            assert np.isclose(p.sum(),k)
        assert len(np.unique(L.grid_points(k)))==k
    p = L.control_marginals('spatial_stratified_legacy_rule','R010')
    assert p.std()==pytest.approx(0.016801503981657608,abs=1e-17)
    expected = np.sum(p*p-p*p[G.permutation(spec.ARMS[3])])/16
    assert expected>0  # Independent unequal-probability arms need not have delta=0.


def test_B10_seed_order_not_pooled_frames():
    m = np.zeros((2,5,64,169),bool);m[:,:,:,1:17]=True
    m[:,4,:16,1]=False;m[:,4,:16,0]=True
    r = S.summarize(m)
    counts = np.array([0,0,0,0,16])
    reference = np.log((1-16/169)/(16/169))
    manual = np.mean(np.log((64-counts+.5)/(counts+.5))-reference)
    wrong = np.log((64-counts.mean()+.5)/(counts.mean()+.5))-reference
    assert r['b10']['video_level']['value'][0]==pytest.approx(manual,abs=1e-14)
    assert abs(manual-wrong)>.1
    wrong_concatenation=np.log((320-counts.sum()+.5)/(counts.sum()+.5))-reference
    assert abs(manual-wrong_concatenation)>.1
    pooled_manual = np.mean(np.log((2*(64-counts)+.5)/(2*counts+.5))-reference)
    assert r['b10']['pooled_marginal_rel_logodds']['value'][0]==pytest.approx(pooled_manual,abs=1e-14)


def test_B10_NA_layers_not_silent_drop():
    m = np.zeros((2,1,64,169),bool);m[1,:,:,0]=True
    r = S.summarize(m)
    assert r['b10']['video_level']['value'] is None
    assert r['b10']['video_level']['invalid_count']==1
    assert r['b10']['pooled_marginal_rel_logodds']['value'] is not None
    m[:]=False;r=S.summarize(m)
    assert r['b10']['pooled_marginal_rel_logodds']['value'] is None
    assert r['P']==0 and r['Q']==0


def test_P_Q_ddof_and_budget_centering():
    p = np.zeros((300,169));p[:150,:10]=.5;p[150:,:10]=1
    r=S.position_summary(p);mu=p.mean(0)
    manual_s=np.sqrt(np.sum((p-mu)**2,axis=0)/299)
    assert r['P']==pytest.approx(np.sqrt(np.sum((mu-mu.mean())**2)/169))
    np.testing.assert_allclose(r['S'],manual_s,atol=1e-15)
    assert r['Q']==pytest.approx(np.sqrt(np.mean(manual_s**2)))
    b=np.repeat(np.arange(300)[:,None]/300,169,axis=1)
    assert S.position_summary(b)['budget_centered_Q']<1e-15


def test_bootstrap_indices_and_undefined_forensic_replicates():
    ix=S.bootstrap_indices();assert ix.shape==(2000,300)
    np.testing.assert_array_equal(ix,np.random.Generator(np.random.PCG64(20260905)).integers(0,300,size=(2000,300),dtype=np.int64))
    x=np.full((300,1,64),np.nan);x[1]=.25
    draws=np.zeros((2000,300),dtype=np.int64);draws[0]=1
    r=S.restricted_geometry_summary(x,[str(i) for i in range(300)],draws)
    assert r['value']==.25 and r['valid_count']==1 and r['invalid_count']==299
    assert r['interval']['value'] is None and r['interval']['invalid_count']==1999 and r['interval']['valid_count']==1
    finite=S.interval(np.arange(2000),'unit')['value']
    np.testing.assert_array_equal(finite,np.quantile(np.arange(2000),[.025,.975],method='linear'))


def test_geometry_rejects_silent_negative_index_and_infinity():
    values=np.zeros((300,1,64));values[-1]=1
    negative=np.full((2000,300),-1,dtype=np.int64)
    with pytest.raises(ValueError):
        S.restricted_geometry_summary(values,[str(i) for i in range(300)],negative)
    values[0,0,0]=np.inf
    with pytest.raises(ValueError):
        S.restricted_geometry_summary(values,[str(i) for i in range(300)],S.bootstrap_indices())


def test_B10_rejects_reference_from_different_mask():
    m=np.zeros((2,1,64,169),bool);m[:,:,:,:16]=True
    c=m.sum(2);u=m.sum(-1).mean(2)
    expected=S.b10_from_counts(c,u)
    assert expected['video_level']['value'] is not None
    with pytest.raises(ValueError):S.b10_from_counts(c,u+1)


def test_paired_bootstrap_recomputes_non_linear_terms_and_multiplicity():
    gray=np.zeros((300,1,64,169),bool);gray[:,:,:,:16]=True
    unblanked=np.zeros_like(gray)
    for v in range(300):unblanked[v,:,:,np.arange(16)+v%8]=True
    indices=S.bootstrap_indices()
    result,traces=S.paired_bootstrap(gray,unblanked,indices)
    for replicate in (0,17,1999):
        draw=indices[replicate]
        counts=unblanked[draw].sum(2).astype(float)
        p=counts[:,0]/64
        mean=p.mean(0);sd=np.sqrt(((p-mean)**2).sum(0)/299)
        assert traces['unblanked-native']['P'][replicate]==pytest.approx(np.sqrt(((mean-mean.mean())**2).mean()))
        assert traces['unblanked-native']['Q'][replicate]==pytest.approx(np.sqrt((sd**2).mean()))
        reference=np.log((1-16/169)/(16/169))
        video=np.log((64-counts+.5)/(counts+.5))-reference
        np.testing.assert_allclose(traces['unblanked-native']['video_B10'][replicate],video.mean((0,1)),rtol=1e-13,atol=1e-13)
        # Scalar math independently applies the registered logit to the same
        # binary64 probability. Algebraically simplifying to log(numerator /
        # complement) changes endpoint rounding by about 1.7e-12 at c=0.
        probabilities=((64-counts).sum(0)+.5)/(64*len(draw)+1)
        pooled=np.array([[math.log(float(q))-math.log1p(-float(q))-reference for q in probabilities[0]]])
        np.testing.assert_allclose(traces['unblanked-native']['pooled_B10'][replicate],pooled.mean(0),rtol=1e-13,atol=1e-13)
    difference=traces['unblanked-native']['P']-traces['controlled-gray']['P']
    np.testing.assert_allclose(result['protocol_difference']['P']['value'],np.quantile(difference,[.025,.975],method='linear'))


def test_numeric_each_vector_dtype_and_zero_classification():
    a=np.ones((2,169,4),np.float32);b=a.copy();assert compare(a,b,'pooled169')['passed']
    a[1,168]=[-1,-1,-1,-1];r=compare(a,b,'pooled169');assert not r['passed'] and r['worst_vector_coordinate']==[1,168]
    assert not compare(a.astype(np.float16),b,'pooled169')['passed']
    assert compare(torch.from_numpy(b),b,'pooled169')['reason']=='original_dtype_or_shape_mismatch'
    a[:]=0;b[:]=0;b[0,0,0]=1e-8;r=compare(a,b,'attn729');assert not r['passed'] and r['zero_classification_mismatches']==1
    with pytest.raises(ValueError):compare(np.ones((1,2),np.float64),np.ones((1,2),np.float64),'pooled169')


def test_official_DySeg_tie_counterexample():
    import flashvid.utils as official
    sim=torch.full((63,),.95,dtype=torch.float32);sim[0]=.8
    new=L.additional_from_similarities(sim)
    old_cuts=torch.topk(sim,7,largest=False).indices.add(1).sort().values.tolist()
    reference=official.additional_segment(torch.where(sim<.9)[0],64,8,sim.clone(),.9,True)
    assert torch.equal(new,reference)
    assert old_cuts==[1,41,42,43,46,47,48]
    assert new.cumsum(0)[:-1].tolist()==[1,41,42,43,44,46,47]
    for scores in (torch.full((63,),.95),torch.linspace(.1,.99,63)):
        ref=official.additional_segment(torch.where(scores<.9)[0],64,8,scores.clone(),.9,True)
        assert torch.equal(L.additional_from_similarities(scores),ref)


def test_original_selection_and_duplicate_pressure():
    from flashvid.token_selection import attn_div_v2_based_token_selection
    generator=torch.Generator().manual_seed(912)
    x=torch.randn((64,169,8),generator=generator);a=torch.rand((64,169),generator=generator)
    for method in spec.INTERNAL:
        for budget in spec.BUDGETS:
            L.acceptance(x,a,method,budget)
    x=torch.full((64,169,2),-1.);x[:,:,1]=0;x[:,0,0]=1000;a=torch.ones((64,169))
    got=L._candidate_segment(x,a,16,'adts_v2')
    _,ref=attn_div_v2_based_token_selection(x.clone(),a.clone(),16)
    assert torch.equal(got,ref) and got.unique().tolist()==[0]
    mask=G.binary_mask(got.numpy(),16,allow_duplicates=True)
    assert np.all(mask.sum(-1)==1)
    with pytest.raises(ValueError):G.binary_mask(got.numpy(),16)


def test_observer_calls_original_and_isolates_mutation():
    from v4.capture import CompressionCapture, tensor_hash
    from flashvid.configuration_flashvid import FlashVidConfig
    import flashvid.utils as u
    generator=torch.Generator().manual_seed(129)
    pooled=torch.randn((64,169,8),generator=generator)
    attn=torch.rand((64,169),generator=generator)
    initial=(tensor_hash(pooled),tensor_hash(attn))
    registry=dict(u.ALL_TOKEN_SELECTION_METHOD)
    with CompressionCapture() as c:
        config=FlashVidConfig(retention_ratio=.1,do_segment=True,segment_threshold=.9,min_segment_num=8,
                              complementary_segment=True,alpha=.7,temporal_threshold=.8,expansion=1.25,
                              token_selection_method='attn_div_v2')
        c.direct(pooled,attn,config)
        lengths,keep=c.selected()
        assert keep.shape==(64,16) and lengths.sum()==64
        assert any(x['call']=='original_selector' and x['original']['qualname']=='attn_div_v2_based_token_selection' for x in c.calls)
    assert (tensor_hash(pooled),tensor_hash(attn))==initial
    assert all(u.ALL_TOKEN_SELECTION_METHOD[k] is v for k,v in registry.items())


def _golden(root, ids):
    coverage=[[v,'controlled-gray','identity'] for v in ids]
    metadata={'kind':'validation','coverage':A.ordered_set(coverage),'fixture_only':True}
    A.write_npz(root/'artifacts/golden.npz',{'fixture':np.arange(169,dtype='<i4')},metadata)
    descriptor,_=A.describe(root,'artifacts/golden.npz',['golden'])
    execution=A.new_execution();execution.update(status='COMPLETE',finished_at='2026-09-05T00:00:00+00:00',
       artifact_paths=[{'key':['golden'],'relative_path':'artifacts/golden.npz'}],
       workers=[{'shard_id':0,'status':'COMPLETE','exit_code':0,'coverage':A.ordered_set(coverage)}])
    science=A.normalize_science({'schema_version':A.SCHEMA_VERSION,'layer':'validation','plan_sha256':spec.PLAN_SHA256,
         'ordered_expected_ids':ids,'subset_kind':'SMOKE_5' if len(ids)==5 else 'FULL_300',
         'dataset_sha256':'1'*64,'definitions':{'fixture_only':True},'provenance_sha256':'2'*64,
         'numeric_environment_sha256':'3'*64,'expected_coverage':coverage,'artifacts':[descriptor],'upstreams':[]})
    manifest={'science':science,'execution':execution,'content_hash_spec':A.HASH_SPEC,'content_sha256':A.digest(A.canonical(science))}
    return manifest,coverage


def test_schema_full_and_smoke_and_hash_invariances(tmp_path):
    for count in (5,300):
        root=tmp_path/str(count);m,expected=_golden(root,['id_'+str(i) for i in range(count)])
        A.validate_manifest(m,root,expected)
        changed=copy.deepcopy(m);changed['execution']=dict(m['execution'],run_id=A.new_execution()['run_id'],started_at='2030-01-01',finished_at='2030-01-02')
        assert A.validate_manifest(changed,root,expected)==m['content_sha256']
        with pytest.raises(ValueError):A.validate_manifest(m,root,expected[:-1])
        swapped=copy.deepcopy(m);swapped['science']['expected_coverage'].reverse()
        assert A.digest(A.canonical(A.normalize_science(swapped['science'])))==m['content_sha256']


@pytest.mark.parametrize('fault',['path_escape','absolute_path','wrong_upstream','nan','failed_shard','missing_tag','science_bit','running_hash','extra_file','symlink','directory','mixed_purpose'])
def test_manifest_negative_fixtures(tmp_path,fault):
    m,expected=_golden(tmp_path,['id_'+str(i) for i in range(5)])
    if fault=='path_escape':m['execution']['artifact_paths'][0]['relative_path']='../x.npz'
    elif fault=='absolute_path':m['execution']['artifact_paths'][0]['relative_path']='/tmp/x.npz'
    elif fault=='wrong_upstream':
        m['science']['upstreams']=[{'role':'features','hash_spec':A.HASH_SPEC,'content_sha256':'4'*64}]
        m['execution']['upstreams']=[{'role':'features','run_id':A.new_execution()['run_id'],'manifest_path':'upstream/manifest.json','content_sha256':'5'*64}]
        m['content_sha256']=A.digest(A.canonical(m['science']))
    elif fault=='nan':m['science']['definitions']['value']=float('nan')
    elif fault=='failed_shard':m['execution']['workers'][0]['status']='FAILED'
    elif fault=='missing_tag':
        m['science']['artifacts'][0]['parsed']['metadata']['coverage'].pop()
        m['content_sha256']=A.digest(A.canonical(m['science']))
    elif fault=='science_bit':m['science']['definitions']['scientific_bit']=1
    elif fault=='running_hash':m['execution']['status']='RUNNING'
    elif fault=='extra_file':(tmp_path/'artifacts/extra.npz').write_bytes(b'extra')
    elif fault=='symlink':
        (tmp_path/'linked.npz').symlink_to(tmp_path/'artifacts/golden.npz');m['execution']['artifact_paths'][0]['relative_path']='linked.npz'
    elif fault=='directory':m['execution']['artifact_paths'][0]['relative_path']='artifacts'
    elif fault=='mixed_purpose':m['science']['expected_coverage'][0].append('geometry_arm')
    with pytest.raises(Exception):A.validate_manifest(m,tmp_path,expected)


def test_NPZ_byte_equality_and_execution_field_rejection(tmp_path):
    metadata={'kind':'validation','coverage':[['unit']],'fixture_only':True}
    a=np.arange(24,dtype='<f4').reshape(4,6)
    A.write_npz(tmp_path/'one.npz',{'b':a,'a':a+1},metadata)
    A.write_npz(tmp_path/'two.npz',{'a':a+1,'b':a},dict(reversed(list(metadata.items()))))
    assert (tmp_path/'one.npz').read_bytes()==(tmp_path/'two.npz').read_bytes()
    with pytest.raises(ValueError):A.write_npz(tmp_path/'bad.npz',{'a':a},{**metadata,'generated_at':'2026'})
    with pytest.raises(FileExistsError):A.write_npz(tmp_path/'one.npz',{'a':a},metadata)
