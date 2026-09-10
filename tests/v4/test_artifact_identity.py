"""D2 payload identity cannot be repaired by simply re-signing a bad manifest."""
import copy
import numpy as np
import pytest

from v4 import artifacts as A, geometry as G, selection as L, spec


def payload():
    key=['-ApL8d6tX5U','controlled-gray','identity',spec.RANDOM[0],'R010',0,'protocol_pair']
    local,stream=L.control(key[0],key[3],key[4],key[5],key[6],key[2])
    mask=G.binary_mask(local,16)
    names={'raw_global_indices':'t000_raw','packed_unique_mask':'t000_mask','unique_k':'t000_unique'}
    arrays={'t000_raw':(local+169*np.arange(64)[:,None]).astype('<i4'),
            't000_mask':np.packbits(mask,axis=-1,bitorder='little'),'t000_unique':mask.sum(-1).astype('<i2')}
    meta={'kind':'masks','coverage':[key],'records':[dict(key=key,**names,nominal_k=16,
                   algorithm_version=spec.METHOD_VERSIONS[key[3]],random_stream=stream,duplicate_slots=[0]*64)],
          'upstream_feature_key':key[:3],'upstream_feature_sha256':'1'*64}
    return arrays,meta


@pytest.mark.parametrize('fault',['wrong_version','wrong_nominal_budget','wrong_stream_seed','wrong_stream_material','wrong_stream_purpose','missing_stream','wrong_duplicates','missing_duplicates','bool_duplicates'])
def test_payload_rejects_resigned_identity_mismatch(fault):
    arrays,meta=payload();A.validate_payload(arrays,meta)
    broken=copy.deepcopy(meta);record=broken['records'][0]
    if fault=='wrong_version':record['algorithm_version']='unregistered/v999'
    elif fault=='wrong_nominal_budget':record['nominal_k']=23
    elif fault=='wrong_stream_seed':record['random_stream']['seed128']+=1
    elif fault=='wrong_stream_material':record['random_stream']['material_hex']='00'
    elif fault=='wrong_stream_purpose':record['key'][-1]='geometry_arm'
    elif fault=='missing_stream':record.pop('random_stream')
    elif fault=='wrong_duplicates':record['duplicate_slots'][0]=1
    elif fault=='missing_duplicates':record.pop('duplicate_slots')
    elif fault=='bool_duplicates':record['duplicate_slots'][0]=False
    with pytest.raises(ValueError):A.validate_payload(arrays,broken)


def test_internal_mask_requires_complete_acceptance_and_upstream_identity():
    arrays,meta=payload();key=meta['coverage'][0];key[3]='adts_v2';key[5]='deterministic'
    record=meta['records'][0];record['key']=key;record['algorithm_version']=spec.METHOD_VERSIONS[key[3]];record['random_stream']=None
    with pytest.raises(ValueError,match='acceptance records'):A.validate_payload(arrays,meta)
    accept={'method':'adts_v2','budget':'R010','full64_segment_lengths':[8]*8,'candidate_vs_independent_official_exact':True}
    meta['acceptance']=[accept];A.validate_payload(arrays,meta)
    for fault in ('inexact','zero_segment','missing_frame','duplicate','wrong_budget','missing_upstream','wrong_upstream'):
        bad=copy.deepcopy(meta)
        if fault=='inexact':bad['acceptance'][0]['candidate_vs_independent_official_exact']=False
        elif fault=='zero_segment':bad['acceptance'][0]['full64_segment_lengths']=[0,64]
        elif fault=='missing_frame':bad['acceptance'][0]['full64_segment_lengths']=[63]
        elif fault=='duplicate':bad['acceptance']*=2
        elif fault=='wrong_budget':bad['acceptance'][0]['budget']='R025'
        elif fault=='missing_upstream':bad.pop('upstream_feature_sha256')
        elif fault=='wrong_upstream':bad['upstream_feature_key']=['wrong']
        with pytest.raises(ValueError):A.validate_payload(arrays,bad)
