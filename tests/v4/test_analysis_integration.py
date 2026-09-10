"""Synthetic full-shape assembly checks; no empirical 300-video claims."""
import copy
import numpy as np
import pytest

from v4 import analysis as AN, artifacts as A, geometry as G, selection as L, spec, statistics as S


def test_nested_nonfinite_json_cannot_hide_inside_uint8():
    from v4.payload_checks import json_array
    for text in ('{"P":NaN}','{"value":[Infinity]}','{"value":-Infinity}'):
        with pytest.raises(ValueError):json_array(np.frombuffer(text.encode(),dtype=np.uint8))


def test_full_shape_analysis_assembly_point_differences_and_seed_outputs(tmp_path):
    method='grid_uniform';budget='R010';ids=[f'synthetic_{i:03d}' for i in range(300)]
    local=np.tile(L.grid_points(16),(64,1));mask=G.binary_mask(local,16)
    all_v=np.broadcast_to(mask,(300,1,64,169)).copy()
    counts=all_v.sum(2).astype('<i4')
    groups={(method,budget,p,purpose,arm):all_v for p in spec.PROTOCOLS for purpose,arm in spec.PURPOSE_ARMS}
    slots={key:counts for key in groups}
    indices=S.bootstrap_indices();result,traces=AN.group_result(groups,method,budget,ids,indices,slots)
    assert result['protocol_difference']['P']==0 and result['protocol_difference']['Q']==0
    assert np.count_nonzero(result['protocol_difference']['per_video_rate_difference'])==0
    assert 'nonuniform positive control' in result['control_theory']['role_and_limits']
    AN.figures(result,tmp_path/'figures')
    assert {f.name for f in (tmp_path/'figures').iterdir()}=={'position.pdf','position.png','geometry.pdf','geometry.png',
                                                          'protocol_difference.pdf','protocol_difference.png','sources.json'}
    for p in spec.PROTOCOLS:
        assert result['position'][p]['Q']==0
        assert result['position'][p]['duplicate_slots']['total']==0
        assert result['position'][p]['unique_k_distribution']['values']==[16]
        assert set(result['coverage'][p])=={purpose+'/'+arm for purpose,arm in spec.PURPOSE_ARMS}
        assert set(result['position'][p]['seed_spread'])=={'P','Q','budget_centered_Q','mean_rate','S','budget_centered_S','video_level','pooled_marginal_rel_logodds'}
        assert result['position'][p]['seed_spread']['mean_rate']['sample_SD']['value'] is None
    arrays={'analysis_json':np.frombuffer(A.canonical(S.jsonable(result)),dtype=np.uint8),'bootstrap_indices':indices.astype('<i8')}
    for pi,metrics in enumerate(traces.values()):
        arrays.update({f'p{pi}_{name}':value.astype('<f8') for name,value in metrics.items()})
    key=[method,budget,'analysis']
    meta={'kind':'analysis','coverage':[key],'calibration_only':False,'ordered_expected_ids':ids,
          'statistical_definition':'v4-A-r4-1','undefined_replicates':{},'bootstrap_seed':20260905,'bootstrap_replicates':2000}
    A.write_npz(tmp_path/'analysis.npz',arrays,meta);A.parse_npz((tmp_path/'analysis.npz').read_bytes())
    for fault in ('bootstrap','paired_point','video_map','missing_seed','extra_forensic','NA_without_reason','undefined_wrong_index','undefined_wrong_counts'):
        changed=dict(arrays);body=copy.deepcopy(S.jsonable(result))
        if fault=='bootstrap':changed['bootstrap_indices']=indices.copy();changed['bootstrap_indices'][0,0]=(changed['bootstrap_indices'][0,0]+1)%300
        elif fault=='paired_point':body['protocol_difference']['P']=1
        elif fault=='video_map':body['position'][spec.PROTOCOLS[0]]['per_video_rate'].pop()
        elif fault=='missing_seed':body['position'][spec.PROTOCOLS[0]]['seed_spread'].pop('mean_rate')
        elif fault=='extra_forensic':body['geometry'][spec.ARMS[1]][spec.PROTOCOLS[0]]['forensic']['99']={}
        elif fault=='NA_without_reason':body['position'][spec.PROTOCOLS[0]]['seed_spread']['mean_rate']['sample_SD'].pop('reason')
        changed_meta=copy.deepcopy(meta)
        if fault.startswith('undefined_'):
            name='p0_video_B10';changed.pop(name)
            changed_meta['undefined_replicates'][name]=[S.na('synthetic_undefined','bootstrap_replicate',0,1,replicate=i) for i in range(2000)]
            body['position_intervals'][spec.PROTOCOLS[0]]['video_B10']=S.na('synthetic_interval','video_cluster_bootstrap_video_B10',0,2000)
            if fault=='undefined_wrong_index':changed_meta['undefined_replicates'][name][1]['replicate']=0
            else:changed_meta['undefined_replicates'][name][1]['invalid_count']=2
        changed['analysis_json']=np.frombuffer(A.canonical(body),dtype=np.uint8)
        with pytest.raises(ValueError):A.validate_payload(changed,changed_meta)
