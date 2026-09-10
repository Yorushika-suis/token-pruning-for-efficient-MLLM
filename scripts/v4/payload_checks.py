"""Semantic checks for JSON embedded in scientific numeric containers."""
import json
import numpy as np

from . import artifacts as A, spec, statistics as S
from .geometry import binary_mask
from .numerics import CAPTURE_AXES


def json_array(array):
    if array.dtype!=np.uint8 or array.ndim!=1:raise ValueError('nested JSON must be a one-dimensional uint8 encoding')
    value=json.loads(array.tobytes())
    A.check_science(value)
    if A.canonical(value)!=array.tobytes():raise ValueError('nested scientific JSON is not canonical')
    return value


def selection_record(row,budget):
    lengths=np.asarray(row['lengths']);indices=np.asarray(row['selected'])
    if lengths.ndim!=1 or lengths.dtype.kind not in 'iu' or np.any(lengths<=0) or lengths.sum()!=64:
        raise ValueError('invalid independently parsed validation segment lengths')
    binary_mask(indices,spec.BUDGETS[budget],allow_duplicates=True)


def numerical_report(report):
    if report.get('passed') is not True or set(report['numeric'])!=set(CAPTURE_AXES) or report['discrete']!={'lengths':True,'selected':True}:
        raise ValueError('incomplete or failed validation comparison')
    shapes={'pixels':[64,3,384,384],'sig729':[64,729,1152],'projected729':[64,729,3584],
            'pooled169':[64,169,3584],'attn729':[64,729],'attn169':[64,169]}
    for capture,r in report['numeric'].items():
        if r.get('passed') is not True or r.get('shape')!=shapes[capture]:
            raise ValueError('failed capture or wrong registered shape')
        if capture=='pixels':
            if r.get('exact') is not True:raise ValueError('pixels were not exact')
            continue
        if r['dtype'] in ('torch.float32','float32'):limits=(1e-5,1e-4,.99999)
        elif r['dtype'] in ('torch.float16','float16','torch.bfloat16','bfloat16'):limits=(1e-3,1e-2,.9999)
        else:raise ValueError('unregistered validation dtype')
        if (r['atol'],r['rtol'],r['required_minimum_cosine'])!=limits or not 0<=r['worst_scaled_error']<=1 or r['minimum_cosine']<limits[2] or r['zero_classification_mismatches']!=0 or r['vector_axis']!=-1:
            raise ValueError('numeric report violates the registered per-vector gate')


def validate_validation(value,meta):
    task=meta.get('task')
    if task not in ('d4','i6') or len(meta['coverage'])!=1 or meta['coverage'][0][1]!=task:
        raise ValueError('unregistered validation task or coverage')
    arms=spec.ARMS if task=='d4' else ('identity',)
    paths=('direct',) if task=='d4' else ('direct','official_wrapper')
    expected={(p,a,b,c,r,path) for p in spec.PROTOCOLS for a in arms for b in spec.BUDGETS for c in (64,16) for r in range(3) for path in paths}
    rows=value['rows'];observed=[tuple(row[k] for k in ('protocol','arm','budget','chunk','repeat','path')) for row in rows]
    if len(rows)!=len(expected) or set(observed)!=expected or value['case_count']!=len(rows) or set(value['tested_capture_points'])!=set(CAPTURE_AXES):
        raise ValueError('validation omitted, repeated or added a capture case')
    for row in rows:
        numerical_report(row['comparison']);selection_record(row,row['budget'])
        if row['path']=='official_wrapper':
            if row.get('stopped_before_llm') is not True:raise ValueError('missing LLM early-stop proof')
        else:
            if set(row['selectors'])!=set(spec.INTERNAL):raise ValueError('validation omitted an internal selector')
            for method,selector in row['selectors'].items():
                if selector['official_reference_exact'] is not True:raise ValueError('selector reference comparison failed')
                selection_record({'lengths':selector['lengths'],'selected':selector['raw_ordered_indices']},row['budget'])
                binary_mask(np.asarray(selector['raw_ordered_indices']),spec.BUDGETS[row['budget']],allow_duplicates=method=='adts_v2')
            original=row['selectors']['adts_v2']
            if row['lengths']!=original['lengths'] or row['selected']!=original['raw_ordered_indices']:
                raise ValueError('actual compression does not match independently verified v2 boundaries/indices')
    for key in ('sampling_frame_indices','sampling_timestamps'):
        x=np.asarray(value[key])
        if x.shape!=(64,) or not np.isfinite(x).all() or np.any(x<0) or np.any(np.diff(x)<0):
            raise ValueError('invalid sampling values in validation payload')
    if task=='d4':
        legacy=value['legacy_seven_tag_diagnostic'];rows=legacy['seven_present_tags']
        names={'adts_v2':'attn_div_v2','adts_v1':'attn_div','attn':'attn','FlashVID-div-ablation@P1U169':'div'}
        expected={b+'_'+names[m] for m in spec.INTERNAL for b in spec.BUDGETS if b=='R010' or m=='adts_v2'}
        absent={b+'_'+names[m] for m in spec.INTERNAL for b in spec.BUDGETS if b!='R010' and m!='adts_v2'}
        if legacy['diagnostic_only'] is not True or legacy['video_id']!=meta['coverage'][0][0] or len(rows)!=7 or {r['tag'] for r in rows}!=expected or set(legacy['absent_not_tested'])!=absent:
            raise ValueError('legacy seven-tag diagnostic coverage is incomplete or mislabeled')
        if any(r['current_candidate_vs_independent_official_exact'] is not True for r in rows):
            raise ValueError('current-source independent reference failed in legacy diagnostic')
        exact=legacy['source_attention_exact'] and all(r['historical_boundaries_exact'] and r['historical_ordered_indices_exact'] for r in rows)
        if legacy['status']!=('EXACT_ON_SAVED_INPUTS' if exact else 'DIFFERENCES_REQUIRE_PARENT_ASSESSMENT'):
            raise ValueError('legacy difference was not truthfully classified')


def validate_na_tree(value):
    if isinstance(value,dict):
        if 'value' in value and value['value'] is None:
            if any(not isinstance(value.get(k),str) or not value[k] for k in ('reason','layer')):
                raise ValueError('NA record requires an explicit reason and applicable layer')
            if any(type(value.get(k)) is not int or value[k]<0 for k in ('valid_count','invalid_count')):
                raise ValueError('NA record requires complete nonnegative integer valid/invalid counts')
        for child in value.values():validate_na_tree(child)
    elif isinstance(value,list):
        for child in value:validate_na_tree(child)


def validate_analysis(value,arrays,meta):
    validate_na_tree(value)
    timing=meta.get('calibration_only',False)
    method,budget,task=meta['coverage'][0]
    if len(meta['coverage'])!=1 or method not in spec.METHODS or budget not in spec.BUDGETS or task!=('analysis_timing' if timing else 'analysis'):
        raise ValueError('analysis logical identity mismatch')
    if value['method']!=method or value['budget']!=budget or value['nominal_k']!=spec.BUDGETS[budget] or value['effective_seeds']!=list(spec.seed_set(method)):
        raise ValueError('analysis method, budget or effective seed identity mismatch')
    ids=value['ordered_video_ids']
    if len(ids)!=300 or len(set(ids))!=300:raise ValueError('analysis requires 300 explicitly identified rows')
    if timing:
        if value.get('calibration_only') is not True or value.get('not_an_inferential_estimate') is not True or value['source_video_ids']!=meta['ordered_expected_ids'] or len(meta['ordered_expected_ids'])!=20:
            raise ValueError('timing payload lacks synthetic-population declaration')
        if value['source_row_indices']!=(np.arange(300)%20).tolist():raise ValueError('calibration replication differs from fixed rule')
    elif ids!=meta['ordered_expected_ids'] or value.get('calibration_only'):
        raise ValueError('formal analysis IDs differ from the manifest census')
    indices=arrays['bootstrap_indices']
    if indices.dtype.str!='<i8' or not np.array_equal(indices,S.bootstrap_indices()):
        raise ValueError('analysis does not contain the registered shared bootstrap table')
    if meta['bootstrap_seed']!=20260905 or meta['bootstrap_replicates']!=2000 or meta['statistical_definition']!='v4-A-r4-1':
        raise ValueError('analysis statistical configuration mismatch')
    if set(value['position'])!=set(spec.PROTOCOLS) or set(value['geometry'])!=set(spec.ARMS[1:]) or set(value['coverage'])!=set(spec.PROTOCOLS):
        raise ValueError('analysis omitted a protocol or geometry arm')
    for p in spec.PROTOCOLS:
        point=value['position'][p]
        for key in ('mean_rate','S','budget_centered_S'):
            if np.asarray(point[key]).shape!=(169,):raise ValueError('position map has wrong shape')
        if np.asarray(point['per_video_rate']).shape!=(300,169) or len(point['per_seed'])!=len(spec.seed_set(method)):
            raise ValueError('video/seed position result coverage mismatch')
        for key in ('P','Q','budget_centered_Q','mean_rate','S','budget_centered_S','video_level','pooled_marginal_rel_logodds'):
            if key not in point['seed_spread']:raise ValueError('seed spread omitted')
        if set(value['coverage'][p])!={purpose+'/'+arm for purpose,arm in spec.PURPOSE_ARMS}:
            raise ValueError('coverage statistics mixed or omitted purpose/arm')
        for arm in spec.ARMS[1:]:
            entry=value['geometry'][arm][p]
            for key in ('stay','follow','delta','hamming_per_slot','jaccard','original_unique_k','transformed_unique_k'):
                if np.asarray(entry[key]['per_video']).shape!=(300,):raise ValueError('geometry video results missing')
            if set(entry['forensic'])!={'1','5','10'}:raise ValueError('forensic sensitivities omitted')
    point_diff=value['protocol_difference']
    for key in ('P','Q'):
        if point_diff[key]!=value['position']['unblanked-native'][key]-value['position']['controlled-gray'][key]:
            raise ValueError('paired protocol point difference is inconsistent')
    if np.asarray(point_diff['per_video_rate_difference']).shape!=(300,169):raise ValueError('paired video rate difference map missing')
    wanted={'analysis_json','bootstrap_indices'}
    undefined=meta.get('undefined_replicates')
    allowed={f'p{pi}_{metric}' for pi in (0,1) for metric in ('P','Q','budget_centered_Q','video_B10','pooled_B10')}
    if not isinstance(undefined,dict) or set(undefined)-allowed:
        raise ValueError('missing or unregistered undefined replicate declaration')
    for pi in (0,1):
        for metric in ('P','Q','budget_centered_Q','video_B10','pooled_B10'):
            name=f'p{pi}_{metric}';shape=(2000,169) if metric.endswith('B10') else (2000,)
            if name in undefined:
                if name in arrays or len(undefined[name])!=2000:raise ValueError('undefined traces dropped or duplicated')
                invalid=0
                for i,row in enumerate(undefined[name]):
                    if not isinstance(row,dict) or type(row.get('replicate')) is not int or row['replicate']!=i or 'value' not in row:
                        raise ValueError('undefined trace has invalid replicate identity')
                    if row['value'] is None:
                        validate_na_tree(row);invalid+=1
                        if row['layer']!='bootstrap_replicate' or row['valid_count']!=0 or row['invalid_count']!=1:
                            raise ValueError('NA bootstrap replicate has inconsistent layer/counts')
                    else:
                        values=np.asarray(row['value'])
                        if values.shape!=shape[1:] or values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
                            raise ValueError('finite bootstrap replicate has invalid shape/domain')
                interval=value['position_intervals'][spec.PROTOCOLS[pi]][metric]
                if interval['valid_count']!=2000-invalid or interval['invalid_count']!=invalid or (invalid>0)!=(interval['value'] is None):
                    raise ValueError('bootstrap interval NA counts differ from preserved replicates')
            else:
                wanted.add(name)
                if arrays[name].shape!=shape or arrays[name].dtype.str!='<f8':raise ValueError('bootstrap trace shape/dtype mismatch')
    if set(arrays)!=wanted:raise ValueError('unregistered or missing analysis array')


def validate_json_payload(arrays,meta):
    if meta.get('fixture_only') is True:
        if not arrays:raise ValueError('empty D2 fixture')
        return
    key='validation_json' if meta['kind']=='validation' else 'analysis_json'
    if key not in arrays:raise ValueError('scientific JSON payload is missing')
    value=json_array(arrays[key])
    if meta['kind']=='validation':
        if set(arrays)!={key}:raise ValueError('unregistered validation array')
        validate_validation(value,meta)
    else:validate_analysis(value,arrays,meta)
