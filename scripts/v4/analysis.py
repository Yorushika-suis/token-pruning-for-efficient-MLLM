"""Descriptive outputs for the fixed 300-video census, without claim gates."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from . import artifacts as A, geometry as G, jobs as J, selection as L, spec, statistics as S


def spread(values, layer):
    x=np.asarray(values,dtype=np.float64)
    valid=np.isfinite(x).reshape(len(x),-1).all(1)
    if not valid.all():
        return S.na('at_least_one_seed_result_is_undefined',layer,valid.sum(),(~valid).sum())
    return {'values':x,'range':np.stack([x.min(0),x.max(0)]),
            'sample_SD':x.std(0,ddof=1) if len(x)>1 else S.na('one_effective_seed',layer,1,0)}


def summarize_frame_metric(values, ids, indices):
    """Frame -> seed -> video, then the common video cluster bootstrap."""
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(len(ids),x.shape[1],64) or x.shape[1] not in (1,5) or not np.isfinite(x).all():
        raise ValueError('invalid primary frame metric')
    per_video_seed=x.mean(-1)
    per_video=per_video_seed.mean(-1)
    draws=np.asarray([per_video[draw].mean() for draw in indices])
    return {'value':per_video.mean(),'per_video':per_video,'per_video_seed':per_video_seed,
            'per_seed':per_video_seed.mean(0),'seed_spread':spread(per_video_seed.mean(0),'frame_metric_seed_spread'),
            'interval':S.interval(draws,'video_cluster_bootstrap_frame_metric'),
            'valid_video_count':len(ids),'invalid_video_count':0,'aggregation':'frames_then_seeds_then_videos'},draws


def collect_masks(plan):
    manifest,root=J.dependency(plan,'masks')
    ids=plan['ids'];video_index={v:i for i,v in enumerate(ids)}
    groups={};slot_counts={};seen=set()
    paths={A.canonical(p['key']):p['relative_path'] for p in manifest['execution']['artifact_paths']}
    for declared in manifest['science']['artifacts']:
        observed,arrays=A.describe(root,paths[A.canonical(declared['key'])],declared['key'])
        if observed!=declared:
            raise ValueError('mask artifact changed during analysis reading')
        metadata=observed['parsed']['metadata']
        for record in metadata['records']:
            v,p,arm,m,b,seed,purpose=record['key']
            full=tuple(record['key'])
            if full in seen:
                raise ValueError('duplicate analysis mask tuple')
            seen.add(full)
            key=(m,b,p,purpose,arm)
            if key not in groups:
                groups[key]=np.zeros((len(ids),len(spec.seed_set(m)),64,169),dtype=bool)
                slot_counts[key]=np.zeros((len(ids),len(spec.seed_set(m)),169),dtype='<i4')
            raw=arrays[record['raw_global_indices']]-169*np.arange(64)[:,None]
            mask=G.binary_mask(raw,spec.BUDGETS[b],allow_duplicates=m=='adts_v2')
            groups[key][video_index[v],spec.seed_set(m).index(seed)]=mask
            slot_counts[key][video_index[v],spec.seed_set(m).index(seed)]=np.bincount(raw.reshape(-1),minlength=169)
    if seen!={tuple(k) for k in spec.mask_keys(ids)}:
        raise ValueError('analysis lacks exact complete mask coverage')
    return groups,slot_counts


def position_results(m,indices):
    result=S.summarize(m)
    result['per_seed_b10']=[S.b10_from_counts(m[:,s:s+1].sum(2),m[:,s:s+1].sum(-1).mean(2)) for s in range(m.shape[1])]
    for key in ('mean_rate','S','budget_centered_S'):
        result['seed_spread'][key]=spread([v[key] for v in result['per_seed']],key+'_seed_spread')
    for key in ('video_level','pooled_marginal_rel_logodds'):
        values=[v[key]['value'] for v in result['per_seed_b10']]
        result['seed_spread'][key]=spread([np.full(169,np.nan) if v is None else v for v in values],key+'_seed_spread')
    k=m.sum(-1)
    result['unique_k_distribution']={'values':np.unique(k).tolist(),'counts':np.unique(k,return_counts=True)[1].tolist(),
                                     'total_frames_including_seeds':k.size,'mean':k.mean(),
                                     'min':int(k.min()),'max':int(k.max())}
    return result


def group_result(groups,method,budget,ids,indices,slot_counts):
    masks={p:groups[method,budget,p,'protocol_pair','identity'] for p in spec.PROTOCOLS}
    point={p:position_results(masks[p],indices) for p in spec.PROTOCOLS}
    intervals,traces=S.paired_bootstrap(masks['controlled-gray'],masks['unblanked-native'],indices)
    result={'method':method,'budget':budget,'nominal_k':spec.BUDGETS[budget],'ordered_video_ids':ids,
            'effective_seeds':list(spec.seed_set(method)),'position':point,'position_intervals':intervals,
            'protocol_difference':{'P':point['unblanked-native']['P']-point['controlled-gray']['P'],
                                   'Q':point['unblanked-native']['Q']-point['controlled-gray']['Q'],
                                   'per_video_rate_difference':point['unblanked-native']['per_video_rate']-point['controlled-gray']['per_video_rate']},
            'geometry':{},'protocol_masks':{},'coverage':{},'control_theory':None}
    for p in spec.PROTOCOLS:
        unique=masks[p].sum(-1)
        binary_counts=masks[p].sum(2)
        raw_counts=slot_counts[method,budget,p,'protocol_pair','identity']
        result['position'][p]['binary_vs_returned_slot_counts']={
            'binary_counts_per_video_seed_position':binary_counts,
            'returned_slot_counts_per_video_seed_position':raw_counts,
            'excess_duplicate_counts_per_video_seed_position':raw_counts-binary_counts,
            'basis':'same_fresh_masks; records_the_effect_of_legacy_bincount_semantics; old_B10_files_preserved'}
        result['position'][p]['duplicate_slots']={'total':int((spec.BUDGETS[budget]-unique).sum()),
                                                'slot_fraction':float((spec.BUDGETS[budget]-unique).sum()/(unique.size*spec.BUDGETS[budget])),
                                                'frames_with_duplicates':int((unique<spec.BUDGETS[budget]).sum())}
        result['coverage'][p]={}
        for purpose,arm in spec.PURPOSE_ARMS:
            mask=groups[method,budget,p,purpose,arm]
            result['coverage'][p][purpose+'/'+arm],_=summarize_frame_metric(G.coverage(mask),ids,indices)
    gray,unblanked=masks['controlled-gray'],masks['unblanked-native']
    union=(gray|unblanked).sum(-1)
    if np.any(union==0):
        raise ValueError('empty protocol mask union')
    for name,value in [('hamming_per_slot',(gray^unblanked).mean(-1)),('jaccard',(gray&unblanked).sum(-1)/union)]:
        result['protocol_masks'][name],_=summarize_frame_metric(value,ids,indices)
    for arm in spec.ARMS[1:]:
        result['geometry'][arm]={}
        delta_draws={}
        for p in spec.PROTOCOLS:
            original=groups[method,budget,p,'geometry_arm','identity']
            transformed=groups[method,budget,p,'geometry_arm',arm]
            values=G.responses(original,transformed,arm)
            entry={}
            for key in ('stay','follow','delta','hamming_per_slot','jaccard','original_unique_k','transformed_unique_k'):
                entry[key],draws=summarize_frame_metric(values[key],ids,indices)
                if key=='delta':delta_draws[p]=draws
            entry['forensic']={}
            for removed in (1,5,10):
                sensitive=G.responses(original,transformed,arm,spec.HOTSPOTS[:removed])
                entry['forensic'][str(removed)]={}
                for key in ('stay','follow','delta'):
                    restricted=S.restricted_geometry_summary(sensitive[key],ids,indices)
                    per_seed=[S.restricted_geometry_summary(sensitive[key][:,s:s+1],ids,indices) for s in range(original.shape[1])]
                    restricted['per_seed']=per_seed
                    restricted['seed_spread']=spread([np.nan if v['value'] is None else v['value'] for v in per_seed],'restricted_geometry_seed_spread')
                    entry['forensic'][str(removed)][key]=restricted
            result['geometry'][arm][p]=entry
        result['geometry'][arm]['protocol_delta_difference']={
            'value':result['geometry'][arm]['unblanked-native']['delta']['value']-result['geometry'][arm]['controlled-gray']['delta']['value'],
            'interval':S.interval(delta_draws['unblanked-native']-delta_draws['controlled-gray'],'paired_protocol_geometry_delta_difference')}
    if method in spec.CONTROLS:
        probabilities=L.control_marginals(method,budget)
        roles={
            'random_uniform_without_replacement':'Uniform k/169 marginal inclusion. Shared protocol coordinates are by design and do not establish protocol equivalence.',
            'grid_uniform':'Fixed-coordinate nonuniform positive control and planar coverage baseline.',
            'spatial_stratified_legacy_rule':'Unequal-area legacy blocks with analytic marginal probabilities; not an equal-probability negative control.',
            'translated_fixed_grid_torus':'Uniform k/169 marginal inclusion under torus shifts; planar coverage radius is not preserved.'}
        result['control_theory']={'marginal_inclusion':probabilities,'marginal_P':probabilities.std(ddof=0),
                                  'role_and_limits':roles[method],
                                  'protocol_common_random_coordinates_are_by_design':method in spec.RANDOM,
                                  'independent_arm_expected_delta':{a:float(np.sum(probabilities**2-probabilities*probabilities[G.permutation(a)])/spec.BUDGETS[budget]) for a in spec.ARMS[1:]}}
    return result,traces


def figures(result, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for col,key in enumerate(('mean_rate','S','budget_centered_S','B10')):
        arrays=[]
        for p in spec.PROTOCOLS:
            value=result['position'][p]['b10']['video_level']['value'] if key=='B10' else result['position'][p][key]
            arrays.append(None if value is None else np.asarray(value).reshape(13,13))
        finite=[a for a in arrays if a is not None]
        lower=min(a.min() for a in finite) if finite else 0
        upper=max(a.max() for a in finite) if finite else 1
        for row,(p,array) in enumerate(zip(spec.PROTOCOLS,arrays)):
            ax=axes[row,col]
            if array is None:ax.text(.5,.5,'Not applicable',ha='center',va='center')
            else:
                im=ax.imshow(array,vmin=lower,vmax=upper,cmap='viridis',interpolation='nearest')
                fig.colorbar(im,ax=ax,shrink=.75)
            ax.set_title(p+'\n'+key)
            ax.set_xlabel('column');ax.set_ylabel('row')
    population='D4 timing fixture: 20 source videos repeated to 300 rows' if result.get('calibration_only') else 'fixed 300-video census'
    fig.suptitle(result['method']+' | '+result['budget']+' | '+population)
    fig.savefig(directory/'position.pdf');fig.savefig(directory/'position.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,arm in zip(axes,spec.ARMS[1:]):
        for p in spec.PROTOCOLS:
            values=np.asarray(result['geometry'][arm][p]['delta']['per_video'])
            ax.hist(values,bins=np.linspace(-1,1,41),histtype='step',label=p)
        ax.set_title(arm);ax.set_xlabel('stay - follow');ax.set_ylabel('videos');ax.legend()
    fig.savefig(directory/'geometry.pdf');fig.savefig(directory/'geometry.png',dpi=160);plt.close(fig)
    # Retain every video-position difference; a video mean alone would conceal
    # fixed-budget coordinate redistribution (its across-position mean is zero).
    difference=np.asarray(result['protocol_difference']['per_video_rate_difference'])
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained',gridspec_kw={'width_ratios':[2,1,1]})
    limit=max(float(np.abs(difference).max()),np.finfo(np.float64).eps)
    im=axes[0].imshow(difference,aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[0].set_xlabel('position j = 13r + c');axes[0].set_ylabel('video index in ordered_video_ids')
    axes[0].set_title('All per-video rate differences')
    fig.colorbar(im,ax=axes[0],label='unblanked-native minus controlled-gray')
    im=axes[1].imshow(difference.mean(0).reshape(13,13),interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[1].set_xlabel('column');axes[1].set_ylabel('row');axes[1].set_title('Mean position-rate difference')
    fig.colorbar(im,ax=axes[1],shrink=.7)
    for i,key in enumerate(('P','Q')):
        value=float(result['protocol_difference'][key])
        ci=result['position_intervals']['protocol_difference'][key]['value']
        if ci is not None:
            axes[2].plot([float(ci[0]),float(ci[1])],[i,i],color='black')
        else:axes[2].annotate('interval not applicable',(value,i),xytext=(0,10),textcoords='offset points')
        axes[2].plot(value,i,'o',color='black')
    axes[2].axvline(0,color='gray',linestyle=':');axes[2].set_yticks([0,1],['P','Q'])
    axes[2].set_xlabel('unblanked-native minus controlled-gray');axes[2].set_title('Paired points and 95% intervals')
    fig.suptitle(result['method']+' | '+result['budget']+' | '+population)
    fig.savefig(directory/'protocol_difference.pdf');fig.savefig(directory/'protocol_difference.png',dpi=160);plt.close(fig)
    J.put(directory/'sources.json',{'scientific_result_sha256':A.digest(A.canonical(S.jsonable(result))),
                                   'all_plotted_values_from_same_group_result':True,
                                   'mean_and_sd_colour_scales_shared_between_protocols':True})


def execute_analysis(plan,shard,root,heartbeat):
    timing=plan['task']=='analysis_timing'
    full_ids=J.read(J.PROJECT/'data/short_subset.json')['short_video_ids']
    if not timing and (plan['ids']!=full_ids or len(plan['ids'])!=300):
        raise ValueError('formal analysis requires the complete fixed census')
    if timing and (plan['ids']!=full_ids[:20] or not plan['science_base']['definitions'].get('calibration_only')):
        raise ValueError('timing fixture must derive only from the declared PILOT_20')
    indices=S.bootstrap_indices()
    groups,slot_counts=collect_masks(plan)
    analysis_ids=plan['ids']
    if timing:
        source_rows=np.arange(300)%20
        groups={k:v[source_rows] for k,v in groups.items()}
        slot_counts={k:v[source_rows] for k,v in slot_counts.items()}
        analysis_ids=[f'D4_timing_synthetic_row_{i:03d}' for i in range(300)]
    work=root/'workers'/str(shard['shard_id'])
    for method,budget in shard['units']:
        mi=spec.METHODS.index(method)
        heartbeat.assert_healthy()
        if (root/'cancel.json').exists():raise RuntimeError('analysis cancelled')
        started=time.monotonic()
        result,traces=group_result(groups,method,budget,analysis_ids,indices,slot_counts)
        if timing:
            result.update(calibration_only=True,not_an_inferential_estimate=True,
                          source_video_ids=plan['ids'],source_row_indices=source_rows)
        arrays={'analysis_json':np.frombuffer(A.canonical(S.jsonable(result)),dtype=np.uint8),
                'bootstrap_indices':indices.astype('<i8')}
        undefined={}
        for pi,(protocol,metrics) in enumerate(traces.items()):
            for metric,values in metrics.items():
                label=f'p{pi}_{metric}'
                if np.isfinite(values).all():arrays[label]=values.astype('<f8')
                else:
                    undefined[label]=[S.na('undefined_bootstrap_estimand','bootstrap_replicate',0,1,replicate=i)
                                      if not np.isfinite(v).all() else {'value':v.tolist(),'replicate':i}
                                      for i,v in enumerate(values)]
        key=[method,budget,plan['task']]
        name=f'm{mi:02d}_{budget}'
        J.submit(root,shard['shard_id'],name,arrays,
                 {'kind':'analysis','coverage':[key],'calibration_only':timing,'statistical_definition':'v4-A-r4-1',
                  'ordered_expected_ids':plan['ids'],
                  'undefined_replicates':undefined,'bootstrap_seed':20260905,'bootstrap_replicates':2000,
                  'interpretation':'descriptive_census_with_conditional_exchangeability_intervals'},key)
        figures(result,work/'figures'/name)
        J.put(work/'unit_timings'/f'{name}.json',{'elapsed_seconds':time.monotonic()-started,
                                                  'unit':[method,budget],'task':plan['task'],'time':J.now()})
