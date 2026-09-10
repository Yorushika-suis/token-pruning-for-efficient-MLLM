"""Registered scientific work units; the parent coordinator owns run status."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import artifacts as A, geometry as G, jobs as J, runtime as R, selection as L, spec
from .capture import CompressionCapture, tensor_hash
from .numerics import CAPTURE_AXES
from .statistics import jsonable


def cpu_capture(capture):
    result = dict(capture)
    result['tensors'] = {k:t.detach().cpu().clone() for k,t in capture['tensors'].items()}
    for key in ('lengths','selected'):
        if key in capture:
            result[key] = capture[key].detach().cpu().clone()
    return result


def capture_arrays(capture):
    arrays = {k:t.detach().cpu().numpy() for k,t in capture.get('tensors',{}).items()}
    arrays.update({k:capture[k].detach().cpu().numpy() for k in ('lengths','selected') if k in capture})
    return arrays


def trace(capture):
    return {k:v for k,v in capture.items() if k not in ('tensors','lengths','selected')}


def call_with_evidence(function, args, label, failure_root, context):
    try:
        return function(*args)
    except BaseException as exc:
        failure_root=Path(failure_root)
        if failure_root.exists():
            exc.add_note('Earlier first-difference evidence retained at '+str(failure_root))
            raise
        failure_root.mkdir(parents=True,exist_ok=False)
        arrays={k:(v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v))
                for k,v in dict(context,**getattr(exc,'arrays',{})).items()}
        A.write_failure_npz(failure_root/'observation_capture.npz',arrays,{'label':label})
        J.put(failure_root/'difference.json',{'label':label,'type':type(exc).__name__,'error':str(exc),
                                           'traceback':traceback.format_exc(),'members':sorted(arrays)})
        raise


def compare_or_save(actual, reference, label, failure_root):
    def save(comparison=None,error=None):
        directory=Path(failure_root)
        if directory.exists():return False
        directory.mkdir(parents=True,exist_ok=False)
        # Full original dtypes, integer ordering and unrounded source inputs.
        for side,capture in [('actual',actual),('reference',reference)]:
            A.write_failure_npz(directory/f'{side}.npz',capture_arrays(capture),
                        {'kind':'validation','coverage':[[label,side]],'failure_capture':True})
            J.put(directory/f'{side}_trace.json',trace(capture))
        J.put(directory/'difference.json',{'label':label,'comparison':comparison,'exception':error})
        return True
    try:
        comparison = R.compare_captures(actual,reference)
    except BaseException as exc:
        save(error={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        exc.add_note('First-difference evidence retained at '+str(failure_root))
        raise
    if not comparison['passed']:
        save(comparison=comparison)
        raise ValueError('first numerical/discrete difference: '+label)
    return comparison


def compress_capture(evaluation, vision, budget):
    config=copy.deepcopy(evaluation.model.flashvid_config)
    config.retention_ratio=int(budget[1:])/100
    # Each call gets new tensors; the stored vision capture remains immutable.
    pooled=vision['tensors']['pooled169'].clone()
    attention=vision['tensors']['attn169'].clone()
    before={k:tensor_hash(vision['tensors'][k]) for k in ('pooled169','attn169')}
    with CompressionCapture() as comp:
        comp.direct(pooled,attention,config)
        lengths,selected=comp.selected()
    if before!={k:tensor_hash(vision['tensors'][k]) for k in before}:
        raise ValueError('budget execution mutated stored pre-compression snapshots')
    return dict(vision,lengths=lengths,selected=selected,compression_trace=comp.calls,
                snapshot_hashes=comp.snapshot_hashes,registry=comp.registry_identities)


def accept_or_save(pooled, attention, method, budget, label, failure_root):
    try:
        return L.acceptance(pooled,attention,method,budget)
    except Exception as exc:
        failure_root=Path(failure_root)
        if failure_root.exists():
            exc.add_note('Earlier first-difference evidence already exists at '+str(failure_root))
            raise
        failure_root.mkdir(parents=True,exist_ok=False)
        arrays={'input_pooled169':pooled.detach().cpu().numpy(),
                'input_attn169':attention.detach().cpu().numpy(),**getattr(exc,'arrays',{})}
        A.write_failure_npz(failure_root/'selector_capture.npz',arrays,
                    {'kind':'validation','coverage':[[label,method,budget]],'failure_capture':True})
        J.put(failure_root/'difference.json',{'label':label,'method':method,'budget':budget,
               'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
               'capture_members':sorted(arrays)})
        raise


def legacy_seven_tag_regression(video_id,work):
    """Supplemental saved-input diagnostic; nine absent tags are never passes."""
    feature_path=R.PROJECT/'runs/stage2_features'/f'{video_id}.npz'
    mask_path=R.PROJECT/'runs/stage1'/f'{video_id}__native.npz'
    raw_feature=feature_path.read_bytes();raw_mask=mask_path.read_bytes()
    hashes={'legacy_feature_sha256':A.digest(raw_feature),'legacy_native_mask_sha256':A.digest(raw_mask)}
    with np.load(io.BytesIO(raw_feature),allow_pickle=False) as f,np.load(io.BytesIO(raw_mask),allow_pickle=False) as old:
        pooled=torch.from_numpy(f['pooled169'].copy()).to('cuda:0')
        attention=torch.from_numpy(f['attn169'].copy()).to('cuda:0')
        pairs=[(m,'R010') for m in spec.INTERNAL]+[('adts_v2',b) for b in ('R015','R020','R025')]
        names={'adts_v2':'attn_div_v2','adts_v1':'attn_div','attn':'attn','FlashVID-div-ablation@P1U169':'div'}
        expected_tags={b+'_'+names[m] for m,b in pairs}
        if {k[5:] for k in old.files if k.startswith('keep_')}!=expected_tags:
            raise ValueError('legacy diagnostic does not contain exactly the seven recorded tags')
        rows=[];evidence={};attn_exact=np.array_equal(f['attn169'],old['cls_attn'])
        for method,budget in pairs:
            tag=budget+'_'+names[method]
            bounds,selected=accept_or_save(pooled,attention,method,budget,'legacy/'+video_id,work/'first_difference')
            current=selected.cpu().numpy();reference=old['keep_'+tag].reshape(64,spec.BUDGETS[budget])-169*np.arange(64)[:,None]
            original_bounds=old['seglen_'+tag]
            G.binary_mask(reference,spec.BUDGETS[budget],allow_duplicates=method=='adts_v2')
            boundaries_exact=np.array_equal(bounds.cpu().numpy(),original_bounds)
            indices_exact=np.array_equal(current,reference)
            rows.append({'tag':tag,'method':method,'budget':budget,'current_candidate_vs_independent_official_exact':True,
                         'historical_boundaries_exact':boundaries_exact,'historical_ordered_indices_exact':indices_exact,
                         'current_lengths':bounds.cpu().tolist(),'historical_lengths':original_bounds.tolist(),
                         'current_ordered_indices':current.tolist(),'historical_ordered_indices':reference.tolist(),
                         'different_return_slots':int(np.count_nonzero(current!=reference))})
            if not boundaries_exact or not indices_exact:
                prefix='case'+str(len(rows));evidence.update({prefix+'_current_indices':current,prefix+'_historical_indices':reference,
                    prefix+'_current_lengths':bounds.cpu().numpy(),prefix+'_historical_lengths':original_bounds})
        if not attn_exact:evidence.update(current_cached_attention=f['attn169'],historical_stage1_attention=old['cls_attn'])
        if evidence:
            A.write_failure_npz(work/'legacy_diagnostics'/f'{video_id}_difference.npz',evidence,
                                {'video_id':video_id,'diagnostic_only':True,**hashes})
    if A.file_hash(feature_path)!=hashes['legacy_feature_sha256'] or A.file_hash(mask_path)!=hashes['legacy_native_mask_sha256']:
        raise ValueError('legacy diagnostic input changed during comparison')
    return {'video_id':video_id,'diagnostic_only':True,'historical_runtime':'unresolved',**hashes,
            'source_attention_exact':attn_exact,'seven_present_tags':rows,
            'absent_not_tested':[b+'_'+names[m] for m in spec.INTERNAL if m!='adts_v2' for b in ('R015','R020','R025')],
            'status':'EXACT_ON_SAVED_INPUTS' if not evidence else 'DIFFERENCES_REQUIRE_PARENT_ASSESSMENT'}


def validate_video(evaluation,video_id,frames,indices,timestamps,task,question,work):
    """Three actual repeats; reverse budget order in repeat 1; 64 and 16x4."""
    rows=[]
    context={'expected_decoded_frames':frames,'expected_frame_indices':indices,'expected_frame_timestamps':timestamps}
    arms=spec.ARMS if task=='d4' else ('identity',)
    for protocol in spec.PROTOCOLS:
        for arm in arms:
            references={}
            for chunk in (64,16):
                for repeat in range(3):
                    budgets=list(spec.BUDGETS)
                    if repeat==1:
                        budgets.reverse()
                    started=time.monotonic()
                    vision=call_with_evidence(R.direct,(evaluation,frames,protocol,arm,chunk),
                                f'{video_id}/{protocol}/{arm}/chunk{chunk}/repeat{repeat}/direct_observation',work/'first_difference',context)
                    for budget in budgets:
                        direct=compress_capture(evaluation,vision,budget)
                        label=f'{video_id}/{protocol}/{arm}/{budget}/chunk{chunk}/repeat{repeat}/direct'
                        if budget not in references:
                            references[budget]=cpu_capture(direct)
                        result=compare_or_save(direct,references[budget],label,work/'first_difference')
                        selectors={}
                        for method in spec.INTERNAL:
                            bounds,selected=accept_or_save(vision['tensors']['pooled169'],vision['tensors']['attn169'],method,budget,
                                                          label,work/'first_difference')
                            selectors[method]={'lengths':bounds.cpu().tolist(),'raw_ordered_indices':selected.cpu().tolist(),
                                               'official_reference_exact':True}
                        row={'protocol':protocol,'arm':arm,'budget':budget,'chunk':chunk,'repeat':repeat,
                             'path':'direct','comparison':result,'selectors':selectors,
                             'lengths':direct['lengths'].cpu().tolist(),'selected':direct['selected'].cpu().tolist()}
                        rows.append(row)
                        J.put(work/'traces'/f'{video_id}_{protocol}_{arm}_{budget}_{chunk}_{repeat}_direct.json',trace(direct))
                        if task=='i6':
                            official=call_with_evidence(R.official_wrapper,
                                (evaluation,video_id,question,frames,indices,timestamps,protocol,budget,chunk),
                                label.replace('/direct','/official_observation'),work/'first_difference',context)
                            comparison=compare_or_save(official,references[budget],label.replace('/direct','/official'),work/'first_difference')
                            rows.append({'protocol':protocol,'arm':arm,'budget':budget,'chunk':chunk,'repeat':repeat,
                                         'path':'official_wrapper','comparison':comparison,
                                         'lengths':official['lengths'].cpu().tolist(),'selected':official['selected'].cpu().tolist(),
                                         'stopped_before_llm':official['stopped_before_llm']})
                            J.put(work/'traces'/f'{video_id}_{protocol}_{arm}_{budget}_{chunk}_{repeat}_official.json',trace(official))
                            del official
                    J.put(work/'timings'/f'{video_id}_{protocol}_{arm}_{chunk}_{repeat}.json',
                          {'elapsed_seconds':time.monotonic()-started,'configuration':[task,protocol,arm,chunk,repeat],
                           'budget_call_order':budgets})
                    del vision
            del references
    wanted={(p,a,b,c,r,path) for p in spec.PROTOCOLS for a in arms for b in spec.BUDGETS for c in (64,16)
            for r in range(3) for path in (('direct','official_wrapper') if task=='i6' else ('direct',))}
    observed={(x['protocol'],x['arm'],x['budget'],x['chunk'],x['repeat'],x['path']) for x in rows}
    if observed!=wanted or len(rows)!=len(wanted):
        raise ValueError('validation work omitted or repeated a planned case')
    legacy=legacy_seven_tag_regression(video_id,work) if task=='d4' else None
    return {'rows':rows,'tested_capture_points':list(CAPTURE_AXES),'legacy_seven_tag_diagnostic':legacy,
            'case_count':len(rows),'sampling_frame_indices':indices.tolist(),'sampling_timestamps':timestamps.tolist(),
            'I6_scope':'only_frozen_sample_and_capture_boundary_not_full_wrapper_or_final_tokens' if task=='i6' else None}


def masks_from_feature(root,shard_id,key,arrays,metadata,feature_descriptor):
    video_id,protocol,arm=key
    pooled=torch.from_numpy(arrays['pooled169']).to('cuda:0')
    attention=torch.from_numpy(arrays['attn169']).to('cuda:0')
    output,records,acceptance={},[],[]
    keys=A.ordered_set([k for k in spec.mask_keys([video_id]) if k[1]==protocol and k[2]==arm])
    deterministic={}
    for i,kkey in enumerate(keys):
        _,_,_,method,budget,seed,purpose=kkey
        nominal=spec.BUDGETS[budget]
        random_stream=None
        if method in spec.INTERNAL:
            cached=deterministic.get((method,budget))
            if cached is None:
                lengths,indices=accept_or_save(pooled,attention,method,budget,'/'.join(key),
                                               Path(root)/'workers'/str(shard_id)/'first_difference')
                cached=indices.cpu().numpy().astype('<i4')
                deterministic[method,budget]=cached
                acceptance.append({'method':method,'budget':budget,'full64_segment_lengths':lengths.cpu().tolist(),
                                   'candidate_vs_independent_official_exact':True})
            local=cached.copy()
        else:
            local,random_stream=L.control(video_id,method,budget,seed,purpose,arm)
        mask=G.binary_mask(local,nominal,allow_duplicates=method=='adts_v2')
        names={'raw_global_indices':f't{i:03d}_raw','packed_unique_mask':f't{i:03d}_mask','unique_k':f't{i:03d}_unique'}
        output[names['raw_global_indices']]=(local+169*np.arange(64)[:,None]).astype('<i4')
        output[names['packed_unique_mask']]=np.packbits(mask,axis=-1,bitorder='little')
        output[names['unique_k']]=mask.sum(-1).astype('<i2')
        records.append({'key':kkey,**names,'nominal_k':nominal,'algorithm_version':spec.METHOD_VERSIONS[method],
                        'random_stream':random_stream,'duplicate_slots':(nominal-mask.sum(-1)).tolist()})
    J.submit(root,shard_id,'_'.join(key),output,
             {'kind':'masks','coverage':keys,'records':records,'acceptance':acceptance,
              'upstream_feature_key':key,'upstream_feature_sha256':feature_descriptor['sha256']},key)


def execute(plan,shard,root,heartbeat):
    task=plan['task'];work=root/'workers'/str(shard['shard_id'])
    if task=='fixture':
        for video_id in shard['units']:
            J.submit(root,shard['shard_id'],video_id,{'probe':np.array([1],dtype='<i4')},
                     {'kind':'validation','coverage':[[video_id,'fixture']],'fixture_only':True},[video_id,'fixture'])
        return
    R.configure()
    from .provenance import numeric_environment
    if A.digest(A.canonical(numeric_environment()))!=plan['science_base']['numeric_environment_sha256']:
        raise ValueError('worker numeric environment differs from registered fingerprint')
    if task in ('analysis','analysis_timing'):
        from .analysis import execute_analysis
        execute_analysis(plan,shard,root,heartbeat)
        return
    features=None
    if task=='masks':
        features,feature_root=J.dependency(plan,'features')
        descriptors={tuple(a['key']):a for a in features['science']['artifacts']}
        paths={tuple(a['key']):a['relative_path'] for a in features['execution']['artifact_paths']}
    else:
        evaluation=R.build()
        provenance=J.read(plan['provenance_file'])
        if R.processor_contract(evaluation._image_processor)!=provenance['science']['actual_processor']:
            raise ValueError('actual original processor configuration differs from D3 registration')
        from .runtime_closure import assert_runtime_closed
        J.put(work/'runtime_closure_after_model_build.json',assert_runtime_closed(provenance))
        questions=None
        if task=='i6':
            manifest_bytes,_,_=R.dataset_snapshot()
            if A.digest(manifest_bytes)!=provenance['science']['dataset_sha256']:
                raise ValueError('I6 parent manifest differs from D3 frozen bytes')
            questions=R.first_questions(manifest_bytes=manifest_bytes)
        if questions is not None:
            if R.prompt_contract()!=provenance['science']['actual_prompt']:
                raise ValueError('actual I6 prompt/template differs from D3 registration')
            registered_questions={v:{'row_number':q['row_number'],'context':q['context'],
                                      'original_document_sha256':A.digest(A.canonical(q['doc']))}
                                  for v,q in questions.items()}
            if registered_questions!=provenance['science']['first_questions']:
                raise ValueError('actual I6 first questions differ from D3 registration')
    for video_id in shard['units']:
        if (root/'cancel.json').exists():
            raise RuntimeError('coordinator cancelled this run')
        heartbeat.assert_healthy()
        started=time.monotonic()
        if task=='masks':
            for key in spec.feature_keys([video_id]):
                descriptor,arrays=A.describe(feature_root,paths[tuple(key)],key)
                if descriptor!=descriptors[tuple(key)]:
                    raise ValueError('feature bytes changed between dependency validation and read')
                masks_from_feature(root,shard['shard_id'],key,arrays,descriptor['parsed']['metadata'],descriptor)
                del arrays
        else:
            frames,indices,timestamps,media=R.decode(R.PROJECT/'data/videos_short'/f'{video_id}.mp4')
            registered=plan['sampling'][video_id]
            if indices.tolist()!=registered['frame_indices'] or timestamps.tolist()!=registered['frame_timestamps']:
                raise ValueError('sampling differs from frozen D3 inputs')
            if A.digest(frames.tobytes())!=registered['decoded_frames_sha256'] or list(frames.shape)!=registered['decoded_shape'] or str(frames.dtype)!=registered['decoded_dtype']:
                raise ValueError('decoded frame bytes/shape/dtype differ from frozen D3 inputs')
            if task in ('d4','i6'):
                result=validate_video(evaluation,video_id,frames,indices,timestamps,task,
                                      questions[video_id] if questions else None,work)
                J.submit(root,shard['shard_id'],video_id,
                         {'validation_json':np.frombuffer(A.canonical(jsonable(result)),dtype=np.uint8)},
                         {'kind':'validation','coverage':[[video_id,task]],'task':task},[video_id,task])
            elif task=='features':
                for p in spec.PROTOCOLS:
                    for arm in spec.ARMS:
                        captured=R.direct(evaluation,frames,p,arm,64)
                        arrays={k:captured['tensors'][k].detach().cpu().numpy() for k in ('pooled169','attn169','sig729','attn729')}
                        arrays.update(frame_indices=indices,frame_timestamps=timestamps)
                        key=[video_id,p,arm]
                        J.submit(root,shard['shard_id'],'_'.join(key),arrays,
                                 {'kind':'features','coverage':[key],'input_protocol':p,'arm_id':arm,
                                  'pixels_sha256':tensor_hash(captured['tensors']['pixels']),
                                  'media_sha256':registered['media_sha256'],'encoder_chunk':64},key)
                        J.put(work/'traces'/('_'.join(key)+'.json'),trace(captured))
                        del captured,arrays
            else:
                raise ValueError('unregistered scientific task')
        J.put(work/'unit_timings'/f'{video_id}.json',{'elapsed_seconds':time.monotonic()-started,
                                                     'unit':video_id,'task':task,'time':J.now()})
