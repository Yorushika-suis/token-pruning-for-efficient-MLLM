"""D3 byte provenance for actual pinned inputs; never reconstruct old runtimes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import urllib.request

from . import artifacts as A, jobs as J, runtime as R, spec

OFFICIAL_COMMIT='983cce6e30d7a8012442bfc7557d3afa61b3572d'


def verify_official_tree(output):
    url='https://codeload.github.com/Fanziyang-v/FlashVID/tar.gz/'+OFFICIAL_COMMIT
    archive=Path(output)/'official_commit_archive.tar.gz'
    try:
        with urllib.request.urlopen(url,timeout=60) as response:
            data=response.read()
            effective_url=response.url
        if not effective_url.startswith('https://codeload.github.com/'):
            raise ValueError('official archive redirected outside its registered origin')
        A.atomic_bytes(archive,data)
    except BaseException as exc:
        J.put(Path(output)/'official_archive_error.json',{'time':J.now(),'url':url,'type':type(exc).__name__,
                    'error':str(exc),'body':exc.read().decode(errors='replace') if hasattr(exc,'read') else None})
        raise
    expected={}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            parts=Path(member.name).parts
            if not parts or parts[0]!='FlashVID-'+OFFICIAL_COMMIT:
                raise ValueError('unexpected official archive prefix')
            if member.isdir():continue
            if not member.isfile() or len(parts)<2 or '..' in parts:
                raise ValueError('unexpected source archive member type/path')
            relative=Path(*parts[1:]).as_posix()
            if relative in expected:raise ValueError('duplicate archive member')
            content=tar.extractfile(member).read()
            expected[relative]={'sha256':A.digest(content),'bytes':len(content)}
    current={}
    for path in (R.PROJECT/'code/FlashVID').rglob('*'):
        if any(x in ('.git','__pycache__') for x in path.parts):continue
        if path.is_symlink():raise ValueError('unexpected symlink in current source')
        if path.is_file():current[str(path.relative_to(R.PROJECT/'code/FlashVID'))]={'sha256':A.file_hash(path),'bytes':path.stat().st_size}
    if current!=expected:
        mismatch=[{'path':path,'current':current.get(path),'official':expected.get(path)}
                  for path in sorted(set(current)|set(expected)) if current.get(path)!=expected.get(path)]
        J.put(Path(output)/'source_tree_difference.json',{'time':J.now(),'differences':mismatch})
        raise ValueError('current source tree does not match the exact official commit')
    return {'commit':OFFICIAL_COMMIT,'archive_sha256':A.file_hash(archive),'url':url,
            'files':A.ordered_set([dict(logical_path=k,**v) for k,v in expected.items()],key=lambda x:x['logical_path']),
            'exact_current_tree_match':True}


def file_record(path, logical):
    path=Path(path)
    actual=path.resolve(strict=True)
    if not actual.is_file():raise ValueError('source is not a regular file')
    before=actual.stat();sha=A.file_hash(actual);after=actual.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('source changed during hashing')
    return {'logical_path':logical,'sha256':sha,'bytes':after.st_size}, {
        'logical_path':logical,'physical_path':str(actual),'declared_path':str(path),
        'sha256':sha,'bytes':after.st_size,'symlink_target':os.readlink(path) if path.is_symlink() else None}


def numeric_environment():
    import torch
    names=('torch','transformers','numpy','decord','accelerate','flash-attn','evaluate','sqlitedict','datasets','jsonschema',
           'pillow','safetensors','einops','huggingface-hub','pyarrow','matplotlib')
    command=['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']
    gpu=subprocess.run(command,capture_output=True,text=True,check=True)
    return {'python':platform.python_version(),'machine':platform.machine(),'libraries':{n:importlib.metadata.version(n) for n in names},
            'torch_runtime_version':torch.__version__,'torch_cuda':torch.version.cuda,
            'cudnn_version':torch.backends.cudnn.version(),'gpu_models_and_drivers':sorted(set(gpu.stdout.splitlines())),
            'precision':'torch.float16','attention_backend':'flash_attention_2','encoder_chunk':64,
            'accepted_validation_chunks':[64,16],'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'torch_threads':torch.get_num_threads()}


def sampling_one(video_id):
    path=R.PROJECT/'data/videos_short'/f'{video_id}.mp4'
    if path.is_symlink():raise ValueError('media symlink is not a registered input')
    record,mapping=file_record(path,'media/'+video_id+'.mp4')
    frames,indices,timestamps,media=R.decode(path)
    sample={'video_id':video_id,'media_sha256':record['sha256'],'frame_indices':indices.tolist(),
            'frame_timestamps':timestamps.tolist(),'total_frames':media['total_frames'],'fps':media['fps'],
            'decoded_frames_sha256':A.digest(frames.tobytes()),'decoded_shape':list(frames.shape),'decoded_dtype':str(frames.dtype)}
    if A.file_hash(path)!=record['sha256']:raise ValueError('media changed during decode')
    return record,mapping,sample


def assert_registered_official(records,official_source):
    prefix='code/FlashVID/'
    registered={r['logical_path'][len(prefix):]:{'bytes':r['bytes'],'sha256':r['sha256']}
                for r in records if r['logical_path'].startswith(prefix)}
    expected={r['logical_path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in official_source['files']}
    if registered!=expected:
        raise ValueError('registered source bytes differ from verified official commit')


def assert_media_entries(ids,directory):
    entries=list(Path(directory).iterdir())
    if {p.name for p in entries}!={v+'.mp4' for v in ids} or any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError('media directory contains missing, extra or nonregular entries')


def build(output):
    """Invoke only after D2 passes; this function does no GPU model inference."""
    output=Path(output)
    if output.exists():raise FileExistsError(str(output))
    gate_bytes=(R.PROJECT/'docs/implementation/v4_20260905/gate_status.json').read_bytes()
    gate=json.loads(gate_bytes)
    if gate['D2']!='COMPLETE':raise ValueError('D2 code/schema/fixtures review gate is not complete')
    output.mkdir(parents=True,exist_ok=False)
    A.atomic_bytes(output/'D2_gate_evidence.json',gate_bytes)
    R.configure()
    manifest_bytes,data,ids=R.dataset_snapshot()
    official_source=verify_official_tree(output)
    from data_manifest import require_video_files
    from adts_harness import resolve_cached_model_sources
    from lmms_eval.models.simple.llava_vid import LlavaVid
    require_video_files(ids,R.PROJECT/'data/videos_short')
    assert_media_entries(ids,R.PROJECT/'data/videos_short')
    if len(ids)!=300:raise ValueError('census does not contain 300 videos')
    records,mappings=[],[]
    def add(path,logical):
        record,mapping=file_record(path,logical);records.append(record);mappings.append(mapping)
    # Complete official source snapshot and this executable implementation.
    for relative in ('code/FlashVID','scripts/v4'):
        for path in sorted((R.PROJECT/relative).rglob('*')):
            if any(p in ('.git','__pycache__') for p in path.parts):continue
            if path.is_file():add(path,str(path.relative_to(R.PROJECT)))
    for relative in ('scripts/adts_harness.py','scripts/data_manifest.py','data/short_subset.json',
                     'docs/implementation/v4_20260905/d2_i6_selection_fixture.json',
                     'docs/reviews/v3_v4_20260905/v4_candidate_a_r4.md'):
        add(R.PROJECT/relative,relative)
    parent_record=next(r for r in records if r['logical_path']=='data/short_subset.json')
    if A.digest(manifest_bytes)!=parent_record['sha256'] or len(manifest_bytes)!=parent_record['bytes']:
        raise ValueError('parent manifest buffer differs from registered source bytes')
    source_directories=[]
    for snapshot in resolve_cached_model_sources():
        snapshot_files=sorted(p for p in snapshot.rglob('*') if p.is_file())
        source_directories.append({'declared_root':str(snapshot),'expected_relative_files':sorted(str(p.relative_to(snapshot)) for p in snapshot_files)})
        for path in snapshot_files:add(path,str(path.relative_to(R.PROJECT)))
        add(snapshot.parents[1]/'refs/main',str((snapshot.parents[1]/'refs/main').relative_to(R.PROJECT)))
    parquet_bytes=Path(data['parquet']).read_bytes()
    add(data['parquet'],'dataset/Video-MME/test-00000-of-00001.parquet')
    if A.digest(parquet_bytes)!=records[-1]['sha256']:
        raise ValueError('question parquet buffer differs from registered source bytes')
    # Registered only for the approved supplemental seven-tag legacy diagnostic.
    for video_id in ids[:20]:
        for relative in (f'runs/stage2_features/{video_id}.npz',f'runs/stage1/{video_id}__native.npz'):
            add(R.PROJECT/relative,'legacy_diagnostic_only/'+relative)
    questions={v:{'row_number':q['row_number'],'context':q['context'],
                  'original_document_sha256':A.digest(A.canonical(q['doc']))}
               for v,q in R.first_questions(parquet_bytes,manifest_bytes=manifest_bytes).items()}
    if set(questions)!=set(ids):raise ValueError('question coverage differs from frozen parent IDs')
    # Populate real processor and numeric libraries before collecting imports.
    processor=R.processor_contract()
    prompt=R.prompt_contract()
    numeric=numeric_environment()
    from .runtime_closure import runtime_inventory, assert_runtime_closed
    runtime_files,runtime_distributions=runtime_inventory()
    already={Path(x['physical_path']).resolve() for x in mappings}
    for path,logical in sorted(runtime_files.items()):
        if path not in already:
            add(path,logical)
            already.add(path)
    J.put(output/'source_hash_progress.json',{'time':J.now(),'registered_files':len(records)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        samples=list(pool.map(sampling_one,ids))
    for record,mapping,sample in samples:records.append(record);mappings.append(mapping)
    assert_registered_official(records,official_source)
    numeric=numeric_environment()
    science={'plan_sha256':spec.PLAN_SHA256,'official_commit':OFFICIAL_COMMIT,'official_source':official_source,
             'files':A.ordered_set(records,key=lambda x:x['logical_path']),'ordered_expected_ids':ids,
             'dataset_sha256':A.digest(manifest_bytes),
             'sampling':{sample['video_id']:sample for _,_,sample in samples},'first_questions':questions,
             'runtime_distributions':runtime_distributions,'actual_processor':processor,'actual_prompt':prompt,
             'processor_contract':{'frames':64,'force_sample':True,'decoded_library':'decord',
                  'sampling':'numpy.linspace(0,total_frames-1,64,dtype=int)','spatial_pool':'average',
                  'newline_position':'frame','conv_template':'qwen_1_5','add_time_instruction':False},
             'numeric_environment':numeric,'numeric_environment_sha256':A.digest(A.canonical(numeric)),
             'historical_runtime':'unresolved; old assets are not formal inputs'}
    science['legacy_diagnostic_scope']={'ordered_ids':ids[:20],'purpose':'supplemental_seven_recorded_tag_regression_only',
                                      'formal_feature_reuse':False,'nine_absent_tag_combinations_not_tested':True}
    A.check_science(science)
    provenance={'science':science,'content_hash_spec':A.HASH_SPEC,'content_sha256':A.digest(A.canonical(science)),
                'execution':{'time':J.now(),'process':J.process_identity(),'files':mappings,
                             'source_directories':source_directories,
                             'D2_gate_evidence_sha256':A.digest(gate_bytes),'D2_gate_evidence_file':'D2_gate_evidence.json',
                             'actual_evaluator_class_module':LlavaVid.__module__}}
    provenance['execution']['runtime_closure_check']=assert_runtime_closed(provenance)
    J.put(output/'provenance.json',provenance)
    J.source_guard({'guarded_files':mappings,'source_directories':source_directories})
    J.put(output/'COMPLETE.json',{'time':J.now(),'provenance_file_sha256':A.file_hash(output/'provenance.json'),
                               'content_sha256':provenance['content_sha256'],'registered_files':len(records)})
    return provenance
