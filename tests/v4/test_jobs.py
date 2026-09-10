"""D2 operational counterexamples: process ownership, liveness and publication."""
from pathlib import Path
import contextlib
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from v4 import artifacts as A, jobs as J, spec


def fixture_plan(root):
    from data_manifest import load_video_ids
    ids=load_video_ids()[:2]
    e=A.new_execution();root=root/e['run_id'];root.mkdir()
    expected=A.ordered_set(J.expected_keys('fixture',ids))
    science={'schema_version':A.SCHEMA_VERSION,'layer':'validation','plan_sha256':spec.PLAN_SHA256,
             'ordered_expected_ids':ids,'subset_kind':'GOLDEN','dataset_sha256':'1'*64,
             'provenance_sha256':'2'*64,'numeric_environment_sha256':'3'*64,
             'definitions':{'fixture_only':True},'expected_coverage':expected,'artifacts':[],'upstreams':[]}
    shards=[{'shard_id':i,'units':[v],'gpu_uuid':None,'coverage':[[v,'fixture']],
             'deadline_seconds':86400} for i,v in enumerate(ids)]
    plan={'run_id':e['run_id'],'plan_sha256':spec.PLAN_SHA256,'task':'fixture','ids':ids,'subset_kind':'GOLDEN',
          'shards':shards,'expected_coverage':expected,'science_base':science,'guarded_files':[],
          'dependencies':[],'d4_unit_max_seconds':None}
    manifest={'science':A.normalize_science(science),'execution':e,'content_hash_spec':A.HASH_SPEC,'content_sha256':None}
    J.put(root/'plan.json',plan);J.put(root/'manifest.json',manifest)
    return root,plan,expected


def test_complete_requires_actual_successful_child_exits(tmp_path):
    root,plan,expected=fixture_plan(tmp_path)
    J.coordinate(root)
    manifest=J.read(root/'manifest.json')
    assert manifest['execution']['status']=='COMPLETE'
    assert len(manifest['execution']['workers'])==2
    for worker in manifest['execution']['workers']:
        assert worker['exit_code']==0 and not J.same_process(worker['process'])
        assert (root/'workers'/str(worker['shard_id'])/'result.json').exists()
    assert A.validate_manifest(manifest,root,expected)==manifest['content_sha256']
    with pytest.raises(ValueError):J.coordinate(root)


def test_failed_worker_cannot_publish_complete(tmp_path,monkeypatch):
    root,plan,expected=fixture_plan(tmp_path)
    real_popen=subprocess.Popen
    def crashing_child(command,**kwargs):
        return real_popen([sys.executable,'-B','-c','import time; time.sleep(0.3); raise SystemExit(23)'],**kwargs)
    monkeypatch.setattr(J.subprocess,'Popen',crashing_child)
    with pytest.raises(RuntimeError):J.coordinate(root)
    manifest=J.read(root/'manifest.json')
    assert manifest['execution']['status']=='FAILED'
    assert manifest['content_sha256'] is None
    assert (root/'cancel.json').exists()
    assert list(root.glob('failure_*.json'))
    with pytest.raises(Exception):A.validate_manifest(manifest,root,expected)


def test_heartbeat_identity_and_180_second_boundary(tmp_path):
    identity=J.process_identity();h=tmp_path/'heartbeat.json';current=time.monotonic()
    J.put(h,{'process':identity,'monotonic_seconds':current-181,'time':J.now()})
    with pytest.raises(TimeoutError):J.check_heartbeat(h,identity,current-181)
    J.put(h,{'process':dict(identity,start_ticks='wrong'),'monotonic_seconds':current,'time':J.now()},replace=True)
    with pytest.raises(ValueError):J.check_heartbeat(h,identity,current)
    with J.Heartbeat(h) as beat:
        J.check_heartbeat(h,identity,current)
        beat.assert_healthy()
    assert J.same_process(identity)


def test_lock_is_exclusive_and_deadlines_not_guessed(tmp_path):
    lock=tmp_path/'run.lock'
    with J.exclusive(lock):
        with pytest.raises(BlockingIOError):
            with J.exclusive(lock):pass
    assert J.fixed_deadline(11,40,full=True)==1920
    assert J.fixed_deadline(1,1,full=True)==1800
    with pytest.raises(ValueError):J.fixed_deadline(None,40,full=True)


def test_shard_overlap_and_source_mutation_fail_closed(tmp_path):
    root,plan,_=fixture_plan(tmp_path)
    plan['shards'][1]['coverage']=plan['shards'][0]['coverage']
    with pytest.raises(ValueError):J.verify_plan(plan,root)
    file=tmp_path/'source.py';file.write_bytes(b'original')
    plan['guarded_files']=[{'physical_path':str(file),'bytes':file.stat().st_size,'sha256':A.file_hash(file)}]
    J.source_guard(plan)
    file.write_bytes(b'changed!')
    with pytest.raises(ValueError):J.source_guard(plan)


def test_analysis_work_units_are_method_budget_groups():
    units=J.work_units('analysis',['v']*300)
    assert len(units)==32 and len({tuple(x) for x in units})==32
    assert J.shard_coverage('analysis',units)==A.ordered_set(J.expected_keys('analysis',[]))
    assert J.fixed_deadline(100,len(units),full=True)==10200
    with pytest.raises(ValueError):J.shard_coverage('analysis',[['attn','R999']])
    with pytest.raises(ValueError):J.shard_coverage('analysis',units+units[:1])
    assert J.work_units('analysis_timing',[])==units
    assert J.shard_coverage('analysis_timing',units)==A.ordered_set([[m,b,'analysis_timing'] for m,b in units])


def test_gpu_occupancy_is_rechecked_immediately_before_launch(monkeypatch):
    from types import SimpleNamespace
    state={'memory':0,'util':0,'process':''}
    def query(args,**kwargs):
        text=state['process'] if '--query-compute-apps=gpu_uuid,pid' in args else f"GPU-one, {state['memory']}, {state['util']}\n"
        return SimpleNamespace(returncode=0,stdout=text,stderr='')
    monkeypatch.setattr(J.subprocess,'run',query)
    plan={'shards':[{'gpu_uuid':'GPU-one'}]}
    assert J.allocated_gpus_idle(plan)['assigned_gpu_uuids']==['GPU-one']
    for key,value in [('memory',1),('util',1),('process','GPU-one, 123\n')]:
        state.update(memory=0,util=0,process='');state[key]=value
        with pytest.raises(ValueError,match='became occupied'):J.allocated_gpus_idle(plan)


def test_unknown_subset_and_task_combinations_cannot_skip_full_gates(tmp_path):
    from data_manifest import load_video_ids
    root,plan,_=fixture_plan(tmp_path)
    for task,subset in [('features','FULL_300X'),('masks','I6_PRESSURE'),('i6','SMOKE_5'),('d4','FULL_300'),('analysis','PILOT_20')]:
        with pytest.raises(ValueError,match='unregistered task/subset'):
            J.required_gates(task,subset)
    plan['ids']=load_video_ids()
    with pytest.raises(ValueError,match='full census cannot use a validation label'):J.verify_plan(plan,root)
    assert J.required_gates('analysis','FULL_300')==['D1','D2','D3a','D3b','D4','D5','D6','D7']


def test_terminal_status_is_rechecked_inside_coordinator_lock(tmp_path,monkeypatch):
    root,_,_=fixture_plan(tmp_path);original=J.exclusive
    @contextlib.contextmanager
    def race(path):
        with original(path):
            if path.name=='coordinator.lock':
                m=J.read(root/'manifest.json');m['execution']['status']='FAILED'
                J.put(root/'manifest.json',m,replace=True)
            yield
    monkeypatch.setattr(J,'exclusive',race)
    with pytest.raises(ValueError,match='terminal runs'):J.coordinate(root)
    assert J.read(root/'manifest.json')['execution']['status']=='FAILED'
    assert not (root/'workers').exists()


def test_same_run_id_cannot_restart_and_cannot_impersonate_launch(tmp_path):
    root,plan,_=fixture_plan(tmp_path)
    J.put(root/'coordinator.owner.json',{'prior_process':'already_claimed'})
    with pytest.raises(FileExistsError):J.coordinate(root)
    assert not (root/'workers').exists()
    second,other,_=fixture_plan(tmp_path);other['task']='features'
    J.put(second/'plan.json',other,replace=True)
    wrong=dict(J.process_identity(),start_ticks='other')
    J.put(second/'coordinator.launch.json',{'process':wrong})
    with pytest.raises(ValueError,match='registered launch process'):J.coordinate(second)
    assert not (second/'workers').exists()


def test_gpu_reservations_are_exclusive_and_assignments_typed(tmp_path,monkeypatch):
    root,plan,_=fixture_plan(tmp_path)
    (tmp_path/'runs').mkdir();monkeypatch.setattr(J,'PROJECT',tmp_path)
    reservation={'shards':[{'gpu_uuid':'GPU-one'}]}
    with J.gpu_allocation_locks(reservation):
        with pytest.raises(BlockingIOError):
            with J.gpu_allocation_locks(reservation):pass
    plan['shards'][0]['gpu_uuid']='GPU-one'
    with pytest.raises(ValueError,match='GPU assignments'):J.verify_plan(plan,root)


def test_measurement_consumer_rederives_actual_source(tmp_path,monkeypatch):
    from v4 import measurements as M
    original={'source_manifest_path':'registered.json','max_unit_seconds':21.5,
              'all_unit_timings':[{'elapsed_seconds':10}],
              'worker_first_unit_including_startup_seconds':[21.5],'recorded_at':'original'}
    path=tmp_path/'measurement.json';J.put(path,original)
    monkeypatch.setattr(M,'derive',lambda source:dict(original,recorded_at='current'))
    assert M.validate_measurement(path)==original
    J.put(path,dict(original,max_unit_seconds=1),replace=True)
    with pytest.raises(ValueError,match='rederived actual COMPLETE'):M.validate_measurement(path)


def test_dependency_role_rejects_wrong_layer_before_consumption(tmp_path):
    path=tmp_path/'manifest.json';J.put(path,{'science':{'layer':'features'}})
    J.put(tmp_path/'plan.json',{'task':'features'})
    plan={'task':'analysis','dependencies':[{'role':'masks','manifest_path':str(path),'manifest_sha256':A.file_hash(path)}]}
    with pytest.raises(ValueError,match='source task/layer'):J.dependency(plan,'masks')


def test_source_guard_binds_loader_alias_and_complete_snapshot(tmp_path):
    blob=tmp_path/'original';blob.write_bytes(b'original')
    other=tmp_path/'other';other.write_bytes(b'changed!')
    snapshot=tmp_path/'snapshot';snapshot.mkdir();alias=snapshot/'config.json';alias.symlink_to(blob)
    plan={'guarded_files':[{'physical_path':str(blob),'declared_path':str(alias),'bytes':8,'sha256':A.file_hash(blob)}],
          'source_directories':[{'declared_root':str(snapshot),'expected_relative_files':['config.json']}]}
    J.source_guard(plan)
    alias.unlink();alias.symlink_to(other)
    with pytest.raises(ValueError,match='alias changed target'):J.source_guard(plan)
    alias.unlink();alias.symlink_to(blob)
    (snapshot/'model.safetensors').write_bytes(b'unregistered alternative')
    with pytest.raises(ValueError,match='snapshot file inventory changed'):J.source_guard(plan)


def test_measured_sidecar_read_rejects_mutation_and_retains_consumed_hash(tmp_path):
    from v4 import measurements as M
    p=tmp_path/'timing.json';J.put(p,{'elapsed_seconds':2})
    descriptors={'timing.json':{'bytes':p.stat().st_size,'sha256':A.file_hash(p)}};consumed=[]
    assert M.verified_sidecar(tmp_path,'timing.json',descriptors,consumed)=={'elapsed_seconds':2}
    J.put(p,{'elapsed_seconds':1},replace=True)
    with pytest.raises(ValueError,match='sidecar bytes differ'):M.verified_sidecar(tmp_path,'timing.json',descriptors,[])
    with pytest.raises(ValueError,match='source/input changed'):J.source_guard({'guarded_files':consumed})
    with pytest.raises(ValueError,match='unregistered measurement'):M.verified_sidecar(tmp_path,'extra.json',descriptors,[])


def test_full_capacity_rejects_understated_measured_work(tmp_path,monkeypatch):
    from v4 import measurements as M
    import time
    from types import SimpleNamespace
    measured={'task':'features','max_artifact_bytes_per_work_unit':1000,
              'total_execution_sidecar_bytes':101,'timed_work_units':20}
    ids=[str(x) for x in range(300)]
    sheet=M.capacity_minimum(measured,'features',ids,8)
    assert sheet['payload_bytes']==300000 and sheet['temporary_bytes']==8000 and sheet['execution_sidecar_bytes']==1515
    assert sheet['minimum_planned_additional_bytes']==309515
    p=tmp_path/'capacity.json';cap={'checked_unix_seconds':time.time(),'available_project_bytes':1000000,
                                'planned_additional_bytes':309514,'measured_worksheet':sheet};J.put(p,cap)
    plan={'subset_kind':'FULL_300','capacity_evidence':str(p),'measurement_file':'frozen','task':'features','ids':ids,'shards':[{}]*8}
    monkeypatch.setattr(M,'validate_measurement',lambda path:measured)
    monkeypatch.setattr(J.os,'statvfs',lambda path:SimpleNamespace(f_bavail=1000000,f_frsize=1))
    with pytest.raises(ValueError,match='actual measured worksheet'):J.allocated_capacity_current(plan)
    cap['planned_additional_bytes']=309515;J.put(p,cap,replace=True);J.allocated_capacity_current(plan)
    cap['measured_worksheet']['payload_bytes']=1;J.put(p,cap,replace=True)
    with pytest.raises(ValueError,match='actual measured worksheet'):J.allocated_capacity_current(plan)
