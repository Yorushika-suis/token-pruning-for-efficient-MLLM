"""Create a fresh run plan, allocate verified idle GPUs and start its supervisors."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import artifacts as A, jobs as J, provenance as P, spec

AUDIT=J.PROJECT/'docs/implementation/v4_20260905'


def gpu_snapshot():
    names=('index','uuid','name','memory.total','memory.used','utilization.gpu')
    r=subprocess.run(['nvidia-smi','--query-gpu='+','.join(names),'--format=csv,noheader,nounits'],capture_output=True,text=True)
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if r.returncode or processes.returncode:
        raise RuntimeError(json.dumps({'gpu':r.stdout,'gpu_error':r.stderr,'processes':processes.stdout,'process_error':processes.stderr}))
    cards=[dict(zip(names,[v.strip() for v in row])) for row in csv.reader(io.StringIO(r.stdout)) if row]
    rows=[{'gpu_uuid':row[0].strip(),'pid':int(row[1]),'process_name':row[2].strip(),'memory_used':row[3].strip()}
          for row in csv.reader(io.StringIO(processes.stdout)) if row]
    return {'time':J.now(),'hostname':os.uname().nodename,'cards':cards,'compute_processes':rows}


def idle_gpus(first,second):
    if first['hostname']!=second['hostname'] or not first['cards'] or not second['cards']:
        raise ValueError('GPU host identity changed or no cards available')
    occupied={r['gpu_uuid'] for x in (first,second) for r in x['compute_processes']}
    ids=[]
    previous={x['uuid']:x for x in first['cards']}
    for card in second['cards']:
        prior=previous.get(card['uuid'])
        if prior and card['uuid'] not in occupied and all(int(c['memory.used'])==0 and int(c['utilization.gpu'])==0 for c in (prior,card)):
            ids.append(card['uuid'])
    if not ids:raise ValueError('no GPU is idle in both actual snapshots')
    return ids


def gate(task,subset):
    return J.current_gate_evidence(task,subset)


def prepare(task,subset,provenance_file,dependency_files=(),measurement=None,capacity_file=None):
    gate_evidence=gate(task,subset)
    P.R.configure()
    from data_manifest import load_video_ids
    full_ids=load_video_ids()
    pressure=J.read(AUDIT/'d2_i6_selection_fixture.json') if subset=='I6_PRESSURE' else None
    if pressure is not None:
        # The D2 fixture is independently hashed in the provenance/guard table.
        ids=pressure['ordered_i6_ids']
    else:
        ids=full_ids[:5] if subset=='SMOKE_5' else full_ids[:20] if subset=='PILOT_20' else full_ids
    if task=='fixture':raise ValueError('fixture launch uses the explicit D2 test harness')
    provenance=J.read(provenance_file)
    confirmation=J.read(Path(provenance_file).parent/'COMPLETE.json')
    if confirmation['provenance_file_sha256']!=A.file_hash(provenance_file) or provenance['content_sha256']!=A.digest(A.canonical(provenance['science'])):
        raise ValueError('D3 source closure is not COMPLETE or its bytes changed')
    if P.numeric_environment()!=provenance['science']['numeric_environment']:
        raise ValueError('current numeric environment differs from D3 fingerprint')
    snapshots=[]
    if task in ('analysis','analysis_timing'):gpus=[None]
    else:
        snapshots.append(gpu_snapshot());time.sleep(5);snapshots.append(gpu_snapshot())
        gpus=idle_gpus(*snapshots)
    dependencies=[]
    for role,path in dependency_files:
        path=Path(path);m=J.read(path)
        dependencies.append({'role':role,'manifest_path':str(path),'manifest_sha256':A.file_hash(path),
                             'content_sha256':m['content_sha256'],'run_id':m['execution']['run_id']})
    unit_max=None
    if measurement:
        from .measurements import validate_measurement
        measured=validate_measurement(measurement)
        if measured['task']!=task or measured['numeric_environment_sha256']!=provenance['science']['numeric_environment_sha256'] or measured['source_stage']!='D4' or measured.get('source_provenance_sha256')!=provenance['content_sha256']:
            raise ValueError('D4 measurement does not match this task/configuration')
        unit_max=measured['max_unit_seconds']
    if subset=='FULL_300':
        if not capacity_file:raise ValueError('fresh quota and full work-budget evidence required')
        cap=J.read(capacity_file)
        age=time.time()-cap['checked_unix_seconds']
        if not (0<=age<=300) or type(cap['planned_additional_bytes']) is not int or cap['planned_additional_bytes']<=0 or cap['available_project_bytes']<cap['planned_additional_bytes']*1.2:
            raise ValueError('quota is stale or lacks 20 percent working headroom')
        if unit_max is None:raise ValueError('no corresponding D4 unit timing for full run')
        from .measurements import capacity_minimum
        worksheet=capacity_minimum(measured,task,ids,min(len(gpus),len(J.work_units(task,ids))))
        if cap.get('measured_worksheet')!=worksheet or cap['planned_additional_bytes']<worksheet['minimum_planned_additional_bytes']:
            raise ValueError('full-run capacity is below the actual measured worksheet')
    execution=A.new_execution()
    root=J.PROJECT/'runs/v4'/execution['run_id']
    root.mkdir(parents=True,exist_ok=False)
    gate_file=root/'gate_evidence.json';J.put(gate_file,gate_evidence)
    expected=A.ordered_set(J.expected_keys(task,ids))
    planned_units=J.work_units(task,ids)
    shards=[]
    for index,gpu in enumerate(gpus[:len(planned_units)]):
        units=planned_units[index:len(planned_units):min(len(gpus),len(planned_units))]
        shards.append({'shard_id':index,'units':units,'gpu_uuid':gpu,
                       'coverage':J.shard_coverage(task,units),
                       'deadline_seconds':J.fixed_deadline(unit_max,len(units),full=subset=='FULL_300')})
    layer='validation' if task in ('d4','i6') else 'analysis' if task=='analysis_timing' else task
    science={'schema_version':A.SCHEMA_VERSION,'layer':layer,'plan_sha256':spec.PLAN_SHA256,
             'ordered_expected_ids':ids,'subset_kind':subset,'dataset_sha256':provenance['science']['dataset_sha256'],
             'provenance_sha256':provenance['content_sha256'],'numeric_environment_sha256':provenance['science']['numeric_environment_sha256'],
             'definitions':{'version':spec.VERSION,'task':task,'fixture_only':False,'calibration_only':task=='analysis_timing','methods':list(spec.METHODS),
                            'method_versions':spec.METHOD_VERSIONS,'budgets':spec.BUDGETS,'protocols':spec.PROTOCOLS,
                            'purpose_arm_pairs':spec.PURPOSE_ARMS,'frames':64,'grid':13,
                            'rng_domain':spec.DOMAIN,'control_seeds':list(spec.SEEDS),'deterministic_seed':'deterministic'},
             'expected_coverage':expected,'artifacts':[],
             'upstreams':[{'role':x['role'],'hash_spec':A.HASH_SPEC,'content_sha256':x['content_sha256']} for x in dependencies]}
    guarded=[{k:x[k] for k in ('physical_path','declared_path','bytes','sha256')} for x in provenance['execution']['files']]
    for path in [provenance_file,gate_file]+([measurement] if measurement else [])+([capacity_file] if capacity_file else [])+([AUDIT/'d2_i6_selection_fixture.json'] if pressure else []):
        guarded.append({'physical_path':str(path),'bytes':Path(path).stat().st_size,'sha256':A.file_hash(path)})
    plan={'run_id':execution['run_id'],'plan_sha256':spec.PLAN_SHA256,'task':task,'subset_kind':subset,'ids':ids,
          'i6_selected_ids':ids if pressure else None,'expected_coverage':expected,'shards':shards,
          'science_base':science,'guarded_files':guarded,'dependencies':dependencies,'d4_unit_max_seconds':unit_max,
          'sampling':provenance['science']['sampling'],'gpu_snapshots':snapshots,
          'provenance_file':str(provenance_file),'capacity_evidence':str(capacity_file) if capacity_file else None}
    plan.update(gate_evidence_file=str(gate_file),measurement_file=str(measurement) if measurement else None)
    plan['source_directories']=provenance['execution']['source_directories']
    execution['runtime']={'plan_file':'plan.json','gpu_snapshots':snapshots,'shard_deadlines':{str(s['shard_id']):s['deadline_seconds'] for s in shards}}
    execution['upstreams']=[{k:x[k] for k in ('role','run_id','manifest_path','content_sha256')} for x in dependencies]
    manifest={'science':A.normalize_science(science),'execution':execution,'content_hash_spec':A.HASH_SPEC,'content_sha256':None}
    J.verify_plan(plan,root);J.source_guard(plan)
    J.put(root/'plan.json',plan);J.put(root/'manifest.json',manifest)
    for mode in ('coordinate','monitor'):
        env=dict(os.environ,PYTHONPATH=str(J.PROJECT/'scripts'),PYTHONDONTWRITEBYTECODE='1')
        with (root/f'{mode}.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B','-m','v4.jobs',mode,str(root)],stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
        identity=J.process_identity(child.pid)
        if identity is None:raise RuntimeError(mode+' exited before registration')
        J.put(root/('coordinator.launch.json' if mode=='coordinate' else 'monitor.launch.json'),
              {'time':J.now(),'process':identity,'launched_monotonic':time.monotonic()})
    return root
