"""Derive D4 timing evidence from actual COMPLETE runs, never guessed deadlines."""
from pathlib import Path
import datetime as dt
import math
import json

from . import artifacts as A, jobs as J


def verified_sidecar(root,relative,descriptors,consumed):
    """Parse precisely the sidecar bytes attested by the COMPLETE manifest."""
    expected=descriptors.get(relative)
    if expected is None:raise ValueError('unregistered measurement sidecar: '+relative)
    path=A.safe_path(root,relative);data=path.read_bytes()
    if len(data)!=expected['bytes'] or A.digest(data)!=expected['sha256']:
        raise ValueError('measurement sidecar bytes differ from COMPLETE manifest: '+relative)
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('measurement sidecar is not an object')
    consumed.append({'physical_path':str(path),'bytes':len(data),'sha256':A.digest(data)})
    return value


def capacity_minimum(measured,task,ids,worker_count):
    """Conservative measured lower bound; fresh project quota is checked separately."""
    if measured['task']!=task:raise ValueError('capacity measurement task differs')
    units=J.work_units(task,ids)
    maximum=measured['max_artifact_bytes_per_work_unit']
    sidecars=measured['total_execution_sidecar_bytes'];observed=measured['timed_work_units']
    if any(type(x) is not int or x<=0 for x in (maximum,observed,worker_count)) or type(sidecars) is not int or sidecars<0:
        raise ValueError('invalid measured capacity inputs')
    if worker_count>len(units):raise ValueError('more workers than planned work units')
    payload=maximum*len(units)
    temporary=maximum*worker_count
    execution=(sidecars*len(units)+observed-1)//observed
    return {'task':task,'ordered_work_units':units,'worker_count':worker_count,
            'measured_maximum_payload_bytes_per_unit':maximum,'payload_bytes':payload,
            'temporary_bytes':temporary,'execution_sidecar_bytes':execution,
            'minimum_planned_additional_bytes':payload+temporary+execution,
            'basis':'maximum measured payload per unit times planned units; one extra unit per worker; measured sidecars scaled by unit count; project quota must additionally retain 20 percent headroom'}


def derive(manifest_file):
    path=Path(manifest_file);root=path.parent;plan=J.read(root/'plan.json')
    manifest_bytes=path.read_bytes();manifest=json.loads(manifest_bytes)
    expected=J.verify_plan(plan,root)
    A.validate_manifest(manifest,root,expected,upstream_resolver=lambda ref:J.dependency(plan,ref['role'])[0])
    if plan['subset_kind'] not in ('SMOKE_5','PILOT_20') or plan['task'] not in ('d4','features','masks','analysis_timing'):
        raise ValueError('timing source is not an accepted D4 configuration')
    task='analysis' if plan['task']=='analysis_timing' else plan['task']
    if task=='analysis' and plan['subset_kind']!='PILOT_20':raise ValueError('analysis timing requires declared pilot calibration')
    rows=[];first_unit_costs=[];consumed=[]
    descriptors={x['relative_path']:x for x in manifest['execution']['sidecars']}
    for shard in plan['shards']:
        worker=root/'workers'/str(shard['shard_id'])
        launch=verified_sidecar(root,str((worker/'launch.json').relative_to(root)),descriptors,consumed)
        items=[verified_sidecar(root,str(p.relative_to(root)),descriptors,consumed)
               for p in sorted((worker/'unit_timings').glob('*.json'))]
        if A.ordered_set([x['unit'] for x in items])!=A.ordered_set(shard['units']):
            raise ValueError('D4 timing omitted, duplicated or added a unit')
        for item in items:
            if item['task']!=plan['task'] or not isinstance(item['elapsed_seconds'],(int,float)) or not math.isfinite(item['elapsed_seconds']) or item['elapsed_seconds']<=0:
                raise ValueError('D4 timing value is not valid for this task')
        first=min(dt.datetime.fromisoformat(x['time']) for x in items)
        first_cost=(first-dt.datetime.fromisoformat(launch['time'])).total_seconds()
        if first_cost<=0:raise ValueError('invalid worker-to-first-unit elapsed time')
        first_unit_costs.append(first_cost);rows.extend(items)
    # Include observed startup/dependency/model loading in the first unit cost.
    maximum=max([x['elapsed_seconds'] for x in rows]+first_unit_costs)
    artifact_sizes={}
    for item in manifest['science']['artifacts']:
        identity=item['key'][0] if task!='analysis' else A.canonical(item['key']).decode()
        artifact_sizes[identity]=artifact_sizes.get(identity,0)+item['bytes']
    result={'source_stage':'D4','task':task,'measured_task':plan['task'],'source_subset':plan['subset_kind'],
            'source_run_id':plan['run_id'],'source_manifest_file_sha256':A.digest(manifest_bytes),
            'source_content_sha256':manifest['content_sha256'],'source_manifest_path':str(path),
            'source_provenance_sha256':manifest['science']['provenance_sha256'],
            'numeric_environment_sha256':manifest['science']['numeric_environment_sha256'],
            'max_unit_seconds':maximum,'all_unit_timings':rows,'worker_first_unit_including_startup_seconds':first_unit_costs,
            'first_unit_includes_observed_startup_and_dependency_validation':True,
            'timing_only_synthetic_rows':task=='analysis','not_an_inferential_estimate':task=='analysis',
            'artifact_bytes_by_work_unit':artifact_sizes,'total_artifact_bytes':sum(artifact_sizes.values()),
            'max_artifact_bytes_per_work_unit':max(artifact_sizes.values()),
            'total_execution_sidecar_bytes':sum(x['bytes'] for x in descriptors.values()),
            'timed_work_units':len(rows),'recorded_at':J.now()}
    J.source_guard(plan)
    J.source_guard({'guarded_files':consumed})
    if A.file_hash(path)!=result['source_manifest_file_sha256']:raise ValueError('timing source changed during collection')
    return result


def collect(manifest_file,output):
    result=derive(manifest_file)
    J.put(output,result)
    return result


def validate_measurement(path):
    declared=J.read(path)
    observed=derive(declared['source_manifest_path'])
    # Timestamp describes when the same measurement was serialized.
    if {k:v for k,v in declared.items() if k!='recorded_at'}!={k:v for k,v in observed.items() if k!='recorded_at'}:
        raise ValueError('D4 measurement differs from rederived actual COMPLETE source evidence')
    return declared
