"""Separate workers, coordinator and monitor for immutable V4 run directories.

Only the coordinator publishes the run manifest. The monitor can mark a lost
coordinator FAILED after its process identity is gone and the lock is acquired;
it never turns a failed run into a success or kills an unowned process.
"""
from __future__ import annotations

import contextlib
import csv
import datetime as dt
import fcntl
import json
import io
import os
import re
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

from . import artifacts as A, spec

PROJECT = Path(__file__).resolve().parents[2]
BEAT_SECONDS, LOST_SECONDS = 30, 180

TASK_SUBSETS={'d4':{'SMOKE_5','PILOT_20'},'i6':{'I6_PRESSURE'},
              'features':{'SMOKE_5','PILOT_20','FULL_300'},'masks':{'SMOKE_5','PILOT_20','FULL_300'},
              'analysis':{'FULL_300'},'analysis_timing':{'PILOT_20'},'fixture':{'GOLDEN'}}


def required_gates(task,subset):
    if subset not in TASK_SUBSETS.get(task,set()):
        raise ValueError('unregistered task/subset combination')
    if task=='fixture':return []
    required=['D1','D2','D3a','D3b']
    if task=='i6':required+=['D4']
    if subset=='FULL_300':
        required+=['D4','D5']
        if task in ('masks','analysis'):required+=['D6']
        if task=='analysis':required+=['D7']
    return required


def current_gate_evidence(task,subset):
    required=required_gates(task,subset)
    if not required:return None
    path=PROJECT/'docs/implementation/v4_20260905/gate_status.json'
    raw=path.read_bytes();g=json.loads(raw)
    if any(g.get(k)!='COMPLETE' for k in required):
        raise ValueError('required implementation gates are not COMPLETE: '+repr(required))
    return {'time':now(),'source_sha256':A.digest(raw),'required':required,
            'statuses':{k:g[k] for k in required}}


@contextlib.contextmanager
def gpu_allocation_locks(plan):
    assigned=sorted({s['gpu_uuid'] for s in plan['shards'] if s['gpu_uuid'] is not None})
    with contextlib.ExitStack() as stack:
        if assigned:
            directory=PROJECT/'runs/v4_gpu_locks';directory.mkdir(exist_ok=True)
            for uuid in assigned:stack.enter_context(exclusive(directory/(uuid+'.lock')))
        yield


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def put(path, value, *, replace=False):
    from .statistics import jsonable
    A.atomic_bytes(path, A.canonical(jsonable(value)), replace=replace)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        # comm is parenthesized and may itself contain spaces or parentheses.
        fields = text[text.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'hostname': os.uname().nodename}
    except FileNotFoundError:
        return None


def same_process(identity):
    if not identity or identity['hostname'] != os.uname().nodename:
        raise ValueError('cannot verify a process on another host')
    return process_identity(identity['pid']) == identity


@contextlib.contextmanager
def exclusive(path):
    with Path(path).open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Heartbeat:
    def __init__(self, path):
        self.path = Path(path)
        self.stop = threading.Event()
        self.identity = process_identity()
        self.failure = None

    def beat(self):
        put(self.path, {'time': now(), 'unix_seconds': time.time(),
                        'monotonic_seconds': time.monotonic(), 'process': self.identity}, replace=True)

    def loop(self):
        while not self.stop.wait(BEAT_SECONDS):
            try:
                self.beat()
            except BaseException as exc:
                self.failure = exc
                return

    def __enter__(self):
        self.beat()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def assert_healthy(self):
        if self.failure:
            raise RuntimeError('heartbeat publication failed') from self.failure

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)


def check_heartbeat(path, identity, launched_monotonic):
    path = Path(path)
    if not path.exists():
        age = time.monotonic() - launched_monotonic
    else:
        h = read(path)
        if h['process'] != identity:
            raise ValueError('heartbeat belongs to another process')
        age = time.monotonic() - h['monotonic_seconds']
        if age < -1:
            raise ValueError('heartbeat timestamp is in the future')
    if age >= LOST_SECONDS:
        raise TimeoutError(f'heartbeat missing for {age:.3f} seconds')


def fixed_deadline(unit_max_seconds, unit_count, *, full):
    if type(unit_count) is not int or unit_count <= 0:
        raise ValueError('positive planned work count required')
    if unit_max_seconds is None:
        if full:
            raise ValueError('full run requires measured D4 maximum for this configuration')
        # D4 exists to measure this quantity. A fixed 24 h initial validation
        # ceiling is operational only; it cannot supply a full-run estimate.
        return 86400
    if not isinstance(unit_max_seconds, (float, int)) or not (0 < unit_max_seconds < float('inf')):
        raise ValueError('invalid D4 unit duration')
    return max(1800, 3 * unit_max_seconds * unit_count + 600)


def source_guard(plan):
    """Rehash the registered executable/input files, including manifest bytes."""
    for item in plan['guarded_files']:
        path = Path(item['physical_path'])
        if item.get('declared_path') and Path(item['declared_path']).resolve(strict=True)!=path.resolve(strict=True):
            raise ValueError('registered source alias changed target: '+item['declared_path'])
        if not path.is_file() or path.stat().st_size != item['bytes'] or A.file_hash(path) != item['sha256']:
            raise ValueError('registered source/input changed: ' + str(path))
    for item in plan.get('source_directories',[]):
        directory=Path(item['declared_root'])
        paths=list(directory.rglob('*'))
        if directory.is_symlink() or any(p.is_symlink() and p.is_dir() for p in paths):
            raise ValueError('unregistered symlink directory in model snapshot')
        actual=sorted(str(p.relative_to(directory)) for p in paths if p.is_file())
        if actual!=item['expected_relative_files'] or any(not p.is_file() and not p.is_dir() for p in paths):
            raise ValueError('registered model snapshot file inventory changed')


def allocated_gpus_idle(plan):
    assigned={s['gpu_uuid'] for s in plan['shards'] if s['gpu_uuid'] is not None}
    if not assigned:return None
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                     capture_output=True,text=True)
    p=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                     capture_output=True,text=True)
    if r.returncode or p.returncode:
        raise RuntimeError(json.dumps({'gpu_stdout':r.stdout,'gpu_stderr':r.stderr,'process_stdout':p.stdout,'process_stderr':p.stderr}))
    cards={row[0].strip():[int(v.strip()) for v in row[1:]] for row in csv.reader(io.StringIO(r.stdout)) if row}
    occupied={row[0].strip() for row in csv.reader(io.StringIO(p.stdout)) if row}
    if assigned.intersection(occupied) or any(cards.get(uuid)!=[0,0] for uuid in assigned):
        raise ValueError('allocated GPU became occupied before worker launch')
    return {'time':now(),'assigned_gpu_uuids':sorted(assigned),'gpu_stdout':r.stdout,'process_stdout':p.stdout}


def allocated_capacity_current(plan):
    if plan['subset_kind']!='FULL_300':return
    from .measurements import validate_measurement, capacity_minimum
    measured=validate_measurement(plan['measurement_file'])
    worksheet=capacity_minimum(measured,plan['task'],plan['ids'],len(plan['shards']))
    cap=read(plan['capacity_evidence']);age=time.time()-cap['checked_unix_seconds']
    if cap.get('measured_worksheet')!=worksheet or type(cap['planned_additional_bytes']) is not int or cap['planned_additional_bytes']<worksheet['minimum_planned_additional_bytes']:
        raise ValueError('full-run capacity is below the actual measured worksheet')
    if not 0<=age<=300 or cap['planned_additional_bytes']<=0 or cap['available_project_bytes']<1.2*cap['planned_additional_bytes']:
        raise ValueError('full-run quota evidence expired or no longer provides registered working headroom')
    mount=os.statvfs(PROJECT)
    if mount.f_bavail*mount.f_frsize<1.2*cap['planned_additional_bytes']:
        raise ValueError('actual mount free bytes are below the declared work headroom')


def expected_keys(task, ids):
    if task == 'features':
        return spec.feature_keys(ids)
    if task == 'masks':
        return spec.mask_keys(ids)
    if task in ('d4', 'i6'):
        return [[v, task] for v in ids]
    if task in ('analysis','analysis_timing'):
        return [[m, b, task] for m in spec.METHODS for b in spec.BUDGETS]
    if task == 'fixture':
        return [[v, 'fixture'] for v in ids]
    raise ValueError('unregistered task')


def work_units(task, ids):
    return [[m,b] for m in spec.METHODS for b in spec.BUDGETS] if task in ('analysis','analysis_timing') else list(ids)


def shard_coverage(task, units):
    if task in ('analysis','analysis_timing'):
        allowed={tuple(x) for x in work_units(task,[])}
        if any(not isinstance(x,list) or len(x)!=2 or tuple(x) not in allowed for x in units):
            raise ValueError('invalid analysis work unit')
        return A.ordered_set([[m,b,task] for m,b in units])
    return A.ordered_set(expected_keys(task,units))


def verify_dependency_roles(plan):
    task=plan['task']
    if task not in TASK_SUBSETS:
        raise ValueError('unregistered dependency consumer task')
    expected=['features'] if task=='masks' else ['masks'] if task in ('analysis','analysis_timing') else []
    roles=[r['role'] for r in plan['dependencies']]
    if sorted(roles)!=expected:
        raise ValueError('consumer task requires its exact dependency role set')


def verify_dependency_join(consumer,producer,manifest):
    science=manifest['science'];base=producer['science_base']
    observed={'actual_runtime_bindings','actual_runtime_libraries'}
    if observed.intersection(base['definitions']) or observed.intersection(consumer['science_base']['definitions']):
        raise ValueError('actual runtime evidence cannot be predeclared in a plan')
    identity=('plan_sha256','dataset_sha256','provenance_sha256','numeric_environment_sha256')
    for field in identity+('ordered_expected_ids','subset_kind','expected_coverage','layer'):
        if science[field]!=base[field]:
            raise ValueError('dependency manifest differs from its producer plan: '+field)
    registered={k:v for k,v in science['definitions'].items() if k not in observed}
    if registered!=base['definitions']:
        raise ValueError('dependency manifest differs from its producer plan: definitions')
    if science['ordered_expected_ids']!=producer['ids'] or science['subset_kind']!=producer['subset_kind']:
        raise ValueError('dependency producer census differs from its manifest')
    if science['ordered_expected_ids']!=consumer['ids'] or science['subset_kind']!=consumer['subset_kind']:
        raise ValueError('dependency census differs from consumer')
    target=consumer['science_base']
    if target['ordered_expected_ids']!=consumer['ids'] or target['subset_kind']!=consumer['subset_kind']:
        raise ValueError('consumer scientific census differs from its plan')
    for field in identity:
        if science[field]!=target[field]:
            raise ValueError('dependency scientific identity differs from consumer: '+field)
    exclude={'task','fixture_only','calibration_only'}
    scientific=lambda d:{k:v for k,v in d.items() if k not in exclude}
    if scientific(registered)!=scientific(target['definitions']):
        raise ValueError('dependency scientific definitions differ from consumer')


def verify_dependency_runtime(producer,manifest,root):
    """Bind derived runtime fields to actual worker sidecars and D3 source bytes."""
    provenance_path=Path(producer['provenance_file'])
    guards=[g for g in producer['guarded_files'] if g['physical_path']==str(provenance_path)]
    if len(guards)!=1:raise ValueError('dependency provenance must have one byte guard')
    consumed=[]
    def verified(path,record):
        raw=path.read_bytes()
        if len(raw)!=record['bytes'] or A.digest(raw)!=record['sha256']:
            raise ValueError('dependency runtime evidence bytes changed: '+str(path))
        consumed.append({'physical_path':str(path),'bytes':len(raw),'sha256':A.digest(raw)})
        return json.loads(raw)
    provenance=verified(provenance_path,guards[0])
    content=A.digest(A.canonical(provenance['science']))
    if content!=provenance['content_sha256'] or content!=producer['science_base']['provenance_sha256']:
        raise ValueError('dependency runtime provenance content differs')
    records=provenance['science']['files']
    known={r['logical_path']:r['sha256'] for r in records}
    if len(known)!=len(records):raise ValueError('duplicate runtime provenance source identity')
    descriptors=manifest['execution']['sidecars']
    bindings={};libraries={}
    for shard in producer['shards']:
        relative='workers/'+str(shard['shard_id'])+'/runtime_closure_final.json'
        found=[r for r in descriptors if r['relative_path']==relative]
        if len(found)!=1:raise ValueError('dependency worker runtime sidecar missing or duplicated')
        closure=verified(A.safe_path(root,relative),found[0])
        if closure['all_registered'] is not True:raise ValueError('dependency worker runtime closure is incomplete')
        for field,dest,fields in [('scientific_bindings',bindings,{'module','attribute','logical_path','sha256'}),
                                  ('scientific_libraries',libraries,{'logical_path','sha256'})]:
            items=closure[field]
            if not isinstance(items,list):raise ValueError('runtime source evidence must be a list')
            seen=set()
            for item in items:
                if not isinstance(item,dict) or set(item)!=fields or any(not isinstance(v,str) or not v for v in item.values()):
                    raise ValueError('invalid runtime source record')
                if known.get(item['logical_path'])!=item['sha256']:
                    raise ValueError('dependency runtime source absent from D3 bytes')
                if field=='scientific_bindings':
                    if item['attribute'] not in ('__file__','__cached__'):raise ValueError('invalid runtime module attribute')
                    key=(item['module'],item['attribute'])
                else:key=item['logical_path']
                if key in seen:raise ValueError('duplicate worker runtime source record')
                seen.add(key)
                if key in dest and dest[key]!=item:raise ValueError('dependency workers disagree on runtime source')
                dest[key]=item
    derived={'actual_runtime_bindings':A.ordered_set(list(bindings.values()),key=lambda x:[x['module'],x['attribute']]),
             'actual_runtime_libraries':A.ordered_set(list(libraries.values()),key=lambda x:x['logical_path'])}
    for field,expected in derived.items():
        if manifest['science']['definitions'].get(field)!=expected:
            raise ValueError('dependency manifest runtime fields differ from worker evidence: '+field)
    source_guard({'guarded_files':consumed})


def verify_plan(plan, root):
    if plan['plan_sha256'] != spec.PLAN_SHA256 or plan['run_id'] != Path(root).name:
        raise ValueError('run identity or approved plan mismatch')
    ids = plan['ids']
    from data_manifest import load_video_ids
    full_ids = load_video_ids()
    subset = plan['subset_kind']
    required=required_gates(plan['task'],subset)
    verify_dependency_roles(plan)
    fixture=plan['task']=='fixture'
    if fixture and plan['science_base']['definitions'].get('fixture_only') is not True:
        raise ValueError('fixture task requires an explicit fixture-only definition')
    if not fixture:
        evidence=read(plan['gate_evidence_file'])
        if evidence['required']!=required or evidence['statuses']!={k:'COMPLETE' for k in required}:
            raise ValueError('required gate evidence differs from registered task')
        guarded=[g for g in plan['guarded_files'] if g['physical_path']==plan['gate_evidence_file']]
        if len(guarded)!=1 or A.file_hash(plan['gate_evidence_file'])!=guarded[0]['sha256']:
            raise ValueError('gate evidence is not byte-bound to this run')
    if plan['task']=='analysis_timing' and (subset!='PILOT_20' or not plan['science_base']['definitions'].get('calibration_only')):
        raise ValueError('analysis timing must be explicitly marked PILOT_20 calibration')
    if subset == 'FULL_300' and ids != full_ids:
        raise ValueError('full census differs from authority')
    if subset in ('SMOKE_5', 'PILOT_20') and ids != full_ids[:5 if subset == 'SMOKE_5' else 20]:
        raise ValueError('validation subset differs from predeclared order')
    if subset == 'I6_PRESSURE':
        frozen=read(PROJECT/'docs/implementation/v4_20260905/d2_i6_selection_fixture.json')
        if ids != frozen['ordered_i6_ids'] or ids != plan['i6_selected_ids']:
            raise ValueError('I6 sample differs from frozen D2 selection')
    if len(ids) != len(set(ids)) or not ids or any(v not in full_ids for v in ids):
        raise ValueError('unsafe or invalid IDs')
    if ids==full_ids and subset!='FULL_300':
        raise ValueError('full census cannot use a validation label')
    gpus=[s['gpu_uuid'] for s in plan['shards']]
    cpu=plan['task'] in ('analysis','analysis_timing','fixture')
    if (cpu and any(g is not None for g in gpus)) or (not cpu and (any(not isinstance(g,str) or not re.fullmatch(r'GPU-[A-Za-z0-9_-]+',g) for g in gpus) or len(gpus)!=len(set(gpus)))):
        raise ValueError('GPU assignments must be valid and distinct; CPU tasks use None')
    if plan.get('measurement_file'):
        from .measurements import validate_measurement
        measured=validate_measurement(plan['measurement_file'])
        if measured['task']!=plan['task'] or measured['max_unit_seconds']!=plan['d4_unit_max_seconds'] or measured['source_provenance_sha256']!=plan['science_base']['provenance_sha256'] or measured['numeric_environment_sha256']!=plan['science_base']['numeric_environment_sha256']:
            raise ValueError('registered D4 measurement differs from task/provenance/deadline')
    elif plan['d4_unit_max_seconds'] is not None or subset=='FULL_300':
        raise ValueError('measured deadline requires its actual D4 evidence file')
    expected = A.ordered_set(expected_keys(plan['task'], ids))
    if plan['expected_coverage'] != expected:
        raise ValueError('caller-supplied expected coverage differs from contract')
    if A.ordered_set([k for shard in plan['shards'] for k in shard['coverage']]) != expected:
        raise ValueError('shards overlap or do not exactly cover the work')
    if A.ordered_set([u for shard in plan['shards'] for u in shard['units']])!=A.ordered_set(work_units(plan['task'],ids)):
        raise ValueError('actual work units do not cover the registered task')
    for shard in plan['shards']:
        if shard['deadline_seconds'] != fixed_deadline(plan['d4_unit_max_seconds'], len(shard['units']), full=subset=='FULL_300'):
            raise ValueError('deadline differs from registered measurement')
        if shard['coverage'] != shard_coverage(plan['task'],shard['units']):
            raise ValueError('shard work and coverage do not agree')
    if len({s['shard_id'] for s in plan['shards']}) != len(plan['shards']):
        raise ValueError('duplicate shard identity')
    if plan['science_base']['definitions'].get('fixture_only') and plan['task'] != 'fixture':
        raise ValueError('fixture cannot enter scientific collection')
    return expected


def dependency(plan, role):
    verify_dependency_roles(plan)
    references = [x for x in plan['dependencies'] if x['role']==role]
    if len(references) != 1:
        raise ValueError('required dependency role missing or duplicated: ' + role)
    ref = references[0]
    path = Path(ref['manifest_path'])
    if any('_DEPRECATED_' in p for p in path.parts) or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('deprecated or symlink dependency')
    raw = path.read_bytes()
    if A.digest(raw) != ref['manifest_sha256']:
        raise ValueError('dependency manifest bytes changed')
    manifest = json.loads(raw)
    source_plan = read(path.parent/'plan.json')
    if role not in ('features','masks') or source_plan['task']!=role or manifest['science']['layer']!=role:
        raise ValueError('dependency role differs from actual source task/layer')
    verify_dependency_join(plan,source_plan,manifest)
    expected = verify_plan(source_plan, path.parent)
    if manifest['science']['definitions'].get('fixture_only') or manifest['science']['definitions'].get('calibration_only'):
        raise ValueError('synthetic fixture/calibration is not a scientific dependency')
    verify_dependency_runtime(source_plan,manifest,path.parent)
    A.validate_manifest(manifest, path.parent, expected,
                        upstream_resolver=lambda u: dependency(source_plan, u['role'])[0])
    if manifest['content_sha256'] != ref['content_sha256'] or manifest['execution']['run_id'] != ref['run_id']:
        raise ValueError('dependency identity mismatch')
    if A.file_hash(path) != ref['manifest_sha256']:
        raise ValueError('dependency manifest changed during reading')
    return manifest, path.parent


def submit(root, shard_id, name, arrays, metadata, key):
    """Each worker owns its staging path and disjoint published subdirectory."""
    root = Path(root)
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('unsafe worker artifact name')
    staging = root/'workers'/str(shard_id)/'staging'/f'{name}.npz'
    A.write_npz(staging, arrays, metadata)
    relative = f'artifacts/shard_{shard_id}/{name}.npz'
    dest = A.safe_path(root, relative, exists=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(staging, dest)
    staging.unlink()
    descriptor, _ = A.describe(root, relative, key)
    receipt = {'artifact': descriptor, 'path': {'key': key, 'relative_path': relative}}
    put(root/'workers'/str(shard_id)/'receipts'/f'{name}.json', receipt)
    return receipt


def mark_failed(root, manifest, exc):
    manifest['execution'].update(status='FAILED', finished_at=now(),
            reason={'type': type(exc).__name__, 'message': str(exc)})
    manifest['content_sha256'] = None
    put(Path(root)/'manifest.json', manifest, replace=True)
    put(Path(root)/('failure_'+str(time.time_ns())+'.json'),
        {'time':now(), 'error_type':type(exc).__name__, 'error':str(exc), 'traceback':traceback.format_exc()})


def coordinate(root):
    root = Path(root)
    plan = read(root/'plan.json')
    manifest = read(root/'manifest.json')
    if manifest['execution']['status'] != 'RUNNING':
        raise ValueError('terminal runs cannot be resumed or overwritten')
    children = []
    with exclusive(root/'coordinator.lock'), contextlib.ExitStack() as lifetime:
        # Read inside the lock so monitor FAILED cannot be overwritten by a
        # stale RUNNING copy. The one-use owner file also forbids same-ID retry.
        manifest=read(root/'manifest.json')
        if manifest['execution']['status']!='RUNNING':
            raise ValueError('terminal runs cannot be resumed or overwritten')
        identity=process_identity()
        if plan['task']!='fixture':
            until=time.monotonic()+5
            while not (root/'coordinator.launch.json').exists() and time.monotonic()<until:time.sleep(.05)
            if read(root/'coordinator.launch.json')['process']!=identity:
                raise ValueError('coordinator is not the registered launch process')
        put(root/'coordinator.owner.json',{'process':identity,'time':now()})
        beat=lifetime.enter_context(Heartbeat(root/'coordinator.heartbeat.json'))
        try:
            expected = verify_plan(plan, root)
            source_guard(plan)
            for ref in plan['dependencies']:
                dependency(plan, ref['role'])
            lifetime.enter_context(gpu_allocation_locks(plan))
            gates=current_gate_evidence(plan['task'],plan['subset_kind'])
            if gates is not None:put(root/'worker_launch_gate_recheck.json',gates)
            fresh=allocated_gpus_idle(plan)
            allocated_capacity_current(plan)
            if fresh is not None:put(root/'worker_launch_gpu_recheck.json',fresh)
            for shard in plan['shards']:
                worker_root = root/'workers'/str(shard['shard_id'])
                worker_root.mkdir(parents=True, exist_ok=False)
                env = dict(os.environ, PYTHONPATH=str(PROJECT/'scripts'), PYTHONDONTWRITEBYTECODE='1',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
                if shard['gpu_uuid'] is not None:
                    env['CUDA_VISIBLE_DEVICES'] = shard['gpu_uuid']
                with (worker_root/'stdout.log').open('xb') as log:
                    child = subprocess.Popen([sys.executable,'-B','-m','v4.jobs','worker',str(root),str(shard['shard_id'])],
                                             env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(child.pid)
                if identity is None:
                    raise RuntimeError('worker exited before identity registration')
                launch = {'process':identity, 'launched_monotonic':time.monotonic(),
                          'time':now(), 'deadline_seconds':shard['deadline_seconds']}
                put(worker_root/'launch.json', launch)
                children.append((child, shard, worker_root, launch))
            while True:
                beat.assert_healthy()
                all_exited = True
                for child, shard, worker_root, launch in children:
                    code = child.poll()
                    if code is None:
                        all_exited = False
                        check_heartbeat(worker_root/'heartbeat.json', launch['process'], launch['launched_monotonic'])
                        if time.monotonic()-launch['launched_monotonic'] >= shard['deadline_seconds']:
                            raise TimeoutError('worker exceeded registered deadline: '+str(shard['shard_id']))
                    elif code != 0:
                        raise RuntimeError(f'worker {shard["shard_id"]} exited {code}')
                if all_exited:
                    break
                time.sleep(1)
            # Coordinator independently reopens actual payload bytes, after exit.
            artifacts, paths, workers = [], [], []
            for child, shard, worker_root, launch in children:
                result = read(worker_root/'result.json')
                if result['status'] != 'COMPLETE' or result['coverage'] != shard['coverage']:
                    raise ValueError('worker did not complete its assigned work')
                rows = [read(p) for p in sorted((worker_root/'receipts').glob('*.json'))]
                for row in rows:
                    observed,_ = A.describe(root,row['path']['relative_path'],row['path']['key'])
                    if observed != row['artifact']:
                        raise ValueError('worker receipt disagrees with actual bytes')
                    artifacts.append(observed); paths.append(row['path'])
                workers.append({'shard_id':shard['shard_id'],'status':'COMPLETE','exit_code':child.returncode,
                                'coverage':shard['coverage'], 'process':launch['process']})
            science = dict(plan['science_base'], expected_coverage=expected, artifacts=artifacts)
            if plan['task']!='fixture':
                bindings={};libraries={}
                for _,_,worker_root,_ in children:
                    closure=read(worker_root/'runtime_closure_final.json')
                    if closure['all_registered'] is not True:raise ValueError('worker runtime closure is incomplete')
                    for item in closure['scientific_bindings']:
                        key=(item['module'],item['attribute'])
                        if key in bindings and bindings[key]!=item:
                            raise ValueError('workers loaded inconsistent runtime module origins')
                        bindings[key]=item
                    for item in closure['scientific_libraries']:
                        key=item['logical_path']
                        if key in libraries and libraries[key]!=item:raise ValueError('workers loaded inconsistent libraries')
                        libraries[key]=item
                science['definitions']=dict(science['definitions'],
                    actual_runtime_bindings=A.ordered_set(list(bindings.values()),key=lambda x:[x['module'],x['attribute']]),
                    actual_runtime_libraries=A.ordered_set(list(libraries.values()),key=lambda x:x['logical_path']))
            manifest['science'] = A.normalize_science(science)
            e = manifest['execution']
            e.update(status='COMPLETE',finished_at=now(),workers=workers,artifact_paths=paths,reason=None)
            sidecars=[]
            for path in sorted((root/'workers').rglob('*')):
                if path.is_symlink():
                    raise ValueError('symlink execution sidecar')
                if path.is_file() and '/staging/' in str(path):
                    raise ValueError('worker left unpublished staging payload')
                if path.is_file():
                    sidecars.append({'relative_path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':A.file_hash(path)})
            e['sidecars'] = sidecars
            manifest['content_sha256'] = A.digest(A.canonical(manifest['science']))
            source_guard(plan)
            A.validate_manifest(manifest,root,expected,upstream_resolver=lambda u:dependency(plan,u['role'])[0])
            beat.assert_healthy()
            put(root/'manifest.json',manifest,replace=True)
        except BaseException as exc:
            # Only these freshly launched children belong to this run. Retain
            # their partial files and request a cooperative exit at unit boundary.
            put(root/'cancel.json', {'time':now(),'reason':str(exc)}, replace=True)
            mark_failed(root,manifest,exc)
            raise


def worker(root, shard_id):
    root = Path(root)
    plan = read(root/'plan.json')
    verify_plan(plan,root)
    shard = next(s for s in plan['shards'] if s['shard_id']==shard_id)
    work = root/'workers'/str(shard_id)
    with exclusive(work/'worker.lock'), Heartbeat(work/'heartbeat.json') as beat:
        try:
            source_guard(plan)
            gates=current_gate_evidence(plan['task'],plan['subset_kind'])
            if gates is not None:put(work/'gate_recheck.json',gates)
            from .worker import execute
            execute(plan,shard,root,beat)
            if plan['task']!='fixture':
                from .runtime_closure import assert_runtime_closed
                put(work/'runtime_closure_final.json',assert_runtime_closed(read(plan['provenance_file'])))
            beat.assert_healthy()
            source_guard(plan)
            if (root/'cancel.json').exists():
                raise RuntimeError('run cancelled by coordinator')
            put(work/'result.json',{'status':'COMPLETE','coverage':shard['coverage'],'time':now()})
        except BaseException as exc:
            put(work/'result.json',{'status':'FAILED','coverage':shard['coverage'],'time':now(),
                                  'reason':{'type':type(exc).__name__,'message':str(exc)},'traceback':traceback.format_exc()})
            raise


def monitor(root):
    root = Path(root)
    launch = read(root/'coordinator.launch.json')
    while True:
        try:
            manifest = read(root/'manifest.json')
            alive=same_process(launch['process'])
        except (OSError,ValueError,KeyError) as exc:
            put(root/'monitor_unverifiable.json',{'time':now(),'type':type(exc).__name__,'error':str(exc),
                                                 'status_not_changed_without_process_evidence':True},replace=True)
            time.sleep(BEAT_SECONDS)
            continue
        if manifest['execution']['status'] != 'RUNNING':
            return
        try:
            check_heartbeat(root/'coordinator.heartbeat.json',launch['process'],launch['launched_monotonic'])
        except BaseException as exc:
            put(root/'monitor_lost.json',{'time':now(),'error':str(exc),'process_still_alive':alive},replace=True)
            if not alive:
                try:
                    with exclusive(root/'coordinator.lock'):
                        latest = read(root/'manifest.json')
                        if latest['execution']['status']=='RUNNING' and not same_process(launch['process']):
                            put(root/'cancel.json',{'time':now(),'reason':'coordinator lost'},replace=True)
                            mark_failed(root,latest,RuntimeError('coordinator heartbeat lost and process exited'))
                except BlockingIOError:
                    pass
        time.sleep(BEAT_SECONDS)


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['coordinate','worker','monitor'])
    ap.add_argument('root',type=Path)
    ap.add_argument('shard_id',nargs='?',type=int)
    args=ap.parse_args()
    if args.mode=='worker':
        worker(args.root,args.shard_id)
    else:
        globals()[args.mode](args.root)


if __name__=='__main__':
    main()
