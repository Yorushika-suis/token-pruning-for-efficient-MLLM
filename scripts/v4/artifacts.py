"""V4 scientific content hashing, deterministic NPZ and fail-closed manifests."""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile

import numpy as np

HASH_SPEC = 'v4-json-1'
SCHEMA_VERSION = 'flashvid-adts-artifact/v4-1'
FORBIDDEN_SCIENCE_KEYS = {'run_id', 'started_at', 'finished_at', 'generated_at', 'mtime',
                          'duration', 'elapsed_seconds', 'physical_path', 'pid', 'worker_pid',
                          'gpu_uuid', 'heartbeat', 'wall_time', 'log_path'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_science(value):
    canonical(value)  # Recursively rejects NaN and Infinity, including metadata.
    if isinstance(value, dict):
        forbidden = FORBIDDEN_SCIENCE_KEYS.intersection(value)
        if forbidden:
            raise ValueError('execution data in science: ' + repr(sorted(forbidden)))
        for child in value.values():
            check_science(child)
    elif isinstance(value, (list,tuple)):
        for child in value:
            check_science(child)


def ordered_set(values, key=lambda x: x):
    pairs = [(canonical(key(v)), v) for v in values]
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError('duplicate logical key')
    return [v for _, v in sorted(pairs, key=lambda x: x[0])]


def normalize_science(science):
    s = json.loads(canonical(science))
    s['expected_coverage'] = ordered_set(s['expected_coverage'])
    s['artifacts'] = ordered_set(s['artifacts'], key=lambda a: a['key'])
    s['upstreams'] = ordered_set(s['upstreams'], key=lambda a: [a['role'], a['hash_spec'], a['content_sha256']])
    check_science(s)
    return s


def safe_path(root, relative, *, exists=True):
    root = Path(root)
    if not isinstance(relative, str) or '\\' in relative or not relative:
        raise ValueError('invalid relative artifact path')
    parts = relative.split('/')
    if any(p in ('', '.', '..') for p in parts) or PurePosixPath(relative).is_absolute():
        raise ValueError('artifact path traversal or absolute path')
    if any('_DEPRECATED_' in p for p in root.parts + tuple(parts)):
        raise ValueError('deprecated inputs are not formal V4 dependencies')
    # Check every component, including root ancestors, before resolving the target.
    for base in [*reversed(root.parents), root]:
        if base.is_symlink():
            raise ValueError('symlink in artifact root')
    p = root
    for part in parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('symlink in artifact path')
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError('artifact escaped run root')
    if exists and (not p.exists() or not stat.S_ISREG(p.lstat().st_mode)):
        raise ValueError('artifact missing or not a regular file')
    return p


def atomic_bytes(path, data, *, replace=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_symlink() or (p.exists() and not replace):
        raise FileExistsError(str(p))
    tmp = p.parent / ('.' + p.name + '.' + str(uuid.uuid4()) + '.tmp')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if replace:
        os.replace(tmp, p)
    else:
        # Link-then-unlink provides no-clobber atomic publication on the same FS.
        os.link(tmp, p)
        tmp.unlink()
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_npz(path, arrays, metadata):
    """ZIP_STORED + NPY v1, little endian, C order, fixed headers/member order."""
    check_science(metadata)
    if 'meta_json' in arrays:
        raise ValueError('meta_json is reserved')
    values = dict(arrays)
    values['meta_json'] = np.frombuffer(canonical(metadata), dtype=np.uint8)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() or p.is_symlink():
        raise FileExistsError(str(p))
    tmp = p.parent / ('.' + p.name + '.' + str(uuid.uuid4()) + '.tmp')
    with tmp.open('xb') as raw:
        with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for name in sorted(values):
                if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
                    raise ValueError('invalid NPZ member name')
                arr = np.asarray(values[name])
                if arr.dtype.kind not in 'biuf' or not np.isfinite(arr).all():
                    raise ValueError('nonfinite or unsupported scientific array')
                arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder('<'))
                info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.extra = b''
                info.comment = b''
                with z.open(info, 'w', force_zip64=True) as f:
                    np.lib.format.write_array(f, arr, version=(1, 0), allow_pickle=False)
        raw.flush()
        os.fsync(raw.fileno())
    os.link(tmp, p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    tmp.unlink()
    return p


def parse_npz(data):
    raw = io.BytesIO(data)
    with zipfile.ZipFile(raw) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or names != sorted(names) or any('/' in n or not n.endswith('.npy') for n in names):
            raise ValueError('invalid or unsorted NPZ members')
        for i in z.infolist():
            if not i.filename[:-4] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in i.filename[:-4]):
                raise ValueError('invalid NPZ member name')
            if i.date_time != (1980, 1, 1, 0, 0, 0) or i.compress_type != zipfile.ZIP_STORED or i.comment or i.create_system!=3 or i.external_attr!=0o100644<<16:
                raise ValueError('NPZ container differs from fixed serialization')
    raw.seek(0)
    with np.load(raw, allow_pickle=False) as z:
        if 'meta_json' not in z.files:
            raise ValueError('scientific metadata missing')
        meta_bytes = z['meta_json']
        if meta_bytes.dtype != np.uint8 or meta_bytes.ndim != 1:
            raise ValueError('invalid metadata encoding')
        meta = json.loads(meta_bytes.tobytes())
        check_science(meta)
        if canonical(meta) != meta_bytes.tobytes():
            raise ValueError('noncanonical scientific metadata')
        arrays, descriptions = {}, {}
        for name in z.files:
            if name == 'meta_json':
                continue
            a = z[name]
            if a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not a.flags.c_contiguous or a.dtype.byteorder == '>':
                raise ValueError('invalid scientific array dtype/order/domain')
            arrays[name] = a
            descriptions[name] = {'shape': list(a.shape), 'dtype': a.dtype.str, 'finite': True}
    if not isinstance(meta.get('coverage'), list) or not meta['coverage']:
        raise ValueError('artifact lacks nonempty parsed coverage')
    if ordered_set(meta['coverage']) != meta['coverage']:
        raise ValueError('artifact coverage must be a sorted logical set')
    validate_payload(arrays, meta)
    return arrays, {'metadata': meta, 'arrays': descriptions, 'finite': True}


def write_failure_npz(path, arrays, metadata):
    """Execution evidence only: preserve even NaN/Inf; never a scientific NPZ."""
    if 'meta_json' in arrays:raise ValueError('reserved forensic metadata member')
    values={name:np.asarray(value) for name,value in arrays.items()}
    if any(v.dtype.kind not in 'biuf' for v in values.values()):
        raise ValueError('forensic captures must be numeric arrays without pickle')
    meta=dict(metadata,kind='forensic_failure',not_a_scientific_dependency=True)
    values['meta_json']=np.frombuffer(canonical(meta),dtype=np.uint8)
    data=io.BytesIO()
    np.savez(data,**values)
    atomic_bytes(path,data.getvalue())
    return Path(path)


def validate_payload(arrays, meta):
    from .spec import ARMS, PROTOCOLS, METHODS, INTERNAL, BUDGETS, PURPOSE_ARMS, METHOD_VERSIONS, RANDOM, seed_set, stream
    from .geometry import binary_mask
    kind = meta.get('kind')
    if kind == 'features':
        shapes = {'pooled169': (64,169,3584), 'attn169': (64,169), 'sig729': (64,729,1152),
                  'attn729': (64,729), 'frame_indices': (64,), 'frame_timestamps': (64,)}
        if set(arrays) != set(shapes) or len(meta['coverage']) != 1:
            raise ValueError('feature members or coverage differ from contract')
        v, p, arm = meta['coverage'][0]
        if p not in PROTOCOLS or arm not in ARMS:
            raise ValueError('invalid feature protocol/arm')
        if meta.get('input_protocol')!=p or meta.get('arm_id')!=arm:
            raise ValueError('feature protocol/arm metadata disagree with coverage')
        for name, shape in shapes.items():
            dtype = '<i8' if name=='frame_indices' else '<f8' if name=='frame_timestamps' else '<f2'
            if arrays[name].shape != shape or arrays[name].dtype.str != dtype:
                raise ValueError('feature shape/dtype mismatch: ' + name)
        if np.any(arrays['frame_indices'] < 0) or np.any(np.diff(arrays['frame_indices']) < 0):
            raise ValueError('invalid frame indices')
        if np.any(arrays['frame_timestamps'] < 0) or np.any(np.diff(arrays['frame_timestamps']) < 0):
            raise ValueError('invalid sampling timestamps')
    elif kind == 'masks':
        records = meta.get('records')
        if not isinstance(records, list) or ordered_set([r['key'] for r in records]) != meta['coverage']:
            raise ValueError('missing or duplicate mask tags')
        wanted = set()
        for r in records:
            key = r['key']
            if len(key) != 7:
                raise ValueError('invalid mask key arity')
            v,p,arm,method,budget,seed,purpose = key
            if p not in PROTOCOLS or method not in METHODS or budget not in BUDGETS or (purpose,arm) not in PURPOSE_ARMS or seed not in seed_set(method):
                raise ValueError('invalid mask coverage key')
            names = [r[k] for k in ('raw_global_indices', 'packed_unique_mask', 'unique_k')]
            if len(set(names)) != 3 or wanted.intersection(names):
                raise ValueError('mask tags alias array members')
            wanted.update(names)
            if not all(n in arrays for n in names):
                raise ValueError('missing tag array')
            raw, packed, unique = [arrays[n] for n in names]
            k = BUDGETS[budget]
            if r.get('nominal_k')!=k or r.get('algorithm_version')!=METHOD_VERSIONS[method]:
                raise ValueError('mask budget or algorithm version differs from registered contract')
            wanted_stream=stream(seed,v,method,purpose,budget,arm) if method in RANDOM else None
            if r.get('random_stream')!=wanted_stream:
                raise ValueError('mask stream material or 128-bit seed differs from its logical key')
            if raw.shape != (64,k) or raw.dtype.str != '<i4' or packed.shape != (64,22) or packed.dtype != np.uint8 or unique.shape != (64,) or unique.dtype.str != '<i2':
                raise ValueError('mask array shape or dtype differs from contract')
            local = raw - np.arange(64, dtype=np.int32)[:,None]*169
            mask = binary_mask(local, k, allow_duplicates=method=='adts_v2')
            if not np.array_equal(np.packbits(mask, axis=-1, bitorder='little'), packed) or not np.array_equal(mask.sum(-1), unique):
                raise ValueError('raw indices, unique mask and unique counts disagree')
            duplicates = r.get('duplicate_slots')
            if not isinstance(duplicates, list) or len(duplicates) != 64 or any(type(x) is not int for x in duplicates) or duplicates != (k-unique).tolist():
                raise ValueError('per-frame duplicate slot registration disagrees with raw indices')
        if wanted != set(arrays):
            raise ValueError('extra unregistered mask array')
        required={(r['key'][3],r['key'][4]) for r in records if r['key'][3] in INTERNAL}
        accepted=meta.get('acceptance',[])
        pairs=[(r['method'],r['budget']) for r in accepted]
        if set(pairs)!=required or len(pairs)!=len(set(pairs)):
            raise ValueError('internal method acceptance records missing, duplicated or extra')
        for r in accepted:
            lengths=r['full64_segment_lengths']
            if r.get('candidate_vs_independent_official_exact') is not True or not isinstance(lengths,list) or not lengths or any(type(n) is not int or n<=0 for n in lengths) or sum(lengths)!=64:
                raise ValueError('internal acceptance is not exact on a full 64-frame segmentation')
        if not meta.get('fixture_only'):
            feature_keys={tuple(r['key'][:3]) for r in records}
            if feature_keys!={tuple(meta.get('upstream_feature_key',[]))} or not isinstance(meta.get('upstream_feature_sha256'),str) or len(meta['upstream_feature_sha256'])!=64 or any(c not in '0123456789abcdef' for c in meta['upstream_feature_sha256']):
                raise ValueError('mask upstream feature identity is missing or inconsistent')
    elif kind in ('validation', 'analysis'):
        from .payload_checks import validate_json_payload
        validate_json_payload(arrays,meta)
    else:
        raise ValueError('unregistered artifact kind')


def describe(root, relative, key):
    p = safe_path(root, relative)
    before = p.stat()
    data = p.read_bytes()
    arrays, parsed = parse_npz(data)
    after = p.stat()
    h = digest(data)
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns) or h != file_hash(p):
        raise ValueError('artifact concurrently changed')
    return {'key': key, 'file_type': 'npz-v4-stored-1', 'bytes': len(data), 'sha256': h, 'parsed': parsed}, arrays


def manifest_schema():
    """Structural schema complements semantic coverage, hash and path checks."""
    hash_type = {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}
    logical_key = {'type': 'array', 'minItems': 1, 'items': {'type': ['string', 'integer']}}
    upstream = {'type': 'object', 'additionalProperties': False,
                'required': ['role', 'hash_spec', 'content_sha256'],
                'properties': {'role': {'type': 'string', 'minLength': 1}, 'hash_spec': {'const': HASH_SPEC}, 'content_sha256': hash_type}}
    artifact = {'type': 'object', 'additionalProperties': False,
                'required': ['key', 'file_type', 'bytes', 'sha256', 'parsed'],
                'properties': {'key': logical_key, 'file_type': {'const': 'npz-v4-stored-1'}, 'bytes': {'type': 'integer', 'minimum': 1},
                               'sha256': hash_type, 'parsed': {'type': 'object', 'required': ['metadata', 'arrays', 'finite'],
                                   'additionalProperties': False, 'properties': {'metadata': {'type': 'object', 'required': ['coverage']},
                                       'arrays': {'type': 'object', 'minProperties': 1}, 'finite': {'const': True}}}}}
    science_props = {'schema_version': {'const': SCHEMA_VERSION}, 'layer': {'enum': ['features', 'masks', 'analysis', 'validation', 'provenance']},
                     'plan_sha256': hash_type, 'ordered_expected_ids': {'type': 'array', 'minItems': 1, 'uniqueItems': True, 'items': {'type': 'string', 'pattern': '^[A-Za-z0-9_-]+$'}},
                     'subset_kind': {'enum': ['SMOKE_5', 'PILOT_20', 'I6_PRESSURE', 'FULL_300', 'GOLDEN']},
                     'dataset_sha256': hash_type, 'definitions': {'type': 'object'}, 'provenance_sha256': hash_type,
                     'numeric_environment_sha256': hash_type, 'expected_coverage': {'type': 'array', 'minItems': 1, 'items': logical_key},
                     'artifacts': {'type': 'array', 'items': artifact}, 'upstreams': {'type': 'array', 'items': upstream}}
    execution_props = {'run_id': {'type': 'string', 'pattern': r'^\d{8}T\d{6}Z_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'},
                       'status': {'enum': ['RUNNING', 'COMPLETE', 'FAILED']}, 'started_at': {'type': 'string'},
                       'finished_at': {'type': ['string', 'null']}, 'responsibility': {'type': 'string', 'minLength': 1},
                       'artifact_paths': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False, 'required': ['key', 'relative_path'],
                            'properties': {'key': logical_key, 'relative_path': {'type': 'string', 'minLength': 1}}}},
                       'sidecars': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                            'required': ['relative_path', 'bytes', 'sha256'], 'properties': {'relative_path': {'type': 'string'},
                            'bytes': {'type': 'integer', 'minimum': 0}, 'sha256': hash_type}}},
                       'upstreams': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                            'required': ['role', 'run_id', 'manifest_path', 'content_sha256'], 'properties': {
                            'role': {'type': 'string'}, 'run_id': {'type': 'string'}, 'manifest_path': {'type': 'string'}, 'content_sha256': hash_type}}},
                       'workers': {'type': 'array', 'items': {'type': 'object', 'required': ['shard_id', 'status', 'exit_code', 'coverage'],
                            'properties': {'shard_id': {'type': 'integer', 'minimum': 0}, 'status': {'enum': ['RUNNING', 'COMPLETE', 'FAILED']},
                            'exit_code': {'type': ['integer', 'null']}, 'coverage': {'type': 'array', 'items': logical_key}}}},
                       'reason': {'type': ['object', 'null']}, 'runtime': {'type': 'object'}}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'additionalProperties': False,
            'required': ['science', 'execution', 'content_hash_spec', 'content_sha256'],
            'properties': {'science': {'type': 'object', 'additionalProperties': False, 'required': list(science_props), 'properties': science_props},
                           'execution': {'type': 'object', 'additionalProperties': False, 'required': list(execution_props), 'properties': execution_props},
                           'content_hash_spec': {'const': HASH_SPEC}, 'content_sha256': {'anyOf': [hash_type, {'type': 'null'}]}},
            'allOf': [{'if': {'properties': {'execution': {'properties': {'status': {'const': 'COMPLETE'}}}}},
                       'then': {'properties': {'content_sha256': hash_type}},
                       'else': {'properties': {'content_sha256': {'type': 'null'}}}}]}


def validate_manifest(manifest, root, expected_coverage, *, upstream_resolver=None, verify_bytes=True):
    import jsonschema
    from .spec import PLAN_SHA256
    jsonschema.Draft202012Validator(manifest_schema()).validate(manifest)
    s, e = manifest['science'], manifest['execution']
    if s['plan_sha256'] != PLAN_SHA256 or normalize_science(s) != s:
        raise ValueError('wrong plan or noncanonical scientific collections')
    if s['expected_coverage'] != ordered_set(expected_coverage):
        raise ValueError('declared coverage differs from independently planned coverage')
    if e['status'] != 'COMPLETE':
        if manifest['content_sha256'] is not None:
            raise ValueError('incomplete run cannot have content hash')
        raise ValueError('dependency is not COMPLETE')
    if not e['finished_at'] or e['reason'] is not None or not e['workers']:
        raise ValueError('COMPLETE lacks successful execution evidence')
    if any(w['status'] != 'COMPLETE' or w['exit_code'] != 0 for w in e['workers']):
        raise ValueError('unsuccessful worker in COMPLETE manifest')
    if len({w['shard_id'] for w in e['workers']}) != len(e['workers']):
        raise ValueError('duplicate worker shard identity')
    workers = [k for w in e['workers'] for k in w['coverage']]
    if ordered_set(workers) != s['expected_coverage']:
        raise ValueError('shard coverage overlaps, misses, or adds work')
    if digest(canonical(s)) != manifest['content_sha256']:
        raise ValueError('science content fingerprint mismatch')
    arts = {canonical(a['key']): a for a in s['artifacts']}
    paths = {canonical(x['key']): x['relative_path'] for x in e['artifact_paths']}
    if len(arts) != len(s['artifacts']) or len(paths) != len(e['artifact_paths']) or set(arts) != set(paths) or len(set(paths.values())) != len(paths):
        raise ValueError('artifact key/path mapping is not one-to-one')
    coverage, snapshots = [], []
    for key, a in arts.items():
        meta=a['parsed']['metadata']
        if meta.get('kind') != s['layer']:
            raise ValueError('artifact kind does not match run layer')
        for flag in ('fixture_only','calibration_only'):
            if bool(meta.get(flag))!=bool(s['definitions'].get(flag)):
                raise ValueError('artifact fixture/calibration status differs from run identity')
        if s['layer']=='analysis' and meta.get('ordered_expected_ids')!=s['ordered_expected_ids']:
            raise ValueError('analysis artifact census differs from manifest authority')
        if verify_bytes:
            observed, _ = describe(root, paths[key], a['key'])
            if observed != a:
                raise ValueError('artifact bytes/metadata mismatch: ' + paths[key])
            snapshots.append((paths[key], a['sha256']))
        coverage.extend(a['parsed']['metadata']['coverage'])
    if ordered_set(coverage) != s['expected_coverage']:
        raise ValueError('parsed artifact coverage is incomplete, duplicate or extra')
    allowed_paths = set(paths.values()) | {x['relative_path'] for x in e['sidecars']}
    if len(allowed_paths) != len(paths) + len(e['sidecars']):
        raise ValueError('duplicate physical file or sidecar path')
    if verify_bytes:
        for x in e['sidecars']:
            file = safe_path(root, x['relative_path'])
            data = file.read_bytes()
            if len(data) != x['bytes'] or digest(data) != x['sha256']:
                raise ValueError('execution sidecar bytes changed')
            if file.suffix == '.json':
                if not isinstance(json.loads(data), dict):
                    raise ValueError('execution sidecar must be a JSON object')
        # All payloads live under artifacts/, preventing hidden extra scientific files.
        actual_paths = set()
        payload_dir = Path(root) / 'artifacts'
        if payload_dir.is_symlink():
            raise ValueError('symlink payload directory')
        for file in payload_dir.rglob('*'):
            if file.is_symlink():
                raise ValueError('symlink under payload directory')
            if file.is_file():
                actual_paths.add(file.relative_to(root).as_posix())
            elif not file.is_dir():
                raise ValueError('nonregular entry under payload directory')
        if actual_paths != {v for v in paths.values()}:
            raise ValueError('unregistered extra or missing scientific files')
    logical_up = {(u['role'], u['hash_spec'], u['content_sha256']) for u in s['upstreams']}
    execution_up = {(u['role'], HASH_SPEC, u['content_sha256']) for u in e['upstreams']}
    if logical_up != execution_up or len(execution_up) != len(e['upstreams']):
        raise ValueError('execution and scientific upstreams do not match')
    if s['upstreams'] and upstream_resolver is None:
        raise ValueError('dependency resolver is required')
    if s['layer'] in ('masks','analysis') and not s['definitions'].get('fixture_only'):
        required_role='features' if s['layer']=='masks' else 'masks'
        if {u['role'] for u in s['upstreams']}!={required_role}:
            raise ValueError('scientific layer lacks its exact required upstream role')
    for u in e['upstreams']:
        actual = upstream_resolver(u)
        if actual['content_sha256'] != u['content_sha256'] or actual['execution']['run_id'] != u['run_id'] or actual['execution']['status'] != 'COMPLETE':
            raise ValueError('wrong or incomplete upstream')
        if s['layer']=='masks' and u['role']=='features':
            source_artifacts={canonical(a['key']):a for a in actual['science']['artifacts']}
            for artifact in s['artifacts']:
                meta=artifact['parsed']['metadata'];source=source_artifacts.get(canonical(meta['upstream_feature_key']))
                if source is None or source['sha256']!=meta['upstream_feature_sha256']:
                    raise ValueError('mask declared feature differs from actual upstream artifact')
    if verify_bytes:
        for relative, h in snapshots:
            if file_hash(safe_path(root, relative)) != h:
                raise ValueError('input changed during manifest validation')
    return manifest['content_sha256']


def new_execution():
    time = dt.datetime.now(dt.timezone.utc)
    run_id = time.strftime('%Y%m%dT%H%M%SZ_') + str(uuid.uuid4())
    return {'run_id': run_id, 'status': 'RUNNING', 'started_at': time.isoformat(), 'finished_at': None,
            'responsibility': 'Codex acting for former Claude and collaborator roles; user and Codex only',
            'artifact_paths': [], 'sidecars': [], 'upstreams': [], 'workers': [], 'reason': None, 'runtime': {}}
