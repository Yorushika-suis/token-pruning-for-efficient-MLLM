"""A real completed definition joins its plan only with byte-bound D3 evidence."""
import copy
import pytest
from v4 import artifacts as A, jobs as J
from test_dependency_join import plans


def fixture(tmp_path):
    consumer,producer,manifest=plans()
    records=[{'logical_path':n,'sha256':c*64,'bytes':1} for n,c in
             [('runtime/a.py','1'),('runtime/b.py','2'),('runtime/lib.so','3')]]
    science={'files':records};content=A.digest(A.canonical(science))
    path=tmp_path/'provenance.json'
    J.put(path,{'science':science,'content_sha256':content})
    for target in (producer['science_base'],consumer['science_base'],manifest['science']):
        target['provenance_sha256']=content
    producer.update(provenance_file=str(path),guarded_files=[{'physical_path':str(path),'bytes':path.stat().st_size,'sha256':A.file_hash(path)}],
                    shards=[{'shard_id':0},{'shard_id':1}])
    binding={'module':'fixture','attribute':'__file__','logical_path':'runtime/a.py','sha256':'1'*64}
    library={'logical_path':'runtime/lib.so','sha256':'3'*64}
    closure={'all_registered':True,'scientific_bindings':[binding],'scientific_libraries':[library]}
    manifest['science']['definitions'].update(actual_runtime_bindings=[binding],actual_runtime_libraries=[library])
    manifest['execution']={'sidecars':[]}
    for shard in producer['shards']:
        p=tmp_path/'workers'/str(shard['shard_id'])/'runtime_closure_final.json'
        J.put(p,closure)
        manifest['execution']['sidecars'].append({'relative_path':str(p.relative_to(tmp_path)),'bytes':p.stat().st_size,'sha256':A.file_hash(p)})
    return consumer,producer,manifest


def rewrite_closure(tmp_path,manifest,change):
    descriptor=manifest['execution']['sidecars'][0]
    path=tmp_path/descriptor['relative_path'];value=J.read(path);change(value)
    J.put(path,value,replace=True)
    descriptor.update(bytes=path.stat().st_size,sha256=A.file_hash(path))


def test_actual_completed_runtime_fields_join_without_changing_registered_definitions(tmp_path):
    consumer,producer,manifest=fixture(tmp_path)
    J.verify_dependency_join(consumer,producer,manifest)
    J.verify_dependency_runtime(producer,manifest,tmp_path)
    manifest['science']['definitions']['budgets']['R010']=17
    with pytest.raises(ValueError,match='producer plan: definitions'):
        J.verify_dependency_join(consumer,producer,manifest)


@pytest.mark.parametrize('change',[
    lambda m:m['execution']['sidecars'].pop(),
    lambda m:m['execution']['sidecars'].append(copy.deepcopy(m['execution']['sidecars'][0])),
    lambda m:m['science']['definitions'].pop('actual_runtime_bindings'),
    lambda m:m['science']['definitions']['actual_runtime_libraries'].clear()])
def test_missing_duplicate_or_inconsistent_runtime_evidence_is_rejected(tmp_path,change):
    _,producer,manifest=fixture(tmp_path);change(manifest)
    with pytest.raises(ValueError):J.verify_dependency_runtime(producer,manifest,tmp_path)


@pytest.mark.parametrize('change',[
    lambda c:c.update(all_registered=False),
    lambda c:c['scientific_bindings'][0].update(sha256='f'*64),
    lambda c:c['scientific_bindings'][0].update(logical_path='unregistered.py'),
    lambda c:c['scientific_bindings'][0].update(attribute='unregistered_attribute'),
    lambda c:c['scientific_bindings'].append(copy.deepcopy(c['scientific_bindings'][0])),
    lambda c:c['scientific_bindings'][0].update(logical_path='runtime/b.py',sha256='2'*64)])
def test_rehashed_sidecar_still_requires_registered_and_consistent_origins(tmp_path,change):
    _,producer,manifest=fixture(tmp_path);rewrite_closure(tmp_path,manifest,change)
    with pytest.raises(ValueError):J.verify_dependency_runtime(producer,manifest,tmp_path)


@pytest.mark.parametrize('target',['provenance','closure'])
def test_actual_evidence_byte_mutation_fails(tmp_path,target):
    _,producer,manifest=fixture(tmp_path)
    path=tmp_path/('provenance.json' if target=='provenance' else 'workers/0/runtime_closure_final.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='bytes changed'):
        J.verify_dependency_runtime(producer,manifest,tmp_path)


def test_runtime_evidence_cannot_be_predeclared(tmp_path):
    consumer,producer,manifest=fixture(tmp_path)
    producer['science_base']['definitions']['actual_runtime_bindings']=[]
    with pytest.raises(ValueError,match='cannot be predeclared'):
        J.verify_dependency_join(consumer,producer,manifest)
