"""Reject a valid upstream attached to the wrong scientific consumer before I/O."""
import copy
import pytest
from v4 import artifacts as A, jobs as J


def plans():
    source={'layer':'features','plan_sha256':'a'*64,'dataset_sha256':'b'*64,
            'provenance_sha256':'c'*64,'numeric_environment_sha256':'d'*64,
            'ordered_expected_ids':['one','two'],'subset_kind':'SMOKE_5',
            'expected_coverage':[['one'],['two']],
            'definitions':{'task':'features','version':'v4-1','fixture_only':False,
                           'calibration_only':False,'budgets':{'R010':16}}}
    producer={'task':'features','ids':['one','two'],'subset_kind':'SMOKE_5',
              'science_base':copy.deepcopy(source),'dependencies':[]}
    target=copy.deepcopy(source);target['layer']='masks';target['definitions']['task']='masks'
    consumer={'task':'masks','ids':['one','two'],'subset_kind':'SMOKE_5',
              'science_base':target,'dependencies':[{'role':'features'}]}
    return consumer,producer,{'science':source}


@pytest.mark.parametrize('task,roles',[
    ('features',['masks']),('d4',['features']),('i6',['features']),
    ('masks',[]),('masks',['masks']),('masks',['features','features']),
    ('analysis',[]),('analysis',['features']),('analysis_timing',['features'])])
def test_wrong_dependency_roles_rejected_before_reads(task,roles):
    plan={'task':task,'dependencies':[{'role':r} for r in roles]}
    with pytest.raises(ValueError,match='exact dependency role set'):
        J.dependency(plan,'features')


@pytest.mark.parametrize('field',[
    'ordered_expected_ids','subset_kind','plan_sha256','dataset_sha256',
    'provenance_sha256','numeric_environment_sha256','definitions'])
def test_dependency_from_another_census_or_source_fails_before_payload_read(tmp_path,monkeypatch,field):
    consumer,producer,manifest=plans()
    # Keep the upstream internally self-consistent; only its consumer is wrong.
    if field=='ordered_expected_ids':
        manifest['science'][field]=['one'];producer['ids']=['one']
    elif field=='definitions':manifest['science'][field]['budgets']['R010']=17
    elif field=='subset_kind':
        manifest['science'][field]='PILOT_20';producer['subset_kind']='PILOT_20'
    else:manifest['science'][field]='e'*64
    producer['science_base']=copy.deepcopy(manifest['science'])
    path=tmp_path/'manifest.json';J.put(path,manifest);J.put(tmp_path/'plan.json',producer)
    consumer['dependencies'][0].update(manifest_path=str(path),manifest_sha256=A.file_hash(path))
    calls=[]
    monkeypatch.setattr(J,'verify_plan',lambda *a:calls.append('verify_plan'))
    monkeypatch.setattr(A,'validate_manifest',lambda *a,**k:calls.append('payload_read'))
    with pytest.raises(ValueError,match='differs from consumer|definitions differ from consumer'):
        J.dependency(consumer,'features')
    assert calls==[]


def test_manifest_cannot_disagree_with_its_producer_and_valid_join_passes():
    consumer,producer,manifest=plans()
    J.verify_dependency_roles(consumer)
    J.verify_dependency_join(consumer,producer,manifest)
    manifest['science']['provenance_sha256']='f'*64
    with pytest.raises(ValueError,match='manifest differs from its producer plan'):
        J.verify_dependency_join(consumer,producer,manifest)


def test_timing_calibration_uses_same_real_pilot_without_claiming_formal_analysis():
    consumer,producer,manifest=plans()
    for p in [consumer,producer]:
        p['subset_kind']='PILOT_20';p['science_base']['subset_kind']='PILOT_20'
    producer['task']='masks';producer['science_base']['layer']='masks'
    producer['science_base']['definitions']['task']='masks'
    manifest['science']=copy.deepcopy(producer['science_base'])
    consumer['task']='analysis_timing';consumer['science_base']['layer']='analysis'
    consumer['science_base']['definitions'].update(task='analysis_timing',calibration_only=True)
    consumer['dependencies']=[{'role':'masks'}]
    J.verify_dependency_roles(consumer)
    J.verify_dependency_join(consumer,producer,manifest)
