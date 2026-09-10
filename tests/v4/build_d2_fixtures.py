"""Create physical SMOKE_5 and FULL_300 golden mask runs for contract testing.

These are labeled fixture_only and must never be used by scientific analysis.
The full fixture is a complete single-method run (grid_uniform), not a claim
that the eight-method research objective or feature extraction is complete.
"""
from pathlib import Path
import hashlib
import json
import sys
import datetime as dt
import struct

import numpy as np

PROJECT=Path('/mnt/tidalfs-hssh01/dataset/ai_tradesearch/flashvid_adts_spatial')
sys.path.insert(0,str(PROJECT/'scripts'))
from v4 import artifacts as A, spec, selection as L, geometry as G


def make(root,ids,subset_kind,dataset_hash):
    root.mkdir(exist_ok=False)
    descriptors,paths=[],[]
    expected=spec.mask_keys(ids,methods=('grid_uniform',))
    for v in ids:
        for p in spec.PROTOCOLS:
            for arm in spec.ARMS:
                coverage=[key for key in spec.mask_keys([v],methods=('grid_uniform',)) if key[1]==p and key[2]==arm]
                coverage=A.ordered_set(coverage)
                arrays,records={},[]
                for i,key in enumerate(coverage):
                    budget=key[4];k=spec.BUDGETS[budget]
                    local=np.tile(L.grid_points(k),(64,1))
                    mask=G.binary_mask(local,k)
                    names={'raw_global_indices':f't{i:03d}_raw','packed_unique_mask':f't{i:03d}_mask','unique_k':f't{i:03d}_unique'}
                    arrays[names['raw_global_indices']]=(local+169*np.arange(64)[:,None]).astype('<i4')
                    arrays[names['packed_unique_mask']]=np.packbits(mask,axis=-1,bitorder='little')
                    arrays[names['unique_k']]=mask.sum(-1).astype('<i2')
                    records.append({'key':key,**names,'nominal_k':k,'algorithm_version':spec.METHOD_VERSIONS['grid_uniform'],
                                    'random_stream':None,'duplicate_slots':(k-mask.sum(-1)).tolist()})
                metadata={'kind':'masks','coverage':coverage,'records':records,'fixture_only':True}
                relative=f'artifacts/{v}_{p}_{arm}.npz'
                A.write_npz(root/relative,arrays,metadata)
                key=[v,p,arm,'grid_uniform']
                d,_=A.describe(root,relative,key)
                descriptors.append(d);paths.append({'key':key,'relative_path':relative})
    execution=A.new_execution()
    execution.update(status='COMPLETE',finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),artifact_paths=paths,
                     workers=[{'shard_id':0,'status':'COMPLETE','exit_code':0,'coverage':A.ordered_set(expected)}])
    source_payload={'purpose':'D2 synthetic contract fixtures only','script_sha256':A.file_hash(__file__),
                    'dataset_sha256':dataset_hash,'methods':['grid_uniform'],'not_scientific_run':True}
    numeric={'numpy':np.__version__,'array_serialization':'npz-v4-stored-1','dtype':'little_endian_i4_u1_i2'}
    science=A.normalize_science({'schema_version':A.SCHEMA_VERSION,'layer':'masks','plan_sha256':spec.PLAN_SHA256,
              'ordered_expected_ids':ids,'subset_kind':subset_kind,'dataset_sha256':dataset_hash,
              'definitions':{'fixture_only':True,'methods':['grid_uniform'],'frames':64,'purpose_arm_pairs':spec.PURPOSE_ARMS,
                             'budgets':spec.BUDGETS,'goal_completion_claim':False},
              'provenance_sha256':A.digest(A.canonical(source_payload)),'numeric_environment_sha256':A.digest(A.canonical(numeric)),
              'expected_coverage':expected,'artifacts':descriptors,'upstreams':[]})
    manifest={'science':science,'execution':execution,'content_hash_spec':A.HASH_SPEC,'content_sha256':A.digest(A.canonical(science))}
    A.validate_manifest(manifest,root,expected)
    A.atomic_bytes(root/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode())
    return {'path':str(root.relative_to(PROJECT)),'manifest_sha256':A.file_hash(root/'manifest.json'),
            'content_sha256':manifest['content_sha256'],'ids':len(ids),'artifacts':len(descriptors),
            'coverage_tuples':len(expected),'artifact_bytes':sum(d['bytes'] for d in descriptors),
            'validated_from_actual_bytes':True,'fixture_only':True,'provenance_payload':source_payload,'numeric_environment':numeric}


def main():
    out=PROJECT/'docs/implementation/v4_20260905/d2_fixtures_r02'
    out.mkdir(exist_ok=False)
    raw=(PROJECT/'data/short_subset.json').read_bytes();ids=json.loads(raw)['short_video_ids']
    assert len(ids)==300 and len(set(ids))==300
    A.atomic_bytes(out/'manifest.schema.json',json.dumps(A.manifest_schema(),indent=2).encode())
    summaries=[make(out/'SMOKE_5',ids[:5],'SMOKE_5',A.digest(raw)),make(out/'FULL_300',ids,'FULL_300',A.digest(raw))]
    vectors=[]
    inputs=[(0,'_KU4X06VNiE',spec.RANDOM[0],'protocol_pair','R010','identity'),
            (4,'fFjv93ACGo8',spec.RANDOM[1],'geometry_arm','R025',spec.ARMS[3]),
            (1,'-ApL8d6tX5U',spec.RANDOM[2],'geometry_arm','R015',spec.ARMS[1])]
    for s,v,m,p,b,a in inputs:
        record=spec.stream(s,v,m,p,b,a)
        fields=[b'flashvid-adts/v4-stream-1',s.to_bytes(8,'big'),v.encode(),(m+'/v4-1').encode(),p.encode(),b.encode(),a.encode()]
        independent=b''.join(len(f).to_bytes(4,'big')+f for f in fields)
        assert record['material_hex']==independent.hex()
        assert record['seed128']==int(hashlib.blake2b(independent,digest_size=16).hexdigest(),16)
        vectors.append({'input':{'run_seed':s,'video_id':v,'method':m,'purpose':p,'budget_id':b,'arm_id':a},
                        **record,'independent_recomputed':True})
    A.atomic_bytes(out/'stream_golden_vectors.json',json.dumps(vectors,indent=2).encode())
    result={'time':dt.datetime.now().astimezone().isoformat(),'golden_runs':summaries,
            'all_distinct_random_streams':spec.check_streams(ids),'fixture_scope':'Complete byte-verified single grid_uniform method over prescribed 5/300 IDs, both protocols and all legal purpose/arm/budget keys. Not formal science or a full-method completion claim.'}
    A.atomic_bytes(out/'summary.json',json.dumps(result,indent=2).encode())
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
