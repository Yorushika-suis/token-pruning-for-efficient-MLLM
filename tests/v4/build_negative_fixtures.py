"""Persist actual malformed D2 fixtures and their complete rejection evidence."""
from pathlib import Path
import copy
import io
import json
import shutil
import traceback
import zipfile

import numpy as np

from v4 import artifacts as A, jobs as J, spec


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=J.PROJECT/'docs/implementation/v4_20260905/d2_fixtures_r02/SMOKE_5'
    original=J.read(source/'manifest.json')
    expected=spec.mask_keys(original['science']['ordered_expected_ids'],methods=('grid_uniform',))
    results=[]
    cases=('missing_real_tag','path_escape','absolute_path','wrong_upstream','NaN_science',
           'failed_shard','changed_scientific_byte','RUNNING_with_hash','extra_file','symlink','directory_as_file','wrong_coverage')
    for case in cases:
        root=output/case;shutil.copytree(source,root)
        manifest=copy.deepcopy(original)
        first=manifest['execution']['artifact_paths'][0]['relative_path']
        if case=='missing_real_tag':
            path=root/first
            with zipfile.ZipFile(path) as archive:
                entries=[(info,archive.read(info.filename)) for info in archive.infolist()]
            removed=next(info.filename for info,_ in entries if info.filename.endswith('_raw.npy'))
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
                for info,data in entries:
                    if info.filename!=removed:archive.writestr(info,data)
            artifact=next(x for x in manifest['science']['artifacts'] if x['key']==manifest['execution']['artifact_paths'][0]['key'])
            artifact.update(bytes=path.stat().st_size,sha256=A.file_hash(path))
            manifest['content_sha256']=A.digest(A.canonical(manifest['science']))
        elif case=='path_escape':manifest['execution']['artifact_paths'][0]['relative_path']='../outside.npz'
        elif case=='absolute_path':manifest['execution']['artifact_paths'][0]['relative_path']='/tmp/outside.npz'
        elif case=='wrong_upstream':
            manifest['science']['upstreams']=[{'role':'features','hash_spec':A.HASH_SPEC,'content_sha256':'4'*64}]
            manifest['execution']['upstreams']=[{'role':'features','run_id':A.new_execution()['run_id'],
                 'manifest_path':'upstream/manifest.json','content_sha256':'5'*64}]
            manifest['content_sha256']=A.digest(A.canonical(manifest['science']))
        elif case=='NaN_science':manifest['science']['definitions']['bad']=float('nan')
        elif case=='failed_shard':manifest['execution']['workers'][0].update(status='FAILED',exit_code=23)
        elif case=='changed_scientific_byte':manifest['science']['definitions']['scientific_change']=1
        elif case=='RUNNING_with_hash':manifest['execution']['status']='RUNNING'
        elif case=='extra_file':(root/'artifacts/unregistered.npz').write_bytes(b'unregistered')
        elif case=='symlink':
            (root/'linked.npz').symlink_to(root/first)
            manifest['execution']['artifact_paths'][0]['relative_path']='linked.npz'
        elif case=='directory_as_file':manifest['execution']['artifact_paths'][0]['relative_path']='artifacts'
        elif case=='wrong_coverage':manifest['science']['expected_coverage'].pop()
        # Deliberately permit JSON's nonstandard NaN only in this invalid fixture.
        # The production canonical serializer continues to reject it.
        (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=True))
        try:
            A.validate_manifest(manifest,root,expected)
        except Exception as exc:
            result={'case':case,'rejected':True,'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                    'manifest_file_sha256':A.file_hash(root/'manifest.json'),'fixture_only':True}
        else:
            result={'case':case,'rejected':False,'fixture_only':True}
            J.put(root/'rejection.json',result)
            raise AssertionError('invalid fixture accepted: '+case)
        J.put(root/'rejection.json',result);results.append(result)
    J.put(output/'summary.json',{'time':J.now(),'source_positive_manifest_sha256':A.file_hash(source/'manifest.json'),
                                'cases':results,'all_rejected':all(x['rejected'] for x in results),
                                'not_scientific_run':True})
    print(json.dumps({'cases':len(results),'all_rejected':True,'output':str(output)}),flush=True)


if __name__=='__main__':
    import sys
    main(sys.argv[1])
