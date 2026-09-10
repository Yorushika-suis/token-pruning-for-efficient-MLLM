"""Unregistered delayed imports cannot pass a runtime closure check."""
from pathlib import Path
import pytest

from v4 import runtime_closure as C, runtime as R


def test_unregistered_lazy_library_fails_before_success(tmp_path,monkeypatch):
    present=tmp_path/'present.so';present.write_bytes(b'present')
    late=tmp_path/'late.so';late.write_bytes(b'late')
    provenance={'execution':{'files':[{'physical_path':str(present),'logical_path':'present.so','sha256':'1'*64}]}}
    monkeypatch.setattr(C,'loaded_files',lambda:(set(),{present,late}))
    with pytest.raises(ValueError,match='absent from D3'):C.assert_runtime_closed(provenance)


def test_actual_official_processor_configuration_is_explicit():
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    processor=SigLipImageProcessor()
    expected=R.processor_contract()
    assert R.processor_contract(processor)==expected
    state=expected['configuration']
    assert state['size']==[384,384] and state['image_mean']==[.5,.5,.5] and state['image_std']==[.5,.5,.5]
    assert state['rescale_factor']==1/255 and state['data_format']=='channels_first'
    processor.image_mean=(0,0,0)
    assert R.processor_contract(processor)!=expected
def test_official_registration_must_match_verified_bytes_and_media_entries(tmp_path):
    from v4 import provenance as P
    official={'files':[{'logical_path':'a.py','sha256':'1'*64,'bytes':10}]}
    records=[{'logical_path':'code/FlashVID/a.py','sha256':'1'*64,'bytes':10}]
    P.assert_registered_official(records,official)
    records[0]['sha256']='2'*64
    with pytest.raises(ValueError,match='registered source bytes'):P.assert_registered_official(records,official)
    (tmp_path/'v.mp4').write_bytes(b'fixture')
    P.assert_media_entries(['v'],tmp_path)
    (tmp_path/'v.mp4.bak').write_bytes(b'extra')
    with pytest.raises(ValueError,match='media directory'):P.assert_media_entries(['v'],tmp_path)


def test_prompt_source_contract_and_task_state_restored_on_entry_failure(monkeypatch):
    import contextlib
    from types import SimpleNamespace
    from data_manifest import PROJECT_ROOT
    monkeypatch.syspath_prepend(str(PROJECT_ROOT/'code/FlashVID/lmms-eval'))
    from lmms_eval.models.simple.llava_vid import LlavaVid
    contract=R.prompt_contract()
    assert contract['conversation_template']=='qwen_1_5'
    assert set(contract['sources'])=={'question_prompt','conversation_get_prompt'}
    assert all(len(x['source_file_sha256'])==64 and 'source_file' not in x for x in contract['sources'].values())
    old={'keep':'original'};config=SimpleNamespace(retention_ratio=.1)
    evaluation=SimpleNamespace(model=SimpleNamespace(flashvid_config=config),task_dict=old)
    @contextlib.contextmanager
    def context(*a,**kw):yield SimpleNamespace()
    def error(*a,**kw):raise ValueError('synthetic entry failure')
    monkeypatch.setattr(R,'observe_preprocessing',context);monkeypatch.setattr(R,'VisionCapture',context)
    monkeypatch.setattr(R,'CompressionCapture',context);monkeypatch.setattr(LlavaVid,'generate_until',error)
    with pytest.raises(ValueError,match='synthetic entry failure'):
        R.official_wrapper(evaluation,'fixture',{'context':'fixture','doc':{}},None,None,None,'controlled-gray','R010')
    assert evaluation.task_dict is old and evaluation.model.flashvid_config is config


def test_parent_snapshot_survives_disk_swap_and_question_coverage_is_exact(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import pyarrow.parquet as pq
    from data_manifest import load_video_ids, PROJECT_ROOT, ManifestError
    monkeypatch.syspath_prepend(str(PROJECT_ROOT/'code/FlashVID/lmms-eval'))
    from lmms_eval.tasks.videomme import utils
    path=tmp_path/'data/short_subset.json';path.parent.mkdir()
    parent={'short_video_ids':['b',{'videoID':'a'}],'parquet':'frozen.parquet'}
    path.write_text(json.dumps(parent));monkeypatch.setattr(R,'PROJECT',tmp_path)
    frozen,data,ids=R.dataset_snapshot()
    assert ids==load_video_ids(path)==['b','a'] and data['parquet']=='frozen.parquet'
    path.write_text(json.dumps({'short_video_ids':['wrong'],'parquet':'wrong.parquet'}))
    rows=[{'videoID':'a','duration':'long'}, {'videoID':'a','duration':'short'},
          {'videoID':'b','duration':'short'}, {'videoID':'a','duration':'short'}]
    seen=[]
    def table(buffer):
        seen.append(buffer.read());return SimpleNamespace(to_pylist=lambda:rows)
    monkeypatch.setattr(pq,'read_table',table)
    monkeypatch.setattr(utils,'videomme_doc_to_text',lambda row,kw:row['videoID']+kw['post_prompt'])
    result=R.first_questions(b'frozen-parquet',manifest_bytes=frozen)
    assert seen==[b'frozen-parquet'] and set(result)=={'a','b'}
    assert result['a']['row_number']==1 and result['b']['row_number']==2
    assert result['a']['context']=='aThe best answer is:'
    with pytest.raises(ValueError,match='first-question coverage'):
        R.first_questions(b'frozen-parquet')
    for bad in [['a','a'],['../a'],[],[{'videoID':'a','extra':1}]]:
        with pytest.raises(ManifestError):
            R.dataset_snapshot(json.dumps(dict(parent,short_video_ids=bad)).encode())
