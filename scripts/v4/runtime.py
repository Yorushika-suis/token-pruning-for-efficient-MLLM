"""Pinned official evaluation entry and direct vision path for D4/D5/D6."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from .capture import CaptureComplete, CompressionCapture, VisionCapture, identity, tensor_hash
from .geometry import transform
from .numerics import compare, CAPTURE_AXES

PROJECT = Path(__file__).resolve().parents[2]


class ObservationMismatch(BaseException):
    """Preserve capture evidence through the official evaluator's except Exception."""
    def __init__(self,message,**arrays):
        super().__init__(message)
        self.arrays=arrays


def processor_contract(processor=None):
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
    if processor is None:processor=SigLipImageProcessor()
    if type(processor) is not SigLipImageProcessor:
        raise ValueError('actual processor class differs from pinned official constructor')
    state={'image_mean':list(processor.image_mean),'image_std':list(processor.image_std),
           'size':list(processor.size),'crop_size':dict(processor.crop_size),
           'resample':int(processor.resample),'rescale_factor':processor.rescale_factor,
           'data_format':processor.data_format.value}
    source=identity(SigLipImageProcessor.preprocess)
    return {'class_module':SigLipImageProcessor.__module__,'class_name':SigLipImageProcessor.__qualname__,
            'preprocess_source_sha256':source['source_file_sha256'],'configuration':state}


def configure():
    os.environ['HF_HOME'] = str(PROJECT/'weights')
    os.environ['HF_HUB_CACHE'] = str(PROJECT/'weights/hub')
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(PROJECT/'weights/hub')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT/'weights/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for name in ('FLASHVID_MODEL_SNAPSHOT','FLASHVID_VISION_TOWER_SNAPSHOT'):
        if os.environ.get(name):
            raise ValueError('unregistered model-source override: '+name)
    sys.dont_write_bytecode = True
    for path in (PROJECT/'code/FlashVID',PROJECT/'code/FlashVID/lmms-eval',PROJECT/'scripts'):
        if str(path) not in sys.path:
            sys.path.insert(0,str(path))
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


def build():
    """Call the unchanged official evaluator constructor; no decoder is run."""
    configure()
    from adts_harness import resolve_cached_model_sources, MODEL_PATH, install_llm_tripwire
    from lmms_eval.models.simple.llava_vid import LlavaVid
    from huggingface_hub import constants as hf_constants
    from transformers.utils import hub as transformers_hub
    for actual in (hf_constants.HF_HUB_CACHE,transformers_hub.TRANSFORMERS_CACHE):
        if Path(actual).resolve()!=(PROJECT/'weights/hub').resolve():
            raise ValueError('imported Hugging Face cache constants differ from registered source root')
    # Verified immutable cached refs, not remote main; D3 hashes actual bytes.
    resolve_cached_model_sources()
    evaluation = LlavaVid(pretrained=MODEL_PATH,torch_dtype='float16',device='cuda:0',device_map='cuda:0',
                          batch_size=1,attn_implementation='flash_attention_2',conv_template='qwen_1_5',
                          max_frames_num=64,force_sample=True,add_time_instruction=False,
                          mm_spatial_pool_mode='average',mm_newline_position='frame',
                          enable_flashvid=True,retention_ratio=0.10,do_segment=True,segment_threshold=0.9,
                          min_segment_num=8,complementary_segment=True,token_selection_method='attn_div_v2',
                          alpha=0.7,temporal_threshold=0.8,expansion=1.25,pruning_layer=20,llm_retention_ratio=0.3)
    model = evaluation.model
    assert model.config.mm_spatial_pool_mode=='average' and model.config.mm_newline_position=='frame'
    assert model.get_vision_tower().num_patches_per_side==27
    install_llm_tripwire(model)
    return evaluation


def decode(path):
    from decord import VideoReader,cpu
    reader = VideoReader(str(path),ctx=cpu(0),num_threads=1)
    n,fps = len(reader),reader.get_avg_fps()
    if n<=0 or not np.isfinite(fps) or fps<=0:
        raise ValueError('invalid media frame count/fps')
    indices = np.linspace(0,n-1,64,dtype=int)
    frames = reader.get_batch(indices.tolist()).asnumpy()
    timestamps = np.asarray([int(i)/fps for i in indices],dtype='<f8')
    return frames,indices.astype('<i8'),timestamps,{'total_frames':n,'fps':fps,'duration':n/fps}


def dataset_snapshot(manifest_bytes=None):
    """Derive the complete dataset selection from one immutable parent buffer."""
    from data_manifest import _canonical_id, ManifestError
    if manifest_bytes is None:
        manifest_bytes=(PROJECT/'data/short_subset.json').read_bytes()
    if not isinstance(manifest_bytes,bytes):raise TypeError('manifest snapshot must be bytes')
    manifest=json.loads(manifest_bytes)
    if not isinstance(manifest,dict):raise ManifestError('dataset manifest must be a JSON object')
    raw_ids=manifest.get('short_video_ids')
    if not isinstance(raw_ids,list) or not raw_ids:raise ManifestError('short_video_ids must be a non-empty list')
    ids=[_canonical_id(value,index) for index,value in enumerate(raw_ids)]
    if len(set(ids))!=len(ids):raise ManifestError('duplicate video IDs in dataset manifest')
    if not isinstance(manifest.get('parquet'),str) or not manifest['parquet']:
        raise ManifestError('dataset parquet path must be a non-empty string')
    return manifest_bytes,manifest,ids


def first_questions(parquet_bytes=None, *, manifest_bytes=None):
    import pyarrow.parquet as pq
    from lmms_eval.tasks.videomme.utils import videomme_doc_to_text
    _,manifest,ids=dataset_snapshot(manifest_bytes)
    if parquet_bytes is None:parquet_bytes=Path(manifest['parquet']).read_bytes()
    table = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()
    first = {}
    for row_number, row in enumerate(table):
        vid = row['videoID']
        if row['duration']=='short' and vid in ids and vid not in first:
            first[vid] = {'row_number':row_number,'doc':row,
                          'context':videomme_doc_to_text(row,{'pre_prompt':'','post_prompt':'The best answer is:'})}
    if set(first) != set(ids):
        raise ValueError('first-question coverage differs from census')
    return first


def prompt_contract():
    from llava.conversation import conv_templates
    from lmms_eval.tasks.videomme.utils import videomme_doc_to_text
    template=conv_templates['qwen_1_5']
    sources={name:{k:v for k,v in identity(fn).items() if k!='source_file'}
             for name,fn in [('question_prompt',videomme_doc_to_text),('conversation_get_prompt',template.get_prompt)]}
    return {'conversation_template':'qwen_1_5','sources':sources,
            'configuration':{'system':template.system,'roles':list(template.roles),'messages':template.messages,
                             'offset':template.offset,'sep_style':template.sep_style.name,'sep':template.sep,
                             'sep2':template.sep2,'version':template.version}}


@torch.inference_mode()
def direct(evaluation, frames, protocol, arm, chunk=64, budget=None):
    model = evaluation.model
    base = evaluation._image_processor.preprocess(frames,return_tensors='pt')['pixel_values']
    if base.shape != (64,3,384,384):
        raise ObservationMismatch('processor does not yield registered pixel shape',actual_processor_pixels=base)
    pixels = transform(base.to(device=evaluation.device,dtype=torch.float16),protocol,arm)
    with VisionCapture(model,chunk) as capture:
        projected,attention = model.encode_images(pixels)
        pooled = model.get_2dPool(projected)
        attn = model.get_2dPool(attention.unsqueeze(-1)).squeeze(-1)
        tensors = capture.tensors()
    for name,x in (('pooled169',pooled),('attn169',attn)):
        if not torch.equal(x,tensors[name]):
            raise ObservationMismatch('direct capture and actual pooling differ',actual_pool=x,reference_pool=tensors[name])
    result = {'tensors':tensors,'trace':capture.trace}
    if budget is not None:
        config = copy.deepcopy(model.flashvid_config)
        config.retention_ratio = int(budget[1:])/100
        with CompressionCapture() as comp:
            comp.direct(pooled.clone(),attn.clone(),config)
            lengths,selected = comp.selected()
        result.update(lengths=lengths,selected=selected,compression_trace=comp.calls,
                      snapshot_hashes=comp.snapshot_hashes,registry=comp.registry_identities)
    return result


@contextlib.contextmanager
def observe_preprocessing(evaluation, protocol, expected_frames, expected_indices, expected_timestamps):
    """Run original sampling/processor; assert shared decoded input bytes explicitly."""
    import lmms_eval.models.simple.llava_vid as official
    original_reader, original_processor = official.VideoReader,evaluation._image_processor
    observed = {'sampling':[],'processor_calls':0}

    class Reader:
        def __init__(self,*args,**kwargs):
            self.actual = original_reader(*args,**kwargs)
        def __len__(self):
            return len(self.actual)
        def get_avg_fps(self):
            return self.actual.get_avg_fps()
        def get_batch(self,indices):
            if not np.array_equal(np.asarray(indices,dtype=np.int64),expected_indices):
                raise ObservationMismatch('official/direct sampling index mismatch',actual_indices=np.asarray(indices),reference_indices=expected_indices)
            times = np.asarray([int(i)/self.actual.get_avg_fps() for i in indices],dtype=np.float64)
            if not np.array_equal(times,expected_timestamps):
                raise ObservationMismatch('official/direct sampling timestamp mismatch',actual_timestamps=times,reference_timestamps=expected_timestamps)
            result = self.actual.get_batch(indices)
            if not np.array_equal(result.asnumpy(),expected_frames):
                raise ObservationMismatch('independent official decode differs from common input bytes',actual_frames=result.asnumpy(),reference_frames=expected_frames)
            observed['sampling'].append({'frame_indices':list(map(int,indices)),'frame_timestamps':times.tolist()})
            return result

    class Processor:
        def preprocess(self,frames,*args,**kwargs):
            if not np.array_equal(frames,expected_frames):
                raise ObservationMismatch('official processor did not receive identical decoded frames',actual_frames=np.asarray(frames),reference_frames=expected_frames)
            result = original_processor.preprocess(frames,*args,**kwargs)
            observed['processor_calls'] += 1
            base = result['pixel_values']
            # Gray is explicit post-processor injection, never described as official default.
            result['pixel_values'] = transform(base,protocol,'identity')
            return result
        def __getattr__(self,name):
            return getattr(original_processor,name)

    official.VideoReader = Reader
    evaluation._image_processor = Processor()
    try:
        yield observed
    finally:
        official.VideoReader = original_reader
        evaluation._image_processor = original_processor


@torch.inference_mode()
def official_wrapper(evaluation,video_id,question,frames,indices,timestamps,protocol,budget,chunk=64):
    from lmms_eval.api.instance import Instance
    from lmms_eval.models.simple.llava_vid import LlavaVid
    from lmms_eval.api.registry import MODEL_REGISTRY
    if MODEL_REGISTRY['llava_vid'] is not LlavaVid:
        raise ValueError('official evaluator registry was replaced')
    model = evaluation.model
    old_config = model.flashvid_config
    had_task_dict='task_dict' in evaluation.__dict__
    old_task_dict=evaluation.__dict__.get('task_dict')
    model.flashvid_config = copy.deepcopy(old_config)
    model.flashvid_config.retention_ratio = int(budget[1:])/100
    evaluation.task_dict = {'v4_i6':{'test':{0:question['doc']}}}
    path = str(PROJECT/'data/videos_short'/f'{video_id}.mp4')
    request = Instance(request_type='generate_until',arguments=(question['context'],
                       {'max_new_tokens':16,'temperature':0,'top_p':1.0,'num_beams':1},
                       lambda doc:[path],0,'v4_i6','test'),idx=0,
                       metadata={'task':'v4_i6','doc_id':0,'repeats':1})
    stopped = False
    try:
        with observe_preprocessing(evaluation,protocol,frames,indices,timestamps) as pre:
            with VisionCapture(model,chunk) as vision, CompressionCapture(stop_after=True) as comp:
                try:
                    # The exact registered official generation entry enters the actual wrapper.
                    LlavaVid.generate_until(evaluation,[request])
                except CaptureComplete:
                    stopped = True
                if not stopped:
                    raise ValueError('official path did not stop at registered capture boundary')
                tensors = vision.tensors()
                for name in ('pooled169','attn169'):
                    if not torch.equal(tensors[name],comp.snapshots[name]):
                        raise ObservationMismatch('actual wrapper pooling differs from independent observed tensor',actual_pool=comp.snapshots[name],reference_pool=tensors[name])
                lengths,selected = comp.selected()
            if pre['processor_calls']!=1 or len(pre['sampling'])!=1:
                raise ValueError('official preprocessor/sampling call coverage mismatch')
        return {'tensors':tensors,'lengths':lengths,'selected':selected,'trace':vision.trace,
                'compression_trace':comp.calls,'registry':comp.registry_identities,
                'snapshot_hashes':comp.snapshot_hashes,'entry':identity(LlavaVid.generate_until),
                'sampling':pre['sampling'],'stopped_before_llm':True,'prompt_identity':prompt_contract(),
                'question_identity':{'row_number':question['row_number'],
                                     'context_sha256':hashlib.sha256(question['context'].encode()).hexdigest()}}
    finally:
        model.flashvid_config = old_config
        if had_task_dict:evaluation.task_dict=old_task_dict
        else:evaluation.__dict__.pop('task_dict',None)


def compare_captures(actual,reference):
    for capture in (actual,reference):
        if set(capture.get('tensors',{}))!=set(CAPTURE_AXES):
            raise ValueError('capture points differ from all six registered observations')
    if {k for k in ('lengths','selected') if k in actual}!={k for k in ('lengths','selected') if k in reference}:
        raise ValueError('discrete capture presence differs between paths')
    numeric = {k:compare(actual['tensors'][k],reference['tensors'][k],k) for k in reference['tensors']}
    discrete = {k:actual[k].dtype==reference[k].dtype and
                torch.equal(actual[k].detach().cpu(),reference[k].detach().cpu())
                for k in ('lengths','selected') if k in reference}
    return {'passed':all(x['passed'] for x in numeric.values()) and all(discrete.values()),
            'numeric':numeric,'discrete':discrete}
