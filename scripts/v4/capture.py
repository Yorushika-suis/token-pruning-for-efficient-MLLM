"""Observe fixed original FlashVID functions; never substitute a selector oracle."""
import contextlib
import functools
import hashlib
import inspect
import sys
import types

import torch

from .selection import original_registry


class CaptureComplete(BaseException):
    """Intentional stop after compression, before any LLM decoder computation."""


def tensor_hash(t):
    x = t.detach().contiguous().cpu()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def identity(fn):
    obj = getattr(fn,'__func__',fn)
    filename = inspect.getsourcefile(obj)
    with open(filename,'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    return {'module':obj.__module__,'qualname':obj.__qualname__,'source_file':filename,
            'source_file_sha256':h,'first_line':inspect.getsourcelines(obj)[1]}


class VisionCapture:
    def __init__(self, model, chunk=64):
        if chunk not in (16,64):
            raise ValueError('registered encoder batch shapes are 64 and 16x4')
        self.model,self.chunk = model,chunk
        self.buffers = {k:[] for k in ('pixels','sig729','attn729','projected729')}
        self.trace = []
        self.handles = []

    def _tower_pre(self,module,args):
        self.buffers['pixels'].append(args[0].detach().clone())

    def _tower_post(self,module,args,out):
        if not isinstance(out,(tuple,list)) or len(out)!=2 or out[0].shape[:2]!=(len(args[0]),729) or out[1].shape!=(len(args[0]),729):
            raise ValueError('official vision tower return differs from the two-output capture contract')
        self.buffers['sig729'].append(out[0].detach().clone())
        self.buffers['attn729'].append(out[1].detach().clone())

    def _projector_post(self,module,args,out):
        self.buffers['projected729'].append(out.detach().clone())

    def __enter__(self):
        tower = self.model.get_vision_tower()
        projector = self.model.get_model().mm_projector
        self.handles = [tower.register_forward_pre_hook(self._tower_pre),tower.register_forward_hook(self._tower_post),
                        projector.register_forward_hook(self._projector_post)]
        self.old_encode = self.model.encode_images
        self.encode_was_instance = 'encode_images' in self.model.__dict__
        self.old_instance_encode = self.model.__dict__.get('encode_images')
        orig = self.old_encode

        def checked(images):
            result=orig(images)
            if not isinstance(result,(tuple,list)) or len(result)!=2 or result[0].shape[:2]!=(len(images),729) or result[1].shape!=(len(images),729):
                raise ValueError('original encode_images return arity/shape differs from capture contract')
            return result

        def encode(instance, images):
            self.trace.append({'call':'encode_images','original':identity(orig),'input_shape':list(images.shape),'chunk':self.chunk})
            # Same fixed original method; concatenate before all 64-frame DySeg work.
            if self.chunk == 64:
                return checked(images)
            parts = [checked(images[i:i+16]) for i in range(0,len(images),16)]
            return tuple(torch.cat([x[j] for x in parts],0) for j in (0,1))

        self.model.encode_images = types.MethodType(encode,self.model)
        return self

    def tensors(self):
        out = {k:torch.cat(v,0) for k,v in self.buffers.items()}
        out['pooled169'] = self.model.get_2dPool(out['projected729'])
        out['attn169'] = self.model.get_2dPool(out['attn729'].unsqueeze(-1)).squeeze(-1)
        return out

    def __exit__(self,*exc):
        if self.encode_was_instance:
            self.model.encode_images = self.old_instance_encode
        else:
            del self.model.encode_images
        for h in self.handles:
            h.remove()


class CompressionCapture:
    def __init__(self, stop_after=False):
        self.stop_after = stop_after
        self.selections = []
        self.segment_lengths = []
        self.calls = []
        self.snapshots = {}

    def __enter__(self):
        import flashvid.llava_arch as arch
        self.utils,originals = original_registry()
        self.registry = dict(self.utils.ALL_TOKEN_SELECTION_METHOD)
        self.orig_comp,self.orig_segment = arch.flashvid_compression,self.utils.segment
        if self.orig_comp is not self.utils.flashvid_compression:
            raise ValueError('official compression binding was replaced')
        self.registry_identities = {str(k):identity(v) for k,v in self.registry.items()}
        for key,fn in self.registry.items():
            self.utils.ALL_TOKEN_SELECTION_METHOD[key] = self._observe_selector(fn)
        self.utils.segment = self._observe_segment
        arch.flashvid_compression = self._observe_compression
        self.arch = arch
        return self

    def _observe_selector(self, original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            result = original(*args,**kwargs)
            keep = result[1].detach().clone()
            self.selections.append(keep)
            self.calls.append({'call':'original_selector','original':identity(original),'shape':list(keep.shape)})
            return result
        return wrapped

    def _observe_segment(self,*args,**kwargs):
        result = self.orig_segment(*args,**kwargs)
        self.segment_lengths.append(result.detach().clone())
        self.calls.append({'call':'original_segment','original':identity(self.orig_segment)})
        return result

    def _observe_compression(self,video_features,cls_attention,flashvid_config):
        self.snapshots = {'pooled169':video_features.detach().clone(),'attn169':cls_attention.detach().clone()}
        before = {k:tensor_hash(v) for k,v in self.snapshots.items()}
        self.calls.append({'call':'original_compression','original':identity(self.orig_comp),
                           'stack':[{'file':x.filename,'function':x.function,'line':x.lineno}
                                    for x in inspect.stack() if '/FlashVID/' in x.filename]})
        # Original TSTM mutates its work tensor; neither the source nor capture aliases it.
        result = self.orig_comp(video_features=video_features.clone(),cls_attention=cls_attention.clone(),flashvid_config=flashvid_config)
        after = {k:tensor_hash(v) for k,v in self.snapshots.items()}
        if before != after or tensor_hash(video_features) != before['pooled169'] or tensor_hash(cls_attention) != before['attn169']:
            raise ValueError('compression mutated immutable input/capture snapshot')
        self.snapshot_hashes = before
        if self.stop_after:
            raise CaptureComplete()
        return result

    def direct(self,pooled,attention,config):
        return self._observe_compression(pooled,attention,config)

    def selected(self):
        if len(self.segment_lengths) != 1 or not self.selections:
            raise ValueError('capture did not observe exactly one full video compression')
        out = torch.cat(self.selections,0)
        if out.shape[0] != 64:
            raise ValueError('capture omitted frames')
        return self.segment_lengths[0],out

    def __exit__(self,*exc):
        self.utils.ALL_TOKEN_SELECTION_METHOD.clear()
        self.utils.ALL_TOKEN_SELECTION_METHOD.update(self.registry)
        self.utils.segment = self.orig_segment
        self.arch.flashvid_compression = self.orig_comp
        original_registry()
