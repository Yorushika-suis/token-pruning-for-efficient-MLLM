"""Original encode/pool observation and budget-order counterexamples on CPU."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from v4 import spec
from v4.capture import CompressionCapture, VisionCapture, tensor_hash


def test_undefined_duplicate_cut_and_zero_norm_stop():
    from v4 import selection as L
    import flashvid.utils as official
    scores=torch.ones(63)
    selected=int(torch.topk(scores,k=6,largest=False).indices[0])
    scores[selected]=.8
    candidate=L.additional_from_similarities(scores)
    reference=official.additional_segment(torch.where(scores<.9)[0],64,8,scores.clone(),.9,True)
    assert torch.equal(candidate,reference) and (candidate==0).any()
    with pytest.raises(ValueError,match='zero frame mean'):
        L.candidate_segments(torch.zeros((64,169,8)))


def test_official_compression_three_repeats_and_reversed_budgets():
    from flashvid.configuration_flashvid import FlashVidConfig
    rng=torch.Generator().manual_seed(9127)
    pooled=torch.randn((64,169,8),generator=rng);attention=torch.rand((64,169),generator=rng)
    hashes=(tensor_hash(pooled),tensor_hash(attention));references={}
    template=FlashVidConfig(retention_ratio=.1,do_segment=True,segment_threshold=.9,min_segment_num=8,
                           complementary_segment=True,alpha=.7,temporal_threshold=.8,expansion=1.25,
                           token_selection_method='attn_div_v2')
    original_config=copy.deepcopy(vars(template));orders=[]
    for repeat in range(3):
        budgets=list(spec.BUDGETS)
        if repeat==1:budgets.reverse()
        orders.append(budgets)
        for budget in budgets:
            config=copy.deepcopy(template);config.retention_ratio=int(budget[1:])/100
            with CompressionCapture() as observed:
                observed.direct(pooled.clone(),attention.clone(),config)
                lengths,selected=observed.selected()
            current=(lengths.clone(),selected.clone())
            if budget not in references:references[budget]=current
            assert all(torch.equal(a,b) for a,b in zip(current,references[budget]))
            assert hashes==(tensor_hash(pooled),tensor_hash(attention))
    assert orders[1]==orders[0][::-1] and len(references)==4
    assert vars(template)==original_config


def model_fixture():
    from flashvid.llava_arch import LlavaMetaForCausalLM_encode_images
    from llava.model.llava_arch import LlavaMetaForCausalLM
    class Tower(torch.nn.Module):
        num_patches_per_side=27
        def forward(self,images):
            frame=images[:,0,0,0,None]
            patch=torch.arange(729,dtype=images.dtype)[None,:]
            signal=frame*1000+patch
            return signal[:,:,None].repeat(1,1,3),signal
    class Model:
        encode_images=LlavaMetaForCausalLM_encode_images
        get_2dPool=LlavaMetaForCausalLM.get_2dPool
        def __init__(self):
            self.tower=Tower();self.mm_projector=torch.nn.Identity()
            self.config=SimpleNamespace(mm_spatial_pool_mode='average')
        def get_model(self):return self
        def get_vision_tower(self):return self.tower
    return Model()


def test_vision_original_encode_pool_chunks_hooks_and_restoration():
    model=model_fixture();original=model.encode_images.__func__
    pixels=torch.arange(64,dtype=torch.float32)[:,None,None,None].expand(-1,3,384,384).clone()
    references=None
    for chunk in (64,16):
        for repeat in range(3):
            with VisionCapture(model,chunk) as observed:
                model.encode_images(pixels)
                captured=observed.tensors()
            assert model.encode_images.__func__ is original and 'encode_images' not in vars(model)
            assert not model.tower._forward_hooks and not model.tower._forward_pre_hooks
            assert not model.mm_projector._forward_hooks
            assert captured['pooled169'].shape==(64,169,3) and captured['attn169'].shape==(64,169)
            torch.testing.assert_close(captured['pixels'],pixels,rtol=0,atol=0)
            expected_patch=(torch.arange(13)[:,None]*54+torch.arange(13)[None,:]*2+14).reshape(-1)
            expected=torch.arange(64)[:,None]*1000+expected_patch
            torch.testing.assert_close(captured['attn169'],expected.float(),rtol=0,atol=0)
            if references is None:references={k:v.clone() for k,v in captured.items()}
            assert all(torch.equal(v,references[k]) for k,v in captured.items())


def test_capture_rejects_extra_original_return_and_restores():
    import types
    model=model_fixture();original=model.encode_images
    def three(instance,images):return (*original(images),torch.zeros(1))
    model.encode_images=types.MethodType(three,model);saved=model.encode_images
    with pytest.raises(ValueError,match='arity'):
        with VisionCapture(model,16):model.encode_images(torch.zeros((64,3,384,384)))
    assert model.encode_images is saved
    assert not model.tower._forward_hooks and not model.mm_projector._forward_hooks
