"""Small synthetic CUDA/CPU observations: no model, media or scientific inference."""
import pytest
import torch
from v4 import runtime as R
from v4.numerics import CAPTURE_AXES


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA/CPU boundary needs a GPU')
def test_integer_capture_equality_across_devices_preserves_values_and_dtype():
    reference={'tensors':{k:torch.ones((1,1,2),dtype=torch.float32) for k in CAPTURE_AXES},
               'lengths':torch.tensor([64],dtype=torch.int64),
               'selected':torch.tensor([[2,2,7]],dtype=torch.int64)}
    actual={'tensors':{k:v.cuda() for k,v in reference['tensors'].items()},
            'lengths':reference['lengths'].cuda(),'selected':reference['selected'].cuda()}
    assert R.compare_captures(actual,reference)['passed']
    actual['selected'][0,2]=8
    r=R.compare_captures(actual,reference);assert not r['passed'] and r['discrete']['selected'] is False
    actual['selected']=reference['selected'].cuda().to(torch.int32)
    r=R.compare_captures(actual,reference);assert not r['passed'] and r['discrete']['selected'] is False
    assert reference['selected'].tolist()==[[2,2,7]] and reference['selected'].dtype==torch.int64
