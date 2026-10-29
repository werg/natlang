import os
import pytest
import torch
from safetensors.torch import load_file
from natlang_neuralese.train.diloco.protocol import MappedInputEvictor, save_tensors


def test_checkpoint_eviction_preserves_output_and_ignores_anonymous(tmp_path):
    path=tmp_path/'source.pt'; original=torch.arange(2_500_000,dtype=torch.float32)
    torch.save({'x': original},path)
    mapped=torch.load(path,mmap=True,weights_only=True)['x']
    evict=MappedInputEvictor(path)
    anonymous=torch.arange(1_000_000,dtype=torch.float32); before=anonymous.clone()
    evict.release(anonymous)
    assert torch.equal(anonymous,before)
    dest=tmp_path/'out.safetensors';save_tensors(dest,{'x':mapped},dtype=torch.bfloat16,input_evictor=evict)
    assert torch.equal(load_file(str(dest))['x'],original.to(torch.bfloat16))
    assert torch.equal(mapped,original)


def test_reject_dirty_checkpoint_mapping(tmp_path):
    path=tmp_path/'source.pt';torch.save({'x':torch.zeros(10000)},path)
    mapped=torch.load(path,mmap=True,weights_only=True)['x'];mapped[0]=1
    with pytest.raises(ValueError,match='private dirty'):
        MappedInputEvictor(path)
