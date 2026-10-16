"""Actual file-backed hints must preserve exact optimizer/RNG continuation."""
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train import checkpoint_readahead as mechanism
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


def assert_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_equal(a,b)
    else:
        assert left==right


@pytest.mark.skipif(not Path('/proc/self/smaps').exists(),reason='Linux mapping advice')
def test_actual_mapping_hint_matches_inode_and_does_not_prefetch_or_copy(tmp_path):
    path=tmp_path/'checkpoint with spaces.pt'
    torch.save({'tensor':torch.arange(1024,dtype=torch.float32)},path)
    state=torch.load(path,mmap=True,weights_only=True)
    alias=tmp_path/'same-inode.pt';alias.hardlink_to(path)
    regions=mechanism.mapped_regions(alias)
    assert regions and any(start<=state['tensor'].data_ptr()<start+length for start,length in regions)
    other=tmp_path/'other.pt';torch.save({'tensor':torch.zeros(1024)},other)
    assert mechanism.mapped_regions(other)==[]
    assert mechanism.advise_checkpoint(alias)==sum(length for _,length in regions)
    mapping=Path('/proc/self/smaps').read_text()
    for start,_ in regions:
        block=mapping[mapping.index(f'{start:x}-'):]
        flags=next(line for line in block.splitlines() if line.startswith('VmFlags:'))
        assert 'sr' in flags.split()  # kernel reports the actual sequential-read hint
    assert torch.equal(state['tensor'],torch.arange(1024,dtype=torch.float32))
    state['tensor'][0]=999
    assert torch.load(path,weights_only=True)['tensor'][0]==0  # normal private mmap remains private


def test_shared_loader_advice_preserves_next_adam_update_and_rng(tmp_path,monkeypatch):
    torch.manual_seed(8217)
    model=torch.nn.Linear(5,3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    model(torch.randn(4,5)).square().mean().backward();optimizer.step();optimizer.zero_grad()
    path=tmp_path/'checkpoint.pt'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},path)
    outputs=[]
    for setting in ('off','sequential'):
        monkeypatch.setenv(mechanism.SETTING,setting)
        before=torch.get_rng_state().clone()
        state=CheckpointPolicy.load(path)
        assert torch.equal(before,torch.get_rng_state())
        restored=torch.nn.Linear(5,3);restored.load_state_dict(state['model'])
        optim=torch.optim.AdamW(restored.parameters(),lr=.9);optim.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        loss=restored(torch.randn(4,5)).square().mean()
        loss.backward();optim.step()
        outputs.append({'model':{k:v.clone() for k,v in restored.state_dict().items()},
            'optimizer':optim.state_dict(),'rng':torch.get_rng_state().clone(),'loss':loss.detach().clone()})
    assert_equal(*outputs)


def test_disabled_or_unavailable_advice_keeps_original_loader(tmp_path,monkeypatch):
    path=tmp_path/'checkpoint.pt';torch.save({'tensor':torch.ones(2)},path)
    monkeypatch.setenv(mechanism.SETTING,'off')
    monkeypatch.setattr(mechanism,'advise_checkpoint',lambda path:pytest.fail('disabled hint called'))
    assert torch.equal(CheckpointPolicy.load(path)['tensor'],torch.ones(2))
    monkeypatch.setenv(mechanism.SETTING,'invalid')
    with pytest.raises(ValueError,match='must be off or sequential'):
        CheckpointPolicy.load(path)


def test_advisory_failure_is_not_a_checkpoint_failure(tmp_path,monkeypatch):
    path=tmp_path/'checkpoint.pt';torch.save({'tensor':torch.ones(2)},path)
    monkeypatch.setenv(mechanism.SETTING,'sequential')
    def unsupported(path):raise OSError('maps unavailable')
    monkeypatch.setattr(mechanism,'mapped_regions',unsupported)
    assert mechanism.advise_checkpoint(path)==0
    assert torch.equal(CheckpointPolicy.load(path)['tensor'],torch.ones(2))
