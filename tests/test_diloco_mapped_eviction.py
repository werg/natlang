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


def test_reject_wrong_file_identity(tmp_path):
    first=tmp_path/'first.pt';other=tmp_path/'other.pt'
    torch.save({'x':torch.ones(10000)},first);torch.save({'x':torch.zeros(10000)},other)
    mapped=torch.load(first,mmap=True,weights_only=True)['x']
    with pytest.raises(ValueError,match='no matching file-backed mapping'):
        MappedInputEvictor(other)
    assert mapped[0].item()==1


def test_native_coordinator_init_releases_actual_mapped_source(tmp_path, monkeypatch):
    import argparse
    import json
    from natlang_neuralese.train import native_diloco
    from natlang_neuralese.train.diloco import protocol
    path=tmp_path/'parent.pt'
    original=torch.arange(10000,dtype=torch.bfloat16)
    torch.save({'model_config':{'architecture':'lfm3'},'weight':original},path)
    mapped=torch.load(path,mmap=True,weights_only=True)['weight']
    monkeypatch.setattr(native_diloco,'_coordinator_surface',lambda *_:{'weight':mapped})
    monkeypatch.setattr(native_diloco,'identity',lambda *_:{'fixture':'mapped-seed'})
    released=[]
    real_release=protocol.MappedInputEvictor.release
    def release(self,tensor):
        released.append(tensor.numel())
        return real_release(self,tensor)
    monkeypatch.setattr(protocol.MappedInputEvictor,'release',release)
    root=tmp_path/'coordinator'
    native_diloco.init(argparse.Namespace(settings=json.dumps({'run_id':'mapped','workers':1,'local_steps':32}),
        parent=path,base=None,role_contract=None,trainer_argv='[]',root=root,quorum=1,fragment_mib=1,
        outer_lr=1.,outer_momentum=0.,grace_seconds=30.,max_staleness=2,ema='off',outer_schedule=None))
    assert released and sum(released)>=original.numel()*2
    assert torch.equal(load_file(str(root/'seed.safetensors'))['weight'],original.float())
    assert torch.equal(mapped,original)
