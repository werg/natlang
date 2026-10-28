import json
import shutil
import tarfile
import uuid
import pytest
import torch
from natlang_neuralese.train.diloco.coordinator import Coordinator, initialize_coordinator, _metadata
from natlang_neuralese.train.diloco.protocol import load_tensors, save_tensors
from test_diloco_recovery import load


@pytest.mark.parametrize('tree_only',[False,True])
def test_compact_capture_rebuilds_public_and_preserves_pending_and_outer_state(tmp_path,tree_only):
    pin=load('diloco_recovery_pin');root=tmp_path/'live';package=tmp_path/'package';package.mkdir()
    (package/'__init__.py').write_text('')
    initialize_coordinator(root,run_id='compact',masters={'weight':torch.arange(8,dtype=torch.float32)},
        identity={'parent':'actual'},workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,
        outer_lr=1.,outer_momentum=.5,ema={'decay':.88,'warmup':0})
    coordinator=Coordinator(root);sessions=[str(uuid.uuid4()),str(uuid.uuid4())]
    for i,session in enumerate(sessions):coordinator.lease(i,session)
    update=tmp_path/'delta.safetensors';save_tensors(update,{'delta':torch.ones(8)})
    for worker in (0,1):coordinator.submit(0,worker,sessions[worker],1,0,1,3,update)
    assert coordinator.status()['versions']==[1]
    save_tensors(update,{'delta':torch.full((8,),2.)})
    coordinator.submit(0,0,sessions[0],2,1,1,3,update)
    before=load_tensors(root/'fragments/00000.safetensors');before_meta=_metadata(root/'fragments/00000.safetensors')
    pending=load_tensors(root/'pending/00000/000.safetensors')['delta'].clone()
    live_public=(root/'public/00000.safetensors').read_bytes()
    target=tmp_path/('tree' if tree_only else 'coordinator.tar')
    captured=pin.coordinator_pin(root,target,package,tree_only=tree_only)
    assert captured['versions']==[1] and captured['last_seq'][0]=={'0':1,'1':1}
    assert captured['queued'][0]['0']['sequence']==2
    assert (root/'public/00000.safetensors').read_bytes()==live_public
    if tree_only:
        assert not (target/'public').exists()
        identity=tmp_path/'identity.json';identity.write_text(json.dumps(captured['file_identity']))
        archive=tmp_path/'tree.tar';pin.archive_coordinator_tree(target,archive,identity)
    else:archive=target
    restored=tmp_path/'restored'
    with tarfile.open(archive) as tar:
        assert not any(name.startswith('diloco-coordinator/public/') for name in tar.getnames())
        tar.extractall(restored,filter='data')
    restored=restored/'diloco-coordinator';assert not (restored/'public').exists()
    recovered=Coordinator(restored)
    for key,value in before.items():assert torch.equal(value,load_tensors(restored/'fragments/00000.safetensors')[key])
    assert _metadata(restored/'fragments/00000.safetensors')==before_meta
    assert torch.equal(load_tensors(restored/'public/00000.safetensors')['theta'],before['theta'])
    assert _metadata(restored/'public/00000.safetensors')['version']=='1'
    assert torch.equal(load_tensors(restored/'pending/00000/000.safetensors')['delta'],pending)
    assert recovered.status()['versions']==[1]
    recovered.submit(0,1,sessions[1],2,1,1,3,update)
    assert recovered.status()['versions']==[2]
    assert json.loads(_metadata(restored/'fragments/00000.safetensors')['last_seq'])=={'0':2,'1':2}


def test_already_pinned_tree_with_public_cache_remains_archivable(tmp_path):
    pin=load('diloco_recovery_pin');root=tmp_path/'live'
    initialize_coordinator(root,run_id='old-capture',masters={'weight':torch.ones(8)},identity={},
                           workers=1,min_quorum=1,fragment_bytes=1024)
    Coordinator(root)
    tree=tmp_path/'old-tree';shutil.copytree(root,tree)
    identity=tmp_path/'identity.json';identity.write_text(json.dumps(pin._inventory(tree,include_public=True)))
    archive=tmp_path/'old.tar';pin.archive_coordinator_tree(tree,archive,identity)
    with tarfile.open(archive) as tar:assert 'diloco-coordinator/public/00000.safetensors' in tar.getnames()
