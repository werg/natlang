import hashlib
import uuid
import pytest
import torch
from natlang_neuralese.train.diloco.coordinator import Coordinator, Conflict, initialize_coordinator
from natlang_neuralese.train.diloco.protocol import save_tensors


def test_same_queued_sequence_requires_exact_delta_and_metadata(tmp_path):
    root=tmp_path/'coord'
    initialize_coordinator(root,run_id='exact-queued',masters={'weight':torch.zeros(8)},identity={'parent':'exact'},
                           workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,outer_lr=1.,outer_momentum=0.)
    coordinator=Coordinator(root)
    sessions=[str(uuid.uuid4()),str(uuid.uuid4())]
    for worker,session in enumerate(sessions):coordinator.lease(worker,session)
    source=tmp_path/'delta.safetensors';delta=torch.arange(coordinator.fragments[0].length,dtype=torch.float32)
    save_tensors(source,{'delta':delta})
    def submit(worker=0,*,tokens=8):
        return coordinator.submit(0,worker,sessions[worker],1,0,2,tokens,source)
    assert submit()['status']=='queued'
    target=root/'pending'/'00000'/'000.safetensors'
    original=hashlib.sha256(target.read_bytes()).hexdigest()
    assert submit()['status']=='already_queued'
    with pytest.raises(Conflict,match='different metadata or delta'):submit(tokens=9)
    save_tensors(source,{'delta':delta+1})
    with pytest.raises(Conflict,match='different metadata or delta'):submit()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==original
    save_tensors(source,{'delta':delta})
    assert submit(worker=1)['status']=='queued'
    assert coordinator.status()['versions']==[1]
