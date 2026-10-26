"""CPU storage pages exact Muon history, without changing the update algorithm."""
import copy
import io
import pytest
import torch

from natlang_neuralese.train.optim import MuonSR
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
from natlang_neuralese.model.shared_state import SharedStateModel
from test_shared_state import tiny
from test_native_optimizer_checkpointing import examples
from natlang_neuralese.train.loop import capture_training_rng_state,restore_training_rng_state


def equal_values(left,right):
    if torch.is_tensor(left):
        assert left.dtype==right.dtype
        assert torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_values(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):equal_values(a,b)
    else:assert left==right


def test_state_storage_rejects_unknown_or_armed_transition():
    p=torch.nn.Parameter(torch.ones(4,4));optimizer=MuonSR([p],lr=1e-4)
    with pytest.raises(ValueError,match='device or cpu'):optimizer.set_state_storage('automatic')
    optimizer._armed=object()
    with pytest.raises(ValueError,match='idle update boundary'):optimizer.set_state_storage('cpu')


@pytest.mark.gpu_heavy
@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual GPU/CPU optimizer state transfers')
@pytest.mark.parametrize('family',['lfm','mellum'])
def test_native_whole_graph_paged_history_updates_rng_and_restart_are_exact(family):
    torch.manual_seed(8642)
    port=tiny(family).to(device='cuda',dtype=torch.bfloat16)
    if family=='mellum':port.hf.natlang_precision='bf16'
    model=SharedStateModel(port,(58,59,60,61),rank=4).to(device='cuda')
    reference=copy.deepcopy(model)
    optimizer,_=make_native_optimizer(model,lr=3e-6,latent_ns='cubic5')
    expected,_=make_native_optimizer(reference,lr=3e-6,latent_ns='cubic5')
    # Populate actual shared-graph history before moving it. Ordinary AdamW
    # interface moments and backbone Muon routing retain their original kinds.
    rng=capture_training_rng_state('cuda')
    native_training_update(reference,expected,examples()[:1],chunk_tokens=3)
    restore_training_rng_state(rng,'cuda')
    native_training_update(model,optimizer,examples()[:1],chunk_tokens=3)
    equal_values(optimizer.state_dict(),expected.state_dict())
    assert any(value['momentum'].is_cuda for value in optimizer.latent.state.values())
    state_before=copy.deepcopy(optimizer.state_dict())
    device_bytes=sum(value.numel()*value.element_size()
        for state in optimizer.latent.state.values()for value in state.values()
        if torch.is_tensor(value)and value.is_cuda)
    torch.cuda.synchronize();allocated_before=torch.cuda.memory_allocated()
    optimizer.latent.set_state_storage('cpu')
    torch.cuda.synchronize()
    assert allocated_before-torch.cuda.memory_allocated()>=device_bytes>0
    equal_values(state_before,optimizer.state_dict())
    assert all(not value['momentum'].is_cuda for value in optimizer.latent.state.values())
    assert all(p.is_cuda for group in optimizer.latent.param_groups for p in group['params'])
    assert optimizer.auxiliary is not None
    for index in range(2):
        rng=capture_training_rng_state('cuda')
        want,_=native_training_update(reference,expected,examples()[index:index+2],chunk_tokens=3)
        after=capture_training_rng_state('cuda')
        restore_training_rng_state(rng,'cuda')
        actual,_=native_training_update(model,optimizer,examples()[index:index+2],chunk_tokens=3)
        equal_values(actual,want);equal_values(after,capture_training_rng_state('cuda'))
        equal_values(model.state_dict(),reference.state_dict())
        equal_values(model.port.hf.state_dict(),reference.port.hf.state_dict())
        equal_values(optimizer.state_dict(),expected.state_dict())
        assert all(not value['momentum'].is_cuda for value in optimizer.latent.state.values())
        if index==0:
            buffer=io.BytesIO();torch.save(optimizer.state_dict(),buffer);buffer.seek(0)
            optimizer.load_state_dict(torch.load(buffer,weights_only=False))
            assert optimizer.latent.state_storage=='cpu'
            assert all(not value['momentum'].is_cuda for value in optimizer.latent.state.values())
            equal_values(optimizer.state_dict(),expected.state_dict())
