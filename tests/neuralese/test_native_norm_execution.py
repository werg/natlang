"""Declared norm dispatch preserves legacy arithmetic independently of storage."""
import pytest
import torch

from natlang_neuralese.model.native_parameter_precision import configure_norm_execution, validate_norm_execution
from natlang_neuralese.model.shared_state import SharedStateModel, channel_checkpoint, restore_channel_checkpoint


@pytest.mark.parametrize('family', ['lfm'])
def test_execution_is_explicit_and_preserves_parameters_rng(family):
    from test_shared_state import tiny
    model = SharedStateModel(tiny(family), (60,61,62,63), rank=4)
    assert model.native_norm_execution == model.port.native_norm_execution == 'source-v1'
    assert 'native_norm_execution' not in channel_checkpoint(model)
    parameters = {name:(id(p),p.detach().clone()) for name,p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    configure_norm_execution(model, 'stable-v1')
    validate_norm_execution(model, {'native_norm_execution':'stable-v1'})
    assert torch.equal(rng, torch.get_rng_state())
    assert all(id(p)==parameters[name][0] and torch.equal(p,parameters[name][1]) for name,p in model.named_parameters())
    saved = channel_checkpoint(model)
    configure_norm_execution(model, None)
    with pytest.raises(ValueError, match='native_norm_execution'):
        restore_channel_checkpoint(model, saved)
    with pytest.raises(ValueError, match='declared profile'):
        validate_norm_execution(model, {'native_norm_execution':'stable-v1'})
    with pytest.raises(ValueError): configure_norm_execution(model, 'invented')


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('profile',['source-v1','stable-v1'])
def test_profiles_graph_cached_execution_and_adjoints(family,profile):
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import Invocation,compile_graph,native_step
    port=tiny(family)
    model=SharedStateModel(port,tuple(range(port.embedding_weight.shape[0]-4,port.embedding_weight.shape[0])),
        rank=4,native_norm_execution=profile)
    torch.nn.init.normal_(model.adapter.up.weight,std=.03)
    plan=compile_graph([Invocation('ordinary',(2,3,4,5,6))])
    flat=model(plan)['h_final']
    prefix=native_step(model,torch.tensor([[2,3]]))
    suffix=native_step(model,torch.tensor([[4,5,6]]),prefix['cache'])
    cached=torch.cat((prefix['h_final'],suffix['h_final']),dim=1)
    torch.testing.assert_close(flat,cached,atol=3e-6,rtol=3e-5)
    weights=torch.randn_like(flat)
    parameters=(model.input_rows,model.adapter.up.weight)
    a=torch.autograd.grad((flat*weights).sum(),parameters)
    b=torch.autograd.grad((cached*weights).sum(),parameters)
    for actual,expected in zip(a,b):torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)
