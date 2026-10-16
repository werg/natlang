"""Exact checkpoint input storage composes with both native architectures."""
import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from natlang_neuralese.common.checkpoint_offload import (
    CheckpointInputOffload, configure_checkpoint_input_offload,
)


def test_only_checkpoint_inputs_are_relocated_and_backend_context_survives(monkeypatch):
    from natlang_neuralese.train.fp8_backend import FP8ExpertAdapter
    from natlang_neuralese.maple.model import DENSE_MOE
    backend = object.__new__(FP8ExpertAdapter)
    native = torch.autograd.graph.save_on_cpu
    saved, modes = [], []
    def observed(**kwargs):
        manager = native(**kwargs)
        pack = manager.pack_hook
        def audited(tensor):
            packed = pack(tensor)
            saved.append((tuple(tensor.shape), packed[1].device.type))
            return packed
        manager.pack_hook = audited
        return manager
    monkeypatch.setattr(torch.autograd.graph, 'save_on_cpu', observed)
    x = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, 11, requires_grad=True)
    old = dict(DENSE_MOE)
    def function(value):
        modes.append(DENSE_MOE.get('fp8'))
        return (value @ weight).sin().square()
    with backend.autocast():
        result = CheckpointInputOffload(backend.checkpoint, pin_memory=False)(
            function, x, use_reentrant=False, preserve_rng_state=True)
    result.sum().backward()
    assert modes == ['grouped', 'grouped']
    assert DENSE_MOE == old
    assert saved and all(device == 'cpu' for _, device in saved)
    assert (tuple(x.shape), 'cpu') in saved
    assert tuple(weight.shape) not in [shape for shape, _ in saved]
    assert torch.isfinite(weight.grad).all() and weight.grad.norm() > 0


def test_wrapper_configuration_is_reversible_and_preserves_backend_identity():
    from types import SimpleNamespace
    backend = lambda *args, **kwargs: checkpoint(*args, **kwargs)
    port = SimpleNamespace(activation_checkpoint=backend)
    first = configure_checkpoint_input_offload(port, enabled=True)
    assert configure_checkpoint_input_offload(port, enabled=True) is first
    assert first.checkpoint is backend
    second = configure_checkpoint_input_offload(port, enabled=True, pin_memory=False)
    assert second is not first and second.checkpoint is backend
    assert configure_checkpoint_input_offload(port, enabled=False) is backend
    assert port.activation_checkpoint is backend
    with pytest.raises(ValueError, match='nonreentrant'):
        first(torch.sin, torch.ones(1), use_reentrant=True)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_graph_outputs_all_gradients_and_rng_are_unchanged(family, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.data.native_graph import parse_graph, graph_loss
    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4).to(device)
    with torch.no_grad():
        model.adapter.up.weight.normal_(0, .02)
    reference = copy.deepcopy(model)
    row = {'schema': 'natlang.neuralese-native-graph/1', 'id': 'checkpoint-offload',
        'split': 'train', 'source_groups': ['offload-test'], 'tokenizer_fingerprint': 'a'*64,
        'nodes': [{'name': 'producer', 'token_ids': [2, 3, 4, 5, 6],
                   'producing': [[1, 5]], 'target_weights': [0, .1, .1, 1, 1]},
                  {'name': 'consumer', 'token_ids': [7, 8, 9, 10],
                   'imports': [{'at': 1, 'source': 'producer', 'start': 1, 'stop': 5}],
                   'target_weights': [0, 0, 1, 1]}]}
    example = parse_graph(row, tokenizer_fingerprint='a'*64)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == 'cuda' else None
    first, _ = graph_loss(model, example, checkpoint_layers=True, chunk_tokens=2)
    first.backward()
    after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state() if device == 'cuda' else None
    torch.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    wrapper = configure_checkpoint_input_offload(reference.port, enabled=True)
    second, _ = graph_loss(reference, example, checkpoint_layers=True, chunk_tokens=2)
    second.backward()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for (name, left), (other, right) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)
    assert torch.equal(after, torch.get_rng_state())
    if cuda_after is not None:
        assert torch.equal(cuda_after, torch.cuda.get_rng_state())
    assert wrapper.calls == reference.port.num_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_checkpoint_input_storage_is_host_memory_for_cuda_inputs(monkeypatch):
    original = torch.autograd.graph.save_on_cpu
    observations = []
    def observed(**kwargs):
        manager = original(**kwargs)
        native_pack = manager.pack_hook
        def pack(tensor):
            result = native_pack(tensor)
            # Nonreentrant checkpoint also saves a zero-length CPU sentinel.
            if tensor.numel():
                observations.append((tensor.device.type, result[1].device.type, result[1].is_pinned()))
            return result
        manager.pack_hook = pack
        return manager
    monkeypatch.setattr(torch.autograd.graph, 'save_on_cpu', observed)
    x = torch.randn(128, 128, device='cuda', requires_grad=True)
    CheckpointInputOffload(checkpoint)(lambda value: value.sin().square(), x,
                                     use_reentrant=False).sum().backward()
    assert observations and all(source == 'cuda' and target == 'cpu' and pinned
                                for source, target, pinned in observations)
    torch.testing.assert_close(x.grad, 2*x.detach().sin()*x.detach().cos(), atol=0, rtol=0)
