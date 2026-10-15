"""Precision contexts survive native activation replay after outer scope exit."""
import torch

from natlang_neuralese.maple.model import DENSE_MOE
from natlang_neuralese.train.fp8_backend import FP8ExpertAdapter
from natlang_neuralese.model.shared_state import SharedStateModel, Invocation, compile_graph
from test_shared_state import tiny


def test_expert_recompute_keeps_forward_precision_after_autocast_exits():
    backend = object.__new__(FP8ExpertAdapter)
    x = torch.tensor([.3, .8], requires_grad=True)
    modes = []
    kernels = []
    previous = DENSE_MOE.get('fp8')
    previous_kernel = DENSE_MOE['kernel']
    DENSE_MOE['kernel'] = 'invariant'
    def forward(value):
        modes.append(DENSE_MOE.get('fp8'))
        kernels.append(DENSE_MOE['kernel'])
        return (value.sin() if DENSE_MOE.get('fp8') == 'grouped' else value.cos()).square()
    try:
        with backend.autocast():
            result = backend.checkpoint(forward, x, use_reentrant=False, preserve_rng_state=True)
        result.sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach().sin() * x.detach().cos(), rtol=0, atol=0)
        assert modes == ['grouped', 'grouped']
        assert kernels == ['grouped', 'grouped']
        assert DENSE_MOE.get('fp8') == previous
        assert DENSE_MOE['kernel'] == 'invariant'
        assert backend.has_executed is False
        modes.clear()
        x.grad = None
        with backend.native_modules():
            result = backend.checkpoint(forward, x, use_reentrant=False)
        result.sum().backward()
        torch.testing.assert_close(x.grad, -2 * x.detach().sin() * x.detach().cos(), rtol=0, atol=0)
        assert modes == [None, None]
    finally:
        DENSE_MOE['fp8'] = previous
        DENSE_MOE['kernel'] = previous_kernel


def test_expert_scope_does_not_claim_execution_and_restores_on_exception():
    backend = object.__new__(FP8ExpertAdapter)
    original = dict(DENSE_MOE)
    try:
        with backend.autocast():
            assert DENSE_MOE['kernel'] == 'grouped'
            assert DENSE_MOE['fp8_stats'] is backend.row_counts
            with backend.native_modules():
                assert DENSE_MOE['fp8'] is None
                assert DENSE_MOE['fp8_stats'] is None
            raise RuntimeError('diagnostic')
    except RuntimeError as error:
        assert str(error) == 'diagnostic'
    assert DENSE_MOE == original
    assert backend.collect_row_counts()['calls'] == 0
    assert backend.has_executed is False


def test_native_graph_calls_the_backends_checkpoint_implementation():
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    model.port.checkpoint_layers = True
    calls = []
    def checkpoint(function, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint as actual
        calls.append(1)
        return actual(function, *args, **kwargs)
    model.port.activation_checkpoint = checkpoint
    result = model(compile_graph([Invocation('one', (1, 2, 3), producing=((1, 3),))]))
    result['h_final'].square().sum().backward()
    assert len(calls) == model.port.num_layers
    assert model.adapter.up.weight.grad is not None
