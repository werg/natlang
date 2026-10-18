"""Real midpoint regression; graph-forward integration is checked separately."""
import pytest
import torch
from natlang_neuralese.model.native_gradient_partitions import (
    GradientPartitions, active_gradient_partitions, gradient_partitions,
    capture_gradient_partitions, token_slices,
)
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope, stored_linear


def test_partition_context_captures_checkpoint_and_restores_nested_context():
    parts = GradientPartitions((20, 0, 14))
    with gradient_partitions(parts):
        captured = capture_gradient_partitions(active_gradient_partitions)
    assert active_gradient_partitions() is None
    with gradient_partitions(GradientPartitions((1,))):
        assert captured() is parts
        assert active_gradient_partitions().lengths == (1,)
    assert [(s.start, s.stop) for s in token_slices(parts, 34)] == [(20, 34), (0, 20)]
    with pytest.raises(RuntimeError, match='provenance'):
        token_slices(parts, 33)


def test_actual_seed67_bf16_midpoint_matches_cached_delivered_gradient():
    torch.manual_seed(67)
    layer = torch.nn.Linear(96, 64, bias=False).to(torch.bfloat16)
    x = torch.randn(34, 96, dtype=torch.bfloat16, requires_grad=True)
    dy = (torch.randn(34, 64) * .01).to(torch.bfloat16)

    def run(*, split=False, partitions=None):
        with ParameterCotangentScope([('weight', layer.weight)], mode='vjp') as scope:
            with gradient_partitions(partitions):
                call = lambda z: stored_linear(z, layer.weight, storage_dtype=layer.weight.dtype)
                y = torch.cat([call(x[:20]), call(x[20:])]) if split else call(x)
            dx, dw = torch.autograd.grad((y * dy).sum(), (x, layer.weight))
            fp32 = scope.entries[id(layer.weight)]['accumulator'].clone()
            count = scope.entries[id(layer.weight)]['count']
            scope.finish()
        return y.detach(), dx, dw, fp32, count

    ordinary = run()
    cached = run(split=True)
    fixed = run(partitions=GradientPartitions((20, 14)))
    # This fixture must actually exercise the original delivered-gradient bug.
    assert not torch.equal(ordinary[2], cached[2])
    assert torch.equal(fixed[0], cached[0])
    assert torch.equal(fixed[1], cached[1])
    assert torch.equal(fixed[2], cached[2])
    assert torch.equal(fixed[3], cached[3])
    assert fixed[4] == cached[4] == 2
    assert layer.weight.grad is None


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('ffn_chunk', [0, 2])
def test_branched_graph_complete_bf16_derivatives_against_cached_execution(
        family, device, checkpointed, ffn_chunk):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from test_shared_state import tiny, graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_gradients import compare_complete_gradients
    import torch.nn.functional as F

    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4).to(device)
    model.port.hf.to(dtype=torch.bfloat16)
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'
    for parameter in model.port.parameters():
        parameter.requires_grad_(True)
    model.port.checkpoint_layers = checkpointed
    model.port.ffn_chunk_tokens = ffn_chunk
    with torch.no_grad():
        model.adapter.up.weight.normal_(std=.03)
    _, named = make_native_optimizer(model, lr=3e-6, in_backward=False)
    plan = graph()
    ids = plan.tensor([t for node in plan.invocations for t in node.token_ids], device)
    labels = ids.roll(-1)
    initial = model.embed(ids[None]).detach()
    outputs = []
    def make(reference):
        x = initial.clone().requires_grad_()
        scope = ParameterCotangentScope.for_model(model, mode='vjp')
        scope.__enter__()
        try:
            out = model(plan, embeddings=x, reference=reference)
            outputs.append(out['h_final'].detach())
            loss = F.cross_entropy(model.logits(out['h_final'])[0].float(), labels)
        except BaseException:
            scope.close()
            raise
        return loss, x, scope
    flat, x, scope = make(False)
    report = compare_complete_gradients(flat, None, named, x, None,
        cached_factory=lambda: make(True), flat_cotangent_scope=scope,
        atol=1e-5, rtol=1e-4)
    failures = {name: row for name, row in report['comparisons'].items() if not row['passed']}
    assert report['passed'], failures
    torch.testing.assert_close(outputs[0], outputs[1], atol=1e-4, rtol=1e-4)
    assert all(parameter.grad is None for _, parameter in named)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_graph_still_visits_each_physical_row_once_per_layer(monkeypatch, family):
    from test_shared_state import tiny, graph
    from natlang_neuralese.model import shared_state
    model = shared_state.SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    model.port.checkpoint_layers = False
    model.port.ffn_chunk_tokens = 2
    plan = graph()
    seen = []
    original = shared_state._actor_layer_inputs
    def observed(bank, groups, port, layer, h, attention):
        seen.append(h.shape[1])
        return original(bank, groups, port, layer, h, attention)
    def forbidden(*args, **kwargs):
        raise AssertionError('flat graph secretly called the cached graph oracle')
    monkeypatch.setattr(shared_state, '_actor_layer_inputs', observed)
    monkeypatch.setattr(shared_state, 'forward_reference', forbidden)
    out = model(plan)
    assert out['h_final'].shape[1] == plan.token_count
    assert seen == [plan.token_count] * model.port.num_layers


@pytest.mark.parametrize('device,backend', [('cpu','cpu_file'), ('cuda','device_fp32')])
def test_embedding_frequency_is_local_to_cached_invocation(device, backend):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_weight_carrier import stored_embedding
    weight = torch.nn.Parameter(torch.ones(9, 4, dtype=torch.bfloat16, device=device))
    ids = torch.tensor([[2, 2, 2, 2, 2]], device=device)
    def run(split):
        with ParameterCotangentScope([('weight',weight)], mode='vjp',
                accumulator_backend=backend) as scope:
            def call(selected):
                return stored_embedding(selected, weight, scale_grad_by_freq=True)
            if split:
                output = torch.cat([call(ids[:,:2]),call(ids[:,2:])],1)
            else:
                with gradient_partitions(GradientPartitions((2,3))):
                    output = call(ids)
            gradient, = torch.autograd.grad(output.sum(), (weight,))
            scope.finish()
        return gradient
    flat, cached = run(False), run(True)
    assert torch.equal(flat,cached)
    assert torch.equal(flat[2],torch.full_like(flat[2],2))



def test_first_order_checkpoint_context_crosses_worker_and_refuses_expired_scope():
    from concurrent.futures import ThreadPoolExecutor
    from natlang_neuralese.model.native_adapters import capture_native_checkpoint_execution
    from natlang_neuralese.model.native_weight_carrier import active_cotangent_scope
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    parts = GradientPartitions((1, 1))
    with ParameterCotangentScope([('weight',parameter)],mode='vjp') as scope:
        with gradient_partitions(parts):
            callback = capture_native_checkpoint_execution(
                lambda: (active_cotangent_scope(), active_gradient_partitions()))
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(active_cotangent_scope).result() is None
            assert pool.submit(callback).result() == (scope, parts)
            assert pool.submit(active_cotangent_scope).result() is None
        scope.finish()
    with pytest.raises(RuntimeError,match='outlived'):
        callback()
