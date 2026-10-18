"""The optional CUDA accumulator shares the CPU-file cotangent contract."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device backend needs a CUDA device')
def test_device_accumulator_matches_cpu_file_across_microbatches_and_optimizer_state(tmp_path):
    torch.manual_seed(73)
    device = torch.device('cuda')
    initial = nn.Linear(11, 7, bias=True).to(device=device, dtype=torch.bfloat16).state_dict()
    models = [nn.Linear(11, 7, bias=True).to(device=device, dtype=torch.bfloat16) for _ in range(2)]
    for model in models:
        model.load_state_dict(initial)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.02) for model in models]
    x = [torch.randn(5, 11, device=device, dtype=torch.bfloat16),
         torch.randn(3, 11, device=device, dtype=torch.bfloat16)]
    cotangents = [torch.randn(5, 7, device=device, dtype=torch.bfloat16),
                  torch.randn(3, 7, device=device, dtype=torch.bfloat16)]
    rng = torch.cuda.get_rng_state().clone()
    reports = []
    gradients = []
    for model, optimizer, backend in zip(models, optimizers, ('cpu_file', 'device_fp32')):
        model.native_cotangent_accumulator_backend = backend
        optimizer.zero_grad(set_to_none=True)
        with ParameterCotangentScope.for_model(model, mode='update', directory=tmp_path) as scope:
            model(x[0]).backward(cotangents[0])
            scope.new_backward()
            model(x[1]).backward(cotangents[1])
            reports.append(scope.allocation_report())
            scope.finish()
        gradients.append({name: parameter.grad.detach().clone()
                          for name, parameter in model.named_parameters()})
        optimizer.step()
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    for name in gradients[0]:
        assert torch.equal(gradients[0][name], gradients[1][name]), name
    for left, right in zip(models[0].parameters(), models[1].parameters()):
        assert torch.equal(left, right)
    for left_parameter, right_parameter in zip(models[0].parameters(), models[1].parameters()):
        left_state = optimizers[0].state[left_parameter]
        right_state = optimizers[1].state[right_parameter]
        for key, left in left_state.items():
            right = right_state[key]
            assert torch.equal(left, right) if torch.is_tensor(left) else left == right
    assert reports[0]['backend'] == 'cpu_file'
    assert reports[1] == {'backend': 'device_fp32', 'allocated_accumulator_bytes': 4 * (7 * 11 + 7),
                          'scratch_peak_bytes': 0, 'allocated_entries': 2}
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize('scale_by_freq', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device backend needs a CUDA device')
def test_device_embedding_accumulator_matches_frequency_and_padding_contract(tmp_path, scale_by_freq):
    torch.manual_seed(91)
    device = torch.device('cuda')
    embedding = nn.Embedding(19, 6, padding_idx=0, scale_grad_by_freq=scale_by_freq).to(
        device=device, dtype=torch.bfloat16)
    reference = nn.Embedding(19, 6, padding_idx=0, scale_grad_by_freq=scale_by_freq).to(
        device=device, dtype=torch.bfloat16)
    cpu_file = nn.Embedding(19, 6, padding_idx=0, scale_grad_by_freq=scale_by_freq).to(
        device=device, dtype=torch.bfloat16)
    reference.load_state_dict(embedding.state_dict())
    cpu_file.load_state_dict(embedding.state_dict())
    ids = torch.tensor([[2, 2, 3, 0], [3, 4, 2, 4]], device=device)
    cotangent = torch.randn(*ids.shape, 6, device=device, dtype=torch.bfloat16)
    with ParameterCotangentScope.for_model(embedding, mode='update', directory=tmp_path,
                                           accumulator_backend='device_fp32') as scope:
        (embedding(ids) * cotangent).sum().backward()
        scope.finish()
    with ParameterCotangentScope.for_model(cpu_file, mode='update', directory=tmp_path,
                                           accumulator_backend='cpu_file') as scope:
        (cpu_file(ids) * cotangent).sum().backward()
        scope.finish()
    (reference(ids) * cotangent).sum().backward()
    # The reference uses PyTorch's stored-dtype embedding rule; the scoped path
    # must preserve its forward and input-independent row reductions exactly.
    assert torch.equal(embedding.weight.grad, reference.weight.grad)
    assert torch.equal(embedding.weight.grad, cpu_file.weight.grad)
    assert scope.allocation_report()['allocated_accumulator_bytes'] == 19 * 6 * 4


def test_device_backend_fails_closed_for_non_cuda_targets(tmp_path):
    parameter = nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match='requires CUDA parameter'):
        ParameterCotangentScope([('weight', parameter)], accumulator_backend='device_fp32',
                                directory=tmp_path)


def test_device_backend_recipe_contract_is_explicit_and_transition_needs_fresh_capacity(tmp_path):
    from natlang_neuralese.train.native_state_sft import build_parser
    from natlang_neuralese.train.recipe import (
        direct_windowing_handoffs, stage_parameter_args, validate_stage_parameters)

    parser = build_parser()
    assert parser.get_default('cotangent_accumulator_backend') is None
    assert stage_parameter_args({'cotangent_accumulator_backend': None}) == []
    assert stage_parameter_args({'cotangent_accumulator_backend': 'device_fp32'}) == [
        '--cotangent-accumulator-backend', 'device_fp32']
    validate_stage_parameters('native_state_sft', {'cotangent_accumulator_backend': 'device_fp32'})
    with pytest.raises(ValueError, match='cotangent_accumulator_backend'):
        validate_stage_parameters('native_state_sft', {'cotangent_accumulator_backend': 'automatic'})
    old = {'stages': [{'id': 'ce', 'kind': 'native_state_sft', 'parameters': {}}]}
    unsafe = {'stages': [{'id': 'ce', 'kind': 'native_state_sft', 'parameters': {
        'cotangent_accumulator_backend': 'device_fp32'}}]}
    with pytest.raises(ValueError, match='fresh exact-parent memory qualification'):
        direct_windowing_handoffs(old, unsafe, out=tmp_path)
    safe = {'stages': [{'id': 'ce', 'kind': 'native_state_sft', 'parameters': {
        'cotangent_accumulator_backend': 'device_fp32',
        'memory_qualification': {'path': 'new.json', 'sha256': 'a' * 64}}}]}
    direct_windowing_handoffs(old, safe, out=tmp_path)


def test_legacy_resume_identity_defaults_backend_without_mutating_saved_identity():
    from natlang_neuralese.train.native_state_sft import _resume_identity_for_comparison

    saved = {'schema': 'native-training-identity/1', 'seed': 7}
    previous = _resume_identity_for_comparison(saved)
    assert previous == {**saved, 'cotangent_accumulator_backend': 'cpu_file'}
    assert saved == {'schema': 'native-training-identity/1', 'seed': 7}
    assert _resume_identity_for_comparison({**saved, 'cotangent_accumulator_backend': 'device_fp32'})[
        'cotangent_accumulator_backend'] == 'device_fp32'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device backend needs a CUDA device')
def test_device_backend_keeps_explicit_higher_order_reference_arithmetic(tmp_path):
    from natlang_neuralese.common.native_higher_order import reference_arithmetic

    layer = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16, device='cuda')
    x = torch.randn(2, 4, dtype=torch.bfloat16, device='cuda')
    with ParameterCotangentScope.for_model(layer, mode='vjp', directory=tmp_path,
                                           accumulator_backend='device_fp32') as scope:
        with reference_arithmetic():
            value = layer(x).float().square().sum()
            first, = torch.autograd.grad(value, (layer.weight,), create_graph=True)
            second, = torch.autograd.grad(first.float().square().sum(), (layer.weight,))
            assert torch.isfinite(first).all() and torch.isfinite(second).all()
            assert float(second.float().norm()) > 0
        scope.finish()
    # Reference arithmetic deliberately uses its differentiable FP32 cast node;
    # it must not allocate or consume the first-order device accumulator.
    assert scope.allocation_report()['allocated_accumulator_bytes'] == 0


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device backend needs a CUDA device')
def test_native_shared_graph_update_selects_backend_from_checkpoint_option(family):
    from test_shared_state import tiny
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, native_training_update

    torch.manual_seed(112)
    device = torch.device('cuda')
    first = SharedStateModel(tiny(family).to(dtype=torch.bfloat16), (60, 61, 62, 63), rank=4).to(device)
    if family == 'mellum':
        first.port.hf.natlang_precision = 'bf16'
    initial = {name: value.detach().clone() for name, value in first.state_dict().items()}
    example = parse_graph({'schema': SCHEMA, 'id': f'backend-{family}', 'split': 'train',
        'source_groups': [f'backend-{family}-train'], 'tokenizer_fingerprint': 'tiny',
        'nodes': [{'name': 'answer', 'token_ids': [4, 5, 6, 7, 8, 9],
                   'target_weights': [0, 0, 1, 1, 1, 1]}]}, tokenizer_fingerprint='tiny')
    outcomes = []
    rng = torch.cuda.get_rng_state().clone()
    for backend in ('cpu_file', 'device_fp32'):
        model = SharedStateModel(tiny(family).to(dtype=torch.bfloat16), (60, 61, 62, 63), rank=4).to(device)
        if family == 'mellum':
            model.port.hf.natlang_precision = 'bf16'
        model.load_state_dict(initial)
        model.native_cotangent_accumulator_backend = backend
        optimizer, _ = make_native_optimizer(model, lr=1e-4, weight_decay=0.)
        torch.cuda.set_rng_state(rng)
        loss, _ = native_training_update(model, optimizer, [example], checkpoint_layers=False,
                                         chunk_tokens=8)
        outcomes.append({'loss': loss.detach().clone(),
                         'state': {name: value.detach().clone() for name, value in model.state_dict().items()},
                         'rng': torch.cuda.get_rng_state().clone()})
    assert torch.equal(outcomes[0]['loss'], outcomes[1]['loss'])
    assert torch.equal(outcomes[0]['rng'], outcomes[1]['rng'])
    for name in outcomes[0]['state']:
        assert torch.equal(outcomes[0]['state'][name], outcomes[1]['state'][name]), name
