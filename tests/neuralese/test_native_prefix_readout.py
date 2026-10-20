"""Read shared causal states once without changing original graph objectives."""
import copy

import pytest
import torch

from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from test_native_prefix_packing import examples
from test_shared_state import tiny


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('chunk_tokens', [1, 3, 128])
def test_reused_readout_preserves_every_parameter_gradient_and_each_graph_weight(family, chunk_tokens):
    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    for parameter in model.port.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        model.adapter.up.weight.normal_(std=.1)
    reference = copy.deepcopy(model)
    rows = examples()  # duplicates, internal terminal, different branches and unequal graph masses
    rows[0].target_weights = (0., .05, 0., 1., .05, 1.)
    rows[1].target_weights = (0., 0., 0., .1, 1.)
    shared, maps = pack_causal_prefixes(rows)
    pairs = [(mapping[position-1], token)
             for example, mapping in zip(rows, maps)
             for position, token in enumerate(example.plan.invocations[0].token_ids)
             if position > 0 and example.target_weights[position] > 0]
    assert len(set(pairs)) < len(pairs)
    # One physical predecessor can predict different branch tokens; never merge labels.
    assert len(set(pairs)) > len({query for query, _ in pairs})
    counts = {'actual':0, 'reference':0}
    for name, owner in [('actual',model), ('reference',reference)]:
        forward = owner.logits
        def counted(states, *, name=name, forward=forward):
            counts[name] += states.shape[0]
            return forward(states)
        owner.logits = counted
    expected, expected_details = packed_graph_loss(reference, rows, share_causal_prefixes=True,
        checkpoint_layers=True, chunk_tokens=chunk_tokens, collect_accuracy=True)
    actual, details = packed_graph_loss(model, rows, share_causal_prefixes=True,
        checkpoint_layers=True, chunk_tokens=chunk_tokens, collect_accuracy=False)
    assert counts == {'actual':len(set(pairs)), 'reference':len(pairs)}
    assert details[0]['readout_forward_rows'] == len(set(pairs))
    assert details[0]['readout_original_rows'] == len(pairs)
    assert [row['weight'] for row in details] == [row['weight'] for row in expected_details]
    assert [row['boundary_targets'] for row in details] == [row['boundary_targets'] for row in expected_details]
    assert all(row['accuracy'] is None for row in details)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    expected.backward()
    for (name, parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name and (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-6, rtol=2e-4, msg=name)


def test_readout_reuse_remains_disabled_without_actual_shared_physical_rows():
    model = SharedStateModel(tiny('lfm'), (58, 59, 60, 61), rank=4)
    _, details = packed_graph_loss(model, examples(), share_causal_prefixes=False, collect_accuracy=False)
    assert details[0]['readout_forward_rows'] == details[0]['readout_original_rows']


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_production_precision_preserves_each_derivative_and_bf16_rounding(family, dtype):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    model = SharedStateModel(tiny(family).to(device='cuda', dtype=dtype),
        (58,59,60,61), rank=4, native_norm_execution='stable-v1')
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'
    configure_norm_precision(model, 'fp32-norms-v1')
    set_native_linear_precision(model, 'bf16x6')
    configure_backbone_training(model.port, 'latent')
    with torch.no_grad():
        model.adapter.up.weight.normal_(std=.1)
    reference = copy.deepcopy(model)
    outputs = []
    for owner, accuracy in [(reference,True), (model,False)]:
        with ParameterCotangentScope.for_model(owner, mode='vjp') as scope:
            loss, details = packed_graph_loss(owner, examples(), share_causal_prefixes=True,
                checkpoint_layers=True, chunk_tokens=3, collect_accuracy=accuracy)
            loss.backward()
            scope.finish()
        outputs.append((loss.detach(), details))
    tolerance = dict(atol=0, rtol=0) if dtype == torch.bfloat16 else dict(atol=3e-6, rtol=2e-4)
    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=0 if dtype == torch.bfloat16 else 1e-6,
                               rtol=0 if dtype == torch.bfloat16 else 1e-6)
    if dtype == torch.bfloat16:
        assert outputs[1][1][0]['readout_forward_rows'] == outputs[1][1][0]['readout_original_rows']
    else:
        assert outputs[1][1][0]['readout_forward_rows'] < outputs[1][1][0]['readout_original_rows']
    for (name,parameter), (other_name, other) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name and (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, **tolerance, msg=name)
