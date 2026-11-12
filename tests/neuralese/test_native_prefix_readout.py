"""Read shared causal states once without changing original graph objectives."""
import copy

import pytest
import torch

from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
from natlang_neuralese.train.native_graph_loss import packed_graph_loss
from test_native_prefix_packing import examples
from test_shared_state import tiny


@pytest.mark.parametrize('family', ['lfm'])
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
