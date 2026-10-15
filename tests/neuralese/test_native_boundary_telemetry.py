"""Held boundary evidence reuses exact CE and never changes the objective."""
import copy

import pytest
import torch

from natlang_neuralese.data.native_graph import graph_loss, parse_graph
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import _evaluate
from test_native_graph import row
from test_shared_state import tiny


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_boundary_sums_match_readout_and_preserve_exact_loss_gradients(family):
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    other = copy.deepcopy(model)
    example = parse_graph(row(), tokenizer_fingerprint='f'*64)
    expected_loss, _ = graph_loss(model, example, chunk_tokens=2)
    rng = torch.get_rng_state().clone()
    actual_loss, detail = graph_loss(other, example, chunk_tokens=2,
                                    boundary_token_ids=(4, 5, 7, 8, 58))
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(actual_loss, expected_loss)
    expected_loss.backward(); actual_loss.backward()
    for (name, first), (other_name, second) in zip(model.named_parameters(), other.named_parameters()):
        assert name == other_name
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            assert torch.equal(first.grad, second.grad), name
    with torch.no_grad():
        result = model(example.plan, retain_states=False)
        logits = model.logits(result['h_final'][0, [0, 1, 4]]).float()
        targets = torch.tensor([4, 5, 8])
        logp = logits.log_softmax(-1)[torch.arange(3), targets]
        for index, (token, weight) in enumerate([(4, .1), (5, 1.), (8, 1.)]):
            actual = detail['boundary_sums'][str(token)]
            assert actual['count'] == 1
            assert actual['weight'] == pytest.approx(weight)
            assert actual['nll_sum'] == pytest.approx(float(-logp[index])*weight, rel=1e-6)
            assert actual['probability_sum'] == pytest.approx(float(logp[index].exp())*weight, rel=1e-6)
            assert actual['correct_weight'] == pytest.approx(float(logits[index].argmax() == token)*weight)
    assert all(value == 0 for value in detail['boundary_sums']['7'].values())
    assert all(value == 0 for value in detail['boundary_sums']['58'].values())


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_evaluator_aggregates_support_without_extra_forward_or_readout(family, monkeypatch):
    model = SharedStateModel(tiny(family), marker_ids=(4, 5, 7, 8), rank=4,
                             token_override_ids=(58, 59))
    example = parse_graph(row(), tokenizer_fingerprint='f'*64)
    calls = {'forward': 0, 'logits': 0}
    for name in calls:
        original = getattr(model, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(model, name, counted)
    model.train()
    measured = _evaluate(model, [example, example], checkpoint_layers=False, chunk_tokens=2)
    assert model.training
    assert calls == {'forward': 2, 'logits': 4}
    boundaries = measured['boundaries']
    assert boundaries['4']['count'] == 2
    assert boundaries['4']['weight'] == pytest.approx(.2)
    assert boundaries['4']['mean_nll'] == pytest.approx(boundaries['4']['nll_sum']/.2)
    assert boundaries['8']['count'] == 2
    assert boundaries['7']['count'] == 0
    assert boundaries['7']['mean_probability'] is None
    assert boundaries['58']['mean_nll'] is None
    assert boundaries['59']['weighted_accuracy'] is None
    assert measured['supervised_tokens'] == 6
    model.eval()
    _evaluate(model, [example], checkpoint_layers=False, chunk_tokens=2)
    assert not model.training


def test_evaluator_restores_mode_on_observation_failure():
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    model.train()
    with pytest.raises(FloatingPointError, match='empty'):
        _evaluate(model, [], checkpoint_layers=False, chunk_tokens=2)
    assert model.training


@pytest.mark.parametrize('ids', [(4, 4), (-1,), (True,), ('4',)])
def test_invalid_observation_inventory_is_rejected(ids):
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    example = parse_graph(row(), tokenizer_fingerprint='f'*64)
    with pytest.raises(ValueError, match='distinct nonnegative integer'):
        graph_loss(model, example, boundary_token_ids=ids)
