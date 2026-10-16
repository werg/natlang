"""Observe rare control targets and update directions without another model pass."""
import pytest
import torch

from test_shared_state import tiny
from test_native_state_sft import graph
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import packed_graph_loss, sparse_row_updates


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_actual_targets_include_prompt_weight_but_not_forced_or_masked_boundaries(family):
    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    model.add_token_overrides((7, 8))
    communication = graph('boundaries')
    communication['nodes'][0].update(token_ids=[1,58,2,59,3,60,4,61,5,58],
                                    target_weights=[0,1,.5,1,0,.05,0,0,0,0])
    ordinary = graph('thinking')
    ordinary['nodes'][0].update(token_ids=[1,7,8], target_weights=[0,.5,1])
    examples = [parse_graph(row, tokenizer_fingerprint='tokenizer-sha') for row in (communication, ordinary)]
    forwards = []
    hook = model.register_forward_hook(lambda *unused: forwards.append(1))
    loss, details = packed_graph_loss(model, examples, checkpoint_layers=False, chunk_tokens=2)
    loss.backward()
    hook.remove()
    assert forwards == [1] and torch.isfinite(loss)
    roles = details[0]['boundary_targets']
    assert roles['58'] == {'count': 1, 'weight': 1.}
    assert roles['59'] == {'count': 1, 'weight': 1.}
    assert roles['60'] == {'count': 1, 'weight': .05}
    assert roles['61'] == {'count': 0, 'weight': 0.}
    assert details[1]['boundary_targets']['7'] == {'count': 1, 'weight': .5}
    assert details[1]['boundary_targets']['8'] == {'count': 1, 'weight': 1.}


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_directions_distinguish_descent_ascent_momentum_and_missing_gradient(family):
    model = SharedStateModel(tiny(family), (58,59,60,61), rank=4)
    parameter = model.input_rows
    before = {'input_rows': parameter.detach().clone()}
    parameter.grad = torch.ones_like(parameter)
    parameter.grad[1].zero_()
    with torch.no_grad():
        parameter[0].sub_(.02)
        parameter[1].add_(.02)
        parameter[2].add_(.02)
    current, rng = parameter.detach().clone(), torch.get_rng_state().clone()
    rows = sparse_row_updates(model, before)['input_rows']['rows']
    assert rows[0]['gradient_update_dot'] < 0
    assert rows[0]['gradient_update_cosine'] == pytest.approx(-1., abs=1e-6)
    assert rows[1]['gradient_present'] and rows[1]['gradient_l2'] == 0
    assert rows[1]['update_l2'] > 0 and rows[1]['gradient_update_cosine'] is None
    assert rows[2]['gradient_update_dot'] > 0
    assert rows[2]['gradient_update_cosine'] == pytest.approx(1., abs=1e-6)
    parameter.grad = None
    missing = sparse_row_updates(model, before)['input_rows']['rows']
    assert all(not row['gradient_present'] and row['gradient_update_dot'] is None for row in missing)
    assert torch.equal(parameter, current) and torch.equal(torch.get_rng_state(), rng)
