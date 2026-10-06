from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.serve import grad
from natlang_neuralese.serve.chat import RequestError
from natlang_neuralese.train.trajectory_state import validate_continuation


def test_value_ce_scores_body_after_exact_forced_prefix_without_boundary_merging(monkeypatch):
    monkeypatch.setattr(grad, 'render_messages', lambda *args, **kwargs: SimpleNamespace(segments=['prompt'], blocks=[], escape_nonce='nonce'))
    session = object.__new__(grad.GradSession)
    tokens = {'value=': [10, 11], '{"x": 1}': [12, 13], 'value={"x": 1}': [10, 99, 13]}
    session.engine = SimpleNamespace(_template=None, specials=None, block_value_type=None, _tokens=tokens.__getitem__)
    session._items = lambda *args: [('tok', 5), ('block', 'leaf')]
    logp = torch.tensor([-1., -2.], requires_grad=True)
    leaves = {'leaf': torch.tensor([1.], requires_grad=True)}
    def score(prompt, target, resolved, write_terms, collect_token_states=False):
        assert prompt == [('tok', 5), ('block', 'leaf'), ('tok', 10), ('tok', 11)]
        assert target == [('tok', 12), ('tok', 13)]
        assert resolved is leaves and not write_terms
        return {'token_logp': logp}
    session._score = score
    loss = session.supervised_continuation_loss([], None, 'value=', '{"x": 1}', leaves)
    assert loss.item() == 1.5
    loss.backward()
    torch.testing.assert_close(logp.grad, torch.tensor([-.5, -.5]))


def test_empty_continuation_is_not_admitted(monkeypatch):
    monkeypatch.setattr(grad, 'render_messages', lambda *args, **kwargs: SimpleNamespace(segments=[], blocks=[], escape_nonce='nonce'))
    session = object.__new__(grad.GradSession)
    session.engine = SimpleNamespace(_template=None, specials=None, block_value_type=None, _tokens=lambda text: [] if not text else [1])
    session._items = lambda *args: []
    with pytest.raises(RequestError, match='target tokens'):
        session.supervised_continuation_loss([], None, '=', '', {})


def test_writer_boundary_supervision_and_horizon_changes_are_explicit():
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {'steps': 2048}, 'files': {'data': 'pinned'}}}
    identity = {'options': {'steps': 3072, 'writer_supervision': 'native-value'}, 'files': {'data': 'pinned'}}
    with pytest.raises(ValueError, match='identical inputs'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['steps', 'writer_supervision'])


def test_gold_stop_boundary_has_terminal_label_on_last_consumed_value_state(monkeypatch):
    monkeypatch.setattr(grad, 'render_messages', lambda *args, **kwargs: SimpleNamespace(segments=[], blocks=[], escape_nonce='nonce'))
    session = object.__new__(grad.GradSession)
    session.engine = SimpleNamespace(_template=None, specials=None, block_value_type=None, _tokens=lambda text: [1, 2] if text == 'body' else [3])
    session._items = lambda *args: []
    states = torch.randn(1, 2, 4, requires_grad=True)
    logits = torch.zeros(1, 2, requires_grad=True)
    def stop(actual, counts):
        assert actual is states
        assert counts.tolist() == [[1, 2]]
        return logits
    session.heads = SimpleNamespace(read_markers=False, stop=stop)
    def score(*args, collect_token_states=False, **kwargs):
        assert collect_token_states
        return {'token_logp': torch.tensor([-1., -2.]), 'token_stop_states': states}
    session._score = score
    loss = session.supervised_continuation_loss([], None, '=', 'body', {}, text_weight=0., stop_weight=1.)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[.25, -.25]]))


def test_gold_stop_supervision_change_is_explicit():
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': {'options': {}, 'files': {}}}
    identity = {'options': {'stop_supervision': 'gold-native-boundary'}, 'files': {}}
    with pytest.raises(ValueError, match='identical inputs'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['stop_supervision'])
