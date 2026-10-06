import copy
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.capacity import checkpoint_write_capacity, set_write_capacity, source_vector_length
from natlang_neuralese.model.heads import StopHead


def test_constant_stop_capacity_preserves_parameters_optimizer_rng_and_logits():
    stop = StopHead(4, 128, hidden=8, use_position=False)
    heads = SimpleNamespace(stop=stop, max_length=128)
    optimizer = torch.optim.AdamW(stop.parameters(), lr=.001)
    h = torch.randn(2, 4)
    stop(h, torch.tensor([1, 128])).sum().backward()
    optimizer.step(); optimizer.zero_grad()
    before = copy.deepcopy(optimizer.state_dict())
    weights = {name: (id(p), p.detach().clone()) for name, p in stop.named_parameters()}
    expected = stop(h, torch.tensor([1, 128]))
    rng = torch.get_rng_state().clone()
    set_write_capacity(heads, 512)
    assert heads.max_length == stop.max_length == 512
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(stop(h, torch.tensor([145, 512])), expected, rtol=0, atol=0)
    for name, p in stop.named_parameters():
        assert id(p) == weights[name][0]
        assert torch.equal(p, weights[name][1])
    after = optimizer.state_dict()
    assert before['param_groups'] == after['param_groups']
    for key, values in before['state'].items():
        for name, value in values.items():
            assert torch.equal(value, after['state'][key][name])


def test_position_dependent_stop_cannot_silently_extend_learned_rows():
    heads = SimpleNamespace(stop=StopHead(4, 128, hidden=8, use_position=True), max_length=128)
    with pytest.raises(ValueError, match='learned rows'):
        set_write_capacity(heads, 512)
    assert heads.max_length == 128


def test_checkpoint_capacity_is_explicit_and_separate_from_constant_row_storage():
    saved = {'stop.position.weight': torch.zeros(129, 64)}
    assert checkpoint_write_capacity(saved, {}) == (128, 128)
    metadata = {'max_length': 512, 'stop_position': False}
    assert checkpoint_write_capacity(saved, metadata) == (512, 128)
    with pytest.raises(ValueError, match='match checkpoint'):
        checkpoint_write_capacity(saved, metadata, 128)
    with pytest.raises(ValueError, match='position-dependent'):
        checkpoint_write_capacity(saved, {**metadata, 'stop_position': True})


def test_gold_length_never_silently_clips_and_explicit_compression_remains_available():
    assert source_vector_length(145, 1, 512) == 145
    assert source_vector_length(145, 2, 128) == 73
    with pytest.raises(ValueError, match='needs 145 vectors'):
        source_vector_length(145, 1, 128)
    for ratio in [0, -1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='sizing controls'):
            source_vector_length(145, ratio, 512)


def test_capacity_change_requires_declared_continuation_curriculum():
    from natlang_neuralese.train.trajectory_state import validate_continuation
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {}, 'files': {'data': 'pinned'}}}
    identity = {'options': {'max_write_vectors': 512}, 'files': {'data': 'pinned'}}
    with pytest.raises(ValueError, match='identical inputs'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['max_write_vectors'])
