"""Actual paired held forward retains raw corpus CE with train policy active."""
from types import SimpleNamespace

import pytest
import torch

from test_native_semantic import setup_pair
from natlang_neuralese.train.native_semantic import joint_loss
from natlang_neuralese.train.native_semantic_refresh import held_evaluation
from natlang_neuralese.train.native_semantic_stage import SourceBoundPair


@pytest.mark.parametrize('phase', ['joint', 'distill'])
def test_semantic_held_uses_raw_weights_and_preserves_state_rng(phase):
    participants, raw, vae = setup_pair()
    row = SourceBoundPair(raw)
    row.source_sha256 = 'b' * 64
    row['split'] = 'test'
    for view in row['models'].values():
        graph = view['graph']
        graph['split'] = 'test'
        # Deliberately eligible for explicit source coverage. Passing the train
        # declaration into this held forward would raise before any metrics.
        graph['conversion'] = {'role_interpretation': 'ordinary-text', 'source_suffix_start': 2}
    source = SimpleNamespace(entries=[{'models': list(participants)}], sha256='b' * 64,
                             rows=lambda: iter([row]))
    owners = [vae, *[participant.model for participant in participants.values()]]
    saved = [[p.detach().clone() for p in owner.parameters()] for owner in owners]
    for owner in owners:
        owner.train()
    torch.manual_seed(29)
    with torch.no_grad():
        for owner in owners:
            owner.eval()
        expected = float(joint_loss(vae, participants, row, phase=phase)[0])
        for owner in owners:
            owner.train()
    torch.manual_seed(29)
    rng = torch.get_rng_state().clone()
    policy = {'sources': [{'sha256':'a' * 64, 'encoding':'ordinary-suffix-raw-v1', 'context_weight':.05}]}
    options = {'_ce_supervision_policy': policy}
    actual, coverage = held_evaluation(participants, vae, source, phase=phase,
        options=options, resolver=None, final=True)
    assert actual['loss'] == pytest.approx(expected, abs=1e-7)
    assert actual['scope'] == 'full' and actual['source_sha256'] == 'b' * 64
    assert coverage['row_visits'] == {row['id']: 1}
    assert options['_ce_supervision_policy'] is policy
    assert torch.equal(torch.get_rng_state(), rng)
    for owner, values in zip(owners, saved):
        assert owner.training
        for parameter, reference in zip(owner.parameters(), values):
            assert torch.equal(parameter, reference)
            assert parameter.grad is None
