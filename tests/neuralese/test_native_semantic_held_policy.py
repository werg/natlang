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


@pytest.mark.parametrize('phase', ['joint','distill'])
def test_semantic_dual_metrics_share_existing_forwards_and_exact_held_binding(phase):
    from natlang_neuralese.data.native_graph import parse_graph,graph_loss
    from natlang_neuralese.train.native_supervision import held_measurements
    participants,raw,vae = setup_pair()
    row = SourceBoundPair(raw)
    row.source_sha256 = 'b'*64
    row['split'] = 'test'
    for view in row['models'].values():
        graph = view['graph']
        graph['split'] = 'test'
        graph['nodes'][0]['name'] = 'producer'
        graph['nodes'][1]['imports'][0]['source'] = 'producer'
        view['export']['node'] = 'producer'
        graph['nodes'][0]['target_weights'][1] = .05
        graph['nodes'][1]['target_weights'][1] = .05
        graph['source_pairing'] = {'pair_id':'fixture','payloads':[{'invocation':'producer'}],
            'consumed_occurrences':[{'consumer_import_at':2}],'source_ids':['w','r']}
    declaration = {'sha256':'b'*64,'encoding':'witnessed-pair-raw-v1','context_weight':.05}
    policy = {'sources':[dict(declaration,sha256='a'*64)],'held_sources':[declaration]}
    source = SimpleNamespace(entries=[{'models':list(participants)}],sha256='b'*64,rows=lambda:iter([row]))
    counts = {name:[] for name in participants}
    hooks = [entry.model.register_forward_hook(lambda _, __, ___, name=name:counts[name].append(1))
             for name,entry in participants.items()]
    torch.manual_seed(29)
    original,_ = held_evaluation(participants,vae,source,phase=phase,options={},resolver=None,final=True)
    baseline = {name:len(value) for name,value in counts.items()}
    for value in counts.values():value.clear()
    torch.manual_seed(29)
    measured,_ = held_evaluation(participants,vae,source,phase=phase,
        options={'_ce_supervision_policy':policy},resolver=None,final=True)
    assert measured['loss'] == original['loss']
    assert {name:len(value) for name,value in counts.items()} == baseline
    for hook in hooks:hook.remove()
    for name in participants:
        assert measured['additional_ce'][name+':control']['target']['weight'] > 0
        assert measured['additional_ce'][name+':adapted-consumer']['declared']['weight'] > 0
    assert measured['additional_ce_bindings'] == [declaration | {'source_sha256':'b'*64}]
