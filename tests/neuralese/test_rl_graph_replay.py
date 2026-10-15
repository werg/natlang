"""Durable value-channel replay and writer gradients on both canonical ports."""
import json
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl.graph_replay import restore_bundle, save_bundle, replay_scores
from natlang_neuralese.serve.store import TensorStore, make_block, encode_block


def test_bundle_roundtrip_recurses_producer_dependencies_and_detects_corruption(tmp_path):
    first = make_block(torch.randn(2, 8), 'dialect')
    second = make_block(torch.randn(3, 8), 'dialect', producer={'context': first.id})
    blocks = {b.id: b for b in [first, second]}
    rollouts = [{'turns': [{'target': {'content': [{'type': 'neuralese', 'id': second.id}]}}]}]
    manifest = save_bundle(rollouts, tmp_path, policy_id='p', fetch=lambda bid: encode_block(blocks[bid]))
    assert set(manifest['blocks']) == set(blocks)
    engine = SimpleNamespace(store=TensorStore(), width=8)
    restore_bundle(engine, tmp_path / 'manifest.json', policy_id='p')
    assert torch.equal(engine.store.get(second.id).payload, second.payload)
    with pytest.raises(ValueError, match='different policy'):
        restore_bundle(engine, tmp_path / 'manifest.json', policy_id='other')
    path = tmp_path / manifest['blocks'][first.id]['path']
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match='bytes changed'):
        restore_bundle(engine, tmp_path / 'manifest.json', policy_id='p')


def test_native_bundle_archives_prompt_write_and_json_argument_references(tmp_path):
    from test_native_state_store import _block
    from natlang_neuralese.serve.store import encode_native_state, native_state_id
    first = _block()
    from dataclasses import replace
    second = replace(first, text='another producer')
    blocks = {native_state_id(block): block for block in (first, second)}
    first_id, second_id = blocks
    rollouts = [{'turns': [{'sampling': {'schema': 'natlang.native-rl-sampling/1',
        'prompt': [{'kind': 'native_state', 'id': first_id}],
        'writes': [{'state_id': second_id}]},
        'messages': [{'tool_calls': [{'function': {'arguments': json.dumps(
            {'value': [{'type': 'native_state', 'id': first_id}]})}}]}]}]}]
    manifest = save_bundle(rollouts, tmp_path, policy_id='p',
                           fetch=lambda bid: encode_native_state(blocks[bid]))
    assert set(manifest['blocks']) == set(blocks)
    assert all(entry['format'] == 'native-state' for entry in manifest['blocks'].values())
    engine = SimpleNamespace(store=TensorStore(), native_state_enabled=True)
    restore_bundle(engine, tmp_path / 'manifest.json', policy_id='p')
    for block_id in blocks:
        assert encode_native_state(engine.store.get_native_state(block_id).block) == encode_native_state(blocks[block_id])
    with pytest.raises(ValueError, match='native-state learner'):
        restore_bundle(SimpleNamespace(store=TensorStore()), tmp_path / 'manifest.json', policy_id='p')


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_consumer_reward_gradient_reaches_recorded_value_writer(family, tmp_path):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve.grad import GradSession
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    if family == 'lfm':
        from test_backbone_policy import tiny_lfm
        backbone = tiny_lfm()
    else:
        from test_mellum_port import tiny_mellum
        from natlang_neuralese.maple.model import load_maple
        from natlang_neuralese.maple.maple_port import MaplePortBackbone
        tiny_mellum(tmp_path)
        model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
        model.natlang_precision = 'bf16'
        backbone = MaplePortBackbone(model, ControlTokens(126, 127), fast=False)
    configure_backbone_training(backbone, 'latent')
    heads = PortHeads(backbone, cutoff=backbone.num_layers, max_length=8, profile='raw-token-v1').eval()
    projection = NeuraleseProjection(backbone.config.hidden_size, hidden=16)
    with torch.no_grad():
        projection.linear.weight.normal_(0, .02)
    store = TensorStore()
    block = store.put(make_block(torch.randn(2, backbone.config.hidden_size), 'dialect'))
    engine = SimpleNamespace(backbone=backbone, heads=heads, device='cpu', store=store,
        width=backbone.config.hidden_size, lookup=store.get, neuralese_projection=projection,
        token_choice_projection=None, stop_ids={2}, resolve_adapters=lambda *_: [])
    engine.advance_token_readout = lambda history, hidden, last_only=False: (backbone.logits(hidden), None)
    session = GradSession(engine)
    # Isolate graph transport/gradient from tokenizer testing: token coordinates
    # are explicit, the backbone and writer/reader implementations are real.
    def target_items(messages, tools, target):
        if messages[0]['content'] == 'producer':
            return [('tok', i) for i in [1, 4, 9]], [('block', block.id)]
        return [('block', block.id), ('tok', 5)], [('tok', 7), ('tok', 8)]
    session._target_items = target_items
    session._term_blocks = lambda term: [block.id]
    producer = {'messages': [{'role': 'user', 'content': 'producer'}],
                'reply': {'role': 'assistant', 'content': [{'type': 'neuralese', 'id': block.id}]}}
    reader = {'messages': [{'role': 'user', 'content': 'consumer'}],
              'target': {'role': 'assistant', 'content': 'answer'}}
    prompt, target = target_items(reader['messages'], None, reader['target'])
    detached = session._score(prompt, target, {}, write_terms=False)['token_logp']
    chained = replay_scores(session, reader, [producer])
    assert torch.equal(chained.detach(), detached.detach())  # same recorded observations
    (-chained.mean()).backward()
    assert projection.linear.weight.grad is not None
    assert projection.linear.weight.grad.abs().sum() > 0
    assert torch.isfinite(projection.linear.weight.grad).all()
    assert torch.equal(store.get(block.id).payload, block.payload)


def test_durable_write_failure_keeps_old_state_and_cleans_pending(tmp_path, monkeypatch):
    from natlang_neuralese.common.jsonio import write_json_durable
    path = tmp_path / 'state.json'
    write_json_durable(path, {'old': True})
    before = path.read_bytes()
    monkeypatch.setattr('natlang_neuralese.common.jsonio.os.fsync', lambda _: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError, match='disk'):
        write_json_durable(path, {'new': True})
    assert path.read_bytes() == before
    assert not (tmp_path / 'state.json.pending').exists()



def test_equal_block_payloads_use_exact_writer_occurrence_and_reject_ambiguity():
    from natlang_neuralese.rl.graph_replay import captured_producer_leaves, producer_turns
    block = 'nz1_' + 'a' * 52
    def capture(token, *, write=False):
        return {'schema': 'natlang.rl-sampling/1', 'temperature': 1., 'guided': False,
            'prompt': [['tokens', [token]]], 'items': [['block', block]] if write else [['tok', 9]],
            'samples': [] if write else [{'item': 0, 'token': 9, 'logp': -2.}]}
    turns = [{'invocation_id': call, 'messages': [], 'target': {}, 'sampling': capture(token, write=True)}
             for call, token in [('first', 1), ('second', 2)]]
    graph = [{'node': f'{call}#write', 'kind': 'block_write', 'turn': f'{call}#turn1', 'block': block}
             for call in ['first', 'second']]
    producers = producer_turns({'turns': turns, 'execution_graph': graph})
    reader = capture(3)
    reader['prompt'] = [['block', block]]
    selected = []
    class Session:
        def _rewritten(self, bid, leaves, indexed, memo):
            selected.append(indexed[bid][1][0])
            return torch.ones(2, 3)
    captured_producer_leaves(Session(), reader, producers, read_bindings={block: 'second#write'})
    assert selected == [('tok', 2)]
    with pytest.raises(ValueError, match='ambiguous'):
        captured_producer_leaves(Session(), reader, producers)
