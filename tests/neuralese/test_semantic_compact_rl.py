"""Actual compact producer capture remains replayable from its durable RL bundle."""
from types import SimpleNamespace
import copy

import pytest
import torch

from test_native_rl_replay import actor_rollout
from test_native_semantic import setup_pair, PayloadTokenizer
from natlang_neuralese.model.semantic_channel import SemanticChannel
from natlang_neuralese.model.semantic_vae import SemanticStateVAE, StateLayout
from natlang_neuralese.rl.native_replay import compile_replay, replay_logps
from natlang_neuralese.rl.graph_replay import save_bundle
from natlang_neuralese.serve.engine import Engine
from natlang_neuralese.serve.store import TensorStore, encode_native_state, decode_native_state


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_compact_actual_producer_archive_and_rl_logp_gradient_replay(tmp_path, family):
    model, roles, identity, _, block, turns = actor_rollout(family)
    block.text = 'abcd'
    participants, row, _ = setup_pair()
    peer = participants['mellum' if family == 'lfm' else 'lfm']
    from natlang_neuralese.train.native_semantic import parse_pair, producer
    example, view = parse_pair(row, participants)['mellum' if family == 'lfm' else 'lfm']
    with torch.no_grad():
        peer_block = producer(peer, example, view, supervise=False)[0]
    vae = SemanticStateVAE({'source': StateLayout.from_block(block), 'peer': StateLayout.from_block(peer_block)},
                           latent_width=4, hidden_width=16)
    tokenizer = PayloadTokenizer([5, 6], [(0, 2), (2, 4)])
    channel = SemanticChannel(vae, {'source': identity, 'peer': peer.identity},
        {'source': tokenizer, 'peer': peer.tokenizer}, adapter_sha256='a'*64)
    engine = object.__new__(Engine)
    engine.native_identity = identity
    engine.model_config = {'tokenizer_fingerprint': identity['tokenizer_fingerprint'], 'architecture': identity['architecture']}
    engine.checkpoint_sha256 = identity['checkpoint_sha256']
    engine.embedding_weight_sha256 = identity['embedding_weight_sha256']
    engine.semantic_channel, engine.store = channel, TensorStore()
    capture = {key: getattr(block, key) for key in ('token_ids', 'positions', 'embeddings', 'layers')}
    with torch.no_grad():
        raw, compact = engine._native_store_capture([5, 6], [capture], text='abcd')
    raw_record = engine.store.put_native_state(raw)
    assert compact.id != raw_record.id
    assert set(compact.block.tensors()) == {'token_ids', 'positions', 'embeddings', 'code'}
    turn = copy.deepcopy(turns[1])
    write = turn['sampling']['writes'][0]
    write.update(state_id=compact.id, raw_state_id=raw_record.id)
    turn['written_nodes'] = {compact.id: 'actual-write'}
    packet = [turn]
    bundle = save_bundle(packet, tmp_path/'bundle', policy_id='actual-source',
                        fetch=lambda key: encode_native_state(engine.store.get_native_state(key).block))
    assert set(bundle['blocks']) == {compact.id, raw_record.id}
    archived = {key: decode_native_state((tmp_path/'bundle'/entry['path']).read_bytes())[0]
                for key, entry in bundle['blocks'].items()}
    replay = compile_replay(packet, identity=identity, roles=roles, fetch_block=archived.__getitem__)
    logps, behavior = replay_logps(model, replay, chunk_tokens=2)
    torch.testing.assert_close(logps.detach(), behavior, atol=2e-6, rtol=2e-5)
    (-logps.mean()).backward()
    assert torch.isfinite(model.adapter.up.weight.grad).all()
    assert model.adapter.up.weight.grad.abs().sum() > 0
    missing = copy.deepcopy(packet)
    del missing[0]['sampling']['writes'][0]['raw_state_id']
    with pytest.raises(ValueError, match='decoding'):
        compile_replay(missing, identity=identity, roles=roles, fetch_block=archived.__getitem__)
