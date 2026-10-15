"""Native graph loss cannot cross invocation/import boundaries accidentally."""
import copy

import pytest
import torch

from natlang_neuralese.data.native_graph import SCHEMA, parse_graph, graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel
from test_shared_state import tiny


def row():
    return {'schema': SCHEMA, 'id': 'native-train-example', 'split': 'train',
        'source_groups': ['native-train-source'], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [3, 4, 5], 'producing': [[1, 3]],
             'target_weights': [0, 0.1, 1]},
            {'name': 'consumer', 'token_ids': [6, 7, 8],
             'imports': [{'at': 1, 'source': 'producer', 'start': 1, 'stop': 3}],
             'target_weights': [0, 0, 1]}]}


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_graph_loss_matches_manual_own_predecessors_and_keeps_source_gradient(family):
    from torch.nn import functional as F
    port = tiny(family)
    model = SharedStateModel(port, marker_ids=(58, 59, 60, 61), rank=4)
    example = parse_graph(row(), tokenizer_fingerprint='f' * 64)
    loss, receipt = graph_loss(model, example, checkpoint_layers=True, chunk_tokens=1)
    output = model(example.plan, retain_states=False)
    manual = F.cross_entropy(model.logits(output['h_final'][0, [0, 1, 4]]).float(),
                             torch.tensor([4, 5, 8]), reduction='none')
    expected = (manual * torch.tensor([0.1, 1., 1.])).sum() / 2.1
    torch.testing.assert_close(loss, expected)
    assert receipt['supervised_tokens'] == 3
    assert receipt['physical_tokens'] == 6 and receipt['logical_slots'] == 8
    assert receipt['allocated_kv_slots'] == receipt['logical_slots'] + receipt['kv_padding_slots']
    loss.backward()
    assert model.adapter.up.weight.grad is not None
    assert torch.isfinite(model.adapter.up.weight.grad).all()
    assert model.adapter.up.weight.grad.norm() > 0


def test_forced_import_boundary_cannot_be_falsely_supervised():
    bad = row()
    bad['nodes'][1]['target_weights'][1] = 1
    with pytest.raises(ValueError, match='consumer predecessor'):
        parse_graph(bad, tokenizer_fingerprint='f' * 64)


@pytest.mark.parametrize('change', ['tokenizer', 'split', 'weight'])
def test_graph_provenance_and_supervision_are_verified(change):
    bad = copy.deepcopy(row())
    if change == 'tokenizer':
        bad['tokenizer_fingerprint'] = 'a' * 64
    elif change == 'split':
        bad['split'] = 'unknown'
    else:
        bad['nodes'][0]['target_weights'][1] = float('nan')
    with pytest.raises(ValueError):
        parse_graph(bad, tokenizer_fingerprint='f' * 64)


def test_native_gradient_service_returns_exact_named_trainable_tensors(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    from natlang_neuralese.serve.store import TensorStore

    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    store = TensorStore(tmp_path)
    engine = SimpleNamespace(shared_model=model, store=store,
        checkpoint_sha256='1' * 64, embedding_weight_sha256='2' * 64,
        model_config={'architecture': 'lfm2', 'tokenizer_fingerprint': 'f' * 64})
    model.requires_grad_(False)
    reply = native_graph_gradient(engine, {
        'schema': 'natlang.native-gradient-request/1', 'objective': 'graph-ce', 'graph': row(),
        'target': {'kind': 'model-parameters', 'names': ['adapter.up.weight']}}, 'test-owner')
    artifact = store.get_native_tensor_artifact(reply['gradient']['id']).artifact
    assert artifact.kind == 'parameter-gradient'
    assert artifact.native_identity['checkpoint_sha256'] == '1' * 64
    assert artifact.provenance['graph_sha256']
    assert artifact.tensors['adapter.up.weight'].shape == model.adapter.up.weight.shape
    assert torch.isfinite(artifact.tensors['adapter.up.weight']).all()
    assert artifact.tensors['adapter.up.weight'].norm() > 0
    assert not model.adapter.up.weight.requires_grad, 'serving freeze state is restored after gradient capture'
    store.close()


def test_native_gradient_service_rejects_held_rows_and_unselected_parameters():
    from types import SimpleNamespace
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    from natlang_neuralese.serve.chat import RequestError
    from natlang_neuralese.serve.store import TensorStore

    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    store = TensorStore()
    engine = SimpleNamespace(shared_model=model, store=store,
        checkpoint_sha256='1' * 64, embedding_weight_sha256='2' * 64,
        model_config={'architecture': 'lfm2', 'tokenizer_fingerprint': 'f' * 64})
    request = {'schema': 'natlang.native-gradient-request/1', 'graph': row(),
               'target': {'kind': 'model-parameters', 'names': ['not_a_native_parameter']}}
    with pytest.raises(RequestError, match='native model parameters'):
        native_graph_gradient(engine, request, None)
    held = row()
    held['split'] = 'test'
    request['graph'] = held
    request['target'] = {'kind': 'model-parameters', 'names': ['adapter.up.weight']}
    with pytest.raises(RequestError, match='train rows only'):
        native_graph_gradient(engine, request, None)
    store.close()


def test_native_gradient_service_returns_producer_kv_and_hybrid_state_gradients(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.model.state_block import NativeStateBlock
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    from natlang_neuralese.serve.store import TensorStore

    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    identity = {'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': 'f' * 64,
                'embedding_weight_sha256': '2' * 64, 'architecture': 'lfm2'}
    layers = []
    for index, layer in enumerate(model.port.layers):
        if model.port.is_attention(index):
            attention = layer.self_attn
            heads = attention.k_proj.out_features // attention.head_dim
            shape = (2, heads, attention.head_dim)
            layers.append({'kind': 'attention', 'raw_k': torch.randn(shape), 'v': torch.randn(shape)})
        else:
            channels = layer.conv.conv.weight.shape[0]
            layers.append({'kind': 'finite_convolution', 'gated_inputs': torch.randn(2, channels)})
    state = NativeStateBlock(text='source', token_ids=torch.tensor([9, 10]), positions=torch.tensor([0, 1]),
        embeddings=torch.randn(2, model.port.embedding_weight.shape[1]), layers=tuple(layers), **identity).validate()
    store = TensorStore(tmp_path)
    stored = store.put_native_state(state, 'test-owner')
    graph = row()
    graph['external_states'] = {'prior': {'id': stored.id, 'token_count': 2}}
    graph['nodes'][1]['imports'] = [{'at': 1, 'source': 'prior', 'start': 0, 'stop': 2}]
    engine = SimpleNamespace(shared_model=model, store=store, checkpoint_sha256=identity['checkpoint_sha256'],
        embedding_weight_sha256=identity['embedding_weight_sha256'], model_config={
            'architecture': identity['architecture'], 'tokenizer_fingerprint': identity['tokenizer_fingerprint']})
    reply = native_graph_gradient(engine, {'schema': 'natlang.native-gradient-request/1', 'graph': graph,
        'target': {'kind': 'native-state', 'id': stored.id}}, 'test-owner')
    gradient = store.get_native_tensor_artifact(reply['gradient']['id']).artifact
    assert gradient.kind == 'state-gradient' and gradient.source_artifact_id == stored.id
    assert 'layers.1.raw_k' in gradient.tensors and 'layers.1.v' in gradient.tensors
    assert 'layers.0.gated_inputs' in gradient.tensors and 'embeddings' not in gradient.tensors
    assert all(torch.isfinite(value).all() for value in gradient.tensors.values())
    assert sum(value.float().norm() for value in gradient.tensors.values()) > 0
    store.close()


def test_native_policy_gradient_scores_imported_kv_and_finite_convolution(tmp_path):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
    from natlang_neuralese.model.state_block import NativeStateBlock
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    from natlang_neuralese.serve.store import TensorStore, native_state_id, encode_native_state
    from natlang_neuralese.rl.native_replay import graph_identity
    from natlang_neuralese.data.native_graph import parse_graph

    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    identity = {'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': 'f' * 64,
                'embedding_weight_sha256': '2' * 64, 'architecture': 'lfm2'}
    layers = []
    for index, layer in enumerate(model.port.layers):
        if model.port.is_attention(index):
            attention = layer.self_attn
            heads = attention.k_proj.out_features // attention.head_dim
            layers.append({'kind': 'attention', 'raw_k': torch.randn(2, heads, attention.head_dim),
                           'v': torch.randn(2, heads, attention.head_dim)})
        else:
            layers.append({'kind': 'finite_convolution', 'gated_inputs': torch.randn(2, layer.conv.conv.weight.shape[0])})
    block = NativeStateBlock(text='prior', token_ids=torch.tensor([9, 10]), positions=torch.tensor([0, 1]),
        embeddings=torch.randn(2, model.port.embedding_weight.shape[1]), layers=tuple(layers), **identity).validate()
    state_id = native_state_id(block)
    graph = {'schema': SCHEMA, 'id': 'native-policy-import-score', 'split': 'train',
        'source_groups': ['policy-import-test'], 'tokenizer_fingerprint': identity['tokenizer_fingerprint'],
        'external_states': {'prior': {'id': state_id, 'token_count': 2}},
        'nodes': [{'name': 'consumer', 'token_ids': [6, 7, 8],
                   'imports': [{'at': 0, 'source': 'prior', 'start': 0, 'stop': 2}],
                   'target_weights': [0, 1, 0]}]}
    example = parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint'])
    store = TensorStore(tmp_path / 'tensor-store')
    store.put_native_state(block, 'test-owner')
    bundle_dir = tmp_path / 'blocks'
    bundle_dir.mkdir()
    block_file = bundle_dir / f'{state_id}.safetensors'
    block_file.write_bytes(encode_native_state(block))
    bundle = bundle_dir / 'bundle.json'
    bundle.write_text(json.dumps({'schema': 'natlang.rl-block-bundle/1', 'policy_id': identity['checkpoint_sha256'],
        'blocks': {state_id: {'path': block_file.name, 'bytes': block_file.stat().st_size,
                              'sha256': sha256_file_hex_cached(block_file)}}}))
    spec = tmp_path / 'objective.json'
    spec.write_text(json.dumps({'schema': 'natlang.native-policy-objective/1', 'kind': 'native-policy',
        'mode': 'rft', 'policy': identity, 'block_bundle': str(bundle),
        'rows': {example.id: {'graph_sha256': canonical_json_sha256_hex(graph_identity(example)),
            'actions': [{'query': 0, 'token_id': 7, 'behavior_logp': -2.0,
                         'temperature': 1.0, 'allowed_token_ids': None}], 'advantage': 1.0}}}))
    engine = SimpleNamespace(shared_model=model, store=store, checkpoint_sha256=identity['checkpoint_sha256'],
        embedding_weight_sha256=identity['embedding_weight_sha256'], model_config={
            'architecture': identity['architecture'], 'tokenizer_fingerprint': identity['tokenizer_fingerprint']})
    reply = native_graph_gradient(engine, {'schema': 'natlang.native-gradient-request/1',
        'objective': {'kind': 'native-policy', 'path': str(spec), 'sha256': sha256_file_hex_cached(spec)},
        'native_identity': identity, 'graph': graph,
        'target': {'kind': 'native-state', 'id': state_id,
                   'names': ['layers.0.gated_inputs', 'layers.1.raw_k', 'layers.1.v']}}, 'test-owner')
    gradient = store.get_native_tensor_artifact(reply['gradient']['id']).artifact
    assert reply['objective'] == 'native-policy-objective/1'
    assert reply['stats']['sampled_tokens'] == 1
    assert gradient.kind == 'state-gradient' and gradient.source_artifact_id == state_id
    assert all(torch.isfinite(value).all() for value in gradient.tensors.values())
    assert all(value.norm() > 0 for value in gradient.tensors.values())
    store.close()
