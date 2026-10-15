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
               'target': {'kind': 'model-parameters', 'names': ['port.layers.0.weight']}}
    with pytest.raises(RequestError, match='trainable native model parameters'):
        native_graph_gradient(engine, request, None)
    held = row()
    held['split'] = 'test'
    request['graph'] = held
    request['target'] = {'kind': 'model-parameters', 'names': ['adapter.up.weight']}
    with pytest.raises(RequestError, match='train rows only'):
        native_graph_gradient(engine, request, None)
    store.close()
