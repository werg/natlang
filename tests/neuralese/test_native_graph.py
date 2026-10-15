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
