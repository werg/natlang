import copy
import hashlib
import json

import pytest

from natlang_neuralese.data.native_graph_quality import (
    mask_exact_target_spans, withdraw_exact_graphs,
)
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.native_sources import _row_bindings


FP = 'f' * 64


def graph(identifier='g1'):
    return {
        'schema': 'natlang.neuralese-native-graph/1', 'id': identifier,
        'split': 'train', 'source_groups': ['source:fixture'], 'tokenizer_fingerprint': FP,
        'nodes': [
            {'name': 'writer', 'token_ids': [3, 4, 5], 'producing': [[1, 3]],
             'target_weights': [0, 1, 1]},
            {'name': 'reader', 'token_ids': [6, 7, 8],
             'imports': [{'at': 1, 'source': 'writer', 'start': 1, 'stop': 3}],
             'target_weights': [0, 0, 1]},
            {'name': 'action', 'token_ids': [9, 10, 11],
             'target_weights': [0, 1, 1]},
        ],
    }


def binding(row):
    parsed = parse_graph(row, tokenizer_fingerprint=FP)
    return _row_bindings([parsed])[row['id']]


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_exact_action_mask_preserves_complete_graph_and_consumer_supervision():
    source = graph()
    changed, receipt = mask_exact_target_spans(source, tokenizer_fingerprint=FP,
        expected_id='g1', expected_binding=binding(source), spans=[{
            'node': 'action', 'token_start': 1, 'token_stop': 3,
            'expected_positive_tokens': 2, 'expected_token_ids_sha256': token_hash([10, 11]),
        }])
    parsed = parse_graph(changed, tokenizer_fingerprint=FP)
    assert receipt['binding_before'] == binding(source)
    assert receipt['binding_after'] == binding(changed)
    assert receipt['nodes'] == 3 and receipt['forward_tokens'] == 9
    assert changed['nodes'][2]['target_weights'] == [0, 0.0, 0.0]
    assert changed['nodes'][0]['target_weights'] == source['nodes'][0]['target_weights']
    assert changed['nodes'][1]['target_weights'] == source['nodes'][1]['target_weights']
    assert parsed.plan.invocations[1].imports[0].source == 'writer'
    assert sum(w > 0 for w in parsed.target_weights) == 3


@pytest.mark.parametrize('tamper', ['binding', 'tokens', 'count'])
def test_exact_action_mask_rejects_stale_review(tamper):
    source = graph()
    expected_binding = binding(source)
    span = {'node': 'action', 'token_start': 1, 'token_stop': 3,
            'expected_positive_tokens': 2, 'expected_token_ids_sha256': token_hash([10, 11])}
    if tamper == 'binding':
        expected_binding = '0' * 64
    elif tamper == 'tokens':
        span['expected_token_ids_sha256'] = token_hash([100, 101])
    else:
        span['expected_positive_tokens'] = 1
    with pytest.raises(ValueError):
        mask_exact_target_spans(source, tokenizer_fingerprint=FP, expected_id='g1',
                                expected_binding=expected_binding, spans=[span])
    assert source['nodes'][2]['target_weights'] == [0, 1, 1]


def test_withdraw_exact_graph_removes_only_reviewed_complete_train_graph():
    rows = [graph('keep-a'), graph('remove'), graph('keep-b')]
    kept, receipt = withdraw_exact_graphs(rows, tokenizer_fingerprint=FP,
        withdrawals={'remove': binding(rows[1])})
    assert [row['id'] for row in kept] == ['keep-a', 'keep-b']
    assert receipt['removed'][0]['id'] == 'remove'
    assert receipt['removed'][0]['node_count'] == 3
    assert receipt['retained_rows'] == 2


def test_withdraw_exact_graph_is_fail_closed_for_duplicate_or_changed_identity():
    source = graph('remove')
    with pytest.raises(ValueError, match='more than once'):
        withdraw_exact_graphs([source, copy.deepcopy(source)], tokenizer_fingerprint=FP,
                              withdrawals={'remove': binding(source)})
    changed = copy.deepcopy(source)
    changed['nodes'][2]['target_weights'][2] = 0
    with pytest.raises(ValueError, match='differs from reviewed binding'):
        withdraw_exact_graphs([changed], tokenizer_fingerprint=FP,
                              withdrawals={'remove': binding(source)})

