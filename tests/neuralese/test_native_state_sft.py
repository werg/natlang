from types import SimpleNamespace

import pytest

from natlang_neuralese.data.native_graph import parse_graph, SCHEMA
from natlang_neuralese.train.native_state_sft import append_only_rows, scheduled_indices, _row_bindings


def graph(identifier, weights=None):
    return {'schema': SCHEMA, 'id': identifier, 'split': 'train', 'source_groups': ['source-' + identifier],
            'tokenizer_fingerprint': 'tokenizer-sha', 'nodes': [
                {'name': 'answer', 'token_ids': [1, 2, 3],
                 'target_weights': weights or [0, 1, 1]}]}


def test_native_data_continuation_only_adds_exact_rows():
    old = [parse_graph(graph('a'), tokenizer_fingerprint='tokenizer-sha'),
           parse_graph(graph('b'), tokenizer_fingerprint='tokenizer-sha')]
    expanded = old + [parse_graph(graph('c'), tokenizer_fingerprint='tokenizer-sha')]
    prior, current = _row_bindings(old), _row_bindings(expanded)
    assert append_only_rows(prior, current, label='training-source') == {'c'}
    with pytest.raises(ValueError, match='modified/removed'):
        append_only_rows(prior, _row_bindings([old[0]]), label='training-source')
    changed = parse_graph(graph('a', [0, .5, 1]), tokenizer_fingerprint='tokenizer-sha')
    with pytest.raises(ValueError, match='modified/removed'):
        append_only_rows(prior, _row_bindings([changed, old[1]]), label='training-source')


def test_append_schedule_preserves_resumed_epoch_prefix_and_deterministic_next_epoch():
    rows = [SimpleNamespace(id=key) for key in ('old-a', 'old-b', 'new-a', 'new-b')]
    old_ids = {'old-a', 'old-b'}
    previous = scheduled_indices(rows[:2], seed=17, epoch=3)
    continued = scheduled_indices(rows, seed=17, epoch=3, prior_ids=old_ids)
    assert [rows[index].id for index in continued[:2]] == [rows[index].id for index in previous]
    assert {rows[index].id for index in continued[2:]} == {'new-a', 'new-b'}
    assert scheduled_indices(rows, seed=17, epoch=4) == scheduled_indices(rows, seed=17, epoch=4)


def test_row_binding_includes_producer_and_target_mask_semantics():
    row = graph('producer')
    row['nodes'][0]['producing'] = [[1, 3]]
    producer = parse_graph(row, tokenizer_fingerprint='tokenizer-sha')
    altered = graph('producer')
    altered['nodes'][0]['producing'] = [[1, 2]]
    assert _row_bindings([producer]) != _row_bindings([
        parse_graph(altered, tokenizer_fingerprint='tokenizer-sha')])
