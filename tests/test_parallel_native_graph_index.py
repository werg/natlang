import json

import pytest

from natlang_neuralese.data.native_sources import NativeGraphSource
from scripts.neuralese_data.parallel_native_graph_index import parallel_rows


def graph(ident, size):
    return {'schema': 'natlang.neuralese-native-graph/1', 'id': ident, 'split': 'train',
        'source_groups': ['world:' + ident], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [{'name': 'text', 'token_ids': [2] * size, 'target_weights': [0] + [1] * (size - 1)}]}


def test_parallel_rows_match_real_serial_parser_in_exact_order_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'index'))
    path = tmp_path / 'graphs.jsonl'
    path.write_text(''.join(json.dumps(graph(f'row-{n}', 11 + n * 7)) + '\n' for n in range(7)))
    serial = NativeGraphSource(path, 'f' * 64, required_split='train', role_ids=[1, 2, 3, 4])
    parallel = list(parallel_rows(path, 'f' * 64, [1, 2, 3, 4], workers=2, max_pending=4))
    assert [item[0] for item in parallel] == serial.entries
    assert [item[1] for item in parallel] == [len(example.plan.slot_sources) for example in serial]
    assert all(item[2] == 0 for item in parallel)


def test_parallel_rejects_duplicate_and_invalid_graphs(tmp_path):
    path = tmp_path / 'graphs.jsonl'
    path.write_text('\n'.join(json.dumps(graph('same', 11)) for _ in range(2)) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        list(parallel_rows(path, 'f' * 64, [], workers=2))
