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


def test_parallel_preparation_seals_the_exact_standard_index_and_geometry(tmp_path, monkeypatch):
    from scripts.neuralese_data.build_acquired_streaming_successor import bind, prepare_parallel_leaf, read
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'index'))
    path = tmp_path / 'graphs.jsonl'
    path.write_text(''.join(json.dumps(graph(f'row-{n}', 10 + n)) + '\n' for n in range(5)))
    reference = {**bind(path), 'rows': 5}
    observed, proof_ref = prepare_parallel_leaf(path, reference, 'f' * 64, [1, 2, 3, 4], {'prior'}, tmp_path, workers=2)
    proof = read(proof_ref['path'])
    assert proof['decision_readouts'] == 0 and proof['rows'] == 5
    # This must be the regular production NativeGraphSource cache, not a new
    # simplified parser or an alternate row-binding dialect.
    source = NativeGraphSource(path, 'f' * 64, required_split='train', role_ids=[1, 2, 3, 4])
    assert [observed[(entry['id'], entry['binding'])] for entry in source.entries] == [10, 11, 12, 13, 14]
    assert source.file_sha256 == reference['sha256']
    broken = graph('bad', 11); broken['nodes'][0]['target_weights'][0] = 1
    path.write_text(json.dumps(broken) + '\n')
    with pytest.raises(ValueError, match='first own token'):
        list(parallel_rows(path, 'f' * 64, [], workers=2))
