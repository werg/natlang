import hashlib
import json
import subprocess

import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from scripts.neuralese_data.assemble_native_semantic_pair_union import assemble


MODELS = ['lfm2:' + 'a' * 64, 'mellum:' + 'b' * 64]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _zst(path, rows):
    text = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=text, check=True,
                                    stdout=subprocess.PIPE).stdout)


def _source_edge(split, serial):
    return {'split': split, 'writer_split': split, 'trajectory': 't' + str(serial),
            'writer_record_id': 'w' + str(serial), 'reader_record_id': 'r' + str(serial)}


def _pair(edge, split, group):
    edge_sha = hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    graph = {'schema': 'natlang.neuralese-native-graph/1', 'split': split, 'source_groups': [group]}
    payload = 'payload-' + edge_sha[:6]
    payload_sha = hashlib.sha256(payload.encode()).hexdigest()
    communication = {'source_id': edge_sha, 'writer_record_id': edge['writer_record_id'],
        'reader_record_id': edge['reader_record_id'], 'write_name': 'result', 'occurrence': '/value',
        'source_binding': {'records': {'sha256': 'd' * 64}}}
    communication_sha = canonical_json_sha256_hex(communication)
    return {'schema': 'natlang.native-semantic-pair/1', 'id': 'e2e-edge:' + edge_sha, 'split': split,
            'source_groups': [group], 'payload_sha256': payload_sha,
            'communication': communication, 'communication_sha256': communication_sha,
            'models': {name: {'graph': graph, 'witness': {'source_id': edge_sha,
                'communication_sha256': communication_sha, 'payload_sha256': payload_sha},
                'export': {'text': payload}} for name in MODELS}}


def _receipt(path, *, sources, outputs, omission_path, repair=None):
    value = {'schema': 'natlang.native-semantic-e2e-build/2', 'source': sources,
             'model_ids': {'lfm25': MODELS[0], 'mellum21': MODELS[1]},
             'outputs': {'pairs': outputs},
             'pair_omissions': {'path': str(omission_path), 'sha256': _sha(omission_path)}}
    if repair is not None:
        value['pair_repair'] = repair
    path.write_text(json.dumps(value))
    return path


def test_pair_union_closes_train_and_held_edges_with_concatenated_frames(tmp_path):
    edges = [_source_edge('train', 1), _source_edge('test', 2)]
    edges_path = tmp_path / 'edges.jsonl'
    edges_path.write_text(''.join(json.dumps(row) + '\n' for row in edges))
    records, pieces = tmp_path / 'records.jsonl', tmp_path / 'pieces.jsonl'
    records.write_text('record binding\n'); pieces.write_text('piece binding\n')
    sources = {name: {'path': str(path), 'sha256': _sha(path)} for name, path in
               [('records', records), ('pieces', pieces), ('edges', edges_path)]}

    base_out, repair_out = tmp_path / 'base', tmp_path / 'repair'
    base_outputs, repair_outputs = {}, {}
    for split, edge, group in [('train', edges[0], 'train-group'), ('test', edges[1], 'held-group')]:
        base_path = base_out / split / 'pairs.jsonl.zst'
        repair_path = repair_out / split / 'pairs.jsonl.zst'
        _zst(base_path, [_pair(edge, split, group)] if split == 'train' else [])
        _zst(repair_path, [_pair(edge, split, group)] if split == 'test' else [])
        base_outputs[split] = {'path': str(base_path), 'sha256': _sha(base_path), 'rows': int(split == 'train')}
        repair_outputs[split] = {'path': str(repair_path), 'sha256': _sha(repair_path), 'rows': int(split == 'test')}
    empty = tmp_path / 'empty-omissions.jsonl'; empty.write_text('')
    base_omissions = tmp_path / 'base-omissions.jsonl'; base_omissions.write_text('')
    base = _receipt(tmp_path / 'base-receipt.json', sources=sources, outputs=base_outputs,
                    omission_path=base_omissions)
    repair = _receipt(tmp_path / 'repair-receipt.json', sources=sources, outputs=repair_outputs,
        omission_path=empty, repair={'mode': 'pairs-only-static-multiread',
                                     'source_omissions_sha256': _sha(base_omissions)})

    receipt = assemble(base, [repair], edges_path, tmp_path / 'union')
    assert receipt['counts'] == {'pairs': {'train': 1, 'test': 1},
        'unresolved_omissions': {'train': 0, 'test': 0}, 'source_edges': 2, 'closed': True}
    for split in ('train', 'test'):
        decoded = subprocess.run(['zstd', '-dcq', receipt['outputs'][split]['path']], check=True,
                                  stdout=subprocess.PIPE).stdout.decode().splitlines()
        assert len(decoded) == 1


def test_pair_union_refuses_an_unaccounted_source_edge(tmp_path):
    edges_path = tmp_path / 'edges.jsonl'
    edges_path.write_text(json.dumps(_source_edge('train', 9)) + '\n')
    records, pieces = tmp_path / 'records.jsonl', tmp_path / 'pieces.jsonl'
    records.write_text('record'); pieces.write_text('piece')
    sources = {name: {'path': str(path), 'sha256': _sha(path)} for name, path in
               [('records', records), ('pieces', pieces), ('edges', edges_path)]}
    output = tmp_path / 'empty.zst'; _zst(output, [])
    omission = tmp_path / 'omissions.jsonl'; omission.write_text('')
    outputs = {split: {'path': str(output), 'sha256': _sha(output), 'rows': 0} for split in ('train', 'test')}
    base = _receipt(tmp_path / 'receipt.json', sources=sources, outputs=outputs, omission_path=omission)
    with pytest.raises(ValueError, match='pair/omission closure'):
        assemble(base, [], edges_path, tmp_path / 'union')
