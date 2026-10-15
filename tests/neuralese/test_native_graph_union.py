import hashlib
import json
import subprocess

import pytest

from scripts.neuralese_data.assemble_native_graph_union import assemble


MODELS = {'lfm25': 'lfm2:' + 'a' * 64, 'mellum21': 'mellum:' + 'b' * 64}
FINGERPRINTS = {'lfm25': 'c' * 64, 'mellum21': 'd' * 64}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zst(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode()
    path.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=text, check=True,
                                    stdout=subprocess.PIPE).stdout)


def _graph(identity, split, groups, fingerprint):
    return {'schema': 'natlang.neuralese-native-graph/1', 'id': identity, 'split': split,
            'source_groups': groups, 'tokenizer_fingerprint': fingerprint,
            'nodes': [{'name': 'text', 'token_ids': [1, 2], 'target_weights': [0, 1]}]}


def _builds(root, *, overlap=False):
    full_root, rest_root = root / 'full', root / 'rest'
    manifest_path = root / 'training' / 'corpus-manifests' / 'full.json'
    base_models, rest_outputs, child_paths = {}, {}, {}
    for alias in ('lfm25', 'mellum21'):
        outputs, child_outputs, rest_splits = {}, [], {}
        for split in ('train', 'test'):
            old_group = 'held-shared' if split == 'test' else 'full-train'
            new_group = 'held-shared' if overlap and split == 'train' else ('rest-held' if split == 'test' else 'rest-train')
            old_path = full_root / alias / split / 'native-graphs.jsonl.zst'
            new_path = rest_root / alias / split / 'native-graphs.jsonl.zst'
            old = [_graph('full-' + split, split, [old_group], FINGERPRINTS[alias])]
            new = [_graph('rest-' + split, split, [new_group], FINGERPRINTS[alias])]
            _write_zst(old_path, old); _write_zst(new_path, new)
            child_outputs.append({'path': str(old_path), 'rows': 1, 'sha256': _sha(old_path), 'split': split})
            rest_splits[split] = {'path': str(new_path), 'rows': 1, 'sha256': _sha(new_path)}
        child_path = full_root / alias / 'receipt.json'
        child_path.parent.mkdir(parents=True, exist_ok=True)
        child_path.write_text(json.dumps({'expanded_tokenizer_fingerprint': FINGERPRINTS[alias],
                                          'outputs': child_outputs}))
        base_models[alias] = {'model_id': MODELS[alias], 'child_receipt': f'{alias}/receipt.json'}
        rest_outputs[alias] = rest_splits
    full_receipt = full_root / 'receipt.json'
    full_receipt.write_text(json.dumps({'schema': 'natlang.native-graph-full-build/1',
        'id': 'fixture-full', 'corpus_manifest': str(manifest_path), 'models': base_models}))
    manifest_files = []
    for path in sorted(full_root.rglob('*')):
        if path.is_file():
            manifest_files.append({'path': str(path.relative_to(full_root)),
                                   'bytes': path.stat().st_size, 'sha256': _sha(path)})
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({'schema': 'natlang.corpus-snapshot/1',
        'id': 'fixture-full', 'path': 'full', 'files': manifest_files}))
    rest_receipt = rest_root / 'receipt.json'
    rest_receipt.write_text(json.dumps({'schema': 'natlang.native-semantic-e2e-build/2',
        'model_ids': MODELS, 'tokenizers': {alias: {'expanded': FINGERPRINTS[alias]} for alias in MODELS},
        'outputs': {'graphs': rest_outputs}}))
    return full_receipt, rest_receipt


def test_native_graph_union_keeps_full_prefix_and_binds_both_splits(tmp_path):
    full_receipt, rest_receipt = _builds(tmp_path)
    result = assemble(full_receipt, rest_receipt, tmp_path / 'union')
    assert result['counts']['train_held_group_intersection'] == 0
    assert result['counts']['per_split'] == {'train': 4, 'test': 4}
    for alias in MODELS:
        for split in ('train', 'test'):
            output = result['models'][alias][split]['path']
            rows = subprocess.run(['zstd', '-dcq', output], check=True,
                                  stdout=subprocess.PIPE).stdout.decode().splitlines()
            assert [json.loads(row)['id'] for row in rows] == ['full-' + split, 'rest-' + split]
            assert result['models'][alias][split]['full_prefix_rows'] == 1
            assert result['models'][alias][split]['appended_rows'] == 1


def test_native_graph_union_rejects_protected_group_crossing(tmp_path):
    full_receipt, rest_receipt = _builds(tmp_path, overlap=True)
    with pytest.raises(ValueError, match='source groups cross train/held'):
        assemble(full_receipt, rest_receipt, tmp_path / 'union')


def test_native_graph_union_rejects_unmanifested_full_receipt_change(tmp_path):
    full_receipt, rest_receipt = _builds(tmp_path)
    full_receipt.write_text(full_receipt.read_text().replace('fixture-full', 'changed-full'))
    with pytest.raises(ValueError, match='FULL corpus manifest does not bind'):
        assemble(full_receipt, rest_receipt, tmp_path / 'union')
