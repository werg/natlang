import json
from pathlib import Path

import pytest

from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from scripts.coreweave.teacher_tranche_admission import file_binding
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import registered_cached_content_index
from scripts.coreweave.teacher_internal_conversion import _same_actor_boundary_inputs, history_policy_args
from scripts.coreweave.teacher_tranche_admission import Tranche


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return file_binding(path)


def test_registered_unavailable_source_uses_only_exact_current_law_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index'))
    ident = 'qwen36-verified-positive-native-graphs-20261022-recovery7853-batch0001-tranche0000'
    graph = {'path': 'lfm25/train/native-graphs.jsonl.zst', 'sha256': 'a' * 64, 'bytes': 123}
    entry_path = 'runs/' + ident
    _write(tmp_path / 'training/neuralese_corpora.json', {
        'corpora': [{'id': ident, 'path': entry_path, 'training_admission': True}],
    })
    _write(tmp_path / 'training/corpus-manifests' / (ident + '.json'), {
        'id': ident, 'path': entry_path, 'files': [graph],
    })
    fingerprint, policy = 'b' * 64, 'c' * 64
    key = {'source_sha256': graph['sha256'], 'tokenizer_fingerprint': fingerprint,
           'policy_sha256': policy, 'kind': 'teacher-content-fingerprints-v1'}
    contents = {'d' * 64: 'exact-native-graph-row'}
    cache = row_index.entry_path('teacher-content', canonical_json_sha256_hex(key))
    row_index.save(cache, {'key': key, 'rows': 4, 'fingerprints': contents,
                           'metadata_sha256': canonical_json_sha256_hex({'rows': 4, 'fingerprints': contents})})

    item, actual = registered_cached_content_index(tmp_path, ident, 'lfm25', fingerprint, policy)
    assert actual == contents
    assert item['sha256'] == graph['sha256']
    assert item['bytes'] == graph['bytes']
    assert item['rows'] == 4
    assert item['cache_only_source'] is True

    with pytest.raises(ValueError, match='no valid exact current-law content index'):
        registered_cached_content_index(tmp_path, ident, 'lfm25', fingerprint, 'e' * 64)

    source = tmp_path / entry_path / graph['path']
    source.parent.mkdir(parents=True)
    source.write_bytes(b'now-present')
    with pytest.raises(ValueError, match='must be read directly'):
        registered_cached_content_index(tmp_path, ident, 'lfm25', fingerprint, policy)


def test_same_source_boundary_is_reused_across_registry_namespaces_but_actor_is_exact(tmp_path):
    source_sha, actor_sha, runtime_sha = '1' * 64, '2' * 64, '3' * 64
    entries, manifests = [], tmp_path / 'training/corpus-manifests'
    for ident, raw_sha, source_actor in (
        ('old-qwen36-batch0041-tranche0000', source_sha, actor_sha),
        ('other-actor-batch0041-tranche0000', source_sha, '4' * 64),
        ('same-actor-other-source-batch0041-tranche0000', '5' * 64, actor_sha),
    ):
        path = f'runs/{ident}'
        files = []
        for model in ('lfm25', 'mellum21'):
            graph_path = tmp_path / path / model / 'train/native-graphs.jsonl.zst'
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_bytes((ident + model).encode())
            receipt_path = graph_path.parents[1] / 'receipt.json'
            receipt_path.write_text(json.dumps({'status': 'passed', 'model_key': model,
                'source': {'sha256': raw_sha},
                'explicit_history_reasoning_policy': 'keep' if model == 'lfm25' else None,
                'failed_assistant_history_policy': 'mask-runtime-failures'}))
            files.extend([{'path': f'{model}/train/native-graphs.jsonl.zst',
                           'sha256': file_binding(graph_path)['sha256'], 'bytes': graph_path.stat().st_size},
                          {'path': f'{model}/receipt.json', 'sha256': file_binding(receipt_path)['sha256'],
                           'bytes': receipt_path.stat().st_size}])
        entries.append({'id': ident, 'path': path, 'training_admission': True,
                        'source_binding': {'raw_source': {'sha256': raw_sha},
                            'actor': {'sha256': source_actor}, 'execution_identity': {'sha256': runtime_sha}}})
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / f'{ident}.json').write_text(json.dumps({'id': ident, 'path': path, 'files': files}))
    registry = tmp_path / 'training/neuralese_corpora.json'
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps({'corpora': entries}))
    current = {'source': {'sha256': source_sha}, 'actor': {'sha256': actor_sha},
               'execution_identity': {'sha256': runtime_sha}}
    selected = _same_actor_boundary_inputs(tmp_path,
        Tranche(41, 0, teacher_namespace='current-qwen36'), current,
        {'lfm25': [], 'mellum21': []})
    assert {row['corpus_id'] for row in selected} == {'old-qwen36-batch0041-tranche0000'}
    assert {row['model'] for row in selected} == {'lfm25', 'mellum21'}


def test_lfm_keeps_declared_history_mode_while_masking_failed_history_for_both_models():
    lfm = history_policy_args('lfm25')
    mellum = history_policy_args('mellum21')
    assert lfm == ['--failed-assistant-history-policy', 'mask-runtime-failures',
                   '--history-reasoning-policy', 'keep']
    assert mellum == ['--failed-assistant-history-policy', 'mask-runtime-failures']


def test_partial_old_content_keeps_only_numeric_new_delta(tmp_path):
    from natlang_neuralese.data.native_graph import SCHEMA
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import graph_content_sha256
    old = {'schema': SCHEMA, 'id': 'old', 'split': 'train', 'source_groups': ['train'],
           'tokenizer_fingerprint': 'tok', 'nodes': [{'name': 'text', 'token_ids': [1, 2],
           'target_weights': [0., 1.]}]}
    equal_content = {**old, 'id': 'relocated-copy'}
    numeric_delta = json.loads(json.dumps(old))
    numeric_delta['id'] = 'new-policy-target'
    numeric_delta['nodes'][0]['target_weights'][1] = .25
    seen = {graph_content_sha256(old)}
    candidates = [equal_content, numeric_delta]
    admitted = [row for row in candidates if graph_content_sha256(row) not in seen]
    assert [row['id'] for row in admitted] == ['new-policy-target']
