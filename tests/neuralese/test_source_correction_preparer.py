import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/prepare_source_correction_continuation.py'
SPEC = importlib.util.spec_from_file_location('source_correction_preparer', SCRIPT)
PREPARER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARER)


def _entry(path):
    payload = Path(path).read_bytes()
    return {'source_binding': {'models': {'lfm25': {
        'model_id': 'lfm2:model', 'tokenizer_fingerprint': 'a' * 64,
        'train': {'path': str(path), 'bytes': len(payload),
                  'sha256': hashlib.sha256(payload).hexdigest()}}}}}


def test_registered_package_role_binds_exact_model_split_bytes(tmp_path):
    graph = tmp_path / 'native-graphs.jsonl.zst'
    graph.write_bytes(b'authenticated package leaf')
    PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
        model_id='lfm2:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
            model_id='mellum:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
            model_id='lfm2:model', fingerprint='b' * 64)
    damaged = _entry(graph)
    damaged['source_binding']['models']['lfm25']['train']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='registered model binding'):
        PREPARER._validate_registered_role(damaged, 'lfm25', 'train', graph,
            model_id='lfm2:model', fingerprint='a' * 64)
    with pytest.raises(ValueError, match='no exact mellum21 source binding'):
        PREPARER._validate_registered_role(_entry(graph), 'mellum21', 'train', graph,
            model_id='lfm2:model', fingerprint='a' * 64)


def test_registered_package_may_use_an_authenticated_external_storage_symlink(tmp_path):
    external = tmp_path / 'external' / 'native-graphs.jsonl.zst'
    external.parent.mkdir()
    external.write_bytes(b'authenticated package leaf')
    package = tmp_path / 'package'
    package.mkdir()
    graph = package / 'native-graphs.jsonl.zst'
    graph.symlink_to(external)
    PREPARER._validate_registered_role(_entry(graph), 'lfm25', 'train', graph,
        model_id='lfm2:model', fingerprint='a' * 64)


def test_flattened_nested_manifest_keeps_resolved_relative_leaf_path(tmp_path):
    import json

    leaf = tmp_path / 'nested' / 'fit.jsonl.zst'
    leaf.parent.mkdir()
    leaf.write_bytes(b'authenticated relative leaf')
    child = leaf.parent / 'child.json'
    child.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': leaf.name,
         'sha256': hashlib.sha256(leaf.read_bytes()).hexdigest()}]}))
    root = tmp_path / 'root.json'
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': str(child),
         'sha256': hashlib.sha256(child.read_bytes()).hexdigest()}]}))

    flattened = PREPARER.flatten(root, {})
    assert flattened == [{'kind': 'native-graphs', 'path': str(leaf.resolve()),
                          'sha256': hashlib.sha256(leaf.read_bytes()).hexdigest()}]


def test_flattened_nested_manifest_preserves_only_bound_cloud_logical_path(tmp_path):
    import json

    leaf = tmp_path / 'staged' / 'fit.jsonl.zst'
    leaf.parent.mkdir()
    leaf.write_bytes(b'cloud staged relative leaf')
    child = leaf.parent / 'child.json'
    child.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': 'fit.jsonl.zst',
         'sha256': hashlib.sha256(leaf.read_bytes()).hexdigest()}]}))
    root = tmp_path / 'root.json'
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '/work/old/child.json',
         'sha256': hashlib.sha256(child.read_bytes()).hexdigest()}]}))
    mapping = {'/work/old/child.json': str(child),
               '/work/old/fit.jsonl.zst': str(leaf)}

    flattened = PREPARER.flatten(root, mapping)
    assert flattened[0]['path'] == '/work/old/fit.jsonl.zst'


def test_mellum_missing_cloud_leaf_relocates_only_through_exact_registry_role(tmp_path):
    import json

    corpus = ('qwen36-verified-positive-native-graphs-20261023-internal-batch0047-tranche0000'
              '-history-policy-keep-repair-v1')
    registry = tmp_path / 'registry'
    package = registry / corpus
    graph = package / 'mellum21' / 'train' / 'native-graphs.jsonl.zst'
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b'registered cloud package graph')
    receipt_path = package / 'mellum21' / 'receipt.json'
    receipt_path.write_text('{}')
    model_id, fingerprint = 'mellum:model', 'b' * 64
    entry = {'id': corpus, 'source_binding': {'models': {'mellum21': {
        'model_id': model_id, 'tokenizer_fingerprint': fingerprint,
        'train': {'path': str(graph), 'bytes': graph.stat().st_size,
                  'sha256': hashlib.sha256(graph.read_bytes()).hexdigest()},
        'receipt': {'path': str(receipt_path), 'bytes': receipt_path.stat().st_size,
                    'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()}}}}}
    (registry / f'{corpus}.entry.json').write_text(json.dumps(entry))
    logical = (f'/work/resume/teacher-append-20261025/artifacts/runs/{corpus}'
               '/mellum21/train/native-graphs.jsonl.zst')
    train = tmp_path / 'train.json'
    train.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': logical,
         'sha256': hashlib.sha256(graph.read_bytes()).hexdigest()}]}))
    held = tmp_path / 'held.json'
    held.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': []}))
    mapping = {}
    cfg = {'model_id': model_id, 'fingerprint': fingerprint}

    proof = PREPARER._relocate_declared_mellum_package_leaves(
        train, held, mapping, cfg=cfg, registry_root=registry)
    assert mapping[logical] == str(graph)
    assert proof[0]['source_sha256'] == hashlib.sha256(graph.read_bytes()).hexdigest()
    assert proof[0]['registry_entry_sha256'] == hashlib.sha256(
        (registry / f'{corpus}.entry.json').read_bytes()).hexdigest()

    bad_train = tmp_path / 'bad-train.json'
    bad_train.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': logical, 'sha256': '0' * 64}]}))
    with pytest.raises(ValueError, match='registry member differs'):
        PREPARER._relocate_declared_mellum_package_leaves(
            bad_train, held, {}, cfg=cfg, registry_root=registry)
