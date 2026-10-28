import hashlib
from pathlib import Path

import pytest

from natlang_neuralese.common import hdd_guard


@pytest.fixture
def hdd(tmp_path, monkeypatch):
    prefix = tmp_path / 'external'
    prefix.mkdir()
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(prefix))
    monkeypatch.setenv('NATLANG_STAGED_INPUTS', str(tmp_path / 'staged'))
    hdd_guard.RECORDS.clear()
    return prefix


def test_nvme_input_untouched_and_hdd_input_staged_with_hash(hdd, tmp_path):
    nvme = tmp_path / 'a.jsonl'
    nvme.write_text('x')
    assert hdd_guard.stage_input(nvme, 'bad') == str(nvme)
    source = hdd / 'd.jsonl'
    source.write_text('data')
    link = tmp_path / 'link.jsonl'
    link.symlink_to(source)
    digest = hashlib.sha256(b'data').hexdigest()
    staged = hdd_guard.stage_input(link, digest)
    assert staged == str(tmp_path / 'staged' / digest / 'd.jsonl')
    assert open(staged).read() == 'data' and hdd_guard.RECORDS[0]['declared_hash']
    assert hdd_guard.stage_input(link) == staged  # undeclared hash is computed


def test_hash_mismatch_refused_and_out_dir_must_be_nvme(hdd, tmp_path):
    source = hdd / 'd.jsonl'
    source.write_text('data')
    with pytest.raises(ValueError, match='SHA-256'):
        hdd_guard.stage_input(source, '0' * 64)
    with pytest.raises(ValueError, match='external HDD'):
        hdd_guard.require_nvme(hdd / 'run')
    hdd_guard.require_nvme(tmp_path / 'run')


def test_noop_without_hdd(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(tmp_path / 'absent'))
    assert not hdd_guard.on_hdd(tmp_path / 'x')
    hdd_guard.require_nvme(tmp_path / 'absent' / 'run')


def test_auto_hdd_root_on_nonrotating_device_is_not_staged(tmp_path, monkeypatch):
    monkeypatch.delenv('NATLANG_HDD_PREFIX', raising=False)
    monkeypatch.setattr(hdd_guard, 'resolve', lambda name: tmp_path)
    monkeypatch.setattr(hdd_guard, '_rotational', lambda path: False)
    assert hdd_guard.hdd_prefix() is None
    assert not hdd_guard.on_hdd(tmp_path / 'run')
    # Explicit selection remains authoritative for development and real HDDs.
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(tmp_path))
    assert hdd_guard.on_hdd(tmp_path / 'run')


def test_stage_input_args_pass_staged_path(hdd, tmp_path):
    from natlang_neuralese.train import recipe
    source = hdd / 'graphs.jsonl'
    source.write_text('t')
    digest = hashlib.sha256(b't').hexdigest()
    kind = 'native_state_sft'
    assert 'native_graphs' in hdd_guard.STAGED_ROLES
    assert 'text_data' not in hdd_guard.STAGED_ROLES
    args = recipe.stage_input_args({'native_graphs': {'path': str(source), 'sha256': digest}}, kind, target='local')
    assert args == ['--native-graphs', str(tmp_path / 'staged' / digest / 'graphs.jsonl')]


def test_native_checkpoint_is_read_in_place_while_graph_corpus_is_staged(hdd, tmp_path):
    """The native checkpoint is loaded once in place; graph corpora are staged for repeated reads."""
    from natlang_neuralese.train import recipe
    state = hdd / 'checkpoint.pt'
    state.write_bytes(b'state')
    graphs = hdd / 'graphs.jsonl'
    graphs.write_text('g')
    assert hdd_guard.input_path('continue_from', state) == str(state)
    assert not (tmp_path / 'staged').exists()
    assert all(r['read_from_hdd'] for r in hdd_guard.RECORDS) and len(hdd_guard.RECORDS) == 1
    staged = recipe._staged_inputs({'continue_from': {'path': str(state)}, 'native_graphs': {'path': str(graphs)}})
    assert staged['continue_from']['path'] == str(state)
    assert staged['native_graphs']['path'].startswith(str(tmp_path / 'staged'))


def test_the_staged_cache_is_bounded_least_recently_used_first(hdd, tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setenv('NATLANG_STAGED_INPUTS_MAX_GB', str(3.5 * 1024 / 2 ** 30))  # room for three 1 KiB files
    paths = []
    for name in 'abc':
        source = hdd / f'{name}.jsonl'
        source.write_bytes(name.encode() * 1024)
        paths.append(hdd_guard.stage_input(source))
        time.sleep(0.01)
    os.utime(paths[0], (time.time() + 5, time.time() + 5))  # a used recently
    source = hdd / 'd.jsonl'
    source.write_bytes(b'd' * 1024)
    hdd_guard.stage_input(source)
    assert os.path.exists(paths[0]) and not os.path.exists(paths[1])


def _json_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_source_manifest_staging_keeps_relative_transitive_leaves_for_real_reader(hdd, tmp_path, monkeypatch):
    import json
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.recipe import _staged_inputs

    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index'))
    rows = hdd / 'trees' / 'rows'
    manifests = hdd / 'trees' / 'manifests'
    rows.mkdir(parents=True)
    manifests.mkdir(parents=True)
    leaf = rows / 'train.jsonl'
    leaf.write_text(json.dumps({'schema': 'natlang.neuralese-native-graph/1', 'id': 'train-row',
        'split': 'train', 'source_groups': ['world-a'], 'tokenizer_fingerprint': 'tok',
        'nodes': [{'name': 'text', 'token_ids': [1, 2], 'target_weights': [0., 1.]}]}) + '\n')
    child = manifests / 'child.json'
    child.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '../rows/train.jsonl', 'sha256': _json_hash(leaf)}]}))
    root = manifests / 'root.json'
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': 'child.json', 'sha256': _json_hash(child)}]}))
    dependencies = [{'path': str(path), 'sha256': _json_hash(path)} for path in (child, leaf)]
    staged = hdd_guard.stage_input(root, _json_hash(root))
    staged_root = Path(staged)
    assert staged_root.read_bytes() == root.read_bytes()
    assert (staged_root.parent / 'child.json').read_bytes() == child.read_bytes()
    assert (staged_root.parent.parent / 'rows' / 'train.jsonl').read_bytes() == leaf.read_bytes()
    leaf.write_text(leaf.read_text().replace('train-row', 'tampered-row'))
    with pytest.raises(ValueError, match='manifest dependency .*SHA-256'):
        _staged_inputs({'native_graphs': {'path': str(root), 'sha256': _json_hash(root),
            'dependencies': dependencies}})
    import shutil
    shutil.rmtree(hdd)
    source = NativeGraphSource(staged_root, 'tok', required_split='train')
    assert [entry['id'] for entry in source.entries] == ['train-row']
    assert any(item['source'] == str(leaf) and item['staged'].endswith('/rows/train.jsonl')
               for item in hdd_guard.RECORDS[-1]['dependencies'])

def test_native_source_reader_routes_absolute_references_to_exact_staged_leaves(hdd, tmp_path):
    import json
    from natlang_neuralese.data.native_sources import NativeGraphSource

    leaf = hdd / 'absolute-row.jsonl'
    leaf.write_text(json.dumps({'schema': 'natlang.neuralese-native-graph/1', 'id': 'absolute-row',
        'split': 'train', 'source_groups': ['world-b'], 'tokenizer_fingerprint': 'tok',
        'nodes': [{'name': 'text', 'token_ids': [3, 4], 'target_weights': [0., 1.]}]}) + '\n')
    child = hdd / 'child.json'
    child.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': str(leaf), 'sha256': _json_hash(leaf)}]}))
    root = hdd / 'nested' / 'deeper' / 'root.json'
    root.parent.mkdir(parents=True)
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '../../child.json', 'sha256': _json_hash(child)}]}))
    staged = Path(hdd_guard.stage_input(root, _json_hash(root)))
    assert staged.read_bytes() == root.read_bytes()
    import shutil
    shutil.rmtree(hdd)
    reader = NativeGraphSource(staged, 'tok', required_split='train')
    assert [entry['id'] for entry in reader.entries] == ['absolute-row']
    assert reader.children[0].path.is_relative_to(Path(staged_guard_root(staged)))
    assert reader.children[0].receipt()['path'] == str(child.resolve())


def test_nvme_to_nvme_nested_manifest_census_stages_hdd_leaf_for_actual_reader(hdd, tmp_path):
    import json
    from natlang_neuralese.data.native_sources import NativeGraphSource

    rows = hdd / 'deep' / 'train.jsonl'
    rows.parent.mkdir(parents=True)
    rows.write_text(json.dumps({'schema': 'natlang.neuralese-native-graph/1', 'id': 'nested-hdd-row',
        'split': 'train', 'source_groups': ['world-c'], 'tokenizer_fingerprint': 'tok',
        'nodes': [{'name': 'text', 'token_ids': [5, 6], 'target_weights': [0., 1.]}]}) + '\n')
    inner = tmp_path / 'nvme-inner.json'
    inner.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': str(rows), 'sha256': _json_hash(rows)}]}))
    root = tmp_path / 'nvme-root.json'
    root.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': inner.name, 'sha256': _json_hash(inner)}]}))
    staged = Path(hdd_guard.stage_input(root, _json_hash(root)))
    assert staged != root
    import shutil
    shutil.rmtree(hdd)
    reader = NativeGraphSource(staged, 'tok', required_split='train')
    assert [entry['id'] for entry in reader.entries] == ['nested-hdd-row']
    assert reader.children[0].receipt()['sources'][0]['path'] == str(rows.resolve())


@pytest.mark.parametrize('role,split', [
    ('native_graphs', 'train'), ('held_native_graphs', 'test'),
    ('continued_native_graphs', 'train'), ('continued_held_native_graphs', 'test'),
    ('foundation_native_graphs', 'train'),
])
def test_each_native_graph_staged_role_uses_real_source_reader(hdd, tmp_path, role, split):
    import json
    from natlang_neuralese.data.native_sources import NativeGraphSource

    leaf = hdd / f'{role}.jsonl'
    leaf.write_text(json.dumps({'schema': 'natlang.neuralese-native-graph/1', 'id': role,
        'split': split, 'source_groups': ['role-test'], 'tokenizer_fingerprint': 'tok',
        'nodes': [{'name': 'text', 'token_ids': [1, 2], 'target_weights': [0., 1.]}]}) + '\n')
    staged = Path(hdd_guard.input_path(role, leaf, _json_hash(leaf)))
    leaf.unlink()
    reader = NativeGraphSource(staged, 'tok', required_split=split)
    assert reader.entries[0]['id'] == role
    assert reader.receipt()['path'] == str(leaf.resolve())


def test_rollouts_role_staging_preserves_exact_rows_for_native_rollout_reader(hdd):
    import json
    from natlang_neuralese.common.jsonio import iter_jsonl_rows

    rollout = hdd / 'rollouts.jsonl'
    row = {'schema': 'natlang.native-rollout/1', 'task_id': 'stage-proof', 'turns': []}
    rollout.write_text(json.dumps(row) + '\n')
    staged = hdd_guard.input_path('rollouts', rollout, _json_hash(rollout))
    rollout.unlink()
    assert list(iter_jsonl_rows(staged))[0][1] == row


def test_pending_native_source_reader_follows_staged_absolute_cohort_and_evidence(hdd, tmp_path):
    import json
    from natlang_neuralese.train.native_pending_sources import declaration, load_pending_sources
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from test_native_state_sft import graph

    rows = hdd / 'pending-rows.jsonl'
    value = dict(graph('pending-row'), tokenizer_fingerprint='tokenizer-sha')
    rows.write_text(json.dumps(value) + '\n')
    graph_manifest = hdd / 'pending-graphs.json'
    graph_manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': str(rows), 'sha256': _json_hash(rows)}]}))
    full_cohort = hdd / 'full-cohort.json'
    full_cohort.write_text(graph_manifest.read_text())
    evidence = hdd / 'evidence.json'
    evidence.write_text('{"verified":true}')
    pending = hdd / 'pending.json'
    pending.write_text(json.dumps(declaration(full_cohort, graph_manifest,
        tokenizer_fingerprint='tokenizer-sha', role_ids=(58, 59, 60, 61),
        evidence=[{'path': str(evidence), 'sha256': sha256_file_hex_cached(evidence)}])))

    staged = hdd_guard.input_path('pending_native_sources', pending, _json_hash(pending))
    import shutil
    shutil.rmtree(hdd)
    loaded = load_pending_sources(staged, tokenizer_fingerprint='tokenizer-sha')
    assert set(loaded['rows']) == {'pending-row'}
    assert str(rows) not in {item['path'] for item in loaded['dependencies']}
    assert all(str(item['path']).startswith(str(tmp_path / 'staged')) for item in loaded['dependencies'])


def staged_guard_root(staged):
    """Locate the directory carrying hdd_guard's per-closure absolute path map."""
    for parent in Path(staged).resolve().parents:
        if (parent / '.natlang-source-map.json').is_file():
            return parent
    raise AssertionError('staged source map missing')


def test_nvme_manifest_fastpath_does_not_hash_leaf_corpus(tmp_path, monkeypatch):
    import json
    manifest = tmp_path / 'graphs.json'
    leaf = tmp_path / 'large-corpus.jsonl'
    leaf.write_text('this path must not be hashed by the staging fast path')
    manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': leaf.name, 'sha256': '0' * 64}]}))
    monkeypatch.setenv('NATLANG_HDD_PREFIX', str(tmp_path / 'other-drive'))
    monkeypatch.setattr(hdd_guard, '_sha', lambda path: (_ for _ in ()).throw(AssertionError('unexpected leaf hash')))
    assert hdd_guard.stage_input(manifest, '1' * 64) == str(manifest)


def test_directory_rl_bundle_staging_preserves_real_bundle_reader_and_repairs_tampered_cache(hdd, tmp_path):
    import json
    import torch
    from natlang_neuralese.model.native_artifact import NativeTensorArtifact, decode_native_tensor_artifact, encode_native_tensor_artifact
    from natlang_neuralese.rl.graph_replay import NativeBundleStore

    identity = {'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': '2' * 64,
        'embedding_weight_sha256': '3' * 64, 'architecture': 'hdd-bundle-reader-test'}
    artifact = NativeTensorArtifact('parameter-set', identity, {'layer.weight': torch.tensor([1., 2.])})
    wire = encode_native_tensor_artifact(artifact)
    _, ident = decode_native_tensor_artifact(wire)
    bundle = hdd / 'bundle'
    bundle.mkdir(parents=True)
    block_path = bundle / (ident + '.nzb')
    block_path.write_bytes(wire)
    manifest = {'schema': 'natlang.rl-block-bundle/1', 'policy_id': identity['checkpoint_sha256'],
        'blocks': {ident: {'path': str(block_path.resolve()), 'sha256': _json_hash(block_path),
            'bytes': len(wire), 'format': 'native-tensor', 'native_identity': identity}}}
    manifest_path = bundle / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    tree_sha = hdd_guard._directory_sha256(bundle)
    staged = Path(hdd_guard.input_path('rl_block_bundle', bundle, tree_sha))
    staged_manifest = staged / 'manifest.json'
    block_path.unlink()
    reader = NativeBundleStore(staged_manifest, policy_id=identity['checkpoint_sha256'])
    from natlang_neuralese.rl.graph_replay import verified_bundle_path
    resolved_block = verified_bundle_path(staged_manifest, ident, policy_id=identity['checkpoint_sha256'])
    assert resolved_block.is_relative_to(staged) and not hdd_guard.on_hdd(resolved_block)
    loaded = reader.get_native_tensor_artifact(ident)
    assert loaded.id == ident and loaded.artifact.tensors['layer.weight'].tolist() == [1., 2.]
    assert json.loads(staged_manifest.read_text()) == manifest

    staged_block = staged / block_path.name
    block_path.write_bytes(wire)
    staged_block.write_bytes(b'corrupt cached copy')
    repaired = Path(hdd_guard.input_path('rl_block_bundle', bundle, tree_sha))
    restored = NativeBundleStore(repaired / 'manifest.json', policy_id=identity['checkpoint_sha256'])
    assert restored.get_native_tensor_artifact(ident).artifact.tensors['layer.weight'].tolist() == [1., 2.]
    with pytest.raises(ValueError, match='directory input .*declared SHA-256'):
        hdd_guard.input_path('rl_block_bundle', bundle, '0' * 64)


def test_portable_stage_argv_keeps_logical_inputs_and_local_uses_staging(monkeypatch):
    from natlang_neuralese.train import recipe
    source = {'path': '/home/werg/logical/train-sources.json', 'sha256': 'a' * 64}
    resolved = {'native_graphs': source, 'foundation_native_graphs': source}
    monkeypatch.setattr(recipe, '_staged_inputs', lambda values: {
        key: {**value, 'path': '/private/staged/tree/train-sources.json'}
        for key, value in values.items()})
    portable = recipe.stage_input_args(resolved, 'native_state_sft')
    assert portable == ['--native-graphs', source['path'], '--foundation-native-graphs', source['path']]
    local = recipe.stage_input_args(resolved, 'native_state_sft', target='local')
    assert local == ['--native-graphs', '/private/staged/tree/train-sources.json',
                     '--foundation-native-graphs', '/private/staged/tree/train-sources.json']
    with pytest.raises(ValueError, match='local or portable'):
        recipe.stage_input_args(resolved, 'native_state_sft', target='unknown')
