"""Exact semantic and metadata preservation across the internal prefix extension."""
import copy
import importlib.util
import subprocess
from pathlib import Path

import pytest
from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('continue_native_source_indexes', ROOT / 'scripts/neuralese_data/continue_native_source_indexes.py')
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


@pytest.fixture
def packages(tmp_path):
    # This reachable immutable production parent predates the prefix
    # extension; its exact compiler is the independent old semantics, not a
    # reverse-patched fixture.
    roots = tmp_path / 'old', tmp_path / 'new'
    for root in roots:
        for name in (*subject.POLICY_FILES, 'common/paths.py'):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if root == roots[0]:
                target.write_bytes(subprocess.check_output(['git', 'show', f'd8bb4642f9dad3c1eff9cd86c1130e4c96cba0af:training/neuralese/natlang_neuralese/{name}'], cwd=ROOT))
            else:
                # This verifier supports exactly the historical prefix change.
                # Later import-adjoint/compiler changes require new indexes,
                # rather than silently widening the cache equivalence proof.
                target.write_bytes((roots[0] / name).read_bytes() if name != 'model/native_graph_plan.py'
                    else subprocess.check_output(['git', 'show',
                        f'5706256bc7f733e9cd13199d4fcb58e05c50ce3a:training/neuralese/natlang_neuralese/{name}'],cwd=ROOT))
    return roots


def test_exact_production_extension(packages):
    proof = subject.prove_extension(*packages)
    assert proof['old_policy_sha256'] != proof['new_policy_sha256']


def test_later_import_adjoint_semantics_cannot_reuse_prefix_only_proof(packages):
    target=packages[1]/'model/native_graph_plan.py'
    target.write_bytes(subprocess.check_output(['git','show',
        '85d263ee8:training/neuralese/natlang_neuralese/model/native_graph_plan.py'],cwd=ROOT))
    with pytest.raises(ValueError):subject.prove_extension(*packages)


@pytest.mark.parametrize('change', ['default', 'guard', 'arithmetic', 'helper', 'other', 'parser'])
def test_semantic_changes_rejected(packages, change):
    target = packages[1] / 'model/native_graph_plan.py'
    text = target.read_text()
    if change == 'default': text = text.replace('cache_prefix: bool = False', 'cache_prefix: bool = True')
    if change == 'guard': text = text.replace('edge.at != 0', 'edge.at != 1')
    if change == 'arithmetic': text = text.replace('physical_count += count', 'physical_count += count + 1')
    if change == 'helper': text = text.replace("value.pop('cache_prefix')", "value.pop('name')")
    if change in ('other', 'parser'):
        target = packages[1] / ('data/native_sources.py' if change == 'other' else 'data/native_graph_schema.py')
        text = target.read_text() + '\n# changed policy\n'
    target.write_text(text)
    with pytest.raises(ValueError): subject.prove_extension(*packages)


def make_index(proof):
    key = {'source_sha256': 'a'*64, 'tokenizer': 'tok', 'split': 'train',
        'policy_sha256': proof['old_policy_sha256'], 'kind': 'native-graphs',
        'pair_model': None, 'role_ids': []}
    entries = [{'id': 'r1', 'binding': 'exact', 'groups': ['protected-origin'],
                'forward_tokens': 7, 'supervised_tokens': 3, 'communication': False}]
    value = {'key': key, 'entries': entries, 'metadata_sha256': digest(entries)}
    path = row_index.entry_path('native-graphs', digest(key))
    row_index.save(path, value)
    return path, row_index.load(path)


def test_copy_preserves_entries_and_is_idempotent(packages, tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    proof = subject.prove_extension(*packages)
    old_path, value = make_index(proof)
    receipt = subject.continue_indexes(proof, ['a'*64, 'b'*64])
    assert receipt['missing_source_sha256'] == ['b'*64]
    item, = receipt['indexes']
    assert item['rows'] == 1
    copied = row_index.load(Path(item['new_index']))
    assert copied['entries'] == value['entries']
    assert copied['key'] == {**value['key'], 'policy_sha256': proof['new_policy_sha256']}
    assert row_index.load(old_path) == value
    assert subject.continue_indexes(proof, ['a'*64, 'b'*64]) == receipt


@pytest.mark.parametrize('corruption', ['metadata', 'filename', 'destination', 'kind'])
def test_corrupt_or_conflicting_index_rejected(packages, tmp_path, monkeypatch, corruption):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    proof = subject.prove_extension(*packages)
    path, value = make_index(proof)
    if corruption == 'metadata':
        value['entries'][0]['forward_tokens'] = 8
        row_index.save(path, value)
    elif corruption == 'kind':
        path.unlink()
        value['key']['kind'] = 'unrelated-index'
        row_index.save(row_index.entry_path('native-graphs', digest(value['key'])), value)
    elif corruption == 'filename':
        path.rename(path.with_name('native-graphs-wrong.pkl'))
    else:
        destination_key = {**value['key'], 'policy_sha256': proof['new_policy_sha256']}
        bad = copy.deepcopy(value)
        bad['key'] = destination_key
        bad['entries'][0]['binding'] = 'different'
        bad['metadata_sha256'] = digest(bad['entries'])
        row_index.save(row_index.entry_path('native-graphs', digest(destination_key)), bad)
    with pytest.raises(ValueError): subject.continue_indexes(proof, ['a'*64])


def test_invalid_hash_rejected(packages):
    with pytest.raises(ValueError): subject.continue_indexes(subject.prove_extension(*packages), ['not-a-source'])


@pytest.mark.parametrize('conflicting', [False, True])
def test_racing_live_writer_is_never_replaced(packages, tmp_path, monkeypatch, conflicting):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    proof = subject.prove_extension(*packages)
    _, old = make_index(proof)
    new_key = {**old['key'], 'policy_sha256': proof['new_policy_sha256']}
    target = row_index.entry_path('native-graphs', digest(new_key))
    winner = copy.deepcopy({**old, 'key': new_key})
    if conflicting:
        winner['entries'][0]['binding'] = 'concurrent-different-value'
        winner['metadata_sha256'] = digest(winner['entries'])
    real_link = subject.os.link
    observed = {}
    def publish_winner_then_link(source, destination):
        assert destination == target
        row_index.save(target, winner)
        observed.update(inode=target.stat().st_ino, raw=target.read_bytes())
        return real_link(source, destination)
    monkeypatch.setattr(subject.os, 'link', publish_winner_then_link)
    if conflicting:
        with pytest.raises(ValueError, match='existing destination index conflicts'):
            subject.continue_indexes(proof, ['a'*64])
    else:
        receipt = subject.continue_indexes(proof, ['a'*64])
        assert len(receipt['indexes']) == 1
    assert target.stat().st_ino == observed['inode']
    assert target.read_bytes() == observed['raw']
    assert row_index.load(target) == winner
    assert not list(target.parent.glob('*.tmp'))


def test_failed_no_clobber_publication_cleans_private_file(packages, tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    proof = subject.prove_extension(*packages)
    _, old = make_index(proof)
    new_key = {**old['key'], 'policy_sha256': proof['new_policy_sha256']}
    target = row_index.entry_path('native-graphs', digest(new_key))
    def denied(*args):
        raise OSError('publication denied')
    monkeypatch.setattr(subject.os, 'link', denied)
    with pytest.raises(OSError, match='publication denied'):
        subject.continue_indexes(proof, ['a'*64])
    assert not target.exists()
    assert not list(target.parent.glob('*.tmp'))


def test_new_loader_hits_continued_index_without_parsing(packages, tmp_path, monkeypatch):
    import importlib
    import json
    import sys
    import types
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'indexes'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    modules = []
    for name, root in zip(('_source_index_old_test', '_source_index_new_test'), packages):
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        monkeypatch.setitem(sys.modules, name, package)
        modules.append(importlib.import_module(name + '.data.native_sources'))
    source = tmp_path / 'rows.jsonl'
    source.write_text(json.dumps({'schema': 'natlang.neuralese-native-graph/1',
        'id': 'real-row', 'split': 'train', 'source_groups': ['real-source'],
        'tokenizer_fingerprint': 'tok', 'nodes': [{'name': 'text',
            'token_ids': [1, 2, 3], 'target_weights': [0., 1., 1.]}]}) + '\n')
    old = modules[0].NativeGraphSource(source, 'tok', required_split='train')
    receipt = subject.continue_indexes(subject.prove_extension(*packages), [subject.sha256_file_hex(source)])
    assert len(receipt['indexes']) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('continued source index must not reparse source rows')
    monkeypatch.setattr(modules[1], '_jsonl', forbidden)
    new = modules[1].NativeGraphSource(source, 'tok', required_split='train')
    assert new.entries == old.entries
    assert new.bindings == old.bindings
    # Source content changes cannot reuse the previous content-addressed index.
    source.write_text(source.read_text().replace('real-row', 'changed-row'))
    with pytest.raises(AssertionError, match='must not reparse'):
        modules[1].NativeGraphSource(source, 'tok', required_split='train')
