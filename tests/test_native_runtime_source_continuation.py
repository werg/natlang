"""Real frozen validators retain all old-valid leaf metadata and positional plans."""
import dataclasses
import importlib
import json
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path

import pytest

from test_native_source_index_continuation import subject

ROOT = Path(__file__).resolve().parents[1]
PARENTS = ('14ca1758cacf99a4dc8f237fea7b8005d2123c40',
           'b50e8823aaa1f92d2d3be853afc1955948a8fd1f')


@pytest.fixture(scope='module')
def frozen_packages(tmp_path_factory):
    root = tmp_path_factory.mktemp('runtime-validators')
    for ref in (*PARENTS, '64ec316e5'):
        package = root / ref
        names = (*subject.POLICY_FILES, 'common/paths.py')
        if ref == '64ec316e5':
            names += ('data/native_source_tree.py',)
        for name in names:
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(['git', 'show',
                f'{ref}:training/neuralese/natlang_neuralese/{name}'], cwd=ROOT))
    return root


def load_source(package, monkeypatch):
    name = '_reviewed_source_' + uuid.uuid4().hex
    module = types.ModuleType(name)
    module.__path__ = [str(package)]
    monkeypatch.setitem(sys.modules, name, module)
    return importlib.import_module(name + '.data.native_sources')


def graph(identifier, *, split='train', external=False):
    result = {'schema': 'natlang.neuralese-native-graph/1', 'id': identifier,
              'split': split, 'source_groups': ['verified-group'],
              'tokenizer_fingerprint': '1'*64,
              'nodes': [{'name': 'producer', 'token_ids': [1, 2, 3],
                         'target_weights': [0, 1, .05], 'producing': [[0, 3]]},
                        {'name': 'consumer', 'token_ids': [4, 5, 6],
                         'target_weights': [0, 0, 1],
                         'imports': [{'source': 'producer', 'at': 1, 'start': 0, 'stop': 3}]}]}
    if external:
        result['external_states'] = {'stored': {'id': 'nzs1_'+'a'*52, 'token_count': 2}}
        result['nodes'][1]['imports'].append(
            {'source': 'stored', 'at': 1, 'start': 0, 'stop': 2})
    return result


def positional_plan(example):
    value = dataclasses.asdict(example.plan)
    assert not value.pop('slot_stops', ())
    for node in value['invocations']:
        assert not node.pop('cache_prefix', False)
        for edge in node['imports']:
            assert not edge.pop('stop_gradient', False)
    return value


@pytest.mark.parametrize('parent', PARENTS)
def test_actual_old_valid_rows_metadata_bindings_and_plans_survive(frozen_packages, tmp_path, monkeypatch, parent):
    from natlang_neuralese.common.hashing import sha256_file_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'indexes'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'digests.jsonl'))
    old = load_source(frozen_packages/parent, monkeypatch)
    new = load_source(frozen_packages/'64ec316e5', monkeypatch)
    rows = [graph('ordinary'), graph('external', external=True)]
    rows[0]['nodes'] = rows[0]['nodes'][:1]
    source = tmp_path/'graphs.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    before = old.NativeGraphSource(source, '1'*64, required_split='train')
    examples = list(before)
    proof = subject.prove_reviewed_runtime_extension(frozen_packages/parent, frozen_packages/'64ec316e5')
    receipt = subject.continue_indexes(proof, [sha256_file_hex(source)])
    assert len(receipt['indexes']) == 1 and not receipt['missing_source_sha256']
    # A hit must not parse; the following actual streamed reads still parse and
    # compare the inherited bindings against the destination implementation.
    parse = new.NativeGraphSource._parse
    monkeypatch.setattr(new.NativeGraphSource, '_parse', lambda *a: pytest.fail('continued cache missed'))
    after = new.NativeGraphSource(source, '1'*64, required_split='train')
    assert before.entries == after.entries and before.bindings == after.bindings
    monkeypatch.setattr(new.NativeGraphSource, '_parse', parse)
    assert [positional_plan(row) for row in after] == [positional_plan(row) for row in examples]
    assert [row.target_weights for row in after] == [row.target_weights for row in examples]
    assert subject.continue_indexes(proof, [sha256_file_hex(source)]) == receipt


@pytest.mark.parametrize('parent', PARENTS)
def test_paired_foundation_and_nested_priority_held_metadata(frozen_packages, tmp_path, monkeypatch, parent):
    monkeypatch.syspath_prepend(str(ROOT/'tests/neuralese'))
    from test_native_state_sft import authenticated_pair
    from natlang_neuralese.common.hashing import sha256_file_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'indexes'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'digests.jsonl'))
    old = load_source(frozen_packages/parent, monkeypatch)
    new = load_source(frozen_packages/'64ec316e5', monkeypatch)
    row = authenticated_pair('foundation')
    source = tmp_path/'pairs.jsonl';source.write_text(json.dumps(row)+'\n')
    roles = (58, 59, 60, 61)
    before = old.NativeGraphSource(source, '1'*64, required_split='train', pair_model='lfm', role_ids=roles)
    proof = subject.prove_reviewed_runtime_extension(frozen_packages/parent, frozen_packages/'64ec316e5')
    subject.continue_indexes(proof, [sha256_file_hex(source)])
    after = new.NativeGraphSource(source, '1'*64, required_split='train', pair_model='lfm', role_ids=roles)
    assert before.entries == after.entries and before.bindings == after.bindings
    assert before.entries[0]['communication']
    assert positional_plan(next(iter(before))) == positional_plan(next(iter(after)))
    held = tmp_path/'held.jsonl';held.write_text(json.dumps(graph('protected', split='test'))+'\n')
    nested = tmp_path/'nested.json'
    nested.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': held.name, 'sha256': sha256_file_hex(held), 'training_priority': 3}]}))
    outer = tmp_path/'outer.json'
    outer.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': nested.name, 'sha256': sha256_file_hex(nested)}]}))
    a = old.NativeGraphSource(outer, '1'*64, required_split='test')
    b = new.NativeGraphSource(outer, '1'*64, required_split='test')
    assert a.entries == b.entries and b.entries[0]['training_priority'] == 3
    assert a.groups == b.groups == {'verified-group'}
    with pytest.raises(ValueError, match='expected train'):
        new.NativeGraphSource(outer, '1'*64, required_split='train')


@pytest.mark.parametrize('side,name', [('old', name) for name in subject.POLICY_FILES]
                         + [('new', name) for name in (*subject.POLICY_FILES, 'data/native_source_tree.py')])
def test_every_unknown_policy_or_helper_mutation_rejected(frozen_packages, tmp_path, side, name):
    old, new = tmp_path/'old', tmp_path/'new'
    shutil.copytree(frozen_packages/PARENTS[0], old)
    shutil.copytree(frozen_packages/'64ec316e5', new)
    path = (old if side == 'old' else new)/name
    path.write_text(path.read_text()+'\n# unknown extension\n')
    with pytest.raises(ValueError, match='reviewed'):
        subject.prove_reviewed_runtime_extension(old, new)


def test_new_import_stop_syntax_needs_fresh_validation(frozen_packages, tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'indexes'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'digests.jsonl'))
    old = load_source(frozen_packages/PARENTS[0], monkeypatch)
    new = load_source(frozen_packages/'64ec316e5', monkeypatch)
    row = graph('new-syntax');row['nodes'][1]['imports'][0]['stop_gradient'] = True
    path = tmp_path/'new.jsonl';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(TypeError, match='stop_gradient'):
        old.NativeGraphSource(path, '1'*64, required_split='train')
    fresh = new.NativeGraphSource(path, '1'*64, required_split='train')
    assert any(next(iter(fresh)).plan.slot_stops)
