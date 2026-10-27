"""Frozen task runtimes own their protection tables and npm manifests."""
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.runtime_inventory import task_runtime_inventory, task_source_reviews


def _runtime(tmp_path, frozen):
    root = tmp_path / 'runtime'
    files = {'prelude.js': '// prelude', 'package.json': '{}', 'dist/teacher/cli.js': '// collector'}
    reviews = 'training/source-reviews/' if frozen else '../training/source-reviews/'
    files.update({reviews + name: '{}' for name in ('holds.jsonl', 'family-holds.json', 'datasets.json')})
    files.update({'package-lock.json': '{}'} if frozen else {'../package.json': '{}', '../package-lock.json': '{}'})
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    if frozen:
        declared = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()}
        (root / 'frozen-runtime.json').write_text(json.dumps({'files': declared}))
    return root


@pytest.mark.parametrize('frozen', [False, True])
def test_inventory_and_execution_choose_the_same_review_assets(tmp_path, frozen):
    root = _runtime(tmp_path, frozen)
    inventory = task_runtime_inventory(root)
    directory = task_source_reviews(root)
    assert directory.resolve() == (root / 'training/source-reviews' if frozen else root.parent / 'training/source-reviews').resolve()
    assert any((root / name).resolve() == (directory / 'family-holds.json').resolve() for name in inventory)
    assert ('package-lock.json' in inventory) is frozen
    assert ('../package-lock.json' in inventory) is not frozen


@pytest.mark.parametrize('name', ['dist/teacher/cli.js', 'package-lock.json', 'training/source-reviews/holds.jsonl'])
def test_frozen_inventory_rejects_changed_executing_assets(tmp_path, name):
    root = _runtime(tmp_path, True)
    (root / name).write_text('changed')
    with pytest.raises(ValueError, match='changed its declared file'):
        task_runtime_inventory(root)


def test_frozen_inventory_never_substitutes_parent_policy(tmp_path):
    root = _runtime(tmp_path, True)
    (root / 'training/source-reviews/holds.jsonl').unlink()
    parent = root.parent / 'training/source-reviews'
    parent.mkdir(parents=True)
    (parent / 'holds.jsonl').write_text('{}')
    with pytest.raises(ValueError, match='pinned npm manifests'):
        task_runtime_inventory(root)


def test_missing_frozen_registry_cannot_execute_with_parent_tables(tmp_path):
    import shutil
    root = _runtime(tmp_path, True)
    shutil.rmtree(root / 'training/source-reviews')
    parent = root.parent / 'training/source-reviews'
    parent.mkdir(parents=True)
    for name in ('holds.jsonl', 'family-holds.json', 'datasets.json'):
        (parent / name).write_text('{}')
    assert task_source_reviews(root) == root / 'training/source-reviews'
    with pytest.raises(ValueError, match='pinned npm manifests'):
        task_runtime_inventory(root)


@pytest.mark.parametrize('frozen', [False, True])
def test_runtime_inventory_binds_dependency_bytes_and_symlink_targets(tmp_path, frozen):
    root = _runtime(tmp_path, frozen)
    package = tmp_path / 'installed' / 'package'
    package.mkdir(parents=True)
    dependency = package / 'index.js'
    dependency.write_text('export const value = 1;\n')
    (root / 'node_modules').mkdir()
    (root / 'node_modules' / 'dep').symlink_to(package, target_is_directory=True)
    original = task_runtime_inventory(root)
    assert original['@dependency-source-links']['node_modules/dep']['resolved'] == str(package)

    dependency.write_text('export const value = 2;\n')
    changed = task_runtime_inventory(root)
    assert changed['@dependency-files'] != original['@dependency-files']

    stable_bytes = changed
    dependency.chmod(0o755)
    executable = task_runtime_inventory(root)
    assert executable['@dependency-files'] != stable_bytes['@dependency-files']

    (root / 'node_modules' / 'dep').unlink()
    replacement = tmp_path / 'installed' / 'replacement'
    replacement.mkdir()
    (replacement / 'index.js').write_text('export const value = 2;\n')
    (root / 'node_modules' / 'dep').symlink_to(replacement, target_is_directory=True)
    retargeted = task_runtime_inventory(root)
    assert retargeted['@dependency-source-links'] != executable['@dependency-source-links']


def test_task_eval_runtime_identity_binds_dependencies_as_files_and_layout(tmp_path, monkeypatch):
    from natlang_neuralese.eval import task_cadence
    from natlang_neuralese.common.runtime_inventory import task_runtime_identity_files

    root = _runtime(tmp_path, False)
    package = root / 'node_modules/dep'
    package.mkdir(parents=True)
    (package / 'index.js').write_text('export const value = 3;\n')
    monkeypatch.setattr(task_cadence, '_git_head', lambda: 'fixture-commit')
    identity = task_cadence.task_eval_runtime_identity({'runtime': str(root)})
    assert identity['schema'] == 'natlang.task-eval-runtime-identity/2'
    assert task_runtime_identity_files(identity) == identity['files']
    dependency = (package / 'index.js').resolve()
    assert any(row['path'] == str(dependency) for row in identity['files'])
    assert identity['dependency_closure']['files']['node_modules/dep/index.js']['sha256']

    (package / 'index.js').write_text('export const value = 4;\n')
    updated = task_cadence.task_eval_runtime_identity({'runtime': str(root)})
    assert updated != identity
    changed = json.loads(json.dumps(updated))
    changed['dependency_closure']['files']['node_modules/dep/index.js']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='binding differs'):
        task_runtime_identity_files(changed)
    changed = json.loads(json.dumps(updated))
    next(row for row in changed['files'] if row.get('runtime_dependency'))['path'] = '../escaped'
    with pytest.raises(ValueError, match='file binding is malformed'):
        task_runtime_identity_files(changed)


def test_saved_task_runtime_identity_v1_is_read_only_historical_shape():
    from natlang_neuralese.common.runtime_inventory import task_runtime_identity_files

    files = [{'path': '/old/runtime/dist/teacher/cli.js', 'bytes': 4, 'sha256': 'a' * 64}]
    identity = {'schema': 'natlang.task-eval-runtime-identity/1', 'files': files}
    assert task_runtime_identity_files(identity) is files


@pytest.mark.parametrize('frozen', [False, True])
def test_runtime_snapshot_receipt_verifies_copied_dependencies_and_retry(tmp_path, frozen):
    import shutil
    from natlang_neuralese.common.runtime_inventory import snapshot_task_runtime

    root = _runtime(tmp_path, frozen)
    external = tmp_path / 'external-package'
    external.mkdir()
    (external / 'index.js').write_text('export const value = 7;\n')
    (root / 'node_modules').mkdir()
    (root / 'node_modules' / 'dep').symlink_to(external, target_is_directory=True)
    destination = tmp_path / 'snapshots'

    snapshot = snapshot_task_runtime(root, destination)
    bound = task_runtime_inventory(snapshot)
    assert bound['@dependency-source-links']['node_modules/dep']['resolved'] == str(external)
    copied = snapshot / 'node_modules/dep/index.js'
    assert copied.read_text() == 'export const value = 7;\n'
    assert not copied.is_symlink()
    assert snapshot_task_runtime(root, destination) == snapshot

    copied.write_text('tampered\n')
    with pytest.raises(ValueError, match='dependency contents changed|prepared task runtime changed'):
        task_runtime_inventory(snapshot)
    with pytest.raises(ValueError, match='dependency contents changed'):
        snapshot_task_runtime(root, destination)

    copied.write_text('export const value = 7;\n')
    receipt_path = snapshot / 'runtime-dependencies.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['source_links']['node_modules/dep']['resolved'] = str(tmp_path / 'different-target')
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt digest differs'):
        task_runtime_inventory(snapshot)
    receipt['sha256'] = hashlib.sha256(json.dumps(
        {key:value for key,value in receipt.items() if key!='sha256'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='snapshot identity'):
        task_runtime_inventory(snapshot)

    # A fresh snapshot location restores the exact source dependency tree.
    repaired = snapshot_task_runtime(root, tmp_path / 'restored-snapshots')
    assert (repaired / 'node_modules/dep/index.js').read_text() == 'export const value = 7;\n'
    nested = snapshot_task_runtime(repaired, tmp_path / 'nested-snapshots')
    assert (nested / 'node_modules/dep/index.js').read_text() == 'export const value = 7;\n'
    assert task_runtime_inventory(nested)['@dependency-files'] == task_runtime_inventory(repaired)['@dependency-files']


def test_snapshot_rejects_dependency_source_change_during_copy(tmp_path, monkeypatch):
    import shutil
    from natlang_neuralese.common import runtime_inventory

    root = _runtime(tmp_path, False)
    (root / 'node_modules').mkdir()
    package = root / 'node_modules/dep'
    package.mkdir(parents=True)
    source_file = package / 'index.js'
    source_file.write_text('original')
    real_copytree = shutil.copytree

    def copy_then_mutate(source, destination, *args, **kwargs):
        result = real_copytree(source, destination, *args, **kwargs)
        if Path(source) == root / 'node_modules':
            source_file.write_text('mutated')
        return result

    monkeypatch.setattr(shutil, 'copytree', copy_then_mutate)
    with pytest.raises(ValueError, match='changed during preparation'):
        runtime_inventory.snapshot_task_runtime(root, tmp_path / 'snapshots')
