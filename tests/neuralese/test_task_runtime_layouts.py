"""Frozen task runtimes own their protection tables and npm manifests."""
import hashlib
import json

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
