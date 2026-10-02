import os
from pathlib import Path

import pytest

from scripts.clone_runtime_tree import clone_runtime_tree


def test_runtime_clone_copies_npm_lock_into_an_independent_inode(tmp_path):
    source = tmp_path / 'sealed-v37'
    destination = tmp_path / 'private-build'
    lock = source / 'node_modules' / '.package-lock.json'
    lock.parent.mkdir(parents=True)
    lock.write_text('{"lockfileVersion":3,"sealed":true}\n')
    package = lock.parent / 'package.json'
    package.write_text('{"name":"fixture"}\n')
    (lock.parent / 'alias').symlink_to('package.json')

    clone_runtime_tree(source, destination)

    copied_lock = destination / 'node_modules' / '.package-lock.json'
    assert (lock.stat().st_dev, lock.stat().st_ino) != (
        copied_lock.stat().st_dev, copied_lock.stat().st_ino
    )
    copied_lock.write_text('{"lockfileVersion":3,"npm-mutated":true}\n')
    assert lock.read_text() == '{"lockfileVersion":3,"sealed":true}\n'
    assert os.readlink(destination / 'node_modules' / 'alias') == 'package.json'
    assert (destination / 'node_modules' / 'alias').read_text() == '{"name":"fixture"}\n'


def test_runtime_clone_rejects_external_symlinks_and_removes_partial_copy(tmp_path):
    source = tmp_path / 'sealed-runtime'
    destination = tmp_path / 'private-runtime'
    outside = tmp_path / 'shared-node-modules'
    source.mkdir()
    outside.mkdir()
    (outside / '.package-lock.json').write_text('{"mutable":true}\n')
    (source / 'node_modules').symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match='external symlink'):
        clone_runtime_tree(source, destination)
    assert not destination.exists()


def test_runtime_clone_refuses_to_overwrite_existing_destination(tmp_path):
    source = tmp_path / 'runtime'
    destination = tmp_path / 'candidate'
    source.mkdir()
    destination.mkdir()
    marker = destination / 'keep'
    marker.write_text('preserve')

    with pytest.raises(FileExistsError, match='destination already exists'):
        clone_runtime_tree(source, destination)
    assert marker.read_text() == 'preserve'


def test_runtime_clone_does_not_delete_destination_created_after_preflight(tmp_path, monkeypatch):
    source = tmp_path / 'runtime'
    destination = tmp_path / 'candidate'
    source.mkdir()
    destination.mkdir()
    marker = destination / 'foreign-file'
    marker.write_text('created by racing process')
    real_exists = Path.exists

    def stale_exists(path):
        if path == destination:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, 'exists', stale_exists)
    with pytest.raises(FileExistsError):
        clone_runtime_tree(source, destination)
    assert marker.read_text() == 'created by racing process'


def test_runtime_clone_rejects_destination_nested_in_source(tmp_path):
    source = tmp_path / 'runtime'
    source.mkdir()
    destination = source / 'private-copy'

    with pytest.raises(ValueError, match='must not contain one another'):
        clone_runtime_tree(source, destination)
    assert not destination.exists()
