import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_check_main_node_modules import prepare


def _package(path: Path, name: str, *, main='index.js'):
    path.mkdir(parents=True)
    (path / 'package.json').write_text(json.dumps({'name': name, 'main': main}))
    (path / main).write_text(f'export default {json.dumps(name)};\n')


def _roots(tmp_path: Path):
    main = tmp_path / 'main'
    worktree = tmp_path / 'worktree'
    (main / 'node_modules' / '@natlang').mkdir(parents=True)
    (main / 'npm-packages' / 'node').mkdir(parents=True)
    (main / 'npm-packages' / 'cli').mkdir(parents=True)
    (main / 'npm-packages' / 'retired').mkdir(parents=True)
    (worktree / 'npm-packages' / 'node').mkdir(parents=True)
    (worktree / 'npm-packages' / 'cli').mkdir(parents=True)
    (worktree / 'tools' / 'local-workspace').mkdir(parents=True)
    (worktree / 'ts-host').mkdir()
    (main / 'node_modules' / 'shared-dependency').mkdir()
    (main / 'node_modules' / 'shared-dependency' / 'package.json').write_text(
        json.dumps({'name': 'shared-dependency', 'main': 'index.js'}))
    (main / 'node_modules' / 'shared-dependency' / 'index.js').write_text('export default "shared";\n')
    for package, version in ((main, 'canonical'), (worktree, 'tested')):
        (package / 'npm-packages' / 'node' / 'package.json').write_text(json.dumps({
            'name': '@natlang/node', 'version': version, 'main': 'index.js'}))
        (package / 'npm-packages' / 'node' / 'index.js').write_text(
            f'export default {json.dumps(version)};\n')
    (main / 'npm-packages' / 'cli' / 'package.json').write_text(json.dumps({
        'name': '@natlang/cli', 'version': '1.0.0'}))
    (main / 'npm-packages' / 'retired' / 'package.json').write_text(json.dumps({
        'name': '@natlang/retired', 'version': '1.0.0'}))
    (worktree / 'npm-packages' / 'cli' / 'package.json').write_text(json.dumps({
        'name': '@natlang/cli', 'version': '1.0.0'}))
    (worktree / 'tools' / 'local-workspace' / 'package.json').write_text(json.dumps({
        'name': 'local-workspace', 'version': '1.0.0'}))
    (worktree / 'ts-host' / 'package.json').write_text(json.dumps({
        'name': '@natlang/typescript-host', 'version': '1.0.0'}))
    (worktree / 'package.json').write_text(json.dumps({
        'name': 'fixture-workspace', 'workspaces': ['ts-host', 'npm-packages/*', 'tools/*']}))
    (main / 'package.json').write_text(json.dumps({
        'name': 'fixture-main', 'workspaces': ['ts-host', 'npm-packages/*']}))
    (main / 'node_modules' / '@natlang' / 'node').symlink_to(
        main / 'npm-packages' / 'node', target_is_directory=True)
    (main / 'node_modules' / '@natlang' / 'cli').symlink_to(
        main / 'npm-packages' / 'cli', target_is_directory=True)
    (main / 'node_modules' / '@natlang' / 'retired').symlink_to(
        main / 'npm-packages' / 'retired', target_is_directory=True)
    (main / 'node_modules' / 'local-workspace').symlink_to(
        main / 'npm-packages' / 'node', target_is_directory=True)
    (worktree / 'node_modules').symlink_to(main / 'node_modules', target_is_directory=True)
    return main, worktree


def test_linked_worktree_shares_third_party_but_uses_its_workspace_packages(tmp_path):
    main, worktree = _roots(tmp_path)
    assert prepare(worktree, main) is True
    assert not (worktree / 'node_modules').is_symlink()
    assert (worktree / 'node_modules' / 'shared-dependency').resolve() == (
        main / 'node_modules' / 'shared-dependency').resolve()
    assert (worktree / 'node_modules' / '@natlang' / 'node').resolve() == (
        worktree / 'npm-packages' / 'node').resolve()
    assert (worktree / 'node_modules' / '@natlang' / 'cli').resolve() == (
        worktree / 'npm-packages' / 'cli').resolve()
    assert (worktree / 'node_modules' / 'local-workspace').resolve() == (
        worktree / 'tools' / 'local-workspace').resolve()
    assert not (worktree / 'node_modules' / '@natlang' / 'retired').exists()

    script = """
const { createRequire } = await import('node:module');
const { pathToFileURL } = await import('node:url');
const require = createRequire(pathToFileURL(process.cwd() + '/probe.mjs'));
console.log(JSON.stringify({
  natlang: require.resolve('@natlang/node'),
  shared: require.resolve('shared-dependency')
}));
"""
    resolved = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script],
        cwd=worktree, text=True))
    assert Path(resolved['natlang']).resolve() == (worktree / 'npm-packages/node/index.js').resolve()
    assert Path(resolved['shared']).resolve() == (main / 'node_modules/shared-dependency/index.js').resolve()
    assert prepare(worktree, main) is False


def test_canonical_checkout_and_unknown_existing_worktree_install_are_untouched(tmp_path):
    main, worktree = _roots(tmp_path)
    original = os.readlink(worktree / 'node_modules')
    assert prepare(main, main) is False
    assert os.readlink(worktree / 'node_modules') == original

    (worktree / 'node_modules').unlink()
    (worktree / 'node_modules').mkdir()
    sentinel = worktree / 'node_modules' / 'sentinel'
    sentinel.write_text('keep')
    (worktree / 'node_modules' / '@natlang').mkdir()
    (worktree / 'node_modules' / '@natlang' / 'node').symlink_to(
        main / 'npm-packages/node', target_is_directory=True)
    with pytest.raises(ValueError, match='non-local/incomplete workspace links'):
        prepare(worktree, main)
    assert sentinel.read_text() == 'keep'
    assert (worktree / 'node_modules' / '@natlang' / 'node').resolve() == (
        main / 'npm-packages/node').resolve()
