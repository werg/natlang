import importlib.util
import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'freeze_native_runtime', Path(__file__).parents[2] / 'scripts/neuralese_data/freeze_native_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_node_closure_is_copied_and_content_bound(tmp_path):
    source = tmp_path / 'source'
    package = source / 'just-bash'
    package.mkdir(parents=True)
    (package / 'package.json').write_text(json.dumps({'version': '3.4.2'}))
    (package / 'index.js').write_text('export default 1')
    first = module.freeze_node_dependencies(source, tmp_path / 'frozen')
    (package / 'index.js').write_text('export default 2')
    assert (tmp_path / 'frozen/just-bash/index.js').read_text() == 'export default 1'
    second = module.freeze_node_dependencies(source, tmp_path / 'next')
    assert first['sha256'] != second['sha256']
    assert first['just_bash_version'] == '3.4.2'


def test_node_closure_requires_collector_dependency(tmp_path):
    source = tmp_path / 'empty'
    source.mkdir()
    with pytest.raises(ValueError, match='just-bash'):
        module.freeze_node_dependencies(source, tmp_path / 'frozen')


def test_archived_commit_is_compiled_instead_of_predecessor_sources(tmp_path, monkeypatch):
    """Exercise git archive and real tsc with a deliberately stale predecessor."""
    repo = tmp_path / 'repo'
    host = repo / 'ts-host'
    (host / 'src/native').mkdir(parents=True)
    (host / 'src/teacher').mkdir()
    (host / 'src/improvement').mkdir()
    (host / 'scripts').mkdir()
    (host / 'src/native/runtime.ts').write_text('export const version = "fresh-commit";\n')
    (host / 'src/native/prompt.ts').write_text('export const prompt = "fresh";\n')
    (host / 'src/teacher/collector.ts').write_text('export { version } from "../native/runtime.js";\n')
    (host / 'src/improvement/authored-source.ts').write_text('export const AUTHORED_IMPROVER = {};\n')
    (host / 'package.json').write_text('{"type":"module"}')
    (host / 'tsconfig.json').write_text(json.dumps({'compilerOptions': {
        'target': 'ES2022', 'module': 'NodeNext', 'rootDir': 'src', 'outDir': 'dist'},
        'include': ['src/**/*.ts']}))
    canonical = Path(__file__).parents[2]
    shutil.copy2(canonical / 'ts-host/scripts/bundle-improver.mjs', host / 'scripts/bundle-improver.mjs')
    app = repo / 'applications/program-improver'
    app.mkdir(parents=True)
    (app / 'main.nl').write_text('Fresh authored program')
    for command in (['init', '-q'], ['add', '.'],
                    ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
                     'commit', '-qm', 'fresh commit']):
        subprocess.run(['git', *command], cwd=repo, check=True, capture_output=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    predecessor = tmp_path / 'predecessor'
    shutil.copytree(host, predecessor / 'ts-host')
    (predecessor / 'ts-host/src/native/runtime.ts').write_text('export const version = "STALE";\n')
    (predecessor / 'ts-host/dist/teacher').mkdir(parents=True)
    (predecessor / 'ts-host/dist/teacher/collector.js').write_text('export const version = "STALE";\n')
    dependencies = tmp_path / 'dependencies'
    (dependencies / 'just-bash').mkdir(parents=True)
    (dependencies / 'just-bash/package.json').write_text('{"version":"3.4.2"}')
    (dependencies / 'typescript').symlink_to((canonical / 'node_modules/typescript').resolve())
    control = tmp_path / 'controls.json'
    control.write_text(json.dumps([{'path': str(host / 'src/native/runtime.ts'),
        'sha256': hashlib.sha256((host / 'src/native/runtime.ts').read_bytes()).hexdigest()}]))
    output = tmp_path / 'frozen'
    monkeypatch.chdir(repo)
    # Protection assembly has its own complete tests. Keep this regression focused on
    # real archive/compile/import behavior, including the formerly overwritten source.
    monkeypatch.setattr(module, 'assemble_protected_runtime_inputs',
                        lambda *a, **kw: {'protected_worlds_sha256': 'a' * 64})
    monkeypatch.setattr(sys, 'argv', ['freeze', '--commit', commit, '--predecessor', str(predecessor),
        '--out', str(output), '--control-bindings', str(control), '--node-dependencies', str(dependencies)])
    module.main()
    value = subprocess.check_output(['node', '--input-type=module', '-e',
        'console.log((await import(process.argv[1])).version)',
        (output / 'ts-host/dist/teacher/collector.js').as_uri()], text=True).strip()
    assert value == 'fresh-commit'
    assert 'fresh-commit' in (output / 'ts-host/src/native/runtime.ts').read_text()
    receipt = json.loads((output / 'runtime-input-closure.json').read_text())
    assert receipt['node_build']['mode'] == 'isolated_typescript'
    assert receipt['node_build']['source_inputs']['ts-host/src/native/runtime.ts'] == \
        hashlib.sha256((host / 'src/native/runtime.ts').read_bytes()).hexdigest()
