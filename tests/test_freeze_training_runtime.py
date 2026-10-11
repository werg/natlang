import json
import shutil
from pathlib import Path

import pytest

from scripts.freeze_training_runtime import freeze


def make_runtime(root):
    files = {
        'dist/native/runtime.js': 'export const runtimeVersion = 1;\n',
        'dist/native/prompt.js': 'export const prompt = "Fixture runtime instructions";\n',
        'dist/teacher/collector.js': 'export const collectorVersion = 1;\n',
        # The provider entrypoint the freeze preflight imports (9610fb65).
        'dist/model/pi-provider.js': 'export const providerVersion = 1;\n',
        'src/native/runtime.ts': 'export const runtimeVersion = 1;\n',
        'scripts/code-corpus/replay.mjs': 'export const replayVersion = 1;\n',
        'prelude.js': 'export const preludeVersion = 1;\n',
        'package.json': '{"name":"fixture","type":"module"}\n',
        'package-lock.json': '{"lockfileVersion":3}\n',
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (root / 'node_modules').mkdir()
    (root / 'node_modules' / 'fixture.txt').write_text('shared dependency')
    return files


def test_freeze_copies_runtime_scripts_and_shares_node_modules(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    files = make_runtime(source)

    manifest = freeze(source, output, compiled_dist=source / 'dist')

    assert manifest['build']['mode'] == 'explicit_compiled_dist'
    assert set(manifest['files']) == set(files)
    assert all((output / name).read_text() == contents for name, contents in files.items())
    # 9610fb65: node_modules is a merged directory of links (local packages over the repository root's hoisted ones).
    assert (output / 'node_modules').is_dir() and not (output / 'node_modules').is_symlink()
    assert (output / 'node_modules' / 'fixture.txt').is_symlink()
    assert (output / 'node_modules' / 'fixture.txt').read_text() == 'shared dependency'
    assert json.loads((output / 'frozen-runtime.json').read_text()) == manifest


def test_freeze_keeps_the_source_review_registry_the_runtime_resolves(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    make_runtime(source)
    reviews = tmp_path / 'training/source-reviews'
    reviews.mkdir(parents=True)
    (reviews / 'holds.jsonl').write_text('{"id":"hold-1"}\n')
    (reviews / 'datasets.json').write_text('{}\n')

    manifest = freeze(source, output, compiled_dist=source / 'dist')
    (reviews / 'holds.jsonl').write_text('{"id":"hold-2"}\n')

    # dist/teacher/source-review.js searches upward for training/source-reviews/holds.jsonl.
    assert (output / 'training/source-reviews/holds.jsonl').read_text() == '{"id":"hold-1"}\n'
    assert 'training/source-reviews/datasets.json' in manifest['files']
    assert freeze(source, output, compiled_dist=source / 'dist') == manifest


def test_existing_freeze_reuses_its_snapshot_after_source_edits(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    make_runtime(source)
    original = freeze(source, output, compiled_dist=source / 'dist')
    (source / 'dist/native/runtime.js').write_text('export const runtimeVersion = 2;\n')
    (source / 'scripts/code-corpus/new-script.mjs').write_text('export const later = true;\n')

    reused = freeze(source, output, compiled_dist=source / 'dist')

    assert reused == original
    assert (output / 'dist/native/runtime.js').read_text() == 'export const runtimeVersion = 1;\n'
    assert not (output / 'scripts/code-corpus/new-script.mjs').exists()


def test_existing_freeze_rejects_snapshot_content_drift(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    make_runtime(source)
    freeze(source, output, compiled_dist=source / 'dist')
    (output / 'dist/native/runtime.js').write_text('tampered\n')

    with pytest.raises(ValueError, match='frozen runtime changed'):
        freeze(source, output, compiled_dist=source / 'dist')


def test_freeze_rejects_advertised_finish_without_compiled_support(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'runtime-host'
    make_runtime(source)
    (source / 'dist/native/prompt.js').write_text('export const prompt = "eval({finish:true})";\n')
    (source / 'dist/native/agent.js').write_text('export const schema = {};\n')
    with pytest.raises(ValueError, match='without its compiled tool schema'):
        freeze(source, output, compiled_dist=source / 'dist')
    assert not (output / 'frozen-runtime.json').exists()


def test_freeze_accepts_finish_when_schema_and_runtime_both_support_it(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'runtime-host'
    make_runtime(source)
    (source / 'dist/native/prompt.js').write_text('export const prompt = "eval({finish:true})";\n')
    (source / 'dist/native/agent.js').write_text('export const schema = {finish: {type: "boolean"}};\n')
    (source / 'dist/native/runtime.js').write_text('export const isFinish = args => args.finish === true;\n')
    assert 'dist/native/agent.js' in freeze(source, output, compiled_dist=source / 'dist')['files']


def make_compile_project(repo):
    host = repo / 'ts-host'
    (host / 'src/native').mkdir(parents=True)
    (host / 'scripts').mkdir()
    (repo / 'applications/program-improver').mkdir(parents=True)
    (repo / 'applications/program-improver/fixture.ts').write_text('export const fixture = true;\n')
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy2(project_root / 'ts-host/scripts/bundle-improver.mjs', host / 'scripts/bundle-improver.mjs')
    (host / 'src/native/runtime.ts').write_text('export const runtimeVersion = 2;\nexport const finishSupported = (args: { finish?: boolean }) => args.finish === true;\n')
    (host / 'src/native/prompt.ts').write_text('export const prompt = "fixture runtime prompt";\n')
    (host / 'src/native/agent.ts').write_text('export const schema = { finish: { type: "boolean" } };\n')
    (host / 'src/model').mkdir()
    (host / 'src/model/pi-provider.ts').write_text('export const providerVersion = 1;\n')  # freeze preflight imports it
    (host / 'tsconfig.json').write_text(json.dumps({
        'compilerOptions': {'target': 'ES2022', 'module': 'NodeNext', 'moduleResolution': 'NodeNext',
                            'strict': True, 'declaration': True, 'outDir': 'dist', 'rootDir': 'src',
                            'skipLibCheck': True}, 'include': ['src/**/*.ts']}))
    (host / 'package.json').write_text('{"name":"fixture","type":"module"}\n')
    source_modules = project_root / 'ts-host/node_modules'
    (host / 'node_modules').symlink_to(source_modules, target_is_directory=True)
    stale = host / 'dist/native'
    stale.mkdir(parents=True)
    (stale / 'runtime.js').write_text('export const runtimeVersion = 1;\n')
    (stale / 'prompt.js').write_text('export const prompt = "old";\n')
    (stale / 'agent.js').write_text('export const schema = {};\n')
    return host


def test_new_default_freeze_compiles_current_typescript_in_isolation(tmp_path):
    source = make_compile_project(tmp_path / 'project')
    output = tmp_path / 'run/runtime-host'

    manifest = freeze(source, output)

    assert manifest['build']['mode'] == 'isolated_typescript'
    assert manifest['build']['source_inputs_sha256']
    assert 'dist/native/runtime.js' in manifest['files']
    assert 'src/native/runtime.ts' in manifest['files']
    assert 'runtimeVersion = 2' in (output / 'dist/native/runtime.js').read_text()
    assert 'runtimeVersion = 1' in (source / 'dist/native/runtime.js').read_text()
    assert 'runtimeVersion = 2' in (output / 'src/native/runtime.ts').read_text()
    assert not (source / 'src/improvement/authored-source.ts').exists()

    (source / 'src/native/runtime.ts').write_text('export const runtimeVersion = 4;\n')
    fresh = freeze(source, tmp_path / 'run/runtime-host-v2')
    assert fresh['build']['source_inputs_sha256'] != manifest['build']['source_inputs_sha256']
    assert 'runtimeVersion = 4' in (tmp_path / 'run/runtime-host-v2/dist/native/runtime.js').read_text()
    assert 'runtimeVersion = 2' in (output / 'dist/native/runtime.js').read_text()


def test_existing_default_freeze_reuses_snapshot_without_recompiling(tmp_path, monkeypatch):
    source = make_compile_project(tmp_path / 'project')
    output = tmp_path / 'run/runtime-host'
    original = freeze(source, output)
    (source / 'src/native/runtime.ts').write_text('export const runtimeVersion = 3;\n')

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError('existing frozen runtime must be reused without a new build')

    monkeypatch.setattr('scripts.freeze_training_runtime.isolated_compiled_runtime', unexpected_compile)
    reused = freeze(source, output)

    assert reused == original
    assert 'runtimeVersion = 2' in (output / 'dist/native/runtime.js').read_text()
