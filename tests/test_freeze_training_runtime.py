import json

import pytest

from scripts.freeze_training_runtime import freeze


def make_runtime(root):
    files = {
        'dist/native/runtime.js': 'export const runtimeVersion = 1;\n',
        'dist/teacher/collector.js': 'export const collectorVersion = 1;\n',
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

    manifest = freeze(source, output)

    assert set(manifest['files']) == set(files)
    assert all((output / name).read_text() == contents for name, contents in files.items())
    assert (output / 'node_modules').is_symlink()
    assert (output / 'node_modules' / 'fixture.txt').read_text() == 'shared dependency'
    assert json.loads((output / 'frozen-runtime.json').read_text()) == manifest


def test_existing_freeze_reuses_its_snapshot_after_source_edits(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    make_runtime(source)
    original = freeze(source, output)
    (source / 'dist/native/runtime.js').write_text('export const runtimeVersion = 2;\n')
    (source / 'scripts/code-corpus/new-script.mjs').write_text('export const later = true;\n')

    reused = freeze(source, output)

    assert reused == original
    assert (output / 'dist/native/runtime.js').read_text() == 'export const runtimeVersion = 1;\n'
    assert not (output / 'scripts/code-corpus/new-script.mjs').exists()


def test_existing_freeze_rejects_snapshot_content_drift(tmp_path):
    source, output = tmp_path / 'ts-host', tmp_path / 'run' / 'runtime-host'
    make_runtime(source)
    freeze(source, output)
    (output / 'dist/native/runtime.js').write_text('tampered\n')

    with pytest.raises(ValueError, match='frozen runtime changed'):
        freeze(source, output)
