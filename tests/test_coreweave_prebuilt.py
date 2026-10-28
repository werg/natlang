"""Prebuilt sandbox artifacts (scripts/coreweave/prebuilt.py) and the job recovery block."""
import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'coreweave'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

import job  # noqa: E402
import prebuilt  # noqa: E402

IMAGE = 'nvcr.io/nvidia/pytorch@sha256:' + 'f' * 64


def make_tar(path: Path, members: dict[str, bytes], links: dict[str, str] | None = None) -> Path:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    path.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    return path


@pytest.fixture
def cas(tmp_path, monkeypatch):
    monkeypatch.setattr(prebuilt, 'CAS', tmp_path / 'cas')
    return tmp_path / 'registry.json'


def test_register_is_content_addressed_and_image_bound_kinds_need_an_image(tmp_path, cas):
    tar = make_tar(tmp_path / 'l.tar.gz', {'llama-cpp-neuralese/bin/llama-server': b'#!x'})
    with pytest.raises(prebuilt.PrebuiltError, match='--image'):
        prebuilt.register('llama', 'llama-cpp', tar, registry_path=cas)
    entry = prebuilt.register('llama', 'llama-cpp', tar, IMAGE, {'git_sha': 'abc'}, registry_path=cas)
    digest = hashlib.sha256(tar.read_bytes()).hexdigest()
    assert entry['sha256'] == digest and Path(entry['path']) == prebuilt.cas_path(digest)
    assert json.loads(cas.read_text())['artifacts']['llama']['git_sha'] == 'abc'


def test_bundle_entries_and_job_image_check(tmp_path, cas):
    tar = make_tar(tmp_path / 'w.tar.gz', {'wheelhouse/added.txt': b''})
    prebuilt.register('wheelhouse', 'wheelhouse', tar, IMAGE, registry_path=cas)
    pairs = prebuilt.bundle_entries(['wheelhouse'], None, registry_path=cas)
    paths = sorted(item['path'] for item, _ in pairs)
    assert paths == ['.coreweave-prebuilt/wheelhouse.json', '.coreweave-prebuilt/wheelhouse.tar.gz']
    manifest = {'files': [item for item, _ in pairs]}
    sources = {item['key']: str(source) for item, source in pairs}
    assert prebuilt.check_job({'id': 'j', 'image': IMAGE}, manifest, sources) == ['wheelhouse']
    with pytest.raises(prebuilt.PrebuiltError, match='built in'):
        prebuilt.check_job({'id': 'j', 'image': 'nvcr.io/nvidia/pytorch:26.09-py3'}, manifest, sources)
    with pytest.raises(prebuilt.PrebuiltError, match='no prebuilt artifact'):
        prebuilt.bundle_entries(['nope'], None, registry_path=cas)


def test_tshost_artifact_is_refused_for_other_sources(tmp_path, cas, monkeypatch):
    tar = make_tar(tmp_path / 't.tar.gz', {'ts-host/dist/index.js': b'1'})
    trees = {path: 'a' * 40 for path in prebuilt.TSHOST_INPUTS}
    prebuilt.register('ts-host', 'ts-host', tar, None, {'inputs': trees}, registry_path=cas)
    monkeypatch.setattr(prebuilt, 'git_trees', lambda sha, repo=None: dict(trees))
    assert len(prebuilt.bundle_entries(['ts-host'], 'HEAD', registry_path=cas)) == 2
    with pytest.raises(prebuilt.PrebuiltError, match='--code-sha'):
        prebuilt.bundle_entries(['ts-host'], None, registry_path=cas)
    monkeypatch.setattr(prebuilt, 'git_trees', lambda sha, repo=None: {**trees, 'ts-host': 'b' * 40})
    with pytest.raises(prebuilt.PrebuiltError, match='ts-host differ'):
        prebuilt.bundle_entries(['ts-host'], 'HEAD', registry_path=cas)


def materialize(root: Path, name: str, kind: str, tar: Path, image=None) -> None:
    directory = root / prebuilt.BUNDLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{name}.tar.gz').write_bytes(tar.read_bytes())
    (directory / f'{name}.json').write_text(json.dumps({'name': name, 'kind': kind, 'image': image,
                                                        'sha256': hashlib.sha256(tar.read_bytes()).hexdigest()}))


def test_install_exports_llama_and_unpacks_tshost(tmp_path):
    root, work = tmp_path / 'repo', tmp_path / 'work'
    (root / 'ts-host' / 'vendor' / 'pkg').mkdir(parents=True)
    materialize(root, 'llama', 'llama-cpp', make_tar(tmp_path / 'l.tar.gz',
                {'llama-cpp-neuralese/bin/llama-server': b'#!/bin/sh\n'}), IMAGE)
    materialize(root, 'ts-host', 'ts-host', make_tar(tmp_path / 't.tar.gz', {'ts-host/dist/index.js': b'x'},
                {'ts-host/node_modules/pkg': '../vendor/pkg'}))
    exports = prebuilt.install(root, work, log=lambda text: None)
    bin_dir = work / 'prebuilt' / 'llama' / 'llama-cpp-neuralese' / 'bin'
    assert f'export NATLANG_LLAMA_SERVER={bin_dir}/llama-server' in exports
    assert 'export NATLANG_CW_TSHOST_PREBUILT=1' in exports
    assert (bin_dir / 'llama-server').is_file() and (root / 'ts-host' / 'dist' / 'index.js').read_text() == 'x'
    assert (root / 'ts-host' / 'node_modules' / 'pkg').is_symlink()


def test_install_refuses_a_tar_that_differs_from_its_descriptor(tmp_path):
    root = tmp_path / 'repo'
    tar = make_tar(tmp_path / 'l.tar.gz', {'a': b'1'})
    materialize(root, 'llama', 'llama-cpp', tar, IMAGE)
    (root / prebuilt.BUNDLE_DIR / 'llama.tar.gz').write_bytes(b'tampered')
    with pytest.raises(prebuilt.PrebuiltError, match='sha256'):
        prebuilt.install(root, tmp_path / 'work', log=lambda text: None)


def recovery_job(**changes):
    value = {'schema': job.SCHEMA, 'id': 'learner', 'kind': 'train', 'placement': {'reason': 'r'}, 'image': IMAGE,
             'resources': {'gpu': 1, 'memory': '64Gi'}, 'lifetime_hours': 2, 'command': ['true'],
             'scratch': {'path': '/work', 'size': '300Gi'}, 'recovery': {'snapshot_interval_seconds': 3600}}
    value.update(changes)
    return value


def test_recovery_block_defaults_and_requires_a_volume():
    assert job.validate(recovery_job())['recovery'] == {'snapshot_interval_seconds': 3600, 'keep': 2}
    with pytest.raises(job.JobError, match='scratch volume'):
        job.validate(recovery_job(scratch={'path': '/work', 'size': '50Gi', 'storage': 'root'}))
    with pytest.raises(job.JobError, match='keep >= 2'):
        job.validate(recovery_job(recovery={'keep': 1}))
    with pytest.raises(job.JobError, match='recovery'):
        job.validate(recovery_job(recovery={'interval': 5}))


def test_lfm3_templates_render_to_valid_jobs_and_run_file(tmp_path):
    spec = importlib.util.spec_from_file_location('lfm3_render', ROOT / 'scripts' / 'coreweave' / 'jobs' / 'lfm3' /
                                                  'render.py')
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    values = {'placeholders': {'RUN_ID': 'lfm3-sft-diloco-test', 'PARENT': '/home/werg/data/p/checkpoint.pt',
                               'TRAINER_ARGS': ['--native-graphs', 'g', '--held-native-graphs', 'h', '--out', '/work/l'],
                               'COORDINATOR_BUNDLE': '/b/c.json', 'LEARNER_BUNDLE': '/b/l.json', 'RECIPE': 'r.json',
                               'STAGE': 's', 'EVAL_ID': 'lfm3-dev-eval-test', 'ACTOR_ID': 'lfm3-actor-test',
                               'GGUF_PATH': '/work/models/m.gguf', 'GGUF_URL': 'https://x/m.gguf',
                               'GGUF_SHA256': 'a' * 64},
              'overrides': {'diloco-sft.run.json': {'settings': {'workers': 2}},
                            'diloco-learner.job.json': {'resources': {'gpu': 1, 'memory': '80Gi'}}}}
    written = {path.name: json.loads(path.read_text()) for path in render.render(values, tmp_path)}
    learner = written['diloco-learner.job.json']
    assert learner['command'][3:] == ['--native-graphs', 'g', '--held-native-graphs', 'h', '--out', '/work/l']
    assert learner['resources'] == {'gpu': 1, 'memory': '80Gi'} and learner['recovery']['keep'] == 2
    run = written['diloco-sft.run.json']
    assert run['settings'] == {'run_id': 'lfm3-sft-diloco-test', 'workers': 2, 'local_steps': 128, 'wire_dtype': 'bf16'}
    assert run['learners']['job'] == str(tmp_path.resolve() / 'diloco-learner.job.json')
    import diloco
    diloco.validate(run)
    for name, value in written.items():
        if name.endswith('.job.json'):
            assert 'gpu' not in value['resources'] or set(value['resources']) <= {'gpu', 'memory'}
    with pytest.raises(SystemExit, match='unfilled'):
        render.render({'placeholders': {}}, tmp_path / 'empty')


def test_bootstrap_is_valid_shell():
    subprocess.run(['bash', '-n', str(ROOT / 'scripts' / 'coreweave' / 'bootstrap.sh')], check=True)
    subprocess.run(['bash', '-n', str(ROOT / 'scripts' / 'coreweave' / 'prebuilt_build.sh')], check=True)
