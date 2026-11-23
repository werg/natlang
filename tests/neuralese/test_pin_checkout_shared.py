"""Experiment pins survive interrupted preparation without trusting foreign or dirty checkouts."""
import importlib.util
import json
import multiprocessing
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(repo):
    spec = importlib.util.spec_from_file_location('pin_checkout_t', ROOT / 'scripts/pin_checkout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = repo
    return module


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    def run(*args):
        return subprocess.run(args, cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    run('git', 'init', '-q')
    (repo / '.gitignore').write_text('.natlang/\nts-host/dist/\n')
    (repo / 'source.txt').write_text('tracked source')
    (repo / 'ts-host').mkdir()
    (repo / 'ts-host/package.json').write_text('{}')
    run('git', 'add', '.')
    run('git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init')
    return repo


def npm_stub(module, monkeypatch, build):
    real = subprocess.run
    def run(cmd, **kwargs):
        if cmd[0] == 'npm':
            return build(**kwargs)
        return real(cmd, **kwargs)
    monkeypatch.setattr(module.subprocess, 'run', run)


def test_pin_links_native_models_and_copies_other_ignored_state(repository, tmp_path, monkeypatch):
    repo = repository
    (repo / '.natlang/packages/native-models').mkdir(parents=True)
    (repo / '.natlang/packages/native-models/m.pt').write_text('weights')
    (repo / '.natlang/state').mkdir()
    (repo / '.natlang/state/a').write_text('state')
    module = load(repo)
    npm_stub(module, monkeypatch, lambda **kw: None)
    dest = tmp_path / 'pin'
    module.create('HEAD', dest)
    models = dest / '.natlang/packages/native-models'
    assert models.is_symlink() and models.resolve() == (repo / '.natlang/packages/native-models').resolve()
    state = dest / '.natlang/state/a'
    assert not state.is_symlink() and state.read_text() == 'state'


def test_interrupted_build_resumes_exact_registered_checkout_and_rebuilds(repository, tmp_path, monkeypatch):
    module = load(repository)
    models = repository / '.natlang/packages/native-models'
    models.mkdir(parents=True)
    (models / 'm.pt').write_text('weights')
    dest = tmp_path / 'pin'
    builds = []
    def build(**kwargs):
        assert not (dest / '.natlang-pin.json').exists()
        assert kwargs['pass_fds']  # ownership survives parent death while build child runs
        builds.append(kwargs['cwd'])
        output = dest / 'ts-host/dist'
        output.mkdir(exist_ok=True)
        (output / 'built').write_text(str(len(builds)))
        if len(builds) == 1:
            raise subprocess.CalledProcessError(143, ['npm'])
    npm_stub(module, monkeypatch, build)
    with pytest.raises(subprocess.CalledProcessError):
        module.create('HEAD', dest)
    assert dest.exists() and not (dest / '.natlang-pin.json').exists()
    assert module.create('HEAD', dest) == dest
    assert (dest / 'ts-host/dist/built').read_text() == '2'
    assert json.loads((dest / '.natlang-pin.json').read_text())['commit'] == module.git('rev-parse', 'HEAD').strip()
    assert (dest / '.natlang/packages/native-models').resolve() == models
    module.create('HEAD', dest)  # completed pin is reused without a third build
    assert len(builds) == 2


@pytest.mark.parametrize('completed', [False, True])
@pytest.mark.parametrize('damage', ['wrong-commit', 'tracked-change', 'staged-change'])
def test_rejects_mismatched_or_dirty_pin_even_with_ready_marker(repository, tmp_path, monkeypatch, completed, damage):
    module = load(repository)
    npm_stub(module, monkeypatch, lambda **kw: None)
    dest = module.create('HEAD', tmp_path / 'pin')
    if not completed:
        (dest / '.natlang-pin.json').unlink()
    if damage == 'wrong-commit':
        (repository / 'source.txt').write_text('second commit')
        module.git('add', 'source.txt')
        module.git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'second')
    else:
        (dest / 'source.txt').write_text('dirty')
        if damage == 'staged-change':
            module.git('add', 'source.txt', cwd=dest)
    with pytest.raises(SystemExit, match='detached checkout|modified tracked'):
        module.create('HEAD', dest)


def test_rejects_foreign_directory_symlink_and_unregistered_git_copy(repository, tmp_path, monkeypatch):
    module = load(repository)
    npm_stub(module, monkeypatch, lambda **kw: None)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / '.natlang-pin.json').write_text(json.dumps({'commit': module.git('rev-parse', 'HEAD').strip()}))
    with pytest.raises(SystemExit, match='registered pin'):
        module.create('HEAD', foreign)
    actual = module.create('HEAD', tmp_path / 'pin')
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(SystemExit, match='registered pin'):
        module.create('HEAD', alias)
    clone = tmp_path / 'clone'
    module.git('clone', '-q', str(repository), str(clone))
    with pytest.raises(SystemExit, match='registered pin'):
        module.create('HEAD', clone)


def test_atomic_marker_failure_can_retry_without_false_completion(repository, tmp_path, monkeypatch):
    module = load(repository)
    npm_stub(module, monkeypatch, lambda **kw: None)
    replace = module.os.replace
    def fail_replace(*args):
        raise OSError('publication interrupted')
    monkeypatch.setattr(module.os, 'replace', fail_replace)
    dest = tmp_path / 'pin'
    with pytest.raises(OSError, match='publication interrupted'):
        module.create('HEAD', dest)
    assert not (dest / '.natlang-pin.json').exists()
    monkeypatch.setattr(module.os, 'replace', replace)
    module.create('HEAD', dest)
    assert json.loads((dest / '.natlang-pin.json').read_text())['source'] == str(repository)


def test_remove_serializes_with_inflight_create(repository, tmp_path, monkeypatch):
    module = load(repository)
    context = multiprocessing.get_context('fork')
    entered, proceed, removing = context.Event(), context.Event(), context.Event()
    dest = tmp_path / 'pin'
    def build(**kwargs):
        entered.set()
        assert proceed.wait(10)
    npm_stub(module, monkeypatch, build)
    creator = context.Process(target=module.create, args=('HEAD', dest))
    def remove():
        removing.set()
        module.remove(dest)
    remover = context.Process(target=remove)
    creator.start()
    try:
        assert entered.wait(10)
        remover.start()
        assert removing.wait(10)
        time.sleep(0.1)
        assert remover.is_alive() and dest.is_dir()
        assert not (dest / '.natlang-pin.json').exists()
        proceed.set()
        creator.join(10)
        remover.join(10)
        assert creator.exitcode == 0 and remover.exitcode == 0
        assert not dest.exists()
    finally:
        proceed.set()
        for process in (creator, remover):
            if process.pid and process.is_alive():
                process.terminate()
                process.join(10)


def test_concurrent_creators_reuse_one_completed_build(repository, tmp_path, monkeypatch):
    module = load(repository)
    context = multiprocessing.get_context('fork')
    entered, proceed = context.Event(), context.Event()
    builds = context.Value('i', 0)
    dest = tmp_path / 'pin'
    def build(**kwargs):
        with builds.get_lock():
            builds.value += 1
        entered.set()
        assert proceed.wait(10)
    npm_stub(module, monkeypatch, build)
    creators = [context.Process(target=module.create, args=('HEAD', dest)) for _ in range(2)]
    creators[0].start()
    try:
        assert entered.wait(10)
        creators[1].start()
        time.sleep(0.1)
        assert creators[1].is_alive()
        proceed.set()
        for process in creators:
            process.join(10)
            assert process.exitcode == 0
        assert builds.value == 1
        assert json.loads((dest / '.natlang-pin.json').read_text())['commit'] == module.git('rev-parse', 'HEAD').strip()
    finally:
        proceed.set()
        for process in creators:
            if process.pid and process.is_alive():
                process.terminate()
                process.join(10)


def test_build_cannot_publish_modified_tracked_sources(repository, tmp_path, monkeypatch):
    module = load(repository)
    dest = tmp_path / 'pin'
    def build(**kwargs):
        (dest / 'source.txt').write_text('changed by build')
    npm_stub(module, monkeypatch, build)
    with pytest.raises(SystemExit, match='modified tracked'):
        module.create('HEAD', dest)
    assert not (dest / '.natlang-pin.json').exists()


def test_interrupted_build_reuses_dangling_leaf_root_and_shared_links(repository, tmp_path, monkeypatch):
    module = load(repository)
    with (repository / '.gitignore').open('a') as stream:
        stream.write('runs/\ndata/\n.venv/\n')
    for name in ('runs', 'data'):
        (repository / name).mkdir()
        (repository / name / 'README').write_text('tracked root container')
        module.git('add', '-f', f'{name}/README')
        (repository / name / 'missing').symlink_to(repository / f'unavailable-{name}')
    module.git('add', '.gitignore')
    module.git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'tracked data containers')
    (repository / '.venv').symlink_to(repository / 'unavailable-venv')
    models = repository / '.natlang/packages/native-models'
    models.parent.mkdir(parents=True)
    models.symlink_to(repository / 'unavailable-models')
    builds = 0
    def build(**kwargs):
        nonlocal builds
        builds += 1
        if builds == 1:
            raise subprocess.CalledProcessError(143, ['npm'])
    npm_stub(module, monkeypatch, build)
    dest = tmp_path / 'pin'
    with pytest.raises(subprocess.CalledProcessError):
        module.create('HEAD', dest)
    paths = ('runs/missing', 'data/missing', '.venv', '.natlang/packages/native-models')
    for name in paths:
        assert (dest / name).is_symlink() and not (dest / name).exists()
    module.create('HEAD', dest)
    for name in paths:
        assert (dest / name).readlink() == repository / name
    assert builds == 2 and (dest / '.natlang-pin.json').exists()


@pytest.mark.parametrize('conflict', ['foreign-link', 'directory', 'file'])
def test_shared_links_reject_conflicting_paths_without_replacing_them(tmp_path, conflict):
    module = load(tmp_path)
    target, intended = tmp_path / 'link', tmp_path / 'intended'
    if conflict == 'foreign-link':
        target.symlink_to(tmp_path / 'wrong-dangling-target')
    elif conflict == 'directory':
        target.mkdir()
    else:
        target.write_text('foreign bytes')
    with pytest.raises(SystemExit, match='intended shared'):
        module.ensure_link(intended, target)
    if conflict == 'foreign-link':
        assert target.readlink() == tmp_path / 'wrong-dangling-target'
    elif conflict == 'directory':
        assert target.is_dir()
    else:
        assert target.read_text() == 'foreign bytes'


def test_nested_ignored_venv_container_and_model_store_resume_without_following_links(repository, tmp_path, monkeypatch):
    module = load(repository)
    venv = repository / '.venv-decider-systempython'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'lib').mkdir()
    (venv / '.gitignore').write_text('*\n')
    (venv / 'bin/python').write_text('canonical interpreter')
    (venv / 'pyvenv.cfg').write_text('canonical config')
    models = repository / '.natlang/packages/native-models'
    models.mkdir(parents=True)
    (models / '.gitignore').write_text('*\n')
    (models / 'weights.pt').write_text('unique model weights')
    (repository / '.natlang/model-alias').symlink_to(models)
    with (repository / '.gitignore').open('a') as stream:
        stream.write('.venv-decider-systempython/\n')
    module.git('add', '.gitignore')
    module.git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'nested ignored environment')
    dest = tmp_path / 'pin'
    module.git('worktree', 'add', '--detach', str(dest), 'HEAD')
    # Reproduce scaffolding left by the old helper: a shared root directory
    # containing exact aliases to individual canonical children.
    (dest / venv.name).mkdir()
    for child in venv.iterdir():
        (dest / venv.name / child.name).symlink_to(child)
    calls = []
    real_git = module.git
    def git(*args, **kwargs):
        if args[:2] == ('ls-files', '-o'):
            return '\n'.join([
                '.venv-decider-systempython/', '.venv-decider-systempython/bin/',
                '.venv-decider-systempython/bin/python', '.venv-decider-systempython/.gitignore',
                '.natlang/', '.natlang/packages/', '.natlang/packages/native-models/',
                '.natlang/packages/native-models/weights.pt', '.natlang/model-alias',
            ]) + '\n'
        return real_git(*args, **kwargs)
    monkeypatch.setattr(module, 'git', git)
    real = subprocess.run
    builds = 0
    before = (models / 'weights.pt').stat()
    def run(cmd, **kwargs):
        nonlocal builds
        if cmd[0] == 'npm':
            builds += 1
            if builds == 1:
                raise subprocess.CalledProcessError(143, cmd)
            return None
        if cmd[0] == 'rsync':
            calls.append(cmd)
            assert 'native-models' not in cmd[-2]  # never an exact/descendant source copy
            assert any(arg.startswith('--exclude=') and 'native-models' in arg for arg in cmd)
        return real(cmd, **kwargs)
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        module.create('HEAD', dest)
    module.create('HEAD', dest)
    assert builds == 2 and calls
    assert not (dest / venv.name).is_symlink()
    assert all(child.is_symlink() and child.readlink() == venv / child.name for child in (dest / venv.name).iterdir())
    assert (dest / '.natlang/packages/native-models').readlink() == models
    assert (dest / '.natlang/model-alias').is_symlink()
    after = (models / 'weights.pt').stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert (models / 'weights.pt').read_text() == 'unique model weights'


def test_shared_container_rejects_unique_files_without_deleting_them(tmp_path):
    module = load(tmp_path)
    source, target = tmp_path / 'canonical', tmp_path / 'view'
    source.mkdir(); target.mkdir()
    (target / 'unique').write_text('must survive')
    with pytest.raises(SystemExit, match='unique data'):
        module.ensure_shared_view(source, target)
    assert (target / 'unique').read_text() == 'must survive'
