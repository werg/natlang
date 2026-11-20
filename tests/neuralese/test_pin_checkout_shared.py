"""Pins link the content-addressed model packages instead of copying them (14 GB per pin)."""
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(repo):
    spec = importlib.util.spec_from_file_location('pin_checkout_t', ROOT / 'scripts/pin_checkout.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPO = repo
    git = module.git
    module.git = lambda *args, cwd=repo: git(*args, cwd=cwd)   # its default cwd was bound to the real checkout
    return module


def test_pin_links_native_models_and_copies_other_ignored_state(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)
    run('git', 'init', '-q')
    (repo / '.gitignore').write_text('.natlang/\n')
    run('git', 'add', '.gitignore')
    run('git', '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init')
    (repo / '.natlang/packages/native-models').mkdir(parents=True)
    (repo / '.natlang/packages/native-models/m.pt').write_text('weights')
    (repo / '.natlang/state').mkdir()
    (repo / '.natlang/state/a').write_text('state')
    module = load(repo)
    real = subprocess.run
    monkeypatch.setattr(module.subprocess, 'run', lambda cmd, **kw: None if cmd[0] == 'npm' else real(cmd, **kw))
    dest = tmp_path / 'pin'
    module.create('HEAD', dest)
    models = dest / '.natlang/packages/native-models'
    assert models.is_symlink() and models.resolve() == (repo / '.natlang/packages/native-models').resolve()
    state = dest / '.natlang/state/a'
    assert not state.is_symlink() and state.read_text() == 'state'
