"""register_corpora.py: entries and manifests go to origin/main from a private worktree, never via the checkout."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import register_corpora as rc  # noqa: E402
import sync_training_corpora as sync  # noqa: E402

REAL = Path(__file__).resolve().parents[1]


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def setup(tmp_path, monkeypatch):
    origin, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    sh('git', 'init', '-q', '--bare', '-b', 'main', str(origin), cwd=tmp_path)
    sh('git', 'clone', '-q', str(origin), str(repo), cwd=tmp_path)
    for args in (('user.email', 't@t'), ('user.name', 't')):
        sh('git', 'config', *args, cwd=repo)
    (repo / 'scripts').mkdir()
    shutil.copy(REAL / 'scripts/sync_training_corpora.py', repo / 'scripts/sync_training_corpora.py')
    (repo / 'training/corpus-manifests').mkdir(parents=True)
    (repo / 'training/corpus-manifests/.keep').write_text('')
    (repo / '.gitignore').write_text('data/\n')
    registry = {'schema': 'natlang.neuralese-corpus-registry/1', 'corpora': [{'id': 'old', 'owner': 'dgx', 'path': 'x'}]}
    (repo / rc.REGISTRY).write_text(json.dumps(registry, indent=2) + '\n')
    sh('git', 'add', '-A', cwd=repo)
    sh('git', 'commit', '-q', '-m', 'init', cwd=repo)
    sh('git', 'push', '-q', 'origin', 'main', cwd=repo)
    data = repo / 'data/corpora/new'
    data.mkdir(parents=True)
    (data / 'a.jsonl').write_text('{"x": 1}\n')
    monkeypatch.setattr(rc, 'REPO', repo)
    monkeypatch.setattr(rc, 'WORKTREE', tmp_path / 'wt')
    monkeypatch.setattr(rc, 'LOCK', tmp_path / 'lock')
    # Guard: every git call must reach the temporary origin, never the real one.
    real_git = rc.git

    def guarded(*args, cwd=None, check=True):
        url = sh('git', 'remote', 'get-url', 'origin', cwd=cwd or rc.REPO).strip()
        assert url == str(origin), f'test git call reached {url}'
        return real_git(*args, cwd=cwd, check=check)

    monkeypatch.setattr(rc, 'git', guarded)
    return origin, repo


def test_registers_through_worktree_and_leaves_checkout_alone(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    # Another session's uncommitted registry edit in the shared checkout must not be committed.
    (repo / rc.REGISTRY).write_text((repo / rc.REGISTRY).read_text().replace('"x"', '"foreign"'))
    entry = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}
    assert rc.register([entry], 'Register new', check=False) == ['new']
    shown = json.loads(sh('git', 'show', f'main:{rc.REGISTRY}', cwd=origin))
    assert [c['id'] for c in shown['corpora']] == ['old', 'new'] and shown['corpora'][0]['path'] == 'x'
    manifest = json.loads(sh('git', 'show', 'main:training/corpus-manifests/new.json', cwd=origin))
    assert manifest['files'][0]['path'] == 'a.jsonl'
    assert '"foreign"' in (repo / rc.REGISTRY).read_text()  # untouched
    # Idempotent: the same entry again changes nothing.
    assert rc.register([entry], 'again', check=False) == []


def test_retries_after_a_lost_push_race(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    real_attempt, calls = rc.attempt, []

    def racing(*args):
        if not calls:  # someone else pushes first: our first push must lose and the retry rebuilds on top
            calls.append(1)
            other = tmp_path / 'other'
            sh('git', 'clone', '-q', str(origin), str(other), cwd=tmp_path)
            (other / 'other.txt').write_text('x')
            sh('git', '-c', 'user.email=o@o', '-c', 'user.name=o', 'commit', '-q', '-m', 'other', '--allow-empty', cwd=other)
            sh('git', 'push', '-q', 'origin', 'HEAD:main', cwd=other)
            raise rc.Retry('simulated race')
        return real_attempt(*args)

    monkeypatch.setattr(rc, 'attempt', racing)
    monkeypatch.setattr(rc.time, 'sleep', lambda s: None)
    entry = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}
    assert rc.register([entry], 'Register new', check=False) == ['new']
    log = sh('git', 'log', '--format=%s', 'main', cwd=origin).split('\n')
    assert log[:2] == ['Register new', 'other']


def test_lost_push_race_reuses_manifest_and_keeps_concurrent_registry_commit(tmp_path, monkeypatch):
    origin, _repo = setup(tmp_path, monkeypatch)
    real_git, calls = rc.git, []

    def race_on_first_push(*args, cwd=None, check=True):
        if args[:1] == ('push',) and not calls:
            calls.append(True)
            other = tmp_path / 'other'
            sh('git', 'clone', '-q', str(origin), str(other), cwd=tmp_path)
            sh('git', '-c', 'user.email=o@o', '-c', 'user.name=o', 'commit', '-q', '-m', 'other',
               '--allow-empty', cwd=other)
            sh('git', 'push', '-q', 'origin', 'HEAD:main', cwd=other)
        return real_git(*args, cwd=cwd, check=check)

    outputs = []
    real_run = subprocess.run

    def capture_publish_output(command, *args, **kwargs):
        result = real_run(command, *args, **kwargs)
        if isinstance(command, (list, tuple)) and any('sync_training_corpora.py' in str(part) for part in command):
            outputs.append(result.stdout)
        return result

    monkeypatch.setattr(rc, 'git', race_on_first_push)
    monkeypatch.setattr(rc.subprocess, 'run', capture_publish_output)
    monkeypatch.setattr(rc.time, 'sleep', lambda _seconds: None)
    entry = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}
    assert rc.register([entry], 'Register new', check=False) == ['new']
    assert len(outputs) == 2
    assert 'reused_verified_manifest' not in outputs[0]
    assert json.loads(outputs[1])['reused_verified_manifest'] is True
    shown = json.loads(sh('git', 'show', f'main:{rc.REGISTRY}', cwd=origin))
    assert [item['id'] for item in shown['corpora']] == ['old', 'new']
    log = sh('git', 'log', '--format=%s', 'main', cwd=origin).split('\n')
    assert log[:2] == ['Register new', 'other']


def test_changed_entry_under_registered_id_is_refused(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    try:
        rc.register([{'id': 'old', 'owner': 'dgx', 'path': 'y'}], 'm', check=False)
    except ValueError as error:
        assert 'immutable' in str(error)
    else:
        raise AssertionError('expected refusal')


def test_publish_reuses_manifest_only_while_entry_and_file_stats_match(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    data = repo / 'data/corpora/new'
    manifests = repo / 'training/corpus-manifests'
    data.mkdir(parents=True)
    manifests.mkdir(parents=True)
    source = data / 'a.jsonl'
    source.write_text('{"x": 1}\n')
    entry = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}
    cache = tmp_path / 'manifest-cache' / 'new.json'

    sync.publish(repo, entry, cache)
    digest = sync.digest

    def no_rehash(_path):
        raise AssertionError('unchanged files should reuse the verified manifest')

    monkeypatch.setattr(sync, 'digest', no_rehash)
    sync.publish(repo, entry, cache)
    assert json.loads((manifests / 'new.json').read_text())['files'][0]['sha256']

    # A newly selected file invalidates the cached file inventory even when prior files are unchanged.
    hashed = []

    def count_hash(path):
        hashed.append(str(path))
        return digest(path)

    monkeypatch.setattr(sync, 'digest', count_hash)
    (data / 'b.jsonl').write_text('{"y": 1}\n')
    try:
        sync.publish(repo, entry, cache)
    except ValueError as error:
        assert 'immutable snapshot changed' in str(error)
    else:
        raise AssertionError('a new selected file must rebuild, not reuse, the old manifest')
    assert len(hashed) == 2

    # Same path and size still cannot reuse a stale manifest after a content write.
    hashed.clear()
    source.write_text('{"x": 2}\n')
    try:
        sync.publish(repo, entry, cache)
    except ValueError as error:
        assert 'immutable snapshot changed' in str(error)
    else:
        raise AssertionError('changed content must rebuild and reject the immutable snapshot')
    assert len(hashed) == 2
