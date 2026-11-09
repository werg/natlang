"""Code identity is recorded provenance (common/code_provenance.py)."""
import json
import subprocess
from pathlib import Path

from natlang_neuralese.common import code_provenance as cp


def init(repo):
    repo.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@e.org', *a],
                                    check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    (repo / 'a.py').write_text('1\n')
    git('add', 'a.py')
    git('commit', '-qm', 'one')
    return git


def test_the_default_root_is_this_checkout_and_the_first_read_is_cached_for_the_process():
    assert cp.repository_root() == Path(__file__).resolve().parents[2]
    first = cp.code_provenance()
    assert first['schema'] == cp.SCHEMA and (first['git_commit'] is None or len(first['git_commit']) == 40)
    assert cp.code_provenance() == first


def test_commit_dirty_flag_and_fresh_reads_follow_the_checkout(tmp_path):
    git = init(tmp_path / 'repo')
    clean = cp.code_provenance(tmp_path / 'repo', fresh=True)
    assert clean == {'schema': cp.SCHEMA, 'git_commit': git('rev-parse', 'HEAD'), 'dirty': False}
    (tmp_path / 'repo/a.py').write_text('2\n')
    assert cp.code_provenance(tmp_path / 'repo', fresh=True)['dirty'] is True
    git('commit', '-qam', 'two')
    newer = cp.code_provenance(tmp_path / 'repo', fresh=True)
    assert newer['git_commit'] != clean['git_commit'] and newer['dirty'] is False
    assert cp.code_changes(clean, newer) == {'git_commit': [clean['git_commit'], newer['git_commit']]}
    assert cp.code_changes(newer, newer) == {}


def test_archive_trees_and_containers_name_their_commit_without_git(tmp_path, monkeypatch):
    (tmp_path / 'runtime-input-closure.json').write_text(json.dumps({'commit': 'a' * 40}))
    assert cp.compute(tmp_path) == {'schema': cp.SCHEMA, 'git_commit': 'a' * 40, 'dirty': None}
    monkeypatch.setenv('NATLANG_CODE_COMMIT', 'b' * 40)
    monkeypatch.setenv('NATLANG_CODE_DIRTY', '1')
    assert cp.compute(tmp_path) == {'schema': cp.SCHEMA, 'git_commit': 'b' * 40, 'dirty': True}
    monkeypatch.delenv('NATLANG_CODE_COMMIT')
    assert cp.compute(tmp_path / 'nowhere')['git_commit'] is None   # never raises
