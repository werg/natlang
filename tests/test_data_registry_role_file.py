"""data_registry.py promote --role: the role's file must be one the dataset holds (semantic-fp M2 tasks are tasks.jsonl;
the grpo-pool default cases.jsonl would have pointed the pool at a missing file)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO / 'scripts'), str(REPO / 'tests/neuralese')]

from natlang_neuralese.common import data_registry as dr  # noqa: E402
from test_data_registry import git_entries, pool, roots  # noqa: E402,F401


def test_promote_refuses_a_role_file_the_dataset_does_not_include(roots, monkeypatch, capsys):  # noqa: F811
    import data_registry as cli
    repo = roots / 'repo'
    pool(roots / 'data_hdd' / 'p1' / 'cases.jsonl', ['a'])
    git_entries(repo, [
        {'id': 'p1', 'owner': 'dgx', 'path': 'data_hdd:p1', 'roles': {'grpo-pool': {'order': 4, 'file': 'cases.jsonl'}}}])
    pool(roots / 'data_hdd' / 'new' / 'tasks.jsonl', ['b'])
    dr.record_dataset('new', roots / 'data_hdd' / 'new', kind='rl-tasks', format='f', license='L', include=['tasks.jsonl'])
    monkeypatch.setattr(dr, 'code_repo', lambda: repo)
    assert cli.main(['promote', 'new', '--role', 'grpo-pool', '--dry-run']) == 1
    assert 'pass --role-file' in capsys.readouterr().err
    assert cli.main(['promote', 'new', '--role', 'grpo-pool', '--role-file', 'tasks.jsonl', '--dry-run']) == 0
    entry = json.loads(capsys.readouterr().out)[0]
    assert entry['roles'] == {'grpo-pool': {'order': 5, 'file': 'tasks.jsonl'}}
