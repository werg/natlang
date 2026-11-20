"""git-history RL source (scripts/rl_tasks/sources/git_history.py): gold answers come from git, and the service-only
reference solver (scripts/rl_tasks/git_history_solve.mjs) reaches the same answers through the model's `git` service,
including a first-parent merge and a file git is told to treat as binary."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.rl_tasks.sources import git_history

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'ts-host'


def _node():
    found = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True)
    return str(found[0]) if found else shutil.which('node')


def _commit(work, message, author, day, files=(), remove=()):
    for path, text in files:
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_text(text)
    for path in remove:
        (work / path).unlink()
    env = {**git_history.GIT_ENV, 'GIT_AUTHOR_NAME': author, 'GIT_AUTHOR_EMAIL': f'{author.split()[0].lower()}@x.org',
           'GIT_COMMITTER_NAME': author, 'GIT_COMMITTER_EMAIL': 'c@x.org',
           'GIT_AUTHOR_DATE': f'2024-03-{day:02d}T12:00:00Z', 'GIT_COMMITTER_DATE': f'2024-03-{day:02d}T12:00:00Z'}
    subprocess.run(['git', '-C', str(work), 'add', '-A'], check=True, env=env)
    subprocess.run(['git', '-C', str(work), 'commit', '-q', '-m', message], check=True, env=env)


def _repo(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(work)], check=True, env=git_history.GIT_ENV)
    body = ['def area(width, height):', '    """Rectangle area of the given sides."""', '    return width * height', '']
    _commit(work, 'Start the geometry helpers', 'Ada Lovelace', 1,
            [('src/geometry.py', '\n'.join(body)), ('README.md', 'Geometry helpers for teaching.\n'),
             ('package.json', json.dumps({'name': 'geo', 'version': '0.1.0'}, indent=2) + '\n')])
    _commit(work, 'feat: add the data table', 'Ada Lovelace', 2, [('data/table.csv', 'x' * 400 + '\n')])
    for i in range(3, 27):
        author = 'Grace Hopper' if i % 3 == 0 else 'Ada Lovelace'
        files = [(f'src/mod{i}.py', f'def helper_{i}(value):\n    return value + {i}\n'),
                 ('notes/log.md', f'Changelog entry number {i} describing the geometry helpers.\n' * (i % 4 + 1))]
        if i == 8:
            body[0] = 'def area(width, height, scale=1):'
            body[2] = '    return width * height * scale'
            files.append(('src/geometry.py', '\n'.join(body)))
        if i == 12:
            files.append(('package.json', json.dumps({'name': 'geo', 'version': '0.2.0'}, indent=2) + '\n'))
        if i == 15:
            files.append(('data/table.csv', 'y' * 400 + '\n'))
        if i == 20:
            body.insert(3, 'def perimeter(width, height):\n    """Perimeter of the rectangle with these sides."""')
            files.append(('src/geometry.py', '\n'.join(body)))
        _commit(work, f'fix: update helper module {i} for the geometry notes' if i % 2 else
                f'Extend the helper module number {i}', author, i, files, remove=['src/mod5.py'] if i == 22 else ())
    subprocess.run(['git', '-C', str(work), 'checkout', '-q', '-b', 'side', 'HEAD~2'], check=True, env=git_history.GIT_ENV)
    _commit(work, 'Side branch rewrites the readme', 'Grace Hopper', 27, [('README.md', 'Geometry helpers for classes.\n')])
    subprocess.run(['git', '-C', str(work), 'checkout', '-q', 'main'], check=True, env=git_history.GIT_ENV)
    env = {**git_history.GIT_ENV, 'GIT_AUTHOR_NAME': 'Ada Lovelace', 'GIT_AUTHOR_EMAIL': 'ada@x.org',
           'GIT_COMMITTER_NAME': 'Ada Lovelace', 'GIT_COMMITTER_EMAIL': 'ada@x.org',
           'GIT_AUTHOR_DATE': '2024-03-28T12:00:00Z', 'GIT_COMMITTER_DATE': '2024-03-28T12:00:00Z'}
    subprocess.run(['git', '-C', str(work), 'merge', '-q', '--no-ff', '-m', 'Merge the readme rewrite', 'side'],
                   check=True, env=env)
    _commit(work, 'Bump the package version again', 'Grace Hopper', 29,
            [('package.json', json.dumps({'name': 'geo', 'version': '0.3.0'}, indent=2) + '\n')])
    repos = tmp_path / 'repos'
    (repos / 'someone').mkdir(parents=True)
    subprocess.run(['git', 'clone', '-q', '--bare', str(work), str(repos / 'someone' / 'geo')], check=True,
                   env=git_history.GIT_ENV)
    return repos


def test_git_history_gold_matches_the_service(tmp_path, monkeypatch):
    if not (RUNTIME / 'dist/native/external.js').exists() or not _node():
        pytest.skip('needs the built ts-host runtime and node')
    monkeypatch.setattr(git_history, 'MAX_BLOB', 300)
    repos = _repo(tmp_path)
    assets = tmp_path / 'assets'
    cases = git_history.cases(repos=str(repos), scan=5, want=1, window=40, out_assets=str(assets),
                              report=str(tmp_path / 'repos.jsonl'))
    kinds = {c['curriculum']['external']['git']['kind'] for c in cases}
    assert {'line_origin', 'commit_files', 'count_commits', 'churn'} <= kinds, kinds
    asset = next(assets.glob('git-history/*.json'))
    snap = json.loads(asset.read_text())
    assert snap['binary_attributes'] == ['data/table.csv']          # larger than MAX_BLOB: git reports it as binary
    assert any(len(c['parents']) == 2 for c in snap['commits'])     # the first-parent merge is in the window
    assert all(c['split'] in ('train', 'dev', 'confirmation') for c in cases)
    for case in cases:
        assert case['semantics']['services']['git'].count(str(assets)) == 1
        assert all(json.dumps(w, sort_keys=True) != json.dumps(case['semantics']['expected'], sort_keys=True)
                   for w in case['curriculum']['external']['checks']['wrong_answers'])
    rows = tmp_path / 'cases.jsonl'
    rows.write_text(''.join(json.dumps(c) + '\n' for c in cases))
    out = tmp_path / 'solve.json'
    subprocess.run([_node(), str(ROOT / 'scripts/rl_tasks/git_history_solve.mjs'), str(RUNTIME), str(rows), str(out)],
                   check=True, capture_output=True, text=True, timeout=120, env={**os.environ})
    report = json.loads(out.read_text())
    assert report['failed'] == [] and report['agreed'] == len(cases)
