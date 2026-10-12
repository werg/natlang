#!/usr/bin/env python3
"""Standing full check of origin/main on the DGX for both machines (coordinator 2026-10-12).

Pop's GPU and memory cannot run the full main check (Python suite, ts-host build, the full TS test files, native
conformance: everything `scripts/check-main.sh` runs). A user timer (`natlang-full-check-main.timer`, every 30 min)
runs `full_check_service.py tick`:

1. A run in progress whose ledger unit has ended is finished: its summary and failures are read from the log, a coord
   note goes to Pop with the commit, the commands, the log path and the failures; every open request to DGX that asks
   for a full check (matches FULL_CHECK) and names commits contained in the checked commit is answered; each failing
   step or test is routed with the last commit that touched its test file (sent to all). The worktree is removed.
2. When no run is in progress and origin/main moved since the last completed run, a clean worktree at origin/main is checked out
   and `scripts/check-main.sh` runs there as ledger unit natlang-full-check-<sha> (medium priority, resumable: a
   preempted run restarts from scratch on relaunch).

State: ~/.local/state/natlang/full-check.json; logs: ~/data/full-check/<sha>.log (kept).
  full_check_service.py tick        (what the timer runs)
  full_check_service.py status
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE = Path.home() / '.local/state/natlang/full-check.json'
BASE = Path.home() / 'data/full-check'
AS = 'dgx-full-check'
FULL_CHECK = re.compile(r'full (main )?check|check-main|full grader|grader check', re.I)
COMMIT = re.compile(r'\b[0-9a-f]{7,40}\b')


def git(*args: str, cwd: Path = REPO, check: bool = True) -> str:
    return subprocess.run(['git', *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + '\n')
    tmp.replace(STATE)


def unit_live(unit: str) -> bool:
    for name in (unit, unit + '-admission'):
        if subprocess.run(['systemctl', '--user', 'is-active', '-q', f'{name}.service']).returncode == 0:
            return True
    try:
        status = json.loads(subprocess.run(['python3', str(REPO / 'scripts/memory_ledger.py'), 'status'],
                                           capture_output=True, text=True).stdout)
        return any(w['unit'] == f'{unit}.service' for w in status.get('waiting', []))
    except (ValueError, KeyError, TypeError):
        return True  # unknown: do not declare it finished


def coord(*args: str) -> str:
    return subprocess.run(['python3', str(REPO / 'scripts/coord.py'), '--as', AS, *args], cwd=REPO,
                          capture_output=True, text=True).stdout


def launch(state: dict, sha: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    wt = BASE / f'wt-{sha[:10]}'
    if wt.exists():
        git('worktree', 'remove', '--force', str(wt), check=False)
    git('worktree', 'add', '-q', '--detach', str(wt), sha)
    log = BASE / f'{sha[:10]}.log'
    unit = f'natlang-full-check-{sha[:10]}'
    command = f'scripts/check-main.sh > {log} 2>&1; echo exit=$? >> {log}'
    # A detached process still belongs to the timer service's cgroup and dies
    # when its oneshot exits. Keep the waiting admission client in its own
    # managed service; the admitted check has its separate ledger-managed unit.
    subprocess.run(['systemd-run', '--user', '--collect', '--unit', unit + '-admission',
                    '--property=Type=exec',
                    '--property=StandardOutput=append:' + str(BASE / f'{sha[:10]}.ledger.log'),
                    '--property=StandardError=inherit',
                    'python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', '28',
                    '--priority', 'medium', '--resumable', '--wait', '86400', '--workdir', str(wt), '--',
                    'sh', '-c', command], check=True)
    state['running'] = {'sha': sha, 'unit': unit, 'worktree': str(wt), 'log': str(log), 'started': time.time(),
                        'command': f'python3 scripts/memory_ledger.py run --unit {unit} --budget-gb 28 --priority '
                                   f'medium --resumable --workdir {wt} -- sh -c "{command}"'}
    save(state)


def failures(text: str) -> dict:
    summary = text[text.rfind('=== summary'):] if '=== summary' in text else ''
    steps = [line.split()[0] for line in summary.splitlines()[1:] if 'FAILED' in line]
    tests = sorted(set(re.findall(r'FAILED (tests/\S+?)(?:::| )', text)))           # pytest
    tests += sorted(set(re.findall(r'^\s*(?:✖|not ok \d+ -) .*?(test/[\w./-]+\.test\.mjs)', text, re.M)))  # node --test
    tests += sorted(set(re.findall(r'^FAIL (test/[\w./-]+\.mjs)', text, re.M)))     # test-files.mjs
    exit_code = re.findall(r'^exit=(\d+)', text, re.M)
    return {'summary': summary.strip(), 'failed_steps': steps, 'failed_tests': sorted(set(tests)),
            'exit': int(exit_code[-1]) if exit_code else None}


def owner_of(test: str, sha: str) -> str:
    path = test.split('::')[0]
    rel = path if path.startswith('tests/') else f'ts-host/{path}'
    return git('log', '-1', '--format=%h %s', sha, '--', rel, check=False) or 'unknown'


def open_full_check_requests() -> list[tuple[str, str]]:
    out = []
    for rid in re.findall(r'── (\S+)\s+request', coord('open')):
        if '-pop-' not in rid:
            continue
        body = coord('show', rid)
        if FULL_CHECK.search(body):
            out.append((rid, body))
    return out


def contained(commit: str, sha: str) -> bool:
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, sha], cwd=REPO,
                          capture_output=True).returncode == 0


def finish(state: dict) -> None:
    run = state['running']
    log = Path(run['log'])
    text = log.read_text(errors='replace') if log.exists() else ''
    result = failures(text)
    complete = result['exit'] is not None and bool(result['summary'])
    sha = run['sha']
    owners = {t: owner_of(t, sha) for t in result['failed_tests']} if complete else {}
    lines = [f'Full main check of {sha[:10]} ({git("log", "-1", "--format=%s", sha)}): '
             + ('INCOMPLETE (no full-suite verdict; retry required)' if not complete else
                'PASSED' if result['exit'] == 0 else f'FAILED (exit {result["exit"]})'),
             f'Command: {run["command"]}', f'Log (DGX): {log}', '', result['summary'] or '(no summary: run incomplete)']
    if owners:
        lines += ['', 'Failing tests (last commit touching the test file):'] + [f'- {t}: {o}' for t, o in owners.items()]
    message = '\n'.join(lines)
    coord('send', '--to', 'pop', '--subject', f'Full main check {sha[:10]}: '
          + ('incomplete' if not complete else 'passed' if result['exit'] == 0 else 'failures'), '--message', message)
    answered = []
    for rid, body in open_full_check_requests() if complete else []:
        commits = [c for c in COMMIT.findall(body.split('\n', 2)[-1]) if git('cat-file', '-t', c, check=False) == 'commit']
        if commits and all(contained(c, sha) for c in commits):
            coord('reply', rid, '-m', f'Covered by the standing full check of {sha[:10]} (contains {", ".join(commits)}).'
                  f'\n\n{message}')
            answered.append(rid)
    if owners:
        coord('send', '--to', 'all', '--subject', f'Full main check {sha[:10]}: {len(owners)} failing test file(s), '
              'owners please fix', '--message', '\n'.join(f'{t}: last touched by {o}' for t, o in owners.items())
              + f'\n\nLog (DGX): {log}')
    if log.exists() and Path(run['worktree']) in log.parents:  # a log inside the worktree is kept beside the others
        kept = BASE / f'{sha[:10]}.log'
        shutil.copy(log, kept)
        run['log'] = str(kept)
    git('worktree', 'remove', '--force', run['worktree'], check=False)
    state.setdefault('history', []).append({**run, **{k: result[k] for k in ('exit', 'failed_steps', 'failed_tests')},
                                            'complete': complete, 'answered': answered, 'finished': time.time()})
    state['history'] = state['history'][-50:]
    if complete:
        state['last_sha'] = sha
    state.pop('running')
    save(state)


def tick(args) -> None:
    state = load()
    if state.get('running'):
        if unit_live(state['running']['unit']):
            return
        finish(state)
    subprocess.run(['git', 'fetch', '-q', 'origin'], cwd=REPO, check=False)
    sha = git('rev-parse', 'origin/main')
    if sha != state.get('last_sha'):
        launch(state, sha)


def status(args) -> None:
    print(json.dumps(load(), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('tick')
    sub.add_parser('status')
    args = ap.parse_args()
    {'tick': tick, 'status': status}[args.cmd](args)


if __name__ == '__main__':
    main()
