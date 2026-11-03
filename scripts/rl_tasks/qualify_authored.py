#!/usr/bin/env python3
"""Qualify authored RL tasks end to end (plans/neuralese/RL_TASK_AUTHORING_KIT.md).

  qualify_authored.py TASKS.jsonl OUT_DIR [--runtime TS_HOST] [--concurrency N]

Steps, each written under OUT_DIR:
 1. convert   authored JSON -> natlang.program/2 cases (scripts/rl_tasks/sources/authored.py)   cases.jsonl
 2. verify    reference replay admitted, every wrong answer rejected, alias acceptance measured   verified/
 3. smoke     the intended call graph runs with scripted helper answers and returns expected     smoke.json
 4. custody   no relation to a protected/held world (ProtectedWorlds registry + LFM3 held-set group closure)
 5. report    per-task PASS/FAIL with reasons                                                       report.json
A task passes when 1-4 pass for it. Alias rejections are reported (fix the oracle or the alias) but do not fail it.
Run through the memory ledger on the DGX.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / 'sources'))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve  # noqa: E402

DEFAULT_RUNTIME = str(resolve('data_hdd', 'natlang-rl-authoring', 'runtime', 'ts-host'))
UNION = resolve('data_nvme', 'broad-rl-tasks-20261031', 'union_groups_fast.py')


def node():
    for candidate in sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True):
        return str(candidate)
    return 'node'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tasks')
    parser.add_argument('out')
    parser.add_argument('--runtime', default=DEFAULT_RUNTIME)
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'NATLANG_HOME': str(out / '.natlang-home'),
           'NATLANG_HASH_CACHE': str(out / 'hash-cache.jsonl'), 'OMP_NUM_THREADS': '1',
           'NODE_OPTIONS': '--max-old-space-size=1500'}
    report = {'schema': 'natlang.rl-authored-qualification/1', 'tasks': str(Path(args.tasks).resolve()),
              'runtime': args.runtime, 'steps': {}}
    converted = subprocess.run([sys.executable, str(HERE / 'sources' / 'authored.py'), args.tasks,
                                str(out / 'cases.jsonl')], capture_output=True, text=True)
    report['steps']['convert'] = json.loads(converted.stdout.strip().splitlines()[-1]) if converted.stdout else {
        'error': converted.stderr[-2000:]}
    cases = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines() if line.strip()] \
        if (out / 'cases.jsonl').exists() else []
    status = {row['id']: {'task': row['curriculum']['external']['upstream_id'], 'split': row['split'],
                          'family': row['family'], 'problems': []} for row in cases}
    if not cases:
        report['result'] = 'FAIL: nothing converted'
        (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps(report['steps'])); return 1
    subprocess.run([node(), str(HERE / 'verify.mjs'), args.runtime, str(out / 'cases.jsonl'), str(out / 'verified'),
                    str(args.concurrency)], env=env, check=True, cwd=REPO)
    verification = json.loads((out / 'verified' / 'verification.json').read_text())
    report['steps']['verify'] = verification['summary']
    for result in verification['results']:
        entry = status[result['id']]
        if not result['reference_ok']:
            entry['problems'].append('reference not admitted: ' + '; '.join(map(str, result['problems'][:2])))
        for wrong in result['wrong']:
            if wrong['accepted'] is not False:
                entry['problems'].append(f'wrong answer not rejected: {json.dumps(wrong["value"])[:200]}')
        for alias in result['aliases']:
            if alias['accepted'] is False:
                entry.setdefault('warnings', []).append(f'alias rejected: {json.dumps(alias["value"])[:200]}')
    subprocess.run([node(), str(HERE / 'structure_smoke.mjs'), args.runtime, str(out / 'cases.jsonl'),
                    str(out / 'smoke.json'), '100000'], env=env, check=True, cwd=REPO)
    smoke = json.loads((out / 'smoke.json').read_text())
    report['steps']['smoke'] = {k: smoke['summary'][k] for k in ('runs', 'accepted')}
    for result in smoke['results']:
        if not result['accepted']:
            status[result['id']]['problems'].append(
                f'smoke run failed ({result.get("status")}): {json.dumps(result.get("error") or result.get("detail"))[:300]}')
    from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
    protected = ProtectedWorlds.load()
    for row in cases:
        related = sorted({kind for kind, _ in world_identities(row, evidence_items=False) & protected.identities})
        if related:
            status[row['id']]['problems'].append('related to a protected world: ' + ', '.join(related))
    if UNION.is_file():
        subprocess.run([sys.executable, str(UNION), str(out / 'cases.jsonl'), str(out / 'protection-union.json')],
                       check=True, cwd=REPO, env={**env, 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'})
        union = json.loads((out / 'protection-union.json').read_text())
        for ident in union['conflicts']:
            status[ident]['problems'].append('held-set group closure conflict')
        report['steps']['custody'] = {'protected_worlds': 'checked', 'group_closure': union['status']}
    passed = [s for s in status.values() if not s['problems']]
    report['passed'] = len(passed)
    report['failed'] = len(status) - len(passed)
    report['tasks_status'] = status
    report['result'] = 'PASS' if not report['failed'] and not report['steps']['convert'].get('errors') else 'FAIL'
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'result': report['result'], 'passed': report['passed'], 'failed': report['failed'],
                      'convert_errors': report['steps']['convert'].get('errors'),
                      'failures': {s['task']: s['problems'] for s in status.values() if s['problems']}}, indent=1))
    return 0 if report['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
