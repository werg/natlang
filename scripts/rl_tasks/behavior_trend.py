#!/usr/bin/env python3
"""Behaviour trend of an RL controller: loop rate on its training rounds and dev16 results of its checkpoints.

    python3 scripts/rl_tasks/behavior_trend.py CONTROLLER_OUT --parent-label LABEL [--label-glob 'raw-*-w4r*']

Loop rate = (budget_exhaustion + repetition) / outcomes: the 2026-11-05/08 regression was a repeat-the-failing-action
attractor that runs out the budget, so it is measured directly, every round, for free from the collected rollouts.
`unresolved` (wall timeout without an answer) is reported separately: often a loop too, but not proven one.

Trend rule (replaces the single-checkpoint >=12/16 gate, whose +-1.7 binomial noise on 16 single-sample cases could
flip the decision by chance). Over this lineage's dev16 results, in step order:
  STOP      latest <= 10/16, or latest has >= 2 looping cases, or the mean of the last two is >= 1.5 below the parent;
  WATCH     training loop rate rose on two consecutive rounds, or one checkpoint below the parent;
  CONTINUE  otherwise.
Read-only; it decides nothing by itself.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'training/neuralese')]
from natlang_neuralese.common.paths import resolve  # noqa: E402

LOOP = ('budget_exhaustion', 'repetition')
EVIDENCE = resolve('data_nvme', 'lfm3-rl-dgx-20261029', 'ops/pop-behavior-eval/evidence')


def loop_rate(counts):
    total = sum(counts.values())
    return (sum(counts.get(k, 0) for k in LOOP) / total if total else None), total


def round_rows(out):
    rows = []
    for score in sorted(Path(out).glob('round-*/score.json')):
        report = json.loads(score.read_text())
        counts = {}
        for group in report.get('groups', []):
            for outcome in group.get('outcomes', []):
                counts[outcome] = counts.get(outcome, 0) + 1
        rate, total = loop_rate(counts)
        rows.append({'round': score.parent.name, 'complete': report.get('collection_complete'), 'students': total,
                     'success': counts.get('task_success', 0), 'loop_rate': rate,
                     'unresolved': counts.get('unresolved', 0), 'groups_with_spread': report.get('groups_with_spread')})
    return rows


def dev16(label):
    # The latest attempt of a label (a '-aN' suffix is a re-run of the same view).
    paths = sorted(glob.glob(str(EVIDENCE / f'{label}*' / 'detail.json')))
    if not paths:
        return None
    detail = json.loads(Path(paths[-1]).read_text())
    counts = detail.get('outcome_counts') or {}
    return {'label': Path(paths[-1]).parent.name, 'step': detail.get('step'),
            'success': counts.get('task_success', detail.get('task_successes')),
            'loops': sum(counts.get(k, 0) for k in LOOP), 'unresolved': counts.get('unresolved', 0),
            'complete': detail.get('complete')}


def decide(parent, results, rounds):
    rates = [r['loop_rate'] for r in rounds if r['loop_rate'] is not None and r['complete']]
    rising = len(rates) >= 3 and rates[-1] > rates[-2] > rates[-3]
    if not results:
        return 'WATCH' if rising else 'CONTINUE', 'no dev16 result of this lineage yet'
    latest = results[-1]
    if latest['success'] <= 10:
        return 'STOP', f"latest dev16 {latest['success']}/16 <= 10"
    if latest['loops'] >= 2:
        return 'STOP', f"latest dev16 has {latest['loops']} looping cases"
    if len(results) >= 2 and (results[-1]['success'] + results[-2]['success']) / 2 <= parent['success'] - 1.5:
        return 'STOP', 'mean of last two dev16 results is >= 1.5 below the parent'
    if rising:
        return 'WATCH', 'training loop rate rose on two consecutive rounds'
    if latest['success'] < parent['success']:
        return 'WATCH', f"latest {latest['success']}/16 below parent {parent['success']}/16 (within noise)"
    return 'CONTINUE', f"latest {latest['success']}/16 vs parent {parent['success']}/16, {latest['loops']} looping"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('out', type=Path, help='controller out directory (holds round-*/score.json)')
    p.add_argument('--parent-label', required=True, help='dev16 evidence label prefix of the parent checkpoint')
    p.add_argument('--label-glob', action='append', default=[], help='evidence label globs of this lineage')
    p.add_argument('--json', action='store_true')
    a = p.parse_args(argv)
    rounds = round_rows(a.out)
    parent = dev16(a.parent_label)
    if parent is None:
        raise SystemExit(f'no dev16 evidence for parent {a.parent_label}')
    labels = sorted({Path(d).name.split('-py')[0] for g in a.label_glob for d in glob.glob(str(EVIDENCE / g))})
    results = sorted((r for r in map(dev16, labels) if r and r['complete']), key=lambda r: r['step'])
    verdict, reason = decide(parent, results, rounds)
    if a.json:
        print(json.dumps({'rounds': rounds, 'parent': parent, 'dev16': results, 'verdict': verdict, 'reason': reason}))
        return
    print('round           students success loop_rate unresolved spread_groups')
    for r in rounds:
        rate = '-' if r['loop_rate'] is None else f"{r['loop_rate']:.2f}"
        print(f"{r['round']:<15} {r['students']:>8} {r['success']:>7} {rate:>9} {r['unresolved']:>10} {r['groups_with_spread']!s:>13}"
              + ('' if r['complete'] else '  (incomplete)'))
    print(f"parent dev16  step {parent['step']}: {parent['success']}/16, {parent['loops']} looping")
    for r in results:
        print(f"dev16         step {r['step']}: {r['success']}/16, {r['loops']} looping, {r['unresolved']} unresolved  ({r['label']})")
    print(f'verdict: {verdict} ({reason})')


if __name__ == '__main__':
    main()
