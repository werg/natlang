#!/usr/bin/env python3
"""Seal a stop-evidence successor of an RL train curriculum written by derive-stop-evidence.mjs.

Checks that the successor differs from its parent only in `curriculum.decisive` of the derived expected-stop rows
(every other row byte-identical, same order, ids, splits and source bindings), and that no expected-stop row is left
without decisive evidence unless it is listed with its reason. Writes `verification.json` (scope: every expected-stop
row, re-executed through the collector) and `receipt.json` binding parent, runtime commit and outputs. Exit 1 on any
violation. Protection intersections are then checked by qualify_rl_train_curriculum.py on the sealed packet.

  publish_stop_evidence_curriculum.py --parent runs/natlang-rl-train-curriculum-lfm3-20261028 \\
      --packet runs/natlang-rl-train-curriculum-lfm3-20261030-stop-evidence --runtime-commit COMMIT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parent', required=True, type=Path)
    ap.add_argument('--packet', required=True, type=Path)
    ap.add_argument('--runtime-commit', required=True)
    a = ap.parse_args()
    parent_cases, cases = a.parent / 'cases.jsonl', a.packet / 'cases.jsonl'
    report = json.loads((a.packet / 'stop-evidence.json').read_text())
    problems = []
    if report['input']['sha256'] != sha_file(parent_cases):
        problems.append('derivation input is not the parent cases.jsonl')
    if report['output']['sha256'] != sha_file(cases):
        problems.append('cases.jsonl changed after derivation')
    derived = {item['id']: item for item in report['derived']}
    unresolved = {item['id']: item for item in report['unresolved']}
    old_lines = parent_cases.read_bytes().splitlines()
    new_lines = cases.read_bytes().splitlines()
    if len(old_lines) != len(new_lines):
        problems.append(f'row count {len(new_lines)} differs from parent {len(old_lines)}')
    changed, stops_without_evidence = 0, []
    for line_no, (old_raw, new_raw) in enumerate(zip(old_lines, new_lines), 1):
        new = json.loads(new_raw)
        if new.get('semantics', {}).get('operation') == 'blocked' and not new['curriculum'].get('decisive') \
                and new['id'] not in unresolved:
            stops_without_evidence.append(new['id'])
        if old_raw == new_raw:
            if new['id'] in derived:
                problems.append(f'{new["id"]} derived but unchanged')
            continue
        old = json.loads(old_raw)
        changed += 1
        if new['id'] not in derived:
            problems.append(f'line {line_no} {new["id"]} changed without a derivation record')
        if [item['marker'] for item in new['curriculum']['decisive']] != [derived.get(new['id'], {}).get('marker')]:
            problems.append(f'{new["id"]} marker differs from its derivation record')
        if old['curriculum'].get('decisive'):
            problems.append(f'{new["id"]} replaced existing decisive evidence')
        old['curriculum']['decisive'] = new['curriculum']['decisive']
        if old != new:
            problems.append(f'{new["id"]} changed beyond curriculum.decisive')
    if changed != len(derived):
        problems.append(f'{changed} rows changed but {len(derived)} derived')
    if stops_without_evidence:
        problems.append(f'{len(stops_without_evidence)} expected-stop rows without evidence and unlisted')
    failing_existing = [item for item in report['existing'] if not item['ok']]
    failures = ([{'id': item['id'], 'check': 'stop-evidence derivation', 'reason': item['reason']}
                 for item in report['unresolved']] +
                [{'id': item['id'], 'check': 'existing markers: reference accepted and immediate give-up rejected',
                  'reference': item['reference'], 'give_up': item['give_up']} for item in failing_existing])
    verification = {
        'schema': 'natlang.rl-curriculum-stop-evidence-verification/1',
        'scope': ('every expected-stop row (semantics.operation blocked) re-executed through the collector at the runtime '
                  'commit: derived rows get verifyCases plus accepted reference, rejected immediate give-up and rejected '
                  'early stop; rows with earlier markers get accepted reference and rejected immediate give-up. All other '
                  'rows are byte-identical to the parent, whose build replayed every reference (assemble_rl_train_curriculum.py).'),
        'runtime_commit': a.runtime_commit,
        'cases': len(report['derived']) + len(report['unresolved']) + len(report['existing']),
        'derived': len(report['derived']), 'existing_markers': len(report['existing']),
        'failures': failures,
        'derivation': {'path': 'stop-evidence.json', 'sha256': sha_file(a.packet / 'stop-evidence.json')},
    }
    (a.packet / 'verification.json').write_text(json.dumps(verification, indent=1) + '\n')
    receipt = {
        'schema': 'natlang.rl-curriculum-stop-evidence-successor/1',
        'parent': {'path': str(a.parent), 'cases_sha256': sha_file(parent_cases), 'rows': len(old_lines),
                   'assembly_receipt_sha256': sha_file(a.parent / 'assembly-receipt.json')},
        'cases': {'path': 'cases.jsonl', 'sha256': sha_file(cases), 'rows': len(new_lines)},
        'runtime_commit': a.runtime_commit,
        'derivation_rule': report['rule'],
        'changed_rows': changed, 'unchanged_rows': len(new_lines) - changed,
        'unresolved_rows': len(report['unresolved']),
        'verification': {'path': 'verification.json', 'sha256': sha_file(a.packet / 'verification.json'),
                         'failures': len(failures)},
        'problems': problems,
    }
    (a.packet / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('changed_rows', 'unchanged_rows', 'unresolved_rows')} |
                     {'failures': len(failures), 'problems': problems[:10]}))
    return 1 if problems or failures else 0


if __name__ == '__main__':
    sys.exit(main())
