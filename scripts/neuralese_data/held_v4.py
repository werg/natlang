#!/usr/bin/env python3
"""held-v4, the semantic-FP held set (plans/semantic-fp/05 §5, 09 B1): reservations, the design-brief gate stratum, and
the combined cases file ``held_eval pin`` stratifies.

    held_v4.py reserve OUT/reservations.json      reserved sources, seeded; the registry role
                                                  held-eval-semantic-fp-reservations names the file
    held_v4.py gates RESERVATIONS OUT.jsonl       the exact gate cases of the reserved design briefs (natlang.program/2)
    held_v4.py combine OUT.jsonl IN.jsonl..       one cases file (first occurrence of an id wins)

Reservations (``held_eval.reserved`` applies them to every training path through ``training_exclusion_ids``):
- 20 design briefs, seed ``held-v4-design-briefs``. Candidates are the 40 briefs the frozen gate packet keeps out of
  its train split (every brief whose number is 4th or 5th in its domain: ts-host/scripts/design-briefs/lib.mjs
  ``applicationSplit``), so no gate of theirs was ever a training task. They rank unexposed first (no row of any
  grpo-pool or memetic pool names them), then by sha256(seed, number).
- 2 of the 11 Astra/Sol applications (``training/semantic_applications/{astra,sol}/*``), seed ``held-v4-astra``, ranked
  the same way.
- A 20% split-group share of every flagship fixture pack (WS6 §1.4), seed ``held-v4-flagship``: a fixture-pack row
  whose split group starts with ``flagship:`` (``flagship:<app>:<group>``) is held when
  ``held_eval.reservation_share_held`` says so. A pack labels its own held split with that function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]

from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.jsonio import iter_jsonl_rows, read_jsonl  # noqa: E402
from natlang_neuralese.rl.held_eval import RESERVATIONS_SCHEMA, reserved  # noqa: E402

BRIEFS = REPO / 'training/semantic_applications/design_briefs'
APPS = REPO / 'training/semantic_applications'
BRIEF_SEED, APP_SEED, FLAGSHIP_SEED = 'held-v4-design-briefs', 'held-v4-astra', 'held-v4-flagship'
BRIEF_COUNT, APP_COUNT, FLAGSHIP_SHARE = 20, 2, 0.2


def _rank(seed, key):
    return hashlib.sha256(f'{seed}\x00{key}'.encode()).hexdigest()


def brief_spec(slug):
    """Groups and patterns of one design brief (``NNN-name``): its gate packet groups, its codebase and fixture paths."""
    number = slug[:3]
    return {'groups': [f'design-brief-app-{number}', f'design-briefs:application-{number}',
                       f'ext-unit:design-brief-gates:design-briefs:application-{number}'],
            'patterns': [rf'(?:^|[/:]){re.escape(slug)}(?:$|[/.:])', rf'design-briefs-v3:{number}:',
                         rf'design-briefs?:application-{number}\b', rf'design-brief-app-{number}\b']}


def app_spec(kind, name):
    return {'groups': [], 'patterns': [rf'semantic_applications/{kind}/{re.escape(name)}\b',
                                       rf'(?:^|[/:])applications/{re.escape(name)}(?:$|/)',
                                       rf'\b{kind}-app(?:lication)?-{re.escape(name)}\b']}


def _pool_rows():
    for path in [*data_registry.role_paths('grpo-pool'), *data_registry.role_paths('grpo-memetic-tasks')]:
        for _, row in iter_jsonl_rows(path):
            yield row


def exposure(specs):
    """{key: number of pool rows a spec reserves}."""
    counts = {key: 0 for key in specs}
    compiled = {key: {'schema': RESERVATIONS_SCHEMA, **spec} for key, spec in specs.items()}
    for row in _pool_rows():
        for key, spec in compiled.items():
            if reserved(spec, row):
                counts[key] += 1
    return counts


def non_train_briefs():
    slugs = sorted(p.name for p in BRIEFS.iterdir() if p.is_dir() and re.match(r'\d{3}-', p.name))
    # applicationSplit: (number - 1) % 5 is 3 (dev) or 4 (test)
    return [s for s in slugs if (int(s[:3]) - 1) % 5 >= 3]


def reserve(out):
    briefs = {s: brief_spec(s) for s in non_train_briefs()}
    apps = {f'{kind}/{p.name}': app_spec(kind, p.name) for kind in ('astra', 'sol')
            for p in sorted((APPS / kind).iterdir()) if p.is_dir()}
    seen = exposure({**briefs, **apps})
    brief_pick = sorted(briefs, key=lambda s: (seen[s] > 0, _rank(BRIEF_SEED, s[:3])))[:BRIEF_COUNT]
    app_pick = sorted(apps, key=lambda a: (seen[a] > 0, _rank(APP_SEED, a)))[:APP_COUNT]
    groups, patterns = [], []
    for key in sorted(brief_pick):
        groups += briefs[key]['groups']
        patterns += briefs[key]['patterns']
    for key in sorted(app_pick):
        patterns += apps[key]['patterns']
    record = {
        'schema': RESERVATIONS_SCHEMA, 'set': 'held-v4',
        'note': 'Sources reserved for held-v4 before its set exists (plans/semantic-fp/09 B1 step 2). Training excludes '
                'every pool row these name from the commit that registered this file (held_eval.reserved, '
                'training_exclusion_ids).',
        'design_briefs': {'seed': BRIEF_SEED, 'candidates': 'the 40 briefs outside the gate packet train split',
                          'rank': '(pool rows naming the brief > 0, sha256(seed, number))',
                          'chosen': [{'brief': k, 'pool_rows_reserved': seen[k]} for k in sorted(brief_pick)],
                          'not_chosen': {k: seen[k] for k in sorted(set(briefs) - set(brief_pick))}},
        'applications': {'seed': APP_SEED, 'candidates': sorted(apps),
                         'chosen': [{'application': k, 'pool_rows_reserved': seen[k]} for k in sorted(app_pick)]},
        'groups': sorted(set(groups)), 'patterns': patterns,
        'share_rules': [{'pattern': r'^flagship:', 'share': FLAGSHIP_SHARE, 'seed': FLAGSHIP_SEED,
                         'note': 'flagship fixture packs (WS6 §1.4): split group flagship:<app>:<group>'}]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if Path(out).exists():
        raise SystemExit(f'{out} exists: reservations are written once (a change is a new held-v4 version)')
    Path(out).write_text(json.dumps(record, indent=1) + '\n')
    return record


def gates(reservations, out):
    """The frozen exact gates (3 per brief: pass, fail, needs-review) of the reserved design briefs."""
    sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
    from scripts.rl_tasks.sources import design_brief_gates
    spec = json.loads(Path(reservations).read_text())
    chosen = {item['brief'][:3] for item in spec['design_briefs']['chosen']}
    rows = [case for case in design_brief_gates.cases() if case['curriculum']['external']['application'] in chosen]
    for row in rows:
        row['curriculum']['held'] = {'set': 'held-v4', 'stratum': 'design-brief-gate'}
    Path(out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return {'rows': len(rows), 'briefs': len(chosen)}


def combine(out, inputs):
    rows, seen = [], set()
    for path in inputs:
        for row in read_jsonl(path):
            if row['id'] not in seen:
                seen.add(row['id'])
                rows.append(row)
    Path(out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    return {'rows': len(rows), 'families': len({r.get('family') for r in rows})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('reserve')
    r.add_argument('out')
    g = sub.add_parser('gates')
    g.add_argument('reservations')
    g.add_argument('out')
    c = sub.add_parser('combine')
    c.add_argument('out')
    c.add_argument('inputs', nargs='+')
    args = parser.parse_args(argv)
    if args.command == 'reserve':
        record = reserve(args.out)
        print(json.dumps({'briefs': record['design_briefs']['chosen'], 'apps': record['applications']['chosen']}))
    elif args.command == 'gates':
        print(json.dumps(gates(args.reservations, args.out)))
    else:
        print(json.dumps(combine(args.out, args.inputs)))


if __name__ == '__main__':
    main()
