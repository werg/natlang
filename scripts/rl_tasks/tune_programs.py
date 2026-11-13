#!/usr/bin/env python3
"""Tune programs from the decision-readout and multi-operation codebase families (program adaptation supply).

Each program is a reusable natlang program directory in the layout `natlang tune` and `tune.structure` read: several
operations (root `.nl` functions, each with its own companion folder), `types.ts`, `package.json`, optional
`tune.services.mjs` (host services built per example from its `reference`), and `examples.json` (an array of root
invocations: function, inputs, exact expected value, world group). Worlds are grouped so the tuner's persisted
grouped split keeps related invocations on one side.

  codebase-<name>   a train-split repository codebase (codebases/<name>) with one operation per exact function of the
                    codebase that the verified RL cases call (trace), a replay operation that folds its transition, and
                    for codebases with a state transition an `operate` operation driving a service to a target
                    (expected: the service's confirmation code). Examples are the verified RL train cases' inputs and
                    expected values (scripts/rl_tasks/sources/codebase_ops.py, qualified by verify.mjs).
  decisions-<set>   a fixed intent set of CLINC150/BANKING77/MASSIVE train items behind a `readout: decision` intent
                    function: route (category), tally (record of counts), select (id list) and desk (assign/escalate
                    through a service, confirmation code) operations. Expected values are computed exactly from the
                    dataset labels with the same state digest as the RL desk service.

  python -I scripts/rl_tasks/tune_programs.py --codebase-cases VERIFIED.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.rl_tasks.common import nl_file  # noqa: E402
from scripts.rl_tasks.sources import codebase_ops, decisions  # noqa: E402

DESK_SERVICES = r'''import { createHash } from 'node:crypto';
export const declarations = { desk: `/** Assign a ticket to a team. A later assign or escalate of the same ticket replaces this one. */
export function assign(args: { ticket: string; team: string }): string;
/** Send a ticket to human review instead of a team. */
export function escalate(args: { ticket: string }): string;
/** Close the session: the confirmation code of how every ticket was routed. */
export function finish(): string;` };
export default function services(example) {
  const desk = example.reference?.desk;
  if (!desk) return {};
  const routed = {};
  return { desk: {
    assign(args) {
      if (!desk.tickets.includes(args.ticket)) return `Error: no ticket ${args.ticket}`;
      if (!desk.teams.includes(args.team)) return `Error: no team ${args.team}; teams: ${desk.teams.join(', ')}`;
      routed[args.ticket] = 'team:' + args.team; return `ticket ${args.ticket} assigned to ${args.team}`;
    },
    escalate(args) {
      if (!desk.tickets.includes(args.ticket)) return `Error: no ticket ${args.ticket}`;
      routed[args.ticket] = 'human'; return `ticket ${args.ticket} escalated to human review`;
    },
    finish() {
      const state = desk.tickets.map(id => [id, routed[id] ?? 'open']);
      return 'desk-' + createHash('sha256').update(desk.salt + JSON.stringify(state)).digest('hex').slice(0, 16);
    },
  } };
}
'''
DRIVE_SERVICES = r'''import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
// The service applies the program's own exact function (operate/__MODULE__), as in the RL drive cases.
const source = readFileSync(new URL('./operate/__MODULE__', import.meta.url), 'utf8').replace('export default function', 'function');
const step = new Function(stripTypeScriptTypes(source) + '\nreturn __NAME__;')();
const PROJECT = s => __PROJECT__;
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon(value[k])]));
  return value;
}
export const declarations = { __SERVICE__: `/** The application's current state. */
export function state(): any;
/** Make one move by its id: the application's own __NAME__ applied to the current state. Each move can be made once; a move the application rejects changes nothing. */
export function apply(args: { id: string }): string;
/** Finish: the confirmation code of the final state and the moves made. */
export function finish(): string;` };
export default function services(example) {
  const world = example.reference?.drive;
  if (!world) return {};
  let current = structuredClone(world.initial);
  const applied = [];
  return { __SERVICE__: {
    state: () => structuredClone(current),
    apply(args) {
      const move = world.moves[args.id];
      if (!move) return `Error: no move ${args.id}`;
      if (applied.includes(args.id)) return `Error: move ${args.id} was already made`;
      try { current = step(structuredClone(current), ...structuredClone(move)); }
      catch (error) { return `Error: rejected: ${error.message}`; }
      applied.push(args.id); return `made ${args.id}`;
    },
    finish: () => '__SERVICE__-' + createHash('sha256').update(world.salt + JSON.stringify([canon(PROJECT(current)), [...applied].sort()])).digest('hex').slice(0, 16),
  } };
}
'''


def write_program(out, name, files, examples, services=None):
    directory = out / name
    if directory.exists():
        raise SystemExit(f'{directory} exists; tune programs are never overwritten')
    for relative, text in files.items():
        if relative.endswith('.nl') and '/' not in relative:
            # Plain `returns: State` (types are read verbatim): tune.structure reads the root result unquoted.
            text = re.sub(r'^returns: "([A-Za-z_][\w<>, \[\]]*)"$', r'returns: \1', text, count=1, flags=re.M)
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (directory / 'package.json').write_text(json.dumps({'name': f'@natlang/tune-{name}', 'private': True, 'type': 'module'}) + '\n')
    (directory / 'natlang.json').write_text('{"schema":"natlang.package/v2"}\n')
    if services:
        (directory / 'tune.services.mjs').write_text(services)
    payload = json.dumps(examples, indent=1, sort_keys=True) + '\n'
    (directory / 'examples.json').write_text(payload)
    return {'program': name, 'path': str(directory), 'roots': len(examples),
            'operations': sorted({row['function'] for row in examples}),
            'examples_sha256': hashlib.sha256(payload.encode()).hexdigest()}


def _op_file(args, returns, instruction, extra=''):
    return nl_file(args, returns, instruction + extra)


def codebase_programs(cases, out):
    """One program per train codebase from its verified RL train cases (trace, fold and drive operations)."""
    by_cb = collections.defaultdict(list)
    for case in cases:
        ext = case['curriculum']['external']
        if case['split'] == 'train' and ext['codebase']['operation'] in ('trace', 'drive'):
            by_cb[ext['codebase']['package'].split('/', 1)[1]].append(case)
    summaries = []
    for name, rows in sorted(by_cb.items()):
        files = codebase_ops._files(name)
        program = {'types.ts': files.get('types.ts', '')}
        examples, services = [], None
        ops = collections.defaultdict(list)
        for case in rows:
            ops[(case['curriculum']['shape'], case['semantics']['files'][case['semantics']['root']].split('---')[1])].append(case)
        main_set = False
        for index, ((shape, header), members) in enumerate(sorted(ops.items())):
            first = members[0]
            root_text = first['semantics']['files'][first['semantics']['root']]
            if shape.startswith('drive'):
                op = 'operate'
            else:
                dotted = next((p[:-3].replace('/', '.') for p in files if p.endswith('.ts') and p[:-3].replace('/', '.') in root_text), 'f')
                op = ('replay_' if shape == 'trace-fold' else 'call_') + dotted.split('.')[-1]
            if not main_set and shape == 'trace-fold':
                op, main_set = 'main', True
            # One fixed instruction per operation (the first case's), its own companion folder holding the codebase.
            program[f'{op}.nl'] = root_text
            for path, text in files.items():
                program[f'{op}/{path}'] = text
            for case in members:
                row = {'function': f'{op}.nl', 'inputs': case['semantics']['inputs'], 'expected': case['semantics']['expected'],
                       'group': f'tune:codebase-{name}:{case["id"]}'}
                if shape.startswith('drive'):
                    svc = next(iter(case['semantics']['services']))
                    spec = codebase_ops.DRIVE[name]
                    src = case['semantics']['services'][svc]
                    initial = json.loads(src.split('let current: any = ', 1)[1].split(';\nconst applied', 1)[0])
                    moves = json.loads(src.split('const MOVES: Record<string, unknown[]> = ', 1)[1].split(';\nconst PROJECT', 1)[0])
                    salt = json.loads(src.split('const SALT = ', 1)[1].split(';\n', 1)[0])
                    row['reference'] = {'drive': {'initial': initial, 'moves': moves, 'salt': salt}}
                    fn = codebase_ops._signature(files[spec['module']])[0]
                    services = (DRIVE_SERVICES.replace('__MODULE__', spec['module']).replace('__NAME__', fn)
                                .replace('__PROJECT__', codebase_ops.DRIVE_PROJECT_TS[name]).replace('__SERVICE__', svc))
                examples.append(row)
        if 'main.nl' not in program:
            # The first trace operation is the program's main entry.
            first_op = sorted(k for k in program if k.endswith('.nl') and '/' not in k)[0]
            program['main.nl'] = program.pop(first_op)
            stem = first_op[:-3]
            for key in [k for k in program if k.startswith(stem + '/')]:
                program['main/' + key[len(stem) + 1:]] = program.pop(key)
            for row in examples:
                if row['function'] == first_op:
                    row['function'] = 'main.nl'
        summaries.append(write_program(out, f'codebase-{name}', program, examples, services))
    return summaries


def decision_programs(out, seed=20261103, per_op=40):
    data = decisions._load()
    summaries = []
    for dataset, k in (('clinc_oos', 6), ('banking77', 6), ('massive_intent', 5)):
        rng = random.Random(f'{seed}|tune|{dataset}')
        pool = decisions.Pool(data[dataset]['train'], rng)
        labels = [l for l in pool.labels(60) if l != 'oos']
        candidates = sorted(rng.sample(labels, k))
        others = [l for l in pool.labels(20) if l not in candidates and l != 'oos']
        teams = decisions.TEAMS[:k - 1]
        routes = {c: teams[i % len(teams)] for i, c in enumerate(candidates)}
        options = candidates + [decisions.OOS]
        child = decisions._decision_file({'request': 'string'}, options, decisions.INTENT_CHILD[0])
        child_msg = decisions._decision_file({'message': 'string'}, candidates, decisions.TALLY_CHILD[0])
        frame = decisions.FRAMES[dataset][0] + '\n\n'
        program = {
            'main.nl': nl_file({'request': 'string', 'routes': 'Record<string, string>'}, 'string', frame + decisions.ROUTE[0]),
            'main/intent.nl': child,
            'tally.nl': nl_file({'messages': '{ id: string; text: string }[]'}, 'Record<string, number>', frame + decisions.TALLY['counts'][0]),
            'tally/intent.nl': child_msg,
            'select.nl': nl_file({'messages': '{ id: string; text: string }[]', 'target': 'string'}, 'string[]',
                                 frame + 'Which messages are about target? Decide each message\'s intent with intent and '
                                 'return the ids of the messages whose intent is target, in any order.'),
            'select/intent.nl': child_msg,
            'desk.nl': nl_file({'tickets': '{ id: string; text: string }[]', 'routes': 'Record<string, string>'}, 'string',
                               frame + decisions.DESK[0]),
            'desk/intent.nl': child,
        }
        examples = []

        def take(label):
            return pool.take(label)

        for i in range(per_op):
            in_scope = rng.random() < 0.75
            item = take(rng.choice(candidates) if in_scope else rng.choice(others))
            if item:
                examples.append({'function': 'main.nl', 'inputs': {'request': item['text'], 'routes': routes},
                                 'expected': routes.get(item['label'], 'human_review'), 'group': f'tune:{dataset}:route:{i}'})
        for i in range(per_op):
            items = [x for x in (take(rng.choice(candidates)) for _ in range(rng.randint(4, 8))) if x]
            msgs = [{'id': f'm{j + 1}', 'text': x['text']} for j, x in enumerate(items)]
            counts = collections.Counter(x['label'] for x in items)
            examples.append({'function': 'tally.nl', 'inputs': {'messages': msgs},
                             'expected': {c: counts.get(c, 0) for c in candidates}, 'group': f'tune:{dataset}:tally:{i}'})
        for i in range(per_op):
            items = [x for x in (take(rng.choice(candidates)) for _ in range(rng.randint(4, 8))) if x]
            msgs = [{'id': f'm{j + 1}', 'text': x['text']} for j, x in enumerate(items)]
            target = rng.choice([x['label'] for x in items])
            examples.append({'function': 'select.nl', 'inputs': {'messages': msgs, 'target': target},
                             'expected': [m['id'] for m, x in zip(msgs, items) if x['label'] == target],
                             'group': f'tune:{dataset}:select:{i}'})
        for i in range(per_op):
            items = [x for x in (take(rng.choice(candidates) if rng.random() < 0.8 else rng.choice(others)) for _ in range(rng.randint(3, 6))) if x]
            tickets = [{'id': f't{100 + j}', 'text': x['text']} for j, x in enumerate(items)]
            salt = hashlib.sha256(f'tune-desk|{dataset}|{i}'.encode()).hexdigest()[:24]
            state = [[t['id'], ('team:' + routes[x['label']]) if x['label'] in routes else 'human'] for t, x in zip(tickets, items)]
            code = 'desk-' + hashlib.sha256((salt + json.dumps(state, separators=(',', ':'), ensure_ascii=False)).encode()).hexdigest()[:16]
            examples.append({'function': 'desk.nl', 'inputs': {'tickets': tickets, 'routes': routes}, 'expected': code,
                             'group': f'tune:{dataset}:desk:{i}',
                             'reference': {'desk': {'tickets': [t['id'] for t in tickets], 'teams': teams, 'salt': salt}}})
        summaries.append(write_program(out, f'decisions-{dataset.replace("_", "-")}', program, examples, DESK_SERVICES))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--codebase-cases', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(line) for line in a.codebase_cases.read_text().splitlines() if line.strip()]
    summaries = codebase_programs(cases, a.out) + decision_programs(a.out)
    (a.out / 'programs-prepared.json').write_text(json.dumps({'schema': 'natlang.program-supply/1', 'programs': summaries,
        'source': 'scripts/rl_tasks/tune_programs.py: verified codebase-ops RL train cases and labelled decision datasets; '
                  'exact oracles only; no generated teacher actions', 'production_exposure': False,
        'capability_lift': None}, indent=1) + '\n')
    print(json.dumps([{k: s[k] for k in ('program', 'roots', 'operations')} for s in summaries]))


if __name__ == '__main__':
    main()
