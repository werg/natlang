#!/usr/bin/env python3
"""Tune programs from the composed Neuralese call-graph families (scripts/rl_tasks/sources/composed.py).

  composed_tune_programs.py SOURCE_DIR OUT_DIR [--train-roots 160]

SOURCE_DIR is an assembled composed task source (cases.jsonl, held/{dev,confirmation}.jsonl: verified, custody
filtered). Each tune program is one reusable natlang program directory in the layout `natlang_neuralese.tune.supply`
writes and `tune.structure` reads: `main.nl` with its callable folder `main/`, `types.ts`, `package.json` and
`tune.examples.jsonl` (`{function, input, expected, group, reference}` rows, one invocation world per split unit). The
same program serves every world; the multi-operation kit has a second root function with its own examples.

Oracles: the RL rows' own outcome oracles, applied by the program's crisp verifier `tune.reward.mjs` (span QA token F1
with alternates, exact, normalized text, numeric tolerance; a port of ts-host `checkOracle` for these kinds). Splits:
`.natlang/tune/splits.json` is written up front so worlds keep the RL source's split (upstream train -> train, upstream
evaluation units -> development / confirmation as `unit_split` drew them) instead of the tuner's hash draw.
No teacher actions; no capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

PARTITION = {'train': 'train', 'dev': 'development', 'confirmation': 'confirmation'}

VERIFIER = r'''// Root verifier of a composed tune program: the RL rows' outcome oracle (example.reference.oracle).
// Port of ts-host evaluation/oracles.ts for the kinds these programs use: span QA token F1 (SQuAD normalization,
// numbers must match), exact, normalized text, numeric tolerance (relative / absolute / percent forms).
const words = value => (String(Array.isArray(value) ? value.join(' ') : value ?? '').normalize('NFKC')
  .toLocaleLowerCase('en-US').match(/[\p{L}\p{N}]+/gu) ?? []).filter(w => !['a', 'an', 'the'].includes(w));
const number = token => { const n = Number(token); return Number.isFinite(n) ? String(n) : token; };
const numbers = value => (String(value).match(/[+-]?\d+(?:\.\d+)?/g) ?? []).map(number).sort();
function f1(actual, expected) {
  const gold = numbers(expected);
  if (gold.length && JSON.stringify(gold) !== JSON.stringify(numbers(actual))) return 0;
  const left = words(actual), right = words(expected);
  if ([actual, expected].some(v => ['yes', 'no', 'noanswer'].includes(String(v).trim().toLowerCase())) &&
      JSON.stringify(left) !== JSON.stringify(right)) return 0;
  if (!left.length || !right.length) return left.length === right.length ? 1 : 0;
  const counts = new Map();
  for (const w of right) counts.set(w, (counts.get(w) ?? 0) + 1);
  let shared = 0;
  for (const w of left) if ((counts.get(w) ?? 0) > 0) { shared++; counts.set(w, counts.get(w) - 1); }
  return 2 * shared / (left.length + right.length);
}
const text = v => String(v ?? '').normalize('NFKC').trim().toLowerCase().replace(/\s+/g, ' ');
function close(got, want, spec) {
  const near = (a, b) => Math.abs(a - b) <= Math.max(spec.abs_tolerance ?? 0, (spec.tolerance ?? 1e-4) * Math.abs(b));
  return near(got, want) || !!spec.percent_equivalent && (near(got, want * 100) || near(got * 100, want));
}
export default function verify(value, example) {
  const spec = example.reference?.oracle ?? { kind: 'exact' };
  const golds = [example.expected, ...(spec.alternates ?? [])];
  if (spec.kind === 'span-qa') return typeof value === 'string' && Math.max(...golds.map(g => f1(value, g))) >= (spec.threshold ?? 0.8);
  if (spec.kind === 'normalized') return golds.some(g => text(value) === text(g));
  if (spec.kind === 'numeric-tolerance') {
    const got = typeof value === 'number' ? value : Number(String(value).replace(/[,$%]/g, ''));
    return Number.isFinite(got) && golds.some(g => close(got, Number(g), spec));
  }
  return golds.some(g => JSON.stringify(g) === JSON.stringify(value));
}
'''


def frontmatter(text):
    _, head, body = text.split('---', 2)
    return head, body


def simple_returns(nl_text, returns):
    """Rewrite one function's `returns:` (unquoted when it is a plain identifier or simple type, as supply writes)."""
    head, body = frontmatter(nl_text)
    head = re.sub(r'^returns: .*$', 'returns: ' + returns, head, flags=re.M)
    return '---' + head + '---' + body


def oracle_of(case):
    o = case['semantics']['oracle']
    if o == 'exact':
        return {'kind': 'exact'}
    if o == 'normalized' or (isinstance(o, dict) and o.get('normalization') == 'qa' and o.get('level') == 'normalized'):
        return {'kind': 'normalized'}
    if isinstance(o, dict) and o.get('level') == 'span':
        return {'kind': 'span-qa', 'threshold': o.get('threshold', 0.8), 'alternates': o.get('alternates', [])}
    if isinstance(o, dict) and o.get('normalization') == 'numeric-tolerance':
        return {'kind': 'numeric-tolerance', **{k: o[k] for k in ('tolerance', 'abs_tolerance', 'percent_equivalent') if k in o}}
    return None


# program name -> (operations: [(structure program, origin source, rename root to, returns override)], types.ts, notes)
PROGRAMS = {
    'composed-multihop-kit': {
        'operations': [('decompose-solve-read', 'musique', 'main', 'string'),
                       ('lookup-read-compare', '2wikimultihopqa', 'answer_by_lookups', 'string')],
        'note': 'depth-3 call graphs, two operations (decompose/solve/read and lookup/read/compare)'},
    'composed-multihop-steps': {
        'operations': [('decompose-solve-read-confirm', 'musique', 'main', 'string')],
        'note': 'depth-3 decompose -> solve_step -> read_passage, plus confirm'},
    'composed-kinship-paths': {
        'operations': [('trace-and-compose-kinship', 'clutrr', 'main', 'Kinship')],
        'types': 'kinship', 'note': 'depth-3 path tracing and relation composition; category result'},
    'composed-article-reducer': {
        'operations': [('article-parts-reducer', 'quality', 'main', 'Choice')],
        'types': 'choice', 'note': 'iterateOn over Neuralese notes across article parts; category result'},
    'composed-story-reducer': {
        'operations': [('story-state-reducer', 'babi', 'main', 'string')],
        'note': 'iterateOn over a Neuralese world state across story chunks'},
    'composed-section-candidates': {
        'operations': [('section-candidates-reducer', 'triviaqa', 'main', 'string')],
        'note': 'iterateOn over a typed Neuralese candidate state across page sections'},
    'composed-report-readers': {
        'operations': [('table-and-text-readers', 'tatqa', 'main', 'number')],
        'note': 'fan-in of a table reader and a text reader into an exact computation (numeric answers only)'},
    'composed-table-pages': {
        'operations': [('rows-and-pages-readers', 'hybridqa', 'main', 'string')],
        'note': 'fan-in of a row finder and per-page readers'},
    'composed-filing-planner': {
        'operations': [('tables-text-planner', 'multihiertt', 'main', 'number')],
        'note': 'fan-in of per-table and text readers into a planner, then exact computation (numeric answers only)'},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--train-roots', type=int, default=160)
    ap.add_argument('--held-roots', type=int, default=30)
    a = ap.parse_args()
    rows = [json.loads(line) for path in (a.source / 'cases.jsonl', a.source / 'held' / 'dev.jsonl',
                                          a.source / 'held' / 'confirmation.jsonl') for line in open(path) if line.strip()]
    by = defaultdict(list)
    for r in rows:
        ext = r['curriculum']['external']
        by[(ext['structure']['program'], ext['origin_source'])].append(r)
    kin_labels = sorted({r['semantics']['expected'] for r in rows if r['curriculum']['external']['origin_source'] == 'clutrr'})
    types = {'kinship': 'export type Kinship = ' + ' | '.join(json.dumps(l) for l in kin_labels) + ';\n',
             'choice': 'export type Choice = "A" | "B" | "C" | "D";\n'}
    a.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, spec in PROGRAMS.items():
        directory = a.out / name
        if directory.exists():
            raise SystemExit(f'{directory} exists; tune programs are never overwritten')
        files, examples, components, counts = {}, [], {}, defaultdict(int)
        for structure, origin, root, returns in spec['operations']:
            cases = sorted(by[(structure, origin)], key=lambda r: r['id'])
            if returns in ('number', 'string'):
                cases = [r for r in cases if r['semantics']['files'][r['semantics']['root']].split('returns: ', 1)[1]
                         .split('\n', 1)[0].strip('"') == returns]
            if not cases:
                raise SystemExit(f'{name}: no cases for {structure}/{origin}')
            # One fixed program text: the first instruction variant (by case id order) of this structure.
            template = cases[0]
            old_root = template['semantics']['root'][:-3]
            for path, text in template['semantics']['files'].items():
                new = root + path[len(old_root):]
                if path == template['semantics']['root'] and returns:
                    text = simple_returns(text, returns)
                files[new] = text
            taken = defaultdict(int)
            for r in cases:
                split = r['split']
                cap = a.train_roots if split == 'train' else a.held_roots
                if taken[split] >= cap or oracle_of(r) is None:
                    continue
                taken[split] += 1
                group = r['curriculum']['split_group']
                components[group] = PARTITION[split]
                ref = r['curriculum']['reference']['root'][-1][1]['value']  # the verified accepted answer form
                expected = r['semantics']['expected'] if not isinstance(r['semantics']['expected'], dict) else ref
                examples.append({'function': f'{root}.nl', 'input': r['semantics']['inputs'], 'expected': expected,
                                 'group': group, 'reference': {'oracle': oracle_of(r), 'task': r['id'],
                                                               'source_ids': r['source_ids']}})
                counts[f'{root}:{split}'] += 1
        if 'types' in spec:
            files['types.ts'] = types[spec['types']]
        for relative, text in files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (directory / 'package.json').write_text(json.dumps({'name': f'@natlang/tune-{name}', 'private': True, 'type': 'module'}) + '\n')
        (directory / 'tune.reward.mjs').write_text(VERIFIER)
        payload = ''.join(json.dumps(e, sort_keys=True) + '\n' for e in examples)
        (directory / 'tune.examples.jsonl').write_text(payload)
        (directory / '.natlang' / 'tune').mkdir(parents=True)
        (directory / '.natlang' / 'tune' / 'splits.json').write_text(json.dumps(
            {'schema': 'natlang.tune-split/v1', 'components': dict(sorted(components.items()))}, indent=1) + '\n')
        summaries.append({'program': name, 'path': str(directory), 'note': spec['note'], 'roots': len(examples),
                          'by_operation_split': dict(counts),
                          'examples_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                          'files': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(files.items())}})
    manifest = {'schema': 'natlang.program-supply/1', 'source': str(a.source), 'programs': summaries,
                'oracle': 'tune.reward.mjs: the RL rows\' verified outcome oracles', 'teacher_generation': False,
                'production_exposure': False, 'capability_lift': None}
    (a.out / 'programs-prepared.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps([{k: s[k] for k in ('program', 'roots', 'by_operation_split')} for s in summaries], indent=1))


if __name__ == '__main__':
    main()
