#!/usr/bin/env python3
"""Matched executable references: errors, missing evidence, success, and repair.

Reference plans supervise training only; deployed execution remains model-driven.
All variants of a generated group share a program_id to prevent split leakage.
"""
import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.corpus import digest, file_digest
from natlang.gen.policy import ReferenceAgent
from natlang.gen.programs import Plan
from natlang.runtime import Runtime
from natlang.values import load_program, dump
from natlang.native import _strip_private


def matched_cases(seed, group):
    rng = random.Random(f'{seed}:{group}:failures')
    n = rng.randint(2, 50)
    word = rng.choice(['amber', 'cedar', 'otter', 'velvet', 'harbor'])
    def case(name, ty, body, steps, expected=None, failure=None, args=None, params='{}', codebase=None, effects=None, fault=None, failed_write=None):
        doc = {'type': f'Lambda<{params}, {ty}>', 'instructions': body, 'args': args or {}}
        if codebase:
            doc['codebase'] = codebase
        if effects:
            doc['effects'] = ['out.emit']
        return {'name': name, 'root': {'$lambda': doc}, 'steps': steps, 'expected': expected,
                'failure': failure, 'effects': effects or [], 'fault': fault, 'allowed_failure_write': failed_write}
    def write(value, ty='Num', path='return'):
        return ('write', {'path': path, 'type': ty, 'value': value})
    def error(message):
        return ('report_error', {'message': message})
    literal = str(n)
    for convert in (False, True):
        yield case('literal_convert' if convert else 'literal_unchanged', 'Num',
                   f'The supplied text is "{literal}". ' + ('Convert it to a number and return that number.' if convert else
                   'Return that text unchanged, preserving its Text type.'),
                   [write(n)] if convert else [error('The required unchanged Text value cannot inhabit the Num return slot.')],
                   n if convert else None, None if convert else 'error', failed_write=write(literal, 'Text')[1] if not convert else None)
    for supplied in (False, True):
        quote = f'{word.title()} shipment {n} arrived.'
        yield case('evidence_present' if supplied else 'evidence_missing', 'Text',
                   'Return an exact copy of the supplied document. If none was supplied, do not invent its contents.',
                   [write(quote, 'Text')] if supplied else [('report_blocker', {'missing': 'The document to quote was not supplied.'})],
                   quote if supplied else None, None if supplied else 'blocked',
                   args={'document': quote} if supplied else {}, params='{ document?: Text }')
    for flag in (False, True):
        yield case('conditional_valid' if flag else 'conditional_error', 'Num',
                   f'If enabled is true, return {n}. Otherwise return the text "{word}" unchanged, preserving its type.',
                   [write(n)] if flag else [error('The selected else branch requires Text but this function must return Num.')],
                   n if flag else None, None if flag else 'error', args={'enabled': flag}, params='{ enabled: Bool }', failed_write=write(word, 'Text')[1] if not flag else None)
    for possible in (False, True):
        upper = n + 2 if possible else n
        yield case('bounds_valid' if possible else 'bounds_impossible', 'Num',
                   f'Return the smallest integer strictly greater than {n} and strictly less than {upper}.',
                   [write(n + 1)] if possible else [error('No integer can satisfy both strict bounds.')],
                   n + 1 if possible else None, None if possible else 'error')
    items = [rng.randint(1, 100) for _ in range(rng.randint(1, 8))]
    fn = {'size_of': {'args': {'items': 'Num[]'}, 'returns': 'Num', 'code': 'return args.items.length'}}
    fault = ('call', {'function': 'size_of', 'to': 'return', 'inputs': {'items': 'args/items'}}, 'rejected')
    for valid in (False, True):
        steps = [('call', {'function': 'size_of', 'to': 'let/size', 'inputs': {'items': 'args/items'}}),
                 ('mark_done', {'start': 1}), write({'size': len(items)}, '{ size: Num }'), ('mark_done', {'start': 2})]
        yield case('binding_repair' if valid else 'binding_error', '{ size: Num }',
                   ('Call size_of(items), saving the number in local size.\nReturn the record { size: size }.' if valid else
                    'Call size_of(items) directly into return.\nDo not wrap, convert, or transform the result.'),
                   steps if valid else [error('size_of returns Num but the required direct destination is a record; wrapping was forbidden.')],
                   {'size': len(items)} if valid else None, None if valid else 'error',
                   args={'items': items}, params='{ items: Num[] }', codebase=fn, fault=fault)
    for valid in (False, True):
        fn = {'send': {'args': {}, 'returns': 'Num', 'effects': ['out.emit'], 'code': f'fx.out.emit({n}); return {n}'}}
        steps = [('call', {'function': 'send', 'to': 'let/sent'}), ('mark_done', {'start': 1})]
        steps += [write(n), ('mark_done', {'start': 2})] if valid else [error('The requested unchanged text conflicts with the Num return type; the preceding emission already happened.')]
        yield case('effect_success' if valid else 'effect_error', 'Num',
                   'Call send(), saving its result as sent.\n' + (f'Return the number {n}.' if valid else
                   f'Then return the text "{word}" unchanged, preserving its type.'),
                   steps, n if valid else None, None if valid else 'error', codebase=fn, effects=[n], failed_write=write(word, 'Text')[1] if not valid else None)


class FaultReference(ReferenceAgent):
    def __init__(self, *args, fault=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault = fault

    def recovery_action(self, session, calls):
        return self.fault


def generate_group(job):
    seed, group = job
    rows, counts = [], Counter()
    for c in matched_cases(seed, group):
        samples = []
        plan = Plan('calls', steps=c['steps'])
        rt = Runtime(lambda lam: FaultReference(plan, samples, fault=c['fault'], recovery_rate=1 if c['fault'] else 0))
        root = load_program(c['root'])
        outcome, value = rt.run_root(root)
        if c['failure']:
            assert outcome.kind == 'quiesced' and outcome.detail.startswith(c['failure'] + ': '), (c, outcome)
            assert samples[-1]['skill'] == ('report_error' if c['failure'] == 'error' else 'report_blocker')
        else:
            assert outcome.kind == 'done' and dump(value) == c['expected'], (c, outcome, value)
        assert rt.emitted == c['effects'], (c, rt.emitted)
        for turn, sample in enumerate(samples):
            rows.append({'id': f'failure-{seed}-{group}-{c["name"]}-{turn}',
                         'program_id': f'failure:{seed}:{group}', 'family': 'failure_' + c['name'],
                         **sample, 'tools': _strip_private(sample['tools'])})
        counts[c['name']] += 1
    return rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', type=int, default=100)
    ap.add_argument('--seed', type=int, default=81)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if min(args.groups, args.workers) < 1:
        ap.error('groups and workers must be positive')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / 'natlang').rglob('*.py')) + [Path(__file__).resolve()]
    sources += sorted((ROOT / 'natlang/prompts').glob('*.md'))
    manifest = {'version': 'failure-references/1', 'seed': args.seed, 'groups': args.groups,
                'sources': digest([(str(p.relative_to(ROOT)), file_digest(p)) for p in sources])}
    counts, turns = Counter(), 0
    # Refuse accidental overwrite; incomplete files are not treated as finished corpora.
    with args.out.open('x') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows, count in pool.map(generate_group, [(args.seed, i) for i in range(args.groups)]):
            for row in rows:
                f.write(json.dumps(row) + '\n')
            counts.update(count)
            turns += len(rows)
    manifest.update({'programs': sum(counts.values()), 'turns': turns, 'cases': dict(counts), 'sha256': file_digest(args.out)})
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
