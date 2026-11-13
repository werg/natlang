#!/usr/bin/env python3
"""Neuralese-declaring variants of existing pool tasks (plans/neuralese/TASK_SUPPLY_20261113.md).

The first Neuralese addition of the experiment plan is data: the same task with a callable folder helper whose declared
result is ``Neuralese<string>`` (a private per-document/per-item/per-part note the root fans in), exactly the
``neuralese-helper`` framing ``common.make_case`` gives half of the split units of a source that has one. A variant
keeps the original's inputs, oracle, expected value, wrong/alias checks, reference, split, ``split_group`` and
``source_groups`` (so the held-v2 closure that excludes the original excludes the variant too); it adds the helper
file and one root-note paragraph, and takes the id ``make_case`` gives the neuralese framing of that item.

Helpers come from two places:
* families whose generator already has a Neuralese framing: the helper file and root note are read from the pool's own
  ``neuralese-helper`` rows of that family (one helper, one note), so a variant is byte-for-byte what the generator
  would have produced for that item;
* ``AUTHORED``: families without one, one helper per family written here (input-specific notes, never a constant cue);
* ``--task-helpers``: per-task helpers (JSONL ``{id, applicable, helper, args, instruction, root_note}``, authored by
  Haiku subagents for measured in-band tasks whose family has neither), checked by ``task_helper_problems``.

Rows: eligible pool tasks (grpo.load_tasks: held-v2 ids, siblings and shared source groups excluded) whose framing is
plain and that declare no Neuralese yet, from families whose measured base pass rate lies in ``--family-band`` plus
every individually measured in-band task, at most ``--per-family`` per family (measured in-band tasks first).

    neuralese_variants.py build --out DIR      # DIR/raw-cases.jsonl + DIR/selection.json; then verify.mjs
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(REPO / 'scripts/neuralese_data')]

from scripts.rl_tasks.common import nl_file, slug, text_digest  # noqa: E402

FRAMING = 'neuralese-helper'

# Families without a generator framing: (helper name, args, instruction, root note). Each note is about the call's own
# inputs (span basis: input-specific content the consumer needs).
AUTHORED = {
    'babi': ('note_facts', {'facts': 'string[]', 'question': 'string'},
             'Read facts, a story or part of one in time order, for what question needs and write a private note of '
             'the facts that decide it, in order, and the state they leave (who is where, who holds what, what is '
             'related how), or that none of them bear on it.',
             'note_facts gives a private note of what a run of facts says about question. You may hand it the facts, '
             'or consecutive groups of them, and answer from the notes.'),
    'ext_clutrr_kinship': ('relation_in', {'sentence': 'string'},
                           'Write a private note of the family relation this sentence states directly, with its '
                           'direction (who is whose what), or that it states none.',
                           'relation_in gives a private note of the relation one sentence of story states. You may '
                           'split story into its sentences, read each with relation_in and combine the notes along '
                           'the path from target to anchor.'),
    'ext_logiqa2_logic_mcq': ('weigh_option', {'passage': 'string', 'question': 'string', 'option': 'string'},
                              'Check option, one candidate answer to question about passage, with strict logic and '
                              'write a private note of what in passage supports it or rules it out.',
                              'weigh_option gives a private note on one option. You may weigh each option with it '
                              'and decide from the notes.'),
    'ext_drop_passage_reasoning': ('read_part', {'part': 'string', 'question': 'string'},
                                   'Read part, a consecutive part of a passage, and write a private note of every '
                                   'name, date and number in it that question needs, with what each refers to, or '
                                   'that it has none.',
                                   'read_part gives a private note of what one part of passage says that question '
                                   'needs. You may split passage into parts, read each, and work out the answer '
                                   'exactly in eval from the notes and passage.'),
    'tatqa': ('read_paragraph', {'question': 'string', 'paragraph': 'string'},
              'Read paragraph, one paragraph of an annual-report excerpt, and write a private note of the figures and '
              'facts in it that question needs, with their units and periods, or that it has none.',
              'read_paragraph gives a private note of one paragraph. You may read the paragraphs with it, then take '
              'exact figures from table and the notes and compute the answer in eval.'),
    'finqa': ('read_text', {'question': 'string', 'sentences': 'string[]'},
              "Read sentences, part of a financial report's text, and write a private note of the figures in them "
              'that question needs, with their units and years, or that they have none.',
              "read_text gives a private note of a group of the report's sentences. You may read text in groups with "
              'it, then compute the answer exactly in eval from the notes and table.'),
    'ext_wikitablequestions_table_qa': ('read_rows', {'question': 'string', 'columns': 'string[]', 'rows': 'string[][]'},
                                        'Read rows (cells in columns order) for everything question depends on, and '
                                        'write a private note of the relevant cells, or that they have none.',
                                        'read_rows gives a private note on a slice of the rows. You may split a long '
                                        'table into slices, read each, and answer from the notes together with your '
                                        'own exact counting and comparison in eval.'),
}


def authored_key(family):
    if family.startswith('ext_babi_task'):
        return 'babi'
    if family.startswith('ext_tatqa_'):
        return 'tatqa'
    if family in ('ext_finqa_report_arithmetic', 'ext_convfinqa_report_conversation'):
        return 'finqa'
    return family if family in AUTHORED else None


def files_of(row):
    return (row.get('semantics') or {}).get('files') or {}


def declares_neuralese(row):
    return any('Neuralese<' in text for text in files_of(row).values() if isinstance(text, str))


def framing_of(row):
    return ((row.get('curriculum') or {}).get('external') or {}).get('framing')


def root_body(text):
    """(frontmatter including its closing ---, body) of an .nl file."""
    if not text.startswith('---'):
        return '', text
    end = text.index('\n---', 3) + 4
    return text[:end], text[end:]


def generator_helpers(rows):
    """{family: (helper basename, helper file text, root note)} from the pool's own neuralese-helper rows, for families
    whose rows all use one helper and one note."""
    seen = collections.defaultdict(set)
    for row in rows:
        if framing_of(row) != FRAMING:
            continue
        sem = row['semantics']
        stem = sem['root'][:-3]
        helpers = [(k.split('/')[-1], v) for k, v in sem['files'].items()
                   if k.startswith(stem + '/') and k.count('/') == stem.count('/') + 1 and 'Neuralese<' in v]
        if len(helpers) != 1:
            seen[row['family']].add(None)
            continue
        note = root_body(sem['files'][sem['root']])[1].rstrip().split('\n\n')[-1].strip()
        seen[row['family']].add((helpers[0][0], helpers[0][1], note))
    return {f: next(iter(v)) for f, v in seen.items() if len(v) == 1 and None not in v
            and not f.endswith('_guided')}


def variant(row, helper_name, helper_text, note):
    """The neuralese-helper framing of ``row``: helper file in the root's folder, note appended to the root body."""
    out = copy.deepcopy(row)
    sem = out['semantics']
    stem = sem['root'][:-3]
    path = f'{stem}/{helper_name}'
    if path in sem['files']:
        raise ValueError(f'{row["id"]}: {path} exists')
    head, body = root_body(sem['files'][sem['root']])
    instruction = body.strip() + '\n\n' + note.strip()
    sem['files'][sem['root']] = head + '\n' + instruction + '\n'
    sem['files'][path] = helper_text
    cur = out['curriculum']
    ext = cur.setdefault('external', {})
    upstream_id, shape = ext.get('upstream_id'), ext.get('shape')
    if upstream_id and shape and row['id'].startswith('ext:') \
            and row['id'].endswith(':' + text_digest(upstream_id + shape + (ext.get('framing') or 'plain'))[:8]):
        out['id'] = row['id'].rsplit(':', 1)[0] + ':' + text_digest(upstream_id + shape + FRAMING)[:8]
    else:
        out['id'] = row['id'] + ':' + text_digest(row['id'] + '|' + FRAMING)[:8]
    ext.update(framing=FRAMING, variant_of=row['id'], instruction_sha256=text_digest(instruction)[:16],
               instruction_id=text_digest(instruction)[:12],
               neuralese_room=ext.get('neuralese_room') or f'{helper_name[:-3]} notes fanned in to the root')
    out['generation'] = {**(out.get('generation') or {}), 'variant_of': row['id'],
                         'variant_method': 'natlang.neuralese_variants/1: neuralese-helper framing of a pool task; '
                                           'inputs, oracle and reference unchanged'}
    return out


def helper_file(args, text):
    return nl_file(args, 'Neuralese<string>', text)


RESERVED = {'call', 'apply', 'bind', 'name', 'length', 'prototype', 'then', 'iterateOn', 'read', 'map', 'view', 'ask',
            'toString', 'constructor'}
PLAIN_TYPES = {'string', 'number', 'boolean', 'string[]', 'string[][]', 'number[]', 'Record<string, string>'}


def task_helper_problems(row, spec):
    """Structural checks of an authored per-task helper before replay qualification (verify.mjs then loads and replays
    the program): a fresh identifier, 1-3 arguments typed with plain types, type text of the root's own arguments or
    named types the program's files mention (verify.mjs rejects an unknown type when it loads the helper), an
    input-specific note request and a root note naming the helper."""
    import re
    problems = []
    name = spec.get('helper') or ''
    sem = row['semantics']
    stem = sem['root'][:-3]
    taken = {Path(k).stem for k in sem['files']} | set(sem.get('inputs') or {})
    if not re.fullmatch(r'[a-z][a-z0-9_]{2,40}', name) or name in RESERVED or name in taken:
        problems.append(f'helper name {name!r}')
    args = spec.get('args') or {}
    root_types = set(re.findall(r'^  \w+: (.*)$', root_body(sem['files'][sem['root']])[0], re.M))
    root_types = {json.loads(t) if t.startswith('"') else t for t in root_types}
    root_types |= {t[:-2] for t in root_types if t.endswith('[]')}
    if not 1 <= len(args) <= 3:
        problems.append(f'{len(args)} args')
    for arg, type_text in args.items():
        named = set(re.findall(r'\b[A-Z]\w*', type_text)) - {'Record', 'Array'}
        known = type_text in PLAIN_TYPES or type_text in root_types or all(
            any(re.search(rf'\b(type|interface)\s+{n}\b|\b{n}\b', t) for t in sem['files'].values()) for n in named)
        if not re.fullmatch(r'[a-z][A-Za-z0-9_]*', arg) or not known:
            problems.append(f'arg {arg}: {type_text}')
    instruction, note = (spec.get('instruction') or '').strip(), (spec.get('root_note') or '').strip()
    if 'private note' not in instruction or len(instruction) > 600:
        problems.append('instruction')
    if not note.startswith(name) or len(note) > 500:
        problems.append('root note')
    if f'{stem}/{name}.nl' in sem['files']:
        problems.append('helper file exists')
    return problems


def build(a):
    import grpo
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    pools = a.pool or grpo.POOLS
    tasks, excluded = grpo.load_tasks(SimpleNamespace(
        pool=pools, held_manifest=a.held_manifest, exclude_manifest=[], task_ids=None,
        exclude_family_regex=r'^ext_codebase-(py|natlang)_codebase_', seed=a.seed, pass_rates=None, pass_band=None,
        pass_min_scored=None, explore=None, max_model_requests=None, case_seconds=None))
    from natlang_neuralese.common.jsonio import read_jsonl
    every = [r for p in pools for r in read_jsonl(p)]
    gen = generator_helpers(every)
    rates = {k: (v['successes'], v['scored']) for k, v in json.loads(Path(a.pass_rates).read_text()).items()
             if v.get('scored', 0) >= 2}
    by_id = {r['id']: r for r in tasks}
    fam = collections.defaultdict(lambda: [0, 0])
    for k, (s, n) in rates.items():
        if k in by_id:
            fam[by_id[k]['family']][0] += s
            fam[by_id[k]['family']][1] += n
    low, high = a.family_band
    in_band = {k for k, (s, n) in rates.items() if k in by_id and 0.1 <= s / n <= 0.9}
    existing_ids = {r['id'] for r in every}
    authored_rows, rejected = [], {}
    for spec in (read_jsonl(a.task_helpers) if a.task_helpers else ()):
        row = by_id.get(spec['id'])
        if row is None or not spec.get('applicable') or declares_neuralese(row):
            rejected[spec['id']] = 'not applicable' if row is not None else 'not an eligible task'
            continue
        problems = task_helper_problems(row, spec)
        if problems:
            rejected[spec['id']] = problems
            continue
        out = variant(row, spec['helper'] + '.nl', helper_file(spec['args'], spec['instruction']), spec['root_note'])
        out['generation']['helper_authoring'] = 'haiku subagent, per task (TASK_SUPPLY_20261113)'
        authored_rows.append(out)
    authored_ids = {r['generation']['variant_of'] for r in authored_rows}
    chosen, skipped = collections.defaultdict(list), collections.Counter()
    order = sorted(tasks, key=lambda r: r['id'])
    random.Random(a.seed).shuffle(order)
    order.sort(key=lambda r: r['id'] not in in_band)
    for row in order:
        family = row['family']
        if declares_neuralese(row) or framing_of(row) not in ('plain', None) or row['id'] in authored_ids:
            continue
        if family in gen:
            name, text, note = gen[family]
        elif authored_key(family):
            name, args, instruction, note = AUTHORED[authored_key(family)]
            name, text = name + '.nl', helper_file(args, instruction)
        else:
            skipped['no helper for family'] += 1
            continue
        s, n = fam.get(family, (0, 0))
        if row['id'] not in in_band and not (n and low <= s / n <= high):
            skipped['family outside band'] += 1
            continue
        if len(chosen[family]) >= a.per_family:
            skipped['family cap'] += 1
            continue
        out = variant(row, name, text, note)
        if out['id'] in existing_ids:
            skipped['variant id already in pool'] += 1
            continue
        chosen[family].append(out)
    rows = [r for f in sorted(chosen) for r in chosen[f]] + authored_rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'raw-cases.jsonl').open('w') as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    selection = {
        'schema': 'natlang.neuralese-variants-selection/1',
        'pools': [{'path': str(p), 'sha256': sha256_file_hex_cached(Path(p))} for p in pools],
        'held_manifest': a.held_manifest, 'held_manifest_sha256': sha256_file_hex_cached(Path(a.held_manifest)),
        'excluded_ids': excluded, 'eligible': len(tasks), 'pass_rates': a.pass_rates,
        'pass_rates_sha256': sha256_file_hex_cached(Path(a.pass_rates)), 'family_band': [low, high],
        'per_family': a.per_family, 'rows': len(rows), 'measured_in_band_rows': sum(r['generation']['variant_of'] in in_band for r in rows),
        'helpers': {'generator': sorted(f for f in chosen if f in gen), 'authored': sorted(f for f in chosen if f not in gen)},
        'by_family': {f: len(v) for f, v in sorted(chosen.items())}, 'skipped': dict(skipped),
        'task_helpers': a.task_helpers and {'path': a.task_helpers, 'sha256': sha256_file_hex_cached(Path(a.task_helpers)),
                                            'rows': len(authored_rows), 'rejected': rejected}}
    (out / 'selection.json').write_text(json.dumps(selection, indent=1) + '\n')
    print(json.dumps({k: selection[k] for k in ('rows', 'measured_in_band_rows', 'skipped')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--pool', action='append', help='pool cases.jsonl (default: grpo.POOLS)')
    b.add_argument('--held-manifest', default=None)
    b.add_argument('--pass-rates', default=None)
    b.add_argument('--family-band', type=float, nargs=2, default=(0.1, 0.9))
    b.add_argument('--per-family', type=int, default=120)
    b.add_argument('--task-helpers', help='per-task authored helpers (JSONL)')
    b.add_argument('--seed', type=int, default=20261113)
    a = parser.parse_args()
    import grpo
    a.held_manifest = a.held_manifest or grpo.HELD
    a.pass_rates = a.pass_rates or str(Path(grpo.HELD).parent.parent / 'exp1-rft' / 'base-pass-rates.json')
    build(a)


if __name__ == '__main__':
    main()
