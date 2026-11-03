#!/usr/bin/env python3
"""Assemble a balanced RL selection pool from verified train task sources.

The 2026-11-03 recipe review found the RL source 98% one dataset-wrapper generator. The pool is the cases file the
interleaved RL stage binds (`rl_task_source`). It keeps verified rows byte-for-byte except scheduling metadata:

1. Generator balance: rows are classed by generator (``generation.generator``; the broad external builder is split
   into dataset / codebase / composed by source, and new builders carry their own generator ids). Water-filling picks
   a pool in which no class exceeds ``--max-generator-share`` of the rows; a class smaller than its cap is kept whole.
2. Within a capped class, rows are taken round-robin across families and levels, preferring effect/state oracles and
   not-yet-taken normalized programs, so the cap drops repeated templates first.
3. Family consolidation (only families with fewer than 20 distinct normalized programs in the pool; procedural
   inline-curriculum families merge by curriculum slice): dataset-wrapper families whose programs are one template per item are scheduling labels, not
   task families. They are merged into their declared domain (``--domain-map``, source -> domain); codebase question
   families (one question template with a substituted symbol) are merged per codebase source and oracle depth. So
   difficulty frontiers and diversity are counted over real breadth. The original family is kept in
   ``curriculum.rl.source_family``; task ids, programs, oracles and splits are unchanged.

Outputs (in ``--out``, a new directory): ``cases.jsonl``, ``receipt.json`` (inputs with SHA-256, per-class counts and
caps, family merges) and ``repetition.json`` (scripts/rl_tasks/measure_repetition.py over the pool).
Only train rows are read; protection and oracle qualification are the inputs' own (each input is a verified source).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import program_features, report  # noqa: E402

# Upstream dataset sources of the broad builder, grouped by the kind of reasoning their tasks need.
DEFAULT_DOMAINS = {
    'hotpotqa': 'multidoc_qa', 'musique': 'multidoc_qa', '2wikimultihopqa': 'multidoc_qa', 'triviaqa': 'reading_qa',
    'squad': 'reading_qa', 'quality': 'reading_qa', 'drop': 'reading_qa', 'qasper': 'reading_qa',
    'hybridqa': 'table_text_qa', 'tatqa': 'table_text_qa', 'finqa': 'table_text_qa', 'convfinqa': 'table_text_qa',
    'multihiertt': 'table_text_qa', 'wikitablequestions': 'table_qa', 'msr_sqa': 'table_qa', 'tabfact': 'table_qa',
    'databench': 'table_qa', 'wikisql': 'sql', 'spider': 'sql', 'bird': 'sql', 'gsm8k': 'math', 'math': 'math',
    'nestful': 'tool_calls', 'xlam': 'tool_calls', 'toolace': 'tool_calls', 'babi': 'logic', 'clutrr': 'logic',
    'logiqa2': 'logic', 'kk': 'logic', 'knights_knaves': 'logic', 'reasoning-gym': 'puzzles', 'countdown': 'puzzles',
    'rg-': 'puzzles', 'msr-sqa': 'table_qa', 'knights-knaves': 'logic', 'mbpp': 'code', 'design-brief': 'design_briefs', 'design_brief_gates': 'design_briefs', 'tau': 'agent_store',
    'apigen': 'agent_store',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def domain_of(row, domains):
    source = str(((row.get('curriculum') or {}).get('external') or {}).get('source') or '')
    for key, domain in domains.items():
        if source == key or source.startswith(key):
            return domain
    return None


def load(paths):
    rows, seen = [], set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('split') != 'train':
                    continue
                if row['id'] in seen:
                    raise SystemExit(f'duplicate task id across inputs: {row["id"]}')
                seen.add(row['id'])
                rows.append((row, program_features(row)))
    return rows


def caps(sizes, share):
    """Largest pool in which every class is at most ``share`` of the rows (water-filling)."""
    if share * len(sizes) < 1:
        raise SystemExit(f'{len(sizes)} generator classes cannot satisfy a {share} share cap')
    capped = set()
    while True:
        free = sum(n for c, n in sizes.items() if c not in capped)
        # total = free + share*total*len(capped)  =>  total = free / (1 - share*len(capped))
        total = free / (1 - share * len(capped))
        over = {c for c, n in sizes.items() if c not in capped and n > share * total}
        if not over:
            limit = int(share * total)
            return {c: (min(n, limit) if c in capped else n) for c, n in sizes.items()}
        capped |= over


def take(members, count):
    """Round-robin across (family, level), preferring effect/state oracles; repeated normalized programs come last."""
    queues = collections.OrderedDict()
    for row, features in sorted(members, key=lambda item: (not item[1]['effect_state_oracle'], item[0]['id'])):
        queues.setdefault((features['family'], features['level']), collections.deque()).append((row, features))
    chosen, programs, repeats = [], set(), collections.OrderedDict()
    while len(chosen) < count and queues:
        for key in list(queues):
            queue = queues[key]
            while queue and queue[0][1]['program'] in programs:
                repeats.setdefault(key, collections.deque()).append(queue.popleft())
            if queue:
                row, features = queue.popleft()
                chosen.append((row, features))
                programs.add(features['program'])
            if not queue:
                del queues[key]
            if len(chosen) >= count:
                break
    while len(chosen) < count and repeats:
        for key in list(repeats):
            chosen.append(repeats[key].popleft())
            if not repeats[key]:
                del repeats[key]
            if len(chosen) >= count:
                break
    return chosen


def assemble(inputs, out, *, share, domains):
    out = Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    rows = load(inputs)
    classes = collections.defaultdict(list)
    for row, features in rows:
        classes[features['generator']].append((row, features))
    limits = caps({c: len(m) for c, m in classes.items()}, share)
    pool = []
    for generator, members in sorted(classes.items()):
        pool += members if limits[generator] >= len(members) else take(members, limits[generator])
    merges = collections.Counter()
    written = []
    family_programs = collections.defaultdict(set)
    for row, features in pool:
        family_programs[features['family']].add(features['program'])
    for row, features in pool:
        rl = dict((row.get('curriculum') or {}).get('rl') or {})
        merged = None
        if len(family_programs[features['family']]) >= 20:
            pass  # a family with 20+ distinct programs is real breadth; keep its own frontier
        elif features['generator'] == 'natlang.inline_curriculum_generator/1':
            merged = 'inline_' + str((row.get('curriculum') or {}).get('slice') or 'other')
        elif features['generator'].endswith(':dataset'):
            domain = domain_of(row, domains)
            merged = 'ext_domain_' + domain if domain else None
        elif features['generator'].endswith(':codebase'):
            # Codebase question types substitute a symbol into one question template per family: schedule them per
            # codebase source and oracle depth (answer questions vs. effect-checked repairs).
            source = str(((row.get('curriculum') or {}).get('external') or {}).get('source') or 'codebase')
            merged = 'ext_codebase_' + source.replace('-', '_') + (
                '_effects' if features['effect_state_oracle'] else '_questions')
        if merged is not None:
            original = rl.get('family', row.get('family'))
            rl.update(source_family=original, family=merged)
            merges[(original, rl['family'])] += 1
            row = {**row, 'curriculum': {**row['curriculum'], 'rl': rl}}
        written.append(row)
    # Interleave classes so a contiguous breadth window spans generators and families.
    by_class = collections.defaultdict(collections.deque)
    for row, (_, features) in zip(written, pool):
        by_class[features['generator']].append(row)
    ordered = []
    while by_class:
        for generator in sorted(list(by_class)):
            ordered.append(by_class[generator].popleft())
            if not by_class[generator]:
                del by_class[generator]
    out.mkdir(parents=True)
    cases = out / 'cases.jsonl'
    with cases.open('w') as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    repetition = report([str(cases)], label='selection-pool')
    (out / 'repetition.json').write_text(json.dumps(repetition, indent=1) + '\n')
    receipt = {'schema': 'natlang.rl-selection-pool/1',
               'inputs': [{'path': str(Path(p).resolve()), 'sha256': sha256(p)} for p in inputs],
               'cases': {'path': 'cases.jsonl', 'sha256': sha256(cases), 'rows': len(ordered)},
               'max_generator_share': share,
               'generator_classes': {c: {'available': len(m), 'kept': min(len(m), limits[c])}
                                     for c, m in sorted(classes.items())},
               'family_merges': [{'from': a, 'to': b, 'rows': n} for (a, b), n in sorted(merges.items())],
               'domains': domains,
               'scope': 'verified train rows kept byte-identical except curriculum.rl.family for merged dataset '
                        'wrapper families (original in curriculum.rl.source_family); no oracle, split or program change'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'cases': str(cases), 'rows': len(ordered), 'classes': receipt['generator_classes']}))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', action='append', required=True, help='verified train cases JSONL (repeatable)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-generator-share', type=float, default=0.2)
    parser.add_argument('--domain-map', help='JSON object source-prefix -> domain (default: built-in map)')
    args = parser.parse_args(argv)
    domains = json.loads(Path(args.domain_map).read_text()) if args.domain_map else DEFAULT_DOMAINS
    assemble(args.cases, args.out, share=args.max_generator_share, domains=domains)


if __name__ == '__main__':
    sys.exit(main())
