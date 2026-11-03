#!/usr/bin/env python3
"""Assemble a balanced RL selection pool from verified train task sources.

The 2026-11-03 recipe review found the RL source 98% one dataset-wrapper generator. The pool is the cases file the
interleaved RL stage binds (`rl_task_source`). It keeps verified rows byte-for-byte except scheduling metadata:

1. Generator balance: rows are classed by generator (``generation.generator``; the broad external builder is split
   into dataset / codebase / composed by source, and new builders carry their own generator ids). Water-filling picks
   a pool in which no class exceeds ``--max-generator-share`` of the rows; a class smaller than its cap is kept whole.
   Effect/state construct share (``--min-effect-share``, default 0.3, the owner bar) is enforced at SELECTION time, never by
   dropping candidates (2026-11-03 supervisor decision: the earlier pool shrank from 11,275 to 6,666 rows and lost 1,653
   distinct programs to reach the share). Every row that survives the generator and family caps stays in the pool; rows
   whose oracle verifies a world after the run (files, service state, repaired code; see
   ``measure_repetition.program_features``) are tagged ``curriculum.rl.mix_class = "effect_state"``, and the receipt records
   the ``class_quota`` the interleaved RL controller must bind (the selector ``natlang.rl-task-curriculum/2`` fills at least
   that share of every round from the tagged rows; ``simulate_selection.py`` measures the sampled share). Rows are never
   duplicated or invented. An author-declared complete returned state with no world is counted separately and not tagged.
   Train rows related to any registered protected packet or held source are dropped before selection.
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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from measure_repetition import program_features, report  # noqa: E402
from natlang_neuralese.common.custody import ProtectedWorlds  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402

# The selector's class quota exceeds the owner bar by this margin so rounds that cannot reach it exactly still clear it.
QUOTA_MARGIN = 0.04

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


def domain_of(row, domains):
    source = str(((row.get('curriculum') or {}).get('external') or {}).get('source') or '')
    for key, domain in domains.items():
        if source == key or source.startswith(key):
            return domain
    return None


def apply_program_caps(rows, caps, dropped):
    """Keep at most ``max_rows_per_program`` rows of each normalized program of the named external sources
    (``curriculum.external.source``): program-poor supplies (one template over thousands of worlds) would otherwise
    shrink the pool's effective program count. The survivors are the rows with the smallest id hashes, which spreads
    them over worlds; nothing is duplicated or altered."""
    import hashlib
    limits = {item['source']: int(item['max_rows_per_program']) for item in caps}
    if not limits:
        return rows
    groups = collections.defaultdict(list)
    kept = []
    for row, features in rows:
        source = str(((row.get('curriculum') or {}).get('external') or {}).get('source') or '')
        if source in limits:
            groups[(source, features['program'])].append((row, features))
        else:
            kept.append((row, features))
    for (source, _), members in groups.items():
        members.sort(key=lambda item: hashlib.sha256(item[0]['id'].encode()).hexdigest())
        kept += members[:limits[source]]
        dropped[source] += max(0, len(members) - limits[source])
    order = {id(item[0]): index for index, item in enumerate(rows)}
    kept.sort(key=lambda item: order[id(item[0])])
    return kept


def load(paths, exclude_ids=(), program_caps=(), dropped=None, protected=None):
    """Verified train rows of the inputs. ``protected`` (ProtectedWorlds) is the fail-closed custody guard: a train row
    whose world is related to any registered evaluation-only packet or held source is never selectable, whatever its
    input's own custody said (an input can predate the registration of its siblings' held rows)."""
    dropped = collections.Counter() if dropped is None else dropped
    excluded = set(exclude_ids)
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
                if row['id'] in excluded:
                    dropped['excluded_ids'] += 1
                    continue
                if protected is not None and protected.related(row):
                    dropped['protected_world_related'] += 1
                    continue
                rows.append((row, program_features(row)))
    return apply_program_caps(rows, program_caps, dropped)


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


def take(members, count, family_cap, family_counts):
    """Round-robin across (family, level), effect/state oracles first inside each queue; repeated normalized programs
    come after every new program; no source family exceeds ``family_cap`` rows of the pool."""
    queues = collections.OrderedDict()
    for row, features in sorted(members, key=lambda item: (not item[1]['effect_state_oracle'], item[0]['id'])):
        queues.setdefault((features['family'], features['level']), collections.deque()).append((row, features))
    chosen, programs, repeats = [], set(), collections.OrderedDict()
    def admit(row, features):
        chosen.append((row, features))
        programs.add(features['program'])
        family_counts[features['family']] += 1
    while len(chosen) < count and queues:
        for key in list(queues):
            queue = queues[key]
            if family_counts[key[0]] >= family_cap:
                del queues[key]
                continue
            while queue and queue[0][1]['program'] in programs:
                repeats.setdefault(key, collections.deque()).append(queue.popleft())
            if queue:
                admit(*queue.popleft())
            if not queue:
                del queues[key]
            if len(chosen) >= count:
                break
    while len(chosen) < count and repeats:
        for key in list(repeats):
            if family_counts[key[0]] < family_cap and repeats[key]:
                admit(*repeats[key].popleft())
            if not repeats[key] or family_counts[key[0]] >= family_cap:
                del repeats[key]
            if len(chosen) >= count:
                break
    return chosen


def usable_rows(members, family_cap):
    per_family = collections.Counter(features['family'] for _, features in members)
    return sum(min(n, family_cap) for n in per_family.values())


def select(classes, total, share, family_share):
    """Choose the pool rows: generator caps (water-filling) plus the family cap, effect/state oracles first inside each
    queue. A family (one program design or template) may hold at most family_share of the pool, effect oracles included:
    the cap is solved jointly with the generator caps (two fixed-point passes suffice for these sizes). No row is
    duplicated or invented, and construct shares are never met by shrinking the pool (the selector enforces them)."""
    family_cap = total
    for _ in range(3):
        limits = caps({g: usable_rows(m, family_cap) for g, m in classes.items()}, share)
        family_cap = max(1, int(family_share * sum(limits.values())))
    pool, family_counts = [], collections.Counter()
    for generator, members in sorted(classes.items()):
        pool += take(members, limits[generator], family_cap, family_counts)
    return pool, family_cap, limits


def assemble(inputs, out, *, share, domains, family_share=0.05, min_effect_share=0.0, exclude_ids=(), program_caps=(),
             protected=None):
    out = Path(out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    dropped = collections.Counter()
    protected = ProtectedWorlds.load() if protected is None else protected
    rows = load([path for path in inputs if Path(path).stat().st_size], exclude_ids, program_caps, dropped, protected)
    classes = collections.defaultdict(list)
    for row, features in rows:
        classes[features['generator']].append((row, features))
    pool, family_cap, limits = select(classes, len(rows), share, family_share)
    effect_rows = sum(1 for _, features in pool if features['effect_state_oracle'])
    quota = {'mode': 'selection-time', 'min_effect_share': min_effect_share, 'rows': len(pool), 'effect_rows': effect_rows,
             'effect_share_of_pool': round(effect_rows / max(1, len(pool)), 4),
             'returned_state_rows': sum(1 for _, features in pool if features['returned_state_oracle']),
             'class_quota': {'effect_state': round(min_effect_share + QUOTA_MARGIN, 4)} if min_effect_share else {},
             'note': 'no row dropped for the share: effect/state rows carry curriculum.rl.mix_class and the curriculum '
                     'selector fills the class_quota of every round from them'}
    merges = collections.Counter()
    written = []
    family_programs = collections.defaultdict(set)
    for row, features in pool:
        family_programs[features['family']].add(features['program'])
    for row, features in pool:
        rl = dict((row.get('curriculum') or {}).get('rl') or {})
        merged = None
        tag = bool(min_effect_share) and features['effect_state_oracle']
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
        if tag:
            rl['mix_class'] = 'effect_state'
        if merged is not None or tag:
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
               'exclude_ids': sorted(exclude_ids), 'program_caps': list(program_caps), 'dropped_before_selection': dict(dropped),
               'protected_worlds_digest': protected.digest,
               'max_generator_share': share, 'max_family_share': family_share, 'family_cap_rows': family_cap,
               'quota': quota,
               'generator_classes': {c: {'available': len(m), 'kept': sum(1 for _, f in pool if f['generator'] == c)}
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
    parser.add_argument('--cases', action='append', default=[], help='verified train cases JSONL (repeatable)')
    parser.add_argument('--spec', help='JSON {"cases": [paths], "max_generator_share", "max_family_share", "min_effect_share", "exclude_ids": [ids], "program_caps": [{"source", "max_rows_per_program"}]}; '
                        'empty or missing files (e.g. a not-yet-filled admitted/cases.jsonl) are skipped')
    parser.add_argument('--out', help='new output directory')
    parser.add_argument('--out-root', help='write a fresh selection-pool-<UTC time> here and update latest.json')
    parser.add_argument('--max-family-share', type=float, default=0.05)
    parser.add_argument('--max-generator-share', type=float, default=0.2)
    parser.add_argument('--min-effect-share', type=float, default=None,
                        help='effect/state construct share (default 0.3, the owner bar; 0 disables), enforced by the '
                             'selector from tagged rows: no candidate row is dropped for it')
    parser.add_argument('--domain-map', help='JSON object source-prefix -> domain (default: built-in map)')
    args = parser.parse_args(argv)
    domains = json.loads(Path(args.domain_map).read_text()) if args.domain_map else DEFAULT_DOMAINS
    inputs, share, family_share = list(args.cases), args.max_generator_share, args.max_family_share
    min_effect = args.min_effect_share
    exclude_ids, program_caps = [], []
    if args.spec:
        spec = json.loads(Path(args.spec).read_text())
        inputs += spec['cases']
        share = spec.get('max_generator_share', share)
        family_share = spec.get('max_family_share', family_share)
        if min_effect is None:
            min_effect = spec.get('min_effect_share')
        exclude_ids += list(spec.get('exclude_ids', []))
        program_caps = list(spec.get('program_caps', []))
    inputs = [path for path in inputs if Path(path).is_file()]
    if not inputs:
        raise SystemExit('no task source files')
    if args.out_root:
        import time
        root = Path(args.out_root)
        out = root / time.strftime('selection-pool-%Y%m%dT%H%M%SZ', time.gmtime())
    elif args.out:
        out = Path(args.out)
    else:
        raise SystemExit('give --out or --out-root')
    receipt = assemble(inputs, out, share=share, domains=domains, family_share=family_share,
                       min_effect_share=0.3 if min_effect is None else min_effect,
                       exclude_ids=exclude_ids, program_caps=program_caps)
    if args.out_root:
        latest = {'pool': str(out.resolve()), 'cases': str((out / 'cases.jsonl').resolve()),
                  'sha256': receipt['cases']['sha256'], 'rows': receipt['cases']['rows']}
        pending = root / 'latest.json.pending'
        pending.write_text(json.dumps(latest, indent=1) + '\n')
        pending.replace(root / 'latest.json')


if __name__ == '__main__':
    sys.exit(main())
