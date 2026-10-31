#!/usr/bin/env python3
"""Assemble the broad external RL task source from per-source verified cases (plans/neuralese/BROAD_RL_TASKS_20261031.md).

  assemble.py --build BUILD_DIR --out runs/natlang-rl-broad-external-20261031 [--legacy-per-family 30]

Train rows: every source's verified train cases (reference accepted, all negative controls rejected), minus any row
related to a protected world (ProtectedWorlds over training/protected-worlds.json) or to this source's own held dev /
confirmation rows (shared ids, groups, bindings, world or visible fingerprints), interleaved source by source and
family by family; plus a small slice of the procedural generator curriculum (whole split units, round-robin over its
families, byte-identical rows) for natlang-specific skills. Held rows: verified dev and confirmation cases, written to
sibling evaluation packets. Writes cases.jsonl, held/{dev,confirmation}.jsonl, assets/ (copied), verification/,
qualification.json (natlang.rl-additional-task-source-qualification/1), diversity.json and receipt.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds, world_identities  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402

LEGACY = ROOT / 'runs/natlang-rl-train-curriculum-lfm3-20261030-stop-evidence/cases.jsonl'


def rows_of(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_sha(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def legacy_slice(per_family, seed):
    rows = rows_of(LEGACY)
    units = collections.defaultdict(list)
    for r in rows:
        c = r['curriculum']
        units[(r['family'], c.get('split_group') or c.get('pair_group') or r['id'])].append(r)
    by_family = collections.defaultdict(list)
    for (family, _), members in sorted(units.items()):
        by_family[family].append(sorted(members, key=lambda m: m['id']))
    rng = random.Random(seed)
    out = []
    for family in sorted(by_family):
        pool = by_family[family]
        rng.shuffle(pool)
        taken = 0
        for members in pool:
            if taken + len(members) > per_family:
                continue
            out.extend(members)
            taken += len(members)
            if taken >= per_family:
                break
    return out


def interleave(groups):
    """Round-robin over sources, then over each source's families, so a rotating breadth window spans both."""
    queues = {}
    for source, rows in sorted(groups.items()):
        fam = collections.defaultdict(collections.deque)
        for r in rows:
            fam[r['family']].append(r)
        order = []
        while fam:
            for name in sorted(list(fam)):
                order.append(fam[name].popleft())
                if not fam[name]:
                    del fam[name]
        queues[source] = collections.deque(order)
    out = []
    while queues:
        for source in sorted(list(queues)):
            out.append(queues[source].popleft())
            if not queues[source]:
                del queues[source]
    return out


def diversity(rows):
    by_source = collections.defaultdict(list)
    for r in rows:
        ext = (r.get('curriculum') or {}).get('external')
        by_source[ext['source'] if ext else 'legacy-procedural-curriculum'].append(r)
    report = {}
    for source, members in sorted(by_source.items()):
        instr = {hashlib.sha256(r['semantics']['files'][r['semantics']['root']].split('---', 2)[-1].encode()).hexdigest()
                 for r in members}
        levels = collections.Counter(str((r['curriculum'].get('rl') or {}).get('level')) for r in members)
        framings = collections.Counter(((r['curriculum'].get('external') or {}).get('framing') or 'plain') for r in members)
        report[source] = {
            'rows': len(members), 'families': len({r['family'] for r in members}),
            'shapes': len({r['curriculum'].get('shape') for r in members}),
            'distinct_instruction_texts': len(instr),
            'distinct_root_signatures': len({r['semantics']['files'][r['semantics']['root']].split('---')[1] for r in members}),
            'split_units': len({r['curriculum'].get('split_group') for r in members}),
            'levels': dict(sorted(levels.items())), 'framings': dict(framings),
            'oracles': dict(collections.Counter(json.dumps(r['semantics'].get('oracle') if not isinstance(
                r['semantics'].get('oracle'), dict) else r['semantics']['oracle'].get('normalization') or
                r['semantics']['oracle'].get('level')) for r in members)),
        }
    total = {'rows': len(rows), 'sources': len(by_source), 'families': len({r['family'] for r in rows}),
             'shapes': len({(r['family'], r['curriculum'].get('shape')) for r in rows}),
             'distinct_instruction_texts': sum(v['distinct_instruction_texts'] for v in report.values()),
             'levels': dict(sorted(collections.Counter(str((r['curriculum'].get('rl') or {}).get('level')) for r in rows).items())),
             'neuralese_helper_rows': sum(((r['curriculum'].get('external') or {}).get('framing') == 'neuralese-helper') for r in rows)}
    return {'total': total, 'by_source': report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--legacy-per-family', type=int, default=30)
    ap.add_argument('--train-cap', type=int, default=1500)
    ap.add_argument('--cap-override', default='reasoning-gym=4000',
                    help='source=rows,... for sources whose many families need more than --train-cap')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--exclude-ids', type=Path, action='append', default=[],
                    help='JSON {"reason": label, "conflicts": [task ids]} (e.g. a held-set union check); those rows are '
                         'left out of train and counted under the label; repeatable')
    ap.add_argument('--seed', type=int, default=20261031)
    a = ap.parse_args()
    excluded = set(filter(None, a.exclude.split(',')))
    caps = {k: int(v) for k, v in (item.split('=') for item in a.cap_override.split(',') if item)}
    out = a.out
    source_id = out.name
    exclusions = {}
    for path in a.exclude_ids:
        value = json.loads(path.read_text())
        for task_id in value['conflicts']:
            exclusions[task_id] = value.get('reason', 'held-set-union-group')
    if out.exists():
        raise SystemExit(f'{out} exists; task sources are immutable')
    (out / 'held').mkdir(parents=True)
    (out / 'verification').mkdir()
    train, held, per_source = collections.defaultdict(list), {'dev': [], 'confirmation': []}, {}
    for verified in sorted(a.build.glob('*/verified/verification.json')):
        source = verified.parent.parent.name
        if source in excluded:
            continue
        report = json.loads(verified.read_text())
        cases = rows_of(verified.parent / 'cases.jsonl')
        if sha_file(verified.parent / 'cases.jsonl') != report['cases_sha256']:
            raise SystemExit(f'{source}: verified cases differ from their verification report')
        shutil.copyfile(verified, out / 'verification' / f'{source}.json')
        rng = random.Random(a.seed)
        rng.shuffle(cases)
        kept_train = [r for r in cases if r['split'] == 'train'][:caps.get(source, a.train_cap)]
        train[source] = kept_train
        for split in ('dev', 'confirmation'):
            held[split] += [r for r in cases if r['split'] == split]
        per_source[source] = {'verification': report['summary'], 'train_selected': len(kept_train),
                              'verification_sha256': sha_file(verified)}
    legacy = legacy_slice(a.legacy_per_family, a.seed)
    train['legacy-procedural-curriculum'] = legacy

    # Custody: protected worlds (registry) plus this source's own held rows.
    protected = ProtectedWorlds.load()
    own_held = ProtectedWorlds.from_cases({'own-held': ('pending', held['dev'] + held['confirmation'])})
    dropped = collections.Counter()
    blocked = protected.identities | own_held.identities
    for source in list(train):
        kept = []
        for r in train[source]:
            # One identity computation per row (inputs can be large); the same relation ProtectedWorlds.related uses.
            kinds = sorted({kind for kind, _ in world_identities(r, evidence_items=False) & blocked})
            if r['id'] in exclusions:
                kinds.append(exclusions[r['id']])
            if kinds:
                dropped[f'{source}:{",".join(kinds)}'] += 1
            else:
                kept.append(r)
        train[source] = kept
    rows = interleave(train)
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate task ids'
    assert all(r['split'] == 'train' for r in rows)

    def write(path, items):
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    write(out / 'cases.jsonl', rows)
    for split, items in held.items():
        write(out / 'held' / f'{split}.jsonl', items)
    if (a.build.parent / 'assets').exists():
        shutil.copytree(a.build.parent / 'assets', out / 'assets')
    div = diversity(rows)
    div['held'] = {split: diversity(items)['total'] for split, items in held.items()}
    (out / 'diversity.json').write_text(json.dumps(div, indent=1) + '\n')
    verification_summary = {'per_source': per_source, 'custody_dropped': dict(dropped)}
    (out / 'verification.json').write_text(json.dumps({
        'schema': 'natlang.broad-rl-assembly-verification/1', 'cases': len(rows), 'failures': [],
        **verification_summary}, indent=1) + '\n')
    proof = {
        'schema': 'natlang.rl-additional-task-source-qualification/1', 'source_id': source_id,
        'source_cases_sha256': sha_file(out / 'cases.jsonl'), 'qualified_task_rows': len(rows),
        'all_train': True, 'all_source_bound': all(r.get('source_ids') and r.get('source_groups') for r in rows),
        'qualification_kind': ('External dataset items converted to typed natlang root functions; every external row '
                               'replayed through the collector (verifyCases: reference accepted and admitted) and '
                               'rejected on every meaningful wrong answer (other items\' gold, perturbations, wrong '
                               'implementations/calls); alternative gold forms probed for false negatives. The legacy '
                               'slice rows are byte-identical rows of the registered procedural curriculum. No teacher '
                               'generations; no student capability qualification claimed.'),
        'training_admission': ('DGX supervisor under standing owner authorization (LFM3 JOLT+ RL stage); whole student '
                               'episodes scored by the collector outcome oracle, actual communication graphs updated whole.'),
        'current_protection_intersections': {'protected_worlds': []},
        'current_protection_bindings': {'registry': REGISTRY, 'registry_sha256': sha_file(ROOT / REGISTRY),
                                        'protected_sha256': protected.digest,
                                        'sources': [json.loads(item) for item in protected.sources],
                                        'own_held_rows': len(held['dev']) + len(held['confirmation'])},
        'verification': {'path': 'verification.json', 'sha256': sha_file(out / 'verification.json')},
        'task_ids': ids, 'union_task_row_sha256': {r['id']: row_sha(r) for r in rows},
    }
    (out / 'qualification.json').write_text(json.dumps(proof, indent=1) + '\n')
    receipt = {'schema': 'natlang.broad-rl-task-source/1', 'source_id': source_id, 'builder': 'scripts/rl_tasks',
               'cases': {'path': 'cases.jsonl', 'sha256': proof['source_cases_sha256'], 'rows': len(rows)},
               'held': {split: {'path': f'held/{split}.jsonl', 'sha256': sha_file(out / 'held' / f'{split}.jsonl'),
                                'rows': len(items)} for split, items in held.items()},
               'legacy_slice': {'source': str(LEGACY.relative_to(ROOT)), 'cases_sha256': sha_file(LEGACY),
                                'rows': len([r for r in rows if r['family'].startswith('curriculum_')])},
               'excluded_sources': sorted(excluded), 'custody_dropped': dict(dropped),
               'id_exclusions': [{'path': str(path), 'sha256': sha_file(path),
                                  'reason': json.loads(path.read_text()).get('reason', 'held-set-union-group'),
                                  'rows': len(json.loads(path.read_text())['conflicts'])} for path in a.exclude_ids],
               'assets_note': 'SQL databases and DataBench tables referenced by SHA-256 from service sources; set '
                              'NATLANG_RL_TASK_ASSETS to another copy of assets/ when this directory moves.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'train': len(rows), 'held': {k: len(v) for k, v in held.items()}, 'dropped': dict(dropped),
                      'total': div['total']}, indent=1))


if __name__ == '__main__':
    main()
