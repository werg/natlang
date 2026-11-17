"""Memetic "improve program P" episodes as GRPO task rows (plans/neuralese/RESTORE_MEMETIC_JOLT_HINDSIGHT_20261117.md
section 3). Crisp and model-free: programs, world splits, episode specs and rows; the policy's evidence and baselines
are measured inside each GRPO group by the workspace (``rl.memetic_workspace.prepare``).

- A program P is one exact natlang source: ``family@sha12`` of its root and ``.nl`` files (a family whose worlds
  paraphrase their root is several programs, and a patch to one would not apply to the others).
- Related worlds (shared source group or split group) form one component; near-duplicate components are dropped (the
  corpus MinHash rule, ``scripts/neuralese_data/dedup.py``). One world per component enters a split, so related worlds
  never straddle support and query.
- Meta-training programs come from the GRPO pool minus the held-v3/v2 closure and the registered memetic held families
  (``training/memetic-held-programs.json``, a protection source). Support worlds are the supervisor's evidence; query
  worlds are its reward and never shown.
- With base pass rates (``--pass-rates``), a program enters only when its support worlds' mean base rate lies in
  [min_success, max_success]: a program the policy always or never solves gives the improver nothing to measure.

    python -m natlang_neuralese.rl.memetic_tasks build --out DIR [--pool cases.jsonl ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex

FAMILY = 'memetic_improve'
STAGE_SCHEMA = 'natlang.memetic-tasks/1'
PACKAGE = Path(__file__).resolve().parents[4] / 'applications' / 'memetic-improver'
# Lean sizes (one group of 4 samples is ~108 nested rollouts, about 13 ordinary groups; design doc section 3).
DEFAULTS = {'support': 2, 'query': 6, 'samples': 2, 'query_samples': 2, 'evaluations': 2, 'min_success': 0.1,
            'max_success': 0.85, 'near_duplicate': 0.9, 'max_worlds_per_program': 40, 'cost_weight': 0.05,
            'episode_seconds': 7200, 'workers': 4, 'case_seconds': 900, 'max_model_requests': 40, 'max_turns': 40,
            'seed': 'memetic-v2'}


def _hash(*parts):
    return hashlib.sha256('\x1f'.join(str(p) for p in parts).encode()).hexdigest()


def program_digest(case):
    """Identity of a world's program: its exact natlang sources and root (inputs and services are the world)."""
    semantics = case.get('semantics') or {}
    files = {path: text for path, text in (semantics.get('files') or {}).items() if path.endswith('.nl')}
    return canonical_json_sha256_hex({'root': semantics.get('root'), 'files': files})


def components(cases):
    """Connected components of related worlds (shared source group or curriculum split group)."""
    parent = list(range(len(cases)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for index, case in enumerate(cases):
        for key in list(case.get('source_groups') or []) + [(case.get('curriculum') or {}).get('split_group')]:
            if key is None:
                continue
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = {}
    for index in range(len(cases)):
        groups.setdefault(find(index), []).append(cases[index])
    return list(groups.values())


def world_text(case):
    """What makes one world of a program differ from another: everything but its .nl sources and the hidden answer."""
    semantics = case.get('semantics') or {}
    files = {path: text for path, text in (semantics.get('files') or {}).items() if not path.endswith('.nl')}
    rest = {key: value for key, value in semantics.items() if key not in ('files', 'expected', 'root')}
    return json.dumps({'world': rest, 'files': files}, sort_keys=True, ensure_ascii=False, default=str)[:400000]


def dedupe(groups, threshold):
    """Drop a component whose representative world nearly repeats a kept one (bottom-k MinHash Jaccard over word
    5-shingles; worlds too short to sketch are compared exactly)."""
    from scripts.neuralese_data.dedup import jaccard_estimate, sketch
    kept, seen = [], []
    for group in groups:
        text = world_text(group[0])
        signature = sketch(text)
        if any((jaccard_estimate(signature, other) >= threshold) if signature and other else text == other_text
               for other, other_text in seen):
            continue
        kept.append(group)
        seen.append((signature, text))
    return kept


def programs(cases, *, settings, excluded=frozenset()):
    """{program key: [one world per component, seeded order]} over train worlds not in ``excluded`` with at least
    ``support + query`` independent worlds."""
    by_program = {}
    for case in cases:
        if case.get('split') != 'train' or case.get('family') in (None, FAMILY) or case.get('id') in excluded:
            continue
        if ((case.get('semantics') or {}).get('world') or {}).get('kind') == 'memetic':
            continue
        by_program.setdefault(f"{case['family']}@{program_digest(case)[:12]}", []).append(case)
    need = settings['support'] + settings['query']
    pool = {}
    for key, rows in sorted(by_program.items()):
        if len(rows) < need:
            continue
        rows = sorted(rows, key=lambda row: _hash(settings['seed'], row['id']))[:settings['max_worlds_per_program']]
        groups = dedupe(sorted(components(rows), key=lambda g: _hash(settings['seed'], g[0]['id'])),
                        settings['near_duplicate'])
        if len(groups) >= need:
            pool[key] = [group[0] for group in groups]
    return pool


def split(worlds, settings):
    return {'support': worlds[:settings['support']],
            'query': worlds[settings['support']:settings['support'] + settings['query']]}


def episode_spec(key, worlds, *, settings):
    """The episode's ``natlang.memetic-episode/1`` spec. ``out``, ``endpoint``, ``runtime`` and ``sampling_config``
    are bound per GRPO group (``grpo.py`` writes the group's copy); the rest is fixed per program."""
    from .memetic_workspace import SPEC_SCHEMA
    semantics = worlds['support'][0]['semantics']
    files = semantics['files']
    return {'schema': SPEC_SCHEMA, 'program': {
                'id': f'memetic:{key}', 'key': key, 'family': key.split('@')[0], 'root': semantics['root'],
                'functions': sorted({Path(path).stem for path in files if path.endswith('.nl')}), 'files': files},
            'support': worlds['support'], 'query': worlds['query'],
            'settings': {name: settings[name] for name in ('min_success', 'max_success', 'query_samples', 'cost_weight')},
            'budget': {'evaluations': settings['evaluations']},
            'eval': {name: settings[name] for name in ('samples', 'workers', 'case_seconds', 'max_model_requests',
                                                       'max_turns')} | {'seed_base': 31000},
            'out': None, 'endpoint': None, 'runtime': None, 'sampling_config': None}


def package_files():
    return {path.relative_to(PACKAGE).as_posix(): path.read_text() for path in sorted(PACKAGE.rglob('*.nl'))}


def episode_case(spec_path, spec, *, settings):
    """One "improve program P" RL task row: the world is the workspace, the reward its graded score."""
    spec_sha = hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
    program = spec['program']
    brief = {'family': program['family'], 'root': program['root'], 'functions': program['functions'],
             'budget': {'evaluations': spec['budget']['evaluations'], 'fits': 0}}
    worlds = spec['support'] + spec['query']
    return {'version': 'natlang.program/2', 'id': f"memetic:{program['key']}:{spec_sha[:16]}", 'kind': 'lambda_source',
            'family': FAMILY, 'source': 'memetic-tasks', 'split': 'train',
            'source_ids': [f'memetic-episode:{spec_sha}'],
            'source_groups': sorted({g for case in worlds for g in case.get('source_groups') or []}),
            'source_revisions': [STAGE_SCHEMA],
            'generation': {'generator': STAGE_SCHEMA, 'teacher_generation': False,
                           'method': 'a GRPO-pool program, its support and query worlds; the reward is the committed '
                                     'change set\'s query success, anchored on the trivial improvements'},
            'curriculum': {'family': FAMILY, 'program_family': program['family'],
                           'world_ids': [case['id'] for case in worlds], 'rl': {'family': FAMILY, 'level': 1}},
            'semantics': {'root': 'improve_program.nl', 'files': package_files(), 'inputs': {'program': brief},
                          'expected': None, 'oracle': 'exact',
                          'world': {'kind': 'memetic', 'task': str(spec_path), 'spec_sha256': spec_sha,
                                    'case_seconds': settings['episode_seconds']}}}


def build(cases, out, *, settings=None, excluded=frozenset(), pass_rates=None, log=print):
    """Write one spec and one row per admitted program; returns the receipt."""
    settings = {**DEFAULTS, **(settings or {})}
    out = Path(out)
    (out / 'specs').mkdir(parents=True, exist_ok=True)
    pool = programs(cases, settings=settings, excluded=excluded)
    rows, skipped = [], {'no_rate': 0, 'out_of_band': 0}
    for key, program_worlds in pool.items():
        worlds = split(program_worlds, settings)
        if pass_rates is not None:
            rates = [pass_rates.get(case['id']) for case in worlds['support']]
            rates = [r['successes'] / r['scored'] for r in rates if r and r.get('scored')]
            if not rates:
                skipped['no_rate'] += 1
            elif not settings['min_success'] <= sum(rates) / len(rates) <= settings['max_success']:
                skipped['out_of_band'] += 1
                continue
        spec = episode_spec(key, worlds, settings=settings)
        path = out / 'specs' / f'{_hash(key)[:16]}.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True) + '\n')
        rows.append(episode_case(path, spec, settings=settings))
    source = out / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    families = sorted({row['curriculum']['program_family'] for row in rows})
    receipt = {'schema': STAGE_SCHEMA, 'settings': settings, 'programs': len(pool), 'rows': len(rows),
               'families': len(families), 'skipped': skipped, 'excluded_ids': len(excluded),
               'cases': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    (out / 'build.json').write_text(json.dumps(receipt, indent=1) + '\n')
    log(json.dumps({key: receipt[key] for key in ('programs', 'rows', 'families', 'skipped')}))
    return receipt


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--pool', action='append', help='task pool cases.jsonl (default: grpo.POOLS)')
    b.add_argument('--held-manifest', help='held manifest whose closure is excluded (default: grpo.HELD)')
    b.add_argument('--pass-rates', help='base pass rates {task_id: {successes, scored}} (default: grpo --pass-rates)')
    b.add_argument('--no-pass-rates', action='store_true')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / 'scripts/neuralese_data'))
    import grpo
    from ..common.custody import ProtectedWorlds
    from ..common.jsonio import read_jsonl
    from .held_eval import exclusion_ids
    cases = [row for pool in (args.pool or grpo.POOLS) for row in read_jsonl(pool)]
    excluded = set(exclusion_ids(args.held_manifest or grpo.HELD, cases))
    protected = ProtectedWorlds.load()
    from ..common.custody import CustodyError
    for case in cases:
        try:
            protected.check(case)
        except CustodyError:
            excluded.add(case['id'])
    rates = None
    if not args.no_pass_rates:
        from ..common.paths import resolve_str
        default = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'exp1-rft', 'base-pass-rates.json')
        rates = json.loads(Path(args.pass_rates or default).read_text())
    build(cases, args.out, excluded=frozenset(excluded), pass_rates=rates)


if __name__ == '__main__':
    main()
