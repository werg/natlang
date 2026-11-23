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
- Every program enters by default (owner 2026-11-18): the group's own preparation skips only a group whose unguided
  support samples all pass (``memetic_workspace.prepare``). With ``--pass-rates`` (repeatable, summed) a build keeps
  only programs whose support worlds all have a measured base rate with mean in [min_success, max_success];
  ``measure`` lists the support worlds still lacking one, as cases for ``scripts/neuralese_data/exp1_collect.py``.

    python -m natlang_neuralese.rl.memetic_tasks measure --out DIR/cases.jsonl [--programs N] [--pass-rates R ...]
    python -m natlang_neuralese.rl.memetic_tasks build --out DIR [--pool cases.jsonl ...] [--pass-rates R ...] [--set NAME=VALUE]
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
# 'case_seconds' and 'workers' are the inline layout's; the default staged layout derives them from the episode's wall
# clock when grpo binds a group (memetic_workspace.bind_layout).
DEFAULTS = {'support': 4, 'query': 6, 'samples': 2, 'query_samples': 2, 'evaluations': 2, 'min_success': 0.1,
            'max_success': 0.85, 'min_rate_samples': 2, 'near_duplicate': 0.9, 'max_worlds_per_program': 40, 'cost_weight': 0.05,
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


def support_rate(support, pass_rates, settings):
    """Mean base pass rate of the support worlds, or None while any of them has fewer than ``min_rate_samples``
    scored episodes."""
    rates = []
    for case in support:
        rate = pass_rates.get(case['id']) or {}
        if (rate.get('scored') or 0) < settings['min_rate_samples']:
            return None
        rates.append(rate['successes'] / rate['scored'])
    return sum(rates) / len(rates)


def merge_rates(values):
    """Sum several ``{task_id: {successes, scored}}`` files' counts per task."""
    out = {}
    for value in values:
        for task_id, rate in value.items():
            entry = out.setdefault(task_id, {'successes': 0, 'scored': 0})
            entry['successes'] += rate.get('successes') or 0
            entry['scored'] += rate.get('scored') or 0
    return out


def unmeasured(cases, *, settings=None, excluded=frozenset(), pass_rates=None, programs_limit=None):
    """Support worlds still lacking a base rate, for up to ``programs_limit`` programs taken one family at a time
    (seeded), as task rows for a base-policy collection. Programs whose measured worlds already put them out of band
    are skipped."""
    settings = {**DEFAULTS, **(settings or {})}
    pass_rates = pass_rates or {}
    by_family = {}
    for key, program_worlds in programs(cases, settings=settings, excluded=excluded).items():
        support = split(program_worlds, settings)['support']
        if support_rate(support, pass_rates, settings) is not None:
            continue
        by_family.setdefault(key.split('@')[0], []).append(support)
    chosen = []
    families = sorted(by_family, key=lambda family: _hash(settings['seed'], family))
    while families and (programs_limit is None or len(chosen) < programs_limit):
        for family in list(families):
            if programs_limit is not None and len(chosen) >= programs_limit:
                break
            chosen.append(by_family[family].pop(0))
            if not by_family[family]:
                families.remove(family)
    need = lambda case: ((pass_rates.get(case['id']) or {}).get('scored') or 0) < settings['min_rate_samples']
    return [case for support in chosen for case in support if need(case)]


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
            rate = support_rate(worlds['support'], pass_rates, settings)
            if rate is None:
                skipped['no_rate'] += 1
                continue
            if not settings['min_success'] <= rate <= settings['max_success']:
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
    b.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                   help='override a DEFAULTS setting (JSON value), e.g. --set query=2 --set evaluations=1 for a smoke')
    m = sub.add_parser('measure', help='support worlds lacking a base rate, as cases for exp1_collect.py')
    m.add_argument('--out', required=True, help='cases.jsonl to write')
    m.add_argument('--programs', type=int, help='at most this many programs (one family at a time)')
    for p in (b, m):
        p.add_argument('--pool', action='append', help='task pool cases.jsonl (default: the registry role grpo-pool)')
        p.add_argument('--held-manifest', help='held manifest whose closure is excluded (default: the registry role held-eval)')
        p.add_argument('--pass-rates', action='append',
                       help='base pass rates {task_id: {successes, scored}}, repeatable and summed (build: none, every '
                            'program enters; measure: exp1-rft)')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / 'scripts/neuralese_data'))
    import grpo
    from ..common.custody import CustodyError, ProtectedWorlds
    from ..common.jsonio import read_jsonl
    from ..common.paths import resolve_str
    from .held_eval import training_exclusion_ids
    cases = [row for pool in (args.pool or grpo.default_pools()) for row in read_jsonl(pool)]
    excluded = set(training_exclusion_ids(cases, args.held_manifest or grpo.default_held()))
    protected = ProtectedWorlds.load()
    for case in cases:
        try:
            protected.check(case)
        except CustodyError:
            excluded.add(case['id'])
    paths = args.pass_rates or ([resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'exp1-rft', 'base-pass-rates.json')]
                                if args.command == 'measure' else [])
    rates = merge_rates(json.loads(Path(path).read_text()) for path in paths) if paths else None
    if args.command == 'measure':
        rows = unmeasured(cases, excluded=frozenset(excluded), pass_rates=rates, programs_limit=args.programs)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        print(json.dumps({'worlds': len(rows), 'out': args.out}))
        return
    settings = {}
    for item in args.set:
        name, _, value = item.partition('=')
        if name not in DEFAULTS:
            raise SystemExit(f'--set {name}: not a setting ({", ".join(DEFAULTS)})')
        settings[name] = json.loads(value)
    build(cases, args.out, settings=settings, excluded=frozenset(excluded), pass_rates=rates)
    from ..common import data_registry
    data_registry.record_dataset(
        data_registry.dataset_id(args.out, 'memetic-tasks'), args.out, kind='rl-tasks',
        include=['cases.jsonl', 'build.json', 'specs/*.json'],
        format='memetic "improve program P" rows (natlang.memetic-tasks/1): cases.jsonl + build.json + specs/',
        inputs=[*(args.pool or grpo.default_pools()), args.held_manifest or grpo.default_held(), *paths],
        license='derived from the grpo task pools (each keeps its license provenance)')


if __name__ == '__main__':
    main()
