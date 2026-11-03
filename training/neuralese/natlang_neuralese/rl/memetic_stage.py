"""Memetic meta-learning as a declared recipe stage (kind ``native_memetic_rl``): "improve program P" episodes.

The backbone, running the natural-language supervisor ``applications/memetic-improver`` (policy), studies one program's
support invocations and improves the program by typed, measured changes through the crisp workspace
(``rl.memetic_workspace``). An episode is an ordinary RL task row (family ``memetic_improve``) whose world is that
workspace, so the interleaved RL controller collects it on-policy with the full prompt contract and whole-graph capture,
the curriculum selector seeks outcome spread over it, and JOLT+ updates it with the other tasks. This module is
mechanism: it builds the program pool and its world splits, prepares each program's evidence and trivial baselines,
writes episode specs and task rows, measures repetition, exports distillation rows and decides promotion.

- Programs P are RL task families of the admitted pool (each family is one natlang program template; its tasks are
  invocation worlds). Related worlds (shared source groups or split groups) form one component and stay on one side.
  Near-duplicate worlds are dropped before splitting (inputs plus root text, difflib ratio).
- Meta-training programs get support worlds (the supervisor sees the parent's attempts on them) and query worlds (the
  reward; never shown). Held programs (a seeded fraction of families, never meta-trained) get support, development and
  confirmation worlds for promotion: development first, confirmation once.
- Curriculum: a program is admitted while the parent's success on its support and query worlds lies in
  [min_success, max_success] (headroom and a learnable signal); outside it the program waits for the next parent.
- Baselines per program and parent: (a) no change; (b) the default crisp improvement: the bank's default coefficient
  fit when a bank service is configured, else mechanical exemplar guidance built from a verified successful support
  attempt (span basis: input-specific, target-relevant content). The reward is the committed candidate's query success
  minus a bounded budget cost; the learner centers on max(group mean, best baseline).
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable
from ..eval.task_packet import read_jsonl
from .memetic_workspace import SPEC_SCHEMA, attempt_digest, attempt_steps

STAGE_SCHEMA = 'natlang.memetic-stage/1'
FAMILY = 'memetic_improve'
PACKAGE = Path(__file__).resolve().parents[4] / 'applications' / 'memetic-improver'
DEFAULTS = {'support': 4, 'query': 4, 'dev': 4, 'confirmation': 4, 'samples': 2, 'held_fraction': 0.2,
            'min_success': 0.1, 'max_success': 0.85, 'near_duplicate': 0.9, 'max_worlds_per_family': 40,
            'evaluations': 4, 'fits': 1, 'cost_weight': 0.05, 'episode_seconds': 5400, 'workers': 4,
            'case_seconds': 900, 'max_model_requests': 48, 'max_turns': 40, 'seed': 'memetic-v1'}


def _hash(*parts):
    return hashlib.sha256('\x1f'.join(str(p) for p in parts).encode()).hexdigest()


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
        keys = list(case.get('source_groups') or []) + [((case.get('curriculum') or {}).get('split_group'))]
        for key in keys:
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
    semantics = case.get('semantics') or {}
    root = (semantics.get('files') or {}).get(semantics.get('root') or '', '')
    return json.dumps(semantics.get('inputs'), sort_keys=True, ensure_ascii=False)[:4000] + '\n' + root[:2000]


def dedupe(groups, threshold):
    """Drop a component whose representative world nearly repeats a kept one (the bank pool's near-duplicate rule)."""
    kept, texts = [], []
    for group in groups:
        text = world_text(group[0])
        if any(difflib.SequenceMatcher(None, text, other, autojunk=False).quick_ratio() >= threshold and
               difflib.SequenceMatcher(None, text, other, autojunk=False).ratio() >= threshold for other in texts):
            continue
        kept.append(group)
        texts.append(text)
    return kept


def program_pool(cases, *, settings, protected=None):
    """{family: {'held': bool, 'components': [[case]]}} over train-split, unprotected cases with enough worlds."""
    families = {}
    for case in cases:
        if case.get('split') != 'train' or case.get('family') in (None, FAMILY):
            continue
        if protected is not None and protected(case):
            continue
        families.setdefault(case['family'], []).append(case)
    pool = {}
    for family, rows in sorted(families.items()):
        rows = sorted(rows, key=lambda row: _hash(settings['seed'], row['id']))[:settings['max_worlds_per_family']]
        groups = dedupe(sorted(components(rows), key=lambda g: _hash(settings['seed'], g[0]['id'])),
                        settings['near_duplicate'])
        held = int(_hash(settings['seed'], 'held', family)[:8], 16) / 0xFFFFFFFF < settings['held_fraction']
        need = settings['support'] + (settings['dev'] + settings['confirmation'] if held else settings['query'])
        if len(groups) >= need:
            pool[family] = {'held': held, 'components': groups}
    return pool


def split_program(entry, settings):
    """Support / query (meta-train) or support / dev / confirmation (held) worlds: one world per component, so
    related worlds never straddle two slices."""
    worlds = [group[0] for group in entry['components']]
    names = ('support', 'dev', 'confirmation') if entry['held'] else ('support', 'query')
    out, at = {}, 0
    for name in names:
        out[name] = worlds[at:at + settings[name]]
        at += settings[name]
    return out


def brief(family, cases):
    """The program as the supervisor sees it: the shared root and function names of the family's representative."""
    semantics = cases[0]['semantics']
    files = semantics['files']
    return {'family': family, 'root': semantics['root'],
            'functions': sorted({Path(path).stem for path in files if path.endswith('.nl')})}


def mechanical_guidance(successes, limit=1500):
    """The default crisp improvement without a bank: the action outline of one verified successful support attempt,
    stated as an example for other invocations (deterministic; input-specific and target-relevant)."""
    for world, result in successes:
        steps = attempt_steps(result, limit=12)
        lines = []
        for step in steps:
            what = step.get('code') or step.get('arguments') or ''
            lines.append(f"{len(lines) + 1}. {step['tool']}" + (f": {what}" if what else ''))
        if lines:
            text = ('A verified successful run of this program on another invocation took these actions:\n'
                    + '\n'.join(lines))
            return text[:limit]
    return None


def prepare_program(family, split, *, evaluate, settings, out):
    """Evidence and baselines of one program under the current parent. ``evaluate(cases, guidance, adapter, name=)``
    is the workspace evaluator (rl.memetic_workspace.production_evaluator or a test double)."""
    out.mkdir(parents=True, exist_ok=True)
    target = split.get('query') or split['dev']
    support = evaluate(split['support'], None, None, name='baseline-support')
    attempts = [{'digest': attempt_digest(result, attempt_id=f'a{i + 1}', world=world),
                 'steps': attempt_steps(result)} for i, (world, result) in enumerate(support['results'])]
    none = evaluate(target, None, None, name='baseline-target-none')
    successes = [(world, result) for (world, result), item in zip(support['results'], attempts)
                 if item['digest']['outcome'] == 'task_success']
    default_text = mechanical_guidance(successes)
    default = (dict(evaluate(target, default_text, None, name='baseline-target-default'), kind='mechanical-guidance',
                    text=default_text) if default_text else None)
    headroom = (support['success'] + none['success']) / 2
    return {'family': family, 'support_success': support['success'], 'attempts': attempts,
            'baselines': {'none': {'success': none['success'], 'dir': none.get('dir')},
                          'default': None if default is None else {'success': default['success'], 'kind': default['kind'],
                                                                   'text': default['text'], 'dir': default.get('dir')}},
            'headroom': headroom,
            'admitted': settings['min_success'] <= headroom <= settings['max_success']}


def episode_spec(family, split, prepared, *, settings, out, parent_sha256):
    program_cases = split['support']
    return {'schema': SPEC_SCHEMA, 'parent_sha256': parent_sha256,
            'program': {'id': f'memetic:{family}', 'family': family, 'files': program_cases[0]['semantics']['files'],
                        **brief(family, program_cases)},
            'support': split['support'], 'query': split.get('query') or split['dev'],
            'evidence': {'attempts': prepared['attempts'], 'success': prepared['support_success']},
            'baselines': prepared['baselines'],
            'budget': {'evaluations': settings['evaluations'], 'fits': settings['fits']},
            'eval': {key: settings[key] for key in ('samples', 'workers', 'case_seconds', 'max_model_requests',
                                                    'max_turns')} | {'seed_base': 31000},
            'cost_weight': settings['cost_weight'], 'out': str(out)}


def package_files():
    return {path.relative_to(PACKAGE).as_posix(): path.read_text()
            for path in sorted(PACKAGE.rglob('*.nl'))}


def episode_case(spec_path, spec, *, settings):
    """One "improve program P" RL task row. The world is the workspace; the reward is its graded score."""
    spec_sha = hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
    program = spec['program']
    brief_value = {'family': program['family'], 'root': program['root'], 'functions': program['functions'],
                   'supportAttempts': len(spec['evidence']['attempts']), 'supportSuccess': spec['evidence']['success'],
                   'budget': spec['budget']}
    world_ids = [case['id'] for case in spec['support'] + spec['query']]
    return {'version': 'natlang.program/2', 'id': f"memetic:{program['family']}:{spec_sha[:16]}", 'kind': 'lambda_source',
            'family': FAMILY, 'source': 'memetic-stage', 'split': 'train',
            'source_ids': [f'memetic-episode:{spec_sha}'],
            'source_groups': sorted({g for case in spec['support'] + spec['query'] for g in case.get('source_groups') or []}),
            'source_revisions': [STAGE_SCHEMA],
            'generation': {'generator': STAGE_SCHEMA, 'teacher_generation': False,
                           'method': 'an RL task family program, its support and query worlds, and the parent outcomes '
                                     'on them; the reward is measured on the query worlds by the production collector'},
            'curriculum': {'family': FAMILY, 'program_family': program['family'], 'world_ids': world_ids,
                           'rl': {'family': FAMILY, 'level': 1}},
            'semantics': {'root': 'improve_program.nl', 'files': package_files(), 'inputs': {'program': brief_value},
                          'expected': None, 'oracle': 'exact',
                          'world': {'kind': 'memetic', 'task': str(spec_path), 'spec_sha256': spec_sha,
                                    'case_seconds': settings['episode_seconds']}}}


def repetition(cases):
    """Diversity of an episode batch: distinct program families, mean pairwise similarity of the program sources and
    of the support evidence (a formulaic generator shows up as near-identical rows)."""
    texts = [json.dumps(case['semantics']['inputs'], sort_keys=True) for case in cases]
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))][:2000]
    similarity = (sum(difflib.SequenceMatcher(None, texts[i], texts[j], autojunk=False).quick_ratio()
                      for i, j in pairs) / len(pairs)) if pairs else 0.
    families = [case['curriculum']['program_family'] for case in cases]
    return {'episodes': len(cases), 'families': len(set(families)),
            'max_episodes_per_family': max((families.count(f) for f in set(families)), default=0),
            'mean_brief_similarity': round(similarity, 4)}


def build(*, cases, out, evaluate_for, parent_sha256, settings, protected=None, families=None, log=print):
    """Prepare programs and write episode specs and the ``memetic_improve`` task source. ``evaluate_for(out_dir, family)``
    returns an evaluator bound to that directory and program."""
    settings = {**DEFAULTS, **(settings or {})}
    out = Path(out)
    pool = program_pool(cases, settings=settings, protected=protected)
    chosen = [f for f in (families or sorted(pool)) if f in pool]
    rows, programs = [], []
    for family in chosen:
        entry = pool[family]
        split = split_program(entry, settings)
        directory = out / 'programs' / _hash(family)[:16]
        prepared_path = directory / f'prepared-{parent_sha256[:16]}.json'
        if prepared_path.exists():
            prepared = json.loads(prepared_path.read_text())
        else:
            prepared = prepare_program(family, split, evaluate=evaluate_for(directory / 'baselines', family),
                                       settings=settings, out=directory)
            write_json_durable(prepared_path, prepared)
        programs.append({'family': family, 'held': entry['held'], 'headroom': prepared['headroom'],
                         'admitted': prepared['admitted'], 'support_success': prepared['support_success'],
                         'baselines': {k: (v or {}).get('success') for k, v in prepared['baselines'].items()}})
        log(json.dumps({'event': 'memetic_program', **programs[-1]}))
        if not prepared['admitted']:
            continue
        spec = episode_spec(family, split, prepared, settings=settings, out=directory / 'episodes', parent_sha256=parent_sha256)
        spec_path = directory / f'spec-{parent_sha256[:16]}.json'
        write_json_durable(spec_path, spec)
        row = episode_case(spec_path, spec, settings=settings)
        (rows if not entry['held'] else []).append(row)
        if entry['held']:
            write_json_durable(directory / 'held-episode.json', row)
    source = out / f'memetic-tasks-{parent_sha256[:16]}.jsonl'
    source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    receipt = {'schema': STAGE_SCHEMA, 'parent_sha256': parent_sha256, 'settings': settings,
               'task_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
               'programs': programs, 'repetition': repetition(rows),
               'held_programs': sorted(f for f in chosen if pool[f]['held'])}
    write_json_durable(out / f'stage-{parent_sha256[:16]}.json', receipt)
    return receipt


def validate_stage_parameters(parameters):
    """Recipe kind ``native_memetic_rl`` parameters (train/recipe.py)."""
    if not isinstance(parameters.get('rl_stage'), str) or not parameters['rl_stage']:
        raise ValueError('memetic RL names the interleaved RL stage it rides on (rl_stage)')
    for key, low in (('memetic_every', 1), ('programs_per_round', 1), ('group', 2)):
        if key in parameters and (type(parameters[key]) is not int or parameters[key] < low):
            raise ValueError(f'memetic RL {key} must be an integer >= {low}')
    settings = parameters.get('settings', {})
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError('memetic RL settings are a subset of ' + ', '.join(sorted(DEFAULTS)))
    gain = parameters.get('promotion_min_gain', 0.05)
    if type(gain) not in (int, float) or not 0 <= gain <= 1:
        raise ValueError('memetic RL promotion_min_gain is in [0, 1]')


def receipt_path(controller_out, parent_sha256):
    return Path(controller_out) / 'memetic' / f'stage-{parent_sha256[:16]}.json'


def bind_round(config, declared, state, parent_sha256):
    """The controller config of a due memetic round, or None. A round is due every ``memetic_every`` rounds; it uses
    the memetic task source built for the current parent (``memetic_stage build --out <controller out>/memetic``);
    without one the controller keeps ordinary rounds and records the pending build."""
    parameters = declared['parameters']
    every = parameters.get('memetic_every', 4)
    if (state.get('round', 0) + 1) % every:
        return None
    path = receipt_path(config['out'], parent_sha256)
    memetic = state.setdefault('interleave', {}).setdefault('memetic', {'rounds': [], 'pending': None})
    if not path.exists():
        memetic['pending'] = {'round': state.get('round', 0), 'parent_sha256': parent_sha256,
                              'reason': 'no memetic task source built for this parent'}
        return None
    receipt = json.loads(path.read_text())
    source = Path(receipt['task_source']['path'])
    if (receipt.get('parent_sha256') != parent_sha256 or not source.is_file() or
            hashlib.sha256(source.read_bytes()).hexdigest() != receipt['task_source']['sha256']):
        raise ValueError('memetic task source does not match its receipt or the current parent')
    if not source.read_text().strip():
        memetic['pending'] = {'round': state.get('round', 0), 'parent_sha256': parent_sha256,
                              'reason': 'no program has curriculum headroom under this parent'}
        return None
    memetic['pending'] = None
    rounds = memetic['rounds']
    if not rounds or rounds[-1]['round'] != state.get('round', 0):
        rounds.append({'round': state.get('round', 0), 'stage': declared['id'], 'parent_sha256': parent_sha256,
                       'task_source': receipt['task_source']})
    return {**config, 'cases': str(source), 'rows_per_round': parameters.get('programs_per_round', 2),
            'group': parameters.get('group', 4)}


def promotion(candidate, published, *, min_gain=0.05):
    """Promote a checkpoint for memetic improvement when its held programs' mean development gain over the trivial
    anchor beats the published parent's by ``min_gain`` (confirmation is spent once, afterwards, through
    common.acceptance). ``candidate``/``published``: {family: gain} on the same held programs."""
    shared = sorted(set(candidate) & set(published))
    if not shared:
        return {'promote': False, 'reason': 'no shared held programs'}
    delta = sum(candidate[f] - published[f] for f in shared) / len(shared)
    return {'promote': delta >= min_gain, 'mean_gain_delta': delta, 'programs': len(shared)}


def distillation_rows(rollouts):
    """Episodes worth distilling: positive anchored gain and a committed change set (the supervisor graphs), with
    their actor identity unchanged. Their improved programs' verified query runs are in each episode's query_dir and
    become SFT graphs through rl.memetic_program.guidance_training_graph (guidance producer span resolved)."""
    rows = []
    for rollout in rollouts:
        memetic = (rollout.get('reward') or {}).get('memetic')
        if memetic and memetic['gain'] > 0 and rollout['reward']['quality'] > memetic['anchor']:
            rows.append({'task_id': rollout['task_id'], 'sample': rollout['sample'], 'gain': memetic['gain'],
                         'policy': rollout.get('policy')})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build', help='prepare programs and write the memetic_improve task source')
    b.add_argument('--cases', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--endpoint', required=True)
    b.add_argument('--families', nargs='*')
    b.add_argument('--settings', default='{}')
    args = parser.parse_args(argv)
    from .memetic_program import actor_identity
    from .memetic_workspace import production_evaluator
    from ..common.custody import ProtectedWorlds
    protected_worlds = ProtectedWorlds.load()
    identity = actor_identity(args.endpoint)
    settings = {**DEFAULTS, **json.loads(args.settings)}

    def evaluate_for(directory, family):
        return production_evaluator({'out': str(directory), 'eval': {**{k: settings[k] for k in (
            'samples', 'workers', 'case_seconds', 'max_model_requests', 'max_turns')}, 'seed_base': 31000},
            'program': {'id': f'memetic:{family}'}}, args.endpoint)

    def protected(case):
        return bool(protected_worlds.related({'id': case['id'], 'source_groups': case.get('source_groups') or [],
                                              'source_ids': case.get('source_ids') or []}))
    receipt = build(cases=list(read_jsonl(args.cases)), out=args.out, evaluate_for=evaluate_for,
                    parent_sha256=identity['checkpoint_sha256'], settings=settings, protected=protected,
                    families=args.families)
    print(json.dumps({k: receipt[k] for k in ('task_source', 'repetition', 'held_programs')}))


if __name__ == '__main__':
    main()
