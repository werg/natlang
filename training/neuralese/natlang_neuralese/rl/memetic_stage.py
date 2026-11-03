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
import contextlib
import difflib
import hashlib
import json
import math
import os
import time
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import write_json_durable
from ..eval.task_packet import read_jsonl
from .memetic_workspace import SPEC_SCHEMA, attempt_digest, attempt_steps

STAGE_SCHEMA = 'natlang.memetic-stage/1'
FAMILY = 'memetic_improve'
PACKAGE = Path(__file__).resolve().parents[4] / 'applications' / 'memetic-improver'
DEFAULTS = {'support': 4, 'query': 4, 'max_query': 12, 'query_rollouts': 24, 'expensive_seconds': 300,
            'dev': 4, 'confirmation': 4, 'samples': 2, 'held_fraction': 0.2,
            'min_success': 0.1, 'max_success': 0.85, 'near_duplicate': 0.9, 'max_worlds_per_family': 40,
            'stale_parents': 3, 'evaluations': 4, 'fits': 1, 'cost_weight': 0.05, 'episode_seconds': 5400,
            'workers': 4, 'case_seconds': 900, 'max_model_requests': 48, 'max_turns': 40, 'ablation': True,
            'promotion_every': 4, 'promotion_samples': 2, 'seed': 'memetic-v1'}


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
    """What makes one world of a program differ from another: its inputs and everything in its semantics except the
    program's own .nl sources (identical within a program by construction) and the hidden answer."""
    semantics = case.get('semantics') or {}
    files = {path: text for path, text in (semantics.get('files') or {}).items() if not path.endswith('.nl')}
    rest = {key: value for key, value in semantics.items() if key not in ('files', 'expected', 'root')}
    return json.dumps({'world': rest, 'files': files}, sort_keys=True, ensure_ascii=False, default=str)[:400000]


def dedupe(groups, threshold):
    """Drop a component whose representative world nearly repeats a kept one: the corpus near-duplicate rule
    (scripts/neuralese_data/dedup.py, bottom-k MinHash Jaccard over word 5-shingles); worlds too short to sketch
    are compared exactly."""
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


def program_pool(cases, *, settings, protected=None):
    """{family: {'held': bool, 'components': [[case]]}} over train-split, unprotected cases with enough worlds."""
    families = {}
    for case in cases:
        if case.get('split') != 'train' or case.get('family') in (None, FAMILY):
            continue
        if protected is not None and protected(case):
            continue
        # A program is one exact source: worlds of a family whose .nl files differ are different programs (most
        # families paraphrase their root per world), and a patch to one would not apply to the others.
        families.setdefault(f"{case['family']}@{program_digest(case)[:12]}", []).append(case)
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
    related worlds never straddle two slices. Query reserves up to ``max_query`` worlds; preparation sizes the slice it
    actually uses by the program's measured cost (``prepare_program``)."""
    worlds = [group[0] for group in entry['components']]
    names = ('support', 'dev', 'confirmation') if entry['held'] else ('support', 'query')
    sizes = {**settings, 'query': max(settings['query'], settings['max_query'])}
    out, at = {}, 0
    for name in names:
        out[name] = worlds[at:at + sizes[name]]
        at += sizes[name]
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


def query_plan(target, *, seconds_per_rollout, settings):
    """The query slice and samples per world for a usable reward signal at bounded cost: cheap programs use every
    reserved query world, expensive ones (above ``expensive_seconds`` per rollout) the minimum; samples per world fill
    ``query_rollouts`` (2 to 4). Every candidate, baseline and ablation of the episode runs on exactly these worlds
    with the same seeds, so their differences are paired."""
    worlds = target if seconds_per_rollout <= settings['expensive_seconds'] else target[:settings['query']]
    samples = min(4, max(settings['samples'], -(-settings['query_rollouts'] // max(1, len(worlds)))))
    return worlds, samples


def binomial_noise(success, rollouts):
    """Standard error of a success rate over ``rollouts`` paired rollouts (reported with every anchor)."""
    return round(math.sqrt(max(0., success * (1 - success)) / max(1, rollouts)), 4)


def prepare_program(family, split, *, evaluate, settings, fitter=None, spec=None):
    """Evidence and baselines of one program under the current parent (run once per program and parent inside the
    episode's actor lease, ``ensure_prepared``). ``evaluate(cases, guidance, adapter, *, name, samples=None)`` is the
    workspace evaluator (rl.memetic_workspace.production_evaluator or a test double)."""
    started = time.monotonic()
    support = evaluate(split['support'], None, None, name='baseline-support')
    rollouts = max(1, len(support['results']))
    seconds = support.get('seconds', time.monotonic() - started) / rollouts
    attempts = [{'digest': attempt_digest(result, attempt_id=f'a{i + 1}', world=world),
                 'steps': attempt_steps(result)} for i, (world, result) in enumerate(support['results'])]
    # Admission reads the support evidence only: selecting programs on the query outcome that also anchors the
    # reward would bias the anchor (regression to the mean).
    headroom = support['success']
    admitted = settings['min_success'] <= headroom <= settings['max_success']
    prepared = {'schema': 'natlang.memetic-prepared/1', 'family': family, 'support_success': support['success'],
                'attempts': attempts, 'headroom': headroom, 'admitted': admitted,
                'seconds_per_rollout': round(seconds, 2), 'baselines': {'none': None, 'default': None}}
    if not admitted:
        return prepared
    target, samples = query_plan(split.get('query') or split['dev'], seconds_per_rollout=seconds, settings=settings)
    none = evaluate(target, None, None, name='baseline-query-none', samples=samples)
    successes = [(world, result) for (world, result), item in zip(support['results'], attempts)
                 if item['digest']['outcome'] == 'task_success']
    default_text = mechanical_guidance(successes)
    default = (evaluate(target, [{'function': None, 'text': default_text}], None, name='baseline-query-default',
                        samples=samples) if default_text else None)
    n = len(target) * samples
    bank = None
    if fitter is not None:
        # The default crisp adaptation: the bank's inner fit with default settings on the verified support runs.
        try:
            fit = fitter(spec, {})
            report = evaluate(target, None, fit['adapter'], name='baseline-query-bank', samples=samples)
            bank = {'success': report['success'], 'kind': 'bank-default-fit', 'adapter': fit['adapter'],
                    'support_loss': fit.get('support_loss'), 'dir': report.get('dir'),
                    'noise': binomial_noise(report['success'], n)}
        except ValueError as error:  # no verified support success to fit on, or a refused fit
            bank = {'success': None, 'kind': 'bank-default-fit', 'unavailable': str(error)[:300]}
    return {**prepared, 'query_ids': [case['id'] for case in target], 'query_samples': samples,
            'baselines': {'none': {'success': none['success'], 'dir': none.get('dir'),
                                   'noise': binomial_noise(none['success'], n)},
                          'default': None if default is None else {
                              'success': default['success'], 'kind': 'mechanical-guidance', 'text': default_text,
                              'dir': default.get('dir'), 'noise': binomial_noise(default['success'], n)},
                          **({'bank': bank} if bank is not None else {})}}


@contextlib.contextmanager
def _locked(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_prepared(spec, evaluate, fitter=None):
    """The episode's per-parent preparation, computed once by whichever rollout of the group loads first (the others
    wait on the lock and read it). Also records the program's support measurement for later admission."""
    path = Path(spec['prepared_path'])
    with _locked(path):
        if path.exists():
            return json.loads(path.read_text())
        prepared = prepare_program(spec['program']['family'], {'support': spec['support'], 'query': spec['query']},
                                   evaluate=evaluate, settings=spec['settings'], fitter=fitter, spec=spec)
        prepared['parent_sha256'] = spec['parent_sha256']
        write_json_durable(path, prepared)
        memo = path.parent / 'measurements.json'
        rows = json.loads(memo.read_text()) if memo.exists() else []
        rows.append({'parent_sha256': spec['parent_sha256'], 'support_success': prepared['support_success'],
                     'admitted': prepared['admitted']})
        write_json_durable(memo, rows)
        return prepared


def episode_spec(family, split, *, settings, out, parent_sha256, prepared_path, stage_out):
    program_cases = split['support']
    return {'schema': SPEC_SCHEMA, 'parent_sha256': parent_sha256,
            'program': {'id': f'memetic:{family}', 'family': family.split('@')[0], 'key': family,
                        'files': program_cases[0]['semantics']['files'],
                        **brief(family, program_cases)},
            'support': split['support'], 'query': split.get('query') or split['dev'],
            'prepared_path': str(prepared_path), 'stage_out': str(stage_out),
            # The parent's unchanged support collection (written by preparation; the bank fit reads only these runs).
            'evidence': {'rollouts': str(Path(out) / 'evals' / parent_sha256[:16] / 'baseline-support' / 'collection'
                                         / 'rollouts.jsonl')},
            'settings': {key: settings[key] for key in ('min_success', 'max_success', 'query', 'query_rollouts',
                                                        'expensive_seconds', 'samples', 'ablation')},
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


def _excluded(memo_path, parents, settings):
    """A program whose latest support measurement had no headroom waits ``stale_parents`` parents before it is
    measured again (the parent changes every update, so a hopeless program would otherwise be re-prepared each round)."""
    if not memo_path.exists():
        return False
    rows = json.loads(memo_path.read_text())
    if not rows or rows[-1]['admitted']:
        return False
    since = len(parents) - 1 - max((i for i, p in enumerate(parents) if p == rows[-1]['parent_sha256']), default=-1)
    return since < settings['stale_parents']


def build(*, cases, out, parent_sha256, settings, protected=None, families=None, log=print):
    """Write episode specs and the ``memetic_improve`` task source for one parent. Crisp and model-free (the
    controller runs it when a memetic round is due); the parent's evidence and baselines are prepared lazily by the
    first rollout of each episode inside the actor lease (``ensure_prepared``)."""
    settings = {**DEFAULTS, **(settings or {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parents_path = out / 'parents.json'
    parents = json.loads(parents_path.read_text()) if parents_path.exists() else []
    if parent_sha256 not in parents:
        parents.append(parent_sha256)
        write_json_durable(parents_path, parents)
    pool = program_pool(cases, settings=settings, protected=protected)
    chosen = [f for f in (families or sorted(pool)) if f in pool]
    rows, held_rows, programs = [], [], []
    for family in chosen:
        entry = pool[family]
        split = split_program(entry, settings)
        directory = out / 'programs' / _hash(family)[:16]
        directory.mkdir(parents=True, exist_ok=True)
        excluded = _excluded(directory / 'measurements.json', parents, settings)
        programs.append({'family': family, 'held': entry['held'], 'excluded_without_headroom': excluded,
                         'worlds': {name: len(value) for name, value in split.items()}})
        log(json.dumps({'event': 'memetic_program', **programs[-1]}))
        if excluded:
            continue
        spec = episode_spec(family, split, settings=settings, out=directory / 'episodes' / parent_sha256[:16],
                            parent_sha256=parent_sha256, stage_out=out,
                            prepared_path=directory / f'prepared-{parent_sha256[:16]}.json')
        spec_path = directory / f'spec-{parent_sha256[:16]}.json'
        write_json_durable(spec_path, spec)
        (held_rows if entry['held'] else rows).append(episode_case(spec_path, spec, settings=settings))
    source = out / f'memetic-tasks-{parent_sha256[:16]}.jsonl'
    source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    held = out / f'memetic-held-{parent_sha256[:16]}.jsonl'
    held.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in held_rows))
    receipt = {'schema': STAGE_SCHEMA, 'parent_sha256': parent_sha256, 'settings': settings,
               'task_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
               'held_source': {'path': str(held), 'sha256': hashlib.sha256(held.read_bytes()).hexdigest()},
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


def protected_filter():
    """The complete current protection union (held manifests and related worlds): such worlds never enter a program."""
    from ..common.custody import ProtectedWorlds
    protected_worlds = ProtectedWorlds.load()
    return lambda case: bool(protected_worlds.related({'id': case['id'], 'source_groups': case.get('source_groups') or [],
                                                       'source_ids': case.get('source_ids') or []}))


def bind_round(config, declared, state, parent_sha256, *, protected=None):
    """The controller config of a due memetic round, or None. A round is due every ``memetic_every`` rounds. The
    controller builds the parent's memetic task source itself (crisp, model-free) from the declared program source;
    the per-parent evidence and baselines are prepared inside the round's actor lease by the episodes."""
    parameters = declared['parameters']
    every = parameters.get('memetic_every', 4)
    if (state.get('round', 0) + 1) % every:
        return None
    path = receipt_path(config['out'], parent_sha256)
    memetic = state.setdefault('interleave', {}).setdefault('memetic', {'pending': None})
    settings = {**DEFAULTS, **parameters.get('settings', {})}
    program_source = declared['program_source']
    if not path.exists():
        source_path = Path(program_source['path'])
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != program_source['sha256']:
            raise ValueError('memetic program source changed after its recipe binding')
        build(cases=list(read_jsonl(source_path)), out=path.parent, parent_sha256=parent_sha256, settings=settings,
              protected=protected if protected is not None else protected_filter())
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
    index = state.get('round', 0) // every  # this memetic round's ordinal (stable across repeated binds)
    rows, group = parameters.get('programs_per_round', 2), parameters.get('group', 4)
    bound = {**config, 'cases': str(source), 'rows_per_round': rows, 'group': group,
             'memetic_round': {'stage_out': str(path.parent), 'parent_sha256': parent_sha256,
                               'receipt': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                               'promotion_due': index % settings['promotion_every'] == 0,
                               'promotion_min_gain': parameters.get('promotion_min_gain', 0.05),
                               'settings': settings}}
    if config.get('task_curriculum') is not None:
        # The selector's budget floor (16 tasks x 8 samples) is for ordinary rounds; an improvement episode runs its
        # program dozens of times, so a memetic round keeps its declared size.
        bound['task_curriculum'] = {**config['task_curriculum'], 'min_rows_per_round': rows, 'min_group': group}
    return bound


def prepare_round(cases_path, *, endpoint, log=print):
    """Inside the memetic round's actor lease, before collection: prepare each selected program's evidence and
    baselines for this parent, so the episodes' own time budget is spent on improving, not on measuring the parent."""
    from .memetic_workspace import production_evaluator
    fitter = None
    if os.environ.get('NATLANG_BANK_SERVICE'):
        from .memetic_fit import bank_fitter
        fitter = lambda episode, fit_spec: bank_fitter(episode, fit_spec, endpoint=os.environ['NATLANG_BANK_SERVICE'])
    prepared = []
    for row in read_jsonl(Path(cases_path)):
        world = (row.get('semantics') or {}).get('world') or {}
        if world.get('kind') != 'memetic':
            continue
        spec = json.loads(Path(world['task']).read_text())
        started = time.monotonic()
        value = ensure_prepared(spec, production_evaluator(spec, endpoint), fitter)
        prepared.append({'program': spec['program']['key'], 'admitted': value['admitted'],
                         'support_success': value['support_success'], 'seconds': round(time.monotonic() - started, 1)})
        log(json.dumps({'event': 'memetic_prepared', **prepared[-1]}))
    return prepared


def after_collection(round_info, *, endpoint, round_out, collect=None, log=print):
    """Inside the memetic round's actor lease, after its collection: export distillation candidates, then run the
    held-program promotion evaluation when due. Failures are recorded, never silently dropped, and do not discard the
    round's collected learning."""
    out = Path(round_info['stage_out'])
    record = {'schema': 'natlang.memetic-after-collection/1', 'round_out': str(round_out),
              'parent_sha256': round_info['parent_sha256']}
    try:
        record['distillation'] = distill(Path(round_out), out, endpoint=endpoint, parent_sha256=round_info['parent_sha256'])
    except Exception as error:  # noqa: BLE001 - recorded for supervision
        record['distillation_error'] = f'{type(error).__name__}: {error}'[:2000]
    if round_info.get('promotion_due'):
        try:
            record['promotion'] = promote(out, round_info, endpoint=endpoint, collect=collect)
        except Exception as error:  # noqa: BLE001 - recorded for supervision
            record['promotion_error'] = f'{type(error).__name__}: {error}'[:2000]
    Path(round_out).mkdir(parents=True, exist_ok=True)
    write_json_durable(Path(round_out) / 'memetic-after-collection.json', record)
    log(json.dumps({'event': 'memetic_after_collection', **{k: v for k, v in record.items() if k != 'distillation'}}))
    return record


def held_gains(rollouts, families):
    """{program family: mean held-development gain over the anchor} from scored held episodes."""
    gains = {}
    for row in rollouts:
        memetic = (row.get('reward') or {}).get('memetic')
        if memetic is not None and row.get('task_id') in families:
            gains.setdefault(families[row['task_id']], []).append(memetic['gain'])
    return {family: sum(values) / len(values) for family, values in gains.items()}


def promote(out, round_info, *, endpoint, collect=None):
    """Held programs (never meta-trained) improved by the current parent's supervisor, scored on their development
    worlds; compared with the previous evaluated parent on the same programs (paired). Confirmation worlds are spent
    once, later, through common.acceptance."""
    parent = round_info['parent_sha256']
    receipt = json.loads((out / f'stage-{parent[:16]}.json').read_text())
    held = Path(receipt['held_source']['path'])
    rows = list(read_jsonl(held)) if held.exists() else []
    if not rows:
        return {'evaluated': False, 'reason': 'no held program has headroom under this parent'}
    if collect is None:
        from .production import collect
    from .memetic_program import actor_identity
    identity = actor_identity(endpoint)
    if identity['checkpoint_sha256'] != parent:
        raise ValueError('promotion actor serves another checkpoint than the memetic parent')
    settings = round_info['settings']
    target = out / 'promotion' / parent[:16]
    report = collect(cases=held, out=target / 'collection', endpoints=[endpoint], policy_id=parent, rows=len(rows),
                     group=settings['promotion_samples'], workers=settings['workers'], root_mode='student',
                     seed_base=41000, case_seconds=settings['episode_seconds'],
                     max_model_requests=4 * settings['max_model_requests'], max_turns=settings['max_turns'],
                     prompt_rendering='full')
    families = {row['id']: row['curriculum']['program_family'] for row in rows}
    gains = held_gains(read_jsonl(target / 'collection' / 'rollouts.jsonl'), families)
    history_path = out / 'promotion' / 'history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    previous = next((item for item in reversed(history) if item['parent_sha256'] != parent and item['gains']), None)
    decision = (promotion(gains, previous['gains'], min_gain=round_info.get('promotion_min_gain', 0.05))
                if previous else {'promote': None, 'reason': 'first evaluated parent; it is the reference'})
    entry = {'parent_sha256': parent, 'gains': gains, 'decision': decision,
             'compared_with': previous['parent_sha256'] if previous else None,
             'collection': {'scored': report.get('scored'), 'rollouts': report.get('rollouts')}}
    history.append(entry)
    write_json_durable(history_path, history)
    return entry


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
                         'policy': rollout.get('policy'), 'query_dir': memetic.get('query_dir'),
                         'committed': memetic.get('committed')})
    return rows


def distill(round_out, out, *, endpoint, parent_sha256, export=None):
    """Distillation candidates of one memetic round: (1) supervisor episodes that beat their anchor (already JOLT+
    trained whole-graph in this round; listed for RFT/SFT with their exact actor); (2) the improved programs' verified
    successful query runs lowered to producer/consumer SFT graphs with their exact guidance import
    (rl.memetic_program.export_training_graphs). Writes ``distill/<parent>/manifest.json``; admission into the SFT
    source goes through the ordinary source-append mechanism."""
    rollouts_path = round_out / 'rollouts.jsonl'
    rows = distillation_rows(read_jsonl(rollouts_path)) if rollouts_path.exists() else []
    target = out / 'distill' / parent_sha256[:16]
    graphs, problems = [], []
    if export is None:
        from .memetic_program import export_training_graphs, actor_identity
        identity = actor_identity(endpoint)

        def export(query_dir, destination, written):
            return export_training_graphs(Path(query_dir) / 'collection', destination,
                                          program_id=written['program_id'], confirmation_ids=set(),
                                          written=written['written'], actor=identity, server=endpoint)
    seen = set()
    for row in rows:
        query_dir = row.get('query_dir')
        written_path = Path(query_dir) / 'guidance-written.json' if query_dir else None
        if not query_dir or query_dir in seen:
            continue
        seen.add(query_dir)
        if not written_path.exists():
            problems.append({'query_dir': query_dir, 'reason': 'no single program-wide guidance state to import'})
            continue
        try:
            graphs.append({'query_dir': query_dir, **export(query_dir, target / hashlib.sha256(query_dir.encode()).hexdigest()[:16],
                                                            json.loads(written_path.read_text()))})
        except ValueError as error:
            problems.append({'query_dir': query_dir, 'reason': str(error)[:500]})
    manifest = {'schema': 'natlang.memetic-distillation/1', 'parent_sha256': parent_sha256,
                'round_out': str(round_out), 'supervisor_episodes': rows, 'guidance_graphs': graphs,
                'problems': problems}
    if rows or graphs or problems:
        target.mkdir(parents=True, exist_ok=True)
        write_json_durable(target / 'manifest.json', manifest)
    return {'supervisor_episodes': len(rows), 'guidance_graph_files': len(graphs), 'problems': len(problems)}


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
    settings = {**DEFAULTS, **json.loads(args.settings)}
    receipt = build(cases=list(read_jsonl(args.cases)), out=args.out,
                    parent_sha256=actor_identity(args.endpoint)['checkpoint_sha256'], settings=settings,
                    protected=protected_filter(), families=args.families)
    print(json.dumps({k: receipt[k] for k in ('task_source', 'held_source', 'repetition', 'held_programs')}))


if __name__ == '__main__':
    main()
