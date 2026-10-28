"""Entrypoint of the joint learned optimizer: every worker resumes from a declared JSON spec.

    python -m natlang_neuralese.tune.learned bank-init --spec SPEC   gradient-subspace basis + source components + keys
    python -m natlang_neuralese.tune.learned targets   --spec SPEC   tuner plans/builds -> joint target records (jsonl)
    python -m natlang_neuralese.tune.learned train     --spec SPEC   resumable outer training (supervised, then policy)
    python -m natlang_neuralese.tune.learned propose   --spec SPEC   apply the trained optimizer: bundle + receipt
    python -m natlang_neuralese.tune.learned ladder    --spec SPEC   ceiling ladder and gaps from measurements
    python -m natlang_neuralese.tune.learned service   --spec SPEC   one adaptation service of the Natlang `adapt` program
    python -m natlang_neuralese.tune.learned serve                   the persistent service: every command and the
                                                                      adaptation services as JSON lines on stdin
    python -m natlang_neuralese.tune.learned collect   --spec SPEC   sampled proposal groups with paired rewards (policy records)
    python -m natlang_neuralese.tune.learned corpus    --spec SPEC   tuner fits on the bank basis + their targets, per program
    python -m natlang_neuralese.tune.learned controls  --spec SPEC   planted-mixture recovery (optimizer vs in-bank fit)
    python -m natlang_neuralese.tune.learned meta-examples --spec SPEC   exact tuning examples for the optimizer's own stages
    python -m natlang_neuralese.tune.learned adopt-bank --spec SPEC  continue training on the latest published bank (after a refit)

Every spec names its exact checkpoint; outputs are written atomically next to the spec's ``out`` path. The crisp
operation baseline (`train` with ``cycles``) alternates label refits after bank publications, supervised updates and
policy updates on recorded proposals; the Natlang `scheduleTraining` policy replaces it through the same spec fields.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import LearnedOptimizerError


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True, default=_jsonable))
    os.replace(temporary, path)


def _jsonable(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


_OPEN = []


def _engine(spec):
    """The exact checkpoint's engine with the operation's durable artifact store (``store``: a directory shared by
    every worker of the operation; one process holds it at a time)."""
    import torch
    from ...serve import load_native_engine
    from ...serve.store import TensorStore
    engine = load_native_engine(spec['checkpoint'], base=spec.get('base'), device=spec.get('device', 'cpu'),
                                dtype=getattr(torch, spec.get('dtype', 'float32')))
    if spec.get('store'):
        engine.store.close()
        engine.store = TensorStore(spec['store'])
    _OPEN.append(engine)
    return engine


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _source_rows(engine, source):
    """A bank source's training rows: lowered graph rows (``rows``) or a tuner plan's training turns (``plan``),
    lowered exactly as the tuner does with reply supervision only."""
    if source.get('plan'):
        from ..build import load_plan
        from ..rows import lower_rows
        _, _, turns, _ = load_plan(source['plan'])
        return lower_rows(engine, turns, prompt_weight=0.)
    return _rows(source['rows'])


def _format(spec):
    """The bank's component-token law from a bank-init spec's ``mixture`` (`mixture.MixtureFormat` fields)."""
    from .mixture import MixtureFormat
    return MixtureFormat(**{key: float(value) for key, value in (spec.get('mixture') or {}).items()})


def bank_init(spec):
    from .bank import new_bank, save_bank, weight_svd_basis
    from .growth import source_component
    from .subspace import pooled_basis, row_loss_weights, row_losses, source_gradients
    from ..targets import module_map
    from ...model.native_adapters import target_parameters
    engine = _engine(spec)
    names, _ = module_map(sorted(target_parameters(engine.shared_model)), classes=spec.get('target_classes'))
    names = sorted(names)
    gradients, descriptors = [], []
    for source in spec['sources']:
        rows = _source_rows(engine, source)
        weights = row_loss_weights(row_losses(engine, rows))
        gradients.append(source_gradients(engine, rows, names, weights=weights))
        descriptors.append({'name': source['name'], 'kind': source['kind'], 'rows': len(rows)})
    basis = pooled_basis(engine.native_identity, gradients, spec['rank'], per_source_rank=spec.get('per_source_rank'),
                         weight_svd=weight_svd_basis(engine, names, spec['rank']), descriptors=descriptors)
    bank = new_bank(engine, basis, _format(spec), quarters=spec.get('quarters', 4))
    components, reports = [], {}
    for source in spec['sources']:
        component, report = source_component(engine, bank, _source_rows(engine, source), source={'name': source['name'], 'kind': source['kind']},
                                             description=source['description'], rank=spec.get('component_rank', 8),
                                             steps=spec.get('component_steps', 20))
        components.append(component)
        reports[source['name']] = report
    bank = bank.with_components(components, reason='bank-init sources')
    manifest = save_bank(engine, bank)
    from ..targets import deploy_quantized
    # Targets a required deploy point re-quantizes (LFM3 NVFP4: w1/w2/w3): folded writes there are measured at that
    # precision before a capability claim (targets.DEPLOY_QUANTIZED).
    _write(spec['out'], {'bank': manifest, 'reports': reports, 'deploy_quantized': deploy_quantized(names)})
    return manifest['snapshot']


def targets(spec):
    """Joint target records from tuner plans and builds (`program.ProgramContext`), appended to ``out`` as they are
    labeled; a rerun skips finished episodes. Spec: ``{checkpoint, store, bank, out, work, fit?, programs: [{plan,
    build?, root?, scopes?, tiers?, subsets?, incremental?, seed?}], max_support?, incremental_max?}``; ``root`` (the
    program's directory) enters each target's lineage, so its proposals can be scored on the program's task roots. ``free-tune`` targets come from the build (its final
    residual and, for fits on the bank basis, every recorded update); ``in-bank-fit`` targets from support subsets of
    the plan's training roots, starting from the base bundle and, when the build sliced the bank basis, from the tuned
    bundle."""
    from .bank import load_bank
    from .program import ProgramContext, read_build
    from .targets import episode_variants, fit_labels, free_tune_targets
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    out, done_path = Path(spec['out']), Path(str(spec['out']) + '.done.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    def emit(key, records):
        with out.open('a') as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, default=_jsonable) + '\n')
        done[key] = [record['id'] for record in records]
        _write(done_path, done)
    work = Path(spec['work'])
    for index, program in enumerate(spec['programs']):
        build = program.get('build')
        tiers = program.get('tiers') or (['free-tune', 'in-bank-fit'] if build else ['in-bank-fit'])
        context = lambda occurrences=None, suffix='', soft=None: ProgramContext(
            engine, bank, program['plan'], out=work / f'program-{index}{suffix}', scopes=program.get('scopes'),
            build=build, occurrences=occurrences, soft=soft)
        full = context()
        name = full.program['id']
        root = {'root': program['root']} if program.get('root') else {}
        if 'free-tune' in tiers and f'{name}:free-tune' not in done:
            emit(f'{name}:free-tune', free_tune_targets(full, fit=spec.get('fit'), incremental=program.get('incremental', True),
                                                       incremental_max=spec.get('incremental_max'),
                                                       lineage={'plan': str(program['plan']), **root}))
        if 'in-bank-fit' not in tiers:
            continue
        tuned = None
        if build:
            state = read_build(build)['state']
            try:
                adapters = {s: a for s, a in state['node_artifacts'].items() if s in full.scopes}
                for adapter in adapters.values():
                    bank.adapter_cores(engine, adapter)
                tuned = {'adapters': adapters, 'states': {**{k: r['state'] for k, r in full.soft.items()}, **state.get('soft_states', {})}}
            except LearnedOptimizerError:
                tuned = None  # a fit on another basis is a behavior target, not a start inside this bank
        roots = sorted({_support_item(turn) for turn in full.turns})
        problem = {'program': name, 'split': 'train', 'support': roots}
        episodes = episode_variants(problem, rng_seed=program.get('seed', 0), subsets=program.get('subsets', 3),
                                    include_empty=False, starts=('base', 'tuned') if tuned else ('base',),
                                    max_size=spec.get('max_support'))
        for number, episode in enumerate(episodes):
            key = f'{name}:in-bank-fit:{number}'
            if key in done:
                continue
            chosen = set(episode['support'])
            occurrences = {_occurrence(turn) for turn in full.turns if _support_item(turn) in chosen}
            pc = context(occurrences, f'-episode-{number}', full.soft)
            incoming_states = tuned['states'] if episode['start'] == 'tuned' else {k: r['state'] for k, r in pc.soft.items()}
            incoming_adapters = tuned['adapters'] if episode['start'] == 'tuned' else {}
            labels, report = fit_labels(pc, incoming_states=incoming_states, incoming_adapters=incoming_adapters, fit=spec.get('fit'))
            pc.persist([*labels['adapters'].values(), *labels['states'].values()])
            emit(key, [pc.request(tier='in-bank-fit', incoming_states=incoming_states, incoming_adapters=incoming_adapters,
                                  labels=labels, lineage={'plan': str(program['plan']), **root, 'build': build, 'episode': episode,
                                                          'fit': report})])
    return sum(len(ids) for ids in done.values())


def _support_item(turn):
    """Support is sampled by root (a turn's whole invocation world stays together); a lowered row is its own item."""
    return turn.get('root') or _occurrence(turn)


def _occurrence(turn):
    from .program import _occurrence as occurrence
    return occurrence(turn)


def _program_root(program):
    """The program's root directory: given, or the one whose `natlang tune` work holds its plan."""
    if program.get('root'):
        return str(Path(program['root']).resolve())
    parts = Path(program['plan']).resolve().parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ('.natlang', 'tune'):
            return str(Path(*parts[:index]))
    return None


def corpus(spec):
    """The optimizer's training corpus from tuner work: for each program's existing tune plan (its admitted teacher
    rows, manifest and student), run the same `tune.build` fit on the bank basis with update boundaries, then label
    its free-tune and in-bank targets. Resumable: a finished fit resumes as complete and labeled episodes are skipped.

    Spec: ``{checkpoint, store, bank, work, out, recipe?, fit?, subsets?, rebase?, programs: [{name, plan, root?, recipe?}]}``;
    ``recipe`` overrides the source plan's recipe (``basis`` is always the bank's basis). ``rebase`` rebinds plans
    prepared for an earlier student to the spec's checkpoint (teacher turns are model-independent and are lowered
    again with this student; the earlier base is recorded as ``rebasedFrom``). A program's ``root`` defaults to the
    directory whose `natlang tune` work holds its plan."""
    from ..build import load_plan, main as build, persist_artifacts
    work = Path(spec['work'])
    bank_manifest = json.loads(Path(spec['bank']).read_text())['bank']
    exported = work / 'bank-artifacts'
    identity = None
    if not exported.is_dir() or spec.get('rebase'):
        engine = _engine(spec)
        identity = dict(engine.native_identity)
        if not exported.is_dir():
            persist_artifacts(engine, [bank_manifest['basis']], exported)
        engine.store.close()
        _OPEN.remove(engine)
    programs = []
    for program in spec['programs']:
        source, manifest, _, _ = load_plan(program['plan'])
        base = Path(program['plan']).parent
        directory = work / program['name']
        directory.mkdir(parents=True, exist_ok=True)
        plan = {key: value for key, value in source.items() if key not in ('parent', 'learned', 'exports')}
        plan['manifest'] = manifest
        if spec.get('rebase') and manifest['base'] != {key: identity[key] for key in manifest['base']}:
            plan['rebasedFrom'] = manifest['base']
            plan['manifest'] = {**manifest, 'base': {key: identity[key] for key in manifest['base']}}
            if 'base' in plan:
                plan['base'] = {key: identity[key] for key in plan['base']}
        plan['rows'] = str((base / source['rows']).resolve())
        plan['artifacts'] = [str(exported.resolve())]
        plan['emitBoundaries'], plan['label'], plan['exports'] = True, f"corpus:{program['name']}", []
        plan['student'] = {**source['student'], 'checkpoint': spec['checkpoint'],
                           'engine': {**source['student'].get('engine', {}), **{k: spec[k] for k in ('base', 'device', 'dtype') if k in spec}}}
        plan['recipe'] = {**source.get('recipe', {}), **spec.get('recipe', {}), **program.get('recipe', {}),
                          'basis': bank_manifest['basis']}
        plan_path = directory / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise LearnedOptimizerError(f'{plan_path} binds another corpus fit; use a new work directory for a changed recipe')
        _write(plan_path, plan)
        if build(['--plan', str(plan_path), '--out', str(directory / 'fit')]) != 0:
            raise LearnedOptimizerError(f'the corpus fit of {program["name"]} failed')
        programs.append({'plan': str(plan_path), 'build': str(directory / 'fit'), 'subsets': spec.get('subsets', 3),
                         **({'root': root} if (root := _program_root(program)) else {}),
                         **({'scopes': program['scopes']} if 'scopes' in program else {})})
    return targets({**spec, 'programs': programs, 'work': str(work / 'targets')})


def controls(spec):
    """Planted controls as telemetry: plant a known mixture on one scope's view, label support rows by the planted
    adapter's exact greedy continuations of the program's own prompts, and compare the trained optimizer's greedy
    proposal and an in-bank fit (the reference) with the planted integers. Spec: ``{checkpoint, store, bank, state,
    plan, work, out, trials?, components?, prompts?, prompt_tokens?, fit?, seed?}``. A failed control is a fault to
    localize, not a verdict on the approach."""
    import torch
    from .bank import load_bank
    from .controls import planted_mixture, planted_rows, recovery
    from .episode import propose as run
    from .fit import ScopeArray, SupportSet, fit_in_bank
    from .mixture import coefficients
    from .program import ProgramContext
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state = json.loads(Path(spec['state']).read_text())
    trainer = OuterTrainer(engine, bank, OuterRecipe.from_json(state['recipe']), [], state=state)
    pc = ProgramContext(engine, bank, spec['plan'], out=Path(spec['work']), scopes=spec.get('scopes'))
    prompts = []
    for row in pc.rows[:spec.get('prompts', 4)]:
        node = row['graph']['nodes'][0]
        first = next((i for i, w in enumerate(node['target_weights']) if w > 0), len(node['token_ids']))
        prompts.append(node['token_ids'][max(0, first - spec.get('prompt_tokens', 48)):first])
    fmt = bank.mixture_format
    lambdas = [scope for scope in pc.scopes if scope.startswith('lambda:')] or pc.scopes
    trials = []
    for trial in range(spec.get('trials', 2)):
        scope = lambdas[trial % len(lambdas)]
        view = pc.views[scope]
        planted = planted_mixture(view, rng_seed=spec.get('seed', 0) * 1009 + trial, components=spec.get('components', 2))
        slots = list(view.slots)
        executed = coefficients(torch.tensor(planted), torch.full((len(bank.group_labels),), fmt.gain_max * fmt.gain_init))
        adapter = bank.compose_adapter(engine, bank.composed_cores(slots, executed), None, {'control': 'planted', 'trial': trial}).id
        binding = next(b['id'] for b in pc.manifest['bindings'] if scope in b['ancestry'])
        rows = planted_rows(engine, adapter, prompts, binding=binding, prefix=f'planted-{trial}')
        context = pc.request()
        context['arrays'] = [item for item in context['arrays'] if item['view']['scope'] == scope]
        _, _, behavior = run(trainer, context, seed=trial)
        proposed = [float.fromhex(x) for x in behavior['arrays'][scope]['distribution']]
        support = SupportSet(engine, pc.manifest, rows)
        fitted = fit_in_bank(engine, bank, support, [ScopeArray(scope, pc.placeholders[scope], slots, list(view.available))],
                             train_states=False, **spec.get('fit', {'steps': 20, 'refine_steps': 5}))
        found = fitted['arrays'][scope]['distribution'].tolist()
        trials.append({'scope': scope, 'planted': planted, 'optimizer': proposed, 'inBankFit': found,
                       'optimizerRecovery': recovery(proposed, planted), 'inBankRecovery': recovery(found, planted)})
    summary = {'schema': 'natlang.lopt.planted-controls/1', 'bank': bank.snapshot, 'writer': trainer.state['writer'],
               'trials': trials, 'meanSupportRecall': {
                   'optimizer': sum(t['optimizerRecovery']['supportRecall'] for t in trials) / max(1, len(trials)),
                   'inBankFit': sum(t['inBankRecovery']['supportRecall'] for t in trials) / max(1, len(trials))}}
    _write(spec['out'], summary)
    return summary['meanSupportRecall']


def meta_examples(spec):
    """Examples for tuning the meta-learner itself (`meta.py`): one counterfactual-best, full-budget episode per
    program plan (each an `apply` plan with ``learned`` settings), written in the tuner's ``tune.examples.jsonl``
    format to ``out``; a rerun skips finished programs. Spec: ``{programs: [{plan, group?}], work, out, steps?,
    refine_steps?, focus?, fit?, learned?}``; ``learned`` (bank, state, store, scopes, ...) is added to plans that do
    not carry it, so corpus plans can be used directly. Tune the optimizer program with them: ``natlang tune <learned-optimizer copy>
    --examples out``."""
    from .apply import Context
    from .meta import episode_examples, write_examples
    out, done_path = Path(spec['out']), Path(str(spec['out']) + '.done.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    for index, program in enumerate(spec['programs']):
        key = str(Path(program['plan']).resolve())
        if key in done:
            continue
        plan_path, work = program['plan'], Path(spec['work']) / f'program-{index}'
        if spec.get('learned'):
            from ..build import load_plan
            source, manifest, _, _ = load_plan(plan_path)
            if 'learned' not in source:
                rows = source['rows'] if Path(source['rows']).is_absolute() else str((Path(plan_path).parent / source['rows']).resolve())
                work.mkdir(parents=True, exist_ok=True)
                plan_path = str(work / 'plan.json')
                _write(plan_path, {**source, 'manifest': manifest, 'rows': rows, 'learned': spec['learned']})
        context = Context(plan_path, work)
        try:
            group = program.get('group') or f"meta:{context.pc.program['id']}"
            examples, receipt = episode_examples(context, steps=spec.get('steps', 4), group=group,
                                                 refine_steps=spec.get('refine_steps', 10), focus=spec.get('focus', True),
                                                 fit=spec.get('fit'))
        finally:
            context.close()
        write_examples(out, examples)
        with Path(str(out) + '.receipts.jsonl').open('a') as stream:
            stream.write(json.dumps(receipt, sort_keys=True, default=_jsonable) + '\n')
        done[key] = len(examples)
        _write(done_path, done)
    return sum(done.values())


def train(spec):
    from .bank import load_bank
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state_path = Path(spec['state'])
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    recipe = OuterRecipe.from_json(spec.get('recipe', {}))
    trainer = OuterTrainer(engine, bank, recipe, _rows(spec['targets']), state=state)
    records = [_rows(path) for path in spec.get('policy_records', [])]
    for _ in range(spec.get('steps', recipe.steps)):
        trainer.step()
        _write(state_path, trainer.state)
    for batch in records:
        trainer.policy_step(batch)
        _write(state_path, trainer.state)
    _write(spec['receipt'], trainer.receipt())
    return trainer.state['step']


def propose(spec):
    from .bank import load_bank
    from .episode import propose as run
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state = json.loads(Path(spec['state']).read_text())
    context = json.loads(Path(spec['context']).read_text())
    trainer = OuterTrainer(engine, bank, OuterRecipe.from_json(state['recipe']), [], state=state)
    bundle, proposal, behavior = run(trainer, context, choose='sample' if spec.get('sample') else None, seed=spec.get('seed', 0))
    _write(spec['out'], {'bundle': bundle, 'proposal': proposal, 'behavior': behavior})
    return bundle['id']


def adopt_bank(spec):
    """After a bank publication: write the published bank as ``out`` (`bank.json` format) and transition the trainer
    state to it (`trainer.adopt_published_bank`). The operation then refits its labels against ``out`` (``corpus``
    into a new work directory) and continues ``train`` with that bank and those targets. Spec: ``{checkpoint, store,
    state, out, recipe?}``; a rerun after a finished adoption is a no-op."""
    from .trainer import OuterRecipe, adopt_published_bank
    state_path, out = Path(spec['state']), Path(spec['out'])
    state = json.loads(state_path.read_text())
    if out.exists() and not state.get('labels_stale') and json.loads(out.read_text())['bank']['snapshot'] == state['bank']:
        return state['bank']
    engine = _engine(spec)
    adopted, published = adopt_published_bank(engine, state, OuterRecipe.from_json(spec.get('recipe', {})))
    _write(out, {'bank': state['published_banks'][-1], 'adopted_at_step': state['step'], 'parent': state['bank']})
    _write(state_path, adopted)
    return published.snapshot


def _outer_trainer(spec):
    from .bank import load_bank
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state = json.loads(Path(spec['state']).read_text())
    return engine, OuterTrainer(engine, bank, OuterRecipe.from_json(state['recipe']), [], state=state)


def _sample_proposals(spec, engine, trainer):
    """Per target context: the optimizer's mixture proposal and one candidate per kept component token of each scope
    (that token executed alone at the scope's gains, the other scopes at their mixture; at most ``perContext`` per
    scope, most probable first), each with its paired shaping reward (mean change of the training-role actions'
    log-likelihood against the incoming bundle): ``[(target, [proposal item])]``; an item keeps its whole bundle under
    ``bundleValue``. The candidates are the finite set the expected-score update scores."""
    from .episode import action_logprobs, candidates, propose as run
    out = []
    for index, target in enumerate(_rows(spec['targets'])):
        before = action_logprobs(engine, _incoming_bundle(engine, target), target['manifest'], target['rows'])
        tokens = max(1, sum(1 for _ in target['rows']))
        def item(choose, candidate, behavior=None):
            bundle, proposal, own = run(trainer, target, choose=choose, seed=spec.get('seed', 0) * 1000003 + index)
            after = action_logprobs(engine, bundle, target['manifest'], target['rows'])
            return {'context': target, 'behavior': behavior or own, 'candidate': candidate, 'bundle': bundle['id'],
                    'proposal': proposal['id'], 'shaping': (sum(after) - sum(before)) / tokens, 'bundleValue': bundle}
        mixture = item(None, None)
        group = [mixture]
        for scope, chosen in candidates(mixture['behavior'], spec.get('perContext')).items():
            group.extend(item({scope: token}, {'scope': scope, 'token': token}, mixture['behavior']) for token in chosen)
        out.append((target, group))
    return out


def _write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(item, sort_keys=True, default=_jsonable) + '\n' for item in items))
    os.replace(temporary, path)


def proposals(spec):
    """Phase one of outer collection with task outcomes. Samples proposals as `collect` does, writes them to ``out``
    (`collect` reads them back with ``proposals``) and writes every proposal, and each target's incoming bundle, as a
    release fragment under ``releases``. For each program it writes ``releases/requests/<program>.json``, the request
    of ``natlang tune ROOT --score-proposals``: the program root (the target lineage's ``root``), and per target its
    support roots (excluded from scoring), its incoming release and its proposals' releases. Targets without a program
    root are recorded without a request: their task component stays absent."""
    from ..soft import text_sha256
    from .apply import bundle_release, write_release
    from .bundle import soft_states_for
    engine, trainer = _outer_trainer(spec)
    sampled = _sample_proposals(spec, engine, trainer)
    releases, requests = Path(spec['releases']), {}
    def release(target, bundle, name, actor):
        directory = releases / name
        if not (directory / 'result.json').exists():
            states = {key: state for binding in target['manifest']['bindings']
                      for key, state in soft_states_for(bundle, binding['id']).items()}
            artifacts, soft_states = bundle_release(engine, target['manifest'], bundle['adapters'], states,
                                                    {write['key']: text_sha256(write['text']) for write in target['writes']})
            write_release(engine, target['manifest'], directory, artifacts, soft_states,
                          {'schema': 'natlang.lopt.proposal-release/1', 'target': target['id'], 'actor': actor,
                           'bundle': bundle.get('id')})
        return str(directory.resolve())
    for target, group in sampled:
        root = target.get('lineage', {}).get('root')
        support = (target.get('lineage', {}).get('episode') or {}).get('support') or sorted({_support_item(row) for row in target['rows']})
        incoming = release(target, _incoming_bundle(engine, target), f"incoming-{target['id'].split(':')[-1][:24]}", 'incoming')
        entry = {'target': target['id'], 'support': support, 'incoming': incoming,
                 'proposals': [{'bundle': item['bundle'], 'release': release(target, item['bundleValue'], item['bundle'].split(':')[-1][:32],
                                                                             'learned-optimizer')} for item in group]}
        if root:
            entry['scaffold'] = target['manifest'].get('digest')
            requests.setdefault(root, {'schema': 'natlang.lopt.score-request/1', 'root': root, 'targets': []})['targets'].append(entry)
    _write_jsonl(spec['out'], [{**{k: v for k, v in item.items() if k != 'bundleValue'}, 'group': index}
                               for index, (_, group) in enumerate(sampled) for item in group])
    paths = {}
    for number, (root, request) in enumerate(sorted(requests.items())):
        path = releases / 'requests' / f'program-{number}.json'
        _write(path, request)
        paths[root] = str(path.resolve())
    return {'proposals': sum(len(group) for _, group in sampled), 'requests': paths}


def collect(spec):
    """Score sampled proposals and record leave-one-out advantages. Proposals are sampled here, or read from
    ``proposals`` (the `proposals` phase's file). The reward is the paired shaping reward plus the task outcome when
    the spec supplies one per proposal bundle (``outcomes``: ``{bundle: {before, after}}`` or the path of such a
    JSON file, as ``natlang tune --score-proposals`` writes); otherwise the task component is absent, never
    invented."""
    from .meta_rl import advantage_snr, leave_one_out_advantages, paired_reward
    if spec.get('proposals'):
        grouped = {}
        for item in _rows(spec['proposals']):
            grouped.setdefault(item.pop('group'), []).append(item)
        sampled = [(None, grouped[index]) for index in sorted(grouped)]
    else:
        engine, trainer = _outer_trainer(spec)
        sampled = _sample_proposals(spec, engine, trainer)
    outcomes = spec.get('outcomes') or {}
    if isinstance(outcomes, str):
        outcomes = json.loads(Path(outcomes).read_text())
    records, groups = [], []
    for _, group in sampled:
        for item in group:
            item.pop('bundleValue', None)
            outcome = outcomes.get(item['bundle'])
            reward = paired_reward(outcome['after'] if outcome else 0., outcome['before'] if outcome else 0.,
                                   shaping=item.pop('shaping'), lambda_shape=float(spec.get('lambda_shape', 1.)))
            reward['taskOutcome'] = outcome
            if outcome is None:
                reward['task'] = None  # no oracle outcome for this proposal: the component is absent, not zero
            item['reward'] = reward
        advantages, info = leave_one_out_advantages([item['reward']['total'] for item in group])
        for item, advantage in zip(group, advantages):
            item['advantage'] = advantage
        groups.append((advantages, info))
        records.extend(group)
    _write_jsonl(spec['out'], records)
    summary = {'records': len(records), 'outcomes': sum(1 for item in records if item['reward']['taskOutcome'] is not None),
               'contrastGroups': sum(1 for _, info in groups if info['contrast']),
               'noContrastGroups': sum(1 for _, info in groups if not info['contrast']), **advantage_snr(groups)}
    if spec.get('summary'):
        _write(spec['summary'], summary)
    return summary


def _incoming_bundle(engine, target):
    """The incoming bundle of a target context: zero composed adapters and the incoming soft states."""
    instructions, skills = {}, {}
    from .bundle import bundle_slot
    for soft, state in target.get('incoming', {}).items():
        field, key = bundle_slot(soft)
        (instructions if field == 'instructions' else skills)[key] = state
    return {'adapters': {**target['placeholders'], **target.get('incoming_adapters', {})}, 'instructions': instructions,
            'skills': skills}


def ladder(spec):
    from .ladder import aggregate, ladder as one
    ladders = [one(item['measurements'], role=item['role']) for item in spec['problems']]
    _write(spec['out'], {'problems': ladders, 'aggregate': aggregate(ladders)})
    return len(ladders)


def service(spec):
    from .apply import service as run
    result = run(spec)
    if spec.get('result'):
        _write(spec['result'], result)
        return spec['result']
    return result


COMMANDS = {'bank-init': bank_init, 'targets': targets, 'train': train, 'propose': propose, 'ladder': ladder, 'proposals': proposals,
            'service': service, 'collect': collect, 'corpus': corpus, 'controls': controls, 'meta-examples': meta_examples,
            'adopt-bank': adopt_bank}


def run_command(command, spec):
    """One command in this process; every engine it opened is closed afterwards."""
    try:
        return COMMANDS[command](spec)
    finally:
        while _OPEN:
            _OPEN.pop().store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=sorted([*COMMANDS, 'serve']))
    parser.add_argument('--spec')
    args = parser.parse_args(argv)
    if args.command == 'serve':
        import sys
        from .apply import serve
        serve(sys.stdin, sys.stdout, commands=run_command)
        return 0
    if not args.spec:
        parser.error('--spec is required')
    try:
        result = run_command(args.command, json.loads(Path(args.spec).read_text()))
    except LearnedOptimizerError as error:
        print(json.dumps({'error': str(error)}))
        return 2
    print(json.dumps({'command': args.command, 'result': result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
