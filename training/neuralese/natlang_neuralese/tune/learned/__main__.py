"""Entrypoint of the joint learned optimizer: every worker resumes from a declared JSON spec.

    python -m natlang_neuralese.tune.learned bank-init --spec SPEC   gradient-subspace basis + source components + keys
    python -m natlang_neuralese.tune.learned targets   --spec SPEC   tuner plans/builds -> joint target records (jsonl)
    python -m natlang_neuralese.tune.learned train     --spec SPEC   resumable outer training (supervised, then policy)
    python -m natlang_neuralese.tune.learned propose   --spec SPEC   apply the trained optimizer: bundle + receipt
    python -m natlang_neuralese.tune.learned ladder    --spec SPEC   ceiling ladder and gaps from measurements
    python -m natlang_neuralese.tune.learned service   --spec SPEC   one adaptation service of the Natlang `adapt` program
    python -m natlang_neuralese.tune.learned collect   --spec SPEC   sampled proposal groups with paired rewards (policy records)
    python -m natlang_neuralese.tune.learned corpus    --spec SPEC   tuner fits on the bank basis + their targets, per program
    python -m natlang_neuralese.tune.learned controls  --spec SPEC   planted-mixture recovery (optimizer vs in-bank fit)
    python -m natlang_neuralese.tune.learned meta-examples --spec SPEC   exact tuning examples for the optimizer's own stages

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
    engine = load_native_engine(spec['checkpoint'], device=spec.get('device', 'cpu'),
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


def _grid(engine, spec):
    from .numbers import GridTokens
    if 'grid_tokens' in spec:
        return GridTokens(**{k: tuple(v) if k == 'digits' else v for k, v in spec['grid_tokens'].items()})
    return GridTokens.from_encoder(lambda text: engine.tokenizer(text, add_special_tokens=False)['input_ids'])


def bank_init(spec):
    from .bank import new_bank, save_bank, weight_svd_basis
    from .growth import source_component
    from .numbers import NumberFormat
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
    bank = new_bank(engine, basis, NumberFormat(float(spec.get('step', .05))), quarters=spec.get('quarters', 4))
    components, reports = [], {}
    for source in spec['sources']:
        component, report = source_component(engine, bank, _source_rows(engine, source), source={'name': source['name'], 'kind': source['kind']},
                                             description=source['description'], rank=spec.get('component_rank', 8),
                                             steps=spec.get('component_steps', 20))
        components.append(component)
        reports[source['name']] = report
    bank = bank.with_components(components, reason='bank-init sources')
    manifest = save_bank(engine, bank)
    _write(spec['out'], {'bank': manifest, 'reports': reports})
    return manifest['snapshot']


def targets(spec):
    """Joint target records from tuner plans and builds (`program.ProgramContext`), appended to ``out`` as they are
    labeled; a rerun skips finished episodes. Spec: ``{checkpoint, store, bank, out, work, fit?, programs: [{plan,
    build?, scopes?, tiers?, subsets?, incremental?, seed?}], max_support?, incremental_max?}``. ``free-tune`` targets come from the build (its final
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
        if 'free-tune' in tiers and f'{name}:free-tune' not in done:
            emit(f'{name}:free-tune', free_tune_targets(full, fit=spec.get('fit'), incremental=program.get('incremental', True),
                                                       incremental_max=spec.get('incremental_max'),
                                                       lineage={'plan': str(program['plan'])}))
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
                                  labels=labels, lineage={'plan': str(program['plan']), 'build': build, 'episode': episode,
                                                          'fit': report})])
    return sum(len(ids) for ids in done.values())


def _support_item(turn):
    """Support is sampled by root (a turn's whole invocation world stays together); a lowered row is its own item."""
    return turn.get('root') or _occurrence(turn)


def _occurrence(turn):
    from .program import _occurrence as occurrence
    return occurrence(turn)


def corpus(spec):
    """The optimizer's training corpus from tuner work: for each program's existing tune plan (its admitted teacher
    rows, manifest and student), run the same `tune.build` fit on the bank basis with update boundaries, then label
    its free-tune and in-bank targets. Resumable: a finished fit resumes as complete and labeled episodes are skipped.

    Spec: ``{checkpoint, store, bank, work, out, recipe?, fit?, subsets?, rebase?, programs: [{name, plan, recipe?}]}``;
    ``recipe`` overrides the source plan's recipe (``basis`` is always the bank's basis). ``rebase`` rebinds plans
    prepared for an earlier student to the spec's checkpoint (teacher turns are model-independent and are lowered
    again with this student; the earlier base is recorded as ``rebasedFrom``)."""
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
                           'engine': {**source['student'].get('engine', {}), **{k: spec[k] for k in ('device', 'dtype') if k in spec}}}
        plan['recipe'] = {**source.get('recipe', {}), **spec.get('recipe', {}), **program.get('recipe', {}),
                          'basis': bank_manifest['basis']}
        plan_path = directory / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise LearnedOptimizerError(f'{plan_path} binds another corpus fit; use a new work directory for a changed recipe')
        _write(plan_path, plan)
        if build(['--plan', str(plan_path), '--out', str(directory / 'fit')]) != 0:
            raise LearnedOptimizerError(f'the corpus fit of {program["name"]} failed')
        programs.append({'plan': str(plan_path), 'build': str(directory / 'fit'), 'subsets': spec.get('subsets', 3),
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
    from .head import executed, shares_from_norms
    from .program import ProgramContext
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state = json.loads(Path(spec['state']).read_text())
    trainer = OuterTrainer(engine, bank, _grid(engine, spec), OuterRecipe.from_json(state['recipe']), [], state=state)
    pc = ProgramContext(engine, bank, spec['plan'], out=Path(spec['work']), scopes=spec.get('scopes'))
    prompts = []
    for row in pc.rows[:spec.get('prompts', 4)]:
        node = row['graph']['nodes'][0]
        first = next((i for i, w in enumerate(node['target_weights']) if w > 0), len(node['token_ids']))
        prompts.append(node['token_ids'][max(0, first - spec.get('prompt_tokens', 48)):first])
    shares = shares_from_norms(bank.group_norms())
    lambdas = [scope for scope in pc.scopes if scope.startswith('lambda:')] or pc.scopes
    trials = []
    for trial in range(spec.get('trials', 2)):
        scope = lambdas[trial % len(lambdas)]
        view = pc.views[scope]
        planted = planted_mixture(view, rng_seed=spec.get('seed', 0) * 1009 + trial, components=spec.get('components', 2))
        slots = list(view.slots)
        coefficients = executed(planted, torch.zeros(len(slots)), torch.zeros(len(slots), len(bank.group_labels)), shares[slots],
                                step=bank.number_format.step, beta=.5, head=False)
        adapter = bank.compose_adapter(engine, bank.composed_cores(slots, coefficients), None, {'control': 'planted', 'trial': trial}).id
        binding = next(b['id'] for b in pc.manifest['bindings'] if scope in b['ancestry'])
        rows = planted_rows(engine, adapter, prompts, binding=binding, prefix=f'planted-{trial}')
        context = pc.request()
        context['arrays'] = [item for item in context['arrays'] if item['view']['scope'] == scope]
        _, _, behavior = run(trainer, context, sample=False, seed=trial)
        proposed = behavior['arrays'][scope]['integers']
        support = SupportSet(engine, pc.manifest, rows)
        fitted = fit_in_bank(engine, bank, support, [ScopeArray(scope, pc.placeholders[scope], slots, list(view.available))],
                             train_states=False, **spec.get('fit', {'steps': 20, 'refine_steps': 5}))
        trials.append({'scope': scope, 'planted': planted, 'optimizer': proposed, 'inBankFit': fitted['arrays'][scope]['integers'],
                       'optimizerRecovery': recovery(proposed, planted),
                       'inBankRecovery': recovery(fitted['arrays'][scope]['integers'], planted)})
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
    trainer = OuterTrainer(engine, bank, _grid(engine, spec), recipe, _rows(spec['targets']), state=state)
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
    trainer = OuterTrainer(engine, bank, _grid(engine, spec), OuterRecipe.from_json(state['recipe']), [], state=state)
    bundle, proposal, behavior = run(trainer, context, sample=spec.get('sample', False), seed=spec.get('seed', 0),
                                     temperature=spec.get('temperature'))
    _write(spec['out'], {'bundle': bundle, 'proposal': proposal, 'behavior': behavior})
    return bundle['id']


def collect(spec):
    """Sample ``perContext`` proposals per target context, score each by the paired shaping reward (mean change of
    the training-role actions' log-likelihood against the incoming bundle) and record leave-one-out advantages. Oracle
    task outcomes are recorded when the spec supplies them per proposal (``outcomes``); otherwise the task component
    is absent, never invented."""
    from .bank import load_bank
    from .episode import action_logprobs, propose as run
    from .meta_rl import advantage_snr, leave_one_out_advantages, paired_reward
    from .trainer import OuterRecipe, OuterTrainer
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    state = json.loads(Path(spec['state']).read_text())
    trainer = OuterTrainer(engine, bank, _grid(engine, spec), OuterRecipe.from_json(state['recipe']), [], state=state)
    records, groups = [], []
    for index, target in enumerate(_rows(spec['targets'])):
        before = action_logprobs(engine, _incoming_bundle(engine, target), target['manifest'], target['rows'])
        group = []
        for sample in range(spec.get('perContext', 4)):
            bundle, proposal, behavior = run(trainer, target, sample=True, seed=spec.get('seed', 0) * 1000003 + index * 101 + sample)
            after = action_logprobs(engine, bundle, target['manifest'], target['rows'])
            tokens = max(1, sum(1 for _ in target['rows']))
            shaping = (sum(after) - sum(before)) / tokens
            outcome = (spec.get('outcomes') or {}).get(bundle['id'])
            reward = paired_reward(outcome['after'] if outcome else 0., outcome['before'] if outcome else 0.,
                                   shaping=shaping, lambda_shape=float(spec.get('lambda_shape', 1.)))
            reward['taskOutcome'] = outcome
            if outcome is None:
                reward['task'] = None  # no oracle outcome for this proposal: the component is absent, not zero
            group.append({'context': target, 'behavior': behavior, 'bundle': bundle['id'], 'proposal': proposal['id'],
                          'old_executor_logprob': sum(after), 'reward': reward})
        advantages, info = leave_one_out_advantages([item['reward']['total'] for item in group])
        for item, advantage in zip(group, advantages):
            item['advantage'] = advantage
        groups.append((advantages, info))
        records.extend(group)
    path = Path(spec['out'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(item, sort_keys=True, default=_jsonable) + '\n' for item in records))
    os.replace(temporary, path)
    summary = {'records': len(records), 'contrastGroups': sum(1 for _, info in groups if info['contrast']),
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


COMMANDS = {'bank-init': bank_init, 'targets': targets, 'train': train, 'propose': propose, 'ladder': ladder,
            'service': service, 'collect': collect, 'corpus': corpus, 'controls': controls, 'meta-examples': meta_examples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=sorted(COMMANDS))
    parser.add_argument('--spec', required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    try:
        result = COMMANDS[args.command](spec)
    except LearnedOptimizerError as error:
        print(json.dumps({'error': str(error)}))
        return 2
    finally:
        while _OPEN:
            _OPEN.pop().store.close()
    print(json.dumps({'command': args.command, 'result': result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
