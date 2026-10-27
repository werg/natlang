"""Entrypoint of the joint learned optimizer: every worker resumes from a declared JSON spec.

    python -m natlang_neuralese.tune.learned bank-init --spec SPEC   gradient-subspace basis + source components + keys
    python -m natlang_neuralese.tune.learned targets   --spec SPEC   in-bank fits -> assembled training targets (jsonl)
    python -m natlang_neuralese.tune.learned train     --spec SPEC   resumable outer training (supervised, then policy)
    python -m natlang_neuralese.tune.learned propose   --spec SPEC   apply the trained optimizer: bundle + receipt
    python -m natlang_neuralese.tune.learned ladder    --spec SPEC   ceiling ladder and gaps from measurements
    python -m natlang_neuralese.tune.learned service   --spec SPEC   one adaptation service of the Natlang `adapt` program

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
        rows = _rows(source['rows'])
        weights = row_loss_weights(row_losses(engine, rows))
        gradients.append(source_gradients(engine, rows, names, weights=weights))
        descriptors.append({'name': source['name'], 'kind': source['kind'], 'rows': len(rows)})
    basis = pooled_basis(engine.native_identity, gradients, spec['rank'], per_source_rank=spec.get('per_source_rank'),
                         weight_svd=weight_svd_basis(engine, names, spec['rank']), descriptors=descriptors)
    bank = new_bank(engine, basis, NumberFormat(float(spec.get('step', .05))), quarters=spec.get('quarters', 4))
    components, reports = [], {}
    for source in spec['sources']:
        component, report = source_component(engine, bank, _rows(source['rows']), source={'name': source['name'], 'kind': source['kind']},
                                             description=source['description'], rank=spec.get('component_rank', 8),
                                             steps=spec.get('component_steps', 20))
        components.append(component)
        reports[source['name']] = report
    bank = bank.with_components(components, reason='bank-init sources')
    manifest = save_bank(engine, bank)
    _write(spec['out'], {'bank': manifest, 'reports': reports})
    return manifest['snapshot']


def targets(spec):
    import torch
    from .assemble import store_fitted_states, training_target
    from .bank import load_bank
    from .evidence import build_view
    from .fit import ScopeArray, SupportSet, fit_in_bank, placeholder_adapters
    from .views import select_view, with_dropout
    engine = _engine(spec)
    bank = load_bank(engine, json.loads(Path(spec['bank']).read_text())['bank'])
    out = []
    for program in spec['programs']:
        manifest = json.loads(Path(program['manifest']).read_text())
        rows = _rows(program['rows'])
        support = SupportSet(engine, manifest, rows)
        placeholders = placeholder_adapters(engine, bank, sorted(program['scopes']))
        for episode in range(program.get('episodes', 1)):
            views = {}
            for scope, declared in sorted(program['scopes'].items()):
                view = select_view(bank, scope, declared['level'], declared['size'], policy=declared.get('policy', 'full-bank-v1'))
                if declared.get('dropout'):
                    view = with_dropout(view, declared['dropout'], seed=program.get('seed', 0) * 1000 + episode)
                views[scope] = view
            arrays = [ScopeArray(scope, placeholders[scope], list(view.slots), list(view.available)) for scope, view in views.items()]
            result = fit_in_bank(engine, bank, support, arrays, soft_states=program.get('incoming_states'),
                                 **spec.get('fit', {}))
            fitted_states = store_fitted_states(engine, result['states'])
            target_adapters = {scope: bank.compose_adapter(engine, bank.composed_cores(list(views[scope].slots),
                               result['arrays'][scope]['executed'])).id for scope in views}
            evidence = build_view(program['contract'], program.get('incoming_bundle', 'incoming'), program.get('evidence', []))
            target = training_target(engine, bank, manifest=manifest, rows=rows, contract=program['contract'], view=evidence,
                                     soft_texts=program.get('soft_texts', {}), incoming_states=program.get('incoming_states', {}),
                                     bank_views=views, target_integers={s: result['arrays'][s]['integers'] for s in views},
                                     target_adapters=target_adapters, target_states=fitted_states,
                                     target_rho={s: result['arrays'][s]['rho'].tolist() for s in views},
                                     target_delta={s: result['arrays'][s]['delta'].tolist() for s in views},
                                     placeholders=placeholders)
            target['fit'] = {'loss': result['loss'], 'saturated': {s: result['arrays'][s]['saturated'] for s in views},
                             'roundedToZero': {s: result['arrays'][s]['roundedToZero'] for s in views}}
            out.append(target)
    path = Path(spec['out'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(item, sort_keys=True, default=_jsonable) + '\n' for item in out))
    os.replace(temporary, path)
    return len(out)


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
            'service': service}


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
