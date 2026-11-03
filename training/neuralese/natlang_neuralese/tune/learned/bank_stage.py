"""`python -m natlang_neuralese.tune.learned.bank_stage`: the ``bank_metalearning`` recipe stage
(``train/adaptation_stages.py``; plans/neuralese/CONTINUOUS_BANK_METALEARNING_20261103.md).

Two commands:

    bank_stage pool --protection REGISTRY --out DIR [--graph-sources MANIFEST_OR_FILE ... --pattern RE [--prefix S]]
        [--rollouts FILE ... --checkpoint CK [--policy SHA]] [--max-tokens N] [--held-fraction F] [--seed S]
        [--max-meta-families N] [--max-held-families N]

builds an episode pool (``DIR/pool.json``, ``DIR/rows.jsonl``) from verified trajectories: native graph sources whose
graph ids name a family and a world (``--pattern`` with named groups), and/or successful oracle-checked RL rollouts
(family: task family; world: task). Every row passes the protected-worlds union.

    bank_stage --episodes DIR/pool.json --protection REGISTRY --continue-from CK --out DIR [--device cuda]
        [stage parameters] [--recipe-overrides JSON] [--adaptation MANIFEST]

runs the stage, resumably: evaluation of the initial bank, ``outer_steps`` first-order outer steps with a held
development evaluation every ``eval_every`` steps, the promotion decision on the final development evaluation, one
confirmation evaluation when it is met, the published bank (``bank-manifest.json``, `bank.save_bank`) and
``report.json`` (``natlang.bank-metalearning-report/1``). Rerunning with a newer ``--continue-from`` continues the same
bank on that parent through a recorded transition. Never advances production lineage exposure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ...common.hashing import sha256_file_hex
from ...train.adaptation_stages import (BANK_METALEARNING, BANK_METALEARNING_DEFAULTS, BANK_METALEARNING_REPORT,
                                        validate_parameters)
from . import meta_bank as MB


def _log(event, **fields):
    print(json.dumps({'event': event, 'time': round(time.time(), 1), **fields}, sort_keys=True), flush=True)


def _write(path, value):
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(value, indent=1, sort_keys=True))
    partial.replace(path)


def _protection(registry):
    from ...common.custody import ProtectedWorlds
    return ProtectedWorlds.load(registry=registry)


def _graph_files(paths):
    files = []
    for path in map(Path, paths):
        if path.suffix == '.json':  # a train-sources manifest: its native-graph files
            manifest = json.loads(path.read_text())
            files += [Path(source['path']) for source in manifest['sources']
                      if source.get('kind') == 'native-graphs' and source['path'].endswith(('.jsonl', '.jsonl.zst'))]
        else:
            files.append(path)
    return files


def pool_command(argv):
    parser = argparse.ArgumentParser(prog='bank_stage pool')
    parser.add_argument('--protection', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--graph-sources', nargs='*', default=[])
    parser.add_argument('--pattern')
    parser.add_argument('--prefix')
    parser.add_argument('--rollouts', nargs='*', default=[])
    parser.add_argument('--checkpoint')
    parser.add_argument('--policy')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--held-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-meta-families', type=int)
    parser.add_argument('--max-held-families', type=int)
    args = parser.parse_args(argv)
    protected = _protection(args.protection)
    rows, counts = [], {}
    if args.graph_sources:
        if not args.pattern:
            parser.error('--graph-sources needs --pattern with named groups family and world')
        files = _graph_files(args.graph_sources)
        # Two passes: index every candidate cheaply, choose the pool's families, then read only their graphs.
        index, counts['graphs'] = MB.graph_rows(files, args.pattern, protected=protected, max_tokens=args.max_tokens,
                                                prefix=args.prefix, index_only=True)
        chosen = MB.build_pool(index, held_fraction=args.held_fraction, max_families=_limits(args), seed=args.seed)
        found, _ = MB.graph_rows(files, args.pattern, protected=protected, max_tokens=args.max_tokens,
                                 prefix=args.prefix, families=set(chosen['families']))
        rows += found
        counts['graph_files'] = len(files)
    if args.rollouts:
        if not args.checkpoint:
            parser.error('--rollouts are rendered with the parent checkpoint\'s tokenizer and template: --checkpoint')
        from ...serve import load_native_engine
        engine = load_native_engine(args.checkpoint, device='cpu')
        found, counts['rollouts'] = MB.rollout_rows(engine, args.rollouts, protected=protected,
                                                    max_tokens=args.max_tokens, policy=args.policy)
        rows += found
    pool = MB.build_pool(rows, held_fraction=args.held_fraction, max_families=_limits(args), seed=args.seed)
    receipt = MB.write_pool(pool, args.out)
    receipt.update({'counts': counts, 'protected_sha256': protected.digest, 'candidate_rows': len(rows),
                    'sources': {'graphs': [str(p) for p in args.graph_sources], 'pattern': args.pattern,
                                'rollouts': list(args.rollouts), 'policy': args.policy}})
    _write(Path(args.out) / 'pool-receipt.json', receipt)
    _log('pool', **{k: v for k, v in receipt.items() if k != 'sources'})
    return 0


def _limits(args):
    if args.max_meta_families or args.max_held_families:
        return {'meta-train': args.max_meta_families or 10 ** 9, 'held': args.max_held_families or 10 ** 9}
    return None


def _engine(checkpoint, device, dtype, store):
    import torch
    from ...serve import load_native_engine
    from ...serve.store import TensorStore
    from .checkpointing import configure_adaptation_checkpointing
    engine = load_native_engine(checkpoint, device=device, dtype=getattr(torch, dtype))
    configure_adaptation_checkpointing(engine, None)
    engine.store.close()
    engine.store = TensorStore(str(store))
    return engine


def parse_parameters(args):
    parameters = dict(BANK_METALEARNING_DEFAULTS)
    for key in BANK_METALEARNING_DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            parameters[key] = value
    if args.recipe_overrides:
        parameters['recipe_overrides'] = json.loads(args.recipe_overrides)
    validate_parameters(BANK_METALEARNING, parameters)
    return parameters


def recipe_of(parameters):
    fields = {key: parameters[key] for key in ('rank', 'components', 'component_rank', 'inner_steps', 'layer_groups',
                                               'residual_rank', 'meta_batch', 'train_bank', 'train_basis')}
    return MB.MetaRecipe(**{**fields, **(parameters.get('recipe_overrides') or {})})


def run(args, parameters):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pool_path = Path(args.episodes).resolve()
    pool = MB.read_pool(pool_path.parent)
    protected = _protection(args.protection)
    for family in pool['families'].values():  # the protection union may have grown since the pool was built
        for world in family['worlds']:
            for row in world['rows']:
                graph = row['graph']
                if protected.related({'id': graph['id'], 'source_groups': graph.get('source_groups') or []}):
                    raise SystemExit(f'pool row {graph["id"]} is now related to a protected world: rebuild the pool')
    if pool['held_fraction'] != parameters['held_fraction']:
        raise SystemExit(f'the pool holds out {pool["held_fraction"]} of families; the stage declares {parameters["held_fraction"]}')
    held = [n for n, f in pool['families'].items() if f['role'] == 'held']
    if len(held) < parameters['min_held_families']:
        raise SystemExit(f'{len(held)} held families < min_held_families {parameters["min_held_families"]}')
    checkpoint = Path(args.continue_from)
    parent = sha256_file_hex(checkpoint)
    # Schedule parameters (how long to run, how often to evaluate) may change on a continuation; everything else
    # binds the operation.
    schedule = ('outer_steps', 'eval_every')
    operation = {'episodes': {'path': str(pool_path), 'sha256': sha256_file_hex(pool_path)},
                 'protection_sha256': protected.digest,
                 'parameters': {k: v for k, v in parameters.items() if k not in schedule}}
    bound = out / 'operation.json'
    if bound.exists() and json.loads(bound.read_text()) != operation:
        raise SystemExit('this stage directory is bound to other episodes, protection or parameters')
    _write(bound, operation)
    recipe = recipe_of(parameters)
    if args.dry_run:  # the cloud pipeline check: inputs, custody and parameters only, no model
        _log('dry_run', held=len(held), meta_train=len(pool['families']) - len(held), recipe=recipe.as_json())
        return 0
    engine = _engine(checkpoint, args.device, args.dtype, out / 'store')
    fits = None
    if recipe.init == 'pca':
        # Data-driven initialization: plain LoRA fits of the meta-training families (saved per family; the same
        # parameterization and pass count as the baseline arm). Fits belong to the parent they were made on.
        fits = MB.family_lora_fits(engine, pool, recipe, MB.select_targets(engine, recipe), out / 'lora-fits' / parent[:16],
                                   log=lambda family, value: _log('family_lora_fit', family=family, **value))
    learner = MB.BankMetaLearner(engine, pool, recipe, out / 'learner', parent_sha256=parent, fits=fits)
    _log('start', step=learner.state['step'], parent=parent, held=len(held), spectrum=learner.state.get('spectrum'),
         meta_train=len(pool['families']) - len(held), targets=learner.state['targets'], groups=learner.state['groups'],
         components=learner.state['components'])
    evaluated = {e['step'] for e in learner.state['evaluations'] if e['split'] == 'dev'}
    def evaluate():
        if learner.state['step'] in evaluated:
            return next(e for e in learner.state['evaluations'] if e['split'] == 'dev' and e['step'] == learner.state['step'])
        result = learner.evaluate('dev', log=lambda family, value: _log('held_family', step=learner.state['step'],
                                                                         family=family, **value))
        evaluated.add(learner.state['step'])
        learner.save()
        _log('evaluation', step=result['step'], mean=result['mean'], meta_wins=result['meta_wins'],
             mean_distinct=result['mean_distinct'], distinct=result['families_distinct_count'], seconds=result['seconds'])
        return result
    evaluate()
    while learner.state['step'] < parameters['outer_steps']:
        entry = learner.step()
        learner.save()
        _log('outer', step=entry['step'], seconds=entry['seconds'], grad=entry['grad_norm'], update=entry['update_norm'],
             regression=entry['regression_unexplained'], step_size=entry['step_size_mean'],
             query=[round(r['query_fitted'], 4) for r in entry['families']],
             support=[(round(r['support_start'], 4), round(r['support_fitted'], 4)) for r in entry['families']])
        if learner.state['step'] % parameters['eval_every'] == 0:
            evaluate()
    final = evaluate()
    held_decision = MB.promotion(final, min_families=parameters['min_held_families'], margin=parameters['promotion_margin'])
    confirmation = {'met': False, 'reason': 'development promotion not met; confirmation not spent'}
    if held_decision['met']:
        done = [e for e in learner.state['evaluations'] if e['split'] == 'confirmation' and e['step'] == learner.state['step']]
        result = done[0] if done else learner.evaluate('confirmation', log=lambda family, value: _log(
            'held_family', split='confirmation', family=family, **value))
        learner.save()
        confirmation = {**MB.promotion(result, min_families=parameters['min_held_families'],
                                       margin=parameters['promotion_margin']), 'step': result['step']}
    manifest = learner.publish()
    learner.save()
    steps = learner.state['history']
    seconds = [h['seconds'] for h in steps]
    report = {'schema': BANK_METALEARNING_REPORT, 'kind': BANK_METALEARNING, 'completed': True,
              'checkpoint_sha256': parent, 'parents': learner.state['parents'], 'bank': manifest['snapshot'],
              'outer_training': {'steps': learner.state['step'], 'train_bank': recipe.train_bank,
                                 'train_basis': recipe.train_basis,
                                 'seconds_per_step': sum(seconds) / len(seconds) if seconds else None,
                                 'passes_per_step': steps[-1]['passes'] if steps else None},
              'held': held_decision, 'confirmation': confirmation,
              'initialization': {'kind': recipe.init, 'spectrum': learner.state.get('spectrum')},
              'evaluations': [{k: e.get(k) for k in ('split', 'step', 'mean', 'meta_wins', 'mean_distinct',
                                                     'meta_wins_distinct', 'families_distinct_count', 'seconds')}
                              for e in learner.state['evaluations']],
              'adaptation_baselines': str(args.adaptation) if args.adaptation else None,
              'production_exposure_advanced': False,
              'note': 'held-family teacher-forced losses on verified trajectories after matched coefficient/LoRA fits; '
                      'task-outcome lift requires rollouts of the adapted programs and is not claimed here'}
    _write(out / 'report.json', report)
    _write(out / 'bank-manifest.json', manifest)
    _log('report', held=held_decision['met'], confirmation=confirmation['met'], steps=learner.state['step'])
    engine.store.close()
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'pool':
        return pool_command(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--protection', required=True)
    parser.add_argument('--continue-from', required=True)
    parser.add_argument('--parent-selection')
    parser.add_argument('--base-snapshot')
    parser.add_argument('--adaptation')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--held-fraction', type=float)
    parser.add_argument('--promotion-margin', type=float)
    for key in ('outer_steps', 'eval_every', 'meta_batch', 'rank', 'components', 'component_rank', 'inner_steps',
                'layer_groups', 'residual_rank', 'min_held_families', 'max_parent_lag_steps'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('train_bank', 'train_basis'):
        parser.add_argument('--' + key.replace('_', '-'), dest=key, action='store_const', const=True)
        parser.add_argument('--no-' + key.replace('_', '-'), dest=key, action='store_const', const=False)
    parser.add_argument('--recipe-overrides')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    return run(args, parse_parameters(args))


if __name__ == '__main__':
    raise SystemExit(main())
