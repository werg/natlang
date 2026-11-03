"""`python -m natlang_neuralese.tune.learned.stage`: the ``learned_optimizer`` recipe stage
(``train/adaptation_stages.py``), the learned-optimizer lane as one resumable stage.

    python -m natlang_neuralese.tune.learned.stage --adaptation ADAPTATION_MANIFEST --protection REGISTRY \
        --continue-from CK [--parent-selection SELECTION] [--base-snapshot DIR] --out DIR [--device cuda] \
        [stage parameters] [--dry-run]

On the tuned programs of a program-adaptation stage (its ``adaptation-manifest.json``):

1. ``specs``: family split into the optimizer's corpus side and held side (``held_fraction`` of families never supply
   bank sources, targets or meta-examples), bank-init/corpus specs and train/meta templates (``lfm3-*-spec.json``)
   with ``train_basis``/``train_bank``/``train_keys`` explicitly enabled;
2. ``bank-init``, ``corpus``; outer cycles (``lane.outer_next``): train to each bank publication, adopt it, refit the
   labels against it;
3. one outer meta-RL round with real task outcomes (proposals, ``natlang tune ROOT --score-proposals`` per program,
   collect, the exact expected-score policy update), then outer cycles again for a bank that update published;
4. ``meta-examples``; planted controls (telemetry: support recall of the optimizer and of an in-bank fit);
5. the controller program: split, protected-world-checked training references, ``natlang tune --rounds``;
6. held arms per held program (in a copy, so the program-adaptation stage stays as it was): the crisp-baseline
   release and the learned controller release, each by ``natlang tune P --learned``, against the program's own
   gradient-fit release from program adaptation, on development roots paired by root; then one confirmation run each;
7. ``report.json`` (``natlang.learned-optimizer-report/1``) and ``optimizer-manifest.json``.

Promotion (``held.met``): over the held programs, learned development successes >= crisp-baseline and >= the gradient
fit, paired by root under identical execution limits, with the learned adaptation's wall time at most the gradient
fit's (matched or less compute), and every confirmation measured once. Each step is a child process with a done
marker; a rerun skips finished steps. ``--dry-run`` runs steps 1-4 on the fixture tunes of a dry program-adaptation
stage (shaping reward only, no task scoring, no held side); it never passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ...common.hashing import sha256_file_hex
from ...train.adaptation_stages import (LEARNED_OPTIMIZER, LEARNED_OPTIMIZER_DEFAULTS, LEARNED_OPTIMIZER_REPORT,
                                        validate_parameters)
from ..stage import REPO, LANE, Tuner, _arms, _read, _write, arm_summary, node_binary, paired_gain
from . import lane

MANIFEST_SCHEMA = 'natlang.learned-optimizer-manifest/1'
# Tiny-fixture settings (dry runs only), as the lane's --dry-run used.
DRY_OVERRIDES = {
    '*': {'device': 'cpu', 'dtype': 'float32'},
    'bank-init': {'rank': 4, 'per_source_rank': 2, 'component_rank': 2, 'component_steps': 2},
    'corpus': {'fit': {'steps': 2, 'refine_steps': 1}, 'subsets': 2, 'max_support': 4, 'incremental_max': 1,
               'recipe': {'batch_rows': 4}},
    'train': {'steps': 3, 'recipe': {'writer': {'kind': 'free-lora', 'rank': 2, 'alpha': 2.0}, 'batch_targets': 1,
                                     'publish_every': 2}},
    'meta': {'steps': 2, 'refine_steps': 1, 'fit': {'steps': 2, 'refine_steps': 1},
             'learned': {'scopes': {'tree': 3, 'folder': 3, 'lambda': 3}, 'steps': 2, 'patience': 2}}}


class Steps:
    """Child-process steps with done markers and logs under ``DIR/steps``."""

    def __init__(self, out, env):
        self.directory = Path(out) / 'steps'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env = env

    def done(self, name):
        return (self.directory / f'{name}.done').exists()

    def run(self, name, command, *, result=None, cwd=REPO, env=None):
        """Run once; ``result``: the command name its last JSON line must report (learned-optimizer CLI)."""
        if self.done(name):
            return _read(self.directory / f'{name}.done')
        log = self.directory / f'{name}.log'
        started = time.time()
        with log.open('a') as stream:
            stream.write(json.dumps({'command': command, 'started': started}) + '\n')
            stream.flush()
            status = subprocess.run(command, env=env or self.env, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode
        if status != 0:
            raise SystemExit(f'{name}: exit status {status}; see {log}')
        last = None
        if result is not None:
            lines = [line for line in log.read_text().splitlines() if line.startswith('{')]
            last = json.loads(lines[-1]) if lines else {}
            if last.get('command') != result:
                raise SystemExit(f'{name}: finished without its {result} result line; see {log}')
        record = {'seconds': time.time() - started, 'result': last}
        _write(self.directory / f'{name}.done', record)
        print(json.dumps({'event': 'step', 'step': name, 'seconds': round(record['seconds'], 1)}), flush=True)
        return record


def learned(steps, command, spec, name=None):
    return steps.run(name or command, [sys.executable, '-m', 'natlang_neuralese.tune.learned', command, '--spec', str(spec)],
                     result=command)


def outer_cycles(steps, op, phase):
    while True:
        action = lane.outer_next(op)
        if action['action'] == 'train':
            learned(steps, 'train', action['spec'], f'{phase}-outer-train-c{action["cycle"]}-s{action["step"]}')
        elif action['action'] == 'adopt':
            learned(steps, 'adopt-bank', action['adopt'], f'{phase}-adopt-c{action["cycle"]}')
            learned(steps, 'corpus', action['corpus'], f'{phase}-corpus-c{action["cycle"]}')
            Path(action['complete']).touch()
        elif action['action'] == 'done':
            return action
        else:
            raise SystemExit(f'outer-next gave no action: {action}')


def outer_rl(steps, op, tuner, *, per_context, score):
    """One outer meta-RL round with real task outcomes (the former outer-rl-round.sh)."""
    directory = Path(op) / 'outer-rl'
    directory.mkdir(parents=True, exist_ok=True)
    train = json.loads((Path(op) / 'train-spec.json').read_text())
    common = {key: train[key] for key in ('checkpoint', 'base', 'store', 'device', 'dtype', 'bank', 'targets', 'state')
              if key in train}
    _write(directory / 'proposals-spec.json', {**common, 'perContext': per_context, 'out': str(directory / 'proposals.jsonl'),
                                               'releases': str(directory / 'releases')})
    _write(directory / 'collect-spec.json', {'proposals': str(directory / 'proposals.jsonl'),
                                             'outcomes': str(directory / 'outcomes.json'),
                                             'out': str(directory / 'records.jsonl')})
    _write(directory / 'policy-spec.json', {**train, 'steps': 0, 'policy_records': [str(directory / 'records.jsonl')]})
    proposed = learned(steps, 'proposals', directory / 'proposals-spec.json', 'outer-rl-proposals')
    if score:
        for index, (root, request) in enumerate(sorted(proposed['result']['result']['requests'].items())):
            name = f'outer-rl-score-{index}'
            if not steps.done(name):
                status = tuner.run(root, '--score-proposals', request, log=steps.directory / f'{name}.log')
                if status != 0:
                    raise SystemExit(f'{name}: natlang tune --score-proposals exit {status}')
                _write(steps.directory / f'{name}.done', {'root': root, 'request': request})
    outcomes = {}
    for path in sorted(glob.glob(str(directory / 'releases' / 'requests' / '*.outcomes.json'))):
        outcomes.update(json.loads(Path(path).read_text()))
    _write(directory / 'outcomes.json', outcomes)
    learned(steps, 'collect', directory / 'collect-spec.json', 'outer-rl-collect')
    learned(steps, 'train', directory / 'policy-spec.json', 'outer-rl-policy')


def stage_tunes(op, adaptation):
    """The program-adaptation stage's tune results, as the lane's ``tunes/`` directory."""
    manifest = json.loads(Path(adaptation).read_text())
    if manifest.get('schema') != 'natlang.program-adaptation-manifest/1':
        raise SystemExit(f'{adaptation} is not a program-adaptation manifest')
    tunes = Path(op) / 'tunes'
    tunes.mkdir(parents=True, exist_ok=True)
    copied = []
    for item in manifest['programs']:
        reference = item.get('tune_result')
        if not reference:
            continue
        if sha256_file_hex(reference['path']) != reference['sha256']:
            raise SystemExit(f'{reference["path"]} changed after program adaptation recorded it')
        target = tunes / f'{item["program"]}.json'
        if not target.exists():
            shutil.copy2(reference['path'], target)
        copied.append(item['program'])
    return manifest, copied


def held_arms(steps, op, tuner, program, baseline, learned_spec, crisp_spec):
    """Crisp-baseline, learned and gradient-fit releases of one held program on development, then confirmation once."""
    name = Path(program).name
    held_root = Path(op) / 'held' / name
    if not held_root.exists():
        pending = held_root.with_name(name + '.pending')
        shutil.rmtree(pending, ignore_errors=True)
        shutil.copytree(program, pending, ignore=shutil.ignore_patterns('node_modules'))
        pending.replace(held_root)

    def tune(label, *args):
        record_path = steps.directory / f'held-{name}-{label}.done'
        if record_path.exists():
            return _read(record_path)
        started = time.time()
        status = tuner.run(held_root, *args, log=steps.directory / f'held-{name}-{label}.log')
        if status != 0:
            raise SystemExit(f'held {name} {label}: natlang tune exit {status}')
        active = _read(held_root / '.natlang' / 'tune' / 'active.json') or {}
        record = {'seconds': time.time() - started, 'release': active.get('release')}
        _write(record_path, record)
        return record

    crisp = tune('crisp', '--learned', str(crisp_spec))
    tune('crisp-development', '--evaluate', 'development', '--release', crisp['release'])
    mine = tune('learned', '--learned', str(learned_spec))
    tune('learned-development', '--evaluate', 'development')
    tune('gradient-development', '--evaluate', 'development', '--release', baseline)
    tune('learned-confirmation', '--evaluate', 'confirmation')
    tune('crisp-confirmation', '--evaluate', 'confirmation', '--release', crisp['release'])
    tune('gradient-confirmation', '--evaluate', 'confirmation', '--release', baseline)
    development = {arm.get('release'): arm for arm in _arms(held_root, 'development').values()}
    arms = {'untuned': next((arm for arm in _arms(held_root, 'development').values() if arm.get('arm') == 'untuned'), None),
            'crisp': development.get(crisp['release']), 'learned': development.get(mine['release']),
            'gradient': development.get(baseline)}
    if any(arm is None for arm in arms.values()):
        raise SystemExit(f'held {name}: a development arm is missing')
    confirmation = {arm.get('release'): arm_summary(arm) for arm in _arms(held_root, 'confirmation').values()}
    return {'program': name, 'releases': {'crisp': crisp['release'], 'learned': mine['release'], 'gradient': baseline},
            'development': {key: arm_summary(arm) for key, arm in arms.items()},
            'learned_vs_crisp': paired_gain(arms['crisp'], arms['learned']),
            'learned_vs_gradient': paired_gain(arms['gradient'], arms['learned']),
            'adaptation_seconds': {'learned': mine['seconds'], 'crisp': crisp['seconds']},
            'confirmation': {key: confirmation.get(release) for key, release in
                             (('crisp', crisp['release']), ('learned', mine['release']), ('gradient', baseline))}}


def held_decision(results, gradient_seconds, minimum):
    reasons = []
    if len(results) < minimum:
        reasons.append(f'{len(results)} held program(s) measured; promotion needs {minimum}')
    totals = {'vs_crisp': 0, 'vs_gradient': 0}
    for item in results:
        for key, field in (('vs_crisp', 'learned_vs_crisp'), ('vs_gradient', 'learned_vs_gradient')):
            comparison = item[field]
            if not comparison.get('paired'):
                reasons.append(f'{item["program"]}: {field} unpaired ({comparison.get("reason")})')
            else:
                totals[key] += comparison['gain']
        seconds = gradient_seconds.get(item['program'])
        if seconds is None:
            reasons.append(f'{item["program"]}: gradient-fit compute unrecorded')
        elif item['adaptation_seconds']['learned'] > seconds:
            reasons.append(f'{item["program"]}: learned adaptation took {item["adaptation_seconds"]["learned"]:.0f}s, '
                           f'more than the gradient fit\'s {seconds:.0f}s')
        if any(value is None for value in item['confirmation'].values()):
            reasons.append(f'{item["program"]}: a confirmation arm is missing')
    if results and totals['vs_crisp'] < 0:
        reasons.append(f'learned trails the crisp baseline by {-totals["vs_crisp"]} development roots')
    if results and totals['vs_gradient'] < 0:
        reasons.append(f'learned trails the gradient fit by {-totals["vs_gradient"]} development roots')
    return {'met': not reasons, 'reason': '; '.join(reasons) or None, 'totals': totals, 'programs': results}


def run(args, parameters):
    op = Path(args.out).resolve()
    op.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.continue_from).resolve()
    if args.dry_run:
        sha = sha256_file_hex(checkpoint)
    else:
        from ...common.checkpoint_digest import verified_checkpoint_digest
        sha = verified_checkpoint_digest(checkpoint)
    if args.parent_selection:
        from ...train.promotion import require_parent_selection
        require_parent_selection(json.loads(Path(args.parent_selection).read_text()), sha)
    operation = {'schema': 'natlang.learned-optimizer-operation/1', 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha,
                 'adaptation': {'path': str(Path(args.adaptation).resolve()), 'sha256': sha256_file_hex(args.adaptation)},
                 'protection': {'path': str(Path(args.protection).resolve()), 'sha256': sha256_file_hex(args.protection)},
                 'parameters': parameters, 'dry_run': bool(args.dry_run)}
    bound = _read(op / 'operation.json')
    if bound is not None and bound != operation:
        raise SystemExit(f'{op}/operation.json binds another checkpoint, adaptation or parameters; use a new stage directory')
    _write(op / 'operation.json', operation)
    manifest, tuned = stage_tunes(op, args.adaptation)
    if manifest['checkpoint_sha256'] != sha:
        raise SystemExit('the program-adaptation stage tuned another backbone checkpoint')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(REPO / 'training' / 'neuralese'), str(REPO),
                                                        *[p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]]),
           'NATLANG_HASH_CACHE': os.environ.get('NATLANG_HASH_CACHE', str(op / 'hash-cache.jsonl'))}
    steps = Steps(op, env)
    overrides = lane._merge(DRY_OVERRIDES if args.dry_run else {}, parameters.get('spec_overrides') or {})
    overrides['train'] = lane._merge(overrides.get('train', {}), {'recipe': {key: True for key in
                                                                          ('train_basis', 'train_bank', 'train_keys')}})
    if not steps.done('specs'):
        lane.specs(op, checkpoint, LANE, base=args.base_snapshot, held_fraction=0 if args.dry_run else parameters['held_fraction'],
                   min_programs=parameters['min_corpus_programs'], extra=overrides)
        _write(steps.directory / 'specs.done', {'programs': tuned})
    learned(steps, 'bank-init', op / 'bank-init-spec.json')
    learned(steps, 'corpus', op / 'corpus-spec.json')
    outer_cycles(steps, op, 'first')
    tuner = None if args.dry_run else Tuner(op, checkpoint, sha, device=args.device, tuner=None,
                                            concurrency=parameters.get('concurrency', 4), base=args.base_snapshot)
    outer_rl(steps, op, tuner, per_context=2 if args.dry_run else parameters['outer_rl_per_context'], score=not args.dry_run)
    outer_cycles(steps, op, 'after-rl')
    learned(steps, 'meta-examples', op / 'meta-spec.json')
    controls = None
    if parameters['planted_controls']:
        train = json.loads((op / 'train-spec.json').read_text())
        corpus = json.loads((op / 'corpus-spec.json').read_text())
        spec = {key: train[key] for key in ('checkpoint', 'base', 'store', 'device', 'dtype', 'bank', 'state') if key in train}
        spec.update(plan=corpus['programs'][0]['plan'], work=str(op / 'controls' / 'work'), out=str(op / 'controls' / 'summary.json'))
        _write(op / 'controls' / 'spec.json', spec)
        learned(steps, 'controls', op / 'controls' / 'spec.json')
        controls = _read(op / 'controls' / 'summary.json')
    held = {'met': False, 'reason': 'dry run: no held side', 'programs': []}
    if not args.dry_run:
        controller = op / 'controller'
        if not controller.exists():
            shutil.copytree(REPO / 'applications' / 'learned-optimizer', controller)
        examples = op / 'meta' / 'tune.examples.jsonl'
        steps.run('controller-split', [node_binary(), str(LANE / 'controller-split.mjs'), str(controller), str(examples)],
                  env={**env, **tuner.env()})
        steps.run('controller-supply', [sys.executable, str(LANE / 'controller-supply.py'), '--controller', str(controller),
                                        '--examples', str(examples), '--root', str(REPO)], env={**env, **tuner.env()})
        if not steps.done('controller-tune'):
            status = tuner.run(controller, '--examples', str(examples), '--rounds', log=steps.directory / 'controller-tune.log')
            if status != 0:
                raise SystemExit(f'controller tune exit {status}')
            _write(steps.directory / 'controller-tune.done', {'status': status})
        active = json.loads((controller / '.natlang' / 'tune' / 'active.json').read_text())
        settings = json.loads((op / 'meta-spec.json').read_text())['learned']
        learned_spec = op / 'learned-controller.json'
        binding = {**settings, 'controller': {'root': str(controller), 'release': active['release']}}
        if learned_spec.exists() and json.loads(learned_spec.read_text()) != binding:
            raise SystemExit('controller binding changed: use a new stage directory')
        _write(learned_spec, binding)
        crisp_spec = op / 'crisp-baseline.json'
        _write(crisp_spec, {**{key: value for key, value in settings.items() if key != 'controller'},
                            'policy': 'crisp-baseline'})
        split = json.loads((op / 'optimizer-split.json').read_text())
        timings = {}
        results = []
        for name in split['held']:
            result = json.loads((op / 'tunes' / f'{name}.json').read_text())
            timing = _read(Path(result['root']).parents[1] / 'tunes' / f'{name}.timing.json')
            if timing:
                timings[name] = timing['seconds']
            results.append(held_arms(steps, op, tuner, result['root'], result['best']['release'], learned_spec, crisp_spec))
        held = held_decision(results, timings, parameters['min_held_programs'])
    summary = lane.summary(op)
    state = _read(json.loads((op / 'train-spec.json').read_text())['state'], {})
    recipe = (state.get('recipe') or {})
    outer = {'steps': summary['outer']['step'], 'published': summary['outer']['published'],
             'adoptions': len(summary['outer']['adoptions']),
             **{key: recipe.get(key) is True for key in ('train_basis', 'train_bank', 'train_keys')}}
    _write(op / 'optimizer-manifest.json', {'schema': MANIFEST_SCHEMA, 'checkpoint_sha256': sha,
                                            'bank': summary['outer']['bank'], 'state': str(json.loads(
                                                (op / 'train-spec.json').read_text())['state']),
                                            'controller': _read(op / 'learned-controller.json'),
                                            'split': summary['optimizer_split']})
    report = {'schema': LEARNED_OPTIMIZER_REPORT, 'kind': LEARNED_OPTIMIZER, 'completed': not args.dry_run,
              'checkpoint_sha256': sha, 'outer_training': outer, 'outer_rl_records': summary['outer_rl_records'],
              'meta_examples': summary['meta_examples'], 'planted_controls': controls, 'held': held,
              'split': summary['optimizer_split'], 'production_exposure_advanced': False, 'dry_run': bool(args.dry_run),
              'note': 'held-family outcomes are paired development measurements with one confirmation run; '
                      'saved banks, steps and passing stages are not capability claims'}
    _write(op / 'report.json', report)
    print(json.dumps({'event': 'report', 'completed': report['completed'], 'held': held['met'],
                      'outer_steps': outer['steps']}), flush=True)
    return 0


def parse_parameters(args):
    parameters = dict(LEARNED_OPTIMIZER_DEFAULTS)
    for key in ('held_fraction', 'min_corpus_programs', 'min_held_programs', 'outer_rl_per_context'):
        if getattr(args, key) is not None:
            parameters[key] = getattr(args, key)
    for key in ('train_basis', 'train_bank', 'train_keys', 'planted_controls'):
        if getattr(args, key) is not None:
            parameters[key] = getattr(args, key)
    if args.spec_overrides is not None:
        parameters['spec_overrides'] = json.loads(args.spec_overrides)
    validate_parameters(LEARNED_OPTIMIZER, parameters)
    return parameters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--adaptation', required=True, help='the program-adaptation stage\'s adaptation-manifest.json')
    parser.add_argument('--protection', required=True)
    parser.add_argument('--continue-from', required=True)
    parser.add_argument('--parent-selection')
    parser.add_argument('--base-snapshot')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--held-fraction', type=float)
    for key in ('min_corpus_programs', 'min_held_programs', 'outer_rl_per_context'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('train_basis', 'train_bank', 'train_keys', 'planted_controls'):
        parser.add_argument('--' + key.replace('_', '-'), dest=key, action='store_const', const=True)
        parser.add_argument('--no-' + key.replace('_', '-'), dest=key, action='store_const', const=False)
    parser.add_argument('--spec-overrides')
    args = parser.parse_args(argv)
    return run(args, parse_parameters(args))


if __name__ == '__main__':
    raise SystemExit(main())
