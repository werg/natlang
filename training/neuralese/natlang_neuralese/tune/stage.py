"""`python -m natlang_neuralese.tune.stage`: the ``program_adaptation`` recipe stage (``train/adaptation_stages.py``).

Runs the existing ``natlang tune`` machinery per program on one earned backbone checkpoint and decides the stage's
task-behaviour entry and exit criteria from the measured paired development arms. Resumable: every step writes its
result before the next starts and a rerun skips finished work; the operation stays bound to its checkpoint, program
manifest, protection registry and parameters.

    python -m natlang_neuralese.tune.stage --programs MANIFEST --protection REGISTRY --continue-from CK \
        [--parent-selection SELECTION] [--base-snapshot DIR] --out DIR [--device cuda] [stage parameters] \
        [--shard I/N] [--dry-run]

Steps (each program in the manifest's order; ``--shard I/N`` runs every N-th program from I, for one-GPU sandboxes,
and leaves the stage report to an unsharded run over the merged directory):

1. stage the programs into ``DIR/programs`` (program files, the persisted grouped split and site IDs, earlier verified
   root records with their import custody; no earlier base-bound releases), and write privileged training references for catalog programs
   (``tune.supply privileged``, custody against every registered evaluation-only packet);
2. ``structure.json``: each program's structural shape (``tune.structure``);
3. entry: the untuned development arm of every program (``natlang tune P --evaluate development``). Entry holds when
   at least ``entry_min_headroom_programs`` programs score strictly between 0 and all roots;
4. tune: ``natlang tune P --rounds`` for every program below all roots, then the selected release's development arm;
5. gain: tuned minus untuned correct roots, paired by root under the same execution limits;
6. confirmation, once, for each release with gain >= ``min_development_gain`` (never used for selection);
7. distillation: ``natlang tune P --distill`` of each confirmed release, collected into ``DIR/distill`` as one
   training-split package for ordinary source admission (``register-distill``; export is not admission);
8. ``report.json`` (``natlang.program-adaptation-report/1``) and the result ``adaptation-manifest.json``.

Program tunes never advance production lineage exposure. ``--dry-run`` replaces steps 3-7 with the fixture stand-in
(``training/neuralese/learned-optimizer/dry_run_fixture.py``); its numbers are not measurements and it never passes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..train.adaptation_stages import (PROGRAM_ADAPTATION, PROGRAM_ADAPTATION_DEFAULTS, PROGRAM_ADAPTATION_REPORT,
                                       validate_parameters)
from .structure import distinct_shapes, program_structure

from ..common.paths import root as _logical_root

# The checkout holding ts-host, applications/ and the lane helpers. The recipe runner executes this module from its
# frozen Python runtime, so it is the logical repo root (NATLANG_REPO), never this file's location.
REPO = _logical_root('repo').resolve()
LANE = REPO / 'training' / 'neuralese' / 'learned-optimizer'
MANIFEST_SCHEMA = 'natlang.program-adaptation-manifest/1'
DISTILL_SCHEMA = 'natlang.program-adaptation-distill/1'
DEFAULT_NODE = '/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node'
# The lane's tuner settings (PROGRAM_TUNING_RUNS_20261026.md limits; reply/action targets only).
DEFAULT_TUNER = {'teacher': 'self', 'backend': 'native', 'softStates': True,
                 'execution': {'maxTurns': 12, 'maxTokens': 4096, 'maxSeconds': 120, 'maxFailureRepairs': 3},
                 'recipe': {'epochs': 3, 'batch_rows': 8, 'prompt_weight': 0},
                 'rounds': {'schedule': ['sft', 'rl', 'rl', 'rl'], 'samples': 4, 'temperature': 0.8, 'patience': 2,
                            'rlRecipe': {'epochs': 1}}}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    pending.replace(path)


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _merge(base, extra):
    out = dict(base)
    for key, value in (extra or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _log(event, **fields):
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def node_binary():
    configured = os.environ.get('NATLANG_NODE')
    if configured:
        return configured
    if Path(DEFAULT_NODE).is_file():
        return DEFAULT_NODE
    found = shutil.which('node')
    if not found:
        raise SystemExit('no node binary: set NATLANG_NODE')
    return found


class Tuner:
    """``natlang tune`` against the stage's own configuration (the bound checkpoint is the native student)."""

    def __init__(self, out, checkpoint, sha, *, device, tuner, concurrency, base=None):
        self.out = Path(out)
        self.config_home = self.out / 'natlang-config'
        settings = _merge(DEFAULT_TUNER, tuner)
        settings['concurrency'] = concurrency
        native = {'checkpoint': str(checkpoint), 'sha256': sha, 'python': sys.executable,
                  'pythonPath': [str(REPO / 'training' / 'neuralese'), str(REPO)], 'device': device,
                  'dtype': 'bfloat16' if device.startswith('cuda') else 'float32', **({'base': str(base)} if base else {})}
        config = {'defaultProfile': 'adaptation', 'profiles': {'adaptation': {'native': native}}, 'tune': settings}
        path = self.config_home / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise SystemExit(f'{path} binds another tuner configuration; use a new stage directory')
        _write(path, config)
        self.settings = settings

    def env(self):
        return {**os.environ, 'NATLANG_CONFIG_HOME': str(self.config_home),
                'PYTHONPATH': os.pathsep.join([str(REPO / 'training' / 'neuralese'), str(REPO),
                                               *[p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]]),
                'NATLANG_HASH_CACHE': os.environ.get('NATLANG_HASH_CACHE', str(self.out / 'hash-cache.jsonl'))}

    def run(self, program, *args, log, timeout=None):
        command = [node_binary(), str(REPO / 'ts-host' / 'bin' / 'natlang.mjs'), 'tune', str(program), *args]
        with open(log, 'a') as stream:
            stream.write(json.dumps({'command': command, 'started': time.time()}) + '\n')
            stream.flush()
            try:
                result = subprocess.run(command, env=self.env(), stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                                        cwd=REPO)
            except subprocess.TimeoutExpired:
                return 'timeout'
        return result.returncode


def _arms(program, partition):
    directory = Path(program) / '.natlang' / 'tune' / 'evaluations' / partition
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob('*.json'))} if directory.is_dir() else {}


def _root_success(root):
    return 1 if (root.get('correct') is True if root.get('oracle') else bool(root.get('ok'))) else 0


def arm_summary(arm):
    roots = arm.get('roots') or []
    return {'arm': arm.get('arm'), 'release': arm.get('release'), 'roots': len(roots),
            'success': sum(_root_success(root) for root in roots), 'execution': arm.get('execution'),
            'tokens': sum((root.get('promptTokens') or 0) + (root.get('completionTokens') or 0) for root in roots),
            'ms': sum(root.get('ms') or 0 for root in roots)}


def paired_gain(untuned, tuned):
    """Correct-root difference over the roots both arms ran, under the same execution limits."""
    if canonical_json_sha256_hex(untuned.get('execution') or {}) != canonical_json_sha256_hex(tuned.get('execution') or {}):
        return {'paired': False, 'reason': 'arms ran under different execution limits'}
    if (untuned.get('concurrency') or 1) != (tuned.get('concurrency') or 1):
        return {'paired': False, 'reason': 'arms ran with different roots in flight (wall-time limits see the load)'}
    base = {root['id']: _root_success(root) for root in untuned.get('roots') or []}
    mine = {root['id']: _root_success(root) for root in tuned.get('roots') or []}
    shared = sorted(set(base) & set(mine))
    if not shared:
        return {'paired': False, 'reason': 'no shared development roots'}
    return {'paired': True, 'roots': len(shared), 'untuned': sum(base[i] for i in shared),
            'tuned': sum(mine[i] for i in shared), 'gain': sum(mine[i] - base[i] for i in shared),
            'gained': [i for i in shared if mine[i] > base[i]], 'lost': [i for i in shared if mine[i] < base[i]]}


def _best_release(program):
    """The release the latest ``--rounds`` journal selected on development."""
    journals = sorted((Path(program) / '.natlang' / 'tune').glob('rounds/*/journal.json'), key=lambda path: path.stat().st_mtime)
    best = json.loads(journals[-1].read_text()).get('best') if journals else None
    return best.get('release') if isinstance(best, dict) else None


def _active_release(program):
    active = _read(Path(program) / '.natlang' / 'tune' / 'active.json')
    return active.get('release') if isinstance(active, dict) else None


def stage_programs(manifest_path, out, names=None):
    """Copy each listed program's files, persisted split and verified root records into ``out/programs``."""
    manifest = json.loads(Path(manifest_path).read_text())
    items = manifest.get('programs')
    if not isinstance(items, list) or not items:
        raise SystemExit(f'{manifest_path}: a program manifest lists {{"programs": [{{"program", "family", "path"}}]}}')
    source_dir = Path(manifest_path).resolve().parent
    if names:
        known = {item['program'] for item in items}
        if set(names) - known:
            raise SystemExit('program_names not in the manifest: ' + ', '.join(sorted(set(names) - known)))
        items = [item for item in items if item['program'] in names]
    target = Path(out) / 'programs'
    staged = []
    for item in items:
        source = Path(item.get('path') or source_dir / item['program'])
        if not source.is_absolute():
            source = source_dir / source
        destination = target / item['program']
        if not (destination / 'main.nl').exists():
            pending = destination.with_name(destination.name + '.pending')
            shutil.rmtree(pending, ignore_errors=True)

            def ignore(directory, names_, root=source):
                relative = Path(directory).relative_to(root)
                if relative == Path('.'):
                    return [name for name in names_ if name == 'node_modules']
                if relative == Path('.natlang'):
                    return [name for name in names_ if name != 'tune']
                if relative == Path('.natlang/tune'):
                    return [name for name in names_ if name not in ('splits.json', 'sites.json', 'data',
                                                                    'pool-import-custody-snapshots')]
                return []
            shutil.copytree(source, pending, ignore=ignore, symlinks=False)
            pending.replace(destination)
        staged.append({**item, 'path': str(destination), 'source_path': str(source)})
    prepared = {**{key: value for key, value in manifest.items() if key != 'programs'}, 'programs': staged,
                'staged_from': {'path': str(Path(manifest_path).resolve()), 'sha256': sha256_file_hex(manifest_path)}}
    path = target / 'programs-prepared.json'
    if path.exists() and json.loads(path.read_text()) != prepared:
        raise SystemExit(f'{path} stages another program set; use a new stage directory')
    _write(path, prepared)
    return staged


def _shard(items, shard):
    if not shard:
        return items
    index, count = shard
    return [item for position, item in enumerate(items) if position % count == index]


def _program_hours_timeout(parameters):
    hours = parameters.get('program_hours')
    return None if hours is None else hours * 3600


def run(args, parameters):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.continue_from).resolve()
    if args.dry_run:
        sha = sha256_file_hex(checkpoint)
    else:
        from ..common.checkpoint_digest import verified_checkpoint_digest
        sha = verified_checkpoint_digest(checkpoint)
    selection = None
    if args.parent_selection:
        from ..train.promotion import require_parent_selection
        selection = require_parent_selection(json.loads(Path(args.parent_selection).read_text()), sha)
    operation = {'schema': 'natlang.program-adaptation-operation/1', 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha,
                 'programs': {'path': str(Path(args.programs).resolve()), 'sha256': sha256_file_hex(args.programs)},
                 'protection': {'path': str(Path(args.protection).resolve()), 'sha256': sha256_file_hex(args.protection)},
                 'parent_selection': ({'path': str(Path(args.parent_selection).resolve()),
                                       'sha256': sha256_file_hex(args.parent_selection)} if selection else None),
                 'parameters': parameters, 'dry_run': bool(args.dry_run), 'pilot': bool(args.pilot)}
    bound = _read(out / 'operation.json')
    if bound is not None and bound != operation:
        raise SystemExit(f'{out}/operation.json binds another checkpoint, program set or parameters; use a new stage directory')
    _write(out / 'operation.json', operation)
    programs = stage_programs(args.programs, out, parameters.get('program_names'))
    catalog = [item['program'] for item in programs if 'variant' in item]
    if catalog and not (out / 'programs' / 'privileged-prepared.json').exists():
        from .supply import main as supply
        supply(['privileged', '--programs', str(out / 'programs'), '--names', *catalog,
                '--registry', str(Path(args.protection).resolve()), '--root', str(REPO)])
    structures = {item['program']: program_structure(item['path']) for item in programs}
    _write(out / 'structure.json', {'schema': 'natlang.program-adaptation-structure/1', 'programs': structures,
                                    'shapes': distinct_shapes(structures.values())})
    tuner = None if args.dry_run else Tuner(out, checkpoint, sha, device=args.device, tuner=parameters.get('tuner'),
                                            concurrency=parameters['concurrency'], base=args.base_snapshot)
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    work = _shard(programs, args.shard)
    timeout = _program_hours_timeout(parameters)
    # Entry: the untuned development arm of every program in this shard.
    for item in work:
        name, root = item['program'], item['path']
        if args.dry_run or 'untuned' in _arms(root, 'development'):
            continue
        status = tuner.run(root, '--evaluate', 'development', log=logs / f'{name}.entry.log', timeout=timeout)
        if status != 0:
            _write(out / 'tunes' / f'{name}.failed.json', {'program': name, 'step': 'entry', 'status': status})
        _log('untuned', program=name, status=status)
    entry = entry_state(out, programs, parameters, dry_run=args.dry_run)
    _write(out / 'entry.json', entry)
    if not entry['met'] and not args.dry_run and not args.pilot:
        _log('entry-not-met', **{key: entry[key] for key in ('reason', 'headroom_programs')})
        return finish(out, programs, parameters, entry, args, sha) if not args.shard else 3
    for item in work:
        name, root = item['program'], item['path']
        result_path = out / 'tunes' / f'{name}.json'
        untuned = _arms(root, 'development').get('untuned')
        if result_path.exists() or (out / 'tunes' / f'{name}.failed.json').exists():
            continue
        if untuned is not None and arm_summary(untuned)['success'] == arm_summary(untuned)['roots']:
            _write(out / 'tunes' / f'{name}.saturated.json', {'program': name, 'untuned': arm_summary(untuned)})
            continue
        started = time.time()
        if args.dry_run:
            status = subprocess.run([sys.executable, str(LANE / 'dry_run_fixture.py'), 'tune', '--program', root,
                                     '--checkpoint', str(checkpoint), '--out', str(result_path)],
                                    env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(REPO / 'training/neuralese'), str(REPO)])},
                                    cwd=REPO).returncode
        else:
            status = tuner.run(root, '--rounds', log=logs / f'{name}.tune.log', timeout=timeout)
            rounds_seconds = time.time() - started
            release = _best_release(root) if status == 0 else None
            if status == 0 and not release:
                status = 'no-selected-release'
            if status == 0:
                status = tuner.run(root, '--evaluate', 'development', '--release', release, log=logs / f'{name}.tune.log')
            if status == 0:
                from .learned.lane import tune_result
                tune_result(root, result_path, item.get('family'))
        if status != 0:
            _write(out / 'tunes' / f'{name}.failed.json', {'program': name, 'step': 'tune', 'status': status,
                                                           'seconds': time.time() - started})
        elif not args.dry_run:  # the gradient fit's wall time: the learned optimizer's matched-compute reference
            _write(out / 'tunes' / f'{name}.timing.json', {'program': name, 'seconds': rounds_seconds,
                                                           'scope': 'natlang tune --rounds wall time (collection, fits, '
                                                                    'rollouts, development selection)'})
        _log('tuned', program=name, status=status, seconds=round(time.time() - started, 1))
    # Confirmation (once) and distillation for every release with a measured development gain.
    for item in work:
        name, root = item['program'], item['path']
        outcome = program_outcome(out, item, parameters)
        if args.dry_run or not outcome.get('gain_met'):
            continue
        if not _arms(root, 'confirmation'):
            status = tuner.run(root, '--evaluate', 'confirmation', log=logs / f'{name}.confirmation.log')
            _log('confirmation', program=name, status=status)
        if parameters['distill'] and not (out / 'distill' / 'programs' / f'{name}.json').exists():
            distill_program(tuner, out, item, logs)
    if args.shard:
        _write(out / 'shards' / f'{args.shard[0]}-of-{args.shard[1]}.json',
               {'programs': [item['program'] for item in work], 'finished': time.time()})
        return 0
    return finish(out, programs, parameters, entry, args, sha)


def entry_state(out, programs, parameters, *, dry_run=False):
    arms = {}
    for item in programs:
        untuned = _arms(item['path'], 'development').get('untuned')
        if untuned is not None:
            arms[item['program']] = arm_summary(untuned)
    headroom = sorted(name for name, arm in arms.items() if 0 < arm['success'] < arm['roots'])
    required = parameters['entry_min_headroom_programs']
    met = len(headroom) >= required and not dry_run
    reason = ('dry run: nothing measured' if dry_run else
              None if met else f'{len(headroom)} of {len(arms)} measured programs have untuned development success '
                               f'strictly between 0 and all roots; entry needs {required}')
    return {'met': met, 'reason': reason, 'required': required, 'headroom_programs': headroom,
            'measured_programs': len(arms), 'untuned': arms}


def program_outcome(out, item, parameters):
    name, root = item['program'], item['path']
    untuned = _arms(root, 'development').get('untuned')
    result = _read(Path(out) / 'tunes' / f'{name}.json')
    record = {'program': name, 'family': item.get('family'),
              'untuned': arm_summary(untuned) if untuned else None,
              'tune_result': ({'path': str(Path(out) / 'tunes' / f'{name}.json'),
                               'sha256': sha256_file_hex(Path(out) / 'tunes' / f'{name}.json')} if result else None),
              'failed': _read(Path(out) / 'tunes' / f'{name}.failed.json'),
              'saturated': (Path(out) / 'tunes' / f'{name}.saturated.json').exists()}
    if not result or not untuned:
        return {**record, 'gain_met': False}
    release = result['best'].get('release')
    tuned = next((arm for arm in _arms(root, 'development').values()
                  if arm.get('release') == release and arm.get('arm') != 'untuned'), None)
    if tuned is None:
        return {**record, 'release': release, 'gain_met': False, 'reason': 'selected release has no development arm'}
    gain = paired_gain(untuned, tuned)
    confirmation = _arms(root, 'confirmation')
    confirmed = {arm_name: arm_summary(arm) for arm_name, arm in confirmation.items()}
    return {**record, 'release': release, 'tuned': arm_summary(tuned), 'gain': gain,
            'gain_met': gain.get('paired') is True and gain['gain'] >= parameters['min_development_gain'],
            'confirmation': confirmed or None,
            'confirmation_release_measured': any(arm.get('release') == release for arm in confirmation.values()),
            'compute': {'rounds': [(r.get('kind'), r.get('status'), r.get('development')) for r in result.get('rounds', [])]}}


def distill_program(tuner, out, item, logs):
    """Distill the active release's verified training-world executions (``natlang tune --distill``)."""
    name, root = item['program'], item['path']
    log = logs / f'{name}.distill.log'
    status = tuner.run(root, '--distill', '--json', log=log)
    release = _active_release(root)
    directory = Path(root) / '.natlang' / 'tune' / 'distill' / str(release)
    lowered = _read(directory / 'lowered.json')
    if status != 0 or lowered is None:
        _write(out / 'distill' / 'programs' / f'{name}.failed.json', {'program': name, 'status': status, 'release': release})
        return None
    graphs = directory / lowered['native_graphs']['path']
    if sha256_file_hex(graphs) != lowered['native_graphs']['sha256']:
        raise SystemExit(f'{graphs} differs from its lowering receipt')
    record = {'program': name, 'release': release, 'export': str(directory), 'manifest': _read(directory / 'manifest.json'),
              'native_graphs': {'path': str(graphs), 'sha256': lowered['native_graphs']['sha256'],
                                'rows': lowered['native_graphs']['rows']}}
    _write(out / 'distill' / 'programs' / f'{name}.json', record)
    return record


def build_distill_package(out, checkpoint_sha):
    """One training-split package of every confirmed release's distilled graphs (``distill/package``)."""
    out = Path(out)
    records = [json.loads(path.read_text()) for path in sorted((out / 'distill' / 'programs').glob('*.json'))
               if not path.name.endswith('.failed.json')]
    if not records:
        return None
    package = out / 'distill' / 'package'
    (package / 'train').mkdir(parents=True, exist_ok=True)
    (package / 'test').mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        for line in Path(record['native_graphs']['path']).read_text().splitlines():
            if line.strip():
                rows.append(line)
    train = package / 'train' / 'native-graphs.jsonl'
    train.write_text(''.join(line + '\n' for line in rows))
    (package / 'test' / 'native-graphs.jsonl').write_text('')
    receipt = {'schema': DISTILL_SCHEMA, 'checkpoint_sha256': checkpoint_sha,
               'programs': [{key: record[key] for key in ('program', 'release', 'native_graphs')} for record in records],
               'train': {'path': 'train/native-graphs.jsonl', 'sha256': sha256_file_hex(train), 'rows': len(rows)},
               'test': {'path': 'test/native-graphs.jsonl', 'rows': 0},
               'split': 'train only: oracle-correct reruns of training roots plus their admitted training rows; '
                        'development, confirmation and protected roots never enter',
               'actor': 'the bound backbone checkpoint serving each program\'s own tuned release',
               'training_admission': False,
               'note': 'conversion, not admission: admit through the registry and a recorded source continuation'}
    _write(package / 'receipt.json', receipt)
    return receipt


def finish(out, programs, parameters, entry, args, sha):
    """The stage report (gate receipt) and its result manifest."""
    outcomes = [program_outcome(out, item, parameters) for item in programs]
    structures = _read(Path(out) / 'structure.json')['programs']
    gained = [item for item in outcomes if item['gain_met']]
    shapes = sorted({structures[item['program']]['key'] for item in gained})
    missing = [item['program'] for item in gained if not item.get('confirmation_release_measured')]
    pending = [item['program'] for item in outcomes
               if not (item['tune_result'] or item['failed'] or item['saturated']) and entry['met']]
    minimum, minimum_shapes = parameters['exit_min_programs'], parameters['exit_min_structural_shapes']
    exit_met = (entry['met'] and not pending and len(gained) >= minimum and len(shapes) >= minimum_shapes
                and not missing and not args.dry_run and not args.pilot)
    reasons = ['pilot: a bounded machinery qualification, never a stage completion'] if args.pilot else []
    if args.dry_run:
        reasons.append('dry run: nothing measured')
    if len(gained) < minimum:
        reasons.append(f'{len(gained)} programs with a paired development gain >= {parameters["min_development_gain"]}; '
                       f'exit needs {minimum}')
    if len(shapes) < minimum_shapes:
        reasons.append(f'gains span {len(shapes)} structural shape(s) (all programs: '
                       f'{len({s["key"] for s in structures.values()})}); exit needs {minimum_shapes}')
    if missing:
        reasons.append('confirmation not yet run for ' + ', '.join(missing))
    if pending:
        reasons.append('programs not yet tuned: ' + ', '.join(pending))
    distill = build_distill_package(out, sha) if parameters['distill'] and not args.dry_run else None
    manifest = {'schema': MANIFEST_SCHEMA, 'checkpoint_sha256': sha,
                'programs': [{key: item.get(key) for key in ('program', 'family', 'release', 'gain_met', 'tune_result')}
                             | {'path': next(p['path'] for p in programs if p['program'] == item['program']),
                                'shape': structures[item['program']]['key']}
                             for item in outcomes],
                'distill': distill}
    _write(Path(out) / 'adaptation-manifest.json', manifest)
    report = {'schema': PROGRAM_ADAPTATION_REPORT, 'kind': PROGRAM_ADAPTATION,
              'completed': entry['met'] and not pending and not args.dry_run and not args.pilot,
              'scope': 'pilot' if args.pilot else 'stage',
              'checkpoint_sha256': sha, 'entry': entry,
              'exit': {'met': exit_met, 'reason': '; '.join(reasons) or None, 'min_programs': minimum,
                       'min_structural_shapes': minimum_shapes, 'gained_programs': len(gained),
                       'structural_shapes': len(shapes), 'gained': [item['program'] for item in gained],
                       'confirmations_missing': missing},
              'programs': outcomes, 'distill': distill, 'production_exposure_advanced': False,
              'dry_run': bool(args.dry_run),
              'note': 'measured by natlang tune on development roots; confirmation run once per selected release and '
                      'never used for selection; program adapters are not production exposure or a broad capability claim'}
    _write(Path(out) / 'report.json', report)
    _log('report', completed=report['completed'], entry=entry['met'], exit=exit_met, gained=len(gained), shapes=len(shapes))
    return 0


def register_distill(out, corpus_id, *, repo=REPO):
    """Stage a stage's distill package as a registry snapshot (``training_admission: false``) for ordinary review and
    a recorded source continuation; publishes its immutable manifest through ``scripts/sync_training_corpora.py``."""
    out = Path(out).resolve()
    receipt = _read(out / 'distill' / 'package' / 'receipt.json')
    if receipt is None or receipt.get('schema') != DISTILL_SCHEMA:
        raise SystemExit(f'{out}: no distill package; run the stage first')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*', corpus_id):
        raise SystemExit('corpus id: lowercase letters, digits, dots and dashes')
    relative = Path('runs') / 'program-adaptation-distill' / corpus_id
    target = Path(repo) / relative
    if not target.exists():
        import zstandard
        for part in ('train', 'test'):
            (target / part).mkdir(parents=True, exist_ok=True)
            source = out / 'distill' / 'package' / receipt[part]['path']
            with source.open('rb') as reader, (target / part / 'native-graphs.jsonl.zst').open('wb') as writer:
                zstandard.ZstdCompressor(level=10).copy_stream(reader, writer)
        shutil.copy2(out / 'distill' / 'package' / 'receipt.json', target / 'receipt.json')
    registry_path = Path(repo) / 'training' / 'neuralese_corpora.json'
    registry = json.loads(registry_path.read_text())
    entry = {'id': corpus_id, 'owner': 'dgx', 'path': str(relative), 'include': ['receipt.json', 'train/native-graphs.jsonl.zst',
                                                                               'test/native-graphs.jsonl.zst'],
             'source_class': 'own-policy-tuned-stack-oracle-verified-training-worlds',
             'split': receipt['split'], 'training_admission': False,
             'admission': f'{receipt["train"]["rows"]} distilled training-world rows from {len(receipt["programs"])} '
                          'confirmed program releases; admission is a separate reviewed source continuation',
             'license_provenance': 'project-generated own-policy executions of supplied programs with exact oracles',
             'note': 'Registration is availability, not optimizer exposure or capability lift.'}
    existing = next((item for item in registry['corpora'] if item['id'] == corpus_id), None)
    if existing is not None and existing != entry:
        raise SystemExit(f'{corpus_id} is already registered differently')
    if existing is None:
        registry['corpora'].append(entry)
        registry_path.write_text(json.dumps(registry, indent=1, ensure_ascii=False) + '\n')
    subprocess.run([sys.executable, str(Path(repo) / 'scripts' / 'sync_training_corpora.py'), 'publish', '--machine', 'dgx',
                    '--id', corpus_id], check=True, cwd=repo)
    return entry


def parse_parameters(args):
    parameters = dict(PROGRAM_ADAPTATION_DEFAULTS)
    for key in PROGRAM_ADAPTATION_DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            parameters[key] = value
    for key in ('program_names', 'tuner'):
        value = getattr(args, key)
        if value is not None:
            parameters[key] = json.loads(value)
    if args.program_hours is not None:
        parameters['program_hours'] = args.program_hours
    validate_parameters(PROGRAM_ADAPTATION, parameters)
    return parameters


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['register-distill']:
        parser = argparse.ArgumentParser(prog='tune.stage register-distill')
        parser.add_argument('--out', required=True)
        parser.add_argument('--id', required=True)
        args = parser.parse_args(argv[1:])
        print(json.dumps({'command': 'register-distill', **register_distill(args.out, args.id)}))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--programs', required=True)
    parser.add_argument('--protection', required=True)
    parser.add_argument('--continue-from', required=True)
    parser.add_argument('--parent-selection')
    parser.add_argument('--base-snapshot')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shard', help='I/N: run every N-th program from I (no report)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pilot', action='store_true',
                        help='bounded machinery qualification: tune the named programs whatever the entry measurement; '
                             'the report is never a completed stage')
    for key in ('entry_min_headroom_programs', 'exit_min_programs', 'exit_min_structural_shapes', 'min_development_gain',
                'concurrency'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('confirmation', 'distill'):
        parser.add_argument('--' + key.replace('_', '-'), dest=key, action='store_const', const=True)
        parser.add_argument('--no-' + key.replace('_', '-'), dest=key, action='store_const', const=False)
    parser.add_argument('--program-names')
    parser.add_argument('--tuner')
    parser.add_argument('--program-hours', type=float)
    args = parser.parse_args(argv)
    if args.shard:
        match = re.fullmatch(r'(\d+)/(\d+)', args.shard)
        if not match or not int(match.group(1)) < int(match.group(2)):
            parser.error('--shard is I/N with 0 <= I < N')
        args.shard = (int(match.group(1)), int(match.group(2)))
    return run(args, parse_parameters(args))


if __name__ == '__main__':
    raise SystemExit(main())
