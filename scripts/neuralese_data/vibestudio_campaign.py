#!/usr/bin/env python3
"""Run the full held-separated Vibestudio Luna queue, retaining evidence for every attempt."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from natlang_neuralese.common import data_registry
from natlang_neuralese.common.code_provenance import code_provenance

from live_teacher_sources import load_episode
from observed_graph_metrics import graph_metrics, aggregate_graph_metrics
from retained_state_archive import prepare_retained_state_packs
from sft_sources import ZstdWriter, held_closure, held_hit, live_holdout_registry, require_registered_open_goal


LUNA = 'openai-codex:gpt-6-luna'
DEFAULT_TEACHER_TIMEOUT_MS = 3_600_000


def read(path):
    return json.loads(path.read_text())


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def publish_immutable(path, value):
    """A resumed controller may reuse an identical record, never replace its history."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f'immutable campaign record differs: {path}')
        return
    publish(path, value)


def verified_episode(path, held, training):
    episode = load_episode(path)
    task = episode['task']
    interaction = task.get('interaction')
    if episode['teacher']['model_id'] != LUNA or not interaction:
        raise ValueError('campaign requires an actual Luna open-goal trajectory')
    if held_hit(task['id'], task['source_ids'], task['source_groups'], held):
        raise ValueError('capture overlaps the held closure')
    require_registered_open_goal({'task_id': task['id'], 'source_groups': task['source_groups'],
                                  'open_goal': interaction['goal']}, training)
    return episode


def campaign_scope(catalog, held_file):
    held = held_closure(held_file)
    training = live_holdout_registry(catalog, held)
    rows = [json.loads(line) for line in catalog.open() if line.strip()]
    cases = {row['scenario']: row for row in rows if row.get('taskId') in training}
    for row in cases.values():
        task_id = row['taskId']
        if held_hit(task_id, [task_id], row['sourceGroups'], held):
            raise ValueError(f'catalog training goal overlaps held: {task_id}')
    if len(cases) != len(training):
        raise ValueError('catalog does not map each training task to a distinct scenario')
    return held, training, cases


def capture_integrity_errors(errors):
    # Older producers placed a faithfully recorded model failure in captureErrors.
    # Keep that original diagnostic, but do not treat it as lost/corrupted evidence.
    return [error for error in (errors or [])
            if error != 'Assigned authored root did not complete successfully']


def outcomes(capture, names, held, training, cases):
    results = {}
    for name in names:
        directory = capture if len(names) == 1 else capture / 'cases' / name
        manifest = directory / 'verified/manifest.json'
        raw = next(directory.glob(f'st_*/trajectory-{name}-full.json.gz'), None)
        if raw is None:
            raw = next(directory.glob(f'st_*/failure-evidence/trajectory-{name}-full.json.gz'), None)
        packet = None
        capture_errors = None
        if raw:
            with gzip.open(raw, 'rt') as stream:
                packet = json.load(stream)
            capture_errors = (((packet.get('execution') or {}).get('diagnostics') or {}).get('natlangTeacher') or {}).get('captureErrors')
        if manifest.is_file():
            episode = verified_episode(manifest, held, training)
            if episode['task_id'] != cases[name]['taskId']:
                raise ValueError(f'verified capture identifies another task: {manifest}')
            results[name] = {'status': 'verified', 'teacher_started': True, 'manifest': str(manifest.relative_to(capture)),
                             'decisions': len(episode['turns'])}
        elif raw:
            test = packet.get('result') or {}
            entry = packet.get('execution') or {}
            teacher = (entry.get('diagnostics') or {}).get('natlangTeacher') or {}
            transport = (teacher.get('executionError') or {}).get('message')
            reason = transport or entry.get('error') or test.get('reason')
            results[name] = {'status': 'export-rejected' if test.get('passed') else 'test-failed',
                             'raw': str(raw.relative_to(capture)), 'reason': reason,
                             'teacher_started': True if (teacher.get('rootCallIds') or teacher.get('calls'))
                             and not capture_integrity_errors(teacher.get('captureErrors')) else None}
        else:
            results[name] = {'status': 'no-capture', 'teacher_started': None}
        if capture_errors:
            results[name]['capture_errors'] = capture_errors
    return results


def workerd_failure(outcome):
    reason = outcome.get('reason') or ''
    return outcome['status'] == 'test-failed' and (
        ('DO RPC ' in reason and 'ECONNRESET' in reason)
        or 'eval: status reconciliation failed for accepted run ' in reason
    )


def infrastructure_failure(outcome, case):
    if capture_integrity_errors(outcome.get('capture_errors')):
        return True
    if workerd_failure(outcome):
        return True
    if outcome['status'] == 'no-capture':
        return True
    reason = outcome.get('reason') or ''
    if outcome.get('teacher_started') is False and (
            '[build.getBuild]' in reason or 'No export ' in reason or
            'Could not resolve ' in reason or 'Cannot find package ' in reason):
        return True
    return case.get('requiresAndroidEmulator') and 'ready disposable Android emulator' in reason


def wave_infrastructure_hold(receipt, cases, threshold=2):
    if threshold < 0:
        raise ValueError('infrastructure hold threshold must be nonnegative')
    failures = [name for name, outcome in receipt.items() if infrastructure_failure(outcome, cases.get(name, {}))]
    if not threshold or len(failures) < threshold:
        return None
    return {'infrastructure_failures': failures, 'infrastructure_hold_threshold': threshold,
            'workerd_resets': [name for name, outcome in receipt.items() if workerd_failure(outcome)],
            'all_captures_missing': bool(receipt) and all(outcome['status'] == 'no-capture' for outcome in receipt.values())}


def prerequisite_holds(cases, host):
    holds = {}
    if any(case.get('requiresAndroidEmulator') for case in cases.values()):
        adb = shutil.which('adb')
        devices = subprocess.run([adb, 'devices'], capture_output=True, text=True, check=False) if adb else None
        emulators = []
        if devices is not None and devices.returncode == 0:
            for line in devices.stdout.splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[0].startswith('emulator-') and fields[1] == 'device':
                    emulators.append(fields[0])
        booted = subprocess.run([adb, '-s', emulators[0], 'shell', 'getprop', 'sys.boot_completed'],
                                capture_output=True, text=True, check=False) if len(emulators) == 1 else None
        if booted is None or booted.returncode or booted.stdout.strip() != '1':
            holds.update({name: 'requires one booted disposable Android emulator'
                          for name, case in cases.items() if case.get('requiresAndroidEmulator')})
    if any(case.get('requiresDeploymentPackage') for case in cases.values()):
        node = shutil.which('node')
        validation = subprocess.run([node, '--input-type=module', '-e', '''
            import path from "node:path";
            import { requireInstalledDeploymentPackage } from "./scripts/lib/owned-deployment-target.mjs";
            const input = process.env.VIBESTUDIO_MANAGED_DEPLOYMENT_PACKAGE?.trim();
            if (!input) process.exit(2);
            await requireInstalledDeploymentPackage(path.resolve(input));
        '''], cwd=host, capture_output=True, check=False) if node else None
        if validation is None or validation.returncode:
            holds.update({name: 'requires a validated installed deployment package'
                          for name, case in cases.items() if case.get('requiresDeploymentPackage')})
    return holds


def transfer(source, remote, destination):
    subprocess.run(['ssh', '-o', 'BatchMode=yes', remote, shlex.join(['mkdir', '-p', str(destination)])], check=True)
    target = f'{remote}:{destination}/'
    exclusions = prepare_retained_state_packs(source)
    common = ['rsync', '-a', '--protect-args', *[f'--exclude={name}' for name in exclusions]]
    flags = [*common, f'{source}/', target]
    subprocess.run(flags, check=True)
    comparison = subprocess.check_output([*common, '--dry-run', '--checksum', '--itemize-changes',
                                          '--delete', f'{source}/', target], text=True)
    if comparison.strip():
        raise RuntimeError(f'archive differs after transfer: {comparison[:400]}')


def remote_archive(args, destination, receipt, capture, *, wave=None):
    root = args.root.resolve()
    transfer(capture, args.archive_host, destination / 'capture')
    metadata = root / 'waves' / f'{destination.name}-archive'
    metadata.mkdir(exist_ok=True)
    publish_immutable(metadata / 'receipt.json', receipt)
    records = [metadata / 'receipt.json']
    if wave is not None:
        publish_immutable(metadata / 'wave.json', wave)
        records.append(metadata / 'wave.json')
    subprocess.run(['rsync', '-a', '--protect-args', *map(str, records),
                    f'{args.archive_host}:{destination}/'], check=True)
    if args.defer_export:
        return
    repo = args.archive_repo
    remote_command = ['env', f'PYTHONPATH={repo}/scripts/neuralese_data:{repo}/training/neuralese:{repo}',
                      'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', str(repo / '.venv-neuralese/bin/python'),
                      str(args.archive / 'vibestudio_campaign.py'), str(args.archive), '--held', str(args.archive_held),
                      'export-wave', '--archive', str(destination), '--receipt', str(destination / 'receipt.json')]
    unit_prefix = f'{root.name}-{destination.name}-export'.replace('_', '-').lower()
    unit = f'{unit_prefix}-{time.time_ns()}'
    ledger = ['python3', str(repo / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', '1.5',
              '--reserve-gb', '15', '--wait', '1200', '--detach', '--resumable', '--priority', 'low',
              '--workdir', str(repo), '--']
    check = ['test', '-f', str(destination / 'exported.json')]
    ssh = ['ssh', '-o', 'BatchMode=yes', args.archive_host]
    if subprocess.run([*ssh, shlex.join(check)]).returncode:
        subprocess.run([*ssh, shlex.join([*ledger, *remote_command])], check=True)
        for _ in range(130):
            if not subprocess.run([*ssh, shlex.join(check)]).returncode:
                break
            time.sleep(10)
        else:
            raise RuntimeError(f'remote export did not complete for {destination}')


def sync_index(args, paths):
    subprocess.run(['rsync', '-a', '--protect-args', *map(str, paths),
                    f'{args.archive_host}:{args.archive}/'], check=True)


def export_trajectories(archive, receipt, held, training):
    # Every direct exporter and archive drainer owns the same cohort lock through
    # registration and publication, including the initial completed-marker check.
    with (archive / '.export.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX)
        return _export_trajectories(archive, receipt, held, training)


def write_graph_telemetry(archive, receipt, admitted=None):
    """Describe retained attempts without admitting, rejecting or rewriting any capture."""
    captures = []
    for name, result in receipt.items():
        outcome = (admitted or {}).get(name) or {}
        category = ('partial' if outcome.get('status') == 'partial' else 'completed') if result['status'] == 'verified' else 'preserved_failure'
        metrics = outcome.get('graph_metrics')
        source, error = None, None
        if result.get('manifest'):
            manifest = archive / 'capture' / result['manifest']
            if manifest.is_file():
                source = {'path': str(manifest.relative_to(archive)), 'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
                if not outcome:
                    try:
                        spec = read(manifest)
                        verification = read(manifest.parent / spec['verification']['path'])
                        if verification.get('status') == 'partial':
                            category = 'partial'
                    except (OSError, ValueError, KeyError, TypeError) as failure:
                        error = str(failure)
        if metrics is None:
            raw = archive / 'capture' / result['raw'] if result.get('raw') else None
            if raw is None and result.get('manifest'):
                directory = (archive / 'capture' / result['manifest']).parent.parent
                raw = next(directory.glob(f'st_*/trajectory-{name}-full.json.gz'), None)
                if raw is None:
                    raw = next(directory.glob(f'st_*/failure-evidence/trajectory-{name}-full.json.gz'), None)
            records = []
            if raw is not None and raw.is_file():
                source = {'path': str(raw.relative_to(archive)), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}
                try:
                    with gzip.open(raw, 'rt') as stream:
                        packet = json.load(stream)
                    teacher = ((packet.get('execution') or {}).get('diagnostics') or {}).get('natlangTeacher') or {}
                    records = [call['record'] for call in teacher.get('calls') or [] if isinstance(call.get('record'), dict)]
                except (OSError, ValueError, TypeError) as failure:
                    error = str(failure)
            metrics = graph_metrics(records)
        captures.append({'case': name, 'status': result['status'], 'category': category, 'metrics': metrics,
                         **({'source': source} if source else {}), **({'telemetry_error': error} if error else {})})
    summary = aggregate_graph_metrics(captures)
    publish(archive / 'graph-metrics.json', {**summary, 'captures': captures})
    return summary


def _export_trajectories(archive, receipt, held, training):
    final = archive / 'trajectory.native.jsonl.zst'
    marker = archive / 'exported.json'
    if marker.exists():
        if not (archive / 'graph-metrics.json').exists():
            write_graph_telemetry(archive, receipt)
        return
    admitted = {}
    output = ZstdWriter(final)
    counts = {'completed_goals': 0, 'partial_trajectories': 0, 'supervised_decisions': 0, 'evidence_only': 0}
    try:
        for name, result in receipt.items():
            if result['status'] == 'verified':
                episode = verified_episode(archive / 'capture' / result['manifest'], held, training)
                outcome = episode['trajectory_ir']['outcome']
                admitted[name] = outcome
                counts['completed_goals'] += outcome.get('task_completed', outcome['status'] == 'passed') is True
                counts['partial_trajectories'] += outcome['status'] == 'partial'
                counts['supervised_decisions'] += len(episode['turns'])
                counts['evidence_only'] += not episode['turns']
                output.write(episode['trajectory_ir'])
    except BaseException:
        output.close(publish=False)
        raise
    output.close()
    graph_summary = write_graph_telemetry(archive, receipt, admitted)
    dataset = None
    if output.rows:
        dataset = data_registry.record_dataset(f'vibestudio-{archive.parent.name}-{archive.name}'.lower(), archive,
                                               kind='teacher-demos', format='native Vibestudio Luna trajectories and capture evidence',
                                               include=['capture/**', 'receipt.json', final.name, 'graph-metrics.json',
                                                        *(['wave.json'] if (archive / 'wave.json').is_file() else [])], rows=output.rows,
                                               extra={'trajectory_counts': counts, 'observed_graph_metrics': graph_summary})
    publish(marker, {'verified': output.rows, 'dataset': dataset['id'] if dataset else None, **counts})


def publish_status(root, cases, seeds, receipts, review_model, attempts, holds, *,
                   attempt_epoch=None, receipt_epochs=None):
    verified, counts, cumulative = set(seeds), Counter(), Counter()
    states, epoch_states = {}, {}
    infrastructure, epoch_infrastructure = Counter(), Counter()
    epochs = [None] * len(receipts) if receipt_epochs is None else receipt_epochs
    if len(epochs) != len(receipts):
        raise ValueError('each immutable receipt must have its original wave epoch')
    for receipt, epoch in zip(receipts, epochs):
        for name, outcome in receipt.items():
            if name not in cases:
                continue  # Retired catalog goals remain in their immutable archived receipts.
            states[name] = outcome['status']
            current = epoch == attempt_epoch
            if current:
                epoch_states[name] = outcome['status']
            if infrastructure_failure(outcome, cases[name]):
                infrastructure[name] += 1
                if current:
                    epoch_infrastructure[name] += 1
                continue
            cumulative[name] += 1
            if current:
                counts[name] += 1
            if outcome['status'] == 'verified':
                verified.add(name)
    rows = {}
    for name, case in cases.items():
        if name in verified:
            status = 'verified'
        elif name in holds:
            status = 'prerequisite-unavailable'
        elif case.get('requiresReview') and not review_model:
            status = 'reviewer-required'
        elif counts[name] < attempts:
            status = 'pending'
        else:
            status = epoch_states[name]
        rows[name] = {'status': status, 'attempts': counts[name], 'attempt_epoch': attempt_epoch,
                      'cumulative_attempts': cumulative[name] + infrastructure[name],
                      'cumulative_teacher_attempts': cumulative[name],
                      'last_outcome': states.get(name), 'epoch_last_outcome': epoch_states.get(name)}
        if name in holds and status == 'prerequisite-unavailable':
            rows[name]['hold_reason'] = holds[name]
        if infrastructure[name]:
            rows[name]['infrastructure_failures'] = infrastructure[name]
        rows[name]['epoch_infrastructure_failures'] = epoch_infrastructure[name]
    summary = {'total': len(cases), 'attempted_goals': sum(cumulative[name] + infrastructure[name] > 0 for name in cases),
               'attempts': sum(cumulative.values()) + sum(infrastructure.values()),
               'attempt_epoch': attempt_epoch,
               'epoch_attempted_goals': len(set(epoch_states)),
               'epoch_attempts': sum(counts.values()) + sum(epoch_infrastructure.values()),
               'epoch_teacher_attempts': sum(counts.values()),
               'infrastructure_failures': sum(infrastructure.values()),
               'counts': dict(Counter(row['status'] for row in rows.values()))}
    publish(root / 'status.json', {'model': LUNA, **summary, 'cases': rows})
    print(json.dumps(summary), flush=True)
    return rows


def next_wave(cases, rows, size):
    pending = [name for name, row in rows.items() if row['status'] == 'pending']
    if not pending:
        return []
    attempts = Counter()
    for name, row in rows.items():
        attempts[cases[name]['workspace']] += row['attempts']
    first = min(pending, key=lambda name: (rows[name]['attempts'], attempts[cases[name]['workspace']]))
    scope = cases[first]
    families = {}
    for name in pending:
        case = cases[name]
        if (rows[name]['attempts'] == rows[first]['attempts'] and case['workspace'] == scope['workspace'] and
                case.get('requiresReview') == scope.get('requiresReview')):
            families.setdefault(case['family'], []).append(name)
    names = []
    for offset in range(size):
        names.extend(family[offset] for family in families.values() if len(family) > offset)
        if len(names) >= size:
            break
    return names[:size]


def collection_environment(release_pins):
    return release_environment(read(release_pins)['release'])


def release_environment(release):
    manifest = Path(release['manifest']).resolve(strict=True)
    if manifest.name != 'release.json' or manifest.parent.name != 'userland-builds':
        raise ValueError(f'invalid workspace release manifest path: {manifest}')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != release['sha256']:
        raise ValueError(f'workspace release manifest does not match its pin: {manifest}')
    # VibeStudio verifies the complete release and its build identity at startup.
    return {**os.environ, 'VIBESTUDIO_WORKSPACE_RELEASE_ROOT': str(manifest.parent.parent)}


def receipt_history(waves, jobs):
    """Legacy receipts keep their historical epoch; their content is never decorated."""
    receipts, epochs = [], []
    for path in sorted(waves.glob('*-result.json')):
        receipts.append(read(path))
        job = jobs / f'{path.name.removesuffix("-result.json")}.json'
        epochs.append(read(job).get('attempt_epoch') if job.exists() else None)
    return receipts, epochs


def attempt_context(args, catalog):
    # Resolve once per controller: changing a pin file cannot relabel an admitted wave.
    pin_bytes = args.release_pins.read_bytes()
    pins = json.loads(pin_bytes)
    environment = release_environment(pins['release'])
    provenance = {'release': pins['release'], 'release_pins_sha256': hashlib.sha256(pin_bytes).hexdigest(),
                  'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(),
                  'model': LUNA, 'driver': code_provenance(), 'host': code_provenance(args.host)}
    epoch = args.attempt_epoch or pins['release']['sha256']
    return epoch, provenance, environment


def unfinished_wave(plans, waves, attempt_epoch, provenance, rows=None):
    """Archive captured historical work; run unstarted work only on its admitted release."""
    for plan in plans:
        if (waves / f'{plan.stem}-result.json').exists():
            continue
        if (waves / f'{plan.stem}-capture').is_dir():
            return plan
        record = read(plan)
        execution = waves / f'{plan.stem}-execution.json'
        if rows is not None and execution.exists() and any(
                rows.get(name, {}).get('status') != 'pending' for name in read(execution)['cases']):
            continue  # Its admitted case list is immutable; schedule a fresh wave for any remainder.
        if record.get('attempt_epoch') == attempt_epoch and record.get('provenance') == provenance:
            return plan
    return None


def drive(args):
    root = args.root.resolve()
    held, training, cases = campaign_scope(root / 'catalog.jsonl', args.held)
    attempt_epoch, provenance, environment = attempt_context(args, root / 'catalog.jsonl')
    seeds = {}
    by_id = {row['taskId']: name for name, row in cases.items()}
    saved_seeds = root / 'reused.json'
    paths = list(map(Path, read(saved_seeds).values())) if saved_seeds.exists() else []
    for path in [*paths, *args.seed]:
        episode = verified_episode(path, held, training)
        if episode['trajectory_ir']['outcome'].get('task_completed') is not True:
            raise ValueError('partial training evidence cannot complete a campaign goal')
        seeds[by_id[episode['task_id']]] = str(path.resolve())
    publish(saved_seeds, seeds)
    waves = root / 'waves'
    jobs = waves / 'jobs'
    jobs.mkdir(exist_ok=True)
    initial = root / 'initial-wave.txt'
    starter = jobs / '0000.json'
    if initial.is_file() and not starter.exists():
        publish_immutable(starter, {'cases': initial.read_text().splitlines(),
                                  'attempt_epoch': attempt_epoch, 'provenance': provenance})
    for index, (name, path) in enumerate(seeds.items()):
        source = Path(path).parent
        remote_archive(args, args.archive / f'seed-{index:04d}',
                       {name: {'status': 'verified', 'manifest': 'manifest.json'}}, source)
    holds = prerequisite_holds(cases, args.host)
    completed_waves = 0
    while True:
        plans = sorted(jobs.glob('[0-9]*.json'))
        receipts, epochs = receipt_history(waves, jobs)
        rows = publish_status(root, cases, seeds, receipts, args.review_model, args.attempts, holds,
                              attempt_epoch=attempt_epoch, receipt_epochs=epochs)
        sync_index(args, [root / 'status.json', root / 'reused.json'])
        if (root / 'stop-after-wave').exists() or completed_waves == args.max_waves:
            return
        started = unfinished_wave(plans, waves, attempt_epoch, provenance, rows)
        if not started:
            names = next_wave(cases, rows, args.wave_size)
            if not names:
                if any(row['status'] in ('prerequisite-unavailable', 'reviewer-required') for row in rows.values()):
                    return
                publish(root / 'finished.json', {'collection_exhausted': True, 'attempt_epoch': attempt_epoch,
                                                 'all_goals_verified': all(row['status'] == 'verified' for row in rows.values())})
                sync_index(args, [root / 'finished.json'])
                return
            started = jobs / f'{int(plans[-1].stem) + 1 if plans else 0:04d}.json'
            publish_immutable(started, {'cases': names, 'attempt_epoch': attempt_epoch, 'provenance': provenance})
        plan = read(started)
        execution = waves / f'{started.stem}-execution.json'
        names = read(execution)['cases'] if execution.exists() else plan['cases']
        first = cases[names[0]]
        capture = waves / f'{started.stem}-capture'
        if not capture.is_dir():
            names = [name for name in names if rows[name]['status'] == 'pending']
            if not names:
                publish_immutable(waves / f'{started.stem}-result.json', {})
                continue
            first = cases[names[0]]
            publish_immutable(execution, {'cases': names, 'attempt_epoch': plan.get('attempt_epoch'),
                                          'provenance': plan.get('provenance'), 'teacher_timeout_ms': args.teacher_timeout_ms})
            if shutil.disk_usage(root).free < args.reserve_disk_gib * 1024**3:
                raise RuntimeError('insufficient local disk headroom for another managed collection wave')
            collect = ['pnpm', '--silent', 'natlang:teacher', '--model', LUNA, '--approve-codex-session',
                       '--cases-per-instance', str(args.wave_size), '--test-timeout-ms', str(args.teacher_timeout_ms), '--out', str(capture)]
            if args.concurrency > 1:
                collect += ['--concurrency', str(args.concurrency)]
            if first.get('requiresReview'):
                collect += ['--review-model', args.review_model]
            for name in names:
                collect += ['--case', name]
            with (waves / f'{started.stem}-collector.log').open('ab') as log:
                completed = subprocess.run(collect, cwd=args.host, env=environment, stdout=log, stderr=subprocess.STDOUT)
            publish(waves / f'{started.stem}-exit.json', {'status': completed.returncode})
            if not capture.is_dir():
                raise RuntimeError(f'collector failed before creating {capture}; see the collector log')

        receipt = outcomes(capture, names, held, training, cases)
        destination = args.archive / started.stem
        remote_archive(args, destination, receipt, capture,
                       wave={'plan': plan, 'execution': read(execution) if execution.exists() else None})
        publish_immutable(waves / f'{started.stem}-result.json', receipt)
        completed_waves += 1
        # The remote capture and receipt are durable. Deferred exports read those copies.
        shutil.rmtree(capture)
        runtime_hold = wave_infrastructure_hold(receipt, cases, getattr(args, 'infrastructure_hold_threshold', 2))
        if runtime_hold is not None:
            publish(root / 'runtime-hold.json', {'wave': started.stem, **runtime_hold})
            (root / 'stop-after-wave').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--held', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('drive')
    run.add_argument('--host', type=Path, required=True)
    run.add_argument('--release-pins', type=Path, required=True,
                     help='JSON pin containing the completed workspace release manifest path and SHA-256')
    run.add_argument('--archive-host', required=True)
    run.add_argument('--archive', type=Path, required=True)
    run.add_argument('--archive-repo', type=Path, required=True)
    run.add_argument('--archive-held', type=Path, required=True)
    run.add_argument('--defer-export', action='store_true',
                     help='archive captures and receipts for a separate export worker')
    run.add_argument('--seed', action='append', type=Path, default=[])
    run.add_argument('--review-model')
    run.add_argument('--attempts', type=int, default=2, help='teacher attempts per goal in the current release epoch')
    run.add_argument('--attempt-epoch', help='explicit retry epoch; defaults to the pinned release manifest SHA-256')
    run.add_argument('--infrastructure-hold-threshold', type=int, default=2,
                     help='hold after this many infrastructure failures in one wave; 0 disables the hold')
    run.add_argument('--wave-size', type=int, default=6)
    run.add_argument('--max-waves', type=int, help='stop after archiving this many waves and syncing status')
    run.add_argument('--concurrency', type=int, default=1)
    run.add_argument('--teacher-timeout-ms', type=int, default=DEFAULT_TEACHER_TIMEOUT_MS,
                     help='per-teacher episode wall time, including native preparation; default 3600000ms; explicit override leaves normal test budgets unchanged')
    run.add_argument('--reserve-disk-gib', type=float, default=8)
    export = sub.add_parser('export-wave')
    export.add_argument('--archive', type=Path, required=True)
    export.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export-wave':
        held, training, _ = campaign_scope(args.root / 'catalog.jsonl', args.held)
        export_trajectories(args.archive, read(args.receipt), held, training)
        return
    reviewer = (args.review_model or '').split(':')[-1].lower().split('-')
    if (args.teacher_timeout_ms < 1 or args.attempts < 1 or args.infrastructure_hold_threshold < 0 or (args.attempt_epoch is not None and not args.attempt_epoch.strip())
            or (args.max_waves is not None and args.max_waves < 1)
            or not 1 <= args.concurrency <= args.wave_size or 'sol' in reviewer or 'luna' in reviewer):
        parser.error('provide positive limits, concurrency within the wave size, and an independent non-Sol reviewer when configured')
    with (args.root / '.controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive(args)


if __name__ == '__main__':
    main()
