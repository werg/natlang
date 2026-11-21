#!/usr/bin/env python3
"""Run the full held-separated Vibestudio Luna queue, retaining evidence for every attempt."""
import argparse
from collections import Counter
import fcntl
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from natlang_neuralese.common import data_registry

from live_teacher_sources import load_episode
from sft_sources import ZstdWriter, held_closure, held_hit, live_holdout_registry, require_registered_open_goal


LUNA = 'openai-codex:gpt-6-luna'


def read(path):
    return json.loads(path.read_text())


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


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


def outcomes(capture, names, held, training, cases):
    results = {}
    for name in names:
        directory = capture if len(names) == 1 else capture / 'cases' / name
        manifest = directory / 'verified/manifest.json'
        raw = next(directory.glob(f'st_*/trajectory-{name}-full.json.gz'), None)
        if raw is None:
            raw = next(directory.glob(f'st_*/failure-evidence/trajectory-{name}-full.json.gz'), None)
        if manifest.is_file():
            episode = verified_episode(manifest, held, training)
            if episode['task_id'] != cases[name]['taskId']:
                raise ValueError(f'verified capture identifies another task: {manifest}')
            results[name] = {'status': 'verified', 'manifest': str(manifest.relative_to(capture)),
                             'decisions': len(episode['turns'])}
        elif raw:
            with gzip.open(raw, 'rt') as stream:
                packet = json.load(stream)
            test = packet.get('result') or {}
            entry = packet.get('execution') or {}
            teacher = (entry.get('diagnostics') or {}).get('natlangTeacher') or {}
            transport = (teacher.get('executionError') or {}).get('message')
            results[name] = {'status': 'export-rejected' if test.get('passed') else 'test-failed',
                             'raw': str(raw.relative_to(capture)), 'reason': transport or entry.get('error') or test.get('reason')}
        else:
            results[name] = {'status': 'no-capture'}
    return results


def transfer(source, remote, destination):
    subprocess.run(['ssh', '-o', 'BatchMode=yes', remote, shlex.join(['mkdir', '-p', str(destination)])], check=True)
    target = f'{remote}:{destination}/'
    flags = ['rsync', '-a', '--protect-args', f'{source}/', target]
    subprocess.run(flags, check=True)
    comparison = subprocess.check_output([*flags[:3], '--dry-run', '--checksum', '--itemize-changes',
                                          '--delete', *flags[3:]], text=True)
    if comparison.strip():
        raise RuntimeError(f'archive differs after transfer: {comparison[:400]}')


def remote_archive(args, destination, receipt, capture):
    root = args.root.resolve()
    transfer(capture, args.archive_host, destination / 'capture')
    metadata = root / 'waves' / f'{destination.name}-archive'
    metadata.mkdir(exist_ok=True)
    publish(metadata / 'receipt.json', receipt)
    target = f'{args.archive_host}:{destination}/receipt.json'
    subprocess.run(['rsync', '-a', '--protect-args', str(metadata / 'receipt.json'), target], check=True)
    repo = args.archive_repo
    remote_command = ['env', f'PYTHONPATH={repo}/scripts/neuralese_data:{repo}/training/neuralese:{repo}',
                      'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', str(repo / '.venv-neuralese/bin/python'),
                      str(args.archive / 'vibestudio_campaign.py'), str(args.archive), '--held', str(args.archive_held),
                      'export-wave', '--archive', str(destination), '--receipt', str(destination / 'receipt.json')]
    unit_prefix = f'{root.name}-{destination.name}-export'.replace('_', '-').lower()
    unit = f'{unit_prefix}-{time.time_ns()}'
    ledger = ['python3', str(repo / 'scripts/memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', '1.5',
              '--reserve-gb', '15', '--wait', '1200', '--detach', '--no-resumable', '--priority', 'low',
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
    final = archive / 'trajectory.native.jsonl.zst'
    marker = archive / 'exported.json'
    if marker.exists():
        return
    output = ZstdWriter(final)
    try:
        for result in receipt.values():
            if result['status'] == 'verified':
                episode = verified_episode(archive / 'capture' / result['manifest'], held, training)
                output.write(episode['trajectory_ir'])
    except BaseException:
        output.close(publish=False)
        raise
    output.close()
    dataset = None
    if output.rows:
        dataset = data_registry.record_dataset(f'vibestudio-{archive.parent.name}-{archive.name}'.lower(), archive,
                                               kind='teacher-demos', format='native Vibestudio Luna trajectories and capture evidence',
                                               include=['capture/**', 'receipt.json', final.name], rows=output.rows)
    publish(marker, {'verified': output.rows, 'dataset': dataset['id'] if dataset else None})


def publish_status(root, cases, seeds, receipts, review_model, attempts):
    verified, counts = set(seeds), Counter()
    states = {}
    for receipt in receipts:
        for name, outcome in receipt.items():
            counts[name] += 1
            states[name] = outcome['status']
            if outcome['status'] == 'verified':
                verified.add(name)
    rows = {}
    for name, case in cases.items():
        if name in verified:
            status = 'verified'
        elif case.get('requiresReview') and not review_model:
            status = 'reviewer-required'
        elif counts[name] < attempts:
            status = 'pending'
        else:
            status = states[name]
        rows[name] = {'status': status, 'attempts': counts[name], 'last_outcome': states.get(name)}
    summary = {'total': len(cases), 'attempted_goals': len(counts), 'attempts': sum(counts.values()),
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


def drive(args):
    root = args.root.resolve()
    held, training, cases = campaign_scope(root / 'catalog.jsonl', args.held)
    seeds = {}
    by_id = {row['taskId']: name for name, row in cases.items()}
    for path in args.seed:
        episode = verified_episode(path, held, training)
        seeds[by_id[episode['task_id']]] = str(path.resolve())
    publish(root / 'reused.json', seeds)
    waves = root / 'waves'
    jobs = waves / 'jobs'
    jobs.mkdir(exist_ok=True)
    initial = root / 'initial-wave.txt'
    starter = jobs / '0000.json'
    if initial.is_file() and not starter.exists():
        publish(starter, {'cases': initial.read_text().splitlines()})
    for index, (name, path) in enumerate(seeds.items()):
        source = Path(path).parent
        remote_archive(args, args.archive / f'seed-{index:04d}',
                       {name: {'status': 'verified', 'manifest': 'manifest.json'}}, source)
    while True:
        plans = sorted(jobs.glob('[0-9]*.json'))
        receipts = [read(wave) for wave in sorted(waves.glob('*-result.json'))]
        rows = publish_status(root, cases, seeds, receipts, args.review_model, args.attempts)
        sync_index(args, [root / 'status.json', root / 'reused.json'])
        if (root / 'stop-after-wave').exists():
            return
        started = next((plan for plan in plans if not (waves / f'{plan.stem}-result.json').exists()), None)
        if not started:
            names = next_wave(cases, rows, args.wave_size)
            if not names:
                publish(root / 'finished.json', {'collection_exhausted': True,
                                                 'all_goals_verified': all(row['status'] == 'verified' for row in rows.values())})
                sync_index(args, [root / 'finished.json'])
                return
            started = jobs / f'{int(plans[-1].stem) + 1 if plans else 0:04d}.json'
            publish(started, {'cases': names})
        names = read(started)['cases']
        first = cases[names[0]]
        capture = waves / f'{started.stem}-capture'
        if not capture.is_dir():
            if shutil.disk_usage(root).free < args.reserve_disk_gib * 1024**3:
                raise RuntimeError('insufficient local disk headroom for another managed collection wave')
            collect = ['pnpm', '--silent', 'natlang:teacher', '--model', LUNA, '--approve-codex-session',
                       '--cases-per-instance', str(args.wave_size), '--test-timeout-ms', '600000', '--out', str(capture)]
            if args.concurrency > 1:
                collect += ['--concurrency', str(args.concurrency)]
            if first.get('requiresReview'):
                collect += ['--review-model', args.review_model]
            for name in names:
                collect += ['--case', name]
            with (waves / f'{started.stem}-collector.log').open('ab') as log:
                completed = subprocess.run(collect, cwd=args.host, stdout=log, stderr=subprocess.STDOUT)
            publish(waves / f'{started.stem}-exit.json', {'status': completed.returncode})
            if not capture.is_dir():
                raise RuntimeError(f'collector failed before creating {capture}; see the collector log')

        receipt = outcomes(capture, names, held, training, cases)
        destination = args.archive / started.stem
        remote_archive(args, destination, receipt, capture)
        publish(waves / f'{started.stem}-result.json', receipt)
        # The matching remote copy and its IR export are durable.
        shutil.rmtree(capture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--held', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('drive')
    run.add_argument('--host', type=Path, required=True)
    run.add_argument('--archive-host', required=True)
    run.add_argument('--archive', type=Path, required=True)
    run.add_argument('--archive-repo', type=Path, required=True)
    run.add_argument('--archive-held', type=Path, required=True)
    run.add_argument('--seed', action='append', type=Path, default=[])
    run.add_argument('--review-model')
    run.add_argument('--attempts', type=int, default=2)
    run.add_argument('--wave-size', type=int, default=6)
    run.add_argument('--concurrency', type=int, default=1)
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
    if args.attempts < 1 or not 1 <= args.concurrency <= args.wave_size or 'sol' in reviewer or 'luna' in reviewer:
        parser.error('provide positive limits, concurrency within the wave size, and an independent non-Sol reviewer when configured')
    with (args.root / '.controller.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        drive(args)


if __name__ == '__main__':
    main()
