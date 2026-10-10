#!/usr/bin/env python3
"""Run an immutable, reviewed fresh campaign with up to five Luna workers."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
from start_reviewed_generation_successor import atomic_json, digest, running
from freeze_training_runtime import tree_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.sha256:
        raise ValueError('Reviewed plan changed')
    plan = json.loads(args.plan.read_text())
    workers = plan['workers']
    reasoning_effort = plan.get('reasoning_effort', 'low')
    if reasoning_effort not in {'low', 'medium', 'high'}:
        raise ValueError('Invalid reviewed reasoning effort')
    if plan.get('root_approved') is not True or not 1 <= len(workers) <= 5:
        raise ValueError('Requires approval for one to five Luna workers')
    for path, expected in plan['artifact_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed: {path}')
    runtime = Path(plan['runtime'])
    if tree_identity(runtime) != json.loads((runtime / 'frozen-runtime.json').read_text())['files']:
        raise ValueError('Frozen runtime changed')
    keys, indices, evidence = set(), set(), set()
    pinned_paths = {str(Path(path).resolve()) for path in plan['artifact_hashes']}
    for worker in workers:
        if worker['queue'] not in plan['artifact_hashes']:
            raise ValueError('Worker queue is not pinned')
        for field in ('journal', 'log'):
            path = Path(worker[field]).resolve()
            if path in evidence or path.exists():
                raise ValueError('Fresh, distinct evidence paths required')
            evidence.add(path)
        for line in Path(worker['queue']).read_text().splitlines():
            entry = json.loads(line)
            source_path = Path(entry['source'])
            if not source_path.is_absolute():
                source_path = Path(plan['cwd']) / source_path
            source_path = str(source_path.resolve())
            identity = (source_path, entry['index'])
            if entry['key'] in keys or identity in indices or entry.get('count', 1) != 1:
                raise ValueError('Workers must have disjoint single-case queues')
            if source_path not in pinned_paths:
                raise ValueError('Queue source is not pinned')
            keys.add(entry['key'])
            indices.add(identity)
    record = Path(plan['launch_record'])
    # A reviewed refill can wait without borrowing any of its predecessor's
    # worker slots. Failed or incomplete predecessors require agent review.
    predecessors = plan.get('predecessors', [])
    for predecessor in predecessors:
        if digest(predecessor['queue']) != predecessor['queue_sha256']:
            raise ValueError('Predecessor queue changed')
    while any(running(p['pid']) for p in predecessors):
        atomic_json(record.with_suffix('.waiting.json'),
                    {'status': 'waiting_for_predecessors', 'plan': str(args.plan), 'time': time.time()})
        time.sleep(30)
    for predecessor in predecessors:
        if digest(predecessor['queue']) != predecessor['queue_sha256']:
            raise ValueError('Predecessor queue changed while waiting')
        finishes = {}
        for line in Path(predecessor['journal']).read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'finish' and not event.get('batch_key'):
                finishes[event['key']] = event
        for line in Path(predecessor['queue']).read_text().splitlines():
            event = finishes.get(json.loads(line)['key'], {})
            if event.get('status') not in {'complete', 'complete_with_skips', 'skipped'} or not event.get(
                    'output_accounting', {}).get('complete'):
                raise ValueError('Predecessor incomplete; agent review required')
    # Recheck pins after a potentially long wait, before creating any worker.
    for path, expected in plan['artifact_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed while waiting: {path}')
    if tree_identity(runtime) != json.loads((runtime / 'frozen-runtime.json').read_text())['files']:
        raise ValueError('Frozen runtime changed while waiting')
    def stop(signum, frame):
        raise KeyboardInterrupt('operator stopped campaign')
    signal.signal(signal.SIGTERM, stop)
    with record.open('x') as stream:
        json.dump({'status': 'starting', 'plan': str(args.plan), 'time': time.time()}, stream)
    # The launch record now owns current status. Do not leave a live waiting
    # marker behind after all pinned predecessors have completed.
    record.with_suffix('.waiting.json').unlink(missing_ok=True)
    processes, launched = [], []
    try:
        for worker in workers:
            command = ['python3', plan['supervisor'], worker['queue'], worker['journal'],
                       '--runtime', str(runtime), '--provider', 'openai-codex',
                       '--model-id', 'gpt-6-luna', '--model-concurrency', '1',
                       '--execution-plans', '--reasoning-effort', reasoning_effort,
                       '--min-free-mib', '1024']
            Path(worker['log']).parent.mkdir(parents=True, exist_ok=True)
            with Path(worker['log']).open('xb') as log:
                process = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            launched.append({**worker, 'pid': process.pid, 'command': command})
            atomic_json(record, {'status': 'running', 'workers': launched, 'plan': str(args.plan)})
        codes = [process.wait() for process in processes]
        atomic_json(record, {'status': 'finished' if not any(codes) else 'finished_with_incomplete_work',
                             'workers': launched, 'exit_codes': codes,
                             'plan': str(args.plan), 'time': time.time()})
        return int(any(codes))
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        interrupted = isinstance(error, KeyboardInterrupt)
        atomic_json(record, {'status': 'operator_stopped' if interrupted else 'launch_failed',
                             'workers': launched, 'plan': str(args.plan), 'time': time.time(),
                             'disposition': 'Unfinished attempts remain unscored; preserve partial evidence.'})
        if interrupted:
            return 130
        raise


if __name__ == '__main__':
    raise SystemExit(main())
