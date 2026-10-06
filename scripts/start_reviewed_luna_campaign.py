#!/usr/bin/env python3
"""Run an immutable, reviewed fresh campaign with up to five Luna workers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from start_reviewed_generation_successor import atomic_json, digest
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
    if plan.get('root_approved') is not True or not 1 <= len(workers) <= 5:
        raise ValueError('Requires approval for one to five Luna workers')
    for path, expected in plan['artifact_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed: {path}')
    runtime = Path(plan['runtime'])
    if tree_identity(runtime) != json.loads((runtime / 'frozen-runtime.json').read_text())['files']:
        raise ValueError('Frozen runtime changed')
    keys, indices, evidence = set(), set(), set()
    for worker in workers:
        for field in ('journal', 'log'):
            path = Path(worker[field]).resolve()
            if path in evidence or path.exists():
                raise ValueError('Fresh, distinct evidence paths required')
            evidence.add(path)
        for line in Path(worker['queue']).read_text().splitlines():
            entry = json.loads(line)
            identity = (entry['source'], entry['index'])
            if entry['key'] in keys or identity in indices or entry.get('count', 1) != 1:
                raise ValueError('Workers must have disjoint single-case queues')
            if entry['source'] not in plan['artifact_hashes']:
                raise ValueError('Queue source is not pinned')
            keys.add(entry['key'])
            indices.add(identity)
    record = Path(plan['launch_record'])
    with record.open('x') as stream:
        json.dump({'status': 'starting', 'plan': str(args.plan), 'time': time.time()}, stream)
    processes, launched = [], []
    try:
        for worker in workers:
            command = ['python3', plan['supervisor'], worker['queue'], worker['journal'],
                       '--runtime', str(runtime), '--provider', 'openai-codex',
                       '--model-id', 'gpt-6-luna', '--model-concurrency', '1',
                       '--execution-plans', '--reasoning-effort', 'low',
                       '--min-free-mib', '1024']
            Path(worker['log']).parent.mkdir(parents=True, exist_ok=True)
            with Path(worker['log']).open('xb') as log:
                process = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            launched.append({**worker, 'pid': process.pid, 'command': command})
            atomic_json(record, {'status': 'running', 'workers': launched, 'plan': str(args.plan)})
        codes = [process.wait() for process in processes]
        atomic_json(record, {'status': 'finished', 'workers': launched, 'exit_codes': codes,
                             'plan': str(args.plan), 'time': time.time()})
        return int(any(codes))
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        atomic_json(record, {'status': 'launch_failed', 'workers': launched, 'plan': str(args.plan)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
