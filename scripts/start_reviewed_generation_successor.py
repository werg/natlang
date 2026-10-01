#!/usr/bin/env python3
"""Start a hash-pinned, root-reviewed successor after its predecessor finishes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def running(pid):
    try:
        return b'run_bonsai_queue.py' in Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.sha256:
        raise ValueError('Successor plan differs from root-reviewed hash')
    plan = json.loads(args.plan.read_text())
    if plan.get('root_approved') is not True:
        raise ValueError('Successor requires explicit root approval')
    provider = plan.get('provider')
    if provider and (provider != 'openai-codex' or plan.get('model_id') != 'gpt-6-luna' or plan.get('model_concurrency') != 1 or len(plan.get('workers', [])) != 2):
        raise ValueError('Luna successors require exactly two workers, one request each')
    if not provider and (plan.get('model_concurrency') != 4 or len(plan.get('workers', [])) != 1):
        raise ValueError('Bonsai successors require one supervisor with four requests')
    if plan['supervisor'] not in plan['artifact_hashes']:
        raise ValueError('Supervisor code must be hash-pinned too')
    record = Path(plan['launch_record'])
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            raise ValueError('Successor has already been claimed; review saved launch evidence')
        while any(running(p['pid']) for p in plan['predecessors']):
            time.sleep(30)
        for predecessor in plan['predecessors']:
            if digest(predecessor['queue']) != predecessor['queue_sha256']:
                raise ValueError('Predecessor queue identity changed')
            keys = {json.loads(line)['key'] for line in Path(predecessor['queue']).read_text().splitlines() if line.strip()}
            finishes = {}
            for line in Path(predecessor['journal']).read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    if event.get('event') == 'finish' and not event.get('batch_key'):
                        finishes[event['key']] = event
            for key in keys:
                event = finishes.get(key, {})
                if event.get('status') not in {'complete', 'complete_with_skips', 'skipped'} or not event.get('output_accounting', {}).get('complete'):
                    raise ValueError(f'Predecessor is incomplete; agent review required: {key}')
        for file, expected in plan['artifact_hashes'].items():
            if digest(file) != expected:
                raise ValueError(f'Reviewed artifact or admission policy changed: {file}')
        for worker in plan['workers']:
            if Path(worker['journal']).exists():
                raise ValueError('Successor journal already exists; refuse duplicate worker')
            if not any(Path(worker['queue']).resolve() == Path(p).resolve() for p in plan['artifact_hashes']):
                raise ValueError('Successor queue lacks an approval hash')
        with record.open('x') as stream:
            launch = dict(status='claimed', claimed_at=time.time(), plan=str(args.plan), plan_sha256=args.sha256, workers=[])
            json.dump(launch, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A crash after claiming is an explicit review state, never an automatic duplicate launch.
        for worker in plan['workers']:
            command = ['python3', plan['supervisor'], worker['queue'], worker['journal'], '--runtime', plan['runtime'],
                       '--case-seconds', str(plan['case_seconds']), '--model-concurrency', str(plan['model_concurrency']),
                       '--min-free-mib', '1024']
            if plan.get('provider'):
                command += ['--provider', plan['provider'], '--model-id', plan['model_id'], '--execution-plans', '--reasoning-effort', 'low']
            Path(worker['log']).parent.mkdir(parents=True, exist_ok=True)
            with Path(worker['log']).open('ab') as log:
                process = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch['workers'].append({**worker, 'pid': process.pid, 'command': command})
            atomic_json(record, launch)
        launch.update(status='running', launched_at=time.time())
        atomic_json(record, launch)
        # Reconcile only this provider's authority. The hourly monitor reads these current paths.
        authority_path = Path(plan['authority'])
        with authority_lock(authority_path):
            authority = json.loads(authority_path.read_text())
            if plan.get('provider'):
                authority.update(luna_workers=launch['workers'], luna_runtime=plan['runtime'], luna_status='reviewed_successor_running')
            else:
                worker = launch['workers'][0]
                authority.update(bonsai_supervisor=worker['pid'], bonsai_queue=worker['queue'], bonsai_journal=worker['journal'],
                                 bonsai_runtime=plan['runtime'], bonsai_status='reviewed_successor_running')
            authority.setdefault('active_journals', []).extend(w['journal'] for w in launch['workers'])
            atomic_json(authority_path, authority)
            if plan.get('provider') and plan.get('luna_state'):
                state_path = Path(plan['luna_state'])
                state = json.loads(state_path.read_text())
                state.update(workers=launch['workers'], runtime=plan['runtime'], status='reviewed_successor_running', latest_launch=str(record))
                atomic_json(state_path, state)
        print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    main()
