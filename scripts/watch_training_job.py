#!/usr/bin/env python3
"""Read-only monitoring wait that ends early when a pinned Docker job exits.

Run in the foreground of an agent's tool session. Wait on that session in
interruptible intervals <=60 seconds; an exit wakes the next diagnostic turn.
This does not restart jobs, alter resources, or grant checkpoint qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def inspect_job(container):
    result = subprocess.run(['docker', 'inspect', '--format', '{{json .}}', container],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('cannot inspect training container: ' + result.stderr.strip())
    value = json.loads(result.stdout)
    # Never return Config/Env: monitoring does not need credentials or commands.
    return {'id': value['Id'], 'state': value['State']}


def inspect_worker(pid):
    """Linux process identity, excluding argv/env and treating zombies as finished."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return {'running': False, 'start_ticks': None}
    return {'running': fields[0] not in {'Z', 'X'}, 'start_ticks': fields[19]}


def wait_for_job(container, seconds, poll_seconds, *, inspect=inspect_job,
                 now=time.monotonic, sleep=time.sleep, worker_pids=(), worker_inspect=inspect_worker):
    deadline = now() + seconds
    pinned = None
    worker_identities = {}
    while True:
        job = inspect(container)
        if pinned is None:
            pinned = job['id']
        elif job['id'] != pinned:
            return {'event': 'job_replaced', 'container': container, 'id': pinned}, 3
        state = job['state']
        if not state.get('Running'):
            code = state.get('ExitCode')
            return {'event': 'job_exited', 'container': container, 'id': pinned,
                    'status': state.get('Status'), 'exit_code': code,
                    'finished_at': state.get('FinishedAt')}, 0 if code == 0 else 2
        for pid in worker_pids:
            worker = worker_inspect(pid)
            if not worker['running']:
                return {'event': 'generation_worker_exited', 'container': container,
                        'worker_pid': pid}, 0
            if pid in worker_identities and worker['start_ticks'] != worker_identities[pid]:
                return {'event': 'generation_worker_replaced', 'container': container,
                        'worker_pid': pid}, 3
            worker_identities[pid] = worker['start_ticks']
        remaining = deadline - now()
        if remaining <= 0:
            return {'event': 'monitoring_cycle_due', 'container': container, 'id': pinned}, 0
        sleep(min(poll_seconds, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--seconds', type=float, default=3000)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--worker-pid', type=int, action='append', default=[],
                        help='also wake when a pinned local generation worker finishes; repeat for each worker')
    args = parser.parse_args()
    if args.seconds < 0 or not 0 < args.poll_seconds <= 60 or any(pid <= 0 for pid in args.worker_pid):
        parser.error('seconds must be nonnegative; poll-seconds must be in (0,60]')
    try:
        result, code = wait_for_job(args.container, args.seconds, args.poll_seconds, worker_pids=args.worker_pid)
    except (RuntimeError, ValueError, KeyError) as error:
        result, code = {'event': 'monitor_failed', 'container': args.container, 'error': str(error)}, 2
    print(json.dumps(result), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
