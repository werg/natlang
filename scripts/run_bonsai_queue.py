#!/usr/bin/env python3
"""Run a frozen, interleaved teacher queue with a hard per-case wall-clock budget.

Timeouts and failures remain explicit in the append-only journal and are not training results.
Restart skips attempted entries; use a new journal for an explicitly reviewed retry pass.
"""
import argparse
import json
import subprocess
import signal
import time
from pathlib import Path


def run_queue(queue, journal, runtime, seconds=600):
    queue, journal, runtime = map(Path, (queue, journal, runtime))
    entries = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    attempted = set()
    if journal.exists():
        attempted = {json.loads(line)['key'] for line in journal.read_text().splitlines()
                     if line.strip() and json.loads(line).get('event') == 'finish'}
    journal.parent.mkdir(parents=True, exist_ok=True)
    def record(value):
        with journal.open('a') as output:
            output.write(json.dumps(value) + '\n')
            output.flush()
        print(json.dumps(value), flush=True)
    for entry in entries:
        if entry['key'] in attempted:
            continue
        command = ['node', '--max-old-space-size=3000', str(runtime / 'dist/teacher/cli.js'),
                   entry['source'], entry['jobs'], entry['output'], '--start', str(entry['index']), '--limit', '1',
                   '--model-id', 'Ternary-Bonsai-2-27B', '--server', 'http://127.0.0.1:8081',
                   '--root-seed', str(entry['seed']), '--workers', '1', '--max-turns', '20',
                   '--model-concurrency', '2', '--max-model-requests', '128', '--kv-tokens', '40000',
                   '--transport-retries', '1', '--file-tools', entry.get('surface', 'all')]
        start = time.monotonic()
        record({'event': 'start', 'key': entry['key'], 'time': time.time(), 'budget_seconds': seconds})
        with Path(entry['log']).open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=seconds)
                status = 'complete' if code == 0 else 'incomplete'
            except subprocess.TimeoutExpired:
                status, code = 'timeout', None
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        record({'event': 'finish', 'key': entry['key'], 'status': status, 'exit_code': code,
                'elapsed_seconds': round(time.monotonic() - start, 1), 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue')
    parser.add_argument('journal')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--case-seconds', type=int, default=600)
    args = parser.parse_args()
    if args.case_seconds < 1:
        parser.error('--case-seconds must be positive')
    def stop(signum, frame):
        raise KeyboardInterrupt('queue stopped')
    signal.signal(signal.SIGTERM, stop)
    run_queue(args.queue, args.journal, args.runtime, args.case_seconds)
