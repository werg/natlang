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
import math
from pathlib import Path


def partial_metrics(entry):
    """Saved replies are activity evidence, not a claim that the task made semantic progress."""
    metrics = dict(saved_turns=0, fresh_model_replies=0, completion_tokens=0, repeated_action_sets=0)
    seen = set()
    for path in Path(entry['jobs']).glob(f"{entry['index']:06d}-*.partial.json"):
        try:
            turns = json.loads(path.read_text())['turns']
        except (OSError, ValueError, KeyError):
            continue
        metrics['saved_turns'] += len(turns)
        for turn in turns:
            response = turn.get('response', {})
            if not response.get('raw_response'):
                continue
            metrics['fresh_model_replies'] += 1
            metrics['completion_tokens'] += response.get('completion_tokens', 0) or 0
            action = json.dumps(response.get('calls', []), sort_keys=True)
            if action in seen:
                metrics['repeated_action_sets'] += 1
            seen.add(action)
    if not metrics['saved_turns']:
        for path in Path(entry['jobs']).glob(f"{entry['index']:06d}-*.result.json"):
            try:
                turns = json.loads(path.read_text())['trajectory']
            except (OSError, ValueError, KeyError):
                continue
            metrics['saved_turns'] += len(turns)
            for turn in turns:
                if not turn.get('raw_response_sha256'):
                    continue
                response = turn.get('model_response', {})
                metrics['fresh_model_replies'] += 1
                metrics['completion_tokens'] += response.get('completion_tokens', 0) or 0
                action = json.dumps(response.get('calls', []), sort_keys=True)
                if action in seen:
                    metrics['repeated_action_sets'] += 1
                seen.add(action)
    metrics['unique_action_sets'] = len(seen)
    return metrics


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
                   '--model-concurrency', '2', '--max-model-requests', str(entry.get('max_model_requests', 128)), '--kv-tokens', '40000',
                   '--transport-retries', '1', '--file-tools', entry.get('surface', 'all')]
        case_seconds = entry.get('case_seconds', seconds)
        if not isinstance(case_seconds, int) or case_seconds < 1:
            raise ValueError('entry case_seconds must be a positive integer')
        start = time.monotonic()
        last_activity, previous = start, partial_metrics(entry)
        record({'event': 'start', 'key': entry['key'], 'time': time.time(), 'budget_seconds': case_seconds})
        with Path(entry['log']).open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                for tick in range(math.ceil(case_seconds / 30)):
                    remaining = case_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, case_seconds)
                    try:
                        code = child.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        current = partial_metrics(entry)
                        if current['saved_turns'] != previous['saved_turns']:
                            last_activity, previous = time.monotonic(), current
                        elif time.monotonic() - last_activity >= 300:
                            raise TimeoutError('no saved reply for 300 seconds')
                        record({'event': 'activity', 'key': entry['key'], 'time': time.time(),
                                'elapsed_seconds': round(time.monotonic() - start, 1), **partial_metrics(entry)})
                else:
                    raise subprocess.TimeoutExpired(command, case_seconds)
                status = 'complete' if code == 0 else 'incomplete'
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                status, code = ('inactivity_timeout' if isinstance(error, TimeoutError) else 'timeout'), None
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
                'elapsed_seconds': round(time.monotonic() - start, 1), 'time': time.time(), **partial_metrics(entry)})


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
