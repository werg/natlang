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
import random
import urllib.request
from pathlib import Path


def local_decode_progress():
    """A long batched reply can be decoding before its next durable checkpoint exists."""
    try:
        with urllib.request.urlopen('http://127.0.0.1:8081/metrics', timeout=2) as response:
            for line in response.read().decode().splitlines():
                if line.startswith('llamacpp:n_decode_total '):
                    value = float(line.split()[1])
                    return value if math.isfinite(value) and value >= 0 else None
    except (OSError, ValueError):
        pass
    return None


def partial_metrics(entry):
    """Saved replies show activity. Repeated action shapes across distinct child inputs do not imply a loop."""
    metrics = dict(saved_turns=0, fresh_model_replies=0, completion_tokens=0, repeated_action_sets=0, repeated_request_hashes=0)
    seen = set()
    requests = set()
    def note_request(turn):
        digest = turn.get("request_sha256")
        if digest:
            if digest in requests:
                metrics["repeated_request_hashes"] += 1
            requests.add(digest)
    indices = range(entry['index'], entry['index'] + entry.get('count', 1))
    # Resolve each root independently: a finished root has a result while its siblings
    # may still have partial checkpoints. Count each root once, preferring the result.
    for index in indices:
        results = list(Path(entry['jobs']).glob(f"{index:06d}-*.result.json"))
        paths = results or list(Path(entry['jobs']).glob(f"{index:06d}-*.partial.json"))
        for path in paths:
            completed = path.name.endswith('.result.json')
            try:
                turns = json.loads(path.read_text())['trajectory' if completed else 'turns']
            except (OSError, ValueError, KeyError):
                continue
            metrics['saved_turns'] += len(turns)
            for turn in turns:
                response = turn.get('model_response' if completed else 'response', {})
                if not (turn.get('raw_response_sha256') if completed else response.get('raw_response')):
                    continue
                note_request(turn)
                metrics['fresh_model_replies'] += 1
                metrics['completion_tokens'] += response.get('completion_tokens', 0) or 0
                action = json.dumps(response.get('calls', []), sort_keys=True)
                if action in seen:
                    metrics['repeated_action_sets'] += 1
                seen.add(action)
    metrics['unique_action_sets'] = len(seen)
    metrics['unique_request_hashes'] = len(requests)
    metrics['repetition_scope'] = 'action shapes across all child calls; not a semantic stall detector'
    return metrics


def retry_deadline(entry):
    """Provider minimum delays survive exhausted retries, timeouts, and supervisor restarts."""
    deadline = 0
    for index in range(entry['index'], entry['index'] + entry.get('count', 1)):
        paths = list(Path(entry['jobs']).glob(f"{index:06d}-*.retry.json"))
        paths.append(Path(entry['jobs']) / f"{index:06d}.error.json")
        for path in paths:
            try:
                row = json.loads(path.read_text())
                value = float(row.get('until', row.get('retry_not_before', 0))) / 1000
                if math.isfinite(value):
                    deadline = max(deadline, value)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return deadline


def retry_waiting(entry):
    return retry_deadline(entry) > time.time()


def failure_cooldown(streak):
    return min(300, 30 * 2 ** min(max(0, streak - 1), 20) * (0.5 + random.random()))


def run_queue(queue, journal, runtime, seconds=600, model_id='Ternary-Bonsai-2-27B',
              provider=None, model_concurrency=None, execution_plans=False, reasoning_effort='low'):
    if model_concurrency is None:
        model_concurrency = 1 if provider else 4
    queue, journal, runtime = map(Path, (queue, journal, runtime))
    entries = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    attempted = set()
    failure_streak, next_allowed_at = 0, 0
    if journal.exists():
        attempted = {json.loads(line)['key'] for line in journal.read_text().splitlines()
                     if line.strip() and json.loads(line).get('event') == 'finish'}
        if provider:
            for line in journal.read_text().splitlines():
                row = json.loads(line)
                if row.get('event') == 'finish' and not row.get('batch_key'):
                    failure_streak = 0 if row['status'] == 'complete' else failure_streak + 1
                    next_allowed_at = row.get('next_allowed_at', 0)
    journal.parent.mkdir(parents=True, exist_ok=True)
    def record(value):
        with journal.open('a') as output:
            output.write(json.dumps(value) + '\n')
            output.flush()
        print(json.dumps(value), flush=True)
    for entry in entries:
        if entry['key'] in attempted:
            continue
        if provider and next_allowed_at > time.time():
            record(dict(event='cooldown', key=entry['key'], time=time.time(),
                        failure_streak=failure_streak, until=next_allowed_at))
            while next_allowed_at > time.time():
                time.sleep(min(30, next_allowed_at - time.time()))
        max_turns = entry.get('max_turns', 20)
        count = entry.get('count', 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
            raise ValueError('entry count must be between one and five')
        if count > 1 and provider:
            raise ValueError('provider queues remain single-case; batching is only enabled for local Bonsai')
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError('entry max_turns must be a positive integer')
        command = ['node', '--max-old-space-size=3000', str(runtime / 'dist/teacher/cli.js'),
                   entry['source'], entry['jobs'], entry['output'], '--start', str(entry['index']), '--limit', str(count),
                   '--model-id', model_id,
                   '--root-seed', str(entry['seed']), '--workers', str(count), '--max-turns', str(max_turns),
                   '--model-concurrency', str(model_concurrency), '--max-model-requests', str(entry.get('max_model_requests', 128)),
                   '--transport-retries', '1', '--file-tools', entry.get('surface', 'all')]
        command += ['--provider', provider, '--context-tokens', '16384', '--reasoning-effort', reasoning_effort] if provider else [
            '--server', 'http://127.0.0.1:8081', '--kv-tokens', '40000']
        if provider:
            command += ['--retry-delay-ms', '15000']
        if execution_plans:
            command.append('--execution-plans')
        case_seconds = entry.get('case_seconds', seconds)
        if not isinstance(case_seconds, int) or case_seconds < 1:
            raise ValueError('entry case_seconds must be a positive integer')
        start = time.monotonic()
        last_activity, previous = start, partial_metrics(entry)
        previous_decode = None
        record({'event': 'start', 'key': entry['key'], 'time': time.time(), 'budget_seconds': case_seconds,
                'max_turns': max_turns})
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
                        decode = local_decode_progress() if not provider else None
                        decoding = decode is not None and previous_decode is not None and decode > previous_decode
                        previous_decode = decode
                        if current['saved_turns'] != previous['saved_turns']:
                            last_activity, previous = time.monotonic(), current
                        elif decoding or (provider and retry_waiting(entry)):
                            last_activity = time.monotonic()
                        elif time.monotonic() - last_activity >= 300:
                            raise TimeoutError('no checkpoint or local decode progress for 300 seconds')
                        record({'event': 'activity', 'key': entry['key'], 'time': time.time(),
                                'elapsed_seconds': round(time.monotonic() - start, 1),
                                'local_decode_total': decode, **partial_metrics(entry)})
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
        if provider:
            failure_streak = 0 if status == 'complete' else failure_streak + 1
            next_allowed_at = max(time.time() + failure_cooldown(failure_streak), retry_deadline(entry)) if failure_streak else 0
        record({'event': 'finish', 'key': entry['key'], 'status': status, 'exit_code': code,
                'failure_streak': failure_streak, 'next_allowed_at': next_allowed_at,
                'elapsed_seconds': round(time.monotonic() - start, 1), 'time': time.time(), **partial_metrics(entry)})
        # Preserve original attempt keys when roots share one collector batch.
        # A terminal batch event covers both attempts, even when only one produced a result.
        for member in entry.get('members', []):
            record({'event': 'finish', 'key': member['key'], 'status': status, 'exit_code': code,
                    'batch_key': entry['key'], 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue')
    parser.add_argument('journal')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--case-seconds', type=int, default=600)
    parser.add_argument('--model-id', default='Ternary-Bonsai-2-27B')
    parser.add_argument('--provider')
    parser.add_argument('--model-concurrency', type=int, help='global request cap, including children (local: 4; provider: 1)')
    parser.add_argument('--execution-plans', action='store_true')
    parser.add_argument('--reasoning-effort', default='low')
    args = parser.parse_args()
    if args.case_seconds < 1:
        parser.error('--case-seconds must be positive')
    if args.model_concurrency is not None and args.model_concurrency < 1:
        parser.error('--model-concurrency must be positive')
    def stop(signum, frame):
        raise KeyboardInterrupt('queue stopped')
    signal.signal(signal.SIGTERM, stop)
    run_queue(args.queue, args.journal, args.runtime, args.case_seconds, args.model_id, args.provider,
              args.model_concurrency, args.execution_plans, args.reasoning_effort)
