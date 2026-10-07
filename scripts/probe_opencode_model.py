#!/usr/bin/env python3
"""Probe a free model through the official OpenCode CLI, retaining retry evidence."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='opencode/exo-free')
    parser.add_argument('--delay-seconds', type=float, default=300)
    parser.add_argument('--max-delay-seconds', type=float, default=1800)
    parser.add_argument('--attempt-seconds', type=float, default=120)
    parser.add_argument('--attempts', type=int, default=0, help='Zero means retry until ready or stopped.')
    args = parser.parse_args()
    if not args.model.startswith('opencode/') or not args.model.endswith('-free'):
        parser.error('This probe only requests explicitly free OpenCode models.')
    if args.delay_seconds < 60 or args.max_delay_seconds < args.delay_seconds or args.attempt_seconds < 1 or args.attempts < 0:
        parser.error('Invalid retry settings; retries must be at least 60 seconds apart.')
    client = args.client.resolve()
    env = os.environ.copy()
    for line in args.credentials.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words and words[0] == 'export':
            words = words[1:]
        if not words:
            continue
        if len(words) != 1 or '=' not in words[0]:
            raise ValueError('Credential file must contain simple KEY=value assignments.')
        name, value = words[0].split('=', 1)
        if name == 'OPENCODE_API_KEY':
            env[name] = value
    secret = env.get('OPENCODE_API_KEY')
    if not secret:
        raise ValueError('No OPENCODE_API_KEY was found.')
    env['OPENCODE_CONFIG_CONTENT'] = json.dumps({
        'provider': {'opencode': {'options': {'apiKey': '{env:OPENCODE_API_KEY}'}}},
        'share': 'disabled', 'autoupdate': False,
    })
    out = args.out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    scratch = out / 'scratch'
    scratch.mkdir()
    state = {'model': args.model, 'official_client': str(client),
             'credential_file': str(args.credentials.resolve()), 'started_at': now(),
             'status': 'starting', 'training_admission': False}
    write_json(out / 'launch.json', state)
    child = None
    def stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    attempt = 0
    try:
        while True:
            attempt += 1
            started = now()
            timed_out = False
            child = subprocess.Popen([
                str(client), 'run', '--model', args.model, '--format', 'json',
                'Reply exactly READY. Do not inspect or change any files.',
            ], cwd=scratch, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                stdout, stderr = child.communicate(timeout=args.attempt_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate()
            stdout = stdout.decode(errors='replace').replace(secret, '[REDACTED]')
            stderr = stderr.decode(errors='replace').replace(secret, '[REDACTED]')
            (out / f'attempt-{attempt:04d}.stdout.jsonl').write_text(stdout)
            (out / f'attempt-{attempt:04d}.stderr.log').write_text(stderr)
            events = []
            for line in stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            text = ''.join(event.get('part', {}).get('text', '')
                           for event in events if event.get('type') == 'text')
            errors = [event.get('error') for event in events if event.get('type') == 'error']
            ready = child.returncode == 0 and not timed_out and not errors and text.strip() == 'READY'
            state.update(status='model_ready' if ready else 'unavailable', attempt=attempt,
                         attempt_started_at=started, observed_at=now(), exit_code=child.returncode,
                         timed_out=timed_out, model_returned_ready=ready, provider_errors=errors)
            write_json(out / f'attempt-{attempt:04d}.receipt.json', state)
            write_json(out / 'status.json', state)
            print(json.dumps({'time': now(), 'attempt': attempt, 'status': state['status']}), flush=True)
            child = None
            if ready:
                return 0
            if args.attempts and attempt >= args.attempts:
                return 1
            delay = min(args.max_delay_seconds, args.delay_seconds * 2 ** min(attempt - 1, 8))
            time.sleep(delay * random.uniform(1.0, 1.1))
    except KeyboardInterrupt:
        state.update(status='operator_stopped', observed_at=now())
        write_json(out / 'status.json', state)
        return 130
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()


if __name__ == '__main__':
    raise SystemExit(main())
