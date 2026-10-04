#!/usr/bin/env python3
"""Finite, resumable skill episode queue; immutable attempts, bounded transport retries."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    staging = path.with_suffix(path.suffix + '.tmp')
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    staging.replace(path)


def retryable(result):
    # Semantic failures are evidence. Only transport/provider failures are retried.
    if result.get('disposition') != 'failed':
        return False
    text = str(result.get('error', '')).lower()
    return any(term in text for term in ['429', 'rate limit', '502', '503', '504',
        'econnreset', 'econnrefused', 'fetch failed', 'socket hang up', 'service unavailable'])


def missing_result_disposition(exit_code, stop_requested):
    # A deliberate stop or process signal is infrastructure evidence, not a model rejection.
    if stop_requested:
        return 'interrupted_attempt_requires_review'
    if exit_code is not None and exit_code < 0:
        return 'collector_terminated_by_signal'
    return 'collector_failed_without_artifact'


def run_queue(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve(strict=True)
    node = args.node.resolve(strict=True)
    data = args.episodes.read_bytes()
    episodes = [json.loads(line) for line in data.splitlines() if line.strip()]
    episodes = [episode for episode in episodes if episode['split'] == 'train']
    if not episodes or len({episode['id'] for episode in episodes}) != len(episodes):
        raise ValueError('nonempty unique train episodes required')
    seal_bytes = (runtime / 'frozen-runtime.json').read_bytes()
    seal = json.loads(seal_bytes)
    if seal.get('schema') != 'natlang.skill-authoring-runtime/1':
        raise ValueError('sealed authoring runtime required')
    # Validate the packet without inference. The child collector rechecks every runtime file.
    preflight = "import {readFileSync} from 'node:fs'; const {validateEpisode}=await import(process.argv[1]); const rows=readFileSync(process.argv[2],'utf8').trim().split('\\n').map(JSON.parse); for(const row of rows){const errors=validateEpisode(row);if(errors.length)throw Error(JSON.stringify(errors));}"
    subprocess.run([str(node), '--input-type=module', '-e', preflight,
        (runtime / 'dist/skills/episode.js').as_uri(), str(args.episodes.resolve())], check=True)
    node_version = subprocess.check_output([str(node), '--version'], text=True).strip()
    if int(node_version.lstrip('v').split('.')[0]) < 22:
        raise ValueError('Node 22 or newer is required')
    identity = {'schema': 'natlang.skill-authoring-queue/1', 'input_sha256': sha(data),
        'episode_ids': [row['id'] for row in episodes], 'runtime': str(runtime),
        'runtime_manifest_sha256': sha(seal_bytes), 'node': str(node), 'node_version': node_version,
        'node_sha256': sha(node.read_bytes()), 'endpoint': args.endpoint, 'model': args.model,
        'experiments': args.experiments, 'ablations': args.ablations, 'workers': args.workers, 'max_attempts': args.max_attempts,
        'backoff_seconds': args.backoff_seconds, 'queue_script_sha256': sha(Path(__file__).read_bytes())}
    if args.database_root:  # graded SQL episodes; absent from identities of queues that never needed it
        identity['database_root'] = str(args.database_root.resolve())
    if args.arena_root:
        identity['arena_root'] = str(args.arena_root.resolve())
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'queue.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != identity:
                raise ValueError('queue resume identity changed; use a fresh queue')
        else:
            write_json(manifest_path, identity)
        stop = threading.Event()
        def interrupt(_signum, _frame):
            stop.set()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, interrupt)

        def collect(episode):
            task = root / 'tasks' / sha(episode['id'].encode())[:20]
            task.mkdir(parents=True, exist_ok=True)
            input_path = task / 'episode.jsonl'
            expected_input = json.dumps(episode, separators=(',', ':')) + '\n'
            if input_path.exists() and input_path.read_text() != expected_input:
                raise ValueError('task input changed')
            if not input_path.exists():
                input_path.write_text(expected_input)
            state_path = task / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {'episode': episode['id'], 'attempts': []}
            if state.get('terminal'):
                return state
            for attempt in range(1, args.max_attempts + 1):
                if stop.is_set():
                    return state
                directory = task / f'attempt-{attempt:03d}'
                result_path = directory / sha(episode['id'].encode())[:20] / 'result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    # Existing unfinished attempt is never overwritten or double-started.
                    if directory.exists():
                        state.update(terminal=True, disposition='unfinished_attempt_requires_review', attempt=attempt)
                        write_json(state_path, state)
                        return state
                    directory.mkdir()
                    command = [str(node), str(runtime / 'scripts/skills/collect-episodes.mjs'),
                        '--episodes', str(input_path), '--out', str(directory), '--limit', '1',
                        '--experiments', str(args.experiments), '--ablations', str(args.ablations), '--endpoint', args.endpoint, '--model', args.model]
                    if args.database_root:
                        command += ['--database-root', str(args.database_root.resolve())]
                    if args.arena_root:
                        command += ['--arena-root', str(args.arena_root.resolve())]
                    with (directory / 'service.log').open('w') as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        write_json(directory / 'launch.json', {'command': command, 'pid': process.pid,
                            'started_at_unix': time.time(), 'episode_sha256': sha(expected_input.encode())})
                        while process.poll() is None:
                            if stop.wait(1):
                                process.terminate()
                                # Collector aborts and journals the attempt; no per-episode wall timeout.
                                process.wait()
                        exit_code = process.returncode
                    if not result_path.exists():
                        state.update(terminal=True, disposition=missing_result_disposition(exit_code, stop.is_set()), exit_code=exit_code,
                                     stop_requested=stop.is_set(),
                                     attempt=attempt, log=str(directory / 'service.log'))
                        write_json(state_path, state)
                        return state
                    result = json.loads(result_path.read_text())
                if result.get('episode') != episode['id']:
                    raise ValueError('result episode mismatch')
                receipt = {'attempt': attempt, 'path': str(result_path), 'sha256': sha(result_path.read_bytes()),
                    'disposition': result.get('disposition'), 'positive': result.get('positive') is True}
                state['attempts'] = [r for r in state['attempts'] if r['attempt'] != attempt] + [receipt]
                state.update(disposition=receipt['disposition'], positive=receipt['positive'])
                if not retryable(result) or attempt == args.max_attempts or stop.is_set():
                    state['terminal'] = True
                    write_json(state_path, state)
                    return state
                delay = min(300, args.backoff_seconds * 2 ** (attempt - 1))
                state.update(terminal=False, retry_delay_seconds=delay)
                write_json(state_path, state)
                if stop.wait(delay):
                    return state
            return state

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(collect, episode) for episode in episodes]
            for future in concurrent.futures.as_completed(futures):
                state = future.result()
                print(json.dumps({key: state.get(key) for key in ['episode', 'disposition', 'positive', 'terminal']}), flush=True)
        states = [json.loads(path.read_text()) for path in (root / 'tasks').glob('*/state.json')]
        summary = {'schema': identity['schema'], 'episodes': len(episodes), 'accounted': len(states),
            'terminal': sum(row.get('terminal') is True for row in states),
            'positive_candidates': sum(row.get('positive') is True for row in states),
            'interrupted': stop.is_set(), 'publication': 'Candidates only; exact offline replay admission required'}
        write_json(root / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['episodes', 'runtime', 'node', 'out']:
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8082')
    parser.add_argument('--model', default='nvidia/Qwen3.6-35B-A3B-NVFP4')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--experiments', type=int, default=3)
    parser.add_argument('--ablations', type=int, default=0)
    parser.add_argument('--database-root', type=Path, help='read-only SQLite databases for graded SQL episodes')
    parser.add_argument('--arena-root', type=Path, help='read-only pinned external game engines')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--backoff-seconds', type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.max_attempts <= 5 or args.experiments < 1 or not 0 <= args.ablations <= 12 or not math.isfinite(args.backoff_seconds) or args.backoff_seconds <= 0:
        parser.error('workers 1..16, attempts 1..5, positive experiments/backoff required')
    print(json.dumps(run_queue(args)), flush=True)


if __name__ == '__main__':
    main()
