#!/usr/bin/env python3
"""Experiment 1 (plans/neuralese/APPROACH_REVIEW_20261111.md section 4): collect N samples per task from one served policy.

Work-stealing slots over one or more actor endpoints (each episode is a fresh `run_bonsai_queue` process, as in the RL
collection), the production sampling capture (`x_natlang_rl_capture`) so every episode keeps its sampled token ids, then
`task_rollouts.score` emits `rollouts.jsonl` (input of `lower_rollout_turns.py`). Resumable: finished entries are skipped.

    exp1_collect.py --out DIR --cases CASES.jsonl --policy-id SHA --endpoint http://host:port=SLOTS [--endpoint ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common.jsonio import read_jsonl  # noqa: E402
from natlang_neuralese.common.sampling import RL_TOP_P  # noqa: E402
from natlang_neuralese.rl import task_rollouts  # noqa: E402
from natlang_neuralese.rl.actor_client import NativeActorClient  # noqa: E402
from natlang_neuralese.rl.held_eval import SEED_BASE, series  # noqa: E402


def serves(endpoint, policy_id):
    try:
        info = json.loads(NativeActorClient(endpoint, timeout=10).get_bytes('/v1/neuralese/info'))
    except Exception:
        return False
    return (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == policy_id


def attest(endpoint, policy_id):
    if not serves(endpoint, policy_id):
        raise SystemExit(f'{endpoint} does not serve {policy_id[:12]}')


def journals(work, key):
    """An episode's attempt journals in attempt order (numeric: a10 follows a9)."""
    return sorted(work.glob(f'{key}.a*.journal.jsonl'), key=lambda path: int(path.name[len(key) + 2:].split('.', 1)[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--policy-id', required=True)
    ap.add_argument('--endpoint', action='append', required=True, help='URL=SLOTS')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--temperature', type=float, default=1.0, help='sampling temperature (0: greedy; sanity checks)')
    ap.add_argument('--top-p', type=float, default=RL_TOP_P, help='nucleus of the sampling law (1.0: untruncated)')
    ap.add_argument('--case-seconds', type=int, default=3600)
    ap.add_argument('--max-model-requests', type=int, default=64)
    ap.add_argument('--max-turns', type=int, default=128)
    ap.add_argument('--context-tokens', type=int, default=131072, help="call budget; follows the actor's served window when smaller")
    ap.add_argument('--seed-base', type=int, default=SEED_BASE)
    ap.add_argument('--attempts', type=int, default=3, help='tries per episode after an infrastructure failure')
    ap.add_argument('--stagger', type=float, default=1.0, help='seconds between slot starts (a simultaneous start of 100+ collector processes peaks in memory)')
    ap.add_argument('--finalize-only', action='store_true')
    ap.add_argument('--finalize', action='store_true', help='score and emit rollouts.jsonl at the end (holds every episode in memory: 6+ GB '
                    'for 1,000 episodes; use exp1_score_shards.py instead)')
    ap.add_argument('--backoff', type=float, default=30., help='seconds to wait after an infrastructure failure before the next attempt')
    args = ap.parse_args()
    out = args.out.resolve()
    endpoints = [(item.rsplit('=', 1)[0].rstrip('/'), int(item.rsplit('=', 1)[1])) for item in args.endpoint]
    for url, _ in endpoints:
        attest(url, args.policy_id)
    if not (out / 'queue.jsonl').exists():
        rows = sum(1 for _ in read_jsonl(args.cases))
        task_rollouts.prepare(SimpleNamespace(
            cases=args.cases, out=out, rows=rows, offset=0, group=args.samples, temperature=args.temperature,
            case_seconds=args.case_seconds, max_model_requests=args.max_model_requests, max_turns=args.max_turns,
            seed_base=args.seed_base, root_mode='student', match_seeds_from=None, policy_checkpoint_sha256=args.policy_id))
        entries = list(read_jsonl(out / 'queue.jsonl'))
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=args.context_tokens)
        entries.sort(key=lambda e: (e['sample'], e['index']))   # sample-major: a cut-short run stays balanced
        (out / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
    entries = list(read_jsonl(out / 'queue.jsonl'))
    from natlang_neuralese.common.sampling import rl_sampling_controls, sampling_law_of
    (out / 'sampling-config.json').write_text(json.dumps(rl_sampling_controls(args.top_p, capture=True)) + '\n')
    spec = {'series': series(), 'endpoints': endpoints, 'policy_id': args.policy_id, 'samples': args.samples,
            'sampling': {**sampling_law_of(args.top_p), 'temperature': args.temperature},
            'started': time.time()}
    (out / 'collect-run.json').write_text(json.dumps(spec, indent=1) + '\n')
    work = out / 'workers'
    work.mkdir(exist_ok=True)

    def finished(key):
        # An infrastructure failure (status 'incomplete': an actor OOM, a dropped connection) is retried from the
        # episode's durable partial; every other finish (complete, a resource limit) is the episode's outcome.
        done = [row for path in journals(work, key) for row in read_jsonl(path) if row.get('event') == 'finish']
        return bool(done) and done[-1].get('status') != 'incomplete'

    todo = [e for e in entries if not finished(e['key'])] if not args.finalize_only else []
    lock, cursor = threading.Lock(), [0]
    stop = threading.Event()
    stats = {'done': 0, 'gave_up': 0, 'started': time.time()}
    down = set()   # endpoints currently not attesting the policy
    env = {**os.environ, 'NATLANG_MODEL_FACING': spec['series']['model_facing']}

    def slot(url, delay):
        time.sleep(delay)
        while not stop.is_set():
            with lock:
                if cursor[0] >= len(todo):
                    return
                entry = todo[cursor[0]]
                cursor[0] += 1
            key = entry['key']
            attempts = 0
            while not stop.is_set():
                wait_for(url)
                # A new journal per attempt, numbered after every earlier one (also a previous run's): run_bonsai_queue
                # skips a key its journal already finished, so reusing a1 after a restart retried nothing.
                number = len(journals(work, key)) + 1
                queue, journal = work / f'{key}.queue.jsonl', work / f'{key}.a{number}.journal.jsonl'
                queue.write_text(json.dumps({**entry, 'rl_endpoint': url}, sort_keys=True) + '\n')
                if run_entry(url, key, queue, journal) or stop.is_set():
                    break
                if not serves(url, args.policy_id):
                    continue   # the actor went away: not the episode's attempt; wait for it to serve again
                attempts += 1
                if attempts >= args.attempts:
                    break
                time.sleep(args.backoff)
            trim(entry)
            with lock:
                stats['done'] += 1
                stats['gave_up'] += not finished(key) and not stop.is_set()
                if stats['done'] % 8 == 0:
                    print(json.dumps({'done': stats['done'], 'gave_up': stats['gave_up'], 'of': len(todo),
                                      'minutes': round((time.time() - stats['started']) / 60, 1)}), flush=True)

    def wait_for(url):
        # An unreachable actor fails every episode at once: slots wait for it instead of spending their attempts.
        while not stop.is_set() and not serves(url, args.policy_id):
            with lock:
                if url not in down:
                    down.add(url)
                    print(json.dumps({'endpoint_down': url, 'at': time.strftime('%H:%M:%S')}), flush=True)
            time.sleep(args.backoff)
        with lock:
            if url in down:
                down.discard(url)
                print(json.dumps({'endpoint_up': url, 'at': time.strftime('%H:%M:%S')}), flush=True)

    def trim(entry):
        # A finished episode's output (results.jsonl) is a byte copy of jobs/*.result.json: keep one (disk: about 6 MB each)
        output = Path(entry['output'])
        if finished(entry['key']) and output.is_file():
            for copy in output.parent.glob('jobs/*.result.json'):
                if copy.stat().st_size == output.stat().st_size and copy.read_bytes() == output.read_bytes():
                    copy.unlink()

    def run_entry(url, key, queue, journal):
        command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue), str(journal), '--runtime',
                   str(REPO / 'ts-host'), '--server', url, '--chat-completions-url', url + '/v1/chat/completions',
                   '--model-id', 'natlang-rl', '--model-concurrency', '1', '--chat-request-config',
                   str(out / 'sampling-config.json'), '--neuralese-server']
        with (work / f'{key}.log').open('a') as log:
            subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        return finished(key)

    def memory_log():
        # attribution of the unit's memory: this driver vs the per-slot children (run_bonsai_queue, node), MB
        def rss(pid):
            try:
                with open(f'/proc/{pid}/statm') as handle:
                    return int(handle.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1e6
            except OSError:
                return 0.
        while not stop.is_set():
            time.sleep(60)
            children = subprocess.run(['pgrep', '-P', str(os.getpid())], capture_output=True, text=True).stdout.split()
            grand = subprocess.run(['pgrep', '-P', ','.join(children)], capture_output=True, text=True).stdout.split() if children else []
            print(json.dumps({'memory_mb': {'driver': round(rss(os.getpid())), 'children': round(sum(rss(p) for p in children)),
                                            'grandchildren': round(sum(rss(p) for p in grand)), 'slots_running': len(children)}}), flush=True)

    threading.Thread(target=memory_log, daemon=True).start()
    slots = [url for url, n in endpoints for _ in range(n)]
    threads = [threading.Thread(target=slot, args=(url, index * args.stagger), daemon=True) for index, url in enumerate(slots)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
    except KeyboardInterrupt:
        stop.set()
    unfinished = sum(1 for e in entries if not finished(e['key']))
    print(json.dumps({'unfinished': unfinished, 'of': len(entries)}), flush=True)
    if not args.finalize and not args.finalize_only:
        # a unit with Restart=on-failure relaunches and takes the unfinished episodes again (finished ones are skipped)
        raise SystemExit(3 if unfinished and not stop.is_set() else 0)
    with (out / 'journal.jsonl').open('w') as merged:
        for entry in entries:
            for path in journals(work, entry['key']):
                merged.write(path.read_text())
    report = task_rollouts.score(out, out / 'rollouts.jsonl')
    print(json.dumps({k: report[k] for k in ('rollouts', 'scored', 'successes', 'outcome_counts')}))


if __name__ == '__main__':
    main()
