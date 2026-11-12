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
from natlang_neuralese.rl import task_rollouts  # noqa: E402
from natlang_neuralese.rl.actor_client import NativeActorClient  # noqa: E402
from natlang_neuralese.rl.held_eval import SEED_BASE, series  # noqa: E402


def attest(endpoint, policy_id):
    info = json.loads(NativeActorClient(endpoint, timeout=10).get_bytes('/v1/neuralese/info'))
    if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') != policy_id:
        raise SystemExit(f'{endpoint} does not serve {policy_id[:12]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--policy-id', required=True)
    ap.add_argument('--endpoint', action='append', required=True, help='URL=SLOTS')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--case-seconds', type=int, default=3600)
    ap.add_argument('--max-model-requests', type=int, default=64)
    ap.add_argument('--max-turns', type=int, default=128)
    ap.add_argument('--context-tokens', type=int, default=32768)
    ap.add_argument('--seed-base', type=int, default=SEED_BASE)
    ap.add_argument('--finalize-only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    endpoints = [(item.rsplit('=', 1)[0].rstrip('/'), int(item.rsplit('=', 1)[1])) for item in args.endpoint]
    for url, _ in endpoints:
        attest(url, args.policy_id)
    if not (out / 'queue.jsonl').exists():
        rows = sum(1 for _ in read_jsonl(args.cases))
        task_rollouts.prepare(SimpleNamespace(
            cases=args.cases, out=out, rows=rows, offset=0, group=args.samples, temperature=1.,
            case_seconds=args.case_seconds, max_model_requests=args.max_model_requests, max_turns=args.max_turns,
            seed_base=args.seed_base, root_mode='student', match_seeds_from=None, policy_checkpoint_sha256=args.policy_id))
        entries = list(read_jsonl(out / 'queue.jsonl'))
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=args.context_tokens)
        entries.sort(key=lambda e: (e['sample'], e['index']))   # sample-major: a cut-short run stays balanced
        (out / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
    entries = list(read_jsonl(out / 'queue.jsonl'))
    (out / 'sampling-config.json').write_text(json.dumps({'guidance': False, 'x_natlang_rl_capture': True}) + '\n')
    spec = {'series': series(), 'endpoints': endpoints, 'policy_id': args.policy_id, 'samples': args.samples,
            'started': time.time()}
    (out / 'collect-run.json').write_text(json.dumps(spec, indent=1) + '\n')
    work = out / 'workers'
    work.mkdir(exist_ok=True)

    def finished(key):
        path = work / f'{key}.journal.jsonl'
        return path.exists() and any(row.get('event') == 'finish' for row in read_jsonl(path))

    todo = [e for e in entries if not finished(e['key'])] if not args.finalize_only else []
    lock, cursor = threading.Lock(), [0]
    stop = threading.Event()
    stats = {'done': 0, 'started': time.time()}
    env = {**os.environ, 'NATLANG_MODEL_FACING': spec['series']['model_facing']}

    def slot(url):
        while not stop.is_set():
            with lock:
                if cursor[0] >= len(todo):
                    return
                entry = todo[cursor[0]]
                cursor[0] += 1
            key = entry['key']
            queue, journal = work / f'{key}.queue.jsonl', work / f'{key}.journal.jsonl'
            queue.write_text(json.dumps({**entry, 'rl_endpoint': url}, sort_keys=True) + '\n')
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue), str(journal), '--runtime',
                       str(REPO / 'ts-host'), '--server', url, '--chat-completions-url', url + '/v1/chat/completions',
                       '--model-id', 'natlang-rl', '--model-concurrency', '1', '--chat-request-config',
                       str(out / 'sampling-config.json'), '--neuralese-server']
            with (work / f'{key}.log').open('a') as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                stats['done'] += 1
                if stats['done'] % 8 == 0:
                    print(json.dumps({'done': stats['done'], 'of': len(todo),
                                      'minutes': round((time.time() - stats['started']) / 60, 1)}), flush=True)

    threads = [threading.Thread(target=slot, args=(url,), daemon=True) for url, n in endpoints for _ in range(n)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
    except KeyboardInterrupt:
        stop.set()
    with (out / 'journal.jsonl').open('w') as merged:
        for entry in entries:
            path = work / f"{entry['key']}.journal.jsonl"
            if path.exists():
                merged.write(path.read_text())
    report = task_rollouts.score(out, out / 'rollouts.jsonl')
    print(json.dumps({k: report[k] for k in ('rollouts', 'scored', 'successes', 'outcome_counts')}))


if __name__ == '__main__':
    main()
