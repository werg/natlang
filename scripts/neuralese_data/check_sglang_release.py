#!/usr/bin/env python3
"""Smoke: the colocated GRPO actor's memory release (grpo.py --actor-release, serve_native_checkpoint.py
--sglang-memory-saver) on the pinned SGLang fork.

Starts the fork's server on an HF export with --enable-memory-saver and a small pool, then checks, as
``grpo.WeightSync.release``/``publish`` drive it: a request in flight survives pause(retract) -> release(tags) ->
resume -> update_weights_from_disk -> continue and finishes; the engine's GPU memory drops on release; greedy token ids
and logprobs after the cycle equal those before it (released weights come back undefined and are reloaded from the
file). Two cycles. Prints one JSON verdict; exit 1 on a failed check.

    check_sglang_release.py --export WORK/actor/sglang-hf [--tags kv_cache,weights,cuda_graph]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def post(url, body=None, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(body or {}).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b'null')


def gpu_mb(pids):
    """GPU memory (MiB) of the given processes and their descendants, from nvidia-smi."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    tree = set(pids)
    for _ in range(3):
        for line in subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True).stdout.splitlines():
            pid, ppid = map(int, line.split())
            if ppid in tree:
                tree.add(pid)
    return sum(int(used) for pid, used in (line.split(', ') for line in out.strip().splitlines()) if int(pid) in tree)


def greedy(url, ids, n=48):
    out = post(url + '/generate', {'input_ids': ids, 'return_logprob': True,
                                   'sampling_params': {'temperature': 0, 'max_new_tokens': n, 'ignore_eos': True}})
    meta = out['meta_info']
    return [t[1] for t in meta['output_token_logprobs']], [t[0] for t in meta['output_token_logprobs']]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', required=True, help='HF export directory (WORK/actor/sglang-hf)')
    p.add_argument('--tags', default='kv_cache,weights,cuda_graph')
    p.add_argument('--port', type=int, default=9611)
    p.add_argument('--max-total-tokens', type=int, default=65536)
    p.add_argument('--mem-fraction', type=float, default=0.08)
    p.add_argument('--hold-s', type=float, default=20., help='seconds the engine stays released')
    args = p.parse_args()
    fork = json.loads((ROOT / 'training/neuralese/sglang-fork.json').read_text())
    python = fork['python']
    cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'training/neuralese'), 'CUDA_HOME': cuda,
           'PATH': os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')]),
           'NATLANG_SGLANG_PROJECTION': 'vendor', 'NATLANG_ADAPTER_SLOTS': '0', 'NATLANG_ADAPTER_RANK': '32'}
    url = f'http://127.0.0.1:{args.port}'
    command = [python, '-m', 'sglang.launch_server', '--model-path', args.export, '--host', '127.0.0.1',
               '--port', str(args.port), '--dtype', 'bfloat16', '--mem-fraction-static', str(args.mem_fraction),
               '--context-length', '8192', '--max-total-tokens', str(args.max_total_tokens), '--max-running-requests', '8',
               '--kv-cache-dtype', 'fp8_e4m3', '--mamba-radix-cache-strategy', 'extra_buffer',
               '--max-mamba-cache-size', '64', '--cuda-graph-max-bs-decode', '8', '--skip-tokenizer-init',
               '--trust-remote-code', '--swa-full-tokens-ratio', '0.5', '--disable-overlap-schedule',
               '--enable-memory-saver']
    server = subprocess.Popen(command, env=env, start_new_session=True)
    verdict = {'tags': args.tags.split(','), 'checks': {}}
    try:
        deadline = time.time() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited {server.returncode}')
            try:
                with urllib.request.urlopen(url + '/health', timeout=5):
                    break
            except OSError:
                if time.time() > deadline:
                    raise TimeoutError('server did not come up')
                time.sleep(5)
        prompt = [1] + list(range(2000, 2300))
        ids0, lp0 = greedy(url, prompt)
        for cycle in range(2):
            flight = {}

            def long_request():
                flight['out'] = post(url + '/generate', {'input_ids': prompt[:120],
                                     'sampling_params': {'temperature': 1.0, 'max_new_tokens': 1500, 'ignore_eos': True}},
                                     timeout=1800)
            thread = threading.Thread(target=long_request)
            thread.start()
            time.sleep(1.5)
            before = gpu_mb([server.pid])
            t = time.time()
            post(url + '/pause_generation', {'mode': 'retract'})
            post(url + '/release_memory_occupation', {'tags': verdict['tags']})
            release_s = time.time() - t
            time.sleep(2)
            released = gpu_mb([server.pid])
            time.sleep(args.hold_s)
            waiting = thread.is_alive()
            t = time.time()
            post(url + '/resume_memory_occupation', {'tags': verdict['tags']})
            resume_s = time.time() - t
            post(url + '/pause_generation', {'mode': 'retract'})
            update = post(url + '/update_weights_from_disk', {'model_path': args.export, 'flush_cache': True,
                                                               'weight_version': f'smoke-{cycle}'})
            post(url + '/continue_generation', {})
            update_s = time.time() - t
            thread.join(1800)
            done = len((flight.get('out') or {}).get('output_ids') or [])
            ids1, lp1 = greedy(url, prompt)
            verdict['checks'][f'cycle{cycle}'] = {
                'gpu_mb_before': before, 'gpu_mb_released': released, 'freed_mb': before - released,
                'release_s': round(release_s, 2), 'resume_and_update_s': round(update_s, 2), 'resume_s': round(resume_s, 2),
                'update_success': bool(update.get('success')), 'in_flight_waited': waiting,
                'in_flight_tokens': done, 'greedy_ids_equal': ids1 == ids0,
                'max_abs_logprob_diff': max(abs(a - b) for a, b in zip(lp0, lp1)) if ids1 == ids0 else None}
        ok = all(c['update_success'] and c['in_flight_waited'] and c['in_flight_tokens'] == 1500 and c['greedy_ids_equal']
                 and c['max_abs_logprob_diff'] == 0 and c['freed_mb'] > 1000 for c in verdict['checks'].values())
        verdict['ok'] = ok
        print(json.dumps(verdict), flush=True)
        return 0 if ok else 1
    finally:
        import signal
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(60)
        except Exception:
            os.killpg(server.pid, signal.SIGKILL)


if __name__ == '__main__':
    sys.exit(main())
