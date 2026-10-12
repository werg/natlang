#!/usr/bin/env python3
"""Teacher serving throughput on one GPU (plans/COREWEAVE_HARDWARE_OPTIMIZATION.md, "Teacher serving").

For each server configuration (an engine command line: vLLM, SGLang or llama-server, each with its quantization,
speculative decoding and cache settings) this launches the server, waits until it answers, drives it with the
natlang workload shape at several concurrency levels, scrapes its Prometheus counters (prefix-cache hit rate,
preemptions), stops it and goes on. Results go to one JSON (schema ``natlang.teacher-bench/1``).

Workload: natlang program calls share a long prefix (system prompt, program header, tool schemas) and differ in a
short tail. ``--prompts FILE.jsonl`` replays real recorded chat requests (``{"messages": [...], "tools": [...]}``
per line); without it a synthetic prompt has a ``--prefix-tokens`` shared head (about 4 characters per token) and a
unique tail, so prefix caching is exercised the way our campaigns exercise it.

  scripts/coreweave/teacher_bench.py --configs scripts/coreweave/teacher-configs.json --out OUT.json
  scripts/coreweave/teacher_bench.py --base-url http://127.0.0.1:8082 --model M --out OUT.json   # a running server

Standard library only, so it runs in any serving image.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

SCHEMA = 'natlang.teacher-bench/1'
WORDS = ('function', 'returns', 'value', 'the', 'input', 'record', 'field', 'list', 'string', 'number', 'call',
         'tool', 'result', 'when', 'each', 'item', 'state', 'program', 'step', 'natural', 'language', 'type')


def synthetic_prompts(count: int, prefix_tokens: int, tail_tokens: int, seed: int = 0) -> list[dict]:
    """Requests with one shared prefix and distinct tails (about 4 characters per token)."""
    rng = random.Random(seed)
    head = ' '.join(rng.choice(WORDS) for _ in range(prefix_tokens * 4 // 6))
    out = []
    for i in range(count):
        tail = ' '.join(rng.choice(WORDS) for _ in range(tail_tokens * 4 // 6))
        out.append({'messages': [{'role': 'system', 'content': 'You execute natlang programs.\n' + head},
                                 {'role': 'user', 'content': f'Case {i}: {tail}\nAnswer with a short JSON value.'}]})
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def one_request(base: str, model: str, request: dict, max_tokens: int, timeout: float) -> dict:
    """Stream one chat completion: time to first token, total time, prompt/completion tokens from usage."""
    url = urllib.parse.urlparse(base)
    body = {'model': model, 'messages': request['messages'], 'max_tokens': max_tokens, 'temperature': 0.7,
            'stream': True, 'stream_options': {'include_usage': True}}
    if request.get('tools'):
        body['tools'] = request['tools']
    tick = time.perf_counter()
    first = None
    usage = {}
    chunks = 0
    try:
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
        conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            return {'ok': False, 'status': response.status, 'error': response.read()[:300].decode(errors='replace')}
        for raw in response:
            line = raw.decode(errors='replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            event = json.loads(data)
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or delta.get(
                        'tool_calls'):
                    chunks += 1
                    if first is None:
                        first = time.perf_counter()
        conn.close()
    except Exception as exc:
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}
    end = time.perf_counter()
    completion = usage.get('completion_tokens', chunks)
    return {'ok': True, 'seconds': end - tick, 'ttft': (first - tick) if first else None,
            'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': completion,
            'decode_tokens_per_second': (completion - 1) / (end - first) if first and end > first and completion > 1
            else None}


def metrics(base: str) -> dict:
    """Selected Prometheus counters (vLLM and SGLang names), summed over labels."""
    url = urllib.parse.urlparse(base)
    try:
        conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=10)
        conn.request('GET', '/metrics')
        text = conn.getresponse().read().decode(errors='replace')
    except Exception:
        return {}
    wanted = ('prefix_cache_hits', 'prefix_cache_queries', 'num_preemptions', 'generation_tokens',
              'prompt_tokens', 'cache_hit_rate', 'spec_decode_num_accepted_tokens', 'spec_decode_num_draft_tokens')
    out: dict = {}
    for line in text.splitlines():
        if line.startswith('#') or ' ' not in line:
            continue
        name, value = line.rsplit(' ', 1)
        base_name = name.split('{', 1)[0]
        for key in wanted:
            if base_name.endswith(key) or base_name.endswith(key + '_total'):
                try:
                    out[key] = out.get(key, 0.0) + float(value)
                except ValueError:
                    pass
    return out


def run_level(base, model, prompts, concurrency, requests, max_tokens, timeout) -> dict:
    queue = list(prompts[:requests])
    lock = threading.Lock()
    results = []

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                request = queue.pop()
            row = one_request(base, model, request, max_tokens, timeout)
            with lock:
                results.append(row)

    before = metrics(base)
    tick = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - tick
    after = metrics(base)
    ok = [r for r in results if r['ok']]
    out_tokens = sum(r['completion_tokens'] or 0 for r in ok)
    in_tokens = sum(r['prompt_tokens'] or 0 for r in ok)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    hit = (delta['prefix_cache_hits'] / delta['prefix_cache_queries']
           if delta.get('prefix_cache_queries') else None)
    ttfts = [r['ttft'] for r in ok if r['ttft'] is not None]
    decode = [r['decode_tokens_per_second'] for r in ok if r['decode_tokens_per_second']]
    return {'concurrency': concurrency, 'requests': len(results), 'ok': len(ok), 'wall_seconds': wall,
            'output_tokens_per_second': out_tokens / wall, 'prompt_tokens_per_second': in_tokens / wall,
            'ttft_p50': percentile(ttfts, .5), 'ttft_p95': percentile(ttfts, .95),
            'per_request_decode_tps_p50': percentile(decode, .5),
            'prefix_cache_hit_rate': hit, 'metrics_delta': delta,
            'errors': [r.get('error') or r.get('status') for r in results if not r['ok']][:5]}


def wait_ready(base: str, process, seconds: float) -> bool:
    url = urllib.parse.urlparse(base)
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=5)
            conn.request('GET', '/v1/models')
            if conn.getresponse().status == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def bench_server(config: dict, a, prompts) -> dict:
    """Launch one configuration (or use a running server), measure every level, stop it."""
    base = config.get('base_url', f"http://127.0.0.1:{config.get('port', 8000)}")
    record = {'name': config['name'], 'command': config.get('command'), 'notes': config.get('notes'), 'levels': []}
    process = None
    log = None
    if config.get('command'):
        log = open(Path(a.out).with_suffix('.' + config['name'] + '.log'), 'w')
        process = subprocess.Popen(config['command'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env=dict(os.environ, **config.get('env', {})))
    tick = time.time()
    try:
        if not wait_ready(base, process, a.startup_seconds):
            record['status'] = 'failed_to_start'
            return record
        record['startup_seconds'] = time.time() - tick
        one_request(base, config['model'], prompts[0], 16, a.request_timeout)  # warm the prefix and graphs
        for level in a.concurrency:
            row = run_level(base, config['model'], prompts, level, max(a.requests, level * 2), a.max_tokens,
                            a.request_timeout)
            record['levels'].append(row)
            print(json.dumps({'config': config['name'], **{k: row[k] for k in (
                'concurrency', 'ok', 'output_tokens_per_second', 'ttft_p50', 'prefix_cache_hit_rate')}}), flush=True)
        record['status'] = 'ok'
        if record['levels']:
            best = max(record['levels'], key=lambda r: r['output_tokens_per_second'])
            record['best'] = {'concurrency': best['concurrency'],
                              'output_tokens_per_second': best['output_tokens_per_second']}
            single = next((r for r in record['levels'] if r['concurrency'] == 1), None)
            if single:
                record['single_stream_decode_tps'] = single['per_request_decode_tps_p50']
    finally:
        if process is not None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        if log:
            log.close()
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--configs', type=Path, help='JSON list of {name, model, command[], port, env, notes}')
    p.add_argument('--only', help='comma-separated config names')
    p.add_argument('--base-url', help='measure an already running server instead')
    p.add_argument('--model', help='--base-url: the served model name')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompts', type=Path, help='JSONL of real chat requests ({"messages", "tools"})')
    p.add_argument('--prefix-tokens', type=int, default=6000)
    p.add_argument('--tail-tokens', type=int, default=400)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--concurrency', type=lambda s: [int(v) for v in s.split(',')], default=[1, 8, 32, 64, 128, 256])
    p.add_argument('--requests', type=int, default=64, help='requests per level (at least 2x the concurrency)')
    p.add_argument('--startup-seconds', type=float, default=1800)
    p.add_argument('--request-timeout', type=float, default=900)
    a = p.parse_args(argv)
    if a.prompts:
        prompts = [json.loads(line) for line in a.prompts.read_text().splitlines() if line.strip()]
    else:
        prompts = synthetic_prompts(max(a.requests, 2 * max(a.concurrency)), a.prefix_tokens, a.tail_tokens)
    if a.base_url:
        configs = [{'name': 'running', 'base_url': a.base_url, 'model': a.model}]
    else:
        configs = json.loads(a.configs.read_text())
        if a.only:
            configs = [c for c in configs if c['name'] in a.only.split(',')]
    document = {'schema': SCHEMA, 'started': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'workload': {'prompts': str(a.prompts) if a.prompts else 'synthetic', 'prefix_tokens': a.prefix_tokens,
                             'tail_tokens': a.tail_tokens, 'max_tokens': a.max_tokens},
                'gpu': subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                       '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
                if not a.base_url else None,
                'configs': []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for config in configs:
        document['configs'].append(bench_server(config, a, prompts))
        a.out.write_text(json.dumps(document, indent=1) + '\n')
    ranked = sorted((c for c in document['configs'] if c.get('best')),
                    key=lambda c: -c['best']['output_tokens_per_second'])
    document['ranking'] = [{'name': c['name'], **c['best'], 'single_stream_decode_tps': c.get('single_stream_decode_tps')}
                           for c in ranked]
    document['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    a.out.write_text(json.dumps(document, indent=1) + '\n')
    print(json.dumps({'event': 'teacher_bench_done', 'ranking': document['ranking']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
