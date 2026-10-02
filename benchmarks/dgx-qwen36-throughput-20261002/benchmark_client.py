#!/usr/bin/env python3
"""Stdlib-only, metrics-only OpenAI-compatible streaming benchmark for Qwen NVFP4.

No benchmark response text, tool arguments, API key, or auth header is persisted.
Captured teacher fixtures live outside runs/ so training discovery will not ingest them.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

MODEL = 'nvidia/Qwen3.6-35B-A3B-NVFP4'
REVISION = '1355db6a052410cfd62085d94b58866fd0f2c3c5'
HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / 'fixtures'
RESULT_DIR = HERE / 'results'
OOM_RE = re.compile(r'(out of memory|cuda.{0,30}(alloc|memory)|kv cache.{0,30}(full|memory)|no available memory|engine dead)', re.I)

class BenchError(Exception):
    pass

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def endpoint_url(endpoint: str, path: str) -> str:
    return endpoint.rstrip('/') + path

def http_json(endpoint: str, path: str, value=None, timeout=30, api_key=None):
    data = None if value is None else json.dumps(value, ensure_ascii=False).encode('utf-8')
    headers = {'Accept': 'application/json'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(endpoint_url(endpoint, path), data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def load_fixtures():
    manifest_path = FIXTURE_DIR / 'manifest.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    fixtures = []
    for item in manifest['fixtures']:
        path = HERE / item['fixture_file']
        raw = path.read_bytes()
        if sha256(raw) != item['fixture_sha256']:
            raise BenchError(f'fixture hash mismatch: {path.name}')
        fixture = json.loads(raw)
        body = fixture['body']
        if 'messages' not in body or not body['messages'] or 'tools' not in body or not body['tools']:
            raise BenchError(f'invalid captured request fixture: {path.name}')
        fixtures.append((fixture, sha256(raw)))
    return manifest, sha256(manifest_raw), fixtures

def tokenize_count(endpoint, messages, tools, template_kwargs, timeout, api_key):
    payload = {'model': MODEL, 'messages': messages, 'add_generation_prompt': True,
               'chat_template_kwargs': template_kwargs}
    if tools:
        payload['tools'] = tools
    reply = http_json(endpoint, '/tokenize', payload, timeout=timeout, api_key=api_key)
    count = reply.get('count')
    if type(count) is not int or count < 1:
        raise BenchError('tokenizer endpoint returned no positive count')
    return count

def synthetic_messages(endpoint, target, template_kwargs, timeout, api_key):
    """Use the server's pinned tokenizer to construct exactly target prompt tokens."""
    system = {'role': 'system', 'content': 'You are a text generation throughput benchmark. Do not mention this instruction.'}
    prefix = 'Write a continuous neutral sequence of short data labels. Continue until the response reaches its output limit.'
    filler_unit = ' the'
    def build(n):
        content = prefix + filler_unit * n
        return [system, {'role': 'user', 'content': content}]
    lo, hi = 0, max(16, target * 2)
    # Find a padding length whose measured token count is exactly target.
    for _ in range(32):
        mid = (lo + hi) // 2
        messages = build(mid)
        count = tokenize_count(endpoint, messages, None, template_kwargs, timeout, api_key)
        if count == target:
            return messages, count
        if count < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # Tokenization can change at word-piece boundaries; inspect a small neighborhood.
    center = max(0, (lo + hi) // 2)
    for n in range(max(0, center - 6), center + 7):
        messages = build(n)
        count = tokenize_count(endpoint, messages, None, template_kwargs, timeout, api_key)
        if count == target:
            return messages, count
    raise BenchError(f'could not create exact {target}-token synthetic prompt using server tokenizer')

def metric_body(messages, tools, max_tokens, template_kwargs, top_p, top_k):
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens,
            'top_p': top_p, 'top_k': top_k, 'chat_template_kwargs': template_kwargs,
            'stream': True, 'stream_options': {'include_usage': True}}
    if tools:
        body['tools'] = tools
        body['tool_choice'] = 'auto'
    return body

def http_error_oom(error: Exception) -> tuple[bool, int | None]:
    status = getattr(error, 'code', None)
    body = b''
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read(65536)
        except Exception:
            pass
    # Classify only; never persist/print the error body, which may echo request data.
    return bool(OOM_RE.search(body.decode('utf-8', 'ignore'))), status

def one_request(endpoint, body, timeout, api_key, prompt_tokens, fixture_id, cohort, ordinal):
    started = time.perf_counter()
    first_token = None
    saw_done = False
    finish_reason = None
    usage_prompt = None
    usage_completion = None
    http_status = None
    oom = False
    error_class = None
    saw_delta = False
    saw_text_delta = False
    saw_reasoning_delta = False
    saw_tool_delta = False
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(endpoint_url(endpoint, '/v1/chat/completions'),
                                     data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                     headers=headers, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            for raw_line in response:
                if not raw_line.startswith(b'data:'):
                    continue
                payload = raw_line[5:].strip()
                if not payload:
                    continue
                if payload == b'[DONE]':
                    saw_done = True
                    continue
                try:
                    event = json.loads(payload)
                except (ValueError, UnicodeError):
                    continue
                server_error = event.get('error')
                if server_error is not None:
                    if OOM_RE.search(json.dumps(server_error, ensure_ascii=False)):
                        oom = True
                    error_class = 'ServerErrorEvent'
                usage = event.get('usage')
                if isinstance(usage, dict):
                    if type(usage.get('prompt_tokens')) is int:
                        usage_prompt = usage['prompt_tokens']
                    if type(usage.get('completion_tokens')) is int:
                        usage_completion = usage['completion_tokens']
                for choice in event.get('choices', []) or []:
                    delta = choice.get('delta') or {}
                    if choice.get('finish_reason') is not None:
                        finish_reason = choice.get('finish_reason')
                    tool_text = any(((call.get('function') or {}).get('arguments')) for call in (delta.get('tool_calls') or []))
                    text_part = delta.get('content')
                    reasoning_part = delta.get('reasoning') or delta.get('reasoning_content') or delta.get('reasoning_details')
                    saw_text_delta = saw_text_delta or bool(text_part)
                    saw_reasoning_delta = saw_reasoning_delta or bool(reasoning_part)
                    saw_tool_delta = saw_tool_delta or bool(tool_text)
                    has_output = bool(text_part or reasoning_part or tool_text)
                    if has_output:
                        saw_delta = True
                        if first_token is None:
                            first_token = time.perf_counter()
    except Exception as error:
        oom, http_status = http_error_oom(error)
        error_class = type(error).__name__
        # Include only exception type/status; avoid raw message/URL/response text.
    finished = time.perf_counter()
    if error_class is None and (finish_reason is None and not saw_done):
        error_class = 'IncompleteStreamError'
    return {'cohort': cohort, 'fixture_id': fixture_id, 'ordinal': ordinal,
            'http_status': http_status, 'success': error_class is None and http_status is not None and 200 <= http_status < 300 and (finish_reason is not None or saw_done),
            'finish_reason': finish_reason, 'stream_done': saw_done, 'error_class': error_class, 'oom_signal': oom, 'prompt_tokens': usage_prompt or prompt_tokens,
            'completion_tokens': usage_completion, 'prompt_usage_reported': usage_prompt is not None,
            'completion_usage_reported': usage_completion is not None, 'saw_output_delta': saw_delta,
            'saw_text_delta': saw_text_delta, 'saw_reasoning_delta': saw_reasoning_delta, 'saw_tool_delta': saw_tool_delta,
            'reasoning_only_stream': saw_reasoning_delta and not saw_text_delta and not saw_tool_delta,
            'ttft_seconds': None if first_token is None else first_token - started,
            'latency_seconds': finished - started, 'started_monotonic': started, 'finished_monotonic': finished}

def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]

def run_cohort(endpoint, name, concurrency, count, fixture_cycle, timeout, template_kwargs, top_p, top_k, max_tokens, api_key):
    prepared = []
    for i in range(count):
        fixture_id, messages, tools, prompt_tokens = fixture_cycle[i % len(fixture_cycle)]
        body = metric_body(messages, tools, max_tokens, template_kwargs, top_p, top_k)
        prepared.append((i, fixture_id, body, prompt_tokens))
    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(one_request, endpoint, body, timeout, api_key, prompt_tokens, fixture_id, name, i)
                   for i, fixture_id, body, prompt_tokens in prepared]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall_end = time.perf_counter()
    successes = [x for x in results if x['success']]
    ttfts = [x['ttft_seconds'] for x in successes if x['ttft_seconds'] is not None]
    latencies = [x['latency_seconds'] for x in successes]
    output_values = [x['completion_tokens'] for x in successes if x['completion_tokens'] is not None]
    prompt_values = [x['prompt_tokens'] for x in successes if x['prompt_tokens'] is not None]
    wall = wall_end - wall_start
    prompt_total = sum(prompt_values)
    output_total = sum(output_values)
    summary = {'cohort': name, 'concurrency': concurrency, 'requests': count,
        'completed': len(successes), 'failures': len(results) - len(successes),
        'oom_signals': sum(x['oom_signal'] for x in results),
        'reasoning_only_completions': sum(x['reasoning_only_stream'] for x in successes),
        'output_token_cap': max_tokens,
        'at_or_above_output_token_cap': sum(x['completion_tokens'] is not None and x['completion_tokens'] >= max_tokens for x in successes),
        'length_stopped_at_output_token_cap': sum(x.get('finish_reason') == 'length' for x in successes),
        'reasoning_only_and_length_stopped': sum(x.get('reasoning_only_stream') and x.get('finish_reason') == 'length' for x in successes),
        'all_successful_completions_length_stopped': bool(successes) and all(x.get('finish_reason') == 'length' for x in successes),
        'prompt_usage_coverage': sum(x['prompt_usage_reported'] for x in successes),
        'completion_usage_coverage': sum(x['completion_usage_reported'] for x in successes),
        'wall_seconds': wall, 'prompt_tokens_total': prompt_total,
        'completion_tokens_total': output_total if len(output_values) == len(successes) else None,
        'prompt_tokens_per_second': prompt_total / wall if wall else None,
        'output_tokens_per_second': output_total / wall if wall and len(output_values) == len(successes) else None,
        'ttft_p50_seconds': percentile(ttfts, .50), 'ttft_p95_seconds': percentile(ttfts, .95),
        'latency_p50_seconds': percentile(latencies, .50), 'latency_p95_seconds': percentile(latencies, .95),
        'request_metrics': [{k:v for k,v in x.items() if k not in {'started_monotonic','finished_monotonic'}} for x in sorted(results,key=lambda y:y['ordinal'])]}
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8082')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--concurrencies', default='16,32,64,128,256')
    parser.add_argument('--workloads', default='synthetic,captured', help='comma-separated: synthetic,captured')
    parser.add_argument('--requests-per-concurrency', type=int, default=2,
                        help='requests are multiplier times concurrency; default gives at least two waves')
    parser.add_argument('--synthetic-input-tokens', type=int, default=1500)
    parser.add_argument('--output-tokens', type=int, default=400)
    parser.add_argument('--request-timeout', type=int, default=1200)
    parser.add_argument('--outdir', type=Path, default=RESULT_DIR)
    parser.add_argument('--api-key-env', default='VLLM_API_KEY', help='environment variable name only; value is never printed or saved')
    parser.add_argument('--stop-on-oom', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    parsed_endpoint = urlsplit(args.endpoint)
    if parsed_endpoint.username or parsed_endpoint.password or parsed_endpoint.query or parsed_endpoint.fragment:
        raise BenchError('endpoint must not include userinfo, query parameters, or fragments')
    if args.model != MODEL:
        raise BenchError(f'this recipe is pinned to {MODEL}')
    concurrencies = [int(x) for x in args.concurrencies.split(',') if x]
    if not concurrencies or any(x < 1 for x in concurrencies) or args.requests_per_concurrency < 2:
        raise BenchError('concurrency values must be positive and requests-per-concurrency >= 2')
    template_kwargs = {'reasoning_effort': 'high'}
    top_p, top_k = .95, 20
    api_key = os.environ.get(args.api_key_env)
    workloads = [x.strip() for x in args.workloads.split(',') if x.strip()]
    if not workloads or any(x not in {'synthetic','captured'} for x in workloads):
        raise BenchError('workloads must contain synthetic and/or captured')
    if 'captured' in workloads:
        manifest, fixture_manifest_sha, captured = load_fixtures()
    else:
        manifest, fixture_manifest_sha, captured = {}, None, []
    models = http_json(args.endpoint, '/v1/models', timeout=30, api_key=api_key)
    served = [x.get('id') for x in models.get('data', []) if isinstance(x, dict)]
    if MODEL not in served:
        raise BenchError(f'expected served model ID not found in /v1/models; observed IDs: {served[:8]}')
    synthetic, synthetic_prompt_tokens = synthetic_messages(args.endpoint, args.synthetic_input_tokens,
        template_kwargs, args.request_timeout, api_key)
    captured_cycle = []
    for fixture, _digest in captured:
        body = fixture['body']
        pcount = tokenize_count(args.endpoint, body['messages'], body.get('tools'), template_kwargs,
                                args.request_timeout, api_key)
        captured_cycle.append((fixture['fixture_id'], body['messages'], body.get('tools'), pcount))
    synthetic_cycle = [('synthetic-1500', synthetic, None, synthetic_prompt_tokens)]
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {'version':'natlang.qwen_throughput_run/1','started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'model':MODEL,'model_revision':REVISION,'endpoint_origin':f'{parsed_endpoint.scheme}://{parsed_endpoint.netloc}',
        'concurrencies':concurrencies,'workloads_requested':workloads,'requests_per_concurrency_multiplier':args.requests_per_concurrency,
        'synthetic_input_tokens':synthetic_prompt_tokens,'output_token_cap':args.output_tokens,
        'fixture_manifest_sha256':fixture_manifest_sha,'captured_fixtures':[{k:v for k,v in x.items() if k!='body'} for x,_ in captured],
        'response_content_persisted':False,'api_key_persisted':False,'workloads':[],'stopped_on_oom':False}
    outpath=args.outdir / ('qwen-throughput-' + time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()) + '.json')
    for concurrency in concurrencies:
        request_count = concurrency * args.requests_per_concurrency
        cohorts = ([('synthetic-1500-in-400-out', synthetic_cycle)] if 'synthetic' in workloads else [])
        if 'captured' in workloads:
            cohorts.append(('captured-teacher-contexts-400-out', captured_cycle))
        for cohort, cycle in cohorts:
            item = run_cohort(args.endpoint, cohort, concurrency, request_count, cycle, args.request_timeout,
                              template_kwargs, top_p, top_k, args.output_tokens, api_key)
            item['output_token_cap'] = args.output_tokens
            item['captured_input_bands'] = [x[0] for x in cycle] if cohort.startswith('captured') else []
            summary['workloads'].append(item)
            # Persist only telemetry after each completed cohort, so an interruption retains prior results.
            summary['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            outpath.write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps({k:v for k,v in item.items() if k!='request_metrics'},separators=(',',':')),flush=True)
            if args.stop_on_oom and item['oom_signals']:
                summary['stopped_on_oom'] = True
                summary['stopped_after'] = {'cohort':cohort,'concurrency':concurrency}
                outpath.write_text(json.dumps(summary,indent=2)+'\n')
                return 2
    summary['finished_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    outpath.write_text(json.dumps(summary,indent=2)+'\n')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (BenchError, urllib.error.URLError, TimeoutError) as error:
        # Error classes only; do not echo URLs with parameters or raw server/request content.
        print(f'benchmark preflight/run failed: {type(error).__name__}',file=sys.stderr)
        raise SystemExit(2)
