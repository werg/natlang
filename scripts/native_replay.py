#!/usr/bin/env python3
"""Replay recorded natlang model requests against chat endpoints: latency, and agreement between two backends.

Input: JSONL turn records with a `wireRequest` (what natlang sent; tune collection writes these). Each request is sent
greedily (temperature 0) to every endpoint in turn. Reports per-endpoint latency, prompt tokens and, with two
endpoints, how many replies are identical and where the first difference falls (docs/NATIVE_MODELS.md).

    native_replay.py TURNS.jsonl --endpoint http://127.0.0.1:20123 [--endpoint http://127.0.0.1:8099 --request-extra JSON]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request


def post(endpoint, body, timeout):
    request = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'content-type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    return time.monotonic() - started, reply


def reply_text(reply):
    message = reply['choices'][0]['message']
    calls = message.get('tool_calls') or []
    return json.dumps({'content': message.get('content') or '', 'tool_calls': [
        {'name': (c.get('function') or {}).get('name'), 'arguments': (c.get('function') or {}).get('arguments')} for c in calls]},
        sort_keys=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('turns')
    parser.add_argument('--endpoint', action='append', required=True)
    parser.add_argument('--request-extra', action='append', default=[],
                        help='JSON merged into requests for the endpoint at the same position (e.g. chat_template_kwargs)')
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--out', help='write per-request results as JSONL')
    args = parser.parse_args(argv)
    extras = [json.loads(value) for value in args.request_extra] + [{}] * len(args.endpoint)
    rows = []
    with open(args.turns) as source:
        for line in source:
            record = json.loads(line)
            if 'wireRequest' in record:
                rows.append(record['wireRequest'])
            if len(rows) >= args.limit:
                break
    results = []
    for index, wire in enumerate(rows):
        body = {key: value for key, value in wire.items() if key not in ('stream', 'stream_options')}
        body.update(temperature=0, max_tokens=min(args.max_tokens, body.get('max_tokens') or args.max_tokens))
        row = {'index': index, 'endpoints': []}
        for endpoint, extra in zip(args.endpoint, extras):
            try:
                seconds, reply = post(endpoint, {**body, **extra}, args.timeout)
                usage = reply.get('usage') or {}
                row['endpoints'].append({'seconds': round(seconds, 3), 'prompt_tokens': usage.get('prompt_tokens'),
                                         'completion_tokens': usage.get('completion_tokens'), 'reply': reply_text(reply)})
            except Exception as error:  # noqa: BLE001 - report and continue
                row['endpoints'].append({'error': f'{type(error).__name__}: {error}'})
        results.append(row)
        print(json.dumps({'index': index, **{f'e{i}_s': e.get('seconds', e.get('error')) for i, e in enumerate(row['endpoints'])}}),
              file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, 'w') as out:
            for row in results:
                out.write(json.dumps(row) + '\n')
    summary = {'requests': len(results), 'endpoints': []}
    for position, endpoint in enumerate(args.endpoint):
        ok = [row['endpoints'][position] for row in results if 'seconds' in row['endpoints'][position]]
        times = [item['seconds'] for item in ok]
        summary['endpoints'].append({'endpoint': endpoint, 'ok': len(ok), 'errors': len(results) - len(ok),
            'first_s': times[0] if times else None, 'median_s': statistics.median(times) if times else None,
            'median_rest_s': statistics.median(times[1:]) if len(times) > 1 else None,
            'median_prompt_tokens': statistics.median([i['prompt_tokens'] for i in ok if i.get('prompt_tokens')] or [0])})
    if len(args.endpoint) >= 2:
        both = [row for row in results if all('reply' in e for e in row['endpoints'][:2])]
        same = [row for row in both if row['endpoints'][0]['reply'] == row['endpoints'][1]['reply']]
        def first_difference(a, b):
            return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        summary['agreement'] = {'compared': len(both), 'identical': len(same),
            'prompt_tokens_equal': sum(row['endpoints'][0]['prompt_tokens'] == row['endpoints'][1]['prompt_tokens'] for row in both),
            'first_difference_chars': [first_difference(row['endpoints'][0]['reply'], row['endpoints'][1]['reply'])
                                       for row in both if row not in same]}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
