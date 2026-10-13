#!/usr/bin/env python3
"""Measure a model-facing error text live: capture real trajectories, then replay the turn after an error per variant.

AGENTS.md "Measure model-facing changes live": a changed error text is sampled on the executor (about 48 samples per
variant) before it is adopted. This tool does it for errors that end a tool result:

  proxy    an OpenAI chat-completions proxy that logs every request and reply (JSONL) while a collector runs against it
  contexts pick, from captured requests, each request whose last tool message holds an error matched by --pattern;
           write the request up to that tool message as a replay context (one per distinct conversation)
  replay   for every context and variant, put the variant's text in place of the matched error and sample the next
           assistant turn N times; classify each reply (see classify) and write per-variant rates

Variant texts come from the runtime itself (``--texts``: JSON {variant: {context-id or "*": text}} produced by
``ts-host/scripts/return-error-texts.mjs``), so what is measured is what the runtime renders.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.server
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, body: dict, timeout: float = 900) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def proxy(args) -> None:
    lock = threading.Lock()
    log = Path(args.log).open('a', encoding='utf-8')

    class Handler(http.server.BaseHTTPRequestHandler):
        def _forward(self, method: str):
            length = int(self.headers.get('Content-Length') or 0)
            data = self.rfile.read(length) if length else None
            request = urllib.request.Request(args.upstream.rstrip('/') + self.path, data=data, method=method,
                                             headers={'Content-Type': self.headers.get('Content-Type', 'application/json')})
            try:
                with urllib.request.urlopen(request, timeout=3600) as response:
                    payload, status = response.read(), response.status
                    kind = response.headers.get('Content-Type', 'application/json')
            except urllib.error.HTTPError as error:
                payload, status, kind = error.read(), error.code, error.headers.get('Content-Type', 'application/json')
            if method == 'POST' and self.path.endswith('/chat/completions'):
                def parsed(raw):
                    try:
                        return json.loads(raw or b'{}')
                    except ValueError:  # a streamed (SSE) reply is kept as text
                        return raw.decode('utf-8', 'replace')
                row = {'request': parsed(data), 'status': status, 'response': parsed(payload)}
                with lock:
                    log.write(json.dumps(row, ensure_ascii=False) + '\n')
                    log.flush()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            self._forward('POST')

        def do_GET(self):
            self._forward('GET')

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'proxy http://127.0.0.1:{server.server_address[1]} -> {args.upstream}', flush=True)
    server.serve_forever()


def tool_text(message: dict) -> str:
    content = message.get('content')
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    return content or ''


def contexts(args) -> None:
    pattern = re.compile(args.pattern)
    seen, out = set(), []
    for line in Path(args.log).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        request = row.get('request') or {}
        messages = request.get('messages') or []
        if not messages or messages[-1].get('role') != 'tool':
            continue
        text = tool_text(messages[-1])
        match = pattern.search(text)
        if not match:
            continue
        # One context per conversation (its first user turn) and error occurrence count: the first rejection is the
        # one a fixed text must repair; later ones are already loops.
        key = (json.dumps(messages[:2], sort_keys=True), args.first_only or len(messages))
        if key in seen:
            continue
        seen.add(key)
        out.append({'id': f'{Path(args.log).stem}-{len(out):03d}', 'matched': match.group(0), 'request': request})
    Path(args.out).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in out), encoding='utf-8')
    print(json.dumps({'contexts': len(out), 'out': args.out}))


ECHO = re.compile(r'bad-action|bad_action')


def classify(reply: dict, wrong_fields: list[str]) -> str:
    """recovered: a return_result whose arguments carry value, or any eval/read_code (work toward the result);
    echo: the reply repeats the diagnostic token (`bad-action`) as code, a value or text; repeat: return_result again
    without value; text: no tool call; other: any other tool call."""
    message = (reply.get('choices') or [{}])[0].get('message') or {}
    calls = message.get('tool_calls') or []
    # The action only: reasoning that quotes the error is not an echo.
    blob = json.dumps({'content': message.get('content'), 'tool_calls': calls}, ensure_ascii=False)
    if ECHO.search(blob):
        return 'echo'
    if not calls:
        return 'text'
    call = calls[0].get('function') or {}
    name = call.get('name')
    try:
        arguments = json.loads(call.get('arguments') or '{}')
    except ValueError:
        arguments = {}
    if name == 'return_result':
        return 'recovered' if isinstance(arguments, dict) and 'value' in arguments else 'repeat'
    if name in ('eval', 'read_code', 'read_page'):
        return 'recovered'
    return 'other'


def classify_binding(reply: dict, name: str) -> str:
    """After an invalid-binding error for `name`: recovered (an eval that uses name's members without declaring name),
    redeclare (declares name again), call_as_function (calls name(...)), return (return_result), text, other."""
    message = (reply.get('choices') or [{}])[0].get('message') or {}
    calls = message.get('tool_calls') or []
    if not calls:
        return 'text'
    call = calls[0].get('function') or {}
    if call.get('name') == 'return_result':
        return 'return'
    try:
        code = json.loads(call.get('arguments') or '{}').get('code') or ''
    except (ValueError, AttributeError):
        code = ''
    if call.get('name') != 'eval':
        return 'other'
    if re.search(rf'\b(?:const|let|var)\s+{re.escape(name)}\b', code):
        return 'redeclare'
    if re.search(rf'(?<![.\w$]){re.escape(name)}\s*\(', code):
        return 'call_as_function'
    if re.search(rf'(?<![.\w$]){re.escape(name)}\s*\.', code):
        return 'recovered'
    return 'other'


def label(reply: dict, row: dict) -> str:
    return classify_binding(reply, row['name']) if row.get('classifier') == 'binding' else classify(reply, [])


def replay(args) -> None:
    texts = json.loads(Path(args.texts).read_text())
    rows = [json.loads(line) for line in Path(args.contexts).read_text().splitlines() if line.strip()]
    pattern = re.compile(args.pattern)
    jobs = []
    for row in rows:
        for variant, by_context in texts.items():
            text = by_context.get(row['id']) or by_context.get('*')
            if text is None:
                continue
            request = json.loads(json.dumps(row['request']))
            last = request['messages'][-1]
            original = tool_text(last)
            replaced = pattern.sub(lambda _m: text, original, count=1)
            last['content'] = replaced
            for sample in range(args.samples):
                body = {**{k: v for k, v in request.items() if k != 'stream_options'},
                        'temperature': args.temperature, 'seed': 1000 + sample, 'stream': False}
                if args.max_tokens:
                    body['max_tokens'] = args.max_tokens
                if args.model:
                    body['model'] = args.model
                jobs.append((row, variant, sample, body))
    results = []

    def run(job):
        row, variant, sample, body = job
        try:
            reply = post(args.url, body)
            kind = label(reply, row)
            message = (reply.get('choices') or [{}])[0].get('message')
        except Exception as error:  # noqa: BLE001 - recorded, not counted
            kind, message = 'error', {'error': str(error)}
        return {'context': row['id'], 'variant': variant, 'sample': sample, 'label': kind, 'message': message,
                **({'classifier': 'binding', 'name': row['name']} if row.get('classifier') == 'binding' else {})}

    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for result in pool.map(run, jobs):
            results.append(result)
            if len(results) % 16 == 0:
                print(f'{len(results)}/{len(jobs)}', flush=True)
    Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results), encoding='utf-8')
    summary = summarize(results)
    print(json.dumps(summary, indent=1))
    Path(args.out).with_suffix('.summary.json').write_text(json.dumps(summary, indent=1))


def summarize(results: list[dict]) -> dict:
    summary = {}
    for result in results:
        counts = summary.setdefault(result['variant'], {})
        counts[result['label']] = counts.get(result['label'], 0) + 1
    for counts in summary.values():
        n = sum(v for k, v in counts.items() if k != 'error')
        counts['n'] = n
        counts['recovered_rate'] = round(counts.get('recovered', 0) / n, 3) if n else None
        counts['echo_rate'] = round(counts.get('echo', 0) / n, 3) if n else None
    return summary


def relabel(args) -> None:
    """Re-classify stored replies (after a classifier change) and rewrite the summary."""
    path = Path(args.results)
    results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for result in results:
        if result['label'] != 'error':
            result['label'] = label({'choices': [{'message': result['message']}]}, result)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results), encoding='utf-8')
    summary = summarize(results)
    path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('proxy')
    p.add_argument('--upstream', required=True)
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--log', required=True)
    p = sub.add_parser('contexts')
    p.add_argument('--log', required=True)
    p.add_argument('--pattern', required=True, help='regex matching the error text in the last tool message')
    p.add_argument('--first-only', action='store_true', help='only the first matching error of a conversation')
    p.add_argument('--out', required=True)
    p = sub.add_parser('replay')
    p.add_argument('--contexts', required=True)
    p.add_argument('--texts', required=True)
    p.add_argument('--pattern', required=True)
    p.add_argument('--url', required=True, help='chat completions URL')
    p.add_argument('--model', default=None)
    p.add_argument('--samples', type=int, default=48)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--out', required=True)
    p = sub.add_parser('relabel')
    p.add_argument('--results', required=True)
    args = parser.parse_args(argv)
    {'proxy': proxy, 'contexts': contexts, 'replay': replay, 'relabel': relabel}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
