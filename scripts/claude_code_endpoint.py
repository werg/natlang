#!/usr/bin/env python3
"""Claude Code's models as an OpenAI-compatible chat-completions endpoint (tool calls included).

The natlang runtime, the collector and teacher_gen reach hosted models through a provider or an OpenAI-compatible
endpoint. Anthropic's models are available here through the Claude Code subscription only (the API account has no
credits), so this service runs each request as one headless Claude Code call (`claude -p`, no tools of its own, no
session, the request's own system prompt) and answers in the OpenAI shape:

  claude_code_endpoint.py serve [--port 8790] [--concurrency 8]
  profile: {"endpoint": "http://127.0.0.1:8790", "model": "claude-opus-5-5"}
  teacher: {"model_id": "claude-opus-5-5", "endpoint": "http://127.0.0.1:8790/v1"}

Each request's messages are rendered as one transcript; the offered tools (name, description, JSON-schema parameters)
are listed after the request's system prompt, and the reply is Claude Code's structured output
`{"tool": NAME, "arguments": {...}}` (one call per turn) or `{"tool": "none", "text": ...}` for a plain reply. A
request without tools gets the plain text reply. A subscription usage limit answers HTTP 429 with an error message
starting "Claude Code usage limit" (teacher_gen treats it as a temporary provider limit, not a teacher verdict).
`reasoning_effort` (low, medium, high, xhigh, max) passes as `--effort`. Usage counts come from Claude Code's result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NONE = 'none'
LIMIT = re.compile(r'usage limit|hit your limit|limit reached|rate limit|overloaded|resets? at', re.I)
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
CALL_INSTRUCTIONS = """

# How you act in this conversation

You act only through the tools listed below. Each reply makes exactly one move: call one tool, written as the
structured output {"tool": TOOL_NAME, "arguments": {...}} whose arguments follow that tool's JSON schema. Reply with
{"tool": "none", "text": "..."} only when no tool call fits.

Tools:
"""


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
    return '' if content is None else json.dumps(content)


def render(messages: list[dict]) -> tuple[str, str]:
    """(system prompt, transcript prompt) of an OpenAI message list."""
    system = '\n\n'.join(text_of(m.get('content')) for m in messages if m.get('role') in ('system', 'developer'))
    parts = []
    for message in messages:
        role = message.get('role')
        if role in ('system', 'developer'):
            continue
        body = text_of(message.get('content'))
        if role == 'assistant':
            for call in message.get('tool_calls') or []:
                fn = call.get('function') or {}
                body += f'\n<tool_call name="{fn.get("name", "")}">{fn.get("arguments", "{}")}</tool_call>'
            reasoning = message.get('reasoning_content') or message.get('reasoning')
            if reasoning:
                body = f'<thinking>{reasoning}</thinking>\n' + body
        elif role == 'tool':
            name = message.get('name') or message.get('tool_call_id') or ''
            parts.append(f'<message role="tool" name="{name}">\n{body}\n</message>')
            continue
        parts.append(f'<message role="{role}">\n{body.strip()}\n</message>')
    prompt = ('The conversation so far, oldest first:\n\n' + '\n\n'.join(parts) +
              '\n\nWrite the assistant\'s next message.')
    return system, prompt


def tool_list(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        fn = tool.get('function') or tool
        lines.append(json.dumps({'name': fn.get('name'), 'description': fn.get('description', ''),
                                 'parameters': fn.get('parameters') or {}}, ensure_ascii=False))
    return '\n'.join(lines)


def schema(tools: list[dict]) -> dict:
    names = [(t.get('function') or t).get('name') for t in tools]
    return {'type': 'object', 'additionalProperties': False, 'required': ['tool'],
            'properties': {'tool': {'type': 'string', 'enum': [*names, NONE]},
                           'arguments': {'type': 'object'}, 'text': {'type': 'string'}}}


class UsageLimit(Exception):
    pass


def claude(model: str, system: str, prompt: str, *, tools: list[dict], effort: str | None, timeout: float) -> dict:
    argv = ['claude', '-p', '--model', model, '--output-format', 'json', '--no-session-persistence',
            '--tools', '', '--strict-mcp-config', '--disable-slash-commands', '--setting-sources', '',
            '--system-prompt', system + (CALL_INSTRUCTIONS + tool_list(tools) if tools else '')]
    if tools:
        argv += ['--json-schema', json.dumps(schema(tools))]
    if effort in EFFORTS:
        argv += ['--effort', effort]
    run = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=timeout,
                         cwd=os.environ.get('CLAUDE_CODE_ENDPOINT_CWD', '/tmp'))
    try:
        result = json.loads(run.stdout)
    except ValueError:
        message = (run.stderr or run.stdout or f'claude exited {run.returncode}').strip()[-800:]
        if LIMIT.search(message):
            raise UsageLimit(message)
        raise RuntimeError(message)
    if result.get('is_error') or result.get('subtype') != 'success':
        message = str(result.get('result') or result.get('api_error_status') or result.get('subtype'))[-800:]
        if LIMIT.search(message) or result.get('api_error_status') in (429, 529):
            raise UsageLimit(message)
        raise RuntimeError(message)
    return result


def completion(request: dict, timeout: float) -> dict:
    tools = [t for t in request.get('tools') or [] if (t.get('function') or t).get('name')]
    system, prompt = render(request.get('messages') or [])
    model = request.get('model') or 'claude-opus-5-5'
    effort = request.get('reasoning_effort') or (request.get('reasoning') or {}).get('effort')
    result = claude(model, system, prompt, tools=tools, effort=effort, timeout=timeout)
    usage = result.get('usage') or {}
    prompt_tokens = sum(int(usage.get(k) or 0) for k in ('input_tokens', 'cache_creation_input_tokens',
                                                          'cache_read_input_tokens'))
    message: dict = {'role': 'assistant', 'content': ''}
    finish = 'stop'
    if tools:
        out = result.get('structured_output')
        if not isinstance(out, dict):
            try:
                out = json.loads(result.get('result') or '')
            except ValueError:
                out = {'tool': NONE, 'text': str(result.get('result') or '')}
        if out.get('tool') and out['tool'] != NONE:
            message['tool_calls'] = [{'id': 'call_' + uuid.uuid4().hex[:16], 'type': 'function',
                                      'function': {'name': out['tool'],
                                                   'arguments': json.dumps(out.get('arguments') or {})}}]
            finish = 'tool_calls'
        else:
            message['content'] = str(out.get('text') or '')
    else:
        message['content'] = str(result.get('result') or '')
    return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion', 'created': int(time.time()),
            'model': model, 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': int(usage.get('output_tokens') or 0),
                      'total_tokens': prompt_tokens + int(usage.get('output_tokens') or 0)}}


def handler(slots: threading.Semaphore, timeout: float, log):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def reply(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip('/').endswith('/models'):
                return self.reply(200, {'object': 'list', 'data': [{'id': m, 'object': 'model'} for m in
                                                                   ('claude-opus-5-5', 'claude-haiku-5-5', 'claude-sonnet-5-5')]})
            self.reply(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            if not self.path.rstrip('/').endswith('/chat/completions'):
                return self.reply(404, {'error': {'message': 'not found'}})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get('content-length') or 0)))
            except ValueError:
                return self.reply(400, {'error': {'message': 'invalid JSON'}})
            started = time.time()
            with slots:
                try:
                    body = completion(request, timeout)
                    status = 200
                except UsageLimit as error:
                    body, status = {'error': {'type': 'rate_limit', 'message': f'Claude Code usage limit: {error}'}}, 429
                except subprocess.TimeoutExpired:
                    body, status = {'error': {'type': 'timeout', 'message': 'Claude Code call timed out'}}, 504
                except Exception as error:  # noqa: BLE001 - every failure is the endpoint's answer
                    body, status = {'error': {'type': 'server_error', 'message': str(error)[-800:]}}, 502
            log(json.dumps({'at': round(started), 'model': request.get('model'), 'status': status,
                            'seconds': round(time.time() - started, 1), 'tools': len(request.get('tools') or []),
                            'usage': body.get('usage')}))
            self.reply(status, body)
    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8790)
    s.add_argument('--concurrency', type=int, default=8, help='Claude Code calls running at once')
    s.add_argument('--timeout', type=float, default=1800, help='seconds per call')
    args = parser.parse_args(argv)
    lock = threading.Lock()

    def log(line):
        with lock:
            print(line, flush=True)
    server = ThreadingHTTPServer((args.host, args.port), handler(threading.Semaphore(args.concurrency), args.timeout, log))
    print(json.dumps({'serving': f'http://{args.host}:{args.port}', 'concurrency': args.concurrency}), flush=True)
    server.serve_forever()
    return 0


if __name__ == '__main__':
    sys.exit(main())
