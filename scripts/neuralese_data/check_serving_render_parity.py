#!/usr/bin/env python3
"""Check that training rows render a request exactly as the serving engine does (bytes and token ids).

For recorded chat requests (eval probe ``requests.jsonl``: {"req": <OpenAI chat request JSON>}), compare:
  1. ``serving_messages(..., entry)`` rebuilt from the recorded request against the request itself (the system prompt
     and tool schemas the current ts-host surface yields must equal what was sent);
  2. the training renderer's prompt (``native_gold_packet``, prompt_rendering full, up to the supervised suffix) against
     llama-server's own chat-template rendering of the same messages and tools (``/apply-template``), byte for byte;
  3. the training token ids of that prompt against llama-server's ``/tokenize`` of its prompt (special tokens parsed).

llama-server runs on CPU (no GPU; CUDA hidden) with the gate's ``--jinja`` and chat-template kwargs. Run under the
memory ledger. Writes a JSON report; exit 1 on any mismatch.

  check_serving_render_parity.py --llama-server BIN --gguf F --requests R.jsonl [...] --out REPORT.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts'))


def post(port, path, body):
    request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def requests(paths, limit):
    seen = set()
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            if 'req' not in row:
                continue
            request = json.loads(row['req'])
            messages = request.get('messages') or []
            if not messages or messages[0].get('role') != 'system':
                continue
            key = json.dumps([messages, request.get('tools')], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            yield path, request
            if limit and len(seen) >= limit:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--llama-server', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--chat-template-kwargs', default='{"preserve_thinking": true}')
    parser.add_argument('--requests', type=Path, nargs='+', required=True)
    parser.add_argument('--entry', default='collector')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--port', type=int, default=18931)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from build_neuralese_gold_text_delta import load_pinned_tokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from natlang_neuralese.data.text_corpus import native_gold_packet
    from natlang_neuralese.data.serving_prompt import serving_messages, load_surface
    tokenizer = load_pinned_tokenizer(args.tokenizer)
    bind_history_reasoning(tokenizer, require_declared=True)
    surface = load_surface()

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    server = subprocess.Popen([str(args.llama_server), '-m', str(args.gguf), '--jinja', '-ngl', '0', '-c', '4096',
                               '--host', '127.0.0.1', '--port', str(args.port),
                               '--chat-template-kwargs', args.chat_template_kwargs],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {'schema': 'natlang.serving-render-parity/1', 'entry': args.entry,
              'surface': surface['_identity'], 'gguf': str(args.gguf),
              'chat_template_kwargs': json.loads(args.chat_template_kwargs), 'cases': []}
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('llama-server did not become healthy')
        for path, request in requests(args.requests, args.limit):
            messages, tools = request['messages'], request.get('tools')
            rebuilt, rebuilt_tools, info = serving_messages(messages, tools, entry=args.entry, surface=surface)
            case = {'source': str(path), 'turns': len(messages), 'mode': info['mode'],
                    'surface_system_equal': rebuilt[0]['content'] == messages[0]['content'],
                    'surface_tools_equal': json.dumps(rebuilt_tools) == json.dumps(tools)}
            # The request's messages end with the context the model answers; a placeholder reply marks the boundary.
            text, ids, suffix = native_gold_packet(tokenizer, messages, {'role': 'assistant', 'content': 'ok'},
                                                   tools or [], prompt_rendering='full')[:3]
            training_prompt = tokenizer.decode(ids[:suffix], skip_special_tokens=False)
            served = post(args.port, '/apply-template', {'messages': messages, 'tools': tools})['prompt']
            served_ids = post(args.port, '/tokenize', {'content': served, 'add_special': False,
                                                       'parse_special': True})['tokens']
            case.update({'prompt_chars': len(served), 'prompt_tokens': len(served_ids),
                         'prompt_bytes_equal': training_prompt == served,
                         'prompt_ids_equal': list(ids[:suffix]) == list(served_ids)})
            if not case['prompt_bytes_equal']:
                at = next((i for i, (a, b) in enumerate(zip(training_prompt, served)) if a != b),
                          min(len(training_prompt), len(served)))
                case['first_difference'] = {'at': at, 'training': training_prompt[max(0, at - 80):at + 80],
                                            'served': served[max(0, at - 80):at + 80]}
            report['cases'].append(case)
    finally:
        server.terminate()
        server.wait(timeout=60)
    keys = ('surface_system_equal', 'surface_tools_equal', 'prompt_bytes_equal', 'prompt_ids_equal')
    report['summary'] = {'cases': len(report['cases']),
                         **{key: sum(case[key] for case in report['cases']) for key in keys}}
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['summary']))
    sys.exit(0 if all(report['summary'][key] == len(report['cases']) for key in keys) and report['cases'] else 1)


if __name__ == '__main__':
    main()
