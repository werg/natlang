#!/usr/bin/env python3
"""Tool-call grammar parity and the learner's law (common.tool_grammar; docs/ENGINE_PARITY.md "Result schemas: tuples").

`decode --endpoint URL --out FILE` runs against a natlang serving front (the Python native engine or the SGLang actor):
- the forced decode of `scripts/wllama_smoke.mjs`'s tuple leg (greedy): one `return_result` tool whose value is the
  runtime's schema of `[string, number]` or `[string, number?]`, told to return `["red", "green", "blue"]` (and, for
  the optional tail, `["red"]`), and an unconstrained control (`value: {}`); a constraining engine keeps every value to
  its type and lets the one-element tail through;
- captured samples of the same prompts at the RL law (temperature 1, top-p 0.95, `x_natlang_rl_capture`), for `learner`.

`learner --captures FILE --checkpoint CKPT --base BASE` scores those captures with the learner's own path
(`lower_rollout_turns.sampled_call` rows, `exp0_sft.law_mask`, `lfm3_fast.token_logprobs`) with and without the
recorded masks, at the capture's own weights: masked, the learner's log-probabilities equal the engine's recorded ones
to the engines' numeric floor and log rho has no offset on the constrained tokens; unmasked, log rho on those tokens is
the log of the allowed mass (the bias the mask removes). Exits 1 when a check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

STRING, NUMBER = {'type': 'string'}, {'type': 'number'}
EXACT = {'type': 'array', 'prefixItems': [STRING, NUMBER], 'minItems': 2, 'maxItems': 2}
# `[string, number?]`: one exact tuple per admitted length (ts-host native/agent.ts schemaOf)
OPTIONAL_TAIL = {'anyOf': [{'type': 'array', 'prefixItems': [STRING], 'minItems': 1, 'maxItems': 1}, EXACT]}
CASES = [{'type': '[string, number]', 'ask': '["red", "green", "blue"]', 'schema': EXACT, 'min': 2, 'max': 2},
         {'type': '[string, number?]', 'ask': '["red", "green", "blue"]', 'schema': OPTIONAL_TAIL, 'min': 1, 'max': 2},
         {'type': '[string, number?]', 'ask': '["red"]', 'schema': OPTIONAL_TAIL, 'min': 1, 'max': 2, 'expect': ['red']},
         {'type': 'control', 'ask': '["red", "green", "blue"]', 'schema': {}, 'expect': ['red', 'green', 'blue']}]


def request(case, **extra):
    """wllama_smoke.mjs `tupleRequest` (the natlang fronts render their own template: reasoning stays on)."""
    return {'messages': [{'role': 'system', 'content': 'You finish a function call by calling return_result.'},
                         {'role': 'user', 'content': f'Call return_result with status "success" and value {case["ask"]}, '
                                                     'written exactly so.'}],
            'tools': [{'type': 'function', 'function': {
                'name': 'return_result', 'description': 'Finish with the result.',
                'parameters': {'type': 'object', 'required': ['status', 'value'], 'properties': {
                    'status': {'type': 'string', 'enum': ['success', 'failed']}, 'value': case['schema']}}}}],
            'tool_choice': 'auto', **extra}


def post(endpoint, body, timeout=900):
    data = json.dumps(body).encode()
    with urllib.request.urlopen(urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions', data=data,
                                                       headers={'content-type': 'application/json'}),
                                timeout=timeout) as response:
        return json.loads(response.read())


def value_of(reply):
    calls = ((reply.get('choices') or [{}])[0].get('message') or {}).get('tool_calls') or []
    if not calls:
        return {'error': json.dumps((reply.get('choices') or [{}])[0].get('message'))[-300:]}
    try:
        return {'value': json.loads(calls[0]['function']['arguments']).get('value')}
    except (KeyError, ValueError, AttributeError):
        return {'error': calls[0]['function'].get('arguments')}


def kept(case, got):
    """None when the value keeps the case's type (wllama_smoke.mjs `tupleKept`), else the problem."""
    if 'error' in got:
        return f'no return_result call: {got["error"]}'
    value = got['value']
    if case['type'] == 'control' and isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if 'expect' in case:
        return None if value == case['expect'] else f'value {json.dumps(value)}, asked {case["ask"]}'
    ok = (isinstance(value, list) and case['min'] <= len(value) <= case['max'] and isinstance(value[0], str)
          and (len(value) < 2 or (isinstance(value[1], (int, float)) and not isinstance(value[1], bool))))
    return None if ok else f'value {json.dumps(value)} breaks {case["type"]}'


def decode(args):
    out, failures = {'endpoint': args.endpoint, 'forced': [], 'captures': []}, []
    for case in CASES:
        started = time.time()
        reply = post(args.endpoint, request(case, max_tokens=args.max_tokens, temperature=0))
        got = value_of(reply)
        problem = kept(case, got)
        out['forced'].append({'type': case['type'], 'asked': case['ask'], **got, 'problem': problem,
                              'seconds': round(time.time() - started, 1),
                              'completion_tokens': (reply.get('usage') or {}).get('completion_tokens')})
        print(json.dumps(out['forced'][-1]), flush=True)
        if problem and (case['type'] != 'control' or not args.unconstrained):
            failures.append(f'{case["type"]} asked {case["ask"]}: {problem}')
    for index, case in enumerate(CASES[:3] * args.samples):
        reply = post(args.endpoint, request(case, max_tokens=args.max_tokens, temperature=1.0, top_p=args.top_p,
                                            seed=1000 + index, x_natlang_rl_capture=True, x_natlang_rl_top_k=0))
        capture = reply.get('x_natlang_rl_capture')
        out['captures'].append({'type': case['type'], 'asked': case['ask'], **value_of(reply), 'capture': capture})
        law = ((capture or {}).get('sampling_law') or {})
        Path(args.out).write_text(json.dumps(out) + '\n')     # kept as it grows: a stopped unit keeps its captures
        print(json.dumps({'capture': index, 'type': case['type'], 'samples': len((capture or {}).get('samples') or []),
                          'tool_grammar': (law.get('tool_grammar') or {}).get('sha256'), **value_of(reply)}), flush=True)
        if not args.unconstrained and law.get('tool_grammar') is None:
            failures.append(f'capture {index} declares no tool grammar')
    out['failures'] = failures
    Path(args.out).write_text(json.dumps(out) + '\n')
    print(json.dumps({'event': 'decode_done', 'failures': failures}), flush=True)
    return 1 if failures else 0


def learner(args):
    import collections
    import numpy as np
    import torch
    import exp0_sft
    from lower_rollout_turns import sampled_call
    from natlang_neuralese.common import tool_grammar
    from natlang_neuralese.train import lfm3_fast
    from natlang_neuralese.train.lfm3_learner import load_effective_policy
    exp0_sft.LAW_BASE['path'] = args.base
    data = json.loads(Path(args.captures).read_text())
    stats = collections.Counter()
    rows = []
    for index, item in enumerate(data['captures']):
        capture = item.get('capture')
        if not capture:
            stats['no_capture'] += 1
            continue
        row = sampled_call(f'check-{index}', 0, capture, stats)
        if row is not None:
            rows.append(row)
    model = load_effective_policy(args.checkpoint, args.base, device='cuda', compile_layers=False)
    model = model[0] if isinstance(model, tuple) else model
    width = model.lm_head.weight.shape[0]
    report, failures = {'calls': len(rows), 'lowering': dict(stats)}, []
    masked_d, unmasked_d, constrained = [], [], []
    for row in rows:
        chain = {'ids': np.concatenate([row['prompt'], row['sample']]),
                 'segs': [(len(row['prompt']), len(row['sample']), row)]}
        ids = torch.from_numpy(chain['ids'].astype(np.int64)).cuda()
        positions = torch.arange(len(row['prompt']) - 1, len(chain['ids']) - 1, device='cuda')
        targets = torch.from_numpy(row['sample'].astype(np.int64)).cuda()
        width = model.lm_head.weight.shape[0]
        law = exp0_sft.law_mask([chain], width)
        with torch.no_grad():
            masked = lfm3_fast.token_logprobs(model, ids, positions, targets, mask=law)
            plain = lfm3_fast.token_logprobs(model, ids, positions, targets)
        recorded = torch.from_numpy(row['logp']).cuda()
        allowed = (law.allowed_rows(0, len(targets), width, 'cuda').sum(-1) if law is not None
                   else torch.full_like(targets, width))
        masked_d.append((masked - recorded).cpu())
        unmasked_d.append((plain - recorded).cpu())
        constrained.append(allowed.cpu())
    if not rows:
        failures.append('no lowered call')
    else:
        m, u, counts = torch.cat(masked_d), torch.cat(unmasked_d), torch.cat(constrained)
        # free text allows every real token (the widest mask: the vocabulary without its padding rows); a constrained
        # position allows fewer
        c = counts < counts.max()
        report['free_text_allowed_tokens'] = int(counts.max())
        def summary(values):
            return ({'n': len(values), 'mean': round(float(values.mean()), 5), 'abs_mean': round(float(values.abs().mean()), 5),
                     'min': round(float(values.min()), 4), 'abs_max': round(float(values.abs().max()), 4)}
                    if len(values) else {'n': 0})
        # log rho = learner - recorded at the capture's own weights: the policy ratio of a first learner step
        report.update(tokens=len(m), constrained_tokens=int(c.sum()),
                      log_rho_masked_all=summary(m), log_rho_masked_constrained=summary(m[c]),
                      log_rho_masked_free=summary(m[~c]), log_rho_unmasked_constrained=summary(u[c]),
                      log_rho_unmasked_free=summary(u[~c]))
        if not int(c.sum()):
            failures.append('no constrained token in the captures')
        if float(m.abs().max()) > args.tolerance:
            failures.append(f'masked learner log-probabilities differ from the recorded ones by {float(m.abs().max()):.3f}')
        if int(c.sum()) and abs(float(m[c].mean())) > args.bias_tolerance:
            failures.append(f'log rho on constrained tokens has mean {float(m[c].mean()):.4f} with the mask')
    report['failures'] = failures
    print(json.dumps(report), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report) + '\n')
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('decode')
    d.add_argument('--endpoint', required=True)
    d.add_argument('--out', required=True)
    d.add_argument('--max-tokens', type=int, default=2048)
    d.add_argument('--samples', type=int, default=4, help='captured samples per constrained case')
    d.add_argument('--top-p', type=float, default=0.95)
    d.add_argument('--unconstrained', action='store_true', help='expect no grammar (an opted-out engine)')
    l = sub.add_parser('learner')
    l.add_argument('--captures', required=True)
    l.add_argument('--checkpoint', required=True)
    l.add_argument('--base', required=True)
    l.add_argument('--tolerance', type=float, default=1.0,
                   help='largest |learner - recorded| logp (nats): the engines differ from the learner by BF16 rounding')
    l.add_argument('--bias-tolerance', type=float, default=0.02, help='largest |mean log rho| on constrained tokens')
    l.add_argument('--out')
    args = parser.parse_args()
    sys.exit({'decode': decode, 'learner': learner}[args.command](args))


if __name__ == '__main__':
    main()
