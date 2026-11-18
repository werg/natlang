"""Speculative decoding keeps the sampling law and the RL capture (any engine that serves both settings).

Usage: check_speculative_law.py --engine sglang|llama --plain URL --spec URL --workload turns.jsonl --out DIR

Two servers of one model, one with speculative decoding and one without, at the actor's law (T=1, top-p 0.95):

  capture  every token a speculative reply draws carries the natlang record of the target model's law for it: its
           logp equals a plain prefill of the same ids at the engine's floor (the same comparison on the plain
           server's own replies gives the floor), and every token has its record.
  law      a few hundred short replies per prompt on each server; per prompt and position a two-sample chi-square
           test of the token marginals (tokens seen fewer than 5 times pooled). Exact speculative sampling leaves
           the law unchanged, so the p-values are uniform; a biased verifier (e.g. accepting drafts too often) moves
           the marginals toward the drafts, which repeat earlier replies, and shows as small p-values.

Prompts are real GRPO calls (``sglang_bench workload``), so drafts come from the prompt and the earlier replies.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import struct
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LAW = {'temperature': 1.0, 'top_p': 0.95}


def _post(url, path, body, timeout=3600):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Sglang:
    def generate(self, url, prompt, max_tokens, stop_ids):
        result = _post(url, '/generate', {'input_ids': prompt, 'sampling_params': {
            **LAW, 'max_new_tokens': max_tokens, 'stop_token_ids': stop_ids, 'skip_special_tokens': False,
            'no_stop_trim': True, 'custom_params': {'natlang_top_k': 1}}})
        records = result['meta_info'].get('natlang_top_k') or []
        return result['output_ids'], [struct.unpack('<f', base64.b64decode(r)[:4])[0] for r in records], None

    def prefill_logps(self, url, prompt, tokens):
        result = _post(url, '/generate', {'input_ids': prompt + tokens, 'return_logprob': True,
                                          'logprob_start_len': len(prompt) - 1,
                                          'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}})
        return [item[0] for item in result['meta_info']['input_token_logprobs']][1:len(tokens) + 1]


class Llama:
    """The managed llama-server: /completion with token prompts and the natlang RL capture (x_natlang_rl_capture). It
    has no prompt log-probabilities, so a token's reference is the plain decode step at its position: a one-token
    reply after the prompt and the earlier tokens (prompt cache), whose top-128 record holds the token's logp (float16)."""

    def generate(self, url, prompt, max_tokens, stop_ids, top_k=0):
        result = _post(url, '/completion', {'prompt': prompt, 'n_predict': max_tokens, **LAW, 'top_k': 0, 'min_p': 0,
                                            'cache_prompt': True, 'x_natlang_rl_capture': True,
                                            'x_natlang_rl_top_k': top_k})
        samples = result['x_natlang_rl_capture']['samples']
        return [s['token_id'] for s in samples], [s['logp'] for s in samples], samples

    def prefill_logps(self, url, prompt, tokens):
        out = []
        for i, token in enumerate(tokens[:LLAMA_REFERENCE_TOKENS]):
            sample = self.generate(url, prompt + tokens[:i], 1, [], top_k=128)[2][0]
            ids = struct.unpack('<128i', base64.b64decode(sample['top_k']['ids']))
            logps = struct.unpack('<128e', base64.b64decode(sample['top_k']['logps']))
            out.append(logps[ids.index(token)] if token in ids else None)
        return out


LLAMA_REFERENCE_TOKENS = 48


def _chi2_sf(x, dof):
    import torch
    return float(torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64), torch.tensor(x / 2, dtype=torch.float64)))


def two_sample_chi2(a, b, min_count=5):
    """(statistic, dof, p) of the token counts a and b (Counters), rare tokens pooled."""
    total = a + b
    common = [t for t, c in total.items() if c >= min_count]
    rows = [[a[t] for t in common], [b[t] for t in common]]
    rows[0].append(sum(a.values()) - sum(rows[0]))
    rows[1].append(sum(b.values()) - sum(rows[1]))
    columns = [i for i in range(len(rows[0])) if rows[0][i] + rows[1][i] > 0]
    if len(columns) < 2:
        return 0.0, 0, 1.0
    n = [sum(row) for row in rows]
    statistic = 0.0
    for i in columns:
        column = rows[0][i] + rows[1][i]
        for r in (0, 1):
            expected = n[r] * column / (n[0] + n[1])
            statistic += (rows[r][i] - expected) ** 2 / expected
    dof = len(columns) - 1
    return statistic, dof, _chi2_sf(statistic, dof)


def capture(engine, url, rows, args):
    def one(row):
        tokens, logps, _ = engine.generate(url, row['prompt'], args.capture_tokens, args.stop_ids)
        prefill = engine.prefill_logps(url, row['prompt'], tokens)
        return {'tokens': len(tokens), 'records': len(logps),
                'deltas': [abs(a - b) for a, b in zip(logps, prefill) if b is not None]}
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(one, rows))
    deltas = sorted(d for r in results for d in r['deltas'])
    return {'replies': len(results), 'tokens': sum(r['tokens'] for r in results),
            'records_cover_tokens': all(r['records'] == r['tokens'] for r in results),
            'abs_delta_mean': sum(deltas) / max(1, len(deltas)),
            'abs_delta_p99': deltas[int(0.99 * (len(deltas) - 1))] if deltas else None,
            'abs_delta_max': deltas[-1] if deltas else None}


def law(engine, plain, spec, rows, args):
    report = []
    for row in rows:
        counts = {}
        for name, url in (('plain', plain), ('spec', spec)):
            with ThreadPoolExecutor(args.concurrency) as pool:
                replies = list(pool.map(lambda _: engine.generate(url, row['prompt'], args.law_tokens, args.stop_ids)[0],
                                        range(args.law_samples)))
            counts[name] = [Counter(r[t] for r in replies if len(r) > t) for t in range(args.law_tokens)]
        for t in range(args.law_tokens):
            statistic, dof, p = two_sample_chi2(counts['plain'][t], counts['spec'][t])
            report.append({'prompt_tokens': len(row['prompt']), 'position': t, 'chi2': statistic, 'dof': dof, 'p': p,
                           'top_plain': counts['plain'][t].most_common(3), 'top_spec': counts['spec'][t].most_common(3)})
    tests = [r for r in report if r['dof'] > 0]
    ps = sorted(r['p'] for r in tests)
    import math
    fisher = -2 * sum(math.log(max(p, 1e-300)) for p in ps)
    return {'tests': len(tests), 'min_p': ps[0] if ps else None, 'bonferroni_min_p': min(1.0, ps[0] * len(ps)) if ps else None,
            'fisher_p': _chi2_sf(fisher, 2 * len(ps)) if ps else None, 'p_below_0.05': sum(p < 0.05 for p in ps),
            'per_test': report}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', choices=('sglang', 'llama'), required=True)
    p.add_argument('--plain', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--workload', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stop-ids', type=int, nargs='*', default=[124900])
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--capture-replies', type=int, default=32)
    p.add_argument('--capture-tokens', type=int, default=256)
    p.add_argument('--law-prompts', type=int, default=3)
    p.add_argument('--law-samples', type=int, default=300)
    p.add_argument('--law-tokens', type=int, default=8)
    p.add_argument('--max-prompt', type=int, default=16000)
    p.add_argument('--parts', nargs='*', default=['capture', 'law'], choices=('capture', 'law'))
    p.add_argument('--prompt-tail', type=int, help='use only the last N tokens of each prompt (a CPU engine)')
    args = p.parse_args(argv)
    engine = Sglang() if args.engine == 'sglang' else Llama()
    rows = [json.loads(line) for line in open(args.workload)]
    rows = [r for r in rows if len(r['prompt']) < args.max_prompt]
    if args.prompt_tail:
        rows = [{**r, 'prompt': r['prompt'][-args.prompt_tail:]} for r in rows]
    random.Random(1).shuffle(rows)
    report = {'engine': args.engine, 'law': LAW}
    if 'capture' in args.parts:
        report['capture_spec'] = capture(engine, args.spec, rows[:args.capture_replies], args)
        report['capture_plain_floor'] = capture(engine, args.plain, rows[:args.capture_replies], args)
        print(json.dumps(report), flush=True)
    if 'law' in args.parts:
        report['law_test'] = law(engine, args.plain, args.spec, rows[args.capture_replies:][:args.law_prompts], args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'speculative-law-{args.engine}-{"-".join(args.parts)}.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: (v if k != 'law_test' else {kk: vv for kk, vv in v.items() if kk != 'per_test'})
                      for k, v in report.items()}), flush=True)


if __name__ == '__main__':
    main()
