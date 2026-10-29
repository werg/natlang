#!/usr/bin/env python3
"""End-to-end check of the SGLang RL actor (serve_native_checkpoint.py --engine sglang) against the native model.

usage: check_sglang_actor_capture.py --checkpoint CKPT --base DIR --out OUT.json [--requests N] [--concurrency C]
                                     [--temperature 1.0] [--max-tokens 128] [--k 128] [--port 8297]

Starts the actor (HF export, the fork's server, the native front), sends N captured chat turns built from real natlang
episode messages (runs/lfm3-engine-bench-20261028/out/capture.jsonl), then replays every capture on the native model
(`native_step` and `common.sampling.masked_log_distribution` with the recorded temperature and `banned_token_ids`,
FP32 reference): |delta| of each sampled token's logp, top-1 agreement and top-k overlap of the behaviour receipt, and
the decoded top-k format. Also records throughput of the batch. Exit status 1 on a capture/format failure or when the
actor is further from FP32 than the native BF16 forward (mean |delta| above 1.25x that floor + 0.005, or top-1
agreement more than 0.01 below it).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))


def turns(path, n):
    rows = []
    for line in open(path):
        row = json.loads(line)
        body = row.get('request') or row.get('body') or row
        if isinstance(body, dict) and body.get('messages'):
            rows.append({'messages': body['messages'], 'tools': body.get('tools')})
        if len(rows) >= n:
            break
    return rows


def post(url, body):
    request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--turns', type=Path, default=ROOT / 'runs/lfm3-engine-bench-20261028/out/capture.jsonl')
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--port', type=int, default=8297)
    parser.add_argument('--replay', type=int, default=16, help='captures replayed on the native model')
    parser.add_argument('--sglang-mem-fraction', type=float, default=0.15)
    parser.add_argument('--store-dir', type=Path)
    args = parser.parse_args()
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    sha = sha256_file_hex_cached(args.checkpoint)
    store = args.store_dir or args.out.parent / 'actor-store'
    log = open(args.out.with_suffix('.actor.log'), 'w')
    actor = subprocess.Popen([sys.executable, str(ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py'),
        '--checkpoint', str(args.checkpoint), '--checkpoint-sha256', sha, '--base', args.base, '--port', str(args.port),
        '--store-dir', str(store), '--engine', 'sglang', '--sglang-mem-fraction', str(args.sglang_mem_fraction)],
        stdout=log, stderr=subprocess.STDOUT)
    url = f'http://127.0.0.1:{args.port}'
    try:
        deadline = time.time() + 1800
        while True:
            if actor.poll() is not None:
                raise RuntimeError(f'actor exited with {actor.returncode}; see {log.name}')
            try:
                with urllib.request.urlopen(url + '/v1/neuralese/info', timeout=10) as response:
                    info = json.loads(response.read())
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(5)
        bodies = [{**turn, 'max_tokens': args.max_tokens, 'temperature': args.temperature,
                   'x_natlang_rl_capture': True, 'x_natlang_rl_top_k': args.k}
                  for turn in turns(args.turns, args.requests)]
        start = time.time()
        with ThreadPoolExecutor(args.concurrency) as pool:
            responses = list(pool.map(lambda body: post(url, body), bodies))
        wall = time.time() - start
    finally:
        actor.terminate()
        try:
            actor.wait(timeout=120)
        except subprocess.TimeoutExpired:
            actor.kill()
    generated = sum(len(r['x_natlang_rl_capture']['samples']) for r in responses)
    result = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha, 'requests': len(responses),
              'concurrency': args.concurrency, 'wall_s': wall, 'sampled_tokens': generated,
              'tokens_per_s': generated / wall, 'serving_engine': info['rl_policy_binding'].get('serving_engine'),
              'finish_reasons': [r['choices'][0]['finish_reason'] for r in responses]}
    print(json.dumps(result), flush=True)
    result.update(replay(args, responses[:args.replay]))
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    sys.exit(1 if result['failures'] else 0)


def replay(args, responses):
    """The native model's law at every recorded sample, against the actor's receipts. Reference: the native forward in
    FP32; the native forward in BF16 against it is the noise floor the BF16 engine is held to."""
    import torch
    from natlang_neuralese.common.sampling import decode_top_k, masked_log_distribution
    from natlang_neuralese.model.native_model import load_native_checkpoint
    from natlang_neuralese.model.shared_state import native_step

    def load(dtype):
        model = load_native_checkpoint(str(args.checkpoint), base=args.base, device='cuda', dtype=dtype,
                                       allow_weights_only=True)[0]
        return model.eval()
    models = {'fp32': load(torch.float32), 'bf16': load(torch.bfloat16)}
    rows, floor, failures = [], [], []
    for response in responses:
        capture = response['x_natlang_rl_capture']
        prompt = [token for segment in capture['prompt'] for token in segment['token_ids']]
        inputs = [item['token_id'] for item in capture['generated_inputs']]
        ids = torch.tensor([prompt + inputs], device='cuda')
        logits = {}
        queries = torch.tensor([sample['query_own_position'] for sample in capture['samples']], device='cuda')
        with torch.no_grad():
            for name, model in models.items():
                hidden = native_step(model, ids, None, producing=False)['h_final'][0]
                logits[name] = model.logits(hidden[queries]).float()
        for row, sample in enumerate(capture['samples']):
            token = sample['token_id']
            laws = {name: masked_log_distribution(value[row], capture['sampling_law']['temperature'],
                                                  banned=sample.get('banned_token_ids')) for name, value in logits.items()}
            law = laws['fp32']
            top_ids, top_logps = decode_top_k(sample['top_k'])
            reference = torch.topk(law, len(top_ids)).indices.cpu()
            rows.append({'logp': abs(sample['logp'] - float(law[token])),
                         'top1': int(top_ids[0]) == int(reference[0]),
                         'overlap': len(set(top_ids.tolist()) & set(reference.tolist())) / len(top_ids),
                         'top_logp': float((top_logps - law[top_ids.to(law.device)].cpu()).abs().max())})
            floor.append({'logp': abs(float(laws['bf16'][token]) - float(law[token])),
                          'top1': int(laws['bf16'].argmax()) == int(reference[0])})
        if len(capture['samples']) != len(inputs) + (capture['finish_reason'] == 'stop'):
            failures.append('samples and generated inputs disagree')

    def summary(values):
        return {'positions': len(values), 'max_abs_logprob': max(r['logp'] for r in values),
                'mean_abs_logprob': sum(r['logp'] for r in values) / len(values),
                'top1_agreement': sum(r['top1'] for r in values) / len(values)}
    report = {**summary(rows), 'topk_overlap': sum(r['overlap'] for r in rows) / len(rows),
              'max_abs_topk_logprob': max(r['top_logp'] for r in rows)}
    noise = summary(floor)
    # The BF16 engine passes when it is no further from FP32 than the native BF16 forward (small margin).
    if report['mean_abs_logprob'] > 1.25 * noise['mean_abs_logprob'] + 0.005:
        failures.append('mean_abs_logprob')
    if report['top1_agreement'] < noise['top1_agreement'] - 0.01:
        failures.append('top1_agreement')
    return {'replay': report, 'native_bf16_noise_floor': noise, 'failures': failures}


if __name__ == '__main__':
    main()
