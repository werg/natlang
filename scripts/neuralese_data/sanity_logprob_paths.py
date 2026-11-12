#!/usr/bin/env python3
"""Learning sanity check 1 (plans/neuralese/LEARNING_SANITY_20261112.md): does what the trainer learned reach the served
model? Scores the exact sampled tokens of a fixed sample of training chains (`exp0_sft.chains_of`) two ways:

    sanity_logprob_paths.py trainer --turns T.pkl --checkpoint C --base B --out X.npz   # exp0_sft/lfm3_fast_train forward (BF16)
    sanity_logprob_paths.py sglang  --turns T.pkl --url http://127.0.0.1:PORT --out X.npz  # SGLang prefill logprobs
    sanity_logprob_paths.py compare A.npz B.npz ...                                       # NLL per file, pairwise |d|

The chain sample is a function of (turns, --chains, --max-tokens, --seed) only, so both paths score identical tokens.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sample_chains(args):
    from exp0_sft import chains_of
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    chains = [c for c in chains_of(episodes) if len(c['ids']) <= args.max_tokens]
    random.Random(args.seed).shuffle(chains)
    return chains[:args.chains]


def targets_of(chain):
    positions = np.concatenate([np.arange(start - 1, start + length - 1) for start, length, _ in chain['segs']])
    targets = np.concatenate([row['sample'] for _, _, row in chain['segs']])
    return positions, targets


def save(out, chains, values):
    np.savez(out, logp=np.concatenate(values), lengths=np.array([len(v) for v in values]),
             tasks=np.array([c['task'] for c in chains]))


def trainer(args):
    import torch
    import exp0_sft
    import lfm3_fast_train
    from natlang_neuralese.common.cuda_context import ensure_context
    ensure_context('cuda')   # GB10: context creation can fail under host-memory fragmentation; retried
    chains = sample_chains(args)
    model, _, _ = exp0_sft.load_model(args.checkpoint, args.base, dtype=torch.bfloat16)
    model.eval()
    lfm3_fast_train.prepare(model)
    values = []
    for chain in chains:
        ids = torch.from_numpy(chain['ids'].astype(np.int64)).cuda()
        positions, targets = targets_of(chain)
        logp = lfm3_fast_train.token_logprobs(model, ids, torch.from_numpy(positions).cuda(),
                                              torch.from_numpy(targets.astype(np.int64)).cuda())
        values.append(logp.float().cpu().numpy())
        print(json.dumps({'tokens': len(ids), 'targets': len(targets), 'nll': float(-values[-1].mean())}), flush=True)
    save(args.out, chains, values)


def sglang(args):
    import urllib.request
    chains = sample_chains(args)

    def one(chain):
        positions, targets = targets_of(chain)
        body = {'input_ids': [int(i) for i in chain['ids']], 'return_logprob': True, 'logprob_start_len': 0,
                'sampling_params': {'temperature': 0.0, 'max_new_tokens': 1}}
        request = urllib.request.Request(args.url + '/generate', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.loads(response.read())
        # input_token_logprobs[i] = (logp of ids[i] given ids[:i], id, _); the first is None
        items = result['meta_info']['input_token_logprobs']
        logp = np.array([items[p + 1][0] for p in positions], dtype=np.float32)
        assert all(items[p + 1][1] == t for p, t in zip(positions, targets))
        return logp
    with ThreadPoolExecutor(args.concurrency) as pool:
        values = list(pool.map(one, chains))
    for chain, value in zip(chains, values):
        print(json.dumps({'tokens': len(chain['ids']), 'targets': len(value), 'nll': float(-value.mean())}), flush=True)
    save(args.out, chains, values)


def compare(args):
    data = {Path(p).stem: np.load(p) for p in args.files}
    names = list(data)
    for name in names:
        logp = data[name]['logp']
        print(json.dumps({'file': name, 'tokens': len(logp), 'nll': float(-logp.mean())}))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = data[a]['logp'] - data[b]['logp']
            print(json.dumps({'a': a, 'b': b, 'mean_signed': float(d.mean()), 'mean_abs': float(np.abs(d).mean()),
                              'p99_abs': float(np.quantile(np.abs(d), 0.99))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('trainer', 'sglang'):
        p = sub.add_parser(name)
        p.add_argument('--turns', required=True)
        p.add_argument('--chains', type=int, default=24)
        p.add_argument('--max-tokens', type=int, default=12000)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--out', required=True)
        if name == 'trainer':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--base', required=True)
        else:
            p.add_argument('--url', required=True)
            p.add_argument('--concurrency', type=int, default=4)
    p = sub.add_parser('compare')
    p.add_argument('files', nargs='+')
    args = parser.parse_args()
    {'trainer': trainer, 'sglang': sglang, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
