#!/usr/bin/env python3
"""Use ablation of learned-operator states (plans/neuralese/OPERATORS_RESTORE_20261117.md §3.5).

For every consumer chain of `operator_rows.py` (a test split), the consumer's target NLL (nats per target token) with:

* ``actual``: the state computed in-graph from its own producer document (the in-context capture, the pass-through
  `lfm3_fast_train.passthrough_plan`, no gradient);
* ``standalone``: the same reply content written from position 0 (`[produce_open] + span`, today's text write):
  whether the in-context capture carries more than its tokens;
* ``shuffled``: the state of another case of the same source (a derangement), its span spliced in at the import;
* ``text-cut``: the runtime's text cut of the value instead of any state (`*.test-textcut.pkl`).

Reported per source: mean NLL per arm, the paired per-case differences against ``actual`` with a sign-flip
permutation p and a case-bootstrap 95% CI, by seed provenance where the rows record it.

    operator_ablation.py --rows DIR --checkpoint C --base B --out report.json [--limit N] [--sources a/b,...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp0_sft  # noqa: E402
from natlang_neuralese.train import lfm3_fast as lfm3_fast_train  # noqa: E402

ARMS = ('actual', 'standalone', 'shuffled', 'text-cut')


def with_span(chain, key, span, document):
    """``chain`` with the import ``key``'s placeholder rows replaced by ``span`` (any length) read from ``document``."""
    (at, _), = [item for item in chain['imports'] if item[1] == key]
    old = chain['sources'][key]
    n = old['stop'] - old['start']
    shift = len(span) - n
    ids = np.concatenate([chain['ids'][:at], np.asarray(span, np.int32), chain['ids'][at + n:]])
    segs = [(start + shift if start > at else start, length, row) for start, length, row in chain['segs']]
    imports = [(start + shift if start > at else start, other) for start, other in chain['imports']]
    sources = {**chain['sources'], key: document}
    return {**chain, 'ids': ids, 'segs': segs, 'imports': imports, 'sources': sources}


def span_of(document):
    return document['ids'][document['start']:document['stop']]


def standalone(chain, produce_open):
    out = chain
    for _, key in chain['imports']:
        span = span_of(chain['sources'][key])
        document = {'ids': np.concatenate([np.asarray([produce_open], np.int32), span]), 'start': 1,
                    'stop': 1 + len(span)}
        out = {**out, 'sources': {**out['sources'], key: document}}
    return out


def shuffled(chains, seed=0):
    """Each chain with every import's state taken from a different chain (a derangement over the list)."""
    order = list(range(len(chains)))
    if len(order) < 2:
        return [None] * len(order)
    rng = random.Random(seed)
    while True:
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            break
    out = []
    for chain, donor_index in zip(chains, order):
        donor = chains[donor_index]
        donor_docs = [donor['sources'][key] for _, key in donor['imports']]
        new = chain
        for index, (_, key) in enumerate(chain['imports']):
            document = donor_docs[index % len(donor_docs)]
            new = with_span(new, key, span_of(document), document)
        out.append(new)
    return out


def chain_nll(model, chain):
    """Mean NLL per target token of one chain (fast path, pass-through imports, no gradient)."""
    ids, positions, targets, weights, lengths, imports = exp0_sft.row_inputs([chain], 'cuda')
    h = lfm3_fast_train.hidden(model, ids, lengths, imports) if imports is not None or len(lengths) > 1 \
        else lfm3_fast_train.hidden(model, ids)
    nll = exp0_sft.token_nll(model, h, positions, targets)
    return float((nll * weights).sum() / weights.sum())


def paired(diffs, rounds=10000, seed=0):
    diffs = np.asarray(diffs, np.float64)
    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    flips = rng.choice([-1., 1.], size=(rounds, len(diffs)))
    p = float(((np.abs((flips * diffs).mean(1)) >= observed).sum() + 1) / (rounds + 1))
    boot = rng.integers(0, len(diffs), size=(rounds, len(diffs)))
    means = diffs[boot].mean(1)
    return {'mean': float(diffs.mean()), 'p': p, 'ci95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            'better_share': float((diffs > 0).mean()), 'n': len(diffs)}


def measure(model, consumers, cuts, produce_open, limit=0):
    consumers = consumers[:limit] if limit else consumers
    cut_by_task = {chain['task']: chain for chain in cuts}
    arms = {'actual': consumers, 'standalone': [standalone(c, produce_open) for c in consumers],
            'shuffled': shuffled(consumers), 'text-cut': [cut_by_task.get(c['task']) for c in consumers]}
    scores = {arm: [] for arm in ARMS}
    with torch.no_grad():
        for index in range(len(consumers)):
            for arm in ARMS:
                chain = arms[arm][index]
                scores[arm].append(None if chain is None else chain_nll(model, chain))
    report = {'cases': len(consumers),
              'mean_nll': {arm: float(np.mean([s for s in values if s is not None])) if any(s is not None for s in values)
                           else None for arm, values in scores.items()}}
    for arm in ARMS[1:]:
        pairs = [other - actual for actual, other in zip(scores['actual'], scores[arm]) if other is not None]
        if len(pairs) >= 2:
            report[f'{arm}_minus_actual'] = paired(pairs)
    report['per_case'] = [{'task': c['task'], **{arm: scores[arm][i] for arm in ARMS}} for i, c in enumerate(consumers)]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', required=True, help='operator_rows.py output directory')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sources', help='comma list of <operator>/<source> (default: every test file)')
    parser.add_argument('--limit', type=int, default=0, help='consumer chains per source')
    args = parser.parse_args()
    rows = Path(args.rows)
    manifest = json.loads((rows / 'manifest.json').read_text())
    produce_open = manifest['role_ids'][0]
    from natlang_neuralese.train.lfm3_learner import load_effective_policy
    model, state, _learner = load_effective_policy(args.checkpoint, args.base, device='cuda', dtype=torch.bfloat16)
    if state and sorted(k for k in state['interface_state'] if k.startswith('adapter.')):
        raise SystemExit('the pass-through embeds producer spans with the embedding: a producer adapter is not modelled')
    wanted = set(args.sources.split(',')) if args.sources else None
    report = {'rows': str(rows), 'checkpoint': args.checkpoint, 'sources': {}}
    for name, entry in sorted(manifest['sources'].items()):
        files = entry.get('files') or {}
        if 'test' not in files or (wanted and name not in wanted):
            continue
        chains = pickle.load(open(files['test']['path'], 'rb'))['chains']
        consumers = [c for c in chains if c['segs'][0][2].get('kind') == 'consumer']
        cuts = pickle.load(open(files['test-textcut']['path'], 'rb'))['chains'] if 'test-textcut' in files else []
        report['sources'][name] = measure(model, consumers, cuts, produce_open, args.limit)
        summary = {k: v for k, v in report['sources'][name].items() if k != 'per_case'}
        print(json.dumps({name: summary}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
