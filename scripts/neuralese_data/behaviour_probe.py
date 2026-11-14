"""Behaviour probe: how far a policy has moved from the start policy on calls the start policy itself sampled, cheap
enough to run after every learner step (DECISIONS 2026-11-14: grpo-v2-run1's drift entered at steps 3-4 and was first seen
by a held eval at step 8).

``build`` takes model calls sampled by the start policy (archived GRPO groups, ``policy`` = its sha) and keeps each
call's prompt and its reasoning up to and including the first ``</think>``. ``measure`` scores them with a model and
returns per reasoning token: the sampled tokens' log-probability (its change from the start policy's: not a KL, the calls
are selected (reasoning closed within 3k tokens) and were sampled under top-p; run1 step 8 raised it by 0.08 nats), the
entropy, the expected ``</think>``
closes per 1,000 reasoning tokens and the log-probability of ``</think>`` where the start policy closed.

    python behaviour_probe.py build --rollouts DIR --policy SHA --out probe.json [--calls 48]
"""
import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

SCHEMA = 'natlang.behaviour-probe/1'


def build(rollouts, policy, out, calls=48, max_prompt_tokens=8000, max_think_tokens=3000, per_group=2, seed=0):
    """Up to ``calls`` start-policy calls without imported state whose reasoning closes within ``max_think_tokens``, at
    most ``per_group`` from one archived group (groups in a seeded order)."""
    import collections
    import zstandard
    from lower_rollout_turns import episode_turns
    archives = sorted(Path(rollouts).glob('g*.rollouts.jsonl.zst'))
    random.Random(seed).shuffle(archives)
    close = None
    kept, stats = [], collections.Counter()
    for archive in archives:
        if len(kept) >= calls:
            break
        taken = 0
        with archive.open('rb') as handle:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(handle), encoding='utf-8'):
                row = json.loads(line)
                if not row.get('turns') or taken >= per_group:
                    continue
                for call in episode_turns(row, stats, None):
                    if call['imports'] or call['policy'] != policy or len(call['prompt']) > max_prompt_tokens:
                        continue
                    if close is None:
                        close = close_id()
                    sample = np.asarray(call['sample'])
                    found = np.flatnonzero(sample == close)
                    if not len(found) or found[0] + 1 > max_think_tokens or found[0] < 16:
                        continue
                    kept.append({'group': archive.name.split('.')[0], 'task_id': row['task_id'],
                                 'prompt': [int(i) for i in call['prompt']],
                                 'think': [int(i) for i in sample[:found[0] + 1]]})
                    taken += 1
                    break
                if len(kept) >= calls:
                    break
    Path(out).write_text(json.dumps({'schema': SCHEMA, 'policy': policy, 'close_id': close, 'calls': kept}) + '\n')
    return len(kept), sum(len(c['prompt']) + len(c['think']) for c in kept)


def close_id():
    from tokenizers import Tokenizer
    from grpo import BASE
    return Tokenizer.from_file(str(Path(BASE) / 'tokenizer.json')).token_to_id('</think>')


def measure(model, probe, pack_tokens=16000, chunk=1024):
    """Per-call means over the reasoning tokens of ``probe`` (``build``'s JSON) under ``model`` (prepared for
    ``lfm3_fast_train``): {'sampled_logp': [...], 'entropy': [...], 'close_per_1k': [...], 'close_logp': [...]}."""
    import torch
    import lfm3_fast_train
    weight = model.get_output_embeddings().weight
    close = probe['close_id']
    out = {'sampled_logp': [], 'entropy': [], 'close_per_1k': [], 'close_logp': []}
    calls = probe['calls']
    batches, batch, size = [], [], 0
    for call in calls:
        n = len(call['prompt']) + len(call['think'])
        if batch and size + n > pack_tokens:
            batches.append(batch)
            batch, size = [], 0
        batch.append(call)
        size += n
    if batch:
        batches.append(batch)
    with torch.no_grad():
        for batch in batches:
            seqs = [call['prompt'] + call['think'] for call in batch]
            ids = torch.tensor([i for s in seqs for i in s], device='cuda')
            h = lfm3_fast_train.hidden(model, ids, [len(s) for s in seqs] if len(seqs) > 1 else None)
            offset = 0
            for call, seq in zip(batch, seqs):
                start = offset + len(call['prompt']) - 1            # the hidden state that predicts think[0]
                n = len(call['think'])
                target = torch.tensor(call['think'], device='cuda')
                logp, entropy, close_lp = [], [], []
                for a in range(0, n, chunk):
                    lf = torch.nn.functional.linear(h[start + a:start + min(n, a + chunk)], weight).float().log_softmax(-1)
                    logp.append(lf.gather(1, target[a:a + chunk, None])[:, 0])
                    entropy.append(-(lf.exp() * lf).sum(-1))
                    close_lp.append(lf[:, close])
                    del lf
                logp, entropy, close_lp = torch.cat(logp), torch.cat(entropy), torch.cat(close_lp)
                out['sampled_logp'].append(logp[:-1].mean().item())
                out['entropy'].append(entropy[:-1].mean().item())
                out['close_per_1k'].append(1000 * close_lp[:-1].exp().mean().item())
                out['close_logp'].append(close_lp[-1].item())
                offset += len(seq)
    return out


def summary(now, start=None):
    """Means of ``measure``'s per-call values, and against the start policy's (same probe): ``logp_change`` (mean over
    calls of now - start sampled logp per token, nats), ``close_ratio`` (expected closes now / start) and the entropy and
    close-logp changes."""
    mean = {k: float(np.mean(v)) for k, v in now.items()}
    record = {k: round(v, 5) for k, v in mean.items()}
    if start is not None:
        base = {k: float(np.mean(v)) for k, v in start.items()}
        record.update(logp_change=round(float(np.mean(np.subtract(now['sampled_logp'], start['sampled_logp']))), 5),
                      close_ratio=round(mean['close_per_1k'] / max(1e-9, base['close_per_1k']), 4),
                      entropy_change=round(mean['entropy'] - base['entropy'], 5),
                      close_logp_change=round(mean['close_logp'] - base['close_logp'], 4))
    return record


def shifted(record, logp=0.02, entropy=0.03, close=(0.67, 1.5)):
    """Whether the probe has moved past the sentinel bounds: sampled-token logp or entropy changed by more than ``logp`` /
    ``entropy`` nats, or expected closes outside ``close`` x the start's. On run1's checkpoints (2026-11-14): step 2 +0.008 /
    -0.007 / 1.10x (inside), step 8 +0.080 / -0.010 / 8.9x, step 12 +0.040 / +0.075 / 1.57x."""
    if 'logp_change' not in record:
        return False
    return (abs(record['logp_change']) > logp or abs(record['entropy_change']) > entropy
            or not close[0] <= record['close_ratio'] <= close[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--rollouts', required=True)
    b.add_argument('--policy', required=True, help='sha256 of the start policy whose calls are kept')
    b.add_argument('--out', required=True)
    b.add_argument('--calls', type=int, default=48)
    b.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    calls, tokens = build(args.rollouts, args.policy, args.out, calls=args.calls, seed=args.seed)
    print(json.dumps({'calls': calls, 'tokens': tokens, 'out': args.out}))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
