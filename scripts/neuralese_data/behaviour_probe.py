"""Behaviour probe: how far a policy has moved from the start policy on calls the start policy itself sampled, cheap
enough to run after every learner step (DECISIONS 2026-11-14: grpo-v2-run1's drift entered at steps 3-4 and was first seen
by a held eval at step 8).

``build`` takes model calls sampled by the start policy (archived GRPO groups, ``policy`` = its sha) and keeps each
call's prompt and its reasoning up to and including the first ``</think>``. ``measure`` scores them with a model and
returns per reasoning token: the sampled tokens' log-probability (its change from the start policy's: not a KL, the calls
are selected (reasoning closed within 3k tokens) and were sampled under top-p; run1 step 8 raised it by 0.08 nats), the
entropy, the expected ``</think>``
closes per 1,000 reasoning tokens and the log-probability of ``</think>`` where the start policy closed.

End of turn (``end_calls``, from 2026-11-19): decision readouts the start policy answered, each the prompt and its chosen
reply ``</think>"VALUE"<|im_end|>``. ``measure`` scores the reply's last position: the log-probability of
``<|im_end|>`` and of a second ``</think>`` there. grpo-v4-run2 moved 0.6-0.96 of that position's mass to a second
``</think>`` by step 12 (DECISIONS 2026-11-19), which the reasoning-token measures above do not see. The probe file keeps
the start policy's own values (``end_start``) so a resumed run without them in its start measurement still compares.

    python behaviour_probe.py build --rollouts DIR --policy SHA --served EXPORT_SHA --out probe.json [--calls 48]
    python behaviour_probe.py extend --probe v1.json --rollouts DIR --out v2.json [--start-checkpoint CKPT]
    python behaviour_probe.py measure --probe probe.json --out probe.jsonl BASE_CKPT CKPT...   # Pop's 8 GB GPU fits it
"""
import argparse
import io
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

SCHEMA = 'natlang.behaviour-probe/1'


def build(rollouts, policy, out, calls=48, max_prompt_tokens=8000, max_think_tokens=3000, per_group=2, seed=0, served=None,
          end_calls=24):
    """Up to ``calls`` start-policy calls without imported state whose reasoning closes within ``max_think_tokens``, at
    most ``per_group`` from one archived group (groups in a seeded order). A call's ``policy`` names the run's start
    checkpoint even after in-place weight updates, so the calls the start policy itself sampled are selected by the
    served weights' sha (``served``: the start policy's ``actor_export`` sha256); behaviour-probe-v0 was built without it
    and 44 of its 48 calls came from updated policies (2026-11-15)."""
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
                    if call['imports'] or call['policy'] != policy or len(call['prompt']) > max_prompt_tokens \
                            or (served is not None and call['served'] != served):
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
    probe = {'schema': SCHEMA, 'policy': policy, 'served': served, 'close_id': close, 'calls': kept}
    if end_calls:
        probe.update(collect_end_calls(rollouts, policy, served, end_calls, seed=seed))
    Path(out).write_text(json.dumps(probe) + '\n')
    return len(kept), sum(len(c['prompt']) + len(c['think']) for c in kept)


def collect_end_calls(rollouts, policy, served, calls=24, max_prompt_tokens=8000, max_reply_tokens=24, per_group=6,
                      per_task=2, seed=0):
    """Up to ``calls`` decision readouts the start policy answered (``served`` as in ``build``), at most ``per_group``
    from one archived group: the prompt and the chosen reply ``</think>`` VALUE ``<|im_end|>`` (at most
    ``max_reply_tokens``), distinct, one per episode and at most ``per_task`` per task. ``rollouts``: one archive directory or
    several. Returns {'end_id', 'end_calls'}. Only a run's first groups were sampled by
    the start policy (grpo-v2-run2: 193 of its decisions in a handful of groups), hence several per group."""
    import collections
    import zstandard
    from lower_rollout_turns import decision_turns
    close, end = close_id(), token_id('<|im_end|>')
    archives = [a for d in ([rollouts] if isinstance(rollouts, (str, Path)) else rollouts)
                for a in sorted(Path(d).glob('g*.rollouts.jsonl.zst'))]
    random.Random(seed).shuffle(archives)
    kept, stats, tasks, seen = [], collections.Counter(), collections.Counter(), set()
    for archive in archives:
        if len(kept) >= calls:
            break
        taken = 0
        with archive.open('rb') as handle:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(handle), encoding='utf-8'):
                if taken >= per_group or len(kept) >= calls:
                    break
                row = json.loads(line)
                if tasks[row.get('task_id')] >= per_task:
                    continue
                for call in decision_turns(row, stats, None) if row.get('turns') else ():
                    reply = [int(i) for i in call['options'][call['chosen']]]
                    if (call['imports'] or call['policy'] != policy or call.get('served') != served
                            or len(call['prompt']) > max_prompt_tokens or not 3 <= len(reply) <= max_reply_tokens
                            or reply[0] != close or reply[-1] != end or reply.count(close) != 1
                            or (tuple(call['prompt']), tuple(reply)) in seen):
                        continue      # sibling episodes repeat a decision exactly: each distinct row once
                    seen.add((tuple(call['prompt']), tuple(reply)))
                    kept.append({'group': '/'.join([*archive.parts[-4:-2], archive.name.split('.')[0]]), 'task_id': row['task_id'],
                                 'prompt': [int(i) for i in call['prompt']], 'reply': reply})
                    taken += 1
                    tasks[row['task_id']] += 1
                    break
    return {'end_id': end, 'end_calls': kept}


def token_id(token):
    from tokenizers import Tokenizer
    from grpo import BASE
    return Tokenizer.from_file(str(Path(BASE) / 'tokenizer.json')).token_to_id(token)


def close_id():
    return token_id('</think>')


# One pack layout for every caller: the fused kernels' BF16 rounding depends on the pack's shape, and on these tail
# tokens it moves a row by up to ~0.4 nats (row 0 of v2: -4.17 packed with 3 others, -4.56 with all 24; FP32 -4.44,
# HF BF16 -3.94; packing itself is exact: a row alone equals it in the same pack). Start and now must share the layout.
END_PACK_TOKENS = 16000


def measure_end(model, probe, pack_tokens=END_PACK_TOKENS):
    """Per end call of ``probe``: {'end_logp': [...], 'reclose_logp': [...]}, the log-probabilities of ``<|im_end|>``
    and of ``</think>`` at the reply's last position (where the start policy ended its turn)."""
    import torch
    from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
    weight = model.get_output_embeddings().weight
    out = {'end_logp': [], 'reclose_logp': []}
    batches, batch, size = [], [], 0
    for call in probe.get('end_calls') or ():
        n = len(call['prompt']) + len(call['reply'])
        if batch and size + n > pack_tokens:
            batches.append(batch)
            batch, size = [], 0
        batch.append(call)
        size += n
    if batch:
        batches.append(batch)
    with torch.no_grad():
        for batch in batches:
            seqs = [call['prompt'] + call['reply'] for call in batch]
            ids = torch.tensor([i for s in seqs for i in s], device='cuda')
            h = lfm3_fast_train.hidden(model, ids, [len(s) for s in seqs] if len(seqs) > 1 else None)
            ends = torch.tensor(list(itertools.accumulate(len(s) for s in seqs)), device='cuda') - 2
            lf = torch.nn.functional.linear(h[ends], weight).float().log_softmax(-1)
            out['end_logp'].extend(lf[:, probe['end_id']].tolist())
            out['reclose_logp'].extend(lf[:, probe['close_id']].tolist())
    return out


def measure(model, probe, pack_tokens=16000, chunk=1024):
    """Per-call means over the reasoning tokens of ``probe`` (``build``'s JSON) under ``model`` (prepared for
    ``lfm3_fast_train``): {'sampled_logp': [...], 'entropy': [...], 'close_per_1k': [...], 'close_logp': [...]}."""
    import torch
    from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
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
    if probe.get('end_calls'):
        out.update(measure_end(model, probe))
    return out


def summary(now, start=None, probe=None):
    """Means of ``measure``'s per-call values, and against the start policy's (same probe): ``logp_change`` (mean over
    calls of now - start sampled logp per token, nats), ``close_ratio`` (expected closes now / start) and the entropy and
    close-logp changes. End of turn: ``end_logp`` (mean), ``end_logp_min``, ``reclose_p`` (mean probability of a second
    ``</think>`` where the start policy ended) and ``end_logp_change`` (mean over calls of now - start; the start's from
    ``start``, else from the probe file's ``end_start``)."""
    mean = {k: float(np.mean(v)) for k, v in now.items() if len(v)}
    record = {k: round(v, 5) for k, v in mean.items()}
    if now.get('end_logp'):
        record.update(end_logp_min=round(float(np.min(now['end_logp'])), 4),
                      reclose_p=round(float(np.mean(np.exp(now['reclose_logp']))), 5))
        record.pop('reclose_logp', None)
        base = start if start is not None and start.get('end_logp') else (probe or {}).get('end_start')
        if base is not None and len(base['end_logp']) == len(now['end_logp']):
            record['end_logp_change'] = round(float(np.mean(np.subtract(now['end_logp'], base['end_logp']))), 4)
    if start is not None:
        base = {k: float(np.mean(v)) for k, v in start.items()}
        record.update(logp_change=round(float(np.mean(np.subtract(now['sampled_logp'], start['sampled_logp']))), 5),
                      close_ratio=round(mean['close_per_1k'] / max(1e-9, base['close_per_1k']), 4),
                      entropy_change=round(mean['entropy'] - base['entropy'], 5),
                      close_logp_change=round(mean['close_logp'] - base['close_logp'], 4))
    return record


def shifted(record, logp=0.02, entropy=0.03, close=(0.67, 1.5), end=1.0):
    """Whether the probe has moved past the sentinel bounds: sampled-token logp or entropy changed by more than ``logp`` /
    ``entropy`` nats, expected closes outside ``close`` x the start's, or the end of turn after a short answer down by
    more than ``end`` nats on average. On run1's checkpoints (2026-11-14): step 2 +0.008 / -0.007 / 1.10x (inside), step 8
    +0.080 / -0.010 / 8.9x, step 12 +0.040 / +0.075 / 1.57x. grpo-v4-run2's end of turn on gap-v1's short-answer rows
    (DECISIONS 2026-11-19): base -0.0 to -2.5 nats, step 12 -0.07 to -13.6."""
    if record.get('end_logp_change', 0.) < -end:
        return True
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
    b.add_argument('--served', required=True, help="sha256 of the start policy's served export (capture actor_export)")
    b.add_argument('--out', required=True)
    b.add_argument('--calls', type=int, default=48)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--end-calls', type=int, default=24, help='decision readouts for the end-of-turn rows (0: none)')
    e = sub.add_parser('extend', help="add end-of-turn rows to an existing probe (its calls kept), from the start policy's "
                                      'decision readouts in --rollouts')
    e.add_argument('--probe', required=True)
    e.add_argument('--rollouts', required=True, nargs='+', help='archive directories (start-policy groups)')
    e.add_argument('--out', required=True)
    e.add_argument('--end-calls', type=int, default=24)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--start-checkpoint', help="the start policy's native checkpoint: its end-of-turn values are stored "
                                              'in the probe (end_start; needs a GPU)')
    m = sub.add_parser('measure', help='probe native checkpoints, one JSON line each (forward only: fits an 8 GB GPU)')
    m.add_argument('--probe', required=True)
    m.add_argument('--start', help="``measure`` JSON of the start policy (else the first checkpoint's)")
    m.add_argument('--out', required=True, help='JSONL, appended; checkpoints already in it are skipped')
    m.add_argument('checkpoints', nargs='+')
    args = parser.parse_args(argv)
    if args.command == 'measure':
        return measure_checkpoints(args)
    if args.command == 'extend':
        return extend(args)
    calls, tokens = build(args.rollouts, args.policy, args.out, calls=args.calls, seed=args.seed, served=args.served,
                          end_calls=args.end_calls)
    print(json.dumps({'calls': calls, 'tokens': tokens, 'out': args.out}))


def extend(args):
    probe = json.loads(Path(args.probe).read_text())
    probe.update(collect_end_calls(args.rollouts, probe['policy'], probe['served'], args.end_calls, seed=args.seed))
    if args.start_checkpoint:
        model = load_probe_model(args.start_checkpoint)
        probe['end_start'] = {**measure_end(model, probe), 'checkpoint': args.start_checkpoint}
    Path(args.out).write_text(json.dumps(probe) + '\n')
    print(json.dumps({'calls': len(probe['calls']), 'end_calls': len(probe['end_calls']), 'out': args.out,
                      **({'end_start': summary({k: v for k, v in probe['end_start'].items() if k != 'checkpoint'})}
                         if 'end_start' in probe else {})}))


def load_probe_model(checkpoint):
    import torch
    import exp0_sft
    from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
    from grpo import BASE
    model, _, _ = exp0_sft.load_model(checkpoint, BASE, device='cpu', dtype=torch.bfloat16)
    model.to('cuda').eval()
    lfm3_fast_train.prepare(model)
    return model


def measure_checkpoints(args):
    import time
    import torch
    probe = json.loads(Path(args.probe).read_text())
    out = Path(args.out)
    done = {json.loads(l)['checkpoint']: json.loads(l) for l in out.read_text().splitlines()} if out.exists() else {}
    start = json.loads(Path(args.start).read_text()) if args.start else None
    for checkpoint in args.checkpoints:
        if checkpoint in done:
            start = start or done[checkpoint]['measure']
            continue
        t = time.time()
        model = load_probe_model(checkpoint)
        now = measure(model, probe, pack_tokens=8000)
        record = summary(now, start, probe)
        start = start or now
        line = {'checkpoint': checkpoint, 'summary': record, 'shifted': shifted(record), 'measure': now,
                'seconds': round(time.time() - t, 1), 'cuda_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        with out.open('a') as handle:
            handle.write(json.dumps(line) + '\n')
        print(json.dumps({k: line[k] for k in ('checkpoint', 'summary', 'shifted', 'seconds', 'cuda_peak_gib')}), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
