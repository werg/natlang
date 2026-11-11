#!/usr/bin/env python3
"""Experiment 0 (plans/neuralese/APPROACH_REVIEW_20261111.md section 4): a plain FP32-master AdamW SFT of a native
LFM3 checkpoint's backbone on per-call rows lowered from verified-successful student episodes
(`lower_rollout_turns.py`), written back as a native checkpoint the existing actor serves.

The native backbone state is a plain HF state dict, so the loop is the vendored HF LFM3 (`third_party/lfm3`) in FP32
with bf16 autocast, assistant (sampled output) tokens only carrying loss. The Neuralese interface (marker rows, token
override rows, producer adapter) is frozen and copied unchanged: the 6 sparse embedding rows are merged into the
embedding for the forward (as `export/native_hf.merged_state` does) and their gradient is masked.

A call's prompt extends the previous call's (prompt + sampled output) of the same invocation in most cases, so such a
chain is one sequence whose sampled segments are all supervised: equal to per-call rows (causal), 10x cheaper.

    exp0_sft.py parity --turns T.pkl --checkpoint C --base B           # HF forward vs the actor's recorded logprobs
    exp0_sft.py train  --turns T.pkl --checkpoint C --base B --out-dir D
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))

MAX_LEN = 32000


def chains_of(episodes, max_len=MAX_LEN):
    """Sequences: [{'ids': int32[], 'segs': [(start, length, row)], 'task', 'round'}]; calls of one invocation whose prompt
    extends the previous call's full text share a sequence."""
    chains = []
    for episode in episodes:
        by_invocation = collections.defaultdict(list)
        for row in episode['rows']:
            by_invocation[row['invocation']].append(row)
        for rows in by_invocation.values():
            rows.sort(key=lambda row: row['index'])
            current = None
            for row in rows:
                ids, sample = row['prompt'], row['sample']
                if len(ids) + len(sample) > max_len:
                    if current is not None:
                        chains.append(current)
                    current = None
                    continue
                if current is not None:
                    full = current['ids']
                    if len(ids) >= len(full) and np.array_equal(ids[:len(full)], full):
                        current['ids'] = np.concatenate([ids, sample])
                        current['segs'].append((len(ids), len(sample), row))
                        continue
                    chains.append(current)
                current = {'ids': np.concatenate([ids, sample]), 'segs': [(len(ids), len(sample), row)],
                           'task': episode['task_id'], 'round': episode['round']}
            if current is not None:
                chains.append(current)
    return chains


def load_model(checkpoint, base, device='cuda', dtype=torch.float32):
    from transformers import AutoConfig, AutoModelForCausalLM
    from natlang_neuralese.third_party import lfm3
    lfm3.register()
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    state = CheckpointPolicy.load(Path(checkpoint))
    config = AutoConfig.from_pretrained(base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, config=config, dtype=dtype, trust_remote_code=False,
                                                 attn_implementation='sdpa')
    backbone = dict(state['backbone_state'])
    embedding = backbone['model.embed_tokens.weight']
    head = backbone.pop('lm_head.weight', None)
    if head is not None and not torch.equal(head, embedding):
        raise ValueError('untied lm_head: this script serves tied-embedding checkpoints only')
    if not config.tie_word_embeddings:
        raise ValueError('tied embeddings expected')
    interface = state['interface_state']
    merged = embedding.clone()
    sparse = [(state['model_config']['marker_ids'], interface['input_rows'])]
    if state['model_config'].get('token_override_ids'):
        sparse.append((state['model_config']['token_override_ids'], interface['extra_input_rows']))
    special = []
    for ids, rows in sparse:
        merged[torch.tensor(list(ids))] = rows.to(merged.dtype)
        special += [int(i) for i in ids]
    backbone['model.embed_tokens.weight'] = merged
    missing = model.load_state_dict({k: v.to(dtype) for k, v in backbone.items()}, strict=False)
    if missing.unexpected_keys or [k for k in missing.missing_keys if k != 'lm_head.weight']:
        raise ValueError(f'state mismatch {missing}')
    model.to(device)
    return model, state, special


def hidden(model, ids):
    return model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]


def _nll(weight, h, target):
    logits = F.linear(h, weight).float()
    return F.cross_entropy(logits, target, reduction='none')


def token_nll(model, h, positions, targets, chunk=2048):
    """NLL (fp32) of ``targets`` predicted from hidden rows ``positions`` (logits only for those rows, chunk-checkpointed)."""
    out = []
    weight = model.lm_head.weight
    for start in range(0, len(positions), chunk):
        rows = h[positions[start:start + chunk]]
        target = targets[start:start + chunk]
        if torch.is_grad_enabled():
            out.append(torch_checkpoint(_nll, weight, rows, target, use_reentrant=False))
        else:
            out.append(_nll(weight, rows, target))
    return torch.cat(out)


def chain_tensors(chain, device):
    ids = torch.from_numpy(chain['ids'].astype(np.int64)).to(device)
    positions = np.concatenate([np.arange(start - 1, start + length - 1) for start, length, _ in chain['segs']])
    targets = np.concatenate([row['sample'] for _, _, row in chain['segs']])
    return ids, torch.from_numpy(positions).to(device), torch.from_numpy(targets.astype(np.int64)).to(device)


def parity(args):
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    chains = [c for c in chains_of(episodes) if len(c['ids']) < args.max_tokens][:args.samples]
    model, _, _ = load_model(args.checkpoint, args.base, dtype=torch.bfloat16 if args.bf16_params else torch.float32)
    model.eval()
    diffs, nll = [], []
    for chain in chains:
        ids, positions, targets = chain_tensors(chain, 'cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.bf16_params):
            h = hidden(model, ids)
            value = -token_nll(model, h, positions, targets)
        recorded = torch.from_numpy(np.concatenate([row['logp'] for _, _, row in chain['segs']])).cuda()
        diffs.append((value.float() - recorded))
        nll.append(-value.float().mean().item())
        d = diffs[-1]
        print(json.dumps({'tokens': len(ids), 'sampled': len(d), 'mean_abs': d.abs().mean().item(), 'mean_signed': d.mean().item(),
                          'p99_abs': d.abs().quantile(0.99).item(), 'nll_hf': nll[-1], 'nll_actor': -recorded.mean().item()}), flush=True)
    d = torch.cat(diffs)
    print(json.dumps({'parity_tokens': len(d), 'mean_abs': d.abs().mean().item(), 'mean_signed': d.mean().item(),
                      'frac_over_0.5': (d.abs() > 0.5).float().mean().item()}))


def write_checkpoint(model, state, special, out_path, source_sha256, step):
    """A pruned (inference-only) native checkpoint: the trained backbone in the source's bf16 layout, the source's interface."""
    backbone = {}
    original = state['backbone_state']
    for name, value in original.items():
        if name == 'lm_head.weight':
            continue
        backbone[name] = value
    trained = {k: v.detach().to('cpu', torch.bfloat16) for k, v in model.state_dict().items() if k != 'lm_head.weight'}
    changed = 0
    for name in list(backbone):
        new = trained[name]
        if name == 'model.embed_tokens.weight':
            new = new.clone()
            new[torch.tensor(special)] = original[name][torch.tensor(special)]
        changed += int((new != original[name]).sum())
        backbone[name] = new.contiguous()
    if 'lm_head.weight' in original:
        backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    out = {k: state[k] for k in ('schema', 'model_config', 'interface_state')}
    out.update(backbone_state=backbone, step=int(state['step']), inference_only=True, optimizer_pruned=True,
               optimizer_pruned_keys=['training'],
               inference_source={'schema': 'natlang.native-inference-source/1', 'training_checkpoint_sha256': source_sha256,
                                 'training_checkpoint_bytes': 0, 'step': int(state['step'])})
    torch.save(out, out_path)
    return changed


def train(args):
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    if args.max_per_task:
        # compute-bound (GB10: about 1k tokens/s of fwd+bwd at 10k context): the shortest verified episodes of each task
        by_task = collections.defaultdict(list)
        for episode in episodes:
            by_task[episode['task_id']].append((sum(len(c['ids']) for c in chains_of([episode])), episode['round'], episode['sample'], episode))
        episodes = [item[-1] for task in sorted(by_task) for item in sorted(by_task[task], key=lambda x: x[:3])[:args.max_per_task]]
    chains = chains_of(episodes)
    loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
    print(json.dumps({'episodes': len(episodes), 'sequences': len(chains), 'sequence_tokens': int(sum(len(c['ids']) for c in chains)),
                      'loss_tokens': int(loss_tokens), 'epochs': args.epochs}), flush=True)
    model, state, special = load_model(args.checkpoint, args.base)
    # FP32 master weights live in the optimizer; the model computes in BF16 (as the actor does) and its BF16 gradients
    # accumulate into FP32 buffers
    masters = {name: p.detach().clone().float() for name, p in model.named_parameters()}
    model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    special_index = torch.tensor(special, device='cuda')
    live = {name: p for name, p in model.named_parameters()}
    parameters = [masters[name] for name in live]
    for master in parameters:
        master.grad = torch.zeros_like(master)
    decay = [p for p in parameters if p.ndim >= 2]
    no_decay = [p for p in parameters if p.ndim < 2]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.}],
                                  lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)  # parameters are the FP32 masters
    embedding_name = next(n for n, p in live.items() if p is model.get_input_embeddings().weight)
    embedding_master = masters[embedding_name]
    saved_rows = embedding_master[special_index].clone()
    rng = random.Random(args.seed)
    order = []
    for _ in range(args.epochs):
        epoch = list(range(len(chains)))
        rng.shuffle(epoch)
        order += epoch
    # optimizer steps: groups of sequences with at least `batch_loss_tokens` loss tokens
    steps, group, tokens = [], [], 0
    for index in order:
        group.append(index)
        tokens += sum(length for _, length, _ in chains[index]['segs'])
        if tokens >= args.batch_loss_tokens:
            steps.append(group)
            group, tokens = [], 0
    if group and tokens >= args.batch_loss_tokens // 2:
        steps.append(group)
    total = len(steps)
    print(json.dumps({'optimizer_steps': total, 'lr': args.lr, 'batch_loss_tokens': args.batch_loss_tokens}), flush=True)
    log = open(Path(args.out_dir) / 'train.jsonl', 'a')
    started = time.time()
    forward_tokens = 0
    for step, group in enumerate(steps):
        fraction = step / max(1, total - 1)
        scale = min(1., (step + 1) / args.warmup) * (args.min_lr_fraction + (1 - args.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * fraction)))
        for g in optimizer.param_groups:
            g['lr'] = args.lr * scale
        denominator = sum(length for i in group for _, length, _ in chains[i]['segs'])
        total_nll, correct = 0., 0
        for i in group:
            chain = chains[i]
            ids, positions, targets = chain_tensors(chain, 'cuda')
            h = hidden(model, ids)
            nll = token_nll(model, h, positions, targets)
            (nll.sum() / denominator).backward()
            for name, p in live.items():
                masters[name].grad.add_(p.grad.float())
                p.grad = None
            total_nll += nll.sum().item()
            forward_tokens += len(ids)
            del h, nll
        embedding_master.grad[special_index] = 0
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        for p in parameters:
            p.grad.zero_()
        with torch.no_grad():
            embedding_master[special_index] = saved_rows
            for name, p in live.items():
                p.copy_(masters[name])
        record = {'step': step + 1, 'of': total, 'loss': total_nll / denominator, 'lr': args.lr * scale, 'grad_norm': norm,
                  'sequences': len(group), 'loss_tokens': denominator, 'elapsed_s': round(time.time() - started, 1),
                  'forward_tokens_per_s': round(forward_tokens / (time.time() - started)), 'cuda_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
    out_path = Path(args.out_dir) / 'checkpoint.pt'
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    source_sha = sha256_file_hex_cached(Path(args.checkpoint))
    with torch.no_grad():
        for name, p in live.items():
            p.copy_(masters[name])   # rounded once, nearest, from the FP32 master
    changed = write_checkpoint(model, state, special, out_path, source_sha, total)
    print(json.dumps({'checkpoint': str(out_path), 'changed_elements': changed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('parity', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--turns', required=True)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--base', required=True)
    p = sub.choices['parity']
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--max-tokens', type=int, default=14000)
    p.add_argument('--bf16-params', action='store_true')
    p = sub.choices['train']
    p.add_argument('--out-dir', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--min-lr-fraction', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--batch-loss-tokens', type=int, default=30000)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--max-per-task', type=int, default=0)
    args = parser.parse_args()
    {'parity': parity, 'train': train}[args.command](args)


if __name__ == '__main__':
    main()
