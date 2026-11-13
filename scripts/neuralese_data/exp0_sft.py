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
    exp0_sft.py bench  --turns T.pkl --checkpoint C --base B           # stock vs fast fwd+bwd on fixed chains: loss, grads, tok/s
    exp0_sft.py train  --chains S.pkl [--val-chains V.pkl] ...          # prebuilt sequences (SFT v2: lower_sft_sources.py select)

A prebuilt chain's segments may carry per-token loss weights (`row['weight']`, e.g. the own-output objective's reasoning
share); the step loss is then the weighted mean over its targets. Each `--val-chains` set reports its weighted loss every
`--val-every` steps and at the end (the run's held-out training curve).

Speed (GB10, 11k-token sequences): the stock HF modelling code runs at about 1.9k fwd+bwd tokens/s because half of the GPU
time is unfused elementwise work and a fifth dense sliding-window attention; `lfm3_fast_train` (compiled per-kind layers,
varlen flash attention with the sliding window, shifted-add conv, fused linear NLL) plus partial activation checkpointing
sized to `--activation-gb` runs about 3x faster with the same BF16 arithmetic (`--stock-model` opts out). A step's chains
are packed into rows up to the length the activation budget holds unrecomputed (`--pack-tokens`).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lfm3_fast_train

MAX_LEN = 32000


def chains_of(episodes, max_len=MAX_LEN):
    """Sequences: [{'ids': int32[], 'segs': [(start, length, row)], 'imports': [(start, block_id)], 'task', 'round'}];
    calls of one invocation whose prompt extends the previous call's full text, with the same imported native state at
    the same positions (``row['imports']``), share a sequence."""
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
                imports = [tuple(i) for i in row.get('imports') or ()]
                if current is not None:
                    full = current['ids']
                    if (len(ids) >= len(full) and np.array_equal(ids[:len(full)], full)
                            and [i for i in imports if i[0] < len(full)] == current['imports']):
                        current['ids'] = np.concatenate([ids, sample])
                        current['segs'].append((len(ids), len(sample), row))
                        current['imports'] = imports
                        continue
                    chains.append(current)
                current = {'ids': np.concatenate([ids, sample]), 'segs': [(len(ids), len(sample), row)],
                           'imports': imports, 'task': episode['task_id'], 'round': episode['round']}
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


def hidden(model, ids, imports=None):
    """Final hidden states (T, D) through the stock HF modules. ``imports`` (``lfm3_fast_train.import_tensors``): at those
    rows each attention layer's normed pre-RoPE keys and values and each conv layer's B*x are the imported state (hooks
    on ``k_norm``/``v_proj`` and on ``in_proj``, where B = 1 and x = the state, as the SGLang fork splices it)."""
    if imports is None:
        return model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    if torch.is_grad_enabled() and getattr(model, 'is_gradient_checkpointing', False):
        raise ValueError('stock-path imports are hooks removed after the forward: no layer recompute (use lfm3_fast_train)')
    rows, handles = imports['rows'], []

    def put(value, last):
        def hook(module, args, output):
            output = output.clone()
            output[0, rows] = value(output).to(output.dtype) if callable(value) else value.reshape(len(rows), *last).to(output.dtype)
            return output
        return hook
    try:
        for layer, state in zip(model.model.layers, imports['layers']):
            if layer.is_attention_layer:
                k, v = state
                handles.append(layer.self_attn.k_norm.register_forward_hook(put(k, k.shape[1:])))
                handles.append(layer.self_attn.v_proj.register_forward_hook(put(v, (v[0].numel(),))))
            else:
                (bx,) = state
                width = bx.shape[-1]
                ones = torch.ones_like(bx)
                handles.append(layer.conv.in_proj.register_forward_hook(
                    put(lambda output, bx=bx, ones=ones: torch.cat([ones, output[0, rows, width:2 * width], bx], -1), None)))
        return model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    finally:
        for handle in handles:
            handle.remove()


ACTIVATION_BYTES_PER_TOKEN_LAYER = 88e3   # measured 84.8 KB of saved activations per token per layer with lfm3_fast_train (2026-11-12)


def resident_tokens(model, activation_gb):
    """The longest sequence whose saved activations fit ``activation_gb`` with no layer recomputed (the default packed
    row: packing past it trades per-forward overhead for recompute)."""
    return int(activation_gb * 2**30 // (ACTIVATION_BYTES_PER_TOKEN_LAYER * len(model.model.layers)))


def set_checkpointing(model, tokens, activation_gb):
    """Recompute (gradient-checkpoint) only as many layers as keep the saved activations of a ``tokens`` sequence under
    ``activation_gb``: a checkpointed layer costs one extra forward (a third of its compute), an unchecked one its activations."""
    layers = model.model.layers
    free = int(min(len(layers), activation_gb * 2**30 // (ACTIVATION_BYTES_PER_TOKEN_LAYER * tokens)))
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index >= free
    return len(layers) - free


def _nll(weight, h, target):
    logits = F.linear(h, weight).float()
    return F.cross_entropy(logits, target, reduction='none')


def token_nll(model, h, positions, targets, chunk=2048):
    """NLL (fp32) of ``targets`` predicted from hidden rows ``positions`` (logits only for those rows, chunk-checkpointed;
    a `lfm3_fast_train.prepare`d model uses its fused linear NLL: the same rounding, logits recomputed once, never stored)."""
    if getattr(model, '_lfm3_fast', None) is not None:
        return lfm3_fast_train.linear_nll(h[positions], model.lm_head.weight, targets)
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


def token_nll_nucleus(model, h, positions, targets, top_p):
    """(nll, nll under each row's own top-p nucleus, target outside the nucleus) of ``targets`` at hidden rows
    ``positions`` (`lfm3_fast_train.linear_nll_nucleus`; prepared models only)."""
    if getattr(model, '_lfm3_fast', None) is None:
        raise ValueError('the nucleus NLL needs a lfm3_fast_train.prepare()d model')
    return lfm3_fast_train.linear_nll_nucleus(h[positions], model.lm_head.weight, targets, top_p)


def chain_tensors(chain, device):
    ids = torch.from_numpy(chain['ids'].astype(np.int64)).to(device)
    positions = np.concatenate([np.arange(start - 1, start + length - 1) for start, length, _ in chain['segs']])
    targets = np.concatenate([row['sample'] for _, _, row in chain['segs']])
    return ids, torch.from_numpy(positions).to(device), torch.from_numpy(targets.astype(np.int64)).to(device)


def chain_weights(chain, device):
    """Per-target loss weights (ones unless the segments declare them)."""
    weights = np.concatenate([row['weight'] if 'weight' in row else np.ones(length, np.float32)
                              for _, length, row in chain['segs']])
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def packed_rows(chains, max_tokens):
    """Greedy, order-preserving packing of ``chains`` into rows of at most ``max_tokens`` tokens (a longer chain is a row
    alone; ``max_tokens`` 0: one chain per row). Returns lists of chains."""
    rows, row, size = [], [], 0
    for chain in chains:
        if chain.get('imports'):
            raise ValueError('exp0_sft trains no imported native state (lower without a NativeStates reader)')
        n = len(chain['ids'])
        if row and (not max_tokens or size + n > max_tokens):
            rows.append(row)
            row, size = [], 0
        row.append(chain)
        size += n
    if row:
        rows.append(row)
    return rows


def row_tensors(row, device):
    """(ids, positions, targets, weights, lengths) of chains packed back to back (`lfm3_fast_train.hidden` lengths)."""
    parts = [chain_tensors(chain, device) for chain in row]
    offsets = np.cumsum([0] + [len(chain['ids']) for chain in row[:-1]])
    ids = torch.cat([p[0] for p in parts])
    positions = torch.cat([p[1] + int(o) for p, o in zip(parts, offsets)])
    targets = torch.cat([p[2] for p in parts])
    weights = torch.cat([chain_weights(chain, device) for chain in row])
    return ids, positions, targets, weights, [len(chain['ids']) for chain in row]


def chain_mass(chain):
    return float(sum(row['weight'].sum() if 'weight' in row else length for _, length, row in chain['segs']))


def load_chains(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)['chains']


def validation_loss(model, forward, chains, pack_tokens=0):
    """Weighted mean NLL over prebuilt validation chains (no gradient; packed rows on the fast path)."""
    total, mass = 0., 0.
    with torch.no_grad():
        for row in packed_rows(chains, pack_tokens):
            ids, positions, targets, weights, lengths = row_tensors(row, 'cuda')
            h = forward(model, ids, lengths) if len(row) > 1 else forward(model, ids)
            nll = token_nll(model, h, positions, targets)
            total += (nll * weights).sum().item()
            mass += weights.sum().item()
    return total / max(mass, 1e-9)


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


def write_checkpoint(model, state, special, out_path, source_sha256, step, trained=None):
    """A pruned (inference-only) native checkpoint: the trained backbone in the source's bf16 layout, the source's interface.
    ``trained``: the backbone already on the host ({name: bf16 tensor}, e.g. a pinned snapshot); else read from ``model``."""
    backbone = {}
    original = state['backbone_state']
    for name, value in original.items():
        if name == 'lm_head.weight':
            continue
        backbone[name] = value
    if trained is None:
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
    run_started = time.time()
    episodes = [] if args.chains else pickle.load(open(args.turns, 'rb'))['episodes']
    if args.max_per_task:
        # compute-bound (GB10: about 1k tokens/s of fwd+bwd at 10k context): the shortest verified episodes of each task
        by_task = collections.defaultdict(list)
        for episode in episodes:
            by_task[episode['task_id']].append((sum(len(c['ids']) for c in chains_of([episode])), episode['round'], episode['sample'], episode))
        episodes = [item[-1] for task in sorted(by_task) for item in sorted(by_task[task], key=lambda x: x[:3])[:args.max_per_task]]
    chains = load_chains(args.chains) if args.chains else chains_of(episodes)
    loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
    val_sets = {Path(path).stem: load_chains(path) for path in args.val_chains}
    print(json.dumps({'episodes': len(episodes), 'sequences': len(chains), 'sequence_tokens': int(sum(len(c['ids']) for c in chains)),
                      'loss_tokens': int(loss_tokens), 'epochs': args.epochs}), flush=True)
    model, state, special = load_model(args.checkpoint, args.base)
    state = {k: v for k, v in state.items() if k != 'training'}   # the source's optimizer state (5.5 GB host): never used
    # FP32 master weights live in the optimizer; the model computes in BF16 (as the actor does) and its BF16 gradients
    # accumulate into FP32 buffers
    masters = {name: p.detach().clone().float() for name, p in model.named_parameters()}
    model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    forward = hidden
    if not args.stock_model:
        lfm3_fast_train.prepare(model)
        forward = lfm3_fast_train.hidden
    pack_tokens = 0 if args.stock_model else args.pack_tokens if args.pack_tokens >= 0 else \
        resident_tokens(model, args.activation_gb)
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
    if args.max_steps:
        steps = steps[:args.max_steps]
    print(json.dumps({'optimizer_steps': total, 'lr': args.lr, 'batch_loss_tokens': args.batch_loss_tokens}), flush=True)
    log = open(Path(args.out_dir) / 'train.jsonl', 'a')
    if val_sets:
        model.eval()
        record = {'step': 0, **{f'val_loss:{name}': validation_loss(model, forward, c, pack_tokens) for name, c in val_sets.items()}}
        model.train()
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
    started = time.time()
    forward_tokens = 0
    phase = collections.Counter()
    best = None
    if args.keep_best_val:
        if not val_sets or f'val_loss:{args.keep_best_val}' not in record:
            raise SystemExit(f'--keep-best-val {args.keep_best_val}: no such --val-chains set')
        best = {'loss': record[f'val_loss:{args.keep_best_val}'], 'step': 0,   # step 0: the unchanged source
                'masters': {name: master.detach().cpu().clone() for name, master in masters.items()}}

    def mark(name, since):
        torch.cuda.synchronize()
        now = time.time()
        phase[name] += now - since
        return now

    for step, group in enumerate(steps):
        if args.stop_file and Path(args.stop_file).exists():
            print(json.dumps({'stopped_by': args.stop_file, 'at_step': step}), flush=True)
            break
        fraction = step / max(1, total - 1)
        scale = min(1., (step + 1) / args.warmup) * (args.min_lr_fraction + (1 - args.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * fraction)))
        for g in optimizer.param_groups:
            g['lr'] = args.lr * scale
        denominator = sum(chain_mass(chains[i]) for i in group)
        total_nll, correct = 0., 0
        step_started = time.time()
        for row in packed_rows([chains[i] for i in group], pack_tokens):
            t = time.time()
            ids, positions, targets, weights, lengths = row_tensors(row, 'cuda')
            if not args.stock_model:
                set_checkpointing(model, len(ids), args.activation_gb)
            h = forward(model, ids, lengths) if len(row) > 1 else forward(model, ids)
            t = mark('forward', t)
            nll = token_nll(model, h, positions, targets) * weights
            (nll.sum() / denominator).backward()
            t = mark('loss_backward', t)
            torch._foreach_add_([masters[name].grad for name in live], [p.grad for p in live.values()])
            for p in live.values():
                p.grad = None
            total_nll += nll.sum().item()
            forward_tokens += len(ids)
            del h, nll
            t = mark('accumulate', t)
        t = time.time()
        embedding_master.grad[special_index] = 0
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)       # zeroes the FP32 accumulators in place (foreach)
        with torch.no_grad():
            embedding_master[special_index] = saved_rows
            torch._foreach_copy_(list(live.values()), [masters[name] for name in live])
        mark('optimizer', t)
        record = {'step': step + 1, 'of': total, 'loss': total_nll / denominator, 'lr': args.lr * scale, 'grad_norm': norm,
                  'sequences': len(group), 'loss_tokens': denominator, 'elapsed_s': round(time.time() - started, 1),
                  'forward_tokens_per_s': round(forward_tokens / (time.time() - started)),
                  'step_seconds': round(time.time() - step_started, 1),
                  'phase_seconds': {k: round(v, 1) for k, v in phase.items()},
                  'cuda_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        if val_sets and ((step + 1) % args.val_every == 0 or step + 1 == len(steps)):
            model.eval()
            record.update({f'val_loss:{name}': validation_loss(model, forward, c, pack_tokens) for name, c in val_sets.items()})
            model.train()
            if args.keep_best_val:
                # the checkpoint written is the FP32 master state at the lowest validation NLL of the named set
                current = record[f'val_loss:{args.keep_best_val}']
                if best is None or current < best['loss']:
                    best = {'loss': current, 'step': step + 1,
                            'masters': {name: master.detach().cpu().clone() for name, master in masters.items()}}
                    if args.save_best:
                        # the live BF16 parameters are the rounded masters of this very step: write them now, so the
                        # best checkpoint exists on disk whatever happens to the run (preemption, an early stop)
                        from natlang_neuralese.common.hashing import sha256_file_hex_cached
                        write_checkpoint(model, state, special, Path(args.out_dir) / 'checkpoint.pt',
                                         sha256_file_hex_cached(Path(args.checkpoint)), total)
                        record['saved_best'] = {'step': step + 1, 'val_loss': current}
                record['best_val'] = {'loss': best['loss'], 'step': best['step']}
        if args.hours and step + 1 == args.replan_step and step + 1 < len(steps):
            # fit the schedule to the wall-clock budget from the measured rate (compile included: conservative)
            rate = forward_tokens / (time.time() - started)
            left = args.hours * 3600 - (time.time() - run_started) - args.reserve_minutes * 60
            mean = np.mean([sum(len(chains[i]['ids']) for i in g) for g in steps[step + 1:]])
            fit = step + 1 + max(0, int(left * rate / mean))
            if fit < len(steps):
                del steps[fit:]
                total = fit
            record['replanned_steps'] = total
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
    if args.max_steps:
        return
    out_path = Path(args.out_dir) / 'checkpoint.pt'
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    source_sha = sha256_file_hex_cached(Path(args.checkpoint))
    if best is not None:
        masters = best['masters']
        print(json.dumps({'kept_best_val': {'loss': best['loss'], 'step': best['step']}}), flush=True)
    with torch.no_grad():
        for name, p in live.items():
            p.copy_(masters[name])   # rounded once, nearest, from the FP32 master
    changed = write_checkpoint(model, state, special, out_path, source_sha, total)
    print(json.dumps({'checkpoint': str(out_path), 'changed_elements': changed}), flush=True)


def bench(args):
    """Stock vs fast forward+backward (BF16, the training dtype) on the same fixed chains: per-chain loss, global gradient
    agreement and tokens/s. The stock pass runs first; the model is then patched in place."""
    episodes = pickle.load(open(args.turns, 'rb'))['episodes']
    chains = [c for c in chains_of(episodes) if args.min_tokens <= len(c['ids']) <= args.max_tokens][:args.chains]
    model, _, _ = load_model(args.checkpoint, args.base)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters()]

    def run(forward, label, before=None):
        grads = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        losses, seconds, tokens = [], 0., 0
        for repeat in range(args.repeats + 1):          # the first pass compiles and warms up
            if repeat == 1:
                seconds, tokens = 0., 0
            for chain in chains:
                ids, positions, targets = chain_tensors(chain, 'cuda')
                if before is not None:
                    before(len(ids))
                torch.cuda.synchronize()
                started = time.time()
                nll = token_nll(model, forward(model, ids), positions, targets)
                (nll.sum() / len(targets)).backward()
                torch.cuda.synchronize()
                seconds += time.time() - started
                tokens += len(ids)
                if repeat == 0:
                    losses.append(nll.mean().item())
                    torch._foreach_add_(grads, [p.grad for p in params])
                for p in params:
                    p.grad = None
        print(json.dumps({'impl': label, 'chain_tokens': [len(c['ids']) for c in chains], 'loss': [round(x, 5) for x in losses],
                          'fwd_bwd_tokens_per_s': round(tokens / seconds), 'cuda_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
        return losses, grads

    reference = None
    if args.fp32_reference:          # the BF16 noise floor: the same chains with FP32 compute (no autocast, no TF32)
        args.repeats, repeats = 0, args.repeats
        _, reference = run(hidden, 'fp32-reference')
        args.repeats = repeats
    model.to(torch.bfloat16)
    stock_losses, stock_grads = run(hidden, 'stock')
    lfm3_fast_train.prepare(model)
    fast_losses, fast_grads = run(lfm3_fast_train.hidden, 'fast', before=lambda n: set_checkpointing(model, n, args.activation_gb))
    def agreement(x, y):
        dot = sum((a * b).sum().item() for a, b in zip(x, y))
        na = math.sqrt(sum((a * a).sum().item() for a in x))
        nb = math.sqrt(sum((b * b).sum().item() for b in y))
        diff = math.sqrt(sum(((a - b) ** 2).sum().item() for a, b in zip(x, y)))
        return {'cosine': dot / (na * nb), 'rel_l2': diff / na, 'norm_a': na, 'norm_b': nb}

    if reference is not None:
        print(json.dumps({'stock_vs_fp32': agreement(reference, stock_grads), 'fast_vs_fp32': agreement(reference, fast_grads)}), flush=True)
    print(json.dumps({'stock_vs_fast': agreement(stock_grads, fast_grads)}), flush=True)
    print(json.dumps({'loss_abs_diff_max': max(abs(a - b) for a, b in zip(stock_losses, fast_losses)),
                      'loss_mean_stock': float(np.mean(stock_losses)), 'loss_mean_fast': float(np.mean(fast_losses)),
                      }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('parity', 'train', 'bench'):
        p = sub.add_parser(name)
        p.add_argument('--turns', required=name != 'train')
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--base', required=True)
    p = sub.choices['parity']
    p.add_argument('--samples', type=int, default=6)
    p.add_argument('--max-tokens', type=int, default=14000)
    p.add_argument('--bf16-params', action='store_true')
    p = sub.choices['bench']
    p.add_argument('--chains', type=int, default=6)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--min-tokens', type=int, default=9000)
    p.add_argument('--max-tokens', type=int, default=12500)
    p.add_argument('--activation-gb', type=float, default=20.)
    p.add_argument('--fp32-reference', action='store_true')
    p = sub.choices['train']
    p.add_argument('--stock-model', action='store_true', help='the unpatched HF layers with full checkpointing (about 3x slower)')
    p.add_argument('--activation-gb', type=float, default=20., help='saved-activation budget per sequence; layers beyond it are recomputed')
    p.add_argument('--pack-tokens', type=int, default=-1, help='pack a step\'s chains back to back into rows of at most this many tokens (documents stay separate); default: the longest row --activation-gb holds without recompute (about 10k at 20 GB); 0: one chain per forward')
    p.add_argument('--max-steps', type=int, default=0, help='stop after this many optimizer steps without writing a checkpoint')
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
    p.add_argument('--chains', help='prebuilt sequences pickle ({"chains": [...]}) instead of --turns')
    p.add_argument('--val-chains', action='append', default=[],
                   help='prebuilt validation sequences (repeatable, named by file stem): weighted NLL every --val-every steps')
    p.add_argument('--val-every', type=int, default=50)
    p.add_argument('--save-best', action='store_true', help='with --keep-best-val: write checkpoint.pt at every new best validation step')
    p.add_argument('--stop-file', help='stop before the next optimizer step when this file exists (the best-validation checkpoint is written)')
    p.add_argument('--keep-best-val', help='name (file stem) of a --val-chains set: write the checkpoint of the step with its lowest NLL (step 0 counts)')
    p.add_argument('--hours', type=float, default=0., help='wall-clock budget: at --replan-step the schedule (cosine '
                   'included) is cut to the steps that fit at the measured rate, keeping --reserve-minutes for eval and save')
    p.add_argument('--replan-step', type=int, default=20)
    p.add_argument('--reserve-minutes', type=float, default=25.)
    args = parser.parse_args()
    from natlang_neuralese.common.cuda_context import ensure_context
    ensure_context()   # GB10: a context can fail to create under host-memory fragmentation; retried
    {'parity': parity, 'train': train, 'bench': bench}[args.command](args)


if __name__ == '__main__':
    main()
