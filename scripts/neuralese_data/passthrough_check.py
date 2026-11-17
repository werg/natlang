#!/usr/bin/env python3
"""Checks of the Neuralese pass-through (plans/neuralese/NEURALESE_PASSTHROUGH_20261117.md) on real rollouts.

  capture  The learner's state for real producer documents against the blocks the SGLang actor captured for them
           (version-0 episodes: every call served by ``--served-sha``, the start export). Per block: the stock HF
           forward's per-layer normed pre-RoPE K/V and conv B*x at the span rows, BF16 and FP32 (the BF16 floor), against
           the stored block. Per consumer call that imports such a block: the sampled tokens' log-probabilities with the
           in-graph state (pass-through) and with the stored block, against the actor's recorded ones.
  bench    Forward+backward tokens/s and CUDA peak on importing chains: pass-through, the same state detached, and no
           producer documents at all (``nostate``: the stored-state path's compute).

Read-only on the rollouts and the store. Run under the memory ledger.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / 'training/neuralese'), str(HERE.parents[1]), str(HERE)]

START_EXPORT = '5ca8a8b1bcabb45ea343a6aae3816804fb4a6639985cb491e72cba4b0312d293'   # grpo-v3 start (5ff07638) export


def version0_episodes(archive, served, limit_groups, first_group=0):
    """Episodes of the archive's first groups whose every sampled call was served by ``served``."""
    import zstandard
    for path in sorted(Path(archive).glob('g*.rollouts.jsonl.zst'))[first_group:first_group + limit_groups]:
        text = zstandard.ZstdDecompressor().stream_reader(path.open('rb')).read().decode()
        for line in text.splitlines():
            if 'block_write' not in line:
                continue
            row = json.loads(line)
            shas = {((t.get('sampling') or {}).get('actor_export') or {}).get('sha256') for t in row['turns']}
            if shas == {served}:
                yield row


def layer_states(model, ids, embeds=None):
    """Per layer {'raw_k', 'v'} or {'gated_inputs'} of a stock HF forward (FP32 copies), every row (``embeds``: the
    input embeddings instead of the model's own)."""
    import torch
    states, handles = [None] * len(model.model.layers), []
    for index, layer in enumerate(model.model.layers):
        states[index] = {}
        if layer.is_attention_layer:
            attn = layer.self_attn
            handles.append(attn.k_norm.register_forward_hook(
                lambda m, a, out, s=states[index]: s.__setitem__('raw_k', out[0].float().clone())))
            handles.append(attn.v_proj.register_forward_hook(
                lambda m, a, out, s=states[index], d=attn.head_dim: s.__setitem__(
                    'v', out[0].float().reshape(out.shape[1], -1, d).clone())))
        else:
            def bx(m, a, out, s=states[index]):
                b, _, x = out[0].chunk(3, dim=-1)
                s['gated_inputs'] = (b * x).float().clone()
            handles.append(layer.conv.in_proj.register_forward_hook(bx))
    with torch.no_grad():
        if embeds is None:
            model.model(input_ids=ids[None], use_cache=False)
        else:
            model.model(inputs_embeds=embeds[None], use_cache=False)
    for handle in handles:
        handle.remove()
    return states


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def capture(args):
    import torch
    import exp0_sft
    import lfm3_fast_train
    from lower_rollout_turns import NativeStates, episode_turns, write_sources
    from natlang_neuralese.serve.store import read_native_state_file
    store = NativeStates(args.store)
    blocks, consumers = {}, []
    for row in version0_episodes(args.rollouts, args.served_sha, args.groups, args.first_group):
        sources = write_sources(row)
        present = {b: s for b, s in sources.items() if store.exists(b)}
        if not present:
            continue
        for b, s in present.items():
            blocks.setdefault(b, s)
        stats = collections.Counter()
        for call in episode_turns(row, stats):
            if call['imports'] and all(k in present for _, k in call['imports']) \
                    and len(call['prompt']) + len(call['sample']) <= args.max_call_tokens:
                consumers.append(call)
        if len(blocks) >= args.blocks and len(consumers) >= args.consumers:
            break
    consumers = consumers[:args.consumers]
    needed = {k for c in consumers for _, k in c['imports']}
    blocks = {k: s for k, s in blocks.items() if k in needed or len(needed) < args.blocks and not needed.add(k)}
    print(json.dumps({'blocks': len(blocks), 'consumers': len(consumers),
                      'document_tokens': [len(s['ids']) for s in blocks.values()]}), flush=True)
    model, _, _ = exp0_sft.load_model(args.checkpoint, args.base)
    model.eval()
    fp32 = {b: layer_states(model, torch.as_tensor(s['ids'], dtype=torch.long, device='cuda')) for b, s in blocks.items()}
    model.to(torch.bfloat16)
    table = None
    if args.producer_table:       # the actor's producer table (its export): the span rows' input embeddings
        from safetensors import safe_open
        with safe_open(args.producer_table, 'pt', device='cuda') as handle:
            table = handle.get_tensor('model.natlang_producer_embed')
    report = []
    for b, s in blocks.items():
        stored = read_native_state_file(store.path(b), b).load()
        assert np.array_equal(np.asarray(stored.token_ids), s['ids'][s['start']:s['stop']]), b
        ids = torch.as_tensor(s['ids'], dtype=torch.long, device='cuda')
        span = slice(s['start'], s['stop'])
        variants = {'learner': layer_states(model, ids)}
        if table is not None:
            embeds = model.model.embed_tokens(ids).clone()
            embeds[span] = table[ids[span]]
            variants['actor_table'] = layer_states(model, ids, embeds)
        entry = {'block': b, 'tokens': s['stop'] - s['start']}
        for label, states in variants.items():
            per_layer = []
            for index, (state, ref, block_layer) in enumerate(zip(states, fp32[b], stored.layers)):
                for field, value in state.items():
                    actor = block_layer[field].to('cuda').float()
                    per_layer.append({'layer': index, 'field': field, 'actor_vs_bf16': rel(actor, value[span]),
                                      'bf16_vs_fp32': rel(value[span], ref[field][span]),
                                      'actor_vs_fp32': rel(actor, ref[field][span])})
            entry[label] = {'actor_vs_bf16_max': max(r['actor_vs_bf16'] for r in per_layer),
                            'actor_vs_bf16_median': float(np.median([r['actor_vs_bf16'] for r in per_layer])),
                            'bf16_vs_fp32_max': max(r['bf16_vs_fp32'] for r in per_layer),
                            'actor_vs_fp32_max': max(r['actor_vs_fp32'] for r in per_layer),
                            'median_by_field': {f: {k: round(float(np.median([r[k] for r in per_layer if r['field'] == f])), 4)
                                                    for k in ('actor_vs_bf16', 'bf16_vs_fp32', 'actor_vs_fp32')}
                                                for f in sorted({r['field'] for r in per_layer})},
                            'by_layer': [(r['layer'], r['field'], round(r['actor_vs_bf16'], 4), round(r['bf16_vs_fp32'], 4),
                                          round(r['actor_vs_fp32'], 4)) for r in per_layer] if label == 'learner' else None,
                            'worst': max(per_layer, key=lambda r: r['actor_vs_bf16'])}
        report.append(entry)
        print(json.dumps(entry), flush=True)
    lfm3_fast_train.prepare(model)
    calls = []
    for call in consumers:
        chain = exp0_sft.chains_of([{'rows': [call], 'task_id': 't', 'round': 0}])[0]
        actor = torch.as_tensor(call['logp'], device='cuda').float()
        out = {}
        for mode in ('pass', 'stored'):
            if mode == 'pass':
                ids, positions, targets, _, lengths, imports = exp0_sft.row_inputs([chain], 'cuda')
            else:
                ids, positions, targets = exp0_sft.chain_tensors(chain, 'cuda')
                lengths = None
                imports = lfm3_fast_train.import_tensors(model, [(r, read_native_state_file(store.path(k), k).load())
                                                                 for r, k in chain['imports']])
            out[mode] = lfm3_fast_train.token_logprobs(model, ids, positions, targets, lengths=lengths, imports=imports)
        none_ids, none_pos, none_t = exp0_sft.chain_tensors(chain, 'cuda')
        out['no_state'] = lfm3_fast_train.token_logprobs(model, none_ids, none_pos, none_t)   # placeholders as tokens
        calls.append({'tokens': len(actor), 'prompt': len(call['prompt']), 'imports': len(call['imports']),
                      **{f'mean_abs_dlogp_{a}_vs_{b}': round(float((out[a] - ref).abs().mean()), 5)
                         for a, b, ref in (('pass', 'stored', out['stored']), ('pass', 'actor', actor),
                                           ('stored', 'actor', actor), ('no_state', 'actor', actor))}})
        print(json.dumps(calls[-1]), flush=True)
    summary = {'blocks': report, 'calls': calls}
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1) + '\n')


def bench(args):
    import torch
    import exp0_sft
    import lfm3_fast_train
    import zstandard
    from lower_rollout_turns import episode_turns
    chains = []
    for path in sorted(Path(args.rollouts).glob('g*.rollouts.jsonl.zst')):
        for line in zstandard.ZstdDecompressor().stream_reader(path.open('rb')).read().decode().splitlines():
            if 'block_write' not in line:
                continue
            row = json.loads(line)
            stats = collections.Counter()
            calls = [c for c in episode_turns(row, stats) if c['imports']]
            for chain in exp0_sft.chains_of([{'rows': calls, 'task_id': row['task_id'], 'round': 0}]):
                if chain['imports'] and args.min_tokens <= len(chain['ids']) <= args.max_tokens and \
                        all(k in chain['sources'] for _, k in chain['imports']):
                    chains.append(chain)
        if len(chains) >= args.chains:
            break
    chains = chains[:args.chains]
    model, _, _ = exp0_sft.load_model(args.checkpoint, args.base, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    lfm3_fast_train.prepare(model)
    params = list(model.parameters())
    print(json.dumps({'chains': len(chains), 'chain_tokens': [len(c['ids']) for c in chains],
                      'producer_document_tokens': [exp0_sft.source_tokens(c) for c in chains],
                      'imports': [len(c['imports']) for c in chains]}), flush=True)
    results = {}
    for mode in args.modes.split(','):
        torch.cuda.reset_peak_memory_stats()
        seconds, tokens, losses = 0., 0, []
        for repeat in range(args.repeats + 1):     # the first pass compiles
            for chain in chains:
                torch.cuda.synchronize()
                started = time.time()
                if mode == 'nostate':     # compute baseline: no producer documents (the stored-state path's cost)
                    ids, positions, targets = exp0_sft.chain_tensors(chain, 'cuda')
                    lengths = imports = None
                else:
                    ids, positions, targets, _, lengths, imports = exp0_sft.row_inputs([chain], 'cuda', mode == 'pass')
                exp0_sft.set_checkpointing(model, len(ids), args.activation_gb)
                h = lfm3_fast_train.hidden(model, ids, lengths, imports)
                nll = exp0_sft.token_nll(model, h, positions, targets)
                (nll.sum() / len(targets)).backward()
                torch.cuda.synchronize()
                if repeat:
                    seconds += time.time() - started
                    tokens += len(ids)
                else:
                    losses.append(round(nll.mean().item(), 5))
                for p in params:
                    p.grad = None
                del h, nll
        results[mode] = {'fwd_bwd_tokens_per_s': round(tokens / seconds), 'seconds': round(seconds, 2),
                         'cuda_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2), 'loss': losses}
        print(json.dumps({'mode': mode, **results[mode]}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1) + '\n')


def main():
    from grpo import BASE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('capture', 'bench'):
        p = sub.add_parser(name)
        p.add_argument('--rollouts', required=True, help='a GRPO rollout archive (g*.rollouts.jsonl.zst)')
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--base', default=BASE)
        p.add_argument('--out')
    p = sub.choices['capture']
    p.add_argument('--store', required=True, help="the run's actor store (blocks/)")
    p.add_argument('--served-sha', default=START_EXPORT)
    p.add_argument('--producer-table', help="the actor export's model.safetensors: also compare with the span rows "
                   "embedded by the actor's producer table (grpo-v3 kept it at the start export's values)")
    p.add_argument('--groups', type=int, default=200)
    p.add_argument('--first-group', type=int, default=0)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--consumers', type=int, default=6)
    p.add_argument('--max-call-tokens', type=int, default=12000)
    p = sub.choices['bench']
    p.add_argument('--chains', type=int, default=6)
    p.add_argument('--min-tokens', type=int, default=3000)
    p.add_argument('--max-tokens', type=int, default=9000)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--activation-gb', type=float, default=4.)
    p.add_argument('--modes', default='pass,frozen')
    args = parser.parse_args()
    from natlang_neuralese.common.cuda_context import ensure_context
    ensure_context()
    {'capture': capture, 'bench': bench}[args.command](args)


if __name__ == '__main__':
    main()
