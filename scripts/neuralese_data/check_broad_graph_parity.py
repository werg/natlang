#!/usr/bin/env python3
"""One-pass graph execution vs explicit cached producer->consumer execution on real broad-graph rows (LFM3).

For each sampled graph: parse it exactly as the native trainer does (``parse_graph``), run the one-pass shared graph
forward and the explicit cached reference (``SharedStateModel(plan, reference=True)``: producers stepped and captured,
state spliced at each import, consumers continued), and compare final hidden states, logits and the weighted target
CE at supervised positions, plus gradients of that CE on the adapter and a few backbone matrices. A small GPU check;
run under the memory ledger at medium priority.

    check_broad_graph_parity.py --graphs FILE.jsonl.zst [FILE ...] --per-file 1 --out REPORT.json
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
from pathlib import Path

import torch

LFM3_FINGERPRINT = '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3'


def rows(path, limit):
    stream = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE).stdout
    out = []
    for line in io.TextIOWrapper(stream, encoding='utf-8'):
        out.append(json.loads(line))
        if len(out) >= limit:
            break
    return out


def relative(a, b):
    return float((a.float() - b.float()).abs().max()) / max(1e-12, float(b.float().abs().max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphs', nargs='+', required=True)
    parser.add_argument('--per-file', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=24000)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from natlang_neuralese.model.lfm3_port import load_lfm3_backbone, Lfm3PortBackbone
    from natlang_neuralese.model.lfm2_port import ControlTokens
    from natlang_neuralese.model.shared_state import SharedStateModel, provision_role_tokens
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.common import stable_softmax
    from types import SimpleNamespace
    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_lfm3_backbone(dtype=dtype, device='cuda')
    tokenizer = copy.deepcopy(tokenizer)
    weight = model.get_input_embeddings().weight
    roles = provision_role_tokens(SimpleNamespace(hf=model, config=model.config, embedding_weight=weight), tokenizer)
    port = Lfm3PortBackbone(model, ControlTokens(roles.produce_open, roles.produce_close), markers=False)
    shared = SharedStateModel(port, roles.ids, rank=8).to('cuda')
    torch.manual_seed(7)
    torch.nn.init.normal_(shared.adapter.up.weight, std=0.03)
    params = dict(model.named_parameters())
    watched_names = ['model.layers.1.conv.in_proj.weight', 'model.layers.2.self_attn.k_proj.weight',
                     'model.layers.23.post_operator_norm.weight']
    watched = [shared.adapter.up.weight, *(params[name] for name in watched_names)]
    for parameter in watched[1:]:
        parameter.requires_grad_(True)
    report = []
    for path in args.graphs:
        for row in rows(path, args.per_file * 4):
            if sum(len(n['token_ids']) for n in row['nodes']) > args.max_tokens:
                continue
            example = parse_graph(row, tokenizer_fingerprint=LFM3_FINGERPRINT)
            plan = example.plan
            ids = [t for node in row['nodes'] for t in node['token_ids']]
            positions, labels, label_weights = [], [], []
            for node, offset in zip(plan.invocations, plan.offsets):
                for local in range(1, len(node.token_ids)):
                    if example.target_weights[offset + local] > 0:
                        positions.append(offset + local - 1); labels.append(int(node.token_ids[local]))
                        label_weights.append(example.target_weights[offset + local])
            positions = torch.tensor(positions, device='cuda')
            labels = torch.tensor(labels, device='cuda')
            label_weights = torch.tensor(label_weights, device='cuda', dtype=torch.float32)

            def loss_of(result):
                logits = shared.logits(result['h_final'][:, positions]).float()[0]
                ce = stable_softmax.cross_entropy(logits, labels, reduction='none')
                return (ce * label_weights).sum() / label_weights.sum(), logits

            flat = shared(plan)
            flat_loss, flat_logits = loss_of(flat)
            flat_grads = torch.autograd.grad(flat_loss, watched)
            reference = shared(plan, reference=True)
            ref_loss, ref_logits = loss_of(reference)
            ref_grads = torch.autograd.grad(ref_loss, watched)
            item = {'graph': row['id'], 'file': str(path), 'family': (row.get('broad_graph') or {}).get('family'),
                    'nodes': len(row['nodes']), 'tokens': len(ids), 'supervised_positions': int(positions.numel()),
                    'imports': sum(len(n['imports']) for n in row['nodes']),
                    'h_final_relative': relative(flat['h_final'], reference['h_final']),
                    'logits_relative': relative(flat_logits, ref_logits),
                    'loss_flat': float(flat_loss), 'loss_reference': float(ref_loss),
                    'argmax_agreement': float((flat_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
                    'gradient_relative': {name: relative(a, b) for name, a, b in
                                          zip(['adapter.up', *watched_names], flat_grads, ref_grads)}}
            print(json.dumps(item), flush=True)
            report.append(item)
            del flat, reference
            torch.cuda.empty_cache()
            if sum(1 for r in report if r['file'] == str(path)) >= args.per_file:
                break
    worst = {key: max(r[key] for r in report) for key in ('h_final_relative', 'logits_relative')} if report else {}
    worst['gradient_relative'] = max((v for r in report for v in r['gradient_relative'].values()), default=None)
    Path(args.out).write_text(json.dumps({'schema': 'natlang.broad-graph-parity/1', 'dtype': args.dtype,
                                          'graphs': report, 'worst': worst}, indent=1) + '\n')
    print(json.dumps({'graphs': len(report), 'worst': worst}))


if __name__ == '__main__':
    main()
