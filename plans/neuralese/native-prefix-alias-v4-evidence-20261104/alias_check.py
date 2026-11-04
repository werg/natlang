"""Physical prefix aliasing vs the production occurrence path on real complete graphs.

A: production BF16 path (share_causal_prefixes=True -> occurrence rows + fixed-linear primal reuse)
B: physical aliasing (pack_causal_prefixes) forced on for BF16
R: FP32 reference of the same model (occurrence path), when --fp32-reference
Reports existing gradient_metrics (atol 1e-5, rtol 1e-4) for B vs A, and relative/cosine of A and B vs R.
"""
import argparse, json, random, sys, time, contextlib
from pathlib import Path
import torch

from natlang_neuralese.model.native_model import load_native_checkpoint
from natlang_neuralese.train.native_state_sft import NativeGraphSource, packed_graph_loss
from natlang_neuralese.train import native_prefix_packing as npp
from natlang_neuralese.model.native_weight_carrier import plain_training_arithmetic
from natlang_neuralese.eval.native_gradients import gradient_metrics

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--graphs', required=True)
p.add_argument('--candidates', type=int, default=40)
p.add_argument('--select-ids', default='')
p.add_argument('--select', type=int, default=3)
p.add_argument('--min-tokens', type=int, default=6000)
p.add_argument('--max-tokens', type=int, default=24000)
p.add_argument('--seed', type=int, default=1104)
p.add_argument('--fp32-reference', action='store_true')
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()

torch.manual_seed(711)
model, _, roles, config, state = load_native_checkpoint(args.checkpoint, device='cuda')
del state
model.native_cotangent_accumulator_backend = 'plain'
model.port.checkpoint_layer_group_size = 2
model.port.ffn_chunk_tokens = 1024
from natlang_neuralese.train.native_state_sft import make_native_optimizer
optimizer, _ = make_native_optimizer(model, lr=1e-6, learned_row_lr=1e-4, interface_row_lr=1e-4)
model.train()
print(json.dumps({'trainable': sum(1 for q in model.parameters() if q.requires_grad)}), flush=True)
source = NativeGraphSource(args.graphs, config['tokenizer_fingerprint'], required_split='train', role_ids=roles.ids)
eligible = [i for i, e in enumerate(source.entries) if args.min_tokens <= e['forward_tokens'] <= args.max_tokens]
rng = random.Random(args.seed)
indices = sorted(rng.sample(eligible, min(args.candidates, len(eligible))))
examples = list(source.iter_indices(indices))
scored = []
for ex in examples:
    shared = npp.pack_causal_prefixes([ex])
    if shared is None or not any(w > 0 for w in ex.target_weights):
        continue
    scored.append((ex.plan.token_count / shared[0].token_count, ex))
scored.sort(key=lambda x: -x[0])
print(json.dumps({'candidates': len(examples), 'shareable': len(scored),
                  'ratios': [round(r, 2) for r, _ in scored[:20]]}), flush=True)
chosen = [ex for _, ex in scored[:args.select]]
if args.select_ids:
    keep=set(args.select_ids.split(','))
    chosen=[ex for _, ex in scored if ex.id in keep] or chosen

original = npp.ordinary_sft_physical_sharing_enabled
v3_rule = lambda m, r: bool(r and m.port.embedding_weight.dtype != torch.bfloat16)
def grads(alias, ex):
    npp.ordinary_sft_physical_sharing_enabled = (lambda m, r: bool(r)) if alias else v3_rule
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.monotonic()
    with plain_training_arithmetic():
        loss, details = packed_graph_loss(model, [ex], checkpoint_layers=True, chunk_tokens=1024,
                                          share_causal_prefixes=True, collect_accuracy=False)
        loss.backward()
    torch.cuda.synchronize()
    out = {n: q.grad.detach().cpu().clone() for n, q in model.named_parameters() if q.grad is not None}
    meta = {'loss': float(loss.detach()), 'seconds': time.monotonic() - t,
            'peak_gb': torch.cuda.max_memory_allocated() / 1e9,
            'physical_tokens': details[0].get('packed_physical_tokens') if details else None}
    npp.ordinary_sft_physical_sharing_enabled = original
    return out, meta

def summary(actual, expected):
    num = den = dot = an = 0.
    worst = []
    passed = 0
    for n, e in expected.items():
        a = actual[n]
        if a.dtype == e.dtype:
            m = gradient_metrics(a, e, atol=1e-5, rtol=1e-4)
            passed += bool(m['passed'])
            worst.append((m['relative_l2'], n))
        a64, e64 = a.double(), e.double()
        num += float(((a64 - e64) ** 2).sum()); den += float((e64 ** 2).sum())
        dot += float((a64 * e64).sum()); an += float((a64 ** 2).sum())
    worst.sort(reverse=True)
    return {'global_relative_l2': (num / den) ** .5, 'global_cosine': dot / (an * den) ** .5,
            'parameters_passed_existing_policy': passed, 'parameters': len(expected),
            'worst_relative_l2': worst[:5]}

report = {'checkpoint': args.checkpoint, 'graphs': []}
for ex in chosen:
    rec = {'id': ex.id, 'forward_tokens': ex.plan.token_count,
           'invocations': len(ex.plan.invocations)}
    a, rec['A'] = grads(False, ex)
    a2, rec['A_repeat'] = grads(False, ex)
    rec['A_repeat_vs_A'] = summary(a2, a); del a2
    b, rec['B'] = grads(True, ex)
    rec['B_vs_A'] = summary(b, a)
    report['graphs'].append(rec)
    print(json.dumps(rec), flush=True)
    args.out.write_text(json.dumps(report, indent=1))
    torch.save({'id': ex.id, 'A': a, 'B': b}, str(args.out) + f'.{len(report["graphs"])}.pt')
    del a, b

if args.fp32_reference:
    model.float()
    for k, ex in enumerate(chosen, 1):
        saved = torch.load(str(args.out) + f'.{k}.pt')
        r, meta = grads(False, ex)
        rec = report['graphs'][k - 1]
        rec['R'] = meta
        rec['A_vs_R'] = summary({n: v.float() for n, v in saved['A'].items()}, r)
        rec['B_vs_R'] = summary({n: v.float() for n, v in saved['B'].items()}, r)
        print(json.dumps({'id': ex.id, 'A_vs_R': rec['A_vs_R'], 'B_vs_R': rec['B_vs_R']}), flush=True)
        args.out.write_text(json.dumps(report, indent=1))
