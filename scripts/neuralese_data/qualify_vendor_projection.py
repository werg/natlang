#!/usr/bin/env python3
"""CUDA correctness gate for vendor projection arithmetic in plain SFT updates (owner, 2026-11-04).

A diagnostic, never a trainer: no optimizer step, no checkpoint write, RNG restored per pass.
On complete real train graphs from a real earned checkpoint, under the production execution profile
(plain cotangents, ffn-stable-v1, aliased-prefix packing, layer checkpointing, FFN/readout chunking):

1. Per projection, exactness against the exact path: every vendor forward, dX and dW product (and, as the
   control, every invariant forward and dX) on sampled rows is compared with FP64 products rounded once to
   the output dtype, through the production comparison policy (eval/native_gradients.gradient_metrics) at the
   declared production tolerances (forward 1e-4, gradient 1e-5, rtol 1e-4). Nothing is relaxed.
2. Coverage: in a vendor pass no fixed-tile Triton projection may run (counted at _dense_mm).
3. Whole graph, invariant versus vendor: CE and complete trainable-parameter gradients. A deliberate rounding
   change is bounded by the declared cross-mode transition bound (model/native_precision.transition_gradient_rtol:
   relative L2 <= 0.1, cosine >= 0.9); both modes are also reported against each other per parameter.
4. Optionally the trainer's own runtime certificate (eval/native_state.qualify_existing, complete production
   trainables) in this vendor-configured process.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
os.environ.setdefault('NATLANG_PROJECTION_ARITHMETIC', 'vendor')

from natlang_neuralese.common.hashing import sha256_file_hex_cached  # noqa: E402
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable  # noqa: E402
from natlang_neuralese.data.native_graph import parse_graph  # noqa: E402
from natlang_neuralese.eval.native_gradients import gradient_metrics  # noqa: E402
from natlang_neuralese.maple import fused_moe  # noqa: E402
from natlang_neuralese.model import native_weight_carrier as carrier  # noqa: E402
from natlang_neuralese.model.native_model import load_native_checkpoint  # noqa: E402
from natlang_neuralese.train.loop import capture_training_rng_state, restore_training_rng_state  # noqa: E402
from natlang_neuralese.train import native_state_sft as sft  # noqa: E402

FORWARD_ATOL, GRADIENT_ATOL, RTOL = 1e-4, 1e-5, 1e-4
CROSS_MODE_RTOL = 0.1


def select_graphs(paths, *, per_file, max_tokens, scan, fingerprint):
    chosen = []
    for path in paths:
        taken = 0
        with open_jsonl_text(path) as stream:
            for index, line in enumerate(stream):
                if index >= scan or taken >= per_file:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('split') != 'train':
                    continue
                if row['tokenizer_fingerprint'] != fingerprint:
                    raise ValueError('qualification graph tokenizer differs from the checkpoint')
                example = parse_graph(row, tokenizer_fingerprint=row['tokenizer_fingerprint'])
                if example.plan.token_count > max_tokens or not any(w > 0 for w in example.target_weights):
                    continue
                chosen.append((str(path), index, example))
                taken += 1
    return chosen


class ProjectionAudit:
    """Sampled-row exactness of every projection product against FP64 products, plus kernel coverage."""

    def __init__(self, samples, seed=1103):
        self.samples = samples
        self.generator = torch.Generator().manual_seed(seed)
        self.rows = {}
        self.dense_calls = 0
        self.vendor_calls = 0

    def _pick(self, count):
        if count <= self.samples:
            return torch.arange(count)
        return torch.randperm(count, generator=self.generator)[:self.samples]

    def record(self, kind, actual, reference, atol, operands):
        # Rigorous criterion: zero elements outside the worst-case FP32-accumulation bound (fused_moe).
        details = []
        beyond_count = fused_moe.accumulation_bound_violations(actual.detach(), *operands, details=details)
        actual, reference = actual.detach().cpu(), reference.detach().cpu()
        metrics = gradient_metrics(actual, reference, atol=atol, rtol=RTOL)  # reported, as for the control kernel
        entry = self.rows.setdefault(kind, {'calls': 0, 'passed': 0, 'strict_passed': 0, 'elements': 0,
                                            'adjacent_bf16_mismatches': 0, 'worst_relative_l2': 0.,
                                            'worst_cosine': 1., 'worst_max_abs': 0., 'failures': []})
        entry['calls'] += 1
        if details and len(entry.setdefault('bound_violation_details', [])) < 16:
            entry['bound_violation_details'].extend(details)
        entry['bound_violations'] = entry.get('bound_violations', 0) + beyond_count
        entry['within_bound_calls'] = entry.get('within_bound_calls', 0) + (beyond_count == 0)
        entry['passed'] += bool(metrics.get('passed'))
        entry['strict_passed'] += bool(metrics.get('strict_passed'))
        entry['elements'] += int(metrics.get('elements', 0))
        entry['adjacent_bf16_mismatches'] += int(metrics.get('adjacent_bf16_mismatches', 0))
        entry['worst_relative_l2'] = max(entry['worst_relative_l2'], float(metrics.get('relative_l2', float('inf'))))
        entry['worst_cosine'] = min(entry['worst_cosine'], float(metrics.get('cosine', -1.)))
        entry['worst_max_abs'] = max(entry['worst_max_abs'], float(metrics.get('max_abs', float('inf'))))
        if not metrics.get('passed') and len(entry['failures']) < 8:
            entry['failures'].append({k: metrics.get(k) for k in ('reason', 'relative_l2', 'cosine', 'max_abs',
                                                                   'mismatched_elements', 'adjacent_bf16_mismatches',
                                                                   'shape', 'dtype')})

    def forward(self, kind, x, w, out):
        if x.dtype != torch.bfloat16:
            return  # the FP32 referee pass is not a BF16 projection under test
        rows = self._pick(x.shape[0]).to(x.device)
        xs = x.index_select(0, rows)
        reference = (xs.double() @ w.double().t()).to(out.dtype)
        self.record(kind, out.index_select(0, rows), reference, FORWARD_ATOL, (xs, w.t()))

    def input_gradient(self, kind, grad, w, dx):
        if grad.dtype != torch.bfloat16:
            return
        rows = self._pick(grad.shape[0]).to(grad.device)
        gs = grad.index_select(0, rows)
        reference = (gs.double() @ w.double()).to(dx.dtype)
        self.record(kind, dx.index_select(0, rows), reference, GRADIENT_ATOL, (gs, w))

    def weight_gradient(self, kind, grad, x, dw):
        if grad.dtype != torch.bfloat16:
            return
        outputs = self._pick(grad.shape[1]).to(grad.device)
        gs = grad.index_select(1, outputs).t()
        reference = (gs.double() @ x.double()).to(dw.dtype)
        self.record(kind, dw.index_select(0, outputs), reference, GRADIENT_ATOL, (gs, x))

    def install(self):
        audit = self
        vendor_forward, vendor_adjoints = fused_moe._VendorLinear.forward, fused_moe._vendor_adjoints
        dense_mm, invariant_forward = fused_moe._dense_mm, fused_moe._invariant_linear_forward
        invariant_input = fused_moe.invariant_input_gradient

        def forward(ctx, x, w):
            out = vendor_forward(ctx, x, w)
            audit.vendor_calls += 1
            with torch.no_grad():
                audit.forward('vendor.forward', x, w, out)
            return out

        def adjoints(x, w, grad, input_gradient, weight_gradient):
            dx, dw = vendor_adjoints(x, w, grad, input_gradient, weight_gradient)
            with torch.no_grad():
                if dx is not None:
                    audit.input_gradient('vendor.dX', grad, w, dx)
                if dw is not None:
                    audit.weight_gradient('vendor.dW', grad, x, dw)
            return dx, dw

        def counted_dense(*args, **kwargs):
            audit.dense_calls += 1
            return dense_mm(*args, **kwargs)

        def checked_forward(x, w, precision):
            out = invariant_forward(x, w, precision)
            if precision == 'ieee' and x.dtype == w.dtype:
                with torch.no_grad():
                    audit.forward('invariant.forward', x, w, out)
            return out

        def checked_input(gradient, weight, *, precision='ieee'):
            dx = invariant_input(gradient, weight, precision=precision)
            if precision == 'ieee' and gradient.dtype == weight.dtype:
                with torch.no_grad():
                    flat = gradient.reshape(-1, gradient.shape[-1])
                    audit.input_gradient('invariant.dX', flat, weight, dx.reshape(-1, dx.shape[-1]))
            return dx

        fused_moe._VendorLinear.forward = staticmethod(forward)
        fused_moe._vendor_adjoints = adjoints
        fused_moe._dense_mm = counted_dense
        fused_moe._invariant_linear_forward = checked_forward
        fused_moe.invariant_input_gradient = checked_input


def gradient_snapshot(named):
    return {name: p.grad.detach().to('cpu', torch.bfloat16, copy=True) if p.grad is not None else None
            for name, p in named}


def compare_gradients(reference, actual):
    dot = ref_sq = act_sq = diff_sq = 0.
    per = []
    for name, value in actual.items():
        base = reference.get(name)
        if base is None or value is None:
            if (base is None) != (value is None):
                per.append({'name': name, 'missing': True})
            continue
        a, b = value.double(), base.double()
        d, r, s = float((a * b).sum()), float((b * b).sum()), float((a * a).sum())
        e = float(((a - b) ** 2).sum())
        dot += d; ref_sq += r; act_sq += s; diff_sq += e
        per.append({'name': name, 'relative_l2': (e ** .5) / max(r ** .5, 1e-30),
                    'cosine': d / max((r * s) ** .5, 1e-30) if r > 0 and s > 0 else float(r == s)})
    return {'relative_l2': (diff_sq ** .5) / max(ref_sq ** .5, 1e-30),
            'cosine': dot / max((ref_sq * act_sq) ** .5, 1e-30),
            'parameters': len(per), 'missing': [row['name'] for row in per if row.get('missing')],
            'worst_parameters': sorted((row for row in per if not row.get('missing')),
                                       key=lambda row: -row['relative_l2'])[:8]}


def fp32_reference(model, batch, named, chunk_tokens, rng, snapshots):
    """The exact-gradient referee: the same weights cast exactly to FP32, IEEE arithmetic, the same packed batch.

    BF16 -> FP32 -> BF16 is an exact round trip, so the BF16 model is restored bit for bit afterwards."""
    before = {name: p.detach().clone() for name, p in list(model.named_parameters())[:4]}
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model.to(torch.float32)
    try:
        loss, _ = run_pass(model, batch, named, 'invariant', chunk_tokens, rng)
        exact = {name: p.grad.detach().to('cpu', torch.float32, copy=True) if p.grad is not None else None
                 for name, p in named}
    finally:
        for _, parameter in named:
            parameter.grad = None
        model.to(torch.bfloat16)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
        torch.set_float32_matmul_precision(flags[2])
    if any(not torch.equal(p.detach(), before[name]) for name, p in list(model.named_parameters())[:4]):
        raise RuntimeError('FP32 round trip changed BF16 weights')
    result = {'ce': loss}
    for mode, snapshot in snapshots.items():
        result[mode] = compare_gradients(exact, snapshot)
        result[mode]['worst_parameters'] = result[mode]['worst_parameters'][:3]
    return result


def run_pass(model, batch, named, mode, chunk_tokens, rng):
    carrier.set_training_projection_arithmetic(mode)
    for _, parameter in named:
        parameter.grad = None
    restore_training_rng_state(rng, 'cuda')
    torch.cuda.synchronize()
    started = time.monotonic()
    with carrier.plain_training_arithmetic():
        loss, _ = sft.packed_graph_loss(model, batch, checkpoint_layers=True, chunk_tokens=chunk_tokens,
                                        share_causal_prefixes=True, collect_accuracy=False)
        loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach()), time.monotonic() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--graphs', type=Path, nargs='+', required=True)
    parser.add_argument('--per-file', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=12000)
    parser.add_argument('--scan', type=int, default=64)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--chunk-tokens', type=int, default=1024)
    parser.add_argument('--certificate', action='store_true')
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--fp32-reference', action='store_true')
    parser.add_argument('--packed-only', action='store_true')
    parser.add_argument('--no-audit', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ['NATLANG_PROJECTION_ARITHMETIC'] != 'vendor':
        raise ValueError('the vendor gate runs in a vendor-configured process')
    started = time.monotonic()
    model, tokenizer, roles, config, state = load_native_checkpoint(args.checkpoint, device='cuda', base=args.base)
    training = state.get('training') or state.get('shared_training') or {}
    del state
    group = sft.configure_checkpoint_layer_group_size(model.port, None, training)
    ffn_chunk = sft.configure_ffn_chunk_tokens(model.port, None, training)
    model.share_causal_prefixes = True
    model.native_cotangent_accumulator_backend = 'plain'
    named = sft.native_trainable_parameters(model)
    model.train(); model.port.checkpoint_layers = True
    examples = select_graphs(args.graphs, per_file=args.per_file, max_tokens=args.max_tokens, scan=args.scan,
                             fingerprint=config['tokenizer_fingerprint'])
    if not examples:
        raise ValueError('no complete train graphs selected')
    batches = [] if args.packed_only else [[example] for _, _, example in examples]
    if len(examples) > 1:
        packed = [example for _, _, example in examples]
        batches.append(packed[:min(4, len(packed))])
    audit = ProjectionAudit(args.samples)
    if not args.no_audit:
        audit.install()
    report = {'schema': 'natlang.vendor-projection-gate/1', 'checkpoint': {
                  'path': str(args.checkpoint), 'sha256': sha256_file_hex_cached(args.checkpoint)},
              'precision': config.get('precision'), 'native_shared_execution': config.get('native_shared_execution'),
              'execution': {'cotangent': 'plain', 'checkpoint_layer_group_size': group, 'ffn_chunk_tokens': ffn_chunk,
                            'chunk_tokens': args.chunk_tokens, 'share_causal_prefixes': True},
              'hardware': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'tolerances': {'forward_atol': FORWARD_ATOL, 'gradient_atol': GRADIENT_ATOL, 'rtol': RTOL,
                             'cross_mode_rtol': CROSS_MODE_RTOL},
              'graphs': [{'path': path, 'line': line, 'id': example.id, 'tokens': example.plan.token_count}
                         for path, line, example in examples],
              'trainable_parameters': len(named), 'optimizer_updates': 0, 'batches': []}
    rng = capture_training_rng_state('cuda')
    for batch in batches:
        row = {'graphs': [example.id for example in batch], 'tokens': sum(e.plan.token_count for e in batch)}
        audit.dense_calls = audit.vendor_calls = 0
        loss_invariant, seconds_invariant = run_pass(model, batch, named, 'invariant', args.chunk_tokens, rng)
        row['invariant'] = {'ce': loss_invariant, 'seconds_with_audit': seconds_invariant,
                            'triton_projection_calls': audit.dense_calls, 'vendor_calls': audit.vendor_calls}
        reference = gradient_snapshot(named)
        audit.dense_calls = audit.vendor_calls = 0
        loss_vendor, seconds_vendor = run_pass(model, batch, named, 'vendor', args.chunk_tokens, rng)
        row['vendor'] = {'ce': loss_vendor, 'seconds_with_audit': seconds_vendor,
                         'triton_projection_calls': audit.dense_calls, 'vendor_calls': audit.vendor_calls}
        row['ce_abs_delta'] = abs(loss_vendor - loss_invariant)
        row['gradients_vendor_vs_invariant'] = compare_gradients(reference, gradient_snapshot(named))
        if args.control:
            # Same-arithmetic repeats: separate run-to-run nondeterminism from the arithmetic change.
            vendor_snapshot = gradient_snapshot(named)
            for mode, baseline, loss_baseline in (('invariant', reference, loss_invariant),
                                                  ('vendor', vendor_snapshot, loss_vendor)):
                loss_repeat, _ = run_pass(model, batch, named, mode, args.chunk_tokens, rng)
                row[mode + '_repeat'] = {'ce_abs_delta': abs(loss_repeat - loss_baseline),
                                         'gradients': compare_gradients(baseline, gradient_snapshot(named))}
            del vendor_snapshot
        if args.fp32_reference:
            vendor_snapshot = gradient_snapshot(named)
            row['fp32_reference'] = fp32_reference(model, batch, named, args.chunk_tokens, rng,
                                                   {'invariant': reference, 'vendor': vendor_snapshot})
            del vendor_snapshot
        del reference
        for _, parameter in named:
            parameter.grad = None
        report['batches'].append(row)
        print(json.dumps({'kind': 'vendor_gate_batch', **{k: v for k, v in row.items()
                                                          if k not in ('gradients_vendor_vs_invariant', 'invariant_repeat', 'vendor_repeat')},
                          'grad_rel_l2': row['gradients_vendor_vs_invariant']['relative_l2'],
                          'grad_cos': row['gradients_vendor_vs_invariant']['cosine'],
                          **{key + '_rel_l2': row[key]['gradients']['relative_l2']
                             for key in ('invariant_repeat', 'vendor_repeat') if key in row}}), flush=True)
    restore_training_rng_state(rng, 'cuda')
    report['projection_exactness'] = audit.rows
    carrier.set_training_projection_arithmetic('vendor')
    if args.certificate:
        from natlang_neuralese.eval.native_state import qualify_existing
        certificate = qualify_existing(model, tokenizer, roles, config,
            checkpoint_sha256=report['checkpoint']['sha256'], forward_atol=FORWARD_ATOL,
            gradient_atol=GRADIENT_ATOL, gradient_parameters=named)
        report['runtime_certificate'] = {key: certificate.get(key) for key in (
            'passed', 'forward_max_abs', 'consumer_ce_abs', 'consumer_gradient_max_abs', 'incremental_max_abs',
            'channel_restore_exact', 'gradient_scope')}
        complete = certificate.get('complete_gradient_qualification') or {}
        report['runtime_certificate']['complete_gradient_passed'] = complete.get('passed')
        report['runtime_certificate']['note'] = ('flat side runs in a VJP carrier scope, so this certificate '
                                                 'exercises the invariant kernel, not vendor projections')
    # Exactness: every sampled vendor and control product lies within the FP32-accumulation rounding bound. The
    # production comparison policy's normwise rtol is reported per kind; the qualified invariant kernel itself does
    # not meet it against FP64 products (a same-arithmetic policy, not a single-GEMM rounding bound).
    exact = all(entry['calls'] > 0 and entry['bound_violations'] == 0 for entry in audit.rows.values())
    coverage = (all(row['vendor']['triton_projection_calls'] == 0 and row['vendor']['vendor_calls'] > 0
                    for row in report['batches'])
                and all(kind in audit.rows for kind in ('vendor.forward', 'vendor.dX', 'vendor.dW')))
    cross = all(row['gradients_vendor_vs_invariant']['relative_l2'] <= CROSS_MODE_RTOL
                and row['gradients_vendor_vs_invariant']['cosine'] >= 1 - CROSS_MODE_RTOL
                and not row['gradients_vendor_vs_invariant']['missing'] for row in report['batches'])
    report['gates'] = {'projection_exactness': exact, 'vendor_coverage': coverage, 'cross_mode_bound': cross,
                       **({'runtime_certificate': bool(report['runtime_certificate']['passed'])}
                          if args.certificate else {})}
    report['passed'] = all(report['gates'].values())
    report['seconds'] = time.monotonic() - started
    write_json_durable(args.out, report)
    print(json.dumps({'kind': 'vendor_gate', 'passed': report['passed'], 'gates': report['gates'],
                      'out': str(args.out)}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
