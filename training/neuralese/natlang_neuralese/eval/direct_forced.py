"""GPU parity and throughput diagnostic for known-token direct writes.

Compare the shared incremental writer with its causal sequence execution on
exact restored weights. This is mechanism evidence, not task qualification.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex
from ..model.neuralese_projection import TokenChoiceProjection
from ..serve.direct_write import DirectWriter
from ..train.direct_export import load_direct_artifact
from .direct_runtime import opened_context


def _compare(left, right):
    if left is None or right is None:
        return {'both_absent': left is None and right is None,
                'relative_l2': 0.0 if left is None and right is None else float('inf')}
    difference = left.double() - right.double()
    return {'max_abs_delta': float(difference.abs().max()),
            'relative_l2': float(difference.norm() / left.double().norm().clamp_min(1e-12)),
            'finite': bool(torch.isfinite(left).all() and torch.isfinite(right).all())}


def diagnose(artifact, base_snapshot, out, lengths, device='cuda', float32_reference=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if float32_reference:
        # Casting weights alone still permits TF32 matmuls. This diagnostic
        # needs a full-precision reference before attributing deltas to BF16.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_direct_artifact(artifact, base_snapshot=base_snapshot, device=device)
    engine = model.engine
    backbone, projection = engine.backbone, model.neuralese_projection
    if float32_reference:
        # Diagnostic only: distinguish reduced-precision sequence/decode kernel
        # differences from positional or convolution-state mistakes.
        backbone.float()
    tcp = getattr(engine, 'token_choice_projection', None)
    synthetic_identity_tcp = tcp is None
    if tcp is None:
        tcp = TokenChoiceProjection(backbone.embedding_weight.shape[1], blocks=2, hidden=256,
                                    kernel=4).to(device)
    # The ordinary vocabulary table remains frozen, as in the native stage.
    # Exercise all other backbone weights, both projections and prefix inputs.
    parameters = []
    for name, parameter in backbone.named_parameters():
        trainable = parameter.shape != backbone.embedding_weight.shape
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(('backbone.' + name, parameter))
    for label, module in [('neuralese_projection', projection), ('token_choice_projection', tcp)]:
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            parameters.append((label + '.' + name, parameter))
    context_ids = engine.tokenizer.encode('The package arrived yesterday. It is ready for collection.',
                                          add_special_tokens=False)
    text_ids = engine.tokenizer.encode('Summarize the evidence, choose the appropriate action, and report the result. ',
                                       add_special_tokens=False)
    consumer_text = 'It is ready.'
    consumer_targets = torch.tensor([engine.tokenizer.encode(consumer_text, add_special_tokens=False)],
                                   device=device)

    def run(length, bulk):
        context = backbone.embed(torch.tensor([context_ids], device=device)).detach().requires_grad_(True)
        tokens = (text_ids * ((length + len(text_ids) - 1) // len(text_ids)))[:length]
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        writer = DirectWriter(backbone, projection, opened_context(engine, context), length, engine.stop_ids,
                              length=length, tokens=tokens, token_choice_projection=tcp)
        if bulk:
            writer.forced_sequence()
        else:
            while not writer.step():
                pass
        payload = writer.complete()[0]
        adjoint = torch.linspace(-1, 1, payload.numel(), device=device).reshape_as(payload)
        linear_loss = (payload.float() * adjoint).mean() + 0.01 * torch.stack(writer.stop_tensors).mean()
        # A downstream ordinary-token decision exercises the nonlinear consumer,
        # rather than qualifying producer deltas against a linear adjoint alone.
        consumer = torch.cat([payload, backbone.embed(consumer_targets[:, :-1]).detach()], 1)
        consumer_final = backbone.forward_embeds(consumer, logits=False)['h_final']
        from ..model.neuralese_projection import token_choice_logits
        consumer_segments = torch.zeros(consumer_final.shape[:2], device=device, dtype=torch.long)
        consumer_logits = token_choice_logits(backbone, consumer_final, tcp, consumer_segments)
        consumer_logits = consumer_logits[:, -consumer_targets.shape[1]:].float()
        consumer_ce = torch.nn.functional.cross_entropy(
            consumer_logits.reshape(-1, consumer_logits.shape[-1]),
            consumer_targets.expand(payload.shape[0], -1).reshape(-1))
        loss = linear_loss + consumer_ce
        gradients = torch.autograd.grad(loss, [context] + [p for _, p in parameters], allow_unused=True)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        cache = {}
        for index, state in enumerate(writer.cache.states):
            for field in ('window', 'k', 'v'):
                if hasattr(state, field):
                    cache[f'{index}.{field}'] = getattr(state, field).detach().cpu()
        return {'payload': payload.detach().cpu(), 'states': torch.stack(writer.states, 1).detach().cpu(),
                'stops': torch.stack(writer.stop_tensors, 1).detach().cpu(), 'cache': cache,
                'cache_lengths': writer.cache.lengths, 'count': writer.count, 'done': writer.done,
                'loss': float(loss.detach()), 'seconds': elapsed,
                'linear_loss': float(linear_loss.detach()), 'consumer_ce': float(consumer_ce.detach()),
                'consumer_logits': consumer_logits.detach().cpu(),
                'peak_gib': torch.cuda.max_memory_allocated() / 1024**3 if device.startswith('cuda') else None,
                'gradients': {name: None if g is None else g.detach().cpu()
                              for name, g in zip(['context'] + [n for n, _ in parameters], gradients)}}

    rows = []
    for length in lengths:
        incremental, sequence = run(length, False), run(length, True)
        tensor_comparisons = {name: _compare(incremental[name], sequence[name])
                              for name in ('payload', 'states', 'stops', 'consumer_logits')}
        cache_comparisons = {name: _compare(value, sequence['cache'][name])
                             for name, value in incremental['cache'].items()}
        gradient_comparisons = {name: _compare(value, sequence['gradients'][name])
                                for name, value in incremental['gradients'].items()}
        row = {'length': length, 'incremental_seconds': incremental['seconds'],
               'sequence_seconds': sequence['seconds'], 'speedup': incremental['seconds'] / sequence['seconds'],
               'incremental_peak_gib': incremental['peak_gib'], 'sequence_peak_gib': sequence['peak_gib'],
               'loss_abs_delta': abs(incremental['loss'] - sequence['loss']),
               'consumer_ce': {name: result['consumer_ce'] for name, result in
                               [('incremental', incremental), ('sequence', sequence)]},
               'consumer_ce_abs_delta': abs(incremental['consumer_ce'] - sequence['consumer_ce']),
               'cache_lengths_equal': incremental['cache_lengths'] == sequence['cache_lengths'],
               'count_and_stop_equal': (incremental['count'], incremental['done']) ==
                                       (sequence['count'], sequence['done']),
               'tensors': tensor_comparisons, 'cache': cache_comparisons, 'gradients': gradient_comparisons}
        rows.append(row)
        (out / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ('length', 'speedup', 'loss_abs_delta',
                                             'incremental_peak_gib', 'sequence_peak_gib')}), flush=True)
        del incremental, sequence
    report = {'schema': 'natlang.direct-forced-sequence-parity/1',
              'artifact_manifest_sha256': sha256_file_hex(Path(artifact) / 'manifest.json'),
              'synthetic_identity_tcp': synthetic_identity_tcp,
              'float32_reference': float32_reference,
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
              'gradient_objective': 'linear payload/stop diagnostic plus downstream consumer token cross entropy',
              'consumer_target_text': consumer_text,
              'rows': rows, 'task_qualified': False, 'autonomous_stopping_qualified': False,
              'scope': 'Known-token sequence versus incremental shared writer; output/cache/consumer-logit and combined parameter/prefix adjoint deltas. Synthetic consumer fixture is mechanism evidence, not task success. No weight updates or automatic tolerance relaxation.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True)
    parser.add_argument('--base-snapshot', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lengths', nargs='+', type=int, default=[8, 32, 102])
    parser.add_argument('--float32-reference', action='store_true')
    args = parser.parse_args()
    diagnose(args.artifact, args.base_snapshot, args.out, args.lengths, args.device, args.float32_reference)


if __name__ == '__main__':
    main()
