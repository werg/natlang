"""Profile declared arithmetic on a complete real graph using the shared trainer.

This is a diagnostic, not a second training implementation. It never writes a
training checkpoint or grants task-quality/readiness qualification. Comparison
modes retain identical source weights; the existing cached execution verifier
uses its unchanged numerical tolerances for each mode.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common.hashing import sha256_file_hex_cached
from ..model.native_model import load_native_model, load_native_checkpoint
from ..model.native_precision import NATIVE_LINEAR_PRECISIONS, set_native_linear_precision
from ..train.native_state_sft import NativeGraphSource, packed_graph_loss
from .native_state import qualify_existing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--base')
    source.add_argument('--checkpoint')
    parser.add_argument('--graphs', required=True, type=Path)
    parser.add_argument('--pair-model')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--modes', nargs='+', choices=NATIVE_LINEAR_PRECISIONS, default=['ieee', 'bf16x6'])
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 2 or args.chunk_tokens < 1 or args.modes[0] != 'ieee':
        raise ValueError('comparison starts with ieee, at least two repeats and positive chunk tokens')
    torch.manual_seed(711)
    if args.checkpoint:
        model, tokenizer, roles, config, _ = load_native_checkpoint(args.checkpoint, device='cuda')
    else:
        model, tokenizer, roles, config = load_native_model(args.base, device='cuda', dtype=getattr(torch, args.dtype))
    rows = NativeGraphSource(args.graphs, config['tokenizer_fingerprint'], required_split='train',
        pair_model=args.pair_model, role_ids=roles.ids)
    row = next(iter(rows))
    # Identity-init adapter otherwise hides down-path gradients. This temporary
    # diagnostic perturbation is declared and never saved as trained weights.
    with torch.no_grad(): model.adapter.up.weight.normal_(std=.003)
    from ..train.backbone_policy import configure_backbone_training
    trainable = configure_backbone_training(model.port, 'latent')
    model.train(); model.port.checkpoint_layers = True
    report = {'schema': 'natlang.native-throughput-diagnostic/1', 'source': config,
        'graph_source': rows.receipt(), 'graph': row.id, 'tokens': row.plan.token_count,
        'supervised_tokens': sum(weight > 0 for weight in row.target_weights),
        'temporary_adapter_up_std': .003, 'optimizer_updates': 0,
        'backbone_policy': 'latent', 'backbone_trainable_parameters': sum(p.numel() for _, p in trainable), 'modes': [],
        'hardware': torch.cuda.get_device_name(), 'torch': torch.__version__}
    package = Path(__file__).resolve().parents[1]
    report['code_files'] = {name: sha256_file_hex_cached(package/name) for name in (
        'model/shared_state.py','model/native_model.py','model/native_precision.py',
        'maple/fused_moe.py','train/native_state_sft.py','train/optim.py','eval/native_throughput.py')}
    reference = None
    for mode in args.modes:
        set_native_linear_precision(model, mode)
        for repeat in range(args.repeats):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.monotonic()
            loss, _ = packed_graph_loss(model, [row], checkpoint_layers=True, chunk_tokens=args.chunk_tokens)
            loss.backward(); torch.cuda.synchronize(); elapsed = time.monotonic() - start
        gradient = torch.cat([value.grad.detach().float().flatten() for value in model.adapter.parameters()])
        if reference is None: reference = loss.detach().clone(), gradient.clone()
        record = {'mode': mode, 'ce': float(loss.detach()), 'ce_delta_from_ieee': float(loss.detach()-reference[0]),
            'producer_adapter_gradient_max_abs': float((gradient-reference[1]).abs().max()),
            'producer_adapter_gradient_relative_l2': float((gradient-reference[1]).norm()/reference[1].norm()),
            'producer_adapter_gradient_cosine': float(F.cosine_similarity(gradient, reference[1], dim=0)),
            'forward_backward_seconds': elapsed, 'supervised_tps': report['supervised_tokens']/elapsed,
            'cuda_lifetime_peak_bytes': torch.cuda.max_memory_allocated()}
        if args.profile:
            model.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                loss, _ = packed_graph_loss(model, [row], checkpoint_layers=True, chunk_tokens=args.chunk_tokens)
                loss.backward()
            record['operators'] = profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=30)
            profile.export_chrome_trace(str(args.out)+'.'+mode+'.trace.json')
        model.zero_grad(set_to_none=True)
        diagnostic_config = {**config, 'native_linear_precision': mode}
        record['cached_reference'] = qualify_existing(model, tokenizer, roles, diagnostic_config,
            checkpoint_sha256='0'*64, forward_atol=1e-4, gradient_atol=1e-5)
        report['modes'].append(record)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    return 0 if all(item['cached_reference']['passed'] for item in report['modes']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
