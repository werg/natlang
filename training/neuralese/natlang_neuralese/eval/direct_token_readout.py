"""Exercise shared causal TCP readout and producer mixture separation.

The TCP is deliberately perturbed, not trained. This is mechanism evidence only,
not qualification of a TCP artifact, Engine, GradSession, or consumer task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex
from ..model.neuralese_projection import TokenChoiceProjection, token_choice_logits
from ..serve.direct_write import write_direct
from ..train.direct_export import load_direct_artifact
from .direct_runtime import opened_context


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    # Qualify the causal equation independently of TF32 matrix-shape low bits.
    torch.set_float32_matmul_precision('highest')
    restored = load_direct_artifact(a.artifact, base_snapshot=a.base_snapshot, device=a.device)
    engine = restored.engine
    backbone = engine.backbone
    for p in backbone.parameters():
        p.requires_grad_(False)
    context = backbone.embed(torch.tensor([engine.tokenizer.encode(
        'The parcel arrived yesterday. Describe its status.', add_special_tokens=False)], device=a.device))
    tcp = TokenChoiceProjection(context.shape[-1], blocks=2, hidden=64, kernel=7).to(a.device)
    # Exercise preceding-position taps, not just an identity/current-state mapping.
    generator = torch.Generator(device=a.device).manual_seed(37)
    with torch.no_grad():
        for block in tcp.stack.blocks:
            block.taps[1:].fill_(0.125)
            block.down.weight.copy_(torch.randn(block.down.weight.shape, generator=generator,
                                               device=a.device) * 0.001)
        plain = write_direct(backbone, restored.neuralese_projection, opened_context(engine, context),
                             16, engine.stop_ids, length=16)
        adapted = write_direct(backbone, restored.neuralese_projection, opened_context(engine, context),
                               16, engine.stop_ids, length=16, token_choice_projection=tcp)
        hidden = torch.stack([adapted.opened.top] + adapted.states, 1)
        segments = torch.zeros(hidden.shape[:2], dtype=torch.long, device=a.device)
        sequence = token_choice_logits(backbone, hidden, tcp, segments)
        incremental = torch.cat([token_choice_logits(backbone, hidden[:, max(0, j+1-adapted.history_limit):j+1],
                                    tcp, segments[:, max(0, j+1-adapted.history_limit):j+1], last_only=True)
                                 for j in range(hidden.shape[1])], 1)
        causal_delta = float((sequence - incremental).abs().max())
        base_delta = float((sequence - backbone.logits(hidden)).abs().max())
        payload_delta = float((plain.complete()[0] - adapted.complete()[0]).abs().max())
        stop_delta = max(abs(x-y) for x, y in zip(plain.stop_logits, adapted.stop_logits))
    state = hidden.detach().float().requires_grad_(True)
    # Compare gradient through a final readout with and without discarded history.
    full = token_choice_logits(backbone, state, tcp, segments, last_only=True).float().square().mean()
    full_gradient, = torch.autograd.grad(full, state)
    suffix = token_choice_logits(backbone, state[:, -adapted.history_limit:], tcp,
                                 segments[:, -adapted.history_limit:], last_only=True).float().square().mean()
    suffix_gradient, = torch.autograd.grad(suffix, state)
    gradient_delta = float((full_gradient - suffix_gradient).abs().max())
    finite = bool(torch.isfinite(full_gradient).all() and torch.isfinite(suffix_gradient).all())
    # Backend matrix shapes may change floating point low bits.
    passed = causal_delta <= 1e-4 and gradient_delta <= 1e-5 and finite and payload_delta == 0 and base_delta > 0 and stop_delta > 0
    report = {'schema': 'natlang.direct-token-readout-mechanism/1',
              'artifact_manifest_sha256': sha256_file_hex(a.artifact / 'manifest.json'),
              'matmul_precision': torch.get_float32_matmul_precision(), 'tcp_config': tcp.stack.config, 'history_limit': adapted.history_limit,
              'sequence_incremental_logits_delta': causal_delta, 'history_gradient_delta': gradient_delta,
              'tcp_vs_base_logits_delta': base_delta, 'fixed_length_payload_delta': payload_delta,
              'stop_log_odds_delta': stop_delta, 'finite_gradients': finite, 'mechanism_passed': passed,
              'tcp_artifact_qualified': False, 'full_runtime_qualified': False,
              'scope': 'Synthetic nonidentity TCP mechanism, base mixture separation; not trained TCP behavior.'}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if not passed:
        raise SystemExit('TCP readout mechanism failed; preserve evidence')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'base-snapshot', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__':
    main()
