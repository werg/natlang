#!/usr/bin/env python3
"""Measure native graph/reference, incremental splice and real channel updates.

This is numerical qualification, not a task-success or full-recipe certificate.
Run on the destination GPU against the actual pinned model inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))

import torch
from natlang_neuralese.model.native_model import load_native_model, native_identity
from natlang_neuralese.model.shared_state import (
    Invocation, ImportSpan, compile_graph, native_step, splice_block, capture_block,
    channel_checkpoint, restore_channel_checkpoint,
)


def qualify(base, *, dtype, device, forward_atol, gradient_atol):
    start = time.monotonic()
    torch.manual_seed(703)
    model, tokenizer, roles, config = load_native_model(base, dtype=dtype, device=device)
    for parameter in model.port.parameters():
        parameter.requires_grad_(False)
    model.port.checkpoint_layers = True
    source = tuple(tokenizer('Write a useful state: the account balance is 42 after adding seven units.',
                             add_special_tokens=False).input_ids)
    target = tuple(tokenizer('Read the state and return the account balance as an integer.',
                             add_special_tokens=False).input_ids)
    source = source[:-4] + (roles.produce_open,) + source[-4:] + (roles.produce_close,)
    target = target[:2] + (roles.consume_open, roles.consume_close) + target[2:]
    lo, hi = len(source) - 5, len(source) - 1
    plan = compile_graph([Invocation('source', source, producing=((lo, hi),)),
                          Invocation('target', target, imports=(ImportSpan(3, 'source', lo, hi),))])
    positions = torch.arange(len(source) + 3, plan.token_count - 1, device=device)
    labels = torch.tensor(target[4:], device=device)

    def objective(result):
        logits = model.logits(result['h_final'][:, positions]).float()
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels)

    flat = model(plan)
    reference = model(plan, reference=True)
    flat_loss, reference_loss = objective(flat), objective(reference)
    flat_gradient = torch.autograd.grad(flat_loss, model.adapter.up.weight)[0]
    reference_gradient = torch.autograd.grad(reference_loss, model.adapter.up.weight)[0]
    forward_error = float((flat['h_final'] - reference['h_final']).detach().abs().max())
    gradient_error = float((flat_gradient - reference_gradient).abs().max())
    ce_error = abs(float(flat_loss.detach()) - float(reference_loss.detach()))
    del flat, reference, flat_loss, reference_loss
    identity = native_identity(model, config, checkpoint_sha256=config['base_inventory_sha256'])
    with torch.no_grad():
        result = model(plan)
        block = capture_block(result, 'source', lo, hi, text=tokenizer.decode(source[lo:hi]), **identity)
        prefix = native_step(model, torch.tensor([target[:3]], device=device))
        cache = splice_block(model, prefix['cache'], block,
                             **{key: value for key, value in identity.items() if key != 'architecture'})
        suffix = native_step(model, torch.tensor([target[3:]], device=device), cache)
        incremental_error = float((suffix['h_final'] - result['h_final'][:, len(source)+3:]).abs().max())
    del result, prefix, suffix, cache, block
    # Bounded interface optimizer check. Backbone training is qualified by the
    # production trainer; this does not substitute an interface-only recipe.
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    loss = objective(model(plan, retain_states=False))
    loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise AssertionError('non-finite native channel gradients')
    before = model.adapter.up.weight.detach().clone()
    optimizer.step()
    changed = bool(torch.any(model.adapter.up.weight != before))
    optimizer.zero_grad(set_to_none=True)
    saved_channel = copy.deepcopy(channel_checkpoint(model))
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        model.adapter.up.weight.add_(1)
    restore_channel_checkpoint(model, saved_channel)
    optimizer.load_state_dict(saved_optimizer)
    torch.testing.assert_close(model.adapter.up.weight, saved_channel['adapter']['up.weight'], rtol=0, atol=0)
    report = dict(schema='natlang.native-numerical-qualification/1', base=str(base), model_config=config,
                  device=str(device), gpu=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
                  forward_max_abs=forward_error, consumer_ce_abs=ce_error,
                  consumer_gradient_max_abs=gradient_error, incremental_max_abs=incremental_error,
                  channel_optimizer_updated=changed, channel_restore_exact=True,
                  backbone_optimizer_tested=False, task_reward_evaluated=False,
                  forward_atol=forward_atol, gradient_atol=gradient_atol,
                  elapsed_seconds=time.monotonic()-start,
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
    report['passed'] = (forward_error <= forward_atol and incremental_error <= forward_atol
                        and ce_error <= forward_atol and gradient_error <= gradient_atol
                        and bool(torch.isfinite(flat_gradient).all()) and bool(flat_gradient.abs().sum() > 0)
                        and changed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--forward-atol', type=float, required=True)
    parser.add_argument('--gradient-atol', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.base, dtype=getattr(torch, args.dtype), device=args.device,
                     forward_atol=args.forward_atol, gradient_atol=args.gradient_atol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
