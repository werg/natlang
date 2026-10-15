"""Exact native-state forward/capture/splice verification on an existing model."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def task_execution_receipt(summary, identity, *, checkpoint_sha256):
    """One actual-harness receipt contract for native SFT and joint training."""
    from ..common.hashing import sha256_file_hex_cached
    score = Path(summary['run'])/'task-score.json' if summary.get('run') else None
    receipt = {'schema': 'natlang.native-task-execution-receipt/1',
        'checkpoint_sha256': identity['checkpoint_sha256'], 'native_identity': identity,
        'packet_sha256': summary.get('packet_sha256'), 'case_indices': summary.get('case_indices'),
        'cases': summary.get('cases'), 'complete': summary.get('complete'), 'eligible': summary.get('eligible'),
        'task_successes': summary.get('task_successes'), 'task_successes_are_qualification_barrier': False,
        'score_path': str(score.resolve()) if score and score.is_file() else None,
        'score_sha256': sha256_file_hex_cached(score) if score and score.is_file() else None}
    receipt['passed'] = bool(identity['checkpoint_sha256'] == checkpoint_sha256 and
                             receipt['complete'] and receipt['eligible'] and receipt['score_sha256'])
    return receipt


def qualify_existing(model, tokenizer, roles, config, *, checkpoint_sha256, forward_atol, gradient_atol,
                     task_execution_receipt=None, backbone_optimizer_updates=0):
    """Verify the live trained instance without reloading a second backbone or changing training state.

    The check compares graph execution with the explicit cached reference, compares
    target CE and adapter gradients, then captures a producer span and consumes it
    by exact native splice. The trainer's optimizer/weights/RNG remain untouched.
    """
    from ..model.native_model import native_identity
    from ..model.shared_state import (Invocation, ImportSpan, capture_block, channel_checkpoint,
                                      compile_graph, native_step, restore_channel_checkpoint, splice_block)

    start = time.monotonic()
    identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha256)
    def probe_tokens(text):
        tokens = tuple(tokenizer(text, add_special_tokens=False).input_ids)
        special = set(getattr(tokenizer, 'all_special_ids', ()))
        if tokens and any(token not in special for token in tokens):
            return tokens
        # A tokenizer with a deliberately tiny lexical vocabulary can map the
        # diagnostic sentence entirely to its zero/padding unknown embedding.
        # Use actual round-tripped lexical text, never synthetic payload rows.
        for _, token in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            if token in special:
                continue
            text = tokenizer.decode([token], skip_special_tokens=False)
            candidate = tuple(tokenizer(text, add_special_tokens=False).input_ids)
            if candidate and any(value not in special for value in candidate):
                return candidate
        raise ValueError('native runtime reference needs a round-tripped lexical token')
    source = probe_tokens('Write a useful state: the account balance is 42 after adding seven units.')
    target = probe_tokens('Read the state and return the account balance as an integer.')
    if not source or not target:
        raise ValueError('native runtime reference needs nonempty source and consumer tokenizations')
    payload_width = min(4, len(source))
    lo = len(source) - payload_width + 1
    source = source[:-payload_width] + (roles.produce_open,) + source[-payload_width:] + (roles.produce_close,)
    hi = lo + payload_width
    prefix_width = min(2, len(target)-1)
    target = target[:prefix_width] + (roles.consume_open, roles.consume_close) + target[prefix_width:]
    import_at = prefix_width + 1
    plan = compile_graph([Invocation('source', source, producing=((lo, hi),)),
                          Invocation('target', target, imports=(ImportSpan(import_at, 'source', lo, hi),))])
    positions = torch.arange(len(source) + import_at, plan.token_count - 1, device=model.input_rows.device)
    labels = torch.tensor(target[import_at+1:], device=model.input_rows.device)
    previous_mode, previous_checkpointing = model.training, model.port.checkpoint_layers
    adapter = model.adapter.up.weight
    try:
        model.eval()
        model.port.checkpoint_layers = True
        flat = model(plan)
        reference = model(plan, reference=True)

        def objective(result):
            logits = model.logits(result['h_final'][:, positions]).float()
            return F.cross_entropy(logits.flatten(0, 1), labels)

        flat_loss, reference_loss = objective(flat), objective(reference)
        flat_gradient = torch.autograd.grad(flat_loss, adapter, retain_graph=False)[0]
        reference_gradient = torch.autograd.grad(reference_loss, adapter, retain_graph=False)[0]
        forward_error = float((flat['h_final'] - reference['h_final']).detach().abs().max())
        gradient_error = float((flat_gradient - reference_gradient).abs().max())
        ce_error = abs(float(flat_loss.detach()) - float(reference_loss.detach()))
        del flat, reference, flat_loss, reference_loss

        with torch.no_grad():
            result = model(plan)
            block = capture_block(result, 'source', lo, hi,
                text=tokenizer.decode(source[lo:hi], skip_special_tokens=False), **identity)
            prefix = native_step(model, torch.tensor([target[:import_at]], device=model.input_rows.device))
            cache = splice_block(model, prefix['cache'], block,
                **{key: value for key, value in identity.items() if key != 'architecture'})
            suffix = native_step(model, torch.tensor([target[import_at:]], device=model.input_rows.device), cache)
            incremental_error = float((suffix['h_final'] - result['h_final'][:, len(source)+import_at:]).abs().max())
            saved_channel = copy.deepcopy(channel_checkpoint(model))
            before = {key: value.detach().clone() for key, value in saved_channel['adapter'].items()}
            restore_channel_checkpoint(model, saved_channel)
            restore_exact = all(torch.equal(model.adapter.state_dict()[key], value)
                                for key, value in before.items())
        del result, prefix, suffix, cache, block
    finally:
        model.port.checkpoint_layers = previous_checkpointing
        model.train(previous_mode)
    passed = (forward_error <= forward_atol and incremental_error <= forward_atol and
              ce_error <= forward_atol and gradient_error <= gradient_atol and
              bool(torch.isfinite(flat_gradient).all()) and bool(flat_gradient.abs().sum() > 0) and restore_exact)
    return {'schema': 'natlang.native-runtime-certificate/1', 'passed': passed,
            'checkpoint_sha256': checkpoint_sha256, 'native_identity': identity,
            'forward_max_abs': forward_error, 'consumer_ce_abs': ce_error,
            'consumer_gradient_max_abs': gradient_error, 'incremental_max_abs': incremental_error,
            'channel_restore_exact': restore_exact, 'forward_atol': forward_atol,
            'gradient_atol': gradient_atol, 'task_execution_receipt': task_execution_receipt,
            'task_successes_are_qualification_barrier': False,
            'backbone_optimizer_tested': backbone_optimizer_updates > 0,
            'backbone_optimizer_updates': int(backbone_optimizer_updates),
            'task_reward_evaluated': bool(task_execution_receipt),
            'elapsed_seconds': time.monotonic() - start}
