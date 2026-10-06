"""Qualify one-stage sketch adjoints and measure their cost on exact weights.

This is an execution/gradient diagnostic, not consumer-quality qualification.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.execution import prefill_write_context, unroll_write, write_generated
from ..train.output_embedding_projection import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8, 32])
    p.add_argument('--context-tokens', type=int, default=256)
    p.add_argument('--stage-batches', type=int, nargs='+', default=[1, 8])
    a = p.parse_args()
    if a.out.exists() or a.context_tokens < 1 or any(n < 1 for n in a.lengths + a.stage_batches):
        raise ValueError('fresh output and positive geometry required')
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(127)
    engine, state = load_recurrence_checkpoint(a.checkpoint, device='cuda', dtype=torch.bfloat16)
    backbone, heads = engine.backbone, engine.heads
    backbone.checkpoint_layers = True
    backbone.elide_checkpoint_rng()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for name, parameter in heads.named_parameters():
        parameter.requires_grad_(not name.startswith('content.reference.'))
    tokens = engine.tokenizer.encode('Context: the blue package arrived on Tuesday; the red package is delayed. ',
                                     add_special_tokens=False)
    tokens = (tokens * ((a.context_tokens + len(tokens) - 1) // len(tokens)))[:a.context_tokens]
    base = backbone.embed(torch.tensor([tokens], device='cuda')).detach()
    rows = []
    for length in a.lengths:
        for mode, stage_batch in [('one_step', 1), *[('local_stage', b) for b in a.stage_batches]]:
            heads.zero_grad(set_to_none=True)
            context = base.clone().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            pre = prefill_write_context(backbone, heads, context)
            with torch.no_grad():
                expected = unroll_write(backbone, heads, pre, length=length)
            branches = []
            def capture(module, inputs, output):
                if torch.is_grad_enabled():
                    branches.append(output)
            hook = heads.feedback.register_forward_hook(capture)
            try:
                written = write_generated(backbone, heads, pre, mode, length=length,
                                          local_stage_batch_size=stage_batch)
            finally:
                hook.remove()
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - start
            support, scope_credit, grads = [], True, []
            if mode == 'local_stage':
                sketches = branches[::2]
                for j in sorted({0, min(1, length - 1), length - 1}):
                    adjoint = torch.linspace(-1, 1, written.final.shape[-1], device='cuda')
                    grads = torch.autograd.grad((written.final[:, j].float() * adjoint).sum(),
                        sketches + [context], allow_unused=True, retain_graph=True)
                    nonzero, offset = [], 0
                    for sketch, grad in zip(sketches, grads[:-1]):
                        if grad is not None:
                            nonzero.extend(offset + i for i in range(sketch.shape[1]) if bool(grad[:, i].any()))
                        offset += sketch.shape[1]
                    support.append({'completion': j, 'sketches_reached': nonzero, 'passed': nonzero == [j]})
                    scope_credit = scope_credit and grads[-1] is not None and bool(grads[-1].any())
            delta = {key: float((getattr(written, key).detach().float() - getattr(expected, key).float()).abs().max())
                     for key in ('inputs', 'shallow', 'final', 'payload', 'stop_logits')}
            replay_delta = (float(written.local_replay_max_abs_error)
                            if written.local_replay_max_abs_error is not None else None)
            # Time ordinary backward separately from support diagnostics.
            torch.cuda.synchronize()
            start = time.perf_counter()
            (written.payload.float().square().mean() + .1 * written.sketch_target_loss).backward()
            torch.cuda.synchronize()
            backward_s = time.perf_counter() - start
            feedback_grad = sum(float(p.grad.float().norm()) for p in heads.feedback.parameters() if p.grad is not None)
            finite = all(bool(torch.isfinite(p.grad).all()) for p in heads.parameters() if p.grad is not None)
            row = dict(mode=mode, stage_batch_size=stage_batch, length=length, context_tokens=len(tokens), forward_seconds=forward_s,
                       backward_seconds=backward_s, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                       rollout_max_abs_errors=delta, local_replay_max_abs_error=replay_delta,
                       gradient_support=support, scope_receives_credit=scope_credit,
                       feedback_gradient_norm=feedback_grad, finite_gradients=finite)
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (a.out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            del written, expected, pre, context, branches, grads
            if mode == 'local_stage':
                del sketches
            gc.collect()
            torch.cuda.empty_cache()
    local = [row for row in rows if row['mode'] == 'local_stage']
    passed = all(all(v == 0 for v in row['rollout_max_abs_errors'].values())
                 and all(g['passed'] for g in row['gradient_support']) and row['scope_receives_credit']
                 and row['feedback_gradient_norm'] > 0 and row['finite_gradients'] for row in local)
    report = dict(schema='natlang.local-stage-sketch-diagnostic/1', checkpoint_sha256=sha(a.checkpoint),
                  checkpoint_step=state['step'], rows=rows, implementation_checks_passed=passed,
                  consumer_task_qualified=False, autonomous_stop_qualified=False,
                  scope='Exact primal plus truncated local replay adjoints; not full-BPTT Jacobian parity or task quality.')
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit('local stage qualification failed')


if __name__ == '__main__':
    main()
