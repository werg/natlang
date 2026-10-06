"""Exercise the shared latent sketch path on actual weights, not a toy model.

Checks top-state projection, causal reference alignment, serving/replay parity
and consumer gradients. These are implementation diagnostics, not task success,
a foundation certificate, or a next-token quality gate for shallow sketches.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..serve.engine import StepWriter
from ..train.execution import prefill_write_context, unroll_write
from ..train.sketch_handoff import install_latent_sketch
from ..train.output_embedding_projection import sha
from ..write import Opened


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cutoffs', type=int, nargs='+', default=[2, 4])
    p.add_argument('--lengths', type=int, nargs='+', default=[8, 32])
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh diagnostic output required')
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    engine, state = load_recurrence_checkpoint(a.checkpoint, device='cuda', dtype=torch.bfloat16)
    parent = engine.heads
    proof = engine.foundation
    backbone = engine.backbone
    # Raw foundation at each preceding ordinary-token state predicts the NEXT
    # embedding, never the token already consumed at that state.
    ids = engine.tokenizer.encode('The package arrived yesterday. Today it is ready for collection.', add_special_tokens=False)
    context = backbone.embed(torch.tensor([ids], device='cuda'))
    rows = []
    for cutoff in a.cutoffs:
        engine.heads, engine.foundation = parent, proof
        torch.manual_seed(71)
        heads = install_latent_sketch(engine, cutoff=cutoff)
        with torch.no_grad():
            ordinary = backbone.forward_embeds(context, logits=False)
            actual = heads.content.reference(ordinary['h_final'][:, :-1])
            expected = backbone.embed(backbone.logits(ordinary['h_final'][:, :-1]).argmax(-1))
            reference_delta = float((actual - expected).abs().max())
        for length in a.lengths:
            with torch.no_grad():
                pre = prefill_write_context(backbone, heads, context)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                written = unroll_write(backbone, heads, pre, length=length)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                writer = StepWriter(backbone, heads, Opened(pre.cache, pre.state, None), length, length=length)
                while not writer.step():
                    pass
                served = writer.complete(0.0, None)[0]
                parity_delta = float((served - written.payload).abs().max())
                # With zero input-space residual the output is the reference of
                # each COMPLETED latent state, not sketch identity or a shift.
                top_delta = float((heads.content.reference(written.final) - written.payload).abs().max())
                bypass_delta = float((written.inputs - written.payload).abs().max())
                # Changing sketches passed to the content module alone cannot
                # change a top-state payload (the writer itself remains causal).
                no_bypass_delta = float((heads.content(torch.zeros_like(written.inputs), written.final) - written.payload).abs().max())
            del writer, served, written, pre
            gc.collect()
            for name, parameter in heads.named_parameters():
                parameter.requires_grad_(not name.startswith('content.reference.'))
            pre = prefill_write_context(backbone, heads, context)
            written = unroll_write(backbone, heads, pre, length=length)
            # Real consumer token CE through the written payload and full stack.
            target = torch.tensor([engine.tokenizer.encode('It is ready.', add_special_tokens=False)], device='cuda')
            consumer = torch.cat([heads.read_embeddings(backbone, written.payload), backbone.embed(target[:, :-1])], 1)
            out = backbone.forward_embeds(consumer, logits=False)
            logits = backbone.logits(out['h_final'][:, -target.shape[1]:]).float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            gradients = {name: float(parameter.grad.float().norm()) for name, parameter in heads.named_parameters()
                         if parameter.grad is not None}
            finite = all(torch.isfinite(parameter.grad).all().item() for parameter in heads.parameters() if parameter.grad is not None)
            sketch_gradient = sum(v for k, v in gradients.items() if k.startswith('feedback.'))
            row = dict(cutoff=cutoff, length=length, writer_s=elapsed, reference_next_token_delta=reference_delta,
                       serving_training_delta=parity_delta, top_state_delta=top_delta,
                       sketch_payload_difference=bypass_delta, no_sketch_bypass_delta=no_bypass_delta,
                       consumer_ce=float(loss.detach()), gradients=gradients, finite_gradients=finite,
                       sketch_receives_consumer_gradient=sketch_gradient > 0,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            print(json.dumps(row), flush=True)
            (a.out/'metrics.jsonl').open('a').write(json.dumps(row)+'\n')
            heads.zero_grad(set_to_none=True)
            del loss, logits, out, consumer, written, pre
            gc.collect()
    passed = all(r['reference_next_token_delta'] == 0 and r['serving_training_delta'] == 0
                 and r['top_state_delta'] == 0 and r['no_sketch_bypass_delta'] == 0
                 and r['finite_gradients'] and r['sketch_receives_consumer_gradient'] for r in rows)
    report = dict(schema='natlang.latent-sketch-diagnostic/1', parent_sha256=sha(a.checkpoint),
                  parent_step=state['step'], layer_types=backbone.layer_types, rows=rows,
                  implementation_checks_passed=passed, runtime_qualified=False, consumer_task_qualified=False,
                  scope='Untrained shallow sketch diagnostic; no shallow fidelity gate or inherited channel certificate.')
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit('latent sketch implementation diagnostic failed')


if __name__ == '__main__':
    main()
