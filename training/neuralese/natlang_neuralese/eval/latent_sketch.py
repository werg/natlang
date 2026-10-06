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
from ..serve.engine import StepWriter, GenerationRequest
from ..train.execution import prefill_write_context, unroll_write, write_generated
from ..train.sketch_handoff import install_latent_sketch
from ..train.output_embedding_projection import sha
from ..write import Opened, read_back
from ..serve.grad import GradSession, encode_text
from ..serve.store import make_block


def _load_parent(path: Path):
    """The channel's parent: a recurrence-training checkpoint (the LFM lineage), or a runtime-qualified foundation
    port (`foundation-v1` recipe output, e.g. the Maple lineage, which starts from its student without port LoRA)."""
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('warmup'):
        from ..serve import load_engine
        engine=load_engine(heads_checkpoint=str(path),device='cuda',dtype=torch.bfloat16)
        return engine, {'step':state['warmup']['step'],
                        'identity':{'options':{'heads':str(path),'sketch_gradient':'local_stage',
                                                'local_stage_batch_size':16}}}
    if state.get('schema') == 'natlang.neuralese_recurrence_checkpoint/1':
        return load_recurrence_checkpoint(path, device='cuda', dtype=torch.bfloat16)
    foundation = state.get('foundation') or {}
    if not (foundation.get('qualified') and foundation.get('runtime_qualified')):
        raise ValueError('parent must be a recurrence checkpoint or a runtime-qualified foundation port')
    from ..serve import load_engine
    engine = load_engine(heads_checkpoint=str(path), device='cuda', dtype=torch.bfloat16)
    return engine, {'step': 0, 'identity': {'options': {'heads': str(path), 'rank': 0}}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cutoffs', type=int, nargs='+', default=[2, 4])
    p.add_argument('--lengths', type=int, nargs='+', default=[8, 32])
    p.add_argument('--profile', choices=['latent-sketch-v1', 'latent-sketch-v2'], default='latent-sketch-v2')
    p.add_argument('--retain-trained-heads', action='store_true',
                   help='Check the loaded learned channel without installing fresh heads or exporting an initialization')
    p.add_argument('--float32-control', action='store_true', help='promote exact loaded BF16 weight values for cache/layout diagnosis; not a production checkpoint')
    a = p.parse_args(argv)
    if a.out.exists():
        raise ValueError('fresh diagnostic output required')
    a.out.mkdir(parents=True)
    torch.set_num_threads(2)
    engine, state = _load_parent(a.checkpoint)
    if a.retain_trained_heads and engine.heads.profile != a.profile:
        raise ValueError('loaded learned channel does not match the requested profile')
    if a.float32_control:
        engine.backbone.float()
        engine.heads.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    parent = engine.heads
    proof = engine.foundation
    backbone = engine.backbone
    if a.retain_trained_heads:
        # This diagnostic audits head/input adjoints without accumulating a
        # training graph through shared context or altering any backbone weight.
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    # Raw foundation at each preceding ordinary-token state predicts the NEXT
    # embedding, never the token already consumed at that state.
    ids = engine.tokenizer.encode('The package arrived yesterday. Today it is ready for collection.', add_special_tokens=False)
    context = backbone.embed(torch.tensor([ids], device='cuda'))
    rows = []
    for cutoff in ([parent.cutoff] if a.retain_trained_heads else a.cutoffs):
        engine.heads, engine.foundation = parent, proof
        torch.manual_seed(71)
        heads = parent if a.retain_trained_heads else install_latent_sketch(engine, cutoff=cutoff, profile=a.profile)
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
                writer = StepWriter(backbone, heads, Opened(pre.cache, pre.state, None, pre.top), length, length=length)
                while not writer.step():
                    pass
                served = writer.complete(0.0, None)[0]
                parity_delta = float((served - written.payload).abs().max())
                # With zero input-space residual the output is the reference of
                # each COMPLETED latent state, not sketch identity or a shift.
                payload_states = heads.payload_states(written.final, pre.top)
                top_delta = float((heads.content(written.inputs, payload_states) - written.payload).abs().max())
                bypass_delta = float((written.inputs - written.payload).abs().max())
                # Changing sketches passed to the content module alone cannot
                # change a top-state payload (the writer itself remains causal).
                no_bypass_delta = float((heads.content(torch.zeros_like(written.inputs), payload_states) - written.payload).abs().max())
                cached = read_back(backbone, heads, pre.cache, written.payload, all_logits=False)
                full = backbone.forward_embeds(torch.cat([context, written.payload], 1), logits=False)
                cross_layout_delta = float((cached['logits'] - backbone.logits(full['h_final'][:, -1:])[:, -1]).abs().max())
                # Keep the production chunk layout for an exact cache-restore
                # control. Merged-prefix BF16 GEMMs/attention are a separate
                # numerical diagnostic, not an exact replay comparator.
                fresh_prefix = backbone.forward_embeds(context, cutoff=heads.cutoff, logits=False)
                recomputed = backbone.forward_embeds(written.payload, cache=fresh_prefix['cache'], logits=False)
                cache_delta = float((cached['logits'] - backbone.logits(recomputed['h_final'][:, -1:])[:, -1]).abs().max())
                cache_lengths_equal = cached['cache'].lengths == recomputed['cache'].lengths
                cache_contents_equal = all(torch.equal(a.window, b.window) if hasattr(a, 'window')
                    else torch.equal(a.k, b.k) and torch.equal(a.v, b.v)
                    for a, b in zip(cached['cache'].states, recomputed['cache'].states))
            del writer, served, written, pre
            gc.collect()
            for name, parameter in heads.named_parameters():
                parameter.requires_grad_(not name.startswith('content.reference.'))
            pre = prefill_write_context(backbone, heads, context)
            gradient_mode = (state['identity']['options'].get('sketch_gradient', 'local_stage')
                             if a.retain_trained_heads else 'one_step' if heads.autoregressive else 'unroll')
            written = write_generated(backbone, heads, pre, gradient_mode, length=length,
                                      local_stage_batch_size=state['identity']['options'].get('local_stage_batch_size', 1))
            sketch_target = float(written.sketch_target_loss.detach()) if written.sketch_target_loss is not None else None
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
                       cache_readback_logit_delta=cache_delta, cache_readback_lengths_equal=cache_lengths_equal,
                       cache_readback_contents_equal=cache_contents_equal, merged_prefix_layout_logit_delta=cross_layout_delta,
                       consumer_ce=float(loss.detach()), sketch_target_loss=sketch_target, gradients=gradients, finite_gradients=finite,
                       sketch_receives_consumer_gradient=sketch_gradient > 0,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (a.out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            heads.zero_grad(set_to_none=True)
            del loss, logits, out, consumer, written, pre
            gc.collect()
    session = GradSession(engine)
    source = encode_text(engine, 'The package arrived yesterday.')
    prompt = [('tok', token) for token in engine.tokenizer.encode('Context: ', add_special_tokens=False)] + [('block', source.id)]
    with torch.no_grad():
        direct = unroll_write(backbone, heads, prefill_write_context(backbone, heads, session._embed_items(prompt, {})), length=8)
        stored = make_block(direct.payload[0], engine.dialect)
        engine.store.put(stored)
    leaf = source.payload.to('cuda').detach().requires_grad_(True)
    direct = unroll_write(backbone, heads, prefill_write_context(backbone, heads, session._embed_items(prompt, {source.id: leaf})), length=8)
    adjoint = torch.linspace(-1, 1, direct.payload.numel(), device='cuda').reshape_as(direct.payload[0]).float()
    direct_gradient, = torch.autograd.grad((direct.payload[0].float() * adjoint).sum(), leaf)
    replay = session._rewritten(stored.id, {source.id: leaf}, {stored.id: ({}, prompt, [], 0)}, {})
    replay_gradient, = torch.autograd.grad((replay * adjoint).sum(), leaf)
    replay_delta = float((replay_gradient - direct_gradient).abs().max())
    replay_value_equal = torch.equal(replay.detach().cpu(), stored.payload)
    passed = replay_delta == 0 and replay_value_equal and all((a.retain_trained_heads or r['reference_next_token_delta'] == 0) and r['serving_training_delta'] == 0
                 and r['top_state_delta'] == 0 and r['no_sketch_bypass_delta'] == 0
                 and r['cache_readback_logit_delta'] == 0 and r['cache_readback_lengths_equal'] and r['cache_readback_contents_equal']
                 and r['finite_gradients'] and r['sketch_receives_consumer_gradient'] for r in rows)
    # Exercise the real public input-gradient and typed host wire surfaces.
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Context: '},
                {'type': 'neuralese', 'id': source.id}, {'type': 'text', 'text': '\nSummarize it.'}]}]
    term = {'kind': 'crossEntropy', 'messages': messages,
            'target': {'role': 'assistant', 'content': 'The package arrived.'}}
    public = session.run({'arguments': [source.id], 'terms': [term]})
    public_gradient = engine.store.get(public['gradients'][source.id]).payload
    input_loss = session._term(term, {source.id: leaf})
    input_gradient, = torch.autograd.grad(input_loss, leaf)
    input_delta = float((public_gradient - input_gradient.detach().cpu()).abs().max())
    with torch.no_grad():
        typed_response = engine.generate(GenerationRequest(messages=[{'role':'user','content':'Describe the package status.'}],
            template={'call':'return_result','arguments':{'status':'success'},'value':'write','value_type':'unknown'},
            neuralese_length=8, max_tokens=128))
    calls = typed_response['choices'][0]['message'].get('tool_calls') or []
    wire = len(calls) == 1 and json.loads(calls[0]['function']['arguments']).get('value') == [
        {'type':'neuralese', 'id':typed_response['neuralese']['blocks'][0]['id'], 'value_type':'unknown'}]
    passed = passed and input_delta == 0 and wire
    report = dict(schema='natlang.latent-sketch-diagnostic/1', parent_sha256=sha(a.checkpoint),
                  parent_step=state['step'], profile=a.profile,
                  sketch_gradient=gradient_mode, layer_types=backbone.layer_types, rows=rows,
                  retained_trained_heads=a.retain_trained_heads, weights_unmodified=a.retain_trained_heads,
                  foundation_requalified=False,
                  float32_diagnostic_control=a.float32_control,
                  producer_replay_gradient_delta=replay_delta, producer_replay_value_equal=replay_value_equal,
                  public_input_gradient_delta=input_delta, typed_wire_value_restored=wire,
                  implementation_checks_passed=passed, runtime_qualified=passed and not a.float32_control, consumer_task_qualified=False,
                  scope='Exact loaded shared writer, cache restore, input/producer gradient replay and typed wire; not foundation requalification, learned task quality or autonomous stopping.')
    (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not passed:
        raise SystemExit('latent sketch implementation diagnostic failed')
    if not a.float32_control and not a.retain_trained_heads:
        from ..train.adapters import lora_state
        parent_state = torch.load(state['identity']['options']['heads'], map_location='cpu', weights_only=False, mmap=True)
        export = {k:parent_state[k] for k in ('backbone','control_head_rows') if k in parent_state}
        export.update(schema='natlang.neuralese-latent-sketch-initialization/1',
                      heads={k:v.detach().cpu() for k,v in heads.state_dict().items()},
                      control_rows=backbone.control_rows.detach().cpu(),
                      lora={k:v.detach().cpu() for k,v in lora_state(backbone).items()},
                      lora_rank=state['identity']['options'].get('rank',0),
                      lora_layers=list(range(backbone.num_layers)),
                      port_config={**heads.port_config(), 'cutoff':heads.cutoff, 'max_length':heads.max_length},
                      foundation={**engine.foundation, 'runtime_qualified':True, 'runtime_report':report},
                      lineage={'parent_checkpoint_sha256':sha(a.checkpoint), 'parent_step':state['step'],
                               'fresh_architecture':True, 'optimizer_inherited':False})
        torch.save(export, a.out/'heads.pt')
        report['heads_sha256'] = sha(a.out/'heads.pt')
        (a.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
