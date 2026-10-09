"""Qualify actual raw serving transport and replay against certified weights.

Known source encoding, a fixed-length greedy writer and input gradients are
controls. They do not qualify semantic compression or autonomous stopping.
"""
import argparse
import json
from pathlib import Path
import torch

from ..train.port_handoff import foundation_port, save_foundation_port
from ..train.output_embedding_projection import sha
from ..train.trajectories import target_write, handover_notes
from ..serve.engine import GenerationRequest
from ..serve.chat import call_reply
from ..serve.grad import GradSession, encode_text
from ..read import build_inputs, splice
from ..write import greedy_continue


def qualify_raw_transport(engine, texts, *, limit=8):
    reference_frozen = all(not getattr(engine.heads.feedback, name).requires_grad and
                           getattr(engine.heads.feedback, name).grad_fn is None
                           for name in ['embedding', 'readout', 'control_rows'])
    rows = []
    with torch.no_grad():
        for text in sorted(texts)[:limit]:
            block = encode_text(engine, text)
            ids = engine._tokens(text)
            raw = engine.backbone.embed(torch.tensor([ids], device=engine.device))[0].float()
            encode_equal = torch.equal(block.payload, raw.cpu())
            # The production reader, with independently tokenized scope pieces.
            prefix, suffix = engine._tokens('Context: '), engine._tokens('\nUse that context.')
            before = engine.backbone.embed(torch.tensor([prefix + ids + suffix], device=engine.device))
            inputs = build_inputs(engine.backbone, [[prefix, block.payload.to(before), suffix]], heads=engine.heads, device=engine.device)
            after = splice(engine.backbone, engine.heads, inputs)
            transport_equal = torch.equal(before, after)
            rows.append({'source_tokens': len(ids), 'encode_equal': encode_equal,
                         'serving_transport_equal': transport_equal})
    messages = [{'role': 'user', 'content': 'Name the capital of France.'}]
    with torch.no_grad():
        ordinary = engine.backbone.forward_embeds(engine.prompt_embeddings(messages, None), logits=False)
        expected, _ = greedy_continue(engine.backbone, ordinary['cache'],
                                     engine.backbone.logits(ordinary['h_final'][:, -1:])[:, -1], 4)
        # Stop policy is intentionally outside this initialization gate.
        stop = engine.heads.stop.mlp_out.bias.clone()
        engine.heads.stop.mlp_out.bias.fill_(-100.)
        response = engine.generate(GenerationRequest(messages=messages, forced=[{'neuralese': 'write'}],
                                                     neuralese_length=4, max_tokens=16))
        block = engine.lookup(response['neuralese']['blocks'][0]['id'])
        writer_equal = torch.equal(block.payload, engine.backbone.embed(torch.tensor([expected], device=engine.device))[0].float().cpu())
        # Typed child returns use a native value boundary rather than a quoted
        # string boundary. Compare serving to an ordinary causal continuation.
        native_prefix = call_reply(lambda m, g: engine.tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=g), "return_result", {"status": "success"}, quoted=False)[0]
        embedded = torch.cat([engine.prompt_embeddings(messages, None),
                              engine.backbone.embed(torch.tensor([engine._tokens(native_prefix)], device=engine.device))], dim=1)
        native = engine.backbone.forward_embeds(embedded, logits=False)
        typed_expected, _ = greedy_continue(engine.backbone, native['cache'],
                                           engine.backbone.logits(native['h_final'][:, -1:])[:, -1], 4)
        typed_response = engine.generate(GenerationRequest(messages=messages,
            template={"call": "return_result", "arguments": {"status": "success"},
                      "value": "write", "value_type": "unknown"}, neuralese_length=4, max_tokens=128))
        typed_block = engine.lookup(typed_response['neuralese']['blocks'][0]['id'])
        typed_writer_equal = torch.equal(typed_block.payload,
            engine.backbone.embed(torch.tensor([typed_expected], device=engine.device))[0].float().cpu())
        calls = typed_response['choices'][0]['message'].get('tool_calls') or []
        typed_wire_passed = len(calls) == 1 and json.loads(calls[0]['function']['arguments']).get('value') == [
            {"type": "neuralese", "id": typed_block.id, "value_type": "unknown"}]
        engine.heads.stop.mlp_out.bias.copy_(stop)
    source = encode_text(engine, 'France has Paris as its capital.')
    gradient_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Context: '},
                         {'type': 'neuralese', 'id': source.id}, {'type': 'text', 'text': '\nName the capital.'}]}]
    replay = GradSession(engine).run({'arguments': [source.id], 'terms': [
        {'kind': 'crossEntropy', 'messages': gradient_messages,
         'target': {'role': 'assistant', 'content': 'Paris'}}]})
    gradient = engine.store.get(replay['gradients'][source.id]).payload
    gradient_passed = bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)

    class DirectRawSession(GradSession):
        """Token-aligned control bypassing port read projection/markers."""
        def _embed_items(self, items, leaves, starts=None):
            pieces, run = [], []
            position = 0
            def flush():
                if run:
                    pieces.append(self.backbone.embed(torch.tensor([run], device=engine.device)))
                    run.clear()
            for kind, value in items:
                if starts is not None:
                    starts.append(position)
                if kind == 'tok':
                    run.append(value)
                    position += 1
                else:
                    flush()
                    payload = self._payload(value, leaves).to(self.backbone.embedding_weight.dtype)[None]
                    pieces.append(payload)
                    position += payload.shape[1]
            flush()
            return torch.cat(pieces, 1)

    leaf = source.payload.to(engine.device).detach().requires_grad_(True)
    direct_loss = DirectRawSession(engine)._term({'kind': 'crossEntropy', 'messages': gradient_messages,
                                                'target': {'role': 'assistant', 'content': 'Paris'}}, {source.id: leaf})
    direct_gradient, = torch.autograd.grad(direct_loss, leaf)
    gradient_equal = torch.equal(gradient, direct_gradient.detach().cpu())
    gradient_delta = float((gradient - direct_gradient.detach().cpu()).abs().max())
    with torch.inference_mode():
        prepared_prompt_equal = torch.equal(engine.prompt_embeddings(gradient_messages, None),
            engine.prompt_embeddings(gradient_messages, None,
                                    prepared=engine._prompt_plan(gradient_messages, None)))
        direct_reply = engine.generate(GenerationRequest(messages=messages, max_tokens=4, seed=73))
    engine.start()
    try:
        queued_reply = engine.submit(GenerationRequest(messages=messages, max_tokens=4, seed=73)).result()
        queued_prompt_equal = (queued_reply['choices'][0]['message'] == direct_reply['choices'][0]['message']
                               and queued_reply['usage'] == direct_reply['usage'])
    finally:
        engine.stop()
    passed = bool(rows) and all(r['encode_equal'] and r['serving_transport_equal'] for r in rows) and writer_equal and typed_writer_equal and typed_wire_passed and gradient_passed and gradient_equal and reference_frozen and prepared_prompt_equal and queued_prompt_equal
    report = {'schema': 'natlang.neuralese-runtime-handoff/1', 'rows': rows,
              'device_type': torch.device(engine.device).type,
              'model_dtype': str(engine.backbone.embedding_weight.dtype),
              'conv_kernel_enabled': getattr(engine.backbone, 'conv_kernel', None) is not None,
              'prepared_prompt_embeddings_equal': prepared_prompt_equal, 'queued_generation_equal': queued_prompt_equal,
              'runtime_transport_passed': passed, 'greedy_fixed_length_writer_equal': writer_equal,
              'typed_greedy_fixed_length_writer_equal': typed_writer_equal, 'typed_wire_value_restored': typed_wire_passed,
              'input_gradient_finite_nonzero': gradient_passed, 'reference_buffers_frozen': reference_frozen,
              'input_gradient_equal_direct_raw': gradient_equal, 'max_abs_input_gradient_delta': gradient_delta,
              'autonomous_stopping_qualified': False, 'semantic_compression_qualified': False,
              'scope': 'exact raw serving transport, fixed-length writer and typed wire, plus input-gradient replay controls; not autonomous stopping or semantic compression'}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['heads', 'checkpoint', 'certificate', 'records', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_length < 4:
        parser.error('positive limit and max-length >=4 required')
    if args.out.exists():
        parser.error('fresh immutable runtime output required')
    torch.set_num_threads(2)
    engine = foundation_port(heads=args.heads, checkpoint=args.checkpoint,
                             certificate=args.certificate, device=args.device, max_length=args.max_length)
    engine.backbone.ffn_chunk_tokens = 2048
    texts = set()
    for row in map(json.loads, args.records.open()):
        name = target_write(row)
        if name and row.get('split') == 'test' and row.get('training_admission', {}).get('approved') is True:
            texts.add(handover_notes(row)[name])
    if not texts:
        raise ValueError('no admitted held source controls')
    report = qualify_raw_transport(engine, texts, limit=args.limit)
    passed = report['runtime_transport_passed']
    report['pins'] = {name: sha(getattr(args, name)) for name in ['heads', 'checkpoint', 'certificate', 'records']}
    args.out.mkdir(parents=True)
    (args.out / 'runtime-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise SystemExit('runtime handoff gate failed')
    save_foundation_port(engine, args.heads, args.out / 'heads.pt', runtime_report=report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
