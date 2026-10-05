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
from ..serve.grad import GradSession, encode_text
from ..read import build_inputs, splice
from ..write import greedy_continue


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
    reference_frozen = all(not getattr(engine.heads.feedback, name).requires_grad and
                           getattr(engine.heads.feedback, name).grad_fn is None
                           for name in ['embedding', 'readout', 'control_rows'])
    engine.backbone.ffn_chunk_tokens = 2048
    texts = set()
    for row in map(json.loads, args.records.open()):
        name = target_write(row)
        if name and row.get('split') == 'test' and row.get('training_admission', {}).get('approved') is True:
            texts.add(handover_notes(row)[name])
    if not texts:
        raise ValueError('no admitted held source controls')
    rows = []
    with torch.no_grad():
        for text in sorted(texts)[:args.limit]:
            block = encode_text(engine, text)
            ids = engine._tokens(text)
            raw = engine.backbone.embed(torch.tensor([ids], device=args.device))[0].float()
            encode_equal = torch.equal(block.payload, raw.cpu())
            # The production reader, with independently tokenized scope pieces.
            prefix, suffix = engine._tokens('Context: '), engine._tokens('\nUse that context.')
            before = engine.backbone.embed(torch.tensor([prefix + ids + suffix], device=args.device))
            inputs = build_inputs(engine.backbone, [[prefix, block.payload.to(before), suffix]], heads=engine.heads, device=args.device)
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
        writer_equal = torch.equal(block.payload, engine.backbone.embed(torch.tensor([expected], device=args.device))[0].float().cpu())
        engine.heads.stop.mlp_out.bias.copy_(stop)
    source = encode_text(engine, 'France has Paris as its capital.')
    gradient_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Context: '},
                         {'type': 'neuralese', 'id': source.id}, {'type': 'text', 'text': '\nName the capital.'}]}]
    replay = GradSession(engine).run({'arguments': [source.id], 'terms': [
        {'kind': 'crossEntropy', 'messages': gradient_messages,
         'target': {'role': 'assistant', 'content': 'Paris'}}]})
    gradient = engine.store.get(replay['gradients'][source.id]).payload
    gradient_passed = bool(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
    passed = all(r['encode_equal'] and r['serving_transport_equal'] for r in rows) and writer_equal and gradient_passed and reference_frozen
    report = {'schema': 'natlang.neuralese-runtime-handoff/1', 'rows': rows,
              'runtime_transport_passed': passed, 'greedy_fixed_length_writer_equal': writer_equal,
              'input_gradient_finite_nonzero': gradient_passed, 'reference_buffers_frozen': reference_frozen,
              'autonomous_stopping_qualified': False, 'semantic_compression_qualified': False,
              'pins': {name: sha(getattr(args, name)) for name in ['heads', 'checkpoint', 'certificate', 'records']}}
    args.out.mkdir(parents=True)
    (args.out / 'runtime-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise SystemExit('runtime handoff gate failed')
    save_foundation_port(engine, args.heads, args.out / 'heads.pt', runtime_report=report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
