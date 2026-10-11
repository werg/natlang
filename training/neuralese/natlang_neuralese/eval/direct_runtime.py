"""Exact direct-artifact transport and producer/input-gradient qualification.

This exercises the shared writer, Engine and GradSession. It establishes channel
implementation properties, not task success or autonomous stopping quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex
from ..serve.direct_write import DirectWriter, write_direct
from ..serve.engine import GenerationRequest
from ..serve.grad import GradSession, encode_text
from ..serve.store import make_block
from ..train.direct_export import load_direct_artifact
from ..train.execution import prefill_write_context
from ..write import Opened, read_back


def opened_context(engine, embeds):
    pre = prefill_write_context(engine.backbone, engine.heads, embeds)
    return Opened(pre.cache, pre.state, engine.backbone.logits(pre.top[:, None])[:, -1], pre.top)


def diagnose(artifact, base_snapshot, out, lengths, device='cuda'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model = load_direct_artifact(artifact, base_snapshot=base_snapshot, device=device)
    engine = model.engine
    backbone, projection = engine.backbone, model.neuralese_projection
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    ids = engine.tokenizer.encode('The package arrived yesterday. It is ready for collection.', add_special_tokens=False)
    context = backbone.embed(torch.tensor([ids], device=device))
    rows = []
    for length in lengths:
        with torch.no_grad():
            opened = opened_context(engine, context)
            direct = write_direct(backbone, projection, opened, length, engine.stop_ids, length=length)
            payload = direct.complete()[0]
            # Scheduler's batched layer path, one row at this qualification point.
            from ..serve.batch import step_rows
            served = DirectWriter(backbone, projection, opened, length, engine.stop_ids, length=length)
            while True:
                value = served.decide()
                if value is None:
                    break
                hidden, caches = step_rows(backbone, value[:, None], [served.cache], range(backbone.num_layers))
                served.advance(value, hidden[:, 0], caches[0])
            serving_delta = float((served.complete()[0] - payload).abs().max())
            cached = read_back(backbone, engine.heads, opened.cache, payload, all_logits=False)
            fresh = opened_context(engine, context)
            recomputed = read_back(backbone, engine.heads, fresh.cache, payload, all_logits=False)
            cache_delta = float((cached['logits'] - recomputed['logits']).abs().max())
            lengths_equal = cached['cache'].lengths == recomputed['cache'].lengths
            contents_equal = all(torch.equal(a.window, b.window) if hasattr(a, 'window') else
                                 torch.equal(a.k, b.k) and torch.equal(a.v, b.v)
                                 for a, b in zip(cached['cache'].states, recomputed['cache'].states))
        for parameter in projection.parameters():
            parameter.requires_grad_(True)
        written = write_direct(backbone, projection, opened_context(engine, context), length, engine.stop_ids,
                               length=length)
        value = written.complete()[0]
        targets = torch.tensor([engine.tokenizer.encode('It is ready.', add_special_tokens=False)], device=device)
        consumer = torch.cat([value, backbone.embed(targets[:, :-1])], 1)
        result = backbone.forward_embeds(consumer, logits=False)
        logits = backbone.logits(result['h_final'][:, -targets.shape[1]:]).float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        gradients = {name: float(p.grad.float().norm()) for name, p in projection.named_parameters() if p.grad is not None}
        finite = bool(gradients) and all(torch.isfinite(p.grad).all().item() for p in projection.parameters() if p.grad is not None)
        row = {'length': length, 'scheduler_writer_delta': serving_delta, 'cache_readback_delta': cache_delta,
               'cache_lengths_equal': lengths_equal, 'cache_contents_equal': contents_equal,
               'consumer_ce': float(loss.detach()), 'projection_gradients': gradients,
               'finite_gradients': finite, 'projection_receives_consumer_gradient': sum(gradients.values()) > 0}
        rows.append(row)
        (out / 'rows.json').write_text(json.dumps(rows, indent=2) + '\n')
        projection.zero_grad(set_to_none=True)
        del loss, result, logits, consumer, written, value, direct, served

    session = GradSession(engine)
    source = encode_text(engine, 'The package arrived yesterday.')
    prompt = [('tok', t) for t in engine.tokenizer.encode('Context: ', add_special_tokens=False)] + [('block', source.id)]
    with torch.no_grad():
        writer = write_direct(backbone, projection, opened_context(engine, session._embed_items(prompt, {})), 8,
                              engine.stop_ids, length=8)
        stored = engine.store.put(make_block(writer.complete()[0][0], engine.dialect))
    leaf = source.payload.to(device).detach().requires_grad_(True)
    writer = write_direct(backbone, projection, opened_context(engine, session._embed_items(prompt, {source.id: leaf})),
                          8, engine.stop_ids, length=8)
    payload = writer.complete()[0][0]
    adjoint = torch.linspace(-1, 1, payload.numel(), device=device).reshape_as(payload).float()
    direct_gradient, = torch.autograd.grad((payload.float() * adjoint).sum(), leaf)
    replay = session._rewritten(stored.id, {source.id: leaf}, {stored.id: ({}, prompt, [], 0)}, {})
    replay_gradient, = torch.autograd.grad((replay * adjoint).sum(), leaf)
    replay_delta = float((direct_gradient - replay_gradient).abs().max())
    replay_equal = torch.equal(replay.detach().cpu(), stored.payload)
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
        response = engine.generate(GenerationRequest(messages=[{'role': 'user', 'content': 'Describe the package status.'}],
            template={'call': 'return_result', 'arguments': {'status': 'success'}, 'value': 'write', 'value_type': 'unknown'},
            neuralese_length=8, max_tokens=128))
    calls = response['choices'][0]['message'].get('tool_calls') or []
    wire = len(calls) == 1 and json.loads(calls[0]['function']['arguments']).get('value') == [
        {'type': 'neuralese', 'id': response['neuralese']['blocks'][0]['id'], 'value_type': 'unknown'}]
    passed = replay_delta == 0 and replay_equal and input_delta == 0 and wire and all(
        r['scheduler_writer_delta'] == 0 and r['cache_readback_delta'] == 0 and r['cache_lengths_equal'] and
        r['cache_contents_equal'] and r['finite_gradients'] and r['projection_receives_consumer_gradient'] for r in rows)
    package = Path(__file__).resolve().parents[1]
    report = {'schema': 'natlang.neuralese-direct-runtime-qualification/1',
              'artifact_manifest_sha256': sha256_file_hex(Path(artifact) / 'manifest.json'),
              'source_checkpoint': model.manifest['training_checkpoint'],
              'runtime_code': {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob('*.py'))},
              'rows': rows, 'producer_replay_gradient_delta': replay_delta, 'producer_replay_value_equal': replay_equal,
              'public_input_gradient_delta': input_delta, 'typed_wire_value_restored': wire,
              'runtime_qualified': passed, 'consumer_task_qualified': False, 'autonomous_stopping_qualified': False,
              'scope': 'Exact restored direct writer, cache readback, input/producer adjoints and typed wire; no weight updates.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise SystemExit('direct runtime qualification failed; preserve diagnostic')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--base-snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lengths', type=int, nargs='+', default=[8, 32])
    args = parser.parse_args()
    if not args.lengths or any(n < 1 for n in args.lengths):
        parser.error('positive write lengths required')
    diagnose(args.artifact, args.base_snapshot, args.out, args.lengths, args.device)


if __name__ == '__main__':
    main()
