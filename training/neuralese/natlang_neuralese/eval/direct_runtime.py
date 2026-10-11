"""Exact direct-artifact transport and producer/input-gradient qualification.

This exercises the shared writer, Engine and GradSession. It establishes channel
implementation properties, not task success or autonomous stopping quality.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _synthetic_tcp(backbone, device, seed=37):
    """A reproducible nonidentity TCP used only for mechanism diagnostics."""
    from ..model.neuralese_projection import TokenChoiceProjection

    tcp = TokenChoiceProjection(backbone.embedding_weight.shape[1], blocks=2, hidden=64, kernel=7).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for block in tcp.stack.blocks:
            block.taps[0].fill_(1.0)
            block.taps[1:].fill_(0.125)
            block.up.weight.copy_(torch.randn(block.up.weight.shape, generator=generator, device=device) * 0.001)
            block.up.bias.copy_(torch.randn(block.up.bias.shape, generator=generator, device=device) * 0.001)
            block.down.weight.copy_(torch.randn(block.down.weight.shape, generator=generator, device=device) * 0.001)
            block.down.bias.copy_(torch.randn(block.down.bias.shape, generator=generator, device=device) * 0.001)
    return tcp


def _tensor_state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _multi_turn_context_diagnostic(engine):
    """Compare GradSession's weighted turn segmentation with a full sequence readout."""
    from ..serve.chat import render_messages

    session = GradSession(engine)
    messages = [
        {'role': 'user', 'content': 'First turn: the parcel is delayed.'},
        {'role': 'assistant', 'content': 'I will check the delivery status.'},
        {'role': 'user', 'content': 'Second turn: what should I do next?'},
    ]
    rendered = render_messages(messages, None, engine._template, engine.specials,
                               block_type=engine.block_value_type)
    items = session._items(rendered.segments, rendered.blocks, rendered.escape_nonce)
    starts = []
    embeds = session._embed_items(items, {}, starts)
    out = engine.backbone.forward_embeds(
        embeds, logits=False, cutoff=engine.heads.cutoff if not engine.heads.read_markers else None)
    weights = session._context_weights(items, feedback_weight=0.5, covered_replies=0)
    shared_logp, shared_weights = session._context_logp(out['h_final'], items, starts, weights)

    # Build the reference labels from each next-token target's owning message, then score
    # the entire hidden sequence in one causal pass. This independently exercises the
    # sequence API against the per-turn weighted path used by _context_logp.
    marker = engine.tokenizer.convert_tokens_to_ids('<|im_start|>')
    segments = torch.full(out['h_final'].shape[:2], -1, device=out['h_final'].device, dtype=torch.long)
    turn = -1
    for index, item in enumerate(items):
        if item == ('tok', marker):
            turn += 1
        if index > 0 and item[0] == 'tok':
            predictor = starts[index] - 1
            if predictor >= 0:
                segments[:, predictor] = turn
    baseline_logits = engine.token_logits(out['h_final'], segments=segments).float()
    target_indices = [i for i in range(1, len(items)) if items[i][0] == 'tok' and weights[i] > 0]
    positions = torch.tensor([starts[i] - 1 for i in target_indices], device=baseline_logits.device)
    targets = torch.tensor([items[i][1] for i in target_indices], device=baseline_logits.device)
    baseline_logp = torch.log_softmax(baseline_logits[0, positions], -1).gather(1, targets[:, None])[:, 0]
    difference = float((shared_logp - baseline_logp).abs().max()) if len(target_indices) else 0.0
    return {
        'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        'prompt_item_count': len(items), 'weighted_target_count': len(target_indices),
        'positive_weight_sum': float(shared_weights.sum()),
        'turn_count': turn + 1, 'sequence_vs_context_logp_max_abs_delta': difference,
        'same_target_positions': len(target_indices) == shared_logp.numel(),
        'mechanism_match': difference <= 1e-5 and len(target_indices) == shared_logp.numel(),
    }


def _guidance_rollback_diagnostic(engine):
    """Exercise the real Engine rollback/replay path, then independently replay its kept prefix."""
    from ..serve.guidance import Settings

    messages = [{'role': 'user', 'content': 'Return a short code example.'}]
    request = GenerationRequest(messages=messages, max_tokens=64,
                                forced=['eval(code="const broken = ;")'],
                                guidance=Settings(repeat=0, run=0, syntax=True, retries=1), seed=901)
    original = engine._guide
    captured = {}

    def capture_after_rollback(seq):
        if not captured and seq.snapshots:
            # Keep only the real request-start snapshot so a rejection traverses the
            # engine's documented restore-and-recompute branch over prior reply tokens.
            seq.snapshots = seq.snapshots[:1]
        before = len(seq.guide.rejections) if seq.guide is not None else 0
        original(seq)
        if captured or seq.guide is None or len(seq.guide.rejections) == before:
            return
        rejection = dict(seq.guide.rejections[before])
        captured.update({
            'rejection': rejection,
            'kept_items': list(seq.items),
            'kept_token_history': None if seq.token_history is None else seq.token_history.detach().clone(),
            'kept_logits': None if seq.logits is None else seq.logits.detach().clone(),
            'snapshot_item_count': int(seq.snapshots[0][0]) if seq.snapshots else None,
            'rollback_items_replayed': len(seq.items) - (int(seq.snapshots[0][0]) if seq.snapshots else 0),
        })

    engine._guide = capture_after_rollback
    try:
        response = engine.generate(request)
    finally:
        engine._guide = original
    if not captured:
        return {'triggered': False, 'finish_reason': response['choices'][0]['finish_reason'],
                'guidance': response.get('x_natlang_guidance', {})}

    plan = engine._prompt_plan(messages, None)
    embeds = engine.prompt_embeddings(messages, None, prepared=plan)
    cutoff = engine.heads.cutoff if not engine.heads.read_markers else None
    with torch.no_grad():
        first = engine.backbone.forward_embeds(embeds, logits=False, cutoff=cutoff)
        _, history = engine.advance_token_readout(None, first['h_final'][:, -1:], last_only=True)
        logits = None
        items = captured['kept_items']
        if items:
            second = engine.backbone.forward_ids(torch.tensor([items], device=engine.device),
                                                 cache=first['cache'], logits=False, cutoff=cutoff)
            score, history = engine.advance_token_readout(history, second['h_final'], last_only=True)
            logits = score[:, -1]
    expected_history = captured.pop('kept_token_history')
    expected_logits = captured.pop('kept_logits')
    history_delta = float((history - expected_history).abs().max()) if expected_history is not None else None
    logits_delta = float((logits - expected_logits).abs().max()) if logits is not None and expected_logits is not None else None
    return {
        'triggered': True,
        'rejection': captured['rejection'],
        'kept_reply_token_count': len(captured['kept_items']),
        'rollback_items_replayed': captured['rollback_items_replayed'],
        'finish_reason': response['choices'][0]['finish_reason'],
        'history_replay_max_abs_delta': history_delta,
        'next_logits_replay_max_abs_delta': logits_delta,
        'restored_state_matches_fresh_replay': history_delta is not None and history_delta <= 1e-5 and
                                                logits_delta is not None and logits_delta <= 1e-5 and
                                                not captured['rejection'].get('accepted', False) and
                                                captured['rollback_items_replayed'] > 0,
    }


def opened_context(engine, embeds):
    pre = prefill_write_context(engine.backbone, engine.heads, embeds)
    return Opened(pre.cache, pre.state, engine.backbone.logits(pre.top[:, None])[:, -1], pre.top)


def diagnose(artifact, base_snapshot, out, lengths, device='cuda', token_choice_diagnostic=False,
             tcp_stream_diagnostic=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model = load_direct_artifact(artifact, base_snapshot=base_snapshot, device=device)
    engine = model.engine
    backbone, projection = engine.backbone, model.neuralese_projection
    synthetic_tcp_identity = None
    if token_choice_diagnostic:
        torch.set_float32_matmul_precision('highest')
        tcp = _synthetic_tcp(backbone, device, seed=37)
        engine.token_choice_projection = tcp
        synthetic_tcp_identity = _tensor_state_sha256(tcp)
        torch.save({'seed': 37, 'config': tcp.stack.config, 'state_dict': tcp.state_dict()},
                   out / 'synthetic-token-choice-state.pt')
    writer_options = {'token_choice_projection': getattr(engine, 'token_choice_projection', None)}
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    ids = engine.tokenizer.encode('The package arrived yesterday. It is ready for collection.', add_special_tokens=False)
    context = backbone.embed(torch.tensor([ids], device=device))
    rows = []
    for length in lengths:
        with torch.no_grad():
            opened = opened_context(engine, context)
            direct = write_direct(backbone, projection, opened, length, engine.stop_ids, length=length, **writer_options)
            payload = direct.complete()[0]
            # Scheduler's batched layer path, one row at this qualification point.
            from ..serve.batch import step_rows
            served = DirectWriter(backbone, projection, opened, length, engine.stop_ids, length=length, **writer_options)
            while True:
                value = served.decide()
                if value is None:
                    break
                hidden, caches = step_rows(backbone, served.next_input(value)[:, None], [served.cache],
                                           range(backbone.num_layers))
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
                               length=length, **writer_options)
        value = written.complete()[0]
        for vector in written.payloads:
            vector.retain_grad()
        for state in written.states:
            if state.requires_grad:
                state.retain_grad()
        targets = torch.tensor([engine.tokenizer.encode('It is ready.', add_special_tokens=False)], device=device)
        consumer = torch.cat([value, backbone.embed(targets[:, :-1])], 1)
        result = backbone.forward_embeds(consumer, logits=False)
        states = result['h_final'][:, -targets.shape[1]:]
        segments = torch.zeros(states.shape[:2], dtype=torch.long, device=states.device)
        logits = engine.token_logits(states, segments).float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        gradients = {name: float(torch.linalg.vector_norm(p.grad, dtype=torch.float64))
                     for name, p in projection.named_parameters() if p.grad is not None}
        feedback_gradients = [float(torch.linalg.vector_norm(vector.grad, dtype=torch.float64)) if vector.grad is not None else None
                              for vector in written.payloads]
        state_gradients = [float(torch.linalg.vector_norm(state.grad, dtype=torch.float64)) if state.grad is not None else None
                           for state in written.states]
        finite = bool(gradients) and all(torch.isfinite(p.grad).all().item() for p in projection.parameters() if p.grad is not None)
        row = {'length': length, 'scheduler_writer_delta': serving_delta, 'cache_readback_delta': cache_delta,
               'cache_lengths_equal': lengths_equal, 'cache_contents_equal': contents_equal,
               'consumer_ce': float(loss.detach()), 'projection_gradients': gradients,
               'payload_gradient_norm_by_position': feedback_gradients,
               'final_state_gradient_norm_by_position': state_gradients,
               'payload_rms_by_position': [float(vector.detach().float().square().mean().sqrt())
                                           for vector in written.payloads],
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
                              engine.stop_ids, length=8, **writer_options)
        stored = engine.store.put(make_block(writer.complete()[0][0], engine.dialect))
    leaf = source.payload.to(device).detach().requires_grad_(True)
    writer = write_direct(backbone, projection, opened_context(engine, session._embed_items(prompt, {source.id: leaf})),
                          8, engine.stop_ids, length=8, **writer_options)
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
    decision_delta = None
    if token_choice_diagnostic:
        options = ['yes', 'no']
        with torch.no_grad():
            decision_scores, own_lengths = session.decision_logprobs(messages, None, options, {})
            prompt_items, rests = session.decision_render(messages, None, options)
            full_scores = [session._score(prompt_items, [('tok', token) for token in rest], {}, False)['token_logp'][-n:].sum()
                           for rest, n in zip(rests, own_lengths)]
            decision_delta = float((decision_scores - torch.stack(full_scores)).abs().max())
    multi_turn = _multi_turn_context_diagnostic(engine) if tcp_stream_diagnostic else None
    guidance_rollback = _guidance_rollback_diagnostic(engine) if tcp_stream_diagnostic else None
    passed = (decision_delta is None or decision_delta <= 1e-4) and replay_delta == 0 and replay_equal and input_delta == 0 and wire and all(
        r['scheduler_writer_delta'] == 0 and r['cache_readback_delta'] == 0 and r['cache_lengths_equal'] and
        r['cache_contents_equal'] and r['finite_gradients'] and r['projection_receives_consumer_gradient'] for r in rows)
    if tcp_stream_diagnostic:
        passed = passed and bool(multi_turn and multi_turn['mechanism_match']) and bool(
            guidance_rollback and guidance_rollback.get('triggered') and
            guidance_rollback.get('restored_state_matches_fresh_replay'))
    package = Path(__file__).resolve().parents[1]
    report = {'schema': 'natlang.neuralese-direct-runtime-qualification/1',
              'artifact_manifest_sha256': sha256_file_hex(Path(artifact) / 'manifest.json'),
              'source_checkpoint': model.manifest['training_checkpoint'],
              'runtime_code': {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob('*.py'))},
              'rows': rows, 'producer_replay_gradient_delta': replay_delta, 'producer_replay_value_equal': replay_equal,
              'public_input_gradient_delta': input_delta, 'typed_wire_value_restored': wire,
              'runtime_qualified': passed and not token_choice_diagnostic,
              'synthetic_tcp_mechanism_passed': passed if token_choice_diagnostic else None,
              'token_choice_diagnostic': token_choice_diagnostic, 'decision_score_delta': decision_delta, 'consumer_task_qualified': False, 'autonomous_stopping_qualified': False,
              'synthetic_token_choice_state_sha256': synthetic_tcp_identity,
              'synthetic_token_choice_state_file_sha256': sha256_file_hex(out / 'synthetic-token-choice-state.pt')
                  if synthetic_tcp_identity is not None else None,
              'tcp_stream_diagnostic': tcp_stream_diagnostic,
              'multi_turn_sequence_comparison': multi_turn,
              'guidance_rollback_replay': guidance_rollback,
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
    parser.add_argument('--token-choice-diagnostic', action='store_true', help='Inject a synthetic TCP; mechanism evidence only')
    parser.add_argument('--tcp-stream-diagnostic', action='store_true',
                        help='Also compare multi-turn causal readout and guided rollback replay; requires synthetic TCP')
    args = parser.parse_args()
    if not args.lengths or any(n < 1 for n in args.lengths):
        parser.error('positive write lengths required')
    if args.tcp_stream_diagnostic and not args.token_choice_diagnostic:
        parser.error('--tcp-stream-diagnostic requires --token-choice-diagnostic')
    diagnose(args.artifact, args.base_snapshot, args.out, args.lengths, args.device,
             args.token_choice_diagnostic, args.tcp_stream_diagnostic)


if __name__ == '__main__':
    main()
