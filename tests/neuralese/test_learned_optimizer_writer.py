"""The learned optimizer's writer: contextual re-encoding, cued weight arrays, exact branch logits and gradients."""
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.model.native_adapters import create_adapter, run_native, target_parameters
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from natlang_neuralese.serve.native_grad import _leaf_block
from natlang_neuralese.tune.learned import numbers as N
from natlang_neuralese.tune.learned.writer import ArrayWrite, StateWrite, Writer

GRID = N.GridTokens(open=10, close=11, separator=12, positive=13, negative=14, digits=tuple(range(20, 30)))
MESSAGES = [{'role': 'user', 'content': 't4 t5 t6 t7'}]


def test_teacher_forced_writer_matches_one_exact_graph_forward(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    writer = Writer(engine, GRID, cue=False)
    array = ArrayWrite('lambda:x', (True, False, True), (None, None, None), targets=(3, 0, -2))
    with torch.no_grad():
        result = writer.run(MESSAGES, [StateWrite('bx', 'instruction', 't8 t9', extra_slots=2)], [array])
    assert result.integers['lambda:x'] == [3, 0, -2]
    assert result.tokens['lambda:x'] == [token for token, _, _ in N.layout(GRID, [3, 0, -2], (True, False, True))]
    state = result.states['bx']
    assert len(state.token_ids) == 2 + 2
    assert float(result.logprobs['lambda:x'][1]) == 0.  # forced slot: no sampling score
    # The same computation as one graph forward over the full token sequence.
    rendered = engine.render_request(MESSAGES, [])
    ids = [t for seg in rendered.segments for t in engine._template_tokens(seg, rendered.escape_nonce)]
    roles = engine.role_tokens
    write = engine._template_tokens(' bx=') + [roles.produce_open]
    start = len(ids) + len(write)
    ids += write + engine._template_tokens('t8 t9') + [GRID.separator] * 2 + [roles.produce_close]
    ids += engine._template_tokens(' lambda:x=')
    decisions = []
    for token, kind, slot in N.layout(GRID, [3, 0, -2], (True, False, True)):
        if kind in ('sign', 'forced-sign'):
            decisions.append(len(ids) - 1)
        ids.append(token)
    model = engine.shared_model
    plan = compile_graph([Invocation('x', tuple(ids), producing=((start, start + 4),))])
    with torch.no_grad():
        graph = model(plan)
        logits = model.logits(graph['h_final'][0, decisions])
    torch.testing.assert_close(result.sign_logits['lambda:x'], logits, rtol=1e-4, atol=1e-4)
    from natlang_neuralese.model.shared_state import capture_invocation_span
    captured = capture_invocation_span(graph, 'x', start, start + 4)
    for ours, theirs in zip(state.layers, captured['layers']):
        for key, value in ours.items():
            if key != 'kind':
                torch.testing.assert_close(value, theirs[key], rtol=1e-4, atol=1e-4)


def test_branch_logits_equal_the_teacher_forced_other_sign(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    writer = Writer(engine, GRID, cue=False)
    with torch.no_grad():
        first = writer.run(MESSAGES, [], [ArrayWrite('s', (True, True), (None, None), targets=(2, 1))])
        flipped = writer.run(MESSAGES, [], [ArrayWrite('s', (True, True), (None, None), targets=(-2, 1))])
    torch.testing.assert_close(first.other_digit_logits['s'][0], flipped.digit_logits['s'][0])
    torch.testing.assert_close(first.sign_logits['s'][0], flipped.sign_logits['s'][0])


def test_cued_sampling_parses_and_gradients_reach_writer_adapter_and_live_keys(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    keys = [engine.native_write_text(f't{30 + index} t{40 + index}')['id'] for index in range(3)]
    names = sorted(target_parameters(engine.shared_model))[:4]
    adapter = create_adapter(engine, {'kind': 'free-lora', 'rank': 2, 'seed': 1, 'parameter_names': names})
    leaves = {key: (value + .01 if key.startswith('lora_B.') else value).clone().requires_grad_(True)
              for key, value in adapter.artifact.tensors.items()}
    live = {key: _leaf_block(engine.store.get_native_state(key).block) for key in keys}
    overrides = {key: block for key, (block, _) in live.items()}
    writer = Writer(engine, GRID, cue=True)
    generator = torch.Generator().manual_seed(0)
    def execute():
        result = writer.run(MESSAGES, [StateWrite('bx', 'instruction', 't8 t9')],
                            [ArrayWrite('s', (True, False, True), tuple(keys))],
                            state_overrides=overrides, sample=True, generator=generator)
        N.parse(GRID, result.tokens['s'], (True, False, True))
        loss = result.logprobs['s'].sum() + sum(v.float().pow(2).sum() for layer in result.states['bx'].layers
                                                for k, v in layer.items() if k != 'kind')
        targets = [*leaves.values(), *[v for _, values in live.values() for v in values.values()]]
        return result, torch.autograd.grad(loss, targets, allow_unused=True)
    result, grads = run_native(engine, execute, [{'id': adapter.id, 'scale': 1.}], leaves={adapter.id: leaves},
                               gradient_precision='fp32')
    assert result.integers['s'][1] == 0
    assert [item['mode'] for item in result.receipt['imports']] == ['same-base-adapter-state'] * 3
    adapter_grads, key_grads = grads[:len(leaves)], grads[len(leaves):]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in adapter_grads)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in key_grads)


def test_cotangent_replay_equals_the_single_graph_gradient(tmp_path):
    """Two-phase replay (detached writer outputs -> executor backward -> writer replay with cotangents) gives the
    same key and writer-adapter gradients as differentiating one connected computation."""
    from natlang_neuralese.model.shared_state import native_step, splice_block
    engine = engine_fixture(tmp_path, 'lfm')
    keys = [engine.native_write_text(f't{30 + i} t{40 + i}')['id'] for i in range(2)]
    names = sorted(target_parameters(engine.shared_model))[:4]
    adapter = create_adapter(engine, {'kind': 'free-lora', 'rank': 2, 'seed': 1, 'parameter_names': names})
    writer = Writer(engine, GRID, cue=True)
    array = ArrayWrite('s', (True, True), tuple(keys), targets=(2, -1))
    writes = [StateWrite('bx', 'instruction', 't8 t9')]
    model = engine.shared_model

    def consumer_loss(block, hidden):
        cache = native_step(model, torch.tensor([[4, 5]]))['cache']
        source = {k: getattr(block, k) for k in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256')}
        cache = splice_block(model, cache, block, **source)
        out = native_step(model, torch.tensor([[6, 7]]), cache)
        return model.logits(out['h_final'][0]).float().logsumexp(-1).sum() + hidden.float().pow(2).sum() * 1e-3

    def fresh():
        leaves = {k: (v + .01 if k.startswith('lora_B.') else v).clone().requires_grad_(True) for k, v in adapter.artifact.tensors.items()}
        live = {key: _leaf_block(engine.store.get_native_state(key).block) for key in keys}
        return leaves, live

    entries = [{'id': adapter.id, 'scale': 1.}]
    # Reference: one connected computation, backward inside the native execution (the run_native contract). Here the
    # consumer runs under the same actor, so both sides include its dependence on the adapter.
    leaves, live = fresh()
    def connected():
        out = writer.run(MESSAGES, writes, [array], state_overrides={k: b for k, (b, _) in live.items()})
        consumer_loss(out.states['bx'], out.decision_hidden['s']).backward()
    run_native(engine, connected, entries, leaves={adapter.id: leaves}, gradient_precision='fp32')
    reference = [*(v.grad.clone() for v in leaves.values()), *(v.grad.clone() for _, values in live.values() for v in values.values())]

    # Two phases: detached writer outputs; the consumer backward (inside its own execution) gives their cotangents.
    leaves, live = fresh()
    with torch.no_grad():
        out = run_native(engine, lambda: writer.run(MESSAGES, writes, [array], state_overrides={k: b for k, (b, _) in live.items()}),
                         entries, leaves={adapter.id: leaves})
    state_block, state_values = _leaf_block(out.states['bx'])
    hidden = out.decision_hidden['s'].detach().clone().requires_grad_(True)
    run_native(engine, lambda: consumer_loss(state_block, hidden).backward(), entries, leaves={adapter.id: leaves},
               gradient_precision='fp32')
    def replay():
        again = writer.run(MESSAGES, writes, [array], state_overrides={k: b for k, (b, _) in live.items()})
        outputs, grads = [again.decision_hidden['s']], [hidden.grad]
        for name, leaf in state_values.items():
            layer, field_name = name.split('.')[1:]
            outputs.append(again.states['bx'].layers[int(layer)][field_name]); grads.append(leaf.grad)
        torch.autograd.backward(outputs, grads)
    run_native(engine, replay, entries, leaves={adapter.id: leaves}, gradient_precision='fp32')
    replayed = [*(v.grad for v in leaves.values()), *(v.grad for _, values in live.values() for v in values.values())]
    errors = []
    for ours, theirs in zip(replayed, reference):
        if theirs is None:
            assert ours is None or float(ours.abs().max()) == 0
        else:
            scale = float(theirs.abs().max())
            errors.append(float((ours - theirs).abs().max()) / max(scale, 1e-30))
    assert max(errors) < 1e-4, errors
