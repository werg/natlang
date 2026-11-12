"""Learned Neuralese operators (model/learned_operators.py) and the shared use ablation (eval/neuralese_ablation.py)."""

import pytest
import torch

from natlang_neuralese.model.learned_operators import (OPERATORS, OperatorRows, bind_tokens, load_rows,
                                                       operator_embeddings, registry_sha256, resolve_spans, save_rows)
from natlang_neuralese.data.native_graph_schema import SCHEMA, parse_graph

FINGERPRINT = 'f' * 64


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(ch) for ch in text]}

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids)


def tiny_model():
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    torch.manual_seed(3)
    model = SharedStateModel(tiny('lfm'), (56, 57, 58, 59), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=.05)
    return model.eval()


def small_bindings(prompt_len=3):
    """Bindings in the tiny vocabulary: prompt IDs 10.."""
    out = {}
    for name, spec in OPERATORS.items():
        out[name] = {'schema': 'natlang.learned-operator/1', 'operator': name, 'version': spec.version,
                     'tokenizer_fingerprint': FINGERPRINT, 'prompt_token_ids': list(range(10, 10 + prompt_len)),
                     'seed_cap': spec.seed_cap, 'prompt_sha256': '0' * 64}
    return out


PO, PC = 56, 57


def seed_of(index):
    """An input-specific seed of varying length (the producer span's starting text)."""
    return [20 + (index * 3 + k) % 9 for k in range(2 + index % 3)]


def graph_row(index, seed=None, *, name='compact-value', prompt=(10, 11, 12)):
    """Producer: prompt rows, a value, produce_open, the seed (producing, CE 1) and produce_close; consumer imports it."""
    seed = seed_of(index) if seed is None else seed
    value = [30 + (index * 7 + k) % 20 for k in range(5)]
    producer = [1, *prompt, *value, PO, *seed, PC]
    p_start, s_start = 1, 2 + len(prompt) + len(value)
    s_stop = s_start + len(seed)
    consumer = [2, 3 + index % 5, 4, 5, 6 + index % 3]
    version = OPERATORS[name].version
    return {'schema': SCHEMA, 'id': f'g{index}', 'split': 'train', 'source_groups': [f'world:{index}'],
            'tokenizer_fingerprint': FINGERPRINT,
            'nodes': [{'name': 'producer', 'token_ids': producer,
                       'target_weights': [0.] * s_start + [1.] * (len(seed) + 1), 'producing': [[s_start, s_stop]]},
                      {'name': 'consumer', 'token_ids': consumer, 'target_weights': [0., 0., 1., 1., 1.],
                       'imports': [{'at': 1, 'source': 'producer', 'start': s_start, 'stop': s_stop}]}],
            'operators': [{'operator': name, 'version': version, 'node': 'producer', 'kind': 'prompt',
                           'start': p_start, 'stop': p_start + len(prompt)},
                          {'operator': name, 'version': version, 'node': 'producer', 'kind': 'seed',
                           'start': s_start, 'stop': s_stop, 'provenance': 'gold'}]}


def test_bind_tokens_round_trips_prompt_and_binds_the_span_cap():
    binding = bind_tokens(CharTokenizer(), 'compact-history', tokenizer_fingerprint=FINGERPRINT)
    spec = OPERATORS['compact-history']
    assert CharTokenizer().decode(binding['prompt_token_ids']) == spec.prompt
    assert binding['seed_cap'] == spec.seed_cap and 'slot_token_ids' not in binding


def test_every_operator_has_dedicated_data_target_distinct_prompt_and_span_cap():
    assert len({spec.prompt for spec in OPERATORS.values()}) == len(OPERATORS)
    assert all(spec.data_target >= 20_000 and spec.seed_cap >= 128 for spec in OPERATORS.values())
    assert len(registry_sha256()) == 64


def test_text_initialized_rows_reproduce_the_text_forward_exactly_and_receive_gradients():
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    rows.reset_to_text(model.port.embedding_weight)
    row = graph_row(0)
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    spans = resolve_spans(row, example.plan, rows.bindings)
    assert [span.kind for span in spans] == ['prompt', 'seed']
    text = model(example.plan)['h_final']
    learned = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))['h_final']
    assert torch.equal(text, learned)
    from natlang_neuralese.data.native_graph import graph_loss_from_result
    result = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))
    loss, _ = graph_loss_from_result(model, example, result)
    loss.backward()
    grad = rows.rows('compact-value').grad
    assert grad is not None and float(grad.abs().sum()) > 0
    assert rows.rows('extract').grad is None


def test_resolve_spans_rejects_wrong_tokens_seeds_outside_production_and_over_cap():
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    row = graph_row(0, prompt=(10, 11, 13))
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match='differ from the bound'):
        resolve_spans(row, example.plan, rows.bindings)
    row = graph_row(0)
    row['nodes'][0]['producing'] = []
    row['nodes'][1]['imports'] = []
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match='producing span'):
        resolve_spans(row, example.plan, rows.bindings)
    capped = {name: {**binding, 'seed_cap': 1} for name, binding in bindings.items()}
    row = graph_row(0)
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match='span cap'):
        resolve_spans(row, example.plan, capped)


def test_seed_schedule_anneals_only_seed_weights():
    from natlang_neuralese.model.learned_operators import seed_factor, seed_schedule, seed_weighted
    schedule = seed_schedule({'start': 1.0, 'end': 0.1, 'updates': 10})
    assert seed_factor(schedule, 0) == 1.0 and seed_factor(schedule, 10) == pytest.approx(0.1)
    assert seed_factor(schedule, 5) == pytest.approx(0.55) and seed_factor(schedule, 50) == pytest.approx(0.1)
    assert seed_factor(None, 3) == 1.0
    for bad in ({'start': 2, 'end': 0, 'updates': 1}, {'start': 1, 'end': 0, 'updates': 0}, {'start': 1, 'end': 0}):
        with pytest.raises(ValueError):
            seed_schedule(bad)
    row = graph_row(1)
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    # Historical schedule /1: the closing marker anneals with the seed (schedule /2: test_guidance_operator.py).
    scaled = seed_weighted(example, 0.25, {**schedule, 'schema': 'natlang.operator-seed-schedule/1'})
    producer = row['nodes'][0]
    s_start, s_stop = producer['producing'][0]
    expected = [weight * (0.25 if s_start <= local <= s_stop else 1.) for local, weight in enumerate(producer['target_weights'])]
    assert list(scaled.target_weights[:len(expected)]) == expected
    assert scaled.target_weights[len(expected):] == example.target_weights[len(expected):]
    assert seed_weighted(example, 1.) is example


def test_rows_save_and_load_round_trip(tmp_path):
    rows = OperatorRows(small_bindings(), 8)
    with torch.no_grad():
        rows.rows('extract').normal_()
    save_rows(rows, tmp_path / 'rows.pt')
    loaded = load_rows(tmp_path / 'rows.pt')
    assert torch.equal(loaded.rows('extract'), rows.rows('extract'))
    assert loaded.bindings == rows.bindings


def test_ablation_externalized_own_state_matches_actual_and_conditions_differ():
    from natlang_neuralese.eval import neuralese_ablation as ablation
    from natlang_neuralese.model.shared_state import capture_block
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    rows.reset_to_text(model.port.embedding_weight)
    with torch.no_grad():
        for name in OPERATORS:
            rows.rows(name).add_(0.3 * torch.randn_like(rows.rows(name)))
    raw = [graph_row(index) for index in range(4)]
    examples = [parse_graph(row, tokenizer_fingerprint=FINGERPRINT) for row in raw]
    spans = {example.id: resolve_spans(row, example.plan, rows.bindings) for row, example in zip(raw, examples)}
    # Importing a row's own captured state as external state reproduces the internal-import graph.
    example = examples[0]
    identity = ablation._identity(model)
    with torch.no_grad():
        embeddings = operator_embeddings(model, rows, example.plan, spans[example.id])
        result = model(example.plan, embeddings=embeddings)
        edge = example.plan.invocations[1].imports[0]
        block = capture_block(result, 'producer', edge.start, edge.stop, text='', **identity)
        weights = ablation.consumer_weights(example)
        own, _ = ablation._score(model, example, example.plan, weights, embeddings, None, None, 128)
        plan, external = ablation.externalized(example, {ablation.internal_edges(example.plan)[0][1:]: block})
        again, _ = ablation._score(model, example, plan, weights, embeddings, external,
                                   {name: identity for name in external}, 128)
    assert again['nll_sum'] == pytest.approx(own['nll_sum'], rel=1e-5, abs=1e-6)
    report = ablation.ablate(model, examples, rows=rows, operator_spans=spans, seed=1)
    assert report['rows'] == 4 and set(report['aggregate']) == {'actual', 'shuffled', 'zeroed'}
    assert report['aggregate']['actual']['tokens'] == 4 * 3
    assert report['aggregate']['actual']['nll'] == pytest.approx(
        sum(row['actual']['nll_sum'] for row in report['per_row']) / report['aggregate']['actual']['weight'])
    assert all(row['donor'] != row['id'] for row in report['per_row'])
    assert report['use_nll_delta']['zeroed-minus-actual'] != 0
    # Consumers that do not import anything are not scored.
    assert ablation.consumer_weights(example)[:len(example.plan.invocations[0].token_ids)] == (
        (0.,) * len(example.plan.invocations[0].token_ids))


def test_serving_generation_steps_match_the_training_graph_capture():
    """The operator endpoint's path (site prefill with prompt rows, then one producing step per span token) equals
    the graph producer's capture of the same span."""
    from natlang_neuralese.model.shared_state import capture_block, concatenate_captures, native_step
    from natlang_neuralese.model.learned_operators import ResolvedSpan
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    rows.reset_to_text(model.port.embedding_weight)
    with torch.no_grad():
        rows.rows('compact-value').add_(0.2 * torch.randn_like(rows.rows('compact-value')))
    row = graph_row(2)
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    spans = resolve_spans(row, example.plan, rows.bindings)
    identity = {'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64, 'embedding_weight_sha256': 'c' * 64,
                'architecture': model.port.config.model_type}
    producer = row['nodes'][0]
    start, stop = producer['producing'][0]
    with torch.no_grad():
        result = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))
        graph_block = capture_block(result, 'producer', start, stop, text='', **identity)
        ids = torch.tensor([producer['token_ids'][:start]])
        prompt = next(span for span in spans if span.kind == 'prompt')
        embeddings = rows.apply(model.embed(ids), [ResolvedSpan('compact-value', 'prompt', 'site', prompt.start, prompt.stop)])
        output = native_step(model, ids, embeddings=embeddings)
        chunks = []
        for token in producer['token_ids'][start:stop]:
            output = native_step(model, torch.tensor([[token]]), output['cache'], producing=True)
            chunks.append(output)
    fields = concatenate_captures(chunks)
    for graph_layer, step_layer in zip(graph_block.layers, fields['layers']):
        for key, value in graph_layer.items():
            if key != 'kind':
                torch.testing.assert_close(step_layer[key].float(), value.float(), rtol=1e-4, atol=1e-5)


def test_runtime_consumer_frames_equal_the_registry():
    import json as _json
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'ts-host/src/native/operator-frames.ts').read_text()
    for name, spec in OPERATORS.items():
        if spec.placement == 'call':
            continue  # the guidance operator's consumers see the engine's guidance turn (test_guidance_operator.py)
        line = next(line for line in source.splitlines() if line.strip().startswith(f"'{name}':"))
        assert f'open: {_json.dumps(spec.consumer_open)}'.replace('"', "'") in line, name
        assert f'close: {_json.dumps(spec.consumer_close)}'.replace('"', "'") in line, name
