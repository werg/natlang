"""Learned Neuralese operators (model/learned_operators.py) and the shared use ablation (eval/neuralese_ablation.py)."""
import dataclasses

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
    """Bindings in the tiny vocabulary: prompt IDs 10.., slot IDs cycling 20..23."""
    out = {}
    for name, spec in OPERATORS.items():
        out[name] = {'schema': 'natlang.learned-operator/1', 'operator': name, 'version': spec.version,
                     'tokenizer_fingerprint': FINGERPRINT, 'prompt_token_ids': list(range(10, 10 + prompt_len)),
                     'slot_token_ids': [20 + index % 4 for index in range(spec.slots)],
                     'prompt_sha256': '0' * 64, 'slot_text_sha256': '0' * 64}
    return out


def graph_row(index, slots, *, name='compact-value', prompt=(10, 11, 12)):
    """Producer: prompt rows, a value, then K slot positions in a producing span; consumer imports the slots."""
    value = [30 + (index * 7 + k) % 20 for k in range(5)]
    producer = [1, *prompt, *value, *slots]
    p_start, s_start = 1, 1 + len(prompt) + len(value)
    consumer = [2, 3 + index % 5, 4, 5, 6 + index % 3]
    return {'schema': SCHEMA, 'id': f'g{index}', 'split': 'train', 'source_groups': [f'world:{index}'],
            'tokenizer_fingerprint': FINGERPRINT,
            'nodes': [{'name': 'producer', 'token_ids': producer, 'target_weights': [0.] * len(producer),
                       'producing': [[s_start, s_start + len(slots)]]},
                      {'name': 'consumer', 'token_ids': consumer, 'target_weights': [0., 0., 1., 1., 1.],
                       'imports': [{'at': 1, 'source': 'producer', 'start': s_start, 'stop': s_start + len(slots)}]}],
            'operators': [{'operator': name, 'version': 1, 'node': 'producer', 'kind': 'prompt',
                           'start': p_start, 'stop': p_start + len(prompt)},
                          {'operator': name, 'version': 1, 'node': 'producer', 'kind': 'slots',
                           'start': s_start, 'stop': s_start + len(slots)}]}


def test_bind_tokens_cycles_slot_text_and_round_trips_prompt():
    binding = bind_tokens(CharTokenizer(), 'compact-history', tokenizer_fingerprint=FINGERPRINT)
    spec = OPERATORS['compact-history']
    assert CharTokenizer().decode(binding['prompt_token_ids']) == spec.prompt
    assert len(binding['slot_token_ids']) == spec.slots
    seed = [ord(ch) for ch in spec.slot_text]
    assert binding['slot_token_ids'][:len(seed)] == seed
    assert binding['slot_token_ids'][len(seed)] == seed[0]


def test_every_operator_has_dedicated_data_target_and_distinct_prompt():
    assert len({spec.prompt for spec in OPERATORS.values()}) == len(OPERATORS)
    assert all(spec.data_target >= 20_000 and spec.slots >= 1 for spec in OPERATORS.values())
    assert len(registry_sha256()) == 64


def test_text_initialized_rows_reproduce_the_text_forward_exactly_and_receive_gradients():
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    rows.reset_to_text(model.port.embedding_weight)
    row = graph_row(0, bindings['compact-value']['slot_token_ids'])
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    spans = resolve_spans(row, example.plan, rows.bindings)
    assert [span.kind for span in spans] == ['prompt', 'slots']
    text = model(example.plan)['h_final']
    learned = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))['h_final']
    assert torch.equal(text, learned)
    from natlang_neuralese.data.native_graph import graph_loss_from_result
    result = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))
    loss, _ = graph_loss_from_result(model, example, result)
    loss.backward()
    for kind in ('prompt', 'slots'):
        grad = rows.rows('compact-value', kind).grad
        assert grad is not None and float(grad.abs().sum()) > 0
    assert rows.rows('extract', 'slots').grad is None


def test_resolve_spans_rejects_wrong_tokens_and_slots_outside_production():
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    row = graph_row(0, bindings['compact-value']['slot_token_ids'], prompt=(10, 11, 13))
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match='differ from the bound'):
        resolve_spans(row, example.plan, rows.bindings)
    row = graph_row(0, bindings['compact-value']['slot_token_ids'])
    row['nodes'][0]['producing'] = []
    row['nodes'][1]['imports'] = []
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    with pytest.raises(ValueError, match='producing span'):
        resolve_spans(row, example.plan, rows.bindings)


def test_rows_save_and_load_round_trip(tmp_path):
    rows = OperatorRows(small_bindings(), 8)
    with torch.no_grad():
        rows.rows('extract', 'slots').normal_()
    save_rows(rows, tmp_path / 'rows.pt')
    loaded = load_rows(tmp_path / 'rows.pt')
    assert torch.equal(loaded.rows('extract', 'slots'), rows.rows('extract', 'slots'))
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
            rows.rows(name, 'slots').add_(0.3 * torch.randn_like(rows.rows(name, 'slots')))
    slots = bindings['compact-value']['slot_token_ids']
    raw = [graph_row(index, slots) for index in range(4)]
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


def test_serving_native_step_site_matches_the_training_graph_capture():
    """The operator endpoint's incremental site (native_step with substituted rows) equals the graph producer."""
    from natlang_neuralese.model.shared_state import capture_block, concatenate_captures, native_step
    from natlang_neuralese.model.learned_operators import ResolvedSpan
    model = tiny_model()
    bindings = small_bindings()
    rows = OperatorRows(bindings, model.port.embedding_weight.shape[1])
    rows.reset_to_text(model.port.embedding_weight)
    with torch.no_grad():
        rows.rows('compact-value', 'slots').add_(0.2 * torch.randn_like(rows.rows('compact-value', 'slots')))
        rows.rows('compact-value', 'prompt').add_(0.2 * torch.randn_like(rows.rows('compact-value', 'prompt')))
    row = graph_row(0, bindings['compact-value']['slot_token_ids'])
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    spans = resolve_spans(row, example.plan, rows.bindings)
    identity = {'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64, 'embedding_weight_sha256': 'c' * 64,
                'architecture': model.port.config.model_type}
    producer = row['nodes'][0]
    start, stop = producer['producing'][0]
    with torch.no_grad():
        result = model(example.plan, embeddings=operator_embeddings(model, rows, example.plan, spans))
        graph_block = capture_block(result, 'producer', start, stop, text='', **identity)
        ids = torch.tensor([producer['token_ids']])
        site = [ResolvedSpan('compact-value', span.kind, 'site', span.start, span.stop) for span in spans]
        embeddings = rows.apply(model.embed(ids), site)
        prefix = native_step(model, ids[:, :start], embeddings=embeddings[:, :start])
        capture = native_step(model, ids[:, start:stop], prefix['cache'], producing=True,
                              embeddings=embeddings[:, start:stop])
    fields = concatenate_captures([capture])
    for graph_layer, step_layer in zip(graph_block.layers, fields['layers']):
        for key, value in graph_layer.items():
            if key != 'kind':
                torch.testing.assert_close(step_layer[key].float(), value.float(), rtol=1e-4, atol=1e-5)


def test_runtime_consumer_frames_equal_the_registry():
    import json as _json
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / 'ts-host/src/neuralese/operator.ts').read_text()
    for name, spec in OPERATORS.items():
        line = next(line for line in source.splitlines() if line.strip().startswith(f"'{name}':"))
        assert f'open: {_json.dumps(spec.consumer_open)}'.replace('"', "'") in line, name
        assert f'close: {_json.dumps(spec.consumer_close)}'.replace('"', "'") in line, name
