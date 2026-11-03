"""The guidance learned operator, seed schedule /2 and guidance cases (fallow register C6)."""
import dataclasses
import glob
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT), str(Path(__file__).resolve().parent)]
LFM3 = glob.glob(str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/*'))

from natlang_neuralese.model import learned_operators as ops  # noqa: E402


def test_guidance_prompt_is_the_writer_instruction_text_the_runtime_renders():
    source = (ROOT / 'applications/guidance-writer/writeGuidance.nl').read_text()
    body = re.split(r'^---\n', source, flags=re.M)[2]
    assert ops.OPERATORS['guidance'].prompt + '.' == body.strip('\n')
    spec = ops.OPERATORS['guidance']
    assert spec.placement == 'call' and spec.seed_cap == 512
    from natlang_neuralese.rl.privileged_context import PrivilegedRendering
    assert spec.consumer_open == PrivilegedRendering.header + '\n'


def test_call_site_replaces_the_single_prompt_occurrence():
    prompt = ops.OPERATORS['guidance'].prompt
    messages = [{'role': 'system', 'content': 'shared'},
                {'role': 'user', 'content': f'You are inside this call: writeGuidance(...)\n\nInstructions:\n{prompt}\n\nIn eval'}]
    part = {'type': 'native_state', 'id': 'x'}
    site = ops.operator_site_messages('guidance', instructions='', source=messages, prompt_part=part)
    assert site[0] == messages[0]
    assert [item.get('type') for item in site[1]['content']] == ['text', 'native_state', 'text']
    assert site[1]['content'][0]['text'].endswith('Instructions:\n') and site[1]['content'][2]['text'] == '\n\nIn eval'
    assert ops.operator_site_messages('guidance', instructions='', source=messages) == messages
    with pytest.raises(ValueError, match='exactly once'):
        ops.operator_site_messages('guidance', instructions='', source=[{'role': 'user', 'content': 'no prompt'}])


def _example(close_weight=1.):
    from test_learned_operators import FINGERPRINT, graph_row
    from natlang_neuralese.data.native_graph_schema import parse_graph
    row = graph_row(2)
    return row, parse_graph(row, tokenizer_fingerprint=FINGERPRINT)


def test_seed_schedule_v2_keeps_the_close_marker_and_v1_keeps_its_history():
    row, example = _example()
    producer = row['nodes'][0]
    start, stop = producer['producing'][0]
    v2 = ops.seed_schedule({'start': 1.0, 'end': 0.1, 'updates': 10})
    assert v2['schema'] == ops.SEED_SCHEDULE_V2 == ops.SEED_SCHEDULE
    scaled = ops.seed_weighted(example, 0.25, v2)
    expected = [w * (0.25 if start <= i < stop else 1.) for i, w in enumerate(producer['target_weights'])]
    assert list(scaled.target_weights[:len(expected)]) == expected and scaled.target_weights[stop] == 1.
    v1 = ops.seed_schedule({'schema': ops.SEED_SCHEDULE_V1, 'start': 1.0, 'end': 0.1, 'updates': 10})
    old = ops.seed_weighted(example, 0.25, v1)
    assert old.target_weights[stop] == 0.25 and list(old.target_weights[start:stop]) == [0.25] * (stop - start)
    with pytest.raises(ValueError):
        ops.seed_schedule({'schema': 'natlang.operator-seed-schedule/9', 'start': 1, 'end': 0, 'updates': 1})


def test_seed_schedule_transition_records_the_schema_change_and_keeps_the_body_anneal():
    v2 = ops.seed_schedule(ops.DEFAULT_SEED_SCHEDULE)
    fresh, none = ops.seed_schedule_transition(None, v2, 40)
    assert fresh['adopted_at_global_step'] == 40 and none is None
    saved = {**ops.seed_schedule({**ops.DEFAULT_SEED_SCHEDULE, 'schema': ops.SEED_SCHEDULE_V1}), 'adopted_at_global_step': 12}
    applied, transition = ops.seed_schedule_transition(saved, v2, 90)
    assert applied['schema'] == ops.SEED_SCHEDULE_V2 and applied['adopted_at_global_step'] == 12
    assert transition['from']['schema'] == ops.SEED_SCHEDULE_V1 and transition['at_global_step'] == 90
    legacy = {key: value for key, value in saved.items() if key != 'schema'}
    assert ops.seed_schedule_transition(legacy, v2, 90)[1]['from']['schema'] is None
    carried, unchanged = ops.seed_schedule_transition({**v2, 'adopted_at_global_step': 7}, v2, 90)
    assert carried['adopted_at_global_step'] == 7 and unchanged is None
    changed, moved = ops.seed_schedule_transition({**v2, 'updates': 5, 'adopted_at_global_step': 7}, v2, 90)
    assert changed['adopted_at_global_step'] == 90 and moved['to']['updates'] == 2000


def test_new_operator_joins_carried_rows_without_touching_them():
    import torch
    from test_learned_operators import CharTokenizer, FINGERPRINT
    embedding = torch.randn(200000, 4)
    tokenizer = CharTokenizer()
    names = sorted(set(ops.OPERATORS) - {'guidance'})
    rows = ops.OperatorRows.text_initialized(embedding, tokenizer, tokenizer_fingerprint=FINGERPRINT, names=names)
    before = {key: value.detach().clone() for key, value in rows.state_dict().items()}
    model = type('M', (), {})()
    model.operator_rows = rows
    model.input_rows = torch.zeros(1, dtype=torch.float32)
    model.port = type('P', (), {'embedding_weight': embedding})()
    record = ops.attach_operator_rows(model, tokenizer, tokenizer_fingerprint=FINGERPRINT)
    assert record['added'] == ['guidance'] and record['parameter_names'] == ['interface.operator_rows.prompt.guidance']
    assert all(torch.equal(rows.state_dict()[key], value) for key, value in before.items())
    ids = torch.tensor(rows.bindings['guidance']['prompt_token_ids'])
    assert torch.equal(rows.rows('guidance'), embedding[ids])
    assert {name for name, _ in rows.named_parameters()} >= {'prompt.guidance'}
    assert ops.attach_operator_rows(model, tokenizer, tokenizer_fingerprint=FINGERPRINT)['added'] == []


def test_mechanical_guidance_seed_is_input_specific_and_skips_feedback():
    from natlang_neuralese.data.operator_seeds import guidance_seed
    facts = [{'id': 'a', 'kind': 'expected_result', 'text': 'The total is 41.'},
             {'id': 'b', 'kind': 'attempt_feedback', 'text': 'Earlier attempt returned 40.'},
             {'id': 'c', 'kind': 'reference_solution', 'text': 'Sum the amounts column.'}]
    seed = guidance_seed(facts)
    assert seed['provenance'] == 'mechanical:guidance-facts'
    assert seed['text'] == '1. A correct result: The total is 41.\n2. A way to get there: Sum the amounts column.'
    other = guidance_seed([{**facts[0], 'text': 'The total is 7.'}])
    assert other['text'] != seed['text']
    with pytest.raises(ValueError):
        guidance_seed([facts[1]])


def _rollout(prompt, facts, *, writer_reply, consumer='task/root'):
    from natlang_neuralese.rl.privileged_context import BLOCK_SCHEMA
    listing = '\n'.join(fact.text for fact in facts)
    writer_messages = [{'role': 'system', 'content': 'shared system'},
                       {'role': 'user', 'content': f'You are inside this call: writeGuidance(consumer, facts)\n\n'
                                                   f'Instructions:\n{prompt}\n\nconsumer {consumer}\nfacts\n{listing}'}]
    receipt = {'schema': BLOCK_SCHEMA, 'invocation': consumer, 'role': 'root', 'mode': 'write-only', 'insert_at': 3,
               'facts': [{'id': fact.id, 'kind': fact.kind, 'sha256': fact.digest} for fact in facts],
               'state': {'id': 'nzs1_x'}, 'token_count': 0}
    return {'id': 'r1', 'task_id': 't1', 'turns': [
        {'invocation_id': 'w1', 'graph_turn': 'w1#turn1', 'messages': writer_messages, 'tools': [],
         'target': {'role': 'assistant', 'content': writer_reply},
         'sampling': {'guidance_policy': {'function': 'writeGuidance', 'sha256': 'e' * 64}, 'policy': {'id': 'p'}}},
        {'invocation_id': consumer, 'graph_turn': consumer + '#turn1',
         'messages': [{'role': 'system', 'content': 'shared system'}, {'role': 'user', 'content': 'Sum the column.'}],
         'tools': [], 'target': {'role': 'assistant', 'content': '41'}, 'sampling': {'privileged': receipt}}]}


def test_cases_from_rollout_links_the_text_writer_and_requires_a_checked_outcome():
    from natlang_neuralese.data.guidance_cases import cases_from_rollout, check_case
    from natlang_neuralese.rl.privileged_context import Fact
    facts = [Fact('f1', 'expected_result', 'The total is 41.'), Fact('f2', 'reference_solution', 'Sum amounts.')]
    rollout = _rollout(ops.OPERATORS['guidance'].prompt, facts, writer_reply='1. Return 41.\n2. Sum amounts.')
    cases, omitted = cases_from_rollout(rollout, facts, outcome={'success': True, 'evidence': {'oracle': 'exact'}},
                                        split_groups=['world:1'])
    assert omitted == [] and len(cases) == 1
    case = check_case(cases[0])
    assert case['guidance']['text'] == '1. Return 41.\n2. Sum amounts.' and case['guidance']['provenance'] == 'checked'
    assert [fact['id'] for fact in case['writer']['facts']] == ['f1', 'f2']
    assert case['consumers'][0]['target']['value']['content'] == '41'
    assert cases_from_rollout(rollout, facts, outcome={'success': False}, split_groups=['w'])[0] == []
    tampered = [Fact('f1', 'expected_result', 'The total is 40.'), facts[1]]
    assert cases_from_rollout(rollout, tampered, outcome={'success': True, 'evidence': {'x': 1}},
                              split_groups=['w'])[1][0]['reason'] == 'receipt-facts-unverified'
    blank = _rollout(ops.OPERATORS['guidance'].prompt, facts, writer_reply='')
    cases, _ = cases_from_rollout(blank, facts, outcome={'success': True, 'evidence': {'x': 1}}, split_groups=['w'])
    assert cases[0]['guidance'] is None


@pytest.mark.skipif(not LFM3, reason='LFM3 tokenizer not cached')
def test_guidance_case_renders_the_real_site_operator_spans_and_engine_import():
    from scripts.neuralese_data.build_operator_graphs import make_renderer
    from natlang_neuralese.data.guidance_cases import cases_from_rollout
    from natlang_neuralese.data.native_graph_schema import parse_graph
    from natlang_neuralese.rl.privileged_context import Fact
    renderer = make_renderer(32768)
    prompt = ops.OPERATORS['guidance'].prompt
    facts = [Fact('f1', 'expected_result', 'The total is 41.'), Fact('f2', 'reference_solution', 'Sum amounts.')]
    rollout = _rollout(prompt, facts, writer_reply='1. Return 41.\n2. Sum amounts.')
    case = cases_from_rollout(rollout, facts, outcome={'success': True, 'evidence': {'oracle': 'exact'}},
                              split_groups=['world:1'])[0][0]
    graph, baseline = renderer.render(case, 'guidance', corpus_id='unit', source_row_sha256='0' * 64, source_ref={})
    example = parse_graph(graph, tokenizer_fingerprint=renderer.fingerprint)
    spans = ops.resolve_spans(graph, example.plan, renderer.bindings)
    assert [span.kind for span in spans] == ['prompt', 'seed']
    producer, consumer = graph['nodes']
    # The site with the prompt placeholder tokenizes exactly as the plain recorded call (exact in-call positions).
    plain, _ = renderer._render(case['writer']['messages'], None, {}, generation=True)
    assert producer['token_ids'][:len(plain)] == plain
    start, stop = producer['producing'][0]
    assert producer['token_ids'][start - 1] == renderer.role_ids[0] and producer['token_ids'][stop] == renderer.role_ids[1]
    assert [i for i, w in enumerate(producer['target_weights']) if w] == list(range(start, stop + 1))
    assert renderer.tokenizer.decode(producer['token_ids'][start:stop]) == '1. Return 41.\n2. Sum amounts.'
    edge = consumer['imports'][0]
    assert consumer['token_ids'][edge['at'] - 1] == renderer.role_ids[2] and consumer['token_ids'][edge['at']] == renderer.role_ids[3]
    rendering = renderer.privileged_rendering()
    turn, offset = rendering.state_turn()
    insert = edge['at'] - offset
    assert consumer['token_ids'][insert:insert + len(turn)] == turn
    assert rendering.insertion_point(consumer['token_ids'][:insert] + consumer['token_ids'][insert + len(turn):])[0] == insert
    assert graph['operator_conversion']['seeds'][0]['provenance'] == 'checked'
    assert 'The total is 41.' in renderer.tokenizer.decode(baseline['nodes'][0]['token_ids'])
    mechanical = {**case, 'guidance': None}
    graph, _ = renderer.render(mechanical, 'guidance', corpus_id='unit', source_row_sha256='1' * 64, source_ref={})
    assert graph['operator_conversion']['seeds'][0]['provenance'] == 'mechanical:guidance-facts'
