import copy

import pytest

from scripts.coreweave.teacher_tranche_admission import verify_teacher_action_accounting


def captured_actions():
    turns = [{'invocation_id': 'call/1', 'request_sha256': 'a' * 64,
              'raw_response_sha256': 'b' * 64, 'model_response': {'raw_calls': []},
              'assistant': {'content': content}} for content in ('', 'correct', 'failed')]
    rows = {'source': {'id': 'source', 'trajectory': turns}}
    decisions = {('source', index): {'approved': index != 2,
                 'reason': 'failed proposal' if index == 2 else None,
                 'invocation_id': 'call/1', 'raw_response_sha256': 'b' * 64}
                 for index in range(3)}
    empty = 'no original provider action or nonempty assistant reply'
    omissions = [dict(source_row_id='source', teacher_turn_index=0, reason=empty,
                      decision={**decisions['source', 0], 'reason': empty}),
                 dict(source_row_id='source', teacher_turn_index=2, reason='failed proposal',
                      decision=copy.deepcopy(decisions['source', 2]))]
    receipt = {'turns': 1, 'deduplication': {'emitted_rows': 1, 'omitted_rows': 0},
               'action_omissions': omissions}
    summary = {'eligible_decisions': 2, 'training_decisions': 3}
    return rows, decisions, receipt, summary


def test_real_admission_filter_conserves_approved_empty_provider_reply():
    inputs = captured_actions()
    before = copy.deepcopy(inputs)
    proof = verify_teacher_action_accounting(*inputs)
    assert inputs == before
    assert proof['source_actions'] == 3
    assert proof['materializer_approved_actions'] == 2
    assert proof['converted_eligible_actions'] == 1
    assert proof['omitted_actions'] == 2
    assert proof['post_materializer_rejections'] == inputs[2]['action_omissions'][:1]


def test_legacy_equal_count_and_exact_dedup_remain_valid():
    rows, decisions, receipt, summary = captured_actions()
    rows['source']['trajectory'][0]['assistant']['content'] = 'another valid response'
    receipt['turns'] = 2
    receipt['deduplication'] = {'emitted_rows': 1, 'omitted_rows': 1}
    receipt['action_omissions'].pop(0)
    proof = verify_teacher_action_accounting(rows, decisions, receipt, summary)
    assert proof['converted_eligible_actions'] == proof['materializer_approved_actions'] == 2
    assert proof['post_materializer_rejections'] == []


@pytest.mark.parametrize('mutation', [
    'summary', 'missing_decision', 'integer_approval', 'unknown_decision',
    'missing_omission', 'duplicate_omission', 'changed_reason', 'changed_decision',
    'typed_payload', 'unverified_rendering', 'wrong_turns', 'wrong_dedup', 'bool_count',
])
def test_rejects_forged_or_lost_action_accounting(mutation):
    rows, decisions, receipt, summary = captured_actions()
    if mutation == 'summary': summary['eligible_decisions'] = 1
    elif mutation == 'missing_decision': del decisions['source', 2]
    elif mutation == 'integer_approval': decisions['source', 0]['approved'] = 1
    elif mutation == 'unknown_decision': decisions['other', 0] = decisions.pop(('source', 0))
    elif mutation == 'missing_omission': receipt['action_omissions'].pop()
    elif mutation == 'duplicate_omission': receipt['action_omissions'].append(receipt['action_omissions'][0])
    elif mutation == 'changed_reason': receipt['action_omissions'][0]['reason'] = 'waived'
    elif mutation == 'changed_decision': receipt['action_omissions'][0]['decision']['approved'] = False
    elif mutation == 'typed_payload': receipt['action_omissions'][0]['teacher_turn_index'] = False
    elif mutation == 'unverified_rendering':
        receipt['action_omissions'].append(dict(source_row_id='source', teacher_turn_index=1,
                                               reason='unverified rendering', decision=decisions['source', 1]))
    elif mutation == 'wrong_turns': receipt['turns'] = 2
    elif mutation == 'wrong_dedup': receipt['deduplication']['omitted_rows'] = 1
    elif mutation == 'bool_count': receipt['deduplication']['emitted_rows'] = True
    with pytest.raises(ValueError):
        verify_teacher_action_accounting(rows, decisions, receipt, summary)


def captured_block_omission():
    import hashlib
    rows, decisions, receipt, summary = captured_actions()
    turn = rows['source']['trajectory'][1]
    block_id, body = 'nz1_exact-captured-block', 'source-bound body'
    turn['context'] = [{'role': 'user', 'content': [{'type': 'neuralese', 'id': block_id}]}]
    turn['tools_offered'] = []
    turn['model_response']['transport_provenance'] = {
        'version': 'text-marker-standin/2', 'learned_vectors': False,
        'qualification_certificate': False, 'training_admission': False,
        'raw_request_sha256': 'a' * 64, 'rendered_request_sha256': 'b' * 64,
        'rendered_messages': [{'role': 'user', 'content': 'captured input lacking its block expansion'}],
        'expanded_input_blocks': [{'id': block_id, 'type': 'Neuralese<string>', 'body': body,
            'body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'learned_vectors': False}]}
    omission = {'source_row_id': 'source', 'teacher_turn_index': 1,
        'unresolved_scope': 'prompt', 'unresolved_block_ids': [block_id],
        'reason': 'captured provider-visible messages do not contain the exact authenticated block expansion',
        'decision': copy.deepcopy(decisions['source', 1])}
    receipt['action_omissions'].append(omission)
    receipt['turns'] = 0
    receipt['deduplication'] = {'emitted_rows': 0, 'omitted_rows': 0}
    return rows, decisions, receipt, summary


def test_actual_renderer_preparation_authenticates_missing_captured_block_omission():
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import render_teacher_packet
    rows, decisions, receipt, summary = captured_block_omission()
    turn = rows['source']['trajectory'][1]
    # This actual renderer branch precedes tokenizer use; none is constructed.
    packet, omitted = render_teacher_packet(None, turn['context'],
        {'role': 'assistant', 'content': 'correct', 'tool_calls': []}, turn['tools_offered'], turn)
    assert packet is None
    assert omitted == {k: v for k, v in receipt['action_omissions'][-1].items()
                       if k not in ('source_row_id', 'teacher_turn_index', 'decision')}
    proof = verify_teacher_action_accounting(rows, decisions, receipt, summary)
    assert proof['source_actions'] == proof['omitted_actions'] == 3
    assert proof['converted_eligible_actions'] == 0
    assert proof['post_materializer_rejections'][-1] == receipt['action_omissions'][-1]


@pytest.mark.parametrize('tamper', ['scope', 'block_id', 'decision', 'reason', 'lost',
                                  'source_context', 'capture_repaired', 'expanded_body'])
def test_render_omission_needs_exact_original_capture_and_complete_payload(tamper):
    import json
    rows, decisions, receipt, summary = captured_block_omission()
    turn = rows['source']['trajectory'][1]
    omission = receipt['action_omissions'][-1]
    if tamper == 'scope': omission['unresolved_scope'] = 'target'
    elif tamper == 'block_id': omission['unresolved_block_ids'] = ['other']
    elif tamper == 'decision': omission['decision']['approved'] = False
    elif tamper == 'reason': omission['reason'] = 'arbitrary renderer failure'
    elif tamper == 'lost': receipt['action_omissions'].pop()
    elif tamper == 'source_context': turn['context'] = [{'role': 'user', 'content': 'ordinary input'}]
    elif tamper == 'expanded_body':
        turn['model_response']['transport_provenance']['expanded_input_blocks'][0]['body'] = 'changed'
    else:
        transport = turn['model_response']['transport_provenance'];block = transport['expanded_input_blocks'][0]
        transport['rendered_messages'][0]['content'] = (
            f'[[Neuralese text block id={block["id"]} type={block["type"]}; exact JSON string body='
            f'{json.dumps(block["body"], ensure_ascii=False, separators=(",", ":"))}]]')
    with pytest.raises(ValueError):
        verify_teacher_action_accounting(rows, decisions, receipt, summary)
