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
