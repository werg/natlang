"""JOLT+ guidance never carries the answer, and a guided success that leaks its guidance never trains in student form
(rl/privileged_context.py case_facts answers, rl/jolt.py guidance_leak; DECISIONS 2026-11-18)."""
import json

from natlang_neuralese.rl.jolt import guidance_controls, guidance_leak, guidance_texts
from natlang_neuralese.rl.privileged_context import answer_leaves, carries, case_facts
from test_jolt_privileged_context import custody, train_case


def test_reference_facts_withhold_the_returned_value_by_default_and_show_opts_out():
    case = train_case(semantics={**train_case()['semantics'], 'expected': 27})
    case['curriculum']['reference']['root'][-1][1]['value'] = 27
    case['curriculum']['decisive'] = ['the total is 27', 'count rows in A1']
    _, facts = case_facts(case, custody=custody())
    reference = next(f for f in facts if f.kind == 'reference_solution')
    assert '27' not in reference.text and 'return_result with the result the steps above computed' in reference.text
    assert 'count(a1)' in reference.text                                   # the procedure stays
    decisive = [f.text for f in facts if f.kind == 'decisive']
    assert decisive == ['Decisive evidence: count rows in A1']             # a procedure fact holding the answer is out
    assert any(f.kind == 'expected_result' for f in facts)                 # answer kinds stay facts (teachers never get them)
    _, shown = case_facts(case, custody=custody(), answers='show')
    assert '27' in next(f for f in shown if f.kind == 'reference_solution').text


def test_answer_leaves_and_carries_read_numbers_and_words_as_hintcheck_does():
    case = train_case(semantics={**train_case()['semantics'], 'expected': {'verdict': 'fail', 'n': 1, 'total': 12.5}})
    assert answer_leaves(case) == ['12.5', '2', '5', 'fail']               # reference/child values too; 1 is too common
    assert carries('Return fail.', ['fail']) == ['fail']
    assert carries('a failure mode', ['fail']) == []
    assert carries('1. read A1\n2. add the rows', ['2']) == []             # list markers and identifiers are not numbers
    assert carries('the sum is 12.50', ['12.5']) == ['12.5']


def test_teacher_controls_name_the_withheld_values():
    controls = guidance_controls({}, role='teacher', custody_receipt={}, facts=[], withheld=['fail'])
    assert controls['x_natlang_guidance']['withheld'] == ['fail']
    assert 'withheld' not in guidance_controls({}, role='teacher', custody_receipt={}, facts=[])['x_natlang_guidance']


def writer_turn(value):
    return {'messages': [], 'target': {'tool_calls': [{'function': {'name': 'return_result', 'arguments': json.dumps(
        {'status': 'success', 'value': value})}}]}, 'sampling': {'guidance_policy': {'function': 'writeGuidance'}}}


def consumer_turn(reasoning, content='', prompt='Judge the claim from evidence.json.'):
    return {'messages': [{'role': 'user', 'content': prompt}],
            'target': {'reasoning_content': reasoning, 'content': content,
                       'tool_calls': [{'function': {'name': 'eval', 'arguments': '{"code":"readEvidenceFile()"}'}}]},
            'sampling': {'prompt': [{'kind': 'tokens', 'token_ids': [1, 2]},
                                    {'kind': 'tokens', 'token_ids': [7, 8, 9], 'privileged': True}]}}


GUIDE = 'Open evidence.json first, then compare each cohort separately before you decide the verdict.'


def test_a_reply_that_quotes_or_names_its_removed_guidance_does_not_train():
    clean = {'turns': [writer_turn(GUIDE), consumer_turn('I will read the evidence file and compare cohorts.')]}
    assert guidance_leak(clean) == []
    quoted = {'turns': [writer_turn(GUIDE), consumer_turn('So: compare each cohort separately before you decide.')]}
    assert guidance_leak(quoted) == ['quoted']
    named = {'turns': [writer_turn(GUIDE), consumer_turn('The guidance mentions cohorts.')]}
    assert guidance_leak(named) == ['named']
    # words the visible prompt itself uses are no quote and no naming
    own = {'turns': [writer_turn(GUIDE), consumer_turn('Per the guidance section: compare each cohort separately before you decide.',
                                                       prompt='Guidance section: compare each cohort separately before you decide.')]}
    assert guidance_leak(own) == []
    # tool-call code that follows the procedure is the point of a hint
    assert guidance_leak({'turns': [writer_turn('Call readEvidenceFile() then compare cohorts one by one carefully.'),
                                    consumer_turn('Reading now.')]}) == []


def test_guidance_that_carried_the_answer_does_not_train_unless_the_prompt_shows_it():
    leaky = {'turns': [writer_turn('Correct result: fail. Check the draft.'), consumer_turn('Checking the draft.')]}
    assert guidance_leak(leaky, answers=['fail']) == ['answer']
    shown = {'turns': [writer_turn('Decide pass or fail from the draft.'),
                       consumer_turn('Checking the draft.', prompt='Return pass or fail.')]}
    assert guidance_leak(shown, answers=['fail']) == []


def test_text_fallback_guidance_is_read_from_the_privileged_tokens():
    row = {'turns': [consumer_turn('so compare each cohort separately before you decide now')]}
    assert guidance_texts(row) == []
    decode = {(7, 8, 9): GUIDE}.get
    assert guidance_texts(row, decode=lambda ids: decode(tuple(ids))) == [GUIDE]
    assert guidance_leak(row, decode=lambda ids: decode(tuple(ids))) == ['quoted']
