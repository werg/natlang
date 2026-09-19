import math

import pytest

from natlang import gbnf
from natlang.decoder import ChatTurn
from natlang.gen.policy import native_text
from natlang.native import call_grammar, written_value_confidence
from natlang.runtime import Runtime, Session
from natlang.tool_agent import ToolAgent, REVIEW_TOOLS
from natlang.types import TypeEnv
from natlang.values import load_program, MISSING


class Scripted:
    def __init__(self, turns, inspect=None):
        self.turns = iter(turns)
        self.requests = []
        self.inspect = inspect

    def chat(self, messages, tools, **kwargs):
        self.requests.append(list(messages))
        if self.inspect:
            self.inspect(messages, tools)
        return next(self.turns)


def proposal(calls=None):
    calls = calls or [('write', {'path': 'return', 'type': 'Num', 'value': 17})]
    return ChatTurn(calls, completion_tokens=1,
                    value_confidence=[None] * (len(calls) - 1) + [{'geometric_mean': 0.1}])


def verdict(decision):
    return ChatTurn([('review_write', {'decision': decision, 'reason': 'Checked the original requested value.'})], completion_tokens=1)


def root_program():
    return load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'effects': ['out.emit'], 'instructions': 'Return 17.'}})


def test_approval_resumes_original_history_and_applies_original_write():
    root = root_program()
    def inspect(messages, tools):
        if tools[0]['function']['name'] == 'review_write':
            assert root.ret is MISSING
            assert 'Are you sure' in messages[-1]['content']
    dec = Scripted([proposal(), verdict('approve'), ChatTurn(text='Done.', completion_tokens=1)], inspect)
    proposals, reviews = [], []
    out, value = Runtime(lambda lam: ToolAgent(dec, careful_threshold=0.5, proposals=proposals, reviews=reviews)).run_root(root)
    assert out.kind == 'done' and value == 17
    assert len(reviews) == 1 and proposals[0]['released']
    assert not any('Are you sure' in m.get('content', '') for m in dec.requests[2])
    assert not any('review_write' in str(m.get('tool_calls', [])) for m in dec.requests[2])


@pytest.mark.parametrize('decision', ['error', 'blocker'])
def test_declining_review_prevents_whole_batch_including_prior_effect_call(decision):
    root = root_program()
    calls = [('run_code', {'code': 'fx.out.emit(1); 1'}), ('write', {'path': 'return', 'type': 'Num', 'value': 17})]
    dec = Scripted([proposal(calls), verdict(decision)])
    rt = Runtime(lambda lam: ToolAgent(dec, careful_threshold=0.5))
    out, _ = rt.run_root(root)
    assert out.kind == 'quiesced' and out.detail.startswith('careful review ' + decision)
    assert root.ret is MISSING and rt.emitted == []


@pytest.mark.parametrize('review', [ChatTurn(text='Yes!', completion_tokens=1),
                                   ChatTurn([('write', {'path': 'return', 'value': 99})], completion_tokens=1)])
def test_unstructured_or_replacement_review_never_applies_write(review):
    root = root_program()
    dec = Scripted([proposal(), review])
    out, _ = Runtime(lambda lam: ToolAgent(dec, careful_threshold=0.5)).run_root(root)
    assert out.kind == 'quiesced' and root.ret is MISSING


def test_review_respects_turn_budget_and_unknown_confidence_is_not_zero():
    root = root_program()
    dec = Scripted([proposal()])
    out, _ = Runtime(lambda lam: ToolAgent(dec, careful_threshold=0.5, max_turns=1)).run_root(root)
    assert out.kind == 'quiesced' and root.ret is MISSING and len(dec.requests) == 1
    root = root_program()
    p = proposal()
    p.value_confidence = [None]
    dec = Scripted([p, ChatTurn(text='Done.', completion_tokens=1)])
    out, val = Runtime(lambda lam: ToolAgent(dec, careful_threshold=0.5)).run_root(root)
    assert out.kind == 'done' and val == 17 and len(dec.requests) == 2


def test_review_grammar_offers_four_verdicts_and_exactly_one_call():
    grammar = call_grammar(REVIEW_TOOLS, allow_reply=False, single_call=True)
    for decision in ('approve', 'withdraw', 'error', 'blocker'):
        call = ('review_write', {'decision': decision, 'reason': 'Because of the instructions.'})
        assert gbnf.accepts(grammar, native_text([call]))
        assert not gbnf.accepts(grammar, native_text([call, call]))
    assert not gbnf.accepts(grammar, 'Yes, sure.')


def test_confidence_aligns_utf8_literal_values_not_path_or_tool_tokens():
    text = "[write(path='return', type='Text', value='café'),\n edit(path='let/x', old='a', new='β')]"
    raw = text.encode()
    ranges = [(raw.index(v), raw.index(v) + len(v)) for v in ("'café'".encode(), "'β'".encode())]
    details = [{'bytes': [b], 'logprob': -0.1 if any(a <= i < z for a, z in ranges) else -20}
               for i, b in enumerate(raw)]
    scores = written_value_confidence(text, details)
    assert len(scores) == 2
    assert all(s['geometric_mean'] == pytest.approx(math.exp(-0.1)) for s in scores)
    assert [s['field'] for s in scores] == ['value', 'new']
    assert written_value_confidence(text, details[:-1]) == []
    details[ranges[0][0]]['logprob'] = None
    assert written_value_confidence(text, details)[0] is None


@pytest.mark.parametrize('ty,value,expected', [('Num', '17', 17), ('Text', 17, '17'),
                                              ('Text', '"17"', '"17"'),
                                              ('{ value: Num }', {'value': 17}, {'value': 17})])
def test_quote_coercion_and_real_value_records(ty, value, expected):
    root = load_program({'$lambda': {'type': f'Lambda<{{}}, {ty}>', 'instructions': 'Return supplied value.'}})
    session = Session(Runtime(None), root, TypeEnv())
    assert session.apply('write', {'path': 'return', 'type': ty, 'value': value}).kind == 'ok'
    assert root.ret == expected


@pytest.mark.parametrize('value', [{'value': 17}, '{"value":17}'])
def test_object_wrappers_never_coerce_to_scalars(value):
    root = root_program()
    session = Session(Runtime(None), root, TypeEnv())
    assert session.apply('write', {'path': 'return', 'type': 'Num', 'value': value}).kind == 'rejected'
    assert root.ret is MISSING


def test_threshold_is_fitted_without_using_held_out_labels():
    from scripts.confidence_probe import calibrate_threshold
    def row(group, values):
        return {'group': group, 'values': [{'confidence': p, 'label': label} for p, label in values]}
    rows = [row(0, [(0.2, 'semantic_mismatch'), (0.8, 'correct')]),
            row(1, [(0.4, 'semantic_mismatch'), (0.9, 'correct')])]
    fit = calibrate_threshold(rows)
    assert fit['threshold'] == pytest.approx(0.5)
    assert fit['held_out_odd_groups']['flagged_bad'] == 1
    rows[1] = row(1, [(0.1, 'correct'), (0.99, 'semantic_mismatch')])
    assert calibrate_threshold(rows)['threshold'] == fit['threshold']


def test_review_reason_can_precede_verdict_in_guided_response():
    from natlang.tool_agent import review_tools
    grammar = call_grammar(review_tools('reason_first'), allow_reply=False, single_call=True)
    call = ('review_write', {'reason': 'The required evidence is absent.', 'decision': 'blocker'})
    assert gbnf.accepts(grammar, native_text([call]))
    assert not gbnf.accepts(grammar, native_text([('review_write', {'decision': 'approve', 'reason': 'Yes.'})]))


def test_withdrawal_bubbles_by_default_without_applying_batch():
    root = root_program()
    dec = Scripted([proposal(), verdict('withdraw')])
    out, _ = Runtime(lambda lam: ToolAgent(dec, careful_threshold=.5)).run_root(root)
    assert out.detail.startswith('careful review withdraw:') and root.ret is MISSING


def test_withdrawal_allows_only_one_reconsideration_and_no_reviewer_reason_leaks():
    root = root_program()
    dec = Scripted([proposal(), verdict('withdraw'), proposal(), verdict('withdraw')])
    proposals = []
    out, _ = Runtime(lambda lam: ToolAgent(dec, careful_threshold=.5, withdrawal_policy='retry', proposals=proposals)).run_root(root)
    assert out.detail.startswith('careful review withdraw:') and root.ret is MISSING
    assert len(proposals) == 2 and all(not p['released'] for p in proposals)
    assert not any('Checked the original' in str(m) for m in dec.requests[2])
    assert 'withdrawn before execution' in dec.requests[2][-1]['content']


def test_fresh_proposal_after_withdrawal_can_succeed():
    root = root_program()
    dec = Scripted([proposal(), verdict('withdraw'), proposal(), verdict('approve'), ChatTurn(text='Done', completion_tokens=1)])
    out, val = Runtime(lambda lam: ToolAgent(dec, careful_threshold=.5, withdrawal_policy='retry')).run_root(root)
    assert out.kind == 'done' and val == 17


@pytest.mark.parametrize('call', [('call', {'function':'size_of', 'to':'return/size'}),
                                  ('mark_done', {'start':1}),
                                  ('write', {'path':'return','type':'Num','source':'let/size'})])
def test_structural_review_needs_no_value_logprobs(call):
    root = root_program()
    reviews = []
    dec = Scripted([ChatTurn([call], completion_tokens=1), verdict('withdraw')])
    out, _ = Runtime(lambda lam: ToolAgent(dec, review_scope='actions', reviews=reviews)).run_root(root)
    assert out.detail.startswith('careful review withdraw:') and root.ret is MISSING
    assert reviews[0]['trigger'] == 'structural' and reviews[0]['confidence'] is None


def test_instruction_reminder_is_verbatim_and_only_in_review_fork():
    from natlang.tool_agent import review_messages
    history = [{'role':'system','content':'Interpreter'}, {'role':'user','content':'If enabled, return 7. Otherwise fail.'},
               {'role':'tool','content':'enabled = false'}]
    for variant in ('repeat_instructions', 'checklist'):
        fork = review_messages(history, [('write', {'path':'return','value':7})], 0, variant)
        assert history[1]['content'] in fork[-1]['content']
        assert fork[:-1] == history and len(history) == 3
        assert 'withdraw' in fork[-1]['content']
    assert 'Original program instructions' not in review_messages(history, [], 0)[-1]['content']
