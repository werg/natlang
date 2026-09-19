"""Distinguish repairing an execution mistake from rewriting an impossible task."""
import pytest

from natlang.decoder import ChatTurn
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import load_program


class Scripted:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def chat(self, messages, tools, **kw):
        self.requests.append(list(messages))
        return next(self.turns)


def action(name, **args):
    return ChatTurn([(name, args)], completion_tokens=5)


def reply():
    return ChatTurn(text='Finished.', completion_tokens=1)


@pytest.mark.parametrize('instructions', ['Return the number 7.', 'Return the text "seven" unchanged; do not convert it to a number.'])
@pytest.mark.parametrize('policy', ['local', 'caller'])
def test_repair_is_not_proof_of_following_instructions(instructions, policy):
    root = load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'instructions': instructions}})
    dec = Scripted([action('write', path='return', type='Text', value='seven'),
                    action('write', path='return', type='Num', value=7), reply()])
    out, value = Runtime(lambda lam: ToolAgent(dec, validation_feedback=policy)).run_root(root)
    if policy == 'local':
        # Same structurally valid repair: correct for the first task, contradictory
        # to the second. Runtime type checking alone cannot distinguish them.
        assert out.kind == 'done' and value == 7 and len(dec.requests) == 3
    else:
        assert out.kind == 'quiesced' and 'validation failed:' in out.detail
        assert len(dec.requests) == 1


@pytest.mark.parametrize('policy', ['local', 'caller'])
def test_incomplete_return_feedback_is_optional_even_with_closed_marks(policy):
    root = load_program({'$lambda': {'type': 'Lambda<{}, { count: Num, evidence: Text }>',
                                    'instructions': 'Return count 3 and supplied evidence.'}})
    root.marks = {1: 'done'}
    dec = Scripted([action('write', path='return/count', type='Num', value=3), reply(),
                    action('write', path='return/evidence', type='Text', value='invented'), reply()])
    out, value = Runtime(lambda lam: ToolAgent(dec, validation_feedback=policy)).run_root(root)
    if policy == 'caller':
        assert out.kind == 'quiesced' and 'return/evidence' in out.detail
        assert len(dec.requests) == 2
    else:
        assert out.kind == 'done' and value['evidence'] == 'invented'
        assert any('return/evidence' in m.get('content', '') for m in dec.requests[-1])


def test_caller_policy_allows_incremental_construction_before_completion():
    root = load_program({'$lambda': {'type': 'Lambda<{}, { a: Num, b: Num }>', 'instructions': 'Return a=1, b=2.'}})
    dec = Scripted([action('write', path='return/a', type='Num', value=1),
                    action('write', path='return/b', type='Num', value=2), reply()])
    out, value = Runtime(lambda lam: ToolAgent(dec, validation_feedback='caller')).run_root(root)
    assert out.kind == 'done' and value == {'a': 1, 'b': 2}


def test_child_failure_reaches_caller_without_replaying_prior_effects():
    root = load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'effects': ['out.emit'],
        'instructions': 'Call worker. If it fails, return -1; otherwise return its value.',
        'codebase': {'worker': {'args': {}, 'returns': 'Num', 'effects': ['out.emit'],
            'instructions': 'Emit 1, then return the text "bad" unchanged.'}}}})
    child = Scripted([action('run_code', code='fx.out.emit(1); 1'),
                      action('write', path='return', type='Text', value='bad')])
    parent = Scripted([action('call', function='worker', to='let/result'),
                       action('write', path='return', type='Num', value=-1), reply()])
    rt = Runtime(lambda lam: ToolAgent(parent if lam is root else child, validation_feedback='caller'))
    out, value = rt.run_root(root)
    assert out.kind == 'done' and value == -1 and rt.emitted == [1]
    assert len(child.requests) == 2
    assert any('validation failed:' in m.get('content', '') for m in parent.requests[1])


@pytest.mark.parametrize('policy', ['local', 'caller'])
def test_explicit_error_reaches_caller_and_stops_remaining_actions(policy):
    root = load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'effects': ['out.emit'],
        'instructions': 'Call worker; on failure return -1.',
        'codebase': {'worker': {'args': {}, 'returns': 'Num', 'effects': ['out.emit'],
            'instructions': 'Emit 1, then return impossible text.'}}}})
    child = Scripted([action('run_code', code='fx.out.emit(1); 1'),
                     ChatTurn([('report_error', {'message': 'Required text conflicts with Num return.'}),
                               ('run_code', {'code': 'fx.out.emit(2); 2'})], completion_tokens=5)])
    parent = Scripted([action('call', function='worker', to='let/result'),
                       action('write', path='return', type='Num', value=-1), reply()])
    rt = Runtime(lambda lam: ToolAgent(parent if lam is root else child, validation_feedback=policy))
    out, value = rt.run_root(root)
    assert out.kind == 'done' and value == -1 and rt.emitted == [1]
    assert len(child.requests) == 2
    assert any('error: Required text conflicts' in m.get('content', '') for m in parent.requests[1])


def test_error_grammar_and_empty_diagnostic_rejection():
    from natlang import gbnf
    from natlang.gen.policy import native_text
    from natlang.native import call_grammar
    from natlang.runtime import Session
    from natlang.surface import ToolSurface
    from natlang.types import TypeEnv
    root = load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'instructions': 'Return impossible text.'}})
    session = Session(Runtime(None), root, TypeEnv())
    surface = ToolSurface()
    assert gbnf.accepts(call_grammar(surface.tools(session)), native_text([
        ('report_error', {'message': 'Required text conflicts with Num return.'})]))
    assert surface.apply(session, 'report_error', {'message': ''}).kind == 'rejected'
    assert not session.completed
    assert 'report_error' not in [t['function']['name'] for t in ToolSurface(error_tool=False).tools(session)]
