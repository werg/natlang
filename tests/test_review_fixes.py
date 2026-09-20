"""Regressions for the independent review: boundaries, effects and measurement."""
import json
import shutil
import time
from pathlib import Path

import pytest

from natlang import js
from natlang.checks import grade
from natlang.corpus import program_id, split_programs
from natlang.decoder import ChatTurn
from natlang.host import load
from natlang.invocation import RunOptions
from natlang.runtime import Runtime, Session, MAX_TOOL_CALLS
from natlang.tool_agent import ToolAgent
from natlang.types import TypeEnv
from natlang.values import load_program, MISSING
from scripts.eval_turns import select_samples, score, summary

ROOT = Path(__file__).resolve().parent.parent


def session(options=None):
    lam = load_program({'$lambda': {'type': 'Lambda<{}, Num>', 'instructions': 'return 1'}})
    return Session(Runtime(None, options=options), lam, TypeEnv())


def test_holdout_reserves_every_turn_of_a_program():
    pairs = [{'id': f'family-{i}-{j}'} for i in range(20) for j in range(5)]
    held, train, manifest = split_programs(pairs, holdout=11, seed=4)
    assert len(held) == 15
    assert {program_id(p) for p in held}.isdisjoint(program_id(p) for p in train)
    assert (held, train, manifest) == split_programs(pairs, 11, 4)
    assert program_id({'id': 'judge-0-0', 'program_id': '71:judge:0'}) == '71:judge:0'
    with pytest.raises(ValueError, match='No training'):
        split_programs(pairs, 100)


def sample(i):
    return {'id': f'judge-{i}-0', 'family': 'judge', 'skill': 'write',
            'messages': [], 'tools': [], 'target': {'tool_calls': []}}


def test_evaluation_manifest_reuses_exact_samples_and_refuses_drift(tmp_path):
    data, manifest = tmp_path / 'ref.jsonl', tmp_path / 'manifest.json'
    rows = [sample(i) for i in range(50)]
    data.write_text('\n'.join(map(json.dumps, rows)))
    picked, identity = select_samples(data, manifest, per_cell=7, seed=5)
    data.write_text('\n'.join(map(json.dumps, reversed(rows))))
    assert select_samples(data, manifest, per_cell=2, seed=100) == (picked, identity)
    changed = {**picked[0], 'messages': ['changed']}
    data.write_text('\n'.join(json.dumps(changed if r['id'] == changed['id'] else r) for r in rows))
    with pytest.raises(ValueError, match='changed'):
        select_samples(data, manifest)


def test_evaluation_handles_empty_corpus_and_separates_replies(tmp_path):
    data = tmp_path / 'empty.jsonl'
    data.write_text('')
    with pytest.raises(ValueError, match='empty'):
        select_samples(data, tmp_path / 'manifest.json')
    assert score(sample(1), ChatTurn())['reply']
    assert summary([]) == {'n': 0, 'exact': 0, 'right_tool': 0}


def test_effect_grade_requires_the_actual_emissions_and_all_checks():
    expected = {'value': 2, 'emitted': [{'id': 'a1'}, {'id': 'a3'}]}
    assert grade(expected, 'done', 2)[0] == '?'
    assert grade(expected, 'done', 2, emitted=[])[0] == 'no'
    assert grade(expected, 'done', 2, emitted=expected['emitted'])[0] == 'yes'
    assert grade({'value': 2, 'checks': [{'kind': 'crisp', 'code': 'value === 3'}]}, 'done', 2)[0] == 'no'


@pytest.mark.parametrize('done', [999, [2, 1], [1, 999], True, [1, 2, 3]])
def test_bad_mark_cannot_partially_commit_a_write(done):
    s = session()
    args = {'path': 'return', 'type': 'Num', 'value': 1, 'done': done}
    assert s.apply('write', args).kind == 'rejected'
    assert s.lam.ret is MISSING and not s.lam.marks
    assert args['done'] == done


def test_bad_mark_cannot_execute_an_effectful_callee():
    root = load(ROOT / 'conformance/programs/14-eval-side-effect.yaml',
                {'alerts': [{'id': 'a1', 'level': 'critical'}]})
    rt = Runtime(None)
    s = Session(rt, root, TypeEnv())
    assert s.apply('call', {'function': 'emit_critical_alerts', 'to': 'return',
                            'inputs': {'alerts': 'args/alerts'}, 'done': 999}).kind == 'rejected'
    assert not rt.emitted and root.ret is MISSING


def test_marks_have_an_independent_call_budget():
    s = session(RunOptions(max_tool_calls=MAX_TOOL_CALLS))
    for _ in range(MAX_TOOL_CALLS):
        assert s.apply('mark_done', {'start': 1}).kind == 'ok'
    assert s.actions == 0
    assert s.apply('mark_done', {'start': 1}).kind == 'budget'


def test_default_episode_has_no_tool_call_limit():
    s = session()
    for _ in range(MAX_TOOL_CALLS + 2):
        assert s.apply('mark_done', {'start': 1}).kind == 'ok'
    assert s.tool_calls == MAX_TOOL_CALLS + 2


class RepeatingDecoder:
    def __init__(self, tokens=1):
        self.tokens, self.calls = tokens, []

    def chat(self, messages, tools, **kw):
        self.calls.append(kw)
        return ChatTurn([('mark_done', {'start': 1})], completion_tokens=self.tokens)


def test_model_turn_budget_counts_bookkeeping():
    dec, s = RepeatingDecoder(), session()
    assert 'budget' in ToolAgent(dec, max_turns=4).run(s)
    assert len(dec.calls) == 4 and s.actions == 0


def test_token_budget_caps_each_request():
    dec = RepeatingDecoder(tokens=3)
    assert 'budget' in ToolAgent(dec, max_tokens=6).run(session())
    assert [c['max_tokens'] for c in dec.calls] == [6, 3]


def test_default_has_no_separate_per_turn_token_cap():
    dec = RepeatingDecoder(tokens=1)
    assert 'budget' in ToolAgent(dec, max_tokens=1000, max_turns=1).run(session())
    assert dec.calls[0]['max_tokens'] == 1000


def test_default_tool_agent_has_no_turn_token_or_wall_clock_budget():
    dec = RepeatingDecoder(tokens=1)
    dec.deadline = None
    agent = ToolAgent(dec)
    assert agent.max_turns is None and agent.max_tokens is None and agent.max_seconds is None
    assert agent.turn_tokens is None
    assert 'budget' in ToolAgent(dec, max_turns=1).run(session())
    assert dec.calls[0]['max_tokens'] is None
    assert dec.deadline is None


def test_failed_effect_is_observed_not_silently_replayed():
    s = session()
    s.lam.effects = ['out.emit']

    class Decoder:
        turns = 0
        def chat(self, messages, tools, **kw):
            self.turns += 1
            if self.turns == 1:
                return ChatTurn([('run_code', {'code': 'fx.out.emit(1); throw new Error("after effect");'})],
                                completion_tokens=10)
            assert any(m['role'] == 'tool' and 'after effect' in m['content'] for m in messages)
            return ChatTurn([('report_blocker', {'missing': 'The effect happened but subsequent code failed.'})],
                            completion_tokens=10)
    dec = Decoder()
    assert 'blocked' in ToolAgent(dec).run(s)
    assert dec.turns == 2 and s.rt.emitted == [1]


@pytest.mark.parametrize('code, expected', [
    ('return countBy(["constructor", "__proto__", "constructor"]);', {'constructor': 2, '__proto__': 1}),
    ('return groupBy(["constructor", "__proto__"], x => x);', {'constructor': ['constructor'], '__proto__': ['__proto__']}),
    ('return indexBy(["constructor", "__proto__"], x => x);', {'constructor': 'constructor', '__proto__': '__proto__'}),
])
def test_dictionary_helpers_accept_every_string_key(code, expected):
    assert js.run(code, {'args': {}}, None, body=True, path='test') == expected


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js optional for TS syntax')
def test_typescript_annotations_are_stripped_before_sandbox_execution():
    code = 'type Row = {n: number}; const x: Row = {n: 3}; return x.n;'
    assert js.run(code, {'args': {}}, None, body=True, path='test') == 3
    assert js.run('const n: number = 4; n + 1', {'args': {}}, None, body=False, path='test') == 5


def test_effectful_infinite_loop_is_killed_after_preserving_host_effect(monkeypatch):
    monkeypatch.setattr(js, 'TIME_LIMIT_S', 0.5)
    calls = []
    started = time.monotonic()
    with pytest.raises(js.JsError, match='wall-clock'):
        js.run('fx.out.emit(7); while(true) {}', {'args': {}}, lambda c, f, a: calls.extend(a),
               body=True, path='test', effectful=True)
    assert calls == [7] and time.monotonic() - started < 3


def test_slow_host_effect_does_not_block_the_interpreter(monkeypatch):
    monkeypatch.setattr(js, 'TIME_LIMIT_S', 0.5)
    def slow(*args):
        time.sleep(1)
    with pytest.raises(js.JsError, match='may still complete'):
        js.run('fx.out.emit(7);', {'args': {}}, slow, body=True, path='test', effectful=True)


def test_native_decoder_reports_tokens_and_honors_deadline(monkeypatch):
    import io
    import urllib.request
    from natlang.decoder import LlamaServerDecoder
    requests = []
    def urlopen(req, timeout):
        requests.append((json.loads(req.data), timeout))
        return io.BytesIO(json.dumps({'content': 'ok', 'tokens_predicted': 2}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    dec = LlamaServerDecoder(timeout=300)
    dec.deadline = time.monotonic() + 1
    result = dec.generate('prompt', grammar=None, max_tokens=3, temperature=0, seed=0, stop=[])
    assert result.completion_tokens == 2 and dec.usage['completion_tokens'] == 2
    assert dec.usage['turns'] == 1 and 0 < requests[0][1] <= 1
    dec.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        dec.generate('prompt', grammar=None, max_tokens=3, temperature=0, seed=0, stop=[])
    assert len(requests) == 1


def test_server_decoder_cannot_override_remaining_token_allowance(monkeypatch):
    import io
    import urllib.request
    from natlang.decoder import LlamaServerDecoder
    def urlopen(req, timeout):
        assert json.loads(req.data)['max_tokens'] == 3
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'ok'}}],
                                     'usage': {'prompt_tokens': 17, 'completion_tokens': 2}}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    dec = LlamaServerDecoder(chat_extra={'max_tokens': 999})
    result = dec.chat([], [], temperature=0, max_tokens=3)
    assert result.completion_tokens == 2 and result.prompt_tokens == 17
    assert dec.usage['completion_tokens'] == 2


def test_server_decoder_omits_default_response_token_cap(monkeypatch):
    import io
    import urllib.request
    from natlang.decoder import LlamaServerDecoder
    def urlopen(req, timeout):
        assert 'max_tokens' not in json.loads(req.data)
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'ok'}}],
                                     'usage': {'completion_tokens': 2}}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    assert LlamaServerDecoder().chat([], [], temperature=0).text == 'ok'


def test_server_decoder_optional_timeout_still_honors_explicit_deadline():
    from natlang.decoder import LlamaServerDecoder
    dec = LlamaServerDecoder()
    with dec.request_scope(deadline=time.monotonic() + 5):
        assert 0 < dec.request_timeout() <= 5


def test_server_decoder_expands_typed_alternatives_and_maps_calls(monkeypatch):
    import io
    import urllib.request
    from natlang.decoder import LlamaServerDecoder
    tool = {"type": "function", "function": {"name": "call", "description": "Invoke a function",
            "parameters": {"type": "object", "properties": {},
                "x-natlang-alternatives": [{"function": {"const": "summarize"},
                    "to": {"const": "let/metrics"}, "inputs": {"type": "object",
                        "properties": {"candidates": {"type": "string"},
                                       "trials": {"type": "string"}},
                        "required": ["candidates", "trials"],
                        "additionalProperties": False}}]}}}
    def urlopen(req, timeout):
        payload = json.loads(req.data)
        variant = payload["tools"][0]["function"]
        assert variant["name"] == "call_alt_0"
        assert variant["parameters"]["properties"]["inputs"]["required"] == ["candidates", "trials"]
        assert "x-natlang-alternatives" not in variant["parameters"]
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "",
            "tool_calls": [{"id": "t1", "type": "function", "function": {
                "name": "call_alt_0", "arguments": json.dumps({"function": "summarize",
                    "to": "let/metrics", "inputs": {"candidates": "args/candidates",
                                                   "trials": "args/trials"}})}}]}}],
            "usage": {"completion_tokens": 12}}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    dec = LlamaServerDecoder(typed_alternatives=True)
    turn = dec.chat([], [tool], temperature=0)
    assert turn.calls == [("call", {"function": "summarize", "to": "let/metrics",
                                   "inputs": {"candidates": "args/candidates", "trials": "args/trials"}})]
    assert turn.raw_calls[0]["function"]["name"] == "call_alt_0"


def test_typed_chat_groups_same_value_type_without_losing_schema():
    from natlang.decoder import LlamaServerDecoder
    value = {"type": "object", "properties": {"id": {"type": "number"}},
             "required": ["id"], "additionalProperties": False}
    alts = [{"path": {"const": path}, "type": {"const": "Item"}, "value": value}
            for path in ("return/first", "return/second")]
    tool = {"type": "function", "function": {"name": "write", "description": "Write",
            "parameters": {"type": "object", "properties": {},
                           "x-natlang-alternatives": alts}}}
    offered = LlamaServerDecoder(typed_alternatives=True).presented_tools([tool])
    assert len(offered) == 1
    params = offered[0]["function"]["parameters"]
    assert params["properties"]["path"] == {"enum": ["return/first", "return/second"]}
    assert params["properties"]["type"] == {"const": "Item"}
    assert params["properties"]["value"] == value


def test_typed_chat_keeps_distinct_copy_source_sets_separate():
    from natlang.decoder import LlamaServerDecoder
    alts = [{"path": {"const": "return/first"}, "type": {"const": "Item"},
             "source": {"enum": ["args/a"]}},
            {"path": {"const": "return/second"}, "type": {"const": "Item"},
             "source": {"enum": ["args/b"]}}]
    tool = {"type": "function", "function": {"name": "write", "description": "Write",
            "parameters": {"type": "object", "properties": {},
                           "x-natlang-alternatives": alts}}}
    offered = LlamaServerDecoder(typed_alternatives=True).presented_tools([tool])
    assert len(offered) == 2
    assert [item["function"]["parameters"]["properties"]["source"]["enum"]
            for item in offered] == [["args/a"], ["args/b"]]
