"""Constrained decoding in the model's native call text: the grammar built from the tool schemas."""
import keyword

from natlang import gbnf
from natlang.native import CALL_OPEN, call_body_grammar, call_grammar, parse_calls
from natlang.surface import ToolSurface

from test_surface import _session

S = ToolSurface()
REC = "{ customer: Text, order_id: Text, amount: Num, phone?: Text }"


def _g(name, **kw):
    doc, s = _session(name)
    return s, call_grammar(S.tools(s), **kw)


def ok(g, text):
    return gbnf.accepts(g, CALL_OPEN + text)


def test_a_typed_write_is_accepted_and_ill_typed_ones_are_not():
    s, g = _g("02-leaf-extraction.yaml")
    good = f"[write(path=\"return\", type=\"{REC}\", value={{'customer': 'Dana Whitfield', 'order_id': '0077', 'amount': 42.5}})]"
    assert ok(g, good)
    assert ok(g, good.replace("42.5}", "42.5, 'phone': '555-0100'}"))                  # optional field
    assert ok(g, good.replace("42.5}", "42.5, 'phone': None}"))                        # "does not apply"
    assert not ok(g, good.replace("42.5}", "42.5, 'phone': ''}"))                      # an invented empty value
    assert not ok(g, good.replace("42.5", "'lots'"))                                   # Num must be a number
    assert not ok(g, good.replace("'order_id': '0077', ", ""))                         # required field missing
    assert not ok(g, good.replace("42.5}", "42.5, 'vip': True}"))                      # unknown field
    assert not ok(g, good.replace('"return"', '"args/note"'))                          # args are read-only
    assert not ok(g, good.replace(REC, "Text"))                                        # the type is not the model's to get wrong
    assert ok(g, '[write(path="return/amount", type="Num", value=42.5)]')              # path, type and value belong together
    assert not ok(g, '[write(path="return/amount", type="Num", value="42.5")]')
    assert not ok(g, '[write(path="return/amount", type="Text", value="42.5")]')


def test_paths_enums_and_ranges_are_limited_to_what_exists():
    s, g = _g("06-map-with-rubric.yaml")
    assert ok(g, '[read(path="args/rubric")]') and ok(g, '[read(path="args")]')
    assert not ok(g, '[read(path="args/secret")]')
    assert ok(g, "[write(path=\"return\", type=\"Label[]\", value=['billing', 'spam'])]")
    assert not ok(g, "[write(path=\"return\", type=\"Label[]\", value=['refund'])]")   # not a Label
    assert ok(g, '[read(path="args/tickets", start=0, end=4)]')                        # five tickets: items 0..4
    assert not ok(g, '[read(path="args/tickets", start=1, end=5)]')
    assert not ok(g, '[launch(missiles=True)]')


def test_several_calls_and_replies():
    s, g = _g("01-leaf-judgment.yaml")
    assert ok(g, '[read(path="args/message"), write(path="return", type="Bool", value=True)]')
    assert gbnf.accepts(g, "It is a complaint, so I wrote true.")
    assert gbnf.accepts(g, "")
    assert not gbnf.accepts(call_grammar(S.tools(s), allow_reply=False), "It is a complaint.")
    assert not gbnf.accepts(call_grammar(S.tools(s), allow_reply=False), "")
    assert gbnf.accepts(call_grammar([], allow_reply=True), "checkpoint note")


def test_parse_calls_and_argument_names():
    assert parse_calls('[write(path="return", type="Bool", value=True)]') == \
        [("write", {"path": "return", "type": "Bool", "value": True})]
    assert parse_calls("<|tool_call_start|>[read(path='a', start=1, end=2), run(paths=['x', 'y'])]<|tool_call_end|>") == \
        [("read", {"path": "a", "start": 1, "end": 2}), ("run", {"paths": ["x", "y"]})]
    doc, s = _session("06-map-with-rubric.yaml")
    for t in S.tools(s):
        for arg in t["function"]["parameters"]["properties"]:
            assert not keyword.iskeyword(arg), (t["function"]["name"], arg)


def test_call_body_grammar_checks_every_visible_byte_after_marker():
    _, session = _session("02-leaf-extraction.yaml")
    grammar = call_body_grammar(S.tools(session))
    good = "\n[write(path='return', type='{ customer: Text, order_id: Text, amount: Num, phone?: Text }', " \
           "value={'customer': 'Dana Whitfield', 'order_id': '0077', 'amount': 42.5})]"
    bad = good.replace("{'customer':", "{customer:")
    assert gbnf.accepts(grammar, good)
    assert not gbnf.accepts(grammar, bad)


def test_probability_trace_keeps_ids_without_treating_empty_token_as_call(monkeypatch):
    from natlang.decoder import Generation
    from natlang.native import NativeCallDecoder
    trace = []
    dec = NativeCallDecoder(probability_log=trace)
    monkeypatch.setattr(dec, 'render', lambda messages, tools: 'prompt')
    details = [{'id': 7, 'token': '', 'logprob': -0.1, 'top_logprobs': []}]
    monkeypatch.setattr(dec, 'generate', lambda *a, **kw: Generation('Done.', [[('', 0.9)]],
                                                                  completion_tokens=1, token_details=details))
    dec.chat([], [], temperature=0)
    assert dec.stats['p_call_first'] == [None]
    assert trace == [{'text': 'Done.', 'tokens': details}]


def test_reasoning_native_deliberates_before_guided_action(monkeypatch):
    from natlang.decoder import Generation
    from natlang.native import ReasoningNativeCallDecoder
    dec = ReasoningNativeCallDecoder(reasoning_tokens=99)
    monkeypatch.setattr(dec, 'render', lambda messages, tools: 'PROMPT')
    generated = []
    def generate(prompt, **kwargs):
        generated.append((prompt, kwargs))
        if len(generated) == 1:
            return Generation('check the requested Boolean', stopped='</think>', completion_tokens=5)
        if len(generated) == 2:
            return Generation('\n', stopped=CALL_OPEN, completion_tokens=1)
        return Generation('[write(path="return", type="Bool", value=True)]', completion_tokens=7)
    monkeypatch.setattr(dec, 'generate', generate)
    _, session = _session('01-leaf-judgment.yaml')
    turn = dec.chat([], S.tools(session), temperature=.2)
    assert generated[0][0] == 'PROMPT<think>'
    assert generated[0][1]['grammar'] is None and generated[0][1]['stop'] == ['</think>']
    assert generated[1][0] == 'PROMPT<think>check the requested Boolean</think>'
    assert generated[1][1]['grammar'] is None and generated[1][1]['stop'] == [CALL_OPEN]
    assert generated[2][0] == 'PROMPT<think>check the requested Boolean</think>\n' + CALL_OPEN
    assert generated[2][1]['grammar']
    assert turn.calls == [('write', {'path':'return', 'type':'Bool', 'value':True})]
    assert turn.completion_tokens == 13
    assert turn.raw_response['choices'][0]['message']['reasoning_content'] == 'check the requested Boolean'
