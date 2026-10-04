import pytest

from natlang_neuralese.serve.guidance import Guide, Settings, code_strings

pytest.importorskip("tree_sitter_typescript")


def _first_rejection(guide, reply):
    for end in range(1, len(reply) + 1):
        verdict = guide.check(reply[:end])
        if verdict:
            return verdict, end
    return None, None


def test_good_code_passes_token_by_token():
    reply = ("<|tool_call_start|>[eval(code='const s = state as any;\\nconst x = s.a.filter((l:any)=>l.amount>0);"
             "\\nif (x.length) {\\n  return x;\\n}\\nreturn [];')]<|tool_call_end|>")
    assert _first_rejection(Guide(Settings(tools=["eval"])), reply) == (None, None)


def test_a_wrong_line_is_rejected_at_its_start_once_complete():
    reply = "<|tool_call_start|>[eval(code='const s = state as any;\\nconst x = s.a.filter((l:any)=>l.amount>0;\\nreturn x;')]"
    (reason, at), end = _first_rejection(Guide(Settings()), reply)
    assert reason == "syntax" and reply[at:].startswith("const x") and reply[:end].endswith("\\n")


def test_repetition_unknown_tools_and_decoding():
    line = "const positiveLines = positiveLines.filter(f);\\n"
    reply = "<|tool_call_start|>[eval(code='const a = 1;\\n" + line * 4
    (reason, at), _ = _first_rejection(Guide(Settings(repeat=3)), reply)
    assert reason == "repetition" and at == reply.index(line) + 2 * len(line)
    assert Guide(Settings(tools=["eval"])).check("<|tool_call_start|>[evaluate(") == ("unknown-tool", 20)
    code, = code_strings("[eval(code='a = \"x\\\\n\";\\nb')]")
    assert code.text == 'a = "x\\n";\nb' and code.closed


def test_accepted_points_are_not_rejected_again():
    guide = Guide(Settings(tools=["eval"]))
    verdict = guide.check("[evaluate(")
    guide.accept(verdict[1])
    assert guide.check("[evaluate(") is None
