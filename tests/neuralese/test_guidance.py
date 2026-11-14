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
    line = "positiveLines = positiveLines.filter(f);\\n"
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


def test_runs_inside_a_line_and_redeclarations():
    run = "<|tool_call_start|>[eval(code='const s = 1;\\nconst t = " + "l.line_amount||" * 5
    (reason, at), _ = _first_rejection(Guide(Settings()), run)
    assert reason == "repetition" and run[at:].startswith("const t")
    twice = "<|tool_call_start|>[eval(code='const p = a.filter(f);\\nconst p = p.filter(g);\\nreturn p;')]"
    (reason, at), _ = _first_rejection(Guide(Settings()), twice)
    assert reason == "redeclaration" and twice[at:].startswith("const p = p")
    scoped = "<|tool_call_start|>[eval(code='for (const x of xs) {\\n  const y = x;\\n}\\nfor (const x of ys) {\\n  const y = x;\\n}\\nreturn 1;')]"
    assert _first_rejection(Guide(Settings()), scoped) == (None, None)


def test_call_names_inside_quoted_code_are_not_transport_tools():
    reply = "<|tool_call_start|>[eval(code='return [Math.max(1, Math.min(2, 3))];')]<|tool_call_end|>"
    assert _first_rejection(Guide(Settings(tools=["eval"])), reply) == (None, None)
    reply = "<|tool_call_start|>[eval(code='const s = \"[bogus()]\"; return s;'), evaluate()]"
    verdict, _ = _first_rejection(Guide(Settings(tools=["eval"])), reply)
    assert verdict == ("unknown-tool", reply.index("evaluate"))


def test_open_reasoning_is_not_checked_and_verdicts_stay_after_it():
    """Reasoning the generation prompt opened is not an action (fork a6401bbd3): a call written inside it is not
    checked, and every verdict after it is a reply offset past its closing tag."""
    thought = "Maybe <tool_call>\n<function=bogus>\n<parameter=code>\nconst a = 1;\nconst a = 2;\n"
    reply = thought + "</think>\n\n<tool_call>\n<function=nope>\n"
    settings = dict(tools=["eval"], call_format="xml", require_call=True)
    verdict, _ = _first_rejection(Guide(Settings(**settings, reasoning_open=True)), reply)
    assert verdict == ("unknown-tool", reply.index("nope"))
    closed = Guide(Settings(**settings, reasoning_open=True))
    assert closed.action_start(thought) is None and closed.action_start(reply) == reply.index("</think>") + 8
    # Not opened by the prompt: the same text is checked from its first character.
    verdict, _ = _first_rejection(Guide(Settings(**settings)), reply)
    assert verdict == ("unknown-tool", reply.index("bogus"))
    # The envelope still requires a call right after the reasoning.
    prose = thought + "</think>\n\nDone."
    verdict, _ = _first_rejection(Guide(Settings(**settings, reasoning_open=True)), prose)
    assert verdict == ("no-call", prose.index("</think>") + 8)
