"""Code bases, calls and locals (spec/CODEBASES.md)."""
from pathlib import Path

import pytest

from natlang import gbnf
from natlang.codebase import load_function
from natlang.diag import Reject
from natlang.gen.policy import native_text
from natlang.host import instantiate, load
from natlang.native import call_grammar
from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.types import TypeEnv, format_type
from natlang.values import coerce, dump

ROOT = Path(__file__).resolve().parent.parent
TRIAGE = ROOT / "examples" / "triage" / "main.nl"
S = ToolSurface()
TICKETS = ["I was charged twice.", "CHEAP WATCHES!!!", "PRODUCTION IS DOWN."]
GOLD = {"classify": lambda a: "spam" if "WATCHES" in a["ticket"] else "billing" if "charged" in a["ticket"] else "technical",
        "is_urgent": lambda a: "DOWN" in a["ticket"] and "calm" not in a.get("__body", ""),
        "shorten": lambda a: " ".join(a["text"].split()[: max(5, len(a["text"].split()) // 2)])}


class Scripted:
    """Leaves answer from GOLD; `summarize` and `main` follow their pseudocode, as a good interpreter would."""
    def __init__(self, lam, log):
        self.lam, self.log = lam, log

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error"), r.text
        self.log.append((self.lam.fn_name, name))
        return r

    def run(self, session):
        f, d = self.lam.fn_name, lambda n, a: self.do(session, n, a)
        if f in GOLD:
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": GOLD[f](self.lam.in_)})
        elif f == "summarize":
            d("write", {"path": "let/draft", "type": "Text", "value": "word " * 150})
            d("call", {"function": "shorten", "to": "return", "init": "let/draft", "until": "is_short", "max": 3})
        else:
            d("call", {"function": "classify", "to": "let/labels", "over": "args/tickets", "inputs": {"rubric": "args/rubric"}})
            flags = d("run_code", {"code": "locals.labels.map(l => l !== 'spam')"}).value
            d("write", {"path": "let/not_spam", "type": "Bool[]", "value": flags})
            d("call", {"function": "select_by_flags", "to": "let/real", "inputs": {"items": "args/tickets", "flags": "let/not_spam"}})
            d("call", {"function": "is_urgent", "to": "let/flags", "over": "let/real"})
            d("call", {"function": "select_by_flags", "to": "let/urgent", "inputs": {"items": "let/real", "flags": "let/flags"}})
            d("call", {"function": "summarize", "to": "return/summary", "inputs": {"tickets": "let/urgent"}})
            d("call", {"function": "count_true", "to": "return/urgent", "inputs": {"flags": "let/flags"}})
            d("call", {"function": "group_count", "to": "return/by_label", "inputs": {"values": "let/labels"}})
        assert session.finish()


def _root():
    return load(TRIAGE, {"tickets": TICKETS, "rubric": "billing: charges\n"})


def test_loader_scopes_lexically_and_shares_definitions():
    fn = load_function(TRIAGE)
    assert set(fn.codebase) == {"classify", "is_urgent", "summarize", "count_true", "select_by_flags", "group_count"}
    assert set(fn.codebase["summarize"].codebase) == {"shorten", "is_short"}        # its own folder only
    assert fn.codebase["classify"].types["Label"].startswith('"billing"')          # types are inherited downwards
    assert fn.codebase["count_true"].kind == "code"


def test_codebase_and_local_count_are_not_language_limits(tmp_path):
    root = tmp_path / "main.nl"
    root.write_text("---\nreturns: Num\n---\nCount the values.\n")
    folder = tmp_path / "main"
    folder.mkdir()
    for i in range(24):
        (folder / f"helper_{i}.nl").write_text("---\nreturns: Num\n---\nReturn one.\n")
    assert len(load_function(root).codebase) == 24
    session = _session()
    for i in range(24):
        assert session.apply("write", {"path": f"let/value_{i}", "type": "Num", "value": i}).kind == "ok"
    assert len(session.lam.let_types) == 24


def test_the_triage_code_base_runs_and_every_call_is_grammatical():
    log = []
    rt = Runtime(lambda lam: Scripted(lam, log))
    out, value = rt.run_root(_root())
    assert out.kind == "done", out.detail
    v = dump(value)
    assert v["urgent"] == 1 and v["by_label"] == {"billing": 1, "spam": 1, "technical": 1}
    assert len(v["summary"].split()) <= 60
    assert ("summarize", "call") in log and rt.episodes_started == 9


def _session():
    root = _root()
    return Session(Runtime(lambda lam: Scripted(lam, [])), root, TypeEnv())


def test_there_are_no_anonymous_lambdas():
    s = _session()
    for ty in ("Task<Text>", "Code<Num>", "Map<Text, Bool>", "Lambda<{}, Text>"):
        r = s.apply("write", {"path": "let/x", "type": ty, "value": {"instructions": "do it"}})
        assert r.kind == "rejected" and "anonymous-lambda" in r.text
    assert "x" not in s.lam.let_types                                     # a refused write creates no local
    assert s.apply("call", {"function": "invent", "to": "let/x"}).kind == "rejected"


def test_calls_are_checked_against_the_signature():
    s = _session()
    assert "bad-call" in s.apply("call", {"function": "classify", "to": "let/l", "inputs": {"ticket": "args/rubric"}}).text
    assert "unknown-field" in s.apply("call", {"function": "is_urgent", "to": "let/u", "inputs": {"nope": "args/rubric"}}).text
    r = s.apply("call", {"function": "is_urgent", "to": "let/u", "over": "args/rubric"})          # not a list
    assert r.kind == "rejected" and "u" not in s.lam.let_types


def test_calls_accept_typed_literal_values_and_reject_double_binding():
    s = _session()
    args = {"function": "select_by_flags", "to": "let/selected",
            "values": {"items": ["a", "b"], "flags": [True, False]}}
    assert gbnf.accepts(call_grammar(S.tools(s)), native_text([("call", args)]))
    result = S.apply(s, "call", args)
    assert result.kind == "done", result.text
    assert dump(s.lam.let["selected"]) == ["a"]
    overlap = S.apply(s, "call", {"function": "select_by_flags", "to": "let/nope",
                                   "inputs": {"items": "args/tickets"},
                                   "values": {"items": ["a"], "flags": [True]}})
    assert overlap.kind == "rejected" and "bad-call" in overlap.text


def test_call_can_update_destination_from_its_previous_value():
    step = load_function(ROOT / "codebases" / "dependency_plan" / "plan.nl").codebase["step"]
    tasks = [{"id": "a", "needs": [], "description": "first"}]
    root = instantiate(step)
    root.in_["state"] = coerce({"tasks": tasks, "order": [], "blocked": ["a"],
                                 "finished": False}, root.type.params.get("state")[0],
                                root.env(TypeEnv()), yaml=False, path="args/state")
    s = Session(Runtime(lambda lam: None), root, TypeEnv())
    assert S.apply(s, "write", {"path": "let/state", "type": "State",
                                "source": "args/state"}).kind == "ok"
    assert S.apply(s, "write", {"path": "let/chosen", "type": "Text", "value": "a"}).kind == "ok"
    result = S.apply(s, "call", {"function": "advance", "to": "let/state",
                                 "inputs": {"state": "let/state", "chosen": "let/chosen"}})
    assert result.kind == "done", result.text
    assert dump(s.lam.let["state"])["order"] == ["a"]


def test_editing_means_copying_into_a_local_first():
    s = _session()
    assert s.apply("edit", {"path": "codebase/is_urgent", "old": "urgent", "new": "calm"}).kind == "rejected"
    assert s.apply("write", {"path": "let/strict", "type": "Function<is_urgent>"}).kind == "ok"
    r = s.apply("edit", {"path": "let/strict/instructions", "old": "Requests, suggestions and cosmetic problems are not urgent.",
                         "new": "Only outages are urgent."})
    assert r.kind == "ok", r.text
    seen = []

    class Peek(Scripted):
        def run(self, session):
            seen.append(self.lam.body)
            return super().run(session)
    s.rt.agent_factory = lambda lam: Peek(lam, [])
    assert s.apply("call", {"function": "let/strict", "to": "let/flags", "over": "args/tickets"}).kind == "done"
    assert all("Only outages" in b for b in seen) and len(seen) == 3
    assert "Only outages" not in s.lam.codebase["is_urgent"].body          # the code base is immutable


def test_call_again_resumes_what_did_not_finish():
    s = _session()
    fail = {"on": True}

    class Flaky(Scripted):
        def run(self, session):
            if fail["on"] and "WATCHES" in self.lam.in_.get("ticket", ""):
                return "could not decide"
            return super().run(session)
    runs = []
    s.rt.agent_factory = lambda lam: (runs.append(1), Flaky(lam, []))[1]
    assert s.apply("call", {"function": "is_urgent", "to": "let/flags", "over": "args/tickets"}).kind == "quiesced"
    fail["on"] = False
    n = len(runs)
    assert s.apply("call", {"function": "is_urgent", "to": "let/flags"}).kind == "done"
    assert len(runs) == n + 1                                              # only the failed item ran again


def test_recursion_is_refused_at_load(tmp_path):
    (tmp_path / "f.nl").write_text("---\nargs: {x: Num}\nreturns: Num\nuses: {g: ./g}\n---\nCall g.\n")
    (tmp_path / "g.nl").write_text("---\nargs: {x: Num}\nreturns: Num\nuses: {f: ./f}\n---\nCall f.\n")
    with pytest.raises(Reject) as e:
        load_function(tmp_path / "f.nl")
    assert "recursion" in str(e.value) and "f -> g -> f" in str(e.value)


def test_a_lambda_with_locals_and_a_code_base_survives_swap_out():
    from natlang.values import dump_state, load_program
    s = _session()
    s.rt.agent_factory = lambda lam: Scripted(lam, [])
    assert s.apply("call", {"function": "is_urgent", "to": "let/flags", "over": "args/tickets"}).kind == "done"
    back = load_program(dump_state(s.lam))
    assert back.let == {"flags": [False, False, True]} and set(back.codebase) == set(s.lam.codebase)
    assert set(back.codebase["summarize"].codebase) == {"shorten", "is_short"}
    s2 = Session(Runtime(lambda lam: Scripted(lam, [])), back, TypeEnv())        # and the run continues from there
    assert s2.apply("call", {"function": "count_true", "to": "return/urgent", "inputs": {"flags": "let/flags"}}).kind == "done"


def test_record_semicolons_and_nested_type_aliases(tmp_path):
    from natlang.codebase import read_type_aliases
    from natlang.types import parse_type, format_type
    source = '// type Ignored = Bad;\nexport type A = { nested: { value: Text; }; note?: "a;b"; };\ntype B = A[];'
    aliases = read_type_aliases(source)
    assert list(aliases) == ['A', 'B']
    assert format_type(parse_type(aliases['A'])) == '{ nested: { value: Text }, note?: "a;b" }'
    (tmp_path / 'types.ts').write_text(source)
    (tmp_path / 'f.nl').write_text('---\nargs:\n  input: A\nreturns: A\n---\nReturn input unchanged.\n')
    assert load_function(tmp_path / 'f.nl').types == aliases
    with pytest.raises(ValueError, match='unterminated'):
        read_type_aliases('type A = { value: Text;')
    with pytest.raises(ValueError, match='duplicate'):
        read_type_aliases('type A = Text; type A = Num;')
