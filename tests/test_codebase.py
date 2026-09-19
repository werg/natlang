"""Code bases, calls and locals (spec/CODEBASES.md)."""
from pathlib import Path

import pytest

from natlang import gbnf
from natlang.codebase import load_function
from natlang.diag import Reject
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.types import TypeEnv, format_type
from natlang.values import dump

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


def test_undeclared_recursion_is_refused_at_load(tmp_path):
    (tmp_path / "f.nl").write_text("---\nargs: {x: Num}\nreturns: Num\nuses: {f: ./f}\n---\nCall f again.\n")
    with pytest.raises(Reject) as e:
        load_function(tmp_path / "f.nl")
    assert "undeclared-recursion" in str(e.value)
    (tmp_path / "g.nl").write_text("---\nargs: {x: Num}\nreturns: Num\nuses: {g: ./g}\nrecursive: true\nmax_depth: 4\n---\nCall g.\n")
    assert "g" in load_function(tmp_path / "g.nl").codebase
