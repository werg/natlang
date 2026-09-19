"""The moderation code base (codebases/moderation): conditionals with early return, and a second opinion obtained
the only way behaviour can be varied - copy a function into a local, edit the copy, call the copy."""
from pathlib import Path

from natlang import gbnf
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.types import format_type
from natlang.values import dump

ROOT = Path(__file__).resolve().parent.parent
S = ToolSurface()
POLICY = "1. No threats of violence.\n2. No advertising.\n3. Stay on topic: gardening.\n"
OLD, NEW = "When in doubt, answer true.", "Answer true only if the violation is unmistakable. When in doubt, answer false."
# post -> rule -> (lenient reading, strict reading, severity)
WORLD = {
    "My tomatoes are finally ripe!": {},
    "BUY CHEAP SEEDS at seedz.example": {"No advertising.": (True, True, "low")},
    "I will find you and hurt you, Tom.": {"No threats of violence.": (True, True, "high")},
    "Someone should really teach Tom a lesson.": {"No threats of violence.": (True, False, "high")},
}


class Interpreter:
    def __init__(self, lam, log):
        self.lam, self.log = lam, log

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error", "quiesced"), (self.lam.fn_name, name, args, r.text)
        self.log.append((name, args.get("function") or args.get("type")))
        return r

    def run(self, session):
        f, a, d = self.lam.fn_name, self.lam.in_, lambda n, x: self.do(session, n, x)
        if f in ("violates", "severity_of"):
            lenient, strict, sev = WORLD[a["post"]].get(a["rule"], (False, False, "low"))
            value = sev if f == "severity_of" else (strict if NEW in self.lam.body else lenient)
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": value})
            return self._done(session)
        ret = lambda action, rules_path, note: (
            d("write", {"path": "return/action", "type": '"allow" | "warn" | "remove" | "escalate"', "value": action}),
            d("write", {"path": "return/rules", "type": "Text[]", **({"source": rules_path} if rules_path else {"value": []})}),
            d("write", {"path": "return/note", "type": "Text", "value": note}))
        d("call", {"function": "split_rules", "to": "let/rules", "inputs": {"policy": "args/policy"}})
        d("call", {"function": "violates", "to": "let/flags", "over": "let/rules", "inputs": {"post": "args/post"}})
        d("call", {"function": "select_by_flags", "to": "let/hits", "inputs": {"items": "let/rules", "flags": "let/flags"}})
        if d("run_code", {"code": "locals.hits.length"}).value == 0:
            ret("allow", None, "No rule applies.")
            return self._done(session)
        d("call", {"function": "severity_of", "to": "let/severities", "over": "let/hits", "inputs": {"post": "args/post"}})
        if not d("run_code", {"code": "locals.severities.includes('high')"}).value:
            ret("warn", "let/hits", "Minor: the author is reminded of the rules.")
            return self._done(session)
        d("write", {"path": "let/strict", "type": "Function<violates>"})
        d("edit", {"path": "let/strict/instructions", "old": OLD, "new": NEW})
        d("call", {"function": "let/strict", "to": "let/confirmed_flags", "over": "let/hits", "inputs": {"post": "args/post"}})
        d("call", {"function": "select_by_flags", "to": "let/confirmed", "inputs": {"items": "let/hits", "flags": "let/confirmed_flags"}})
        if d("run_code", {"code": "locals.confirmed.length"}).value == 0:
            ret("escalate", "let/hits", "A serious rule may apply, but it is not clear-cut. A human should look.")
        else:
            ret("remove", "let/confirmed", "A serious violation, confirmed on a strict reading.")
        return self._done(session)

    def _done(self, session):
        assert session.finish()


def moderate(post):
    log = []
    rt = Runtime(lambda lam: Interpreter(lam, log))
    out, value = rt.run_root(load(ROOT / "codebases" / "moderation" / "moderate.nl", {"post": post, "policy": POLICY}))
    assert out.kind == "done", out.detail
    return dump(value), log


def test_four_outcomes():
    posts = list(WORLD)
    assert moderate(posts[0])[0]["action"] == "allow"
    assert moderate(posts[1])[0] == {"action": "warn", "rules": ["No advertising."], "note": "Minor: the author is reminded of the rules."}
    removed, log = moderate(posts[2])
    assert removed["action"] == "remove" and removed["rules"] == ["No threats of violence."]
    assert ("write", "Function<violates>") in log and ("call", "let/strict") in log
    assert moderate(posts[3])[0]["action"] == "escalate"                       # the strict reader was not sure


def test_the_second_opinion_is_only_sought_when_needed():
    _, log = moderate(list(WORLD)[1])
    assert ("write", "Function<violates>") not in log
