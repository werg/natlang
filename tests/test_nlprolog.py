"""The natural-language Prolog code base (codebases/nlprolog): no recursion; forward chaining by repeat-until."""
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
KB = ["Socrates is a man.", "Every man is mortal.", "Whoever is mortal and wise is remembered.", "Socrates is wise.",
      "Plato is wise."]
RULES = {"Every man is mortal.": ("is mortal.", ["is a man."]),                      # conclusion, conditions
         "Whoever is mortal and wise is remembered.": ("is remembered.", ["is mortal.", "is wise."])}


def conclusions(a):
    concl, conds = RULES[a["rule"]]
    subjects = {k.split(" ")[0] for k in a["known"]}
    return [f"{s} {concl}" for s in sorted(subjects)
            if all(f"{s} {c}" in a["known"] for c in conds) and f"{s} {concl}" not in a["known"]]


LEAF = {"is_rule": lambda a: a["statement"] in RULES, "same_claim": lambda a: a["known"] == a["goal"],
        "conclusions": conclusions}


class Interpreter:
    """Follows the pseudocode of solve and derive as a good interpreter would; leaves answer from LEAF."""
    def __init__(self, lam, log):
        self.lam, self.log = lam, log

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error", "quiesced"), (self.lam.fn_name, name, args, r.text)
        self.log.append((self.lam.fn_name, name, args.get("function")))
        return r

    def run(self, session):
        f, d = self.lam.fn_name, lambda n, a: self.do(session, n, a)
        if f in LEAF:
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": LEAF[f](self.lam.in_)})
        elif f == "derive":
            d("call", {"function": "conclusions", "to": "let/found", "over": "args/rules", "inputs": {"known": "args/state/known"}})
            d("call", {"function": "merge", "to": "return", "inputs": {"state": "args/state", "found": "let/found"}})
        else:
            d("call", {"function": "is_rule", "to": "let/rule_flags", "over": "args/kb"})
            flags = d("run_code", {"code": "locals.rule_flags.map(f => !f)"}).value
            d("write", {"path": "let/fact_flags", "type": "Bool[]", "value": flags})
            d("call", {"function": "select_by_flags", "to": "let/rules", "inputs": {"items": "args/kb", "flags": "let/rule_flags"}})
            d("call", {"function": "select_by_flags", "to": "let/facts", "inputs": {"items": "args/kb", "flags": "let/fact_flags"}})
            start = d("run_code", {"code": "({ known: locals.facts, derived: [], grew: true })"}).value
            d("write", {"path": "let/start", "type": "State", "value": start})
            d("call", {"function": "derive", "to": "let/final", "init": "let/start", "until": "settled", "max": 6,
                       "inputs": {"rules": "let/rules"}})
            d("call", {"function": "same_claim", "to": "let/hits", "over": "let/final/known", "inputs": {"goal": "args/goal"}})
            d("call", {"function": "any_true", "to": "let/found", "inputs": {"flags": "let/hits"}})
            verdict = "yes" if d("read", {"path": "let/found"}).value else "unknown"
            d("write", {"path": "return/verdict", "type": '"yes" | "unknown"', "value": verdict})
            d("write", {"path": "return/derived", "type": "Text[]", "source": "let/final/derived"})
        assert session.finish()


def _solve(goal, kb=KB):
    log = []
    rt = Runtime(lambda lam: Interpreter(lam, log), max_episodes=400)
    out, value = rt.run_root(load(ROOT / "codebases" / "nlprolog" / "solve.nl", {"goal": goal, "kb": kb}))
    assert out.kind == "done", out.detail
    return dump(value), log


def test_two_levels_of_rules_need_two_rounds_and_a_third_to_settle():
    v, log = _solve("Socrates is remembered.")
    assert v == {"verdict": "yes", "derived": ["Socrates is mortal.", "Socrates is remembered."]}
    assert sum(1 for fn, tool, callee in log if fn == "derive" and callee == "merge") == 3


def test_a_fact_needs_no_rule():
    assert _solve("Plato is wise.")[0]["verdict"] == "yes"


def test_open_world_unknown_is_not_no():
    v, _ = _solve("Plato is remembered.")                                 # Plato is wise, but nothing says he is a man
    assert v["verdict"] == "unknown" and "Plato is remembered." not in v["derived"]
