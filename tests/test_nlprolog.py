"""The natural-language Prolog code base (codebases/nlprolog): declared recursion, conditionals, open world."""
from pathlib import Path

from natlang import gbnf
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.values import dump

ROOT = Path(__file__).resolve().parent.parent
S = ToolSurface()
KB = ["Socrates is a man.", "Every man is mortal.", "Whoever is mortal and wise is remembered.", "Socrates is wise.",
      "Plato is wise."]
FACTS = {"Socrates is a man.", "Socrates is wise.", "Plato is wise."}
RULES = {"Every man is mortal.": ("is mortal.", ["is a man."]),                     # conclusion suffix, condition suffixes
         "Whoever is mortal and wise is remembered.": ("is remembered.", ["is mortal.", "is wise."])}
subject = lambda goal: goal.split(" ")[0]
LEAF = {"states": lambda a: a["statement"] in FACTS and a["statement"] == a["goal"],
        "is_rule_for": lambda a: a["statement"] in RULES and a["goal"].endswith(RULES[a["statement"]][0]),
        "conditions_for": lambda a: [f"{subject(a['goal'])} {c}" for c in RULES[a["rule"]][1]]}


class Unfinished(Exception):
    pass


class Interpreter:
    """Follows the pseudocode of solve and try_rule as a good interpreter would; leaves answer from LEAF."""
    def __init__(self, lam, log):
        self.lam, self.log = lam, log

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error"), (self.lam.fn_name, name, args, r.text)
        self.log.append((self.lam.fn_name, name, args.get("function")))
        if r.kind == "quiesced":                       # a call that did not finish is passed upward, never papered over
            raise Unfinished(r.text.splitlines()[0])
        return r

    def run(self, session):
        try:
            return self._run(session)
        except Unfinished as e:
            return f"a call did not finish: {e}"

    def _run(self, session):
        f, d = self.lam.fn_name, lambda n, a: self.do(session, n, a)
        if f in LEAF:
            from natlang.types import format_type
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": LEAF[f](self.lam.in_)})
        elif f == "try_rule":
            d("call", {"function": "conditions_for", "to": "let/conditions", "inputs": {"rule": "args/rule", "goal": "args/goal"}})
            d("call", {"function": "solve", "to": "let/results", "over": "let/conditions", "inputs": {"kb": "args/kb"}})
            d("call", {"function": "all_yes", "to": "return", "inputs": {"results": "let/results", "rule": "args/rule"}})
        else:
            d("call", {"function": "states", "to": "let/stated", "over": "args/kb", "inputs": {"goal": "args/goal"}})
            d("call", {"function": "any_true", "to": "let/found", "inputs": {"flags": "let/stated"}})
            if d("read", {"path": "let/found"}).value:
                d("call", {"function": "first_selected", "to": "let/fact", "inputs": {"items": "args/kb", "flags": "let/stated"}})
                proof = d("run_code", {"code": "[locals.fact]"}).value
                d("write", {"path": "return", "type": "Answer", "value": {"verdict": "yes", "proof": proof}})
            else:
                d("call", {"function": "is_rule_for", "to": "let/concludes", "over": "args/kb", "inputs": {"goal": "args/goal"}})
                d("call", {"function": "select_by_flags", "to": "let/rules", "inputs": {"items": "args/kb", "flags": "let/concludes"}})
                if d("run_code", {"code": "locals.rules.length"}).value == 0:
                    d("write", {"path": "return", "type": "Answer", "value": {"verdict": "unknown", "proof": []}})
                else:
                    d("call", {"function": "try_rule", "to": "let/attempts", "over": "let/rules",
                               "inputs": {"goal": "args/goal", "kb": "args/kb"}})
                    d("call", {"function": "first_yes", "to": "return", "inputs": {"answers": "let/attempts"}})
        assert session.finish()


def _solve(goal, kb=KB, **kw):
    log = []
    rt = Runtime(lambda lam: Interpreter(lam, log), **kw)
    out, value = rt.run_root(load(ROOT / "codebases" / "nlprolog" / "solve.nl", {"goal": goal, "kb": kb}))
    return out, (dump(value) if out.kind == "done" else None), log, rt


def test_a_fact_is_found_directly():
    out, v, log, rt = _solve("Socrates is wise.")
    assert v == {"verdict": "yes", "proof": ["Socrates is wise."]}
    assert not any(fn == "try_rule" for _, _, fn in log)                  # the branch not taken is not carried out


def test_two_levels_of_rules_with_a_proof():
    out, v, log, rt = _solve("Socrates is remembered.")
    assert out.kind == "done" and v["verdict"] == "yes"
    assert v["proof"] == ["Whoever is mortal and wise is remembered.", "Every man is mortal.", "Socrates is a man.",
                          "Socrates is wise."]


def test_open_world_unknown_is_not_no():
    out, v, log, rt = _solve("Plato is remembered.")                      # Plato is wise, but nothing says he is a man
    assert v == {"verdict": "unknown", "proof": []}


def test_declared_recursion_is_bounded():
    loop = ["Whoever is lucky is happy.", "Whoever is happy is lucky."]

    LEAF_LOOP = {"Whoever is lucky is happy.": ("is happy.", ["is lucky."]), "Whoever is happy is lucky.": ("is lucky.", ["is happy."])}
    RULES.update(LEAF_LOOP)
    try:
        out, v, log, rt = _solve("Ann is happy.", kb=loop, max_episodes=400, max_depth=30)
    finally:
        for k in LEAF_LOOP:
            RULES.pop(k)
    # never an endless descent, never a guess: the goal that comes back round is the identical lambda (the
    # harness's cycle guard); a chain of ever-new goals would instead stop at the declared max_depth
    assert out.kind == "quiesced" and _deepest(rt)
    assert rt.episodes_started < 200


def _deepest(rt) -> str:
    return " ".join(t["result"] for t in rt.trace
                    if "max_depth" in t.get("result", "") or "identical to a lambda" in t.get("result", ""))
