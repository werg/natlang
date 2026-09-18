"""The generated decoding grammars must accept every action of the canonical traces
at the state where it is taken, and refuse actions the tree or types rule out."""
from pathlib import Path

import pytest
import yaml

from natlang import gbnf
from natlang.actions import parse_action
from natlang.agents import OracleAgent
from natlang.grammar import body_grammar, header_grammar
from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import coerce, load_program

from test_canonical_traces import PROGRAMS, TRACED, oracle, parse_trace


class CheckingReplay:
    def __init__(self, actions, failures):
        self.actions, self.failures = actions, failures

    def run(self, session):
        for text in self.actions:
            header, _, body = text.partition("\n")
            hg = header_grammar(session)
            if not gbnf.accepts(hg, header + "\n"):
                self.failures.append(f"header refused: {header}")
            bg = body_grammar(session, parse_action(text))
            if bg is None:
                if body.strip():
                    self.failures.append(f"unexpected body for: {header}")
            else:
                b = body if body == "" or body.endswith("\n") else body + "\n"
                if not gbnf.accepts(bg, b):
                    self.failures.append(f"body refused for: {header}\n{body}")
            session.act(text)
            if session.completed:
                return None
        return "replay ended"


def _root(doc):
    root = load_program(doc["program"])
    env = root.env(TypeEnv())
    for name, value in (doc.get("inputs") or {}).items():
        root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    return root


@pytest.mark.parametrize("path", TRACED, ids=lambda p: p.stem)
def test_grammar_accepts_canonical_trace(path):
    doc = yaml.safe_load(path.read_text())
    root = _root(doc)
    failures = []
    actions = [a for a, _ in parse_trace(doc["canonical_trace"])]
    rt = Runtime(lambda lam: CheckingReplay(actions, failures) if lam is root else OracleAgent(oracle, lam))
    out, _ = rt.run_root(root)
    assert not failures, "\n".join(failures)
    assert out.kind == "done"


def _session(name):
    doc = yaml.safe_load((PROGRAMS / name).read_text())
    return Session(Runtime(None), _root(doc), TypeEnv())


@pytest.mark.parametrize("header", [
    "read args/nonexistent",          # no such path
    "set args/tickets : Text[]",      # a lambda's own args are not writable
    "set return : Num",               # type does not fit the slot
    "reduce return",                  # nothing pending yet
    "edit args/tickets",              # not a Text node
    "launch missiles",                # not a tool
])
def test_header_grammar_refuses(header):
    s = _session("06-map-with-rubric.yaml")
    assert not gbnf.accepts(header_grammar(s), header + "\n")


def test_body_grammar_is_typed():
    s = _session("02-leaf-extraction.yaml")
    a = parse_action("set return : { customer: Text, order_id: Text, amount: Num, phone?: Text }\n")
    g = body_grammar(s, a)
    assert gbnf.accepts(g, 'customer: "Dana"\norder_id: "0077"\namount: 42.5\n')
    assert gbnf.accepts(g, 'customer: "Dana"\n')                       # a draft with holes
    assert not gbnf.accepts(g, 'customer: "Dana"\namount: lots\n')     # Num must be a number
    assert not gbnf.accepts(g, 'customer: "Dana"\nvip: true\n')        # unknown field


def test_enum_body_and_narrowing():
    s = _session("06-map-with-rubric.yaml")
    s.act("set return : Map<Text, Label>\nfn:\n  $lambda:\n    type: 'Lambda<{ item: Text }, Label>'\n    instructions: x")
    hg = header_grammar(s)
    assert gbnf.accepts(hg, 'set return : Map<Text, "billing" | "spam">\n')      # narrowed enum
    assert not gbnf.accepts(hg, 'set return : Map<Text, "refund">\n')            # not a member
