"""Replay the canonical traces of the conformance programs through the runtime.

The root lambda replays its recorded actions. Every other natural-language
lambda is answered by a small Python oracle, standing in for the model.
"""
import re
from pathlib import Path

import pytest
import yaml

from natlang import js
from natlang.agents import Close, OracleAgent, ReplayAgent
from natlang.nodes import MISSING, Lambda
from natlang.runtime import Runtime
from natlang.values import coerce, dump, load_program

ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = ROOT / "conformance" / "programs"


def parse_trace(text):
    """-> [(action_text, expected_kind)]"""
    out, cur, exp = [], None, None
    for line in text.splitlines():
        if line.startswith(">>> "):
            if cur is not None:
                out.append(("\n".join(cur), exp))
            cur, exp = [line[4:]], None
        elif line.startswith("<<< "):
            m = re.search(r"\b(completed|replaced|quiesced|done|ok|refused|rejected)\b", line)
            exp = m.group(1) if m else None
        elif cur is not None and exp is None:
            body = line[4:] if line.startswith("    ") else line.strip()  # bodies are indented 4
            if body.strip() != "(empty body)":
                cur.append(body)
    if cur is not None:
        out.append(("\n".join(cur), exp))
    return out


URGENT = ("down", "failing", "corrupting", "breach", "stopped", "500 errors")


def oracle(lam: Lambda):
    text, i = lam.body, lam.in_
    if "urgent?" in text:
        return any(w in i["item"].lower() for w in URGENT)
    if "Count how many of" in text:
        n = sum(1 for f in i["flags"] if f)
        return f"ALERT: {n} urgent" if n > 5 else "ok"
    if "according to `args/rubric`" in text:
        t = i["item"].lower()
        if "watches" in t or "90% off" in t:
            return "spam"
        return "billing" if any(w in t for w in ("charged", "invoice", "card")) else "technical"
    if "a question or a complaint" in text:
        t = i["item"]
        if not re.search(r"[a-z]{4,}", t.lower()) or "zzv" in t:
            return Close("The ticket is not readable text; it is neither a question nor a complaint.")
        return "question" if t.strip().endswith("?") else "complaint"
    if "noticeably shorter" in text:
        return "The shop will be closed on Monday."
    if "headline" in text.lower():
        if lam.ret is not MISSING:  # reopened: the draft return holds the previous attempt
            return "Revenue grows four percent on northern sales"
        return "Quarterly revenue grows four percent on strong northern sales as costs hold"
    if "serves the customer" in text:
        return "decline"
    raise AssertionError(f"no oracle for: {text!r}")


TRACED = sorted(p for p in PROGRAMS.glob("*.yaml") if "canonical_trace" in p.read_text())


@pytest.mark.parametrize("path", TRACED, ids=lambda p: p.stem)
def test_canonical_trace(path):
    doc = yaml.safe_load(path.read_text())
    root = load_program(doc["program"])
    env_inner = root.env(__import__("natlang.types", fromlist=["TypeEnv"]).TypeEnv())
    for name, value in (doc.get("inputs") or {}).items():
        ft = root.type.params.get(name)[0]
        root.in_[name] = coerce(value, ft, env_inner, yaml=False, path=f"args/{name}")

    steps = parse_trace(doc["canonical_trace"])
    log = []

    def factory(lam):
        return ReplayAgent([a for a, _ in steps], log) if lam is root else OracleAgent(oracle, lam)

    rt = Runtime(factory)
    out, value = rt.run_root(root)

    got = [kind for _, kind, _ in log]
    want = [k or "ok" for _, k in steps]
    detail = "\n".join(f"{h:<50} {k:<10} {t.splitlines()[0] if t else ''}" for h, k, t in log)
    assert got == want, f"\n{detail}"
    assert out.kind == "done", detail

    exp = doc["expect"]
    if "value" in exp:
        assert dump(value) == exp["value"]
    for chk in exp.get("checks", []):
        if chk["kind"] == "crisp":
            ok = js.run(f"return (function(value){{ return {chk['code']} }})(args.value)",
                        {"args": {"value": dump(value)}}, None, body=True, path="check")
            assert ok is True, chk["code"]
    assert len(log) <= (doc.get("lint") or {}).get("max_actions", 99)
