"""Agents drive a Session. An agent's `run` returns a closing note, or None if it completed."""
from __future__ import annotations

from typing import Any, Callable, Optional

import yaml

from .nodes import Lambda
from .types import format_type


def empty_instructions(session) -> Any:
    n = max(1, len(session.lam.body.splitlines()))
    return session.act(f"edit {session.lam.kind}[1..{n}]")


class StubAgent:
    """Scripted behaviour for harness tests, keyed by a substring of the instructions.

    rules: { "<substring>": [step, ...] | [ {when_item: x, do: [...]}, {otherwise: true, do: [...]} ] }
    steps: {close: note} | {action: text, expect: {...}} | {set_return_from: path}
           | {empty_instructions: true}
    """

    def __init__(self, rules: dict, lam: Lambda, failures: list):
        self.rules, self.lam, self.failures = rules, lam, failures

    def _steps(self):
        for key, spec in self.rules.items():
            if key not in self.lam.body:
                continue
            if spec and isinstance(spec[0], dict) and "do" in spec[0]:
                for rule in spec:
                    if "when_item" in rule and self.lam.in_.get("item") == rule["when_item"]:
                        return rule["do"]
                for rule in spec:
                    if "when_item" not in rule:
                        return rule["do"]
                return []
            return spec
        return [{"close": "no stub rule matched"}]

    def run(self, session) -> Optional[str]:
        for step in self._steps():
            if "close" in step:
                return step["close"]
            if "action" in step:
                r = session.act(step["action"])
                exp = step.get("expect") or {}
                if "result" in exp and r.kind != exp["result"]:
                    self.failures.append(f"stub action {step['action']!r}: got {r.kind}, want {exp['result']}")
                for c in exp.get("codes", []):
                    if c not in r.codes:
                        self.failures.append(f"stub action {step['action']!r}: missing code {c}, got {r.codes}")
            elif "set_return_from" in step:
                session.act(f"copy {step['set_return_from']} to return")
            elif step.get("empty_instructions"):
                empty_instructions(session)
            if session.completed:
                return None
        return None if session.completed else "stub ran out of steps"


class OracleAgent:
    """Answers leaf lambdas from a Python function; used to replay reference traces.

    oracle(lam) returns the value for `return`, or a `Close(note)`.
    """

    def __init__(self, oracle: Callable[[Lambda], Any], lam: Lambda):
        self.oracle, self.lam = oracle, lam

    def run(self, session) -> Optional[str]:
        out = self.oracle(self.lam)
        if isinstance(out, Close):
            return out.note
        t = format_type(self.lam.type.returns)
        body = out if isinstance(out, str) and t == "Text" else yaml.safe_dump(out, default_flow_style=True).strip()
        if body.endswith("\n..."):
            body = body[:-4]
        r = session.act(f"set return : {t}\n{body}")
        if r.kind != "ok":
            return f"oracle value rejected: {r.text}"
        empty_instructions(session)
        return None if session.completed else "could not complete"


class Close:
    def __init__(self, note: str):
        self.note = note


class ReplayAgent:
    """Replays a fixed list of action texts (a canonical trace) for one lambda."""

    def __init__(self, actions: list, log: list):
        self.actions, self.log = actions, log

    def run(self, session) -> Optional[str]:
        for text in self.actions:
            r = session.act(text)
            self.log.append((text.splitlines()[0], r.kind, r.text))
            if session.completed:
                return None
        return "replay ended without completing"
