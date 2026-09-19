"""Grading a conformance outcome the way the spec states it: `expect.value` (equality), `expect.status`,
`expect.checks` / `expect.note_checks` (crisp code over `value`, or a yes/no question put to a judge model)."""
from __future__ import annotations

import json
from typing import Callable, Optional

from . import js

JUDGE = ("You are grading an answer. Read the answer and the question about it, then reply with exactly one "
         "word: yes or no.\n\nAnswer being graded:\n{subject}\n\nQuestion: {question}")


def make_judge(decoder) -> Callable[[str, str], Optional[bool]]:
    def judge(subject: str, question: str) -> Optional[bool]:
        turn = decoder.chat([{"role": "user", "content": JUDGE.format(subject=subject, question=question)}], [],
                            temperature=0.0, max_tokens=8)
        word = turn.text.strip().lower().lstrip("*\"' ")
        return True if word.startswith("yes") else False if word.startswith("no") else None
    return judge


def run_checks(checks: list, subject, judge=None) -> list:
    """-> [(check, passed)] with passed in True | False | None (None: could not be evaluated)."""
    out = []
    for c in checks or []:
        if c.get("kind") == "crisp":
            try:
                got = js.run("const value = self.value;\nreturn (" + c["code"] + ");", {"value": subject}, None,
                             body=True, path="check")
                out.append((c, bool(got)))
            except js.JsError:
                out.append((c, False))
        elif c.get("kind") == "judge":
            text = subject if isinstance(subject, str) else json.dumps(subject, ensure_ascii=False)
            got = judge(text, c["question"]) if judge else None
            out.append((c, None if got is None else got == c.get("answer", True)))
        else:
            out.append((c, None))
    return out


def grade(expect: dict, kind: str, value, note: str = "", judge=None, *, emitted=None):
    """Grade all stated requirements, including exact outputs and observable effects.

    Missing effect observations are ungraded, never an implicit pass.
    """
    results = []
    wanted_status = expect.get("status", "done")
    if kind != wanted_status:
        return "no", [f"status {kind}, expected {wanted_status}"]
    if "value" in expect:
        results.append(({"code": "value differs"}, value == expect["value"]))
    results += run_checks(expect.get("checks"), value, judge)
    results += run_checks(expect.get("note_checks"), note, judge)
    if "emitted" in expect or "effect_checks" in expect:
        if emitted is None:
            results.append(({"question": "effect observations unavailable"}, None))
        else:
            if "emitted" in expect:
                results.append(({"code": "emitted records differ"}, emitted == expect["emitted"]))
            results += run_checks(expect.get("effect_checks"), emitted, judge)
    failed = [c.get("code") or c.get("question") for c, ok in results if ok is False]
    unknown = [c.get("question") for c, ok in results if ok is None]
    return ("no" if failed else "?" if unknown else "yes"), failed + [f"unjudged: {u}" for u in unknown]
