"""Shared helpers for turning decision examples into text and measuring readout quality."""
from __future__ import annotations

import json
import re

from .decision import scores

_INSTRUCTIONS = re.compile(r"Instructions:\n(.*?)\n\nIn eval", re.S)
_STATE = re.compile(r"^state: string = (.*)\nDeclared state", re.S)


def write_prompt(cases: list[dict]) -> str:
    """The family's instructions and a few cases with their answers, asking for a note on how to decide."""
    def text(row, role):
        return next(m["content"] for m in row["messages"] if m["role"] == role and isinstance(m["content"], str))
    instructions = _INSTRUCTIONS.search(text(cases[0], "user"))
    examples = []
    for row in cases:
        state = _STATE.search(text(row, "tool"))
        answer = row["options"][max(range(len(row["gold"])), key=lambda i: row["gold"][i])]
        examples.append(f"Input: {json.loads(state.group(1)) if state else '?'}\nAnswer: {answer}")
    return (f"A function decides: {instructions.group(1).strip() if instructions else '?'}\n"
            f"Allowed answers: {', '.join(cases[0]['options'])}\n\nExamples:\n" + "\n\n".join(examples) +
            "\n\nWrite a short note for whoever answers this function next: how to decide well.")


def quality(kind: str, predicted: list[float], gold: list[float]) -> float:
    result = scores(kind, predicted, gold)
    return 1 - result["rps"] if "rps" in result else 1 - result["brier"] / 2
