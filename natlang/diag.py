"""Diagnostics (SPEC 6.5)."""
from __future__ import annotations

from dataclasses import dataclass

REJECT = "reject"
BLOCKS = "blocks-commit"
HOLE = "hole"


@dataclass(frozen=True)
class Diagnostic:
    path: str
    code: str
    severity: str = REJECT
    expected: str = ""
    got: str = ""

    def __str__(self) -> str:
        if self.code == "hole":
            return f"{self.path}: hole, missing {self.expected}".rstrip()
        parts = [f"{self.path}: {self.code}"]
        if self.expected:
            parts.append(f"expected {self.expected}")
        if self.got:
            parts.append(f"got {self.got}")
        return ", ".join(parts)


class Reject(Exception):
    """The action is invalid. Nothing enters the tree."""

    def __init__(self, *diags: Diagnostic):
        self.diags = list(diags)
        super().__init__("; ".join(str(d) for d in diags))


class Refuse(Exception):
    """A commit point was not satisfied. Nothing changes."""

    def __init__(self, *diags: Diagnostic):
        self.diags = list(diags)
        super().__init__("; ".join(str(d) for d in diags))


def reject(path: str, code: str, expected: str = "", got: str = "") -> Reject:
    return Reject(Diagnostic(path, code, REJECT, expected, got))
