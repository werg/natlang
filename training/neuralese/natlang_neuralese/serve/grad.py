"""Gradient replay sessions (S4 §4.4) — interface only.

A `grad` request replays a recorded execution graph (spec/NEURALESE_GRAPH.md): it restores context revisions,
supplies recorded effect results, holds discrete choices fixed, recomputes the differentiable path and returns
gradients for the requested argument entries as store entries. The server reports `grad: false` in
`/v1/neuralese/info` and answers `POST /v1/neuralese/grad` with 501 until this is implemented.
"""

from __future__ import annotations

from typing import Protocol


class GradSession(Protocol):
    def replay(self, graph: dict, arguments: list[str], objective: dict) -> dict:
        """Return {"loss": float, "gradients": {argument_id: block_id}} for one recorded graph."""


class Unavailable(NotImplementedError):
    code = "neuralese-grad-unavailable"


def grad_session() -> GradSession:
    raise Unavailable("gradient replay sessions are not implemented in the reference server yet")
