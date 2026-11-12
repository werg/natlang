"""The current serving request for recorded natlang tool-loop calls (training rows that match what ts-host sends).

Recorded trajectories carry the system prompt and tool schemas of the runtime version that produced them.
``serving_messages`` rebuilds the request this ts-host sends for the same call with ts-host's own code
(``ts-host/src/teacher/serving-request.ts`` through data/ts_host.py): the entry point's base system prompt (the same for
every call), the call's guidance and surface paragraphs as the last paragraph of its opening message (the agent's
``callNotes``), and the runtime's tool list. A record that is not a tool-loop call, or that calls a tool the current
surface does not offer, is returned unchanged with the reason; callers decide whether to keep its recorded rendering.
Nothing here grants admission or changes split, source or target bytes.
"""
from __future__ import annotations

import hashlib

from .ts_host import identity, request

# The tool-loop entry points: ``collector`` (eval probes, RL and teacher collection) and ``runtime`` (natlang run). Both
# send ts-host's SYSTEM_PROMPT since 2026-11-12 (natlang run had no generation-guidance appendix before), so ``mix`` and
# either entry rebuild the same request.
ENTRY_POINTS = ('collector', 'runtime')

# ``mix``: each record's entry point is drawn deterministically from its id with these shares.
ENTRY_SHARES = (('collector', 0.85), ('runtime', 0.15))


def entry_for(record_id: str, entry: str) -> str:
    """The entry point of one record: ``entry`` itself, or for ``mix`` a deterministic draw from ``ENTRY_SHARES``."""
    if entry != 'mix':
        return entry
    draw = int.from_bytes(hashlib.sha256(('serving-entry\0' + record_id).encode()).digest()[:8], 'big') / 2 ** 64
    for name, share in ENTRY_SHARES:
        if draw < share:
            return name
        draw -= share
    return ENTRY_SHARES[-1][0]


def serving_identity() -> dict:
    """The ts-host code that rebuilds the requests (bind it in a rendered corpus's inputs)."""
    return dict(identity())


def serving_messages(messages, tools, *, target=None, entry: str = 'collector', rendering: str = 'full'):
    """``(messages, tools, report)``: the current serving request of ``entry`` for a recorded call, rendered in
    ``rendering`` (``full`` or ``program-header``)."""
    if entry not in ENTRY_POINTS:
        raise ValueError(f'unknown serving entry point {entry!r} (declared: {ENTRY_POINTS})')
    out = request({'op': 'serving', 'messages': messages, 'tools': tools or [], 'target': target,
                   'entry': entry, 'rendering': rendering})
    report = {**out['report'], 'ts_host': serving_identity()}
    if report['mode'] == 'recorded':
        return messages, tools, report
    return out['messages'], out['tools'], report
