"""Behavioural identity of the serving/training prompt-rendering contract (schema /2).

The contract is what the executing model sees of a request: ``data/prompt_rendering.apply_prompt_rendering`` for the
mode (full | program-header) over the shared piece table, then the serving prompt assembly
``serve/engine.Engine.render_request`` -> ``serve/chat.render_messages`` (content escaping, tool-call argument
normalisation, Neuralese block placeholders), plus the declared per-backbone history-reasoning policy
(``serve/chat.BACKBONE_HISTORY_REASONING``). The backbone's own chat template is part of the base binding, not of
this contract, so a fixed stub template (canonical JSON of the plain messages and tools it receives) stands in.

Identity /2 = sha256 over: schema, mode, the sha256 of the pinned fixture inputs
(``training/prompt-pieces/rendering-contract-fixtures.json``), the sha256 of the canonical rendered outputs of every
fixture case, and the piece table's sha256. A refactor or an unrelated helper keeps it; a change to what a request
renders to alters it. Identity /1 (natlang.prompt-rendering-identity/1) hashed the source bytes of
prompt_rendering.py, serve/engine.py and serve/chat.py, so any edit there refused every export bound before it.

Compatibility: ``training/prompt-pieces/rendering-identity-equivalence.json`` maps each known superseded identity (a
/1 identity, or a /2 identity of an earlier fixture set) to the /2 identity of the exact code it was bound to,
computed by running the current fingerprint over that commit's blobs
(``scripts/neuralese_data/prompt_rendering_equivalence.py``; re-verified from git by
tests/neuralese/test_rendering_contract.py). A binding is accepted when its identity equals the current /2 identity,
or when the table maps it to the current /2 identity. A behaviour change therefore still refuses old bindings,
unless the table declares an ADOPTION for it: a deliberate rendering change that applies to artifacts bound before
it (first: 95bc6eca, program-header serving keeps externally supplied tool schemas; native natlang requests send none
in that mode, so weights trained without schemas see the same native prompts). An adoption maps the /2 identity
just before its commit to the /2 identity at it, both recomputed from git by the test; a binding is accepted when
its /2 identity reaches the current one through adoptions. Undeclared behaviour changes still refuse.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "natlang.prompt-rendering-identity/2"
_PACKAGE = Path(__file__).resolve().parents[1]
_PIECE_DIR = _PACKAGE.parents[1] / "prompt-pieces"
FIXTURES_PATH = _PIECE_DIR / "rendering-contract-fixtures.json"
PIECES_PATH = _PIECE_DIR / "pieces.json"
EQUIVALENCE_PATH = _PIECE_DIR / "rendering-identity-equivalence.json"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fixture_cases(fixtures: dict) -> list[tuple[str, list, list]]:
    """(case id, messages, tools): every system prompt before every conversation, each conversation without a system
    turn, and a conversation with a second system turn (refused by program-header)."""
    cases = []
    for conversation in fixtures["conversations"]:
        for system in fixtures["system_prompts"]:
            cases.append((f"{system['id']}|{conversation['id']}",
                          [{"role": "system", "content": system["text"]}] + conversation["messages"], fixtures["tools"]))
        cases.append((f"no-system|{conversation['id']}", list(conversation["messages"]), fixtures["tools"]))
        system = fixtures["system_prompts"][0]
        cases.append((f"{system['id']}|{conversation['id']}|no-tools",
                      [{"role": "system", "content": system["text"]}] + conversation["messages"], None))
    first = fixtures["conversations"][0]["messages"]
    cases.append(("late-system", first[:1] + [{"role": "system", "content": "late"}] + first[1:], fixtures["tools"]))
    return cases


def rendered_outputs(mode: str, fixtures: dict | None = None) -> list[dict]:
    """The canonical rendering of every fixture case through the serving request renderer for ``mode``."""
    from types import SimpleNamespace

    from ..serve.engine import Engine

    fixtures = fixtures if fixtures is not None else json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    block_types = fixtures.get("block_types") or {}

    def template(messages, tools):
        return _canonical({"messages": messages, "tools": tools, "add_generation_prompt": True})

    stub = SimpleNamespace(prompt_rendering=mode, _template=template, specials=tuple(fixtures["specials"]),
                           block_value_type=lambda block_id: block_types.get(block_id))
    outputs = []
    for case_id, messages, tools in fixture_cases(fixtures):
        try:
            rendered = Engine.render_request(stub, json.loads(_canonical(messages)),
                                             None if tools is None else json.loads(_canonical(tools)))
        except Exception as error:  # a refusal is behaviour too; its type is recorded, not its wording
            outputs.append({"case": case_id, "error": type(error).__name__})
            continue
        nonce = rendered.escape_nonce
        outputs.append({"case": case_id, "blocks": list(rendered.blocks),
                        "segments": [s.replace(nonce, "{nonce}") if isinstance(s, str) else s
                                     for s in rendered.segments]})
    return outputs


def compute_identity(mode: str) -> dict:
    """The /2 identity of the code and piece table this process imports (uncached)."""
    from ..serve import chat
    from .prompt_rendering import PROMPT_RENDERINGS

    if mode not in PROMPT_RENDERINGS:
        raise ValueError(f"unknown prompt_rendering {mode!r} (declared: {PROMPT_RENDERINGS})")
    fixtures_sha = hashlib.sha256(FIXTURES_PATH.read_bytes()).hexdigest()
    pieces_sha = hashlib.sha256(PIECES_PATH.read_bytes()).hexdigest()
    behaviour = {"renderings": list(PROMPT_RENDERINGS),
                 "history_reasoning": dict(chat.BACKBONE_HISTORY_REASONING),
                 "cases": rendered_outputs(mode)}
    rendered_sha = hashlib.sha256(_canonical(behaviour).encode()).hexdigest()
    identity = {"schema": SCHEMA, "mode": mode, "fixtures_sha256": fixtures_sha,
                "rendered_sha256": rendered_sha, "pieces_sha256": pieces_sha}
    identity["identity_sha256"] = hashlib.sha256(
        "\0".join((SCHEMA, mode, fixtures_sha, rendered_sha, pieces_sha)).encode()).hexdigest()
    return identity


_CACHE: dict = {}


def behavioural_identity(mode: str) -> dict:
    if mode not in _CACHE:
        _CACHE[mode] = compute_identity(mode)
    return dict(_CACHE[mode])


def _table() -> dict:
    if not EQUIVALENCE_PATH.exists():
        return {}
    return json.loads(EQUIVALENCE_PATH.read_text(encoding="utf-8"))


def equivalence_table() -> dict:
    return _table().get("identities") or {}


def adoptions() -> list:
    """Declared rendering changes that apply to artifacts bound before them (owner/agent decisions with a commit):
    each maps the /2 identity just before the change to the /2 identity at it, for one mode."""
    return _table().get("adoptions") or []


def _reaches_current(v2: str, mode: str, current: str) -> bool:
    steps = [a for a in adoptions() if a.get("mode") == mode]
    for _ in range(len(steps) + 1):
        if v2 == current:
            return True
        nxt = next((a["to_v2"] for a in steps if a.get("from_v2") == v2), None)
        if nxt is None:
            return False
        v2 = nxt
    return False


def binding_accepted(identity_sha256, mode: str) -> bool:
    """Whether a recorded identity (any schema) binds the current contract's behaviour for ``mode``: it is the current
    /2 identity, or the table maps it to a /2 identity that is current or reaches it through declared adoptions."""
    if not isinstance(identity_sha256, str):
        return False
    current = behavioural_identity(mode)["identity_sha256"]
    if identity_sha256 == current:
        return True
    entry = equivalence_table().get(identity_sha256)
    if entry is not None:
        return entry.get("mode") == mode and _reaches_current(entry.get("v2_identity_sha256"), mode, current)
    return _reaches_current(identity_sha256, mode, current) if any(
        a.get("from_v2") == identity_sha256 for a in adoptions()) else False


def comparison_binding(mode: str) -> dict:
    """The identity a stable comparison key uses for ``mode``: the table's designated canonical older identity while
    it binds the current behaviour (directly or through declared adoptions), so evaluation keys recorded before /2
    still compare; else the /2 identity."""
    current = behavioural_identity(mode)
    for entry in equivalence_table().values():
        if (entry.get("mode") == mode and entry.get("comparison_canonical") and
                _reaches_current(entry.get("v2_identity_sha256"), mode, current["identity_sha256"])):
            return dict(entry["identity"])
    return current
