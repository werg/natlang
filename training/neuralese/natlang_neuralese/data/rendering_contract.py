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

The identity is used to verify the active rendering behavior. Historical packages retain their own exact source and
identity metadata; current serving does not accept legacy rendering bindings or adoption aliases.
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
