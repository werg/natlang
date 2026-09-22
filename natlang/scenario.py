"""Whole-run trace admission against semantic outcome and action obligations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trace import TraceReader


@dataclass(frozen=True)
class ScenarioContract:
    outcome: str
    value: Any = None
    effects: tuple | None = None
    required_actions: tuple = ()
    constrained_calls: tuple = ()


CALL_MODES = {"run_function", "for_each", "fold", "repeat"}


def action_matches(actual: dict, required: dict) -> bool:
    """Compare tools-v2 semantic obligations with either v2 or tools-v4 actions."""
    aname, aargs = actual.get("name"), actual.get("arguments", {})
    rname, rargs = required.get("name"), required.get("arguments", {})
    if aname == rname:
        return all(aargs.get(key) == value for key, value in rargs.items())
    if rname == "write" and aname == "write_value":
        translated = {"path": aargs.get("destination"), "type": aargs.get("type"), "value": aargs.get("value")}
        return all(translated.get(key) == value for key, value in rargs.items())
    if rname == "write" and aname == "copy_value":
        translated = {"path": aargs.get("destination"), "source": aargs.get("source")}
        return all(translated.get(key) == value for key, value in rargs.items())
    if rname == "call" and aname in CALL_MODES:
        translated = {"function": aargs.get("function"), "to": aargs.get("save_as"),
                      "inputs": dict(zip((rargs.get("inputs") or {}).keys(), aargs.get("inputs") or []))}
        if aname in ("for_each", "fold"):
            translated["over"] = aargs.get("items")
        if aname in ("fold", "repeat"):
            translated["init"] = aargs.get("initial")
        if aname == "repeat":
            translated.update(until=aargs.get("until"), max=aargs.get("at_most"))
        return all(translated.get(key) == value for key, value in rargs.items())
    return False


def admit(reader: TraceReader, contract: ScenarioContract) -> dict:
    replay = reader.replay_observations()
    if replay["outcome"] != contract.outcome:
        raise ValueError(f"outcome changed: {replay['outcome']} != {contract.outcome}")
    if contract.outcome == "done" and replay["final"] != contract.value:
        raise ValueError("final captured value does not match the contract")
    requested = [event for event in replay["effects"] if event["phase"] == "requested"]
    observed_effects = tuple((e["capability"], e.get("args")) for e in requested)
    if contract.effects is not None and observed_effects != contract.effects:
        raise ValueError("ordered effect sequence does not match the contract")
    if contract.effects is not None:
        completed = {(e.get("call_id"), e["capability"], e["sequence"])
                     for e in replay["effects"] if e["phase"] == "completed"}
        if any((e.get("call_id"), e["capability"], e["sequence"]) not in completed
               for e in requested):
            raise ValueError("required effect was requested but did not complete")
    for rule in contract.constrained_calls:
        for event in replay["actions"]:
            if event.get("name") in ({"call"} | CALL_MODES) and event.get("arguments", {}).get("function") == rule["function"]:
                required = {"name": "call", "arguments": rule}
                if not action_matches(event, required):
                    raise ValueError("required call destination or inputs changed")
    executed = replay["actions"]  # proposals are separate records; rejected actions were still attempted
    cursor = 0
    for required in contract.required_actions:
        while cursor < len(executed) and not action_matches(executed[cursor], required):
            cursor += 1
        if cursor == len(executed):
            raise ValueError(f"required ordered action absent: {required}")
        cursor += 1
    return {"admitted": True, "trace_version": reader.manifest["version"],
            "source_sha256": reader.manifest.get("source_sha256"),
            "run_id": reader.manifest.get("run_id"), "coverage": replay["coverage"]}
