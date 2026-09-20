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
            if event.get("name") == "call" and event.get("arguments", {}).get("function") == rule["function"]:
                arguments = event["arguments"]
                if arguments.get("to") != rule["to"] or arguments.get("inputs", {}) != rule["inputs"]:
                    raise ValueError("required call destination or inputs changed")
    executed = replay["actions"]  # proposals are separate records; rejected actions were still attempted
    cursor = 0
    for required in contract.required_actions:
        while cursor < len(executed) and (executed[cursor].get("name") != required.get("name") or
                                         any(executed[cursor].get("arguments", {}).get(k) != v
                                             for k, v in required.get("arguments", {}).items())):
            cursor += 1
        if cursor == len(executed):
            raise ValueError(f"required ordered action absent: {required}")
        cursor += 1
    return {"admitted": True, "trace_version": reader.manifest["version"],
            "source_sha256": reader.manifest.get("source_sha256"),
            "run_id": reader.manifest.get("run_id"), "coverage": replay["coverage"]}
