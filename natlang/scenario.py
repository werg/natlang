"""Whole-run trace admission against semantic outcome and action obligations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .trace import TraceReader


@dataclass(frozen=True)
class ScenarioContract:
    outcome: str
    value: Any = None
    effects: tuple = ()
    required_actions: tuple = ()


def admit(reader: TraceReader, contract: ScenarioContract) -> dict:
    replay = reader.replay_observations()
    if replay["outcome"] != contract.outcome:
        raise ValueError(f"outcome changed: {replay['outcome']} != {contract.outcome}")
    if contract.outcome == "done" and replay["final"] != contract.value:
        raise ValueError("final captured value does not match the contract")
    requested = [event for event in replay["effects"] if event["phase"] == "requested"]
    observed_effects = tuple((e["capability"], e.get("args")) for e in requested)
    if contract.effects and observed_effects != contract.effects:
        raise ValueError("ordered effect sequence does not match the contract")
    applied = [e for e in replay["actions"] if e["outcome"] in ("ok", "done", "completed")]
    cursor = 0
    for required in contract.required_actions:
        while cursor < len(applied) and (applied[cursor].get("name") != required.get("name") or
                                         any(applied[cursor].get("arguments", {}).get(k) != v
                                             for k, v in required.get("arguments", {}).items())):
            cursor += 1
        if cursor == len(applied):
            raise ValueError(f"required ordered action absent: {required}")
        cursor += 1
    return {"admitted": True, "trace_version": reader.manifest["version"],
            "source_sha256": reader.manifest.get("source_sha256"),
            "run_id": reader.manifest.get("run_id"), "coverage": replay["coverage"]}
