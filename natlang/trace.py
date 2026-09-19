"""Versioned, append-only execution observations and an offline reader."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


VERSION = "reduction-trace/1"


def _view(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else {"$opaque": "nonfinite-number"}
    if isinstance(value, (list, tuple)):
        return [_view(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _view(v) for k, v in value.items()}
    return {"$opaque": type(value).__name__, "reconstructable": False}


class TraceRecorder:
    def __init__(self, manifest: dict, path: Path | None = None):
        self.events: list[dict] = []
        self._lock = threading.Lock()
        self.path = Path(path) if path is not None else None
        self._stream = self.path.open("w", encoding="utf-8") if self.path else None
        self.emit("manifest", **manifest)

    def emit(self, kind: str, **data) -> dict:
        with self._lock:
            event = _view({"version": VERSION, "seq": len(self.events), "kind": kind, **data})
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
            clean = json.loads(encoded)
            self.events.append(clean)
            if self._stream:
                self._stream.write(encoded + "\n")
                self._stream.flush()
            return clean

    def close(self):
        if self._stream:
            self._stream.close()
            self._stream = None

    def reopen(self):
        if self.path is not None and self._stream is None:
            self._stream = self.path.open("a", encoding="utf-8")


class TraceReader:
    def __init__(self, events: list[dict]):
        if not events or events[0].get("kind") != "manifest":
            raise ValueError("trace has no manifest")
        if any(event.get("version") != VERSION or event.get("seq") != i
               for i, event in enumerate(events)):
            raise ValueError("unsupported or discontinuous trace")
        self.events = events

    @classmethod
    def open(cls, path: Path):
        with Path(path).open(encoding="utf-8") as stream:
            return cls([json.loads(line) for line in stream if line.strip()])

    @property
    def manifest(self) -> dict:
        return self.events[0]

    def of_kind(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["kind"] == kind]

    def final_state(self) -> Any:
        snapshots = self.of_kind("state")
        if not snapshots:
            raise ValueError("trace has no captured state")
        return snapshots[-1]["value"]

    def coverage(self) -> dict:
        snapshots = self.of_kind("state")
        incomplete = any("$stream" in json.dumps(event.get("value")) or
                         '"$opaque"' in json.dumps(event.get("value")) for event in snapshots)
        return {"state_reconstructable": bool(self.of_kind("state")),
                "live_source_reconstructable": not incomplete,
                "native_effects_replayable": False,
                "effect_count": len(self.of_kind("effect")),
                "mode": "recorded-observations-only"}

    def replay_observations(self) -> dict:
        """Read captured decisions and state; never call model, eval or host effects."""
        states = self.of_kind("state")
        if not states or states[0].get("phase") != "initial" or states[-1].get("phase") != "final":
            raise ValueError("trace lacks complete initial/final observations")
        return {"initial": states[0]["value"], "final": states[-1]["value"],
                "outcome": states[-1].get("outcome"),
                "actions": self.of_kind("action"), "effects": self.of_kind("effect"),
                "coverage": self.coverage()}
