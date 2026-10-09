"""Source (a) of refine-judge: mine `refinement_check` and `refinement_shadow` events into unlabeled pairs.

Inputs are call-store roots (`calls.sqlite` plus `blobs/`, ts-host/src/calls/store-core.ts) and trace files (JSONL of
trace objects with an `events` list, or of bare events). The output is a list of pairs `(value, predicate)` for the
teacher judge to label (`ts-host/scripts/refine-data/cli.mjs label`); nothing is labelled here. The checker's own verdict
in the event is recorded as `observed` but never used as the label: it is the student-side judge's output, not ground truth.

Events carry only a 400-character preview of the value. A preview that was cut is completed from the call store when the
recorded output hashes to the event's `value_sha256`; otherwise the pair is counted as unrecoverable and left out.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .common import canonical_json_sha256_hex, normalize_predicate, predicate_id, sha256_hex

KINDS = ("refinement_check", "refinement_shadow")
TRUNCATED = re.compile(r" … \((\d+) chars\)$")


def events_of_trace_file(path: Path):
    """Events of a trace JSONL: lines that are `{events: [...]}` objects or bare events."""
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            batch = row["events"] if isinstance(row, dict) and isinstance(row.get("events"), list) else [row]
            for event in batch:
                if isinstance(event, dict) and event.get("kind") in KINDS:
                    yield {"event": event, "origin": {"trace": str(path), "line": number}}


class StoreReader:
    """Read-only view of a call store: events by call and recorded output values."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = sqlite3.connect(f"file:{self.root / 'calls.sqlite'}?mode=ro", uri=True)

    def blob(self, digest: str) -> str | None:
        try:
            return (self.root / "blobs" / digest[:2] / digest[2:]).read_text(encoding="utf-8")
        except OSError:
            return None

    def calls_with_events(self):
        return [row[0] for row in self.db.execute("SELECT call_id FROM calls WHERE events_hash IS NOT NULL ORDER BY started_at")]

    def events(self, call_id: str):
        row = self.db.execute("SELECT events_hash FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        text = self.blob(row[0]) if row and row[0] else None
        return [json.loads(line) for line in (text or "").splitlines() if line.strip()]

    def output(self, call_id: str):
        """The call's recorded output value, or (False, None) when it was not kept in full."""
        row = self.db.execute("SELECT record_hash FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        record = self.blob(row[0]) if row else None
        ref = json.loads(record).get("output") if record else None
        if not ref or not ref.get("complete"):
            return False, None
        text = self.blob(ref["hash"])
        return (True, json.loads(text)) if text is not None else (False, None)

    def close(self):
        self.db.close()


def events_of_store(reader: StoreReader):
    for call_id in reader.calls_with_events():
        for event in reader.events(call_id):
            if isinstance(event, dict) and event.get("kind") in KINDS:
                event.setdefault("call_id", call_id)
                yield {"event": event, "origin": {"store": str(reader.root), "call_id": call_id}}


def _walk(value, path: str):
    """Navigate an obligation path (`return`, `return/subject`, `return/2`) from a call's output."""
    for part in path.split("/")[1:]:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(path)
    return value


def recover_value(event: dict, readers: list[StoreReader]):
    """The full value of an event whose preview was cut, from a store that recorded it; (ok, value)."""
    call_id, path, want = event.get("call_id"), event.get("path"), event.get("value_sha256")
    if not (call_id and isinstance(path, str) and want):
        return False, None
    for reader in readers:
        ok, output = reader.output(call_id)
        if not ok:
            continue
        try:
            value = _walk(output, path)
        except KeyError:
            continue
        if canonical_json_sha256_hex(value) == want:
            return True, value
    return False, None


def mine(sources: list[dict], readers: list[StoreReader] | None = None) -> tuple[list[dict], dict]:
    """Unlabeled pairs from `sources` (items of `events_of_*`), deduplicated by (predicate, value digest)."""
    readers = readers or []
    pairs: dict[str, dict] = {}
    stats = {"events": 0, "check": 0, "shadow": 0, "duplicates": 0, "truncated_recovered": 0, "truncated_unrecoverable": 0,
             "without_value": 0, "shadow_disagreements": 0}
    for item in sources:
        event, origin = item["event"], item["origin"]
        stats["events"] += 1
        stats["check" if event["kind"] == "refinement_check" else "shadow"] += 1
        predicate, preview = event.get("predicate"), event.get("value")
        if not isinstance(predicate, str) or not isinstance(preview, str):
            stats["without_value"] += 1
            continue
        value, rendered_only = preview, True
        if TRUNCATED.search(preview):
            ok, full = recover_value(event, readers)
            if not ok:
                stats["truncated_unrecoverable"] += 1
                continue
            value, rendered_only = full, False
            stats["truncated_recovered"] += 1
        digest = event.get("value_sha256") or sha256_hex(preview.encode("utf-8"))
        key = predicate_id(predicate) + ":" + digest[:24]
        entry = pairs.get(key)
        if entry is None:
            entry = pairs[key] = {"id": f"mined:{key}", "predicate": normalize_predicate(predicate), "value": value,
                                  "value_rendered_only": rendered_only, "value_sha256": digest, "observed": [], "origins": [],
                                  "shadow": []}
        else:
            stats["duplicates"] += 1
        if event["kind"] == "refinement_check":
            entry["observed"].append({"outcome": event.get("outcome"), "probability": event.get("probability"),
                                      "judge": event.get("judge"), "source": event.get("source"), "phase": event.get("phase")})
        else:
            entry["shadow"].append({"crisp": event.get("crisp"), "nl": event.get("nl"), "agree": event.get("agree"),
                                    "probability": event.get("probability"), "judge": event.get("judge")})
            if event.get("agree") is False:
                stats["shadow_disagreements"] += 1
        if len(entry["origins"]) < 5:
            entry["origins"].append(origin)
    out = sorted(pairs.values(), key=lambda pair: pair["id"])
    for pair in out:
        pair["shadow_disagreement"] = any(item.get("agree") is False for item in pair["shadow"])
    stats["pairs"] = len(out)
    return out, stats
