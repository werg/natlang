"""Read-only progress summaries for running natlang traces and experiment journals."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from applications.experiment_lab import inspect_journal


def inspect_trace(path: Path) -> dict:
    path = Path(path)
    counts = Counter()
    last = None
    active_requests = {}
    last_action = None
    last_action_result = None
    repeat_actions = 0
    tokens = 0
    prompt_tokens = 0
    largest_tool_schema = 0
    total_tool_schema = 0
    largest_prompt = 0
    started_requests = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A concurrent reader may encounter the writer's incomplete last line.
                break
            counts[event["kind"]] += 1
            last = event
            if event["kind"] == "model_request":
                key = (event.get("call_id"), event.get("turn"))
                if event.get("phase") == "start":
                    active_requests[key] = event.get("observed_at")
                    started_requests += 1
                    schema_bytes = event.get("tool_schema_bytes") or 0
                    largest_tool_schema = max(largest_tool_schema, schema_bytes)
                    total_tool_schema += schema_bytes
                elif event.get("phase") in ("end", "error"):
                    active_requests.pop(key, None)
                    tokens += event.get("completion_tokens") or 0
                    turn_prompt = event.get("prompt_tokens") or 0
                    prompt_tokens += turn_prompt
                    largest_prompt = max(largest_prompt, turn_prompt)
            elif event["kind"] == "action":
                signature = (event.get("call_id"), event.get("name"),
                             json.dumps(event.get("arguments"), sort_keys=True))
                repeat_actions = repeat_actions + 1 if signature == last_action else 1
                last_action = signature
                last_action_result = {"name": event.get("name"), "outcome": event.get("outcome"),
                                      "diagnostics": event.get("diagnostics")}
    if last is None:
        raise ValueError("empty trace")
    stamp = last.get("observed_at")
    age = None
    if stamp:
        age = max(0, round((datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()))
    return {"path": str(path), "events": sum(counts.values()),
            "last_seq": last.get("seq"), "last_kind": last.get("kind"),
            "last_observed_at": stamp, "seconds_since_event": age,
            "actions": counts["action"], "reductions": counts["reduction"],
            "model_requests_started": started_requests,
            "model_requests_in_flight": [{"call_id": call_id, "turn": turn, "started_at": started}
                                         for (call_id, turn), started in active_requests.items()],
            "completion_tokens_reported": tokens,
            "prompt_tokens_reported": prompt_tokens,
            "largest_prompt_tokens": largest_prompt,
            "largest_tool_schema_bytes": largest_tool_schema,
            "total_tool_schema_bytes_presented": total_tool_schema,
            "consecutive_identical_actions": repeat_actions,
            "last_action": last_action_result}


def inspect_run(*, journal: Path | None = None, trace_dir: Path | None = None) -> dict:
    if journal is None and trace_dir is None:
        raise ValueError("supply a journal or trace directory")
    result = {"journal": (inspect_journal(journal) if Path(journal).exists()
                           else {"path": str(journal), "status": "not_started"})
              if journal is not None else None,
              "traces": []}
    if trace_dir is not None:
        result["traces"] = [inspect_trace(path) for path in sorted(Path(trace_dir).rglob("*.jsonl"))]
    return result
