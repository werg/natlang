"""Trajectory continuation records rebuilt from upstream SWE-rebench OpenHands trajectories (S1 §3.1, §4.1).

bgkit's compaction stores used a cons-cell spine and segment indices over these trajectories.
Here records are rebuilt at natural boundaries instead: an earlier stretch of the session (whole
messages, ending before an assistant turn) is the source, the task statement and the most recent
assistant turns with their observations are the consumer's context, and the next assistant
message is the target. The upstream OpenHands system prompt is scaffolding and is dropped;
the upstream tool schemas are kept for the consumer.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .common import Reject
from .records import VERSION, exact_refs_from, group_key, license_, seal, source, strip_markup

TRAJECTORIES = Path("/mnt/external/bgkit-data/trajectories_ext/swe_rebench_openhands/trajectories.parquet")
BENCH_KEYS = Path("/mnt/external/bgkit-data/trajectories_ext/bench_dedupe/bench_keys.json")
CONVERTER = "scripts/neuralese_data/swe_compaction.py@1"
UPSTREAM = "nebius/SWE-rebench-openhands-trajectories"
TEACHER = {"model": "Qwen/Qwen3-Coder-480B-A35B-Instruct", "provider": "nebius (dataset)"}

RECENT_TURNS = 2          # assistant turns (with their observations) the consumer sees verbatim
MIN_SOURCE_MESSAGES = 4   # an earlier stretch shorter than this is not worth a block
MAX_SOURCE_CHARS = 200_000


def load_bench_instances(path: Path = BENCH_KEYS) -> set:
    """Benchmark instance ids (SWE-bench and relatives) that must stay out of training."""
    if not path.exists():
        return set()
    keys = json.loads(path.read_text()).get("keys", [])
    return {k[1] for k in keys if isinstance(k, list) and len(k) >= 2 and k[0] == "id"}


def _message(m: dict) -> dict:
    out = {"role": m.get("role")}
    content = m.get("content")
    out["content"] = strip_markup(content) if isinstance(content, str) else None
    if m.get("name"):
        out["name"] = m["name"]
    if m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    calls = m.get("tool_calls")
    if calls:
        out["tool_calls"] = [{"id": c.get("id"), "type": c.get("type") or "function",
                              "function": {"name": (c.get("function") or {}).get("name"),
                                           "arguments": (c.get("function") or {}).get("arguments")}} for c in calls]
    return out


def prune(value):
    """Drop nulls that Arrow's struct unification adds to upstream tool schemas."""
    if isinstance(value, dict):
        return {k: prune(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [prune(v) for v in value]
    return value


def _text_of(messages: list) -> str:
    parts = []
    for m in messages:
        if m.get("content"):
            parts.append(m["content"])
        for c in m.get("tool_calls") or []:
            parts.append(str((c.get("function") or {}).get("arguments") or ""))
    return "\n".join(parts)


def windows_for(trajectory: list, windows: int, seed: str) -> list[tuple[int, int]]:
    """(recent_start, target) pairs at assistant-turn boundaries, chosen deterministically."""
    assistant = [i for i, m in enumerate(trajectory) if i >= 2 and m.get("role") == "assistant"]
    candidates = [(assistant[k - RECENT_TURNS], assistant[k]) for k in range(RECENT_TURNS, len(assistant))
                  if assistant[k - RECENT_TURNS] - 2 >= MIN_SOURCE_MESSAGES]
    if not candidates:
        return []
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(candidates)), min(windows, len(candidates))))
    return [candidates[p] for p in picks]


def convert_window(row_index: int, row: dict, recent_start: int, target_index: int, bench: set) -> dict:
    trajectory = row["trajectory"]
    if len(trajectory) < 2 or trajectory[0].get("role") != "system" or trajectory[1].get("role") != "user":
        raise Reject("unexpected-trajectory-head")
    task = _message(trajectory[1])
    earlier = [_message(m) for m in trajectory[2:recent_start]]
    recent = [_message(m) for m in trajectory[recent_start:target_index]]
    target = _message(trajectory[target_index])
    if not earlier:
        raise Reject("empty-source")
    source_text = _text_of(earlier)
    if len(source_text) > MAX_SOURCE_CHARS:
        raise Reject(f"oversize-source: {len(source_text)} chars")
    if not (target.get("content") or target.get("tool_calls")):
        raise Reject("empty-target")
    resolved = bool(row.get("resolved"))
    instance, repo = row.get("instance_id"), row.get("repo")
    if not instance or not repo:
        raise Reject("missing-group: instance or repo")
    protected = instance in bench
    record = {
        "version": VERSION,
        "id": f"upstream:swe-rebench-openhands:{row['trajectory_id']}:{target_index}",
        "family": "trajectory_continuation_swe",
        "task": "continue",
        "sources": [source("trajectory", messages=earlier, exact_refs=exact_refs_from(source_text),
                           meta={"span": [2, recent_start]})],
        "writer": {
            "instructions": "Read this earlier part of the coding session so that the task can be continued from the most recent steps.",
            "result_type": "Neuralese<SessionHistory>",
            "context": [task],
        },
        "consumer": {"context": [task, *recent], "tools": prune(list(row.get("tools") or [])),
                     "withheld": ["sources"]},
        "target": {"kind": "message", "value": target, "alternatives": []},
        "outcome": ({"label": "checked", "checked": "swe-rebench-tests"} if resolved else
                    {"label": "teacher", "checked": None})
                   | {"details": {
                       "resolved": resolved,
                       "exit_status": row.get("exit_status"),
                       "checker_evidence": {
                           "origin": "upstream_resolved_status",
                           "raw_status_field": "resolved",
                           "raw_status": row.get("resolved"),
                           "local_execution": False,
                       },
                   }},
        "lineage": {"project": "upstream", "store": UPSTREAM, "store_version": None, "row": row_index,
                    "upstream": UPSTREAM, "upstream_id": row["trajectory_id"], "upstream_revision": None,
                    "teacher": TEACHER, "converter": CONVERTER, "sha256": "",
                    "notes": {"instance_id": instance, "upstream_system_prompt": "dropped (OpenHands scaffolding)",
                              "benchmark_protected": protected}},
        "license": license_("CC-BY-4.0", False, "Trajectories CC-BY-4.0; embedded repository code under each repository's licence."),
        "split": "test" if protected else "train",
        "split_groups": [group_key("repo", repo), group_key("swe-instance", instance)],
    }
    return seal(record)


def iter_rows(limit_trajectories: int | None, seed: int = 0, path: Path = TRAJECTORIES):
    from .common import parquet_rows

    yield from parquet_rows(path, limit=limit_trajectories, seed=seed)
