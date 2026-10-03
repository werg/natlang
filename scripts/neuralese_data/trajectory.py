"""Multi-turn trajectories → `continue` port records at natural boundaries (S1 §3, §4.1).

Every agent corpus is first normalised into a `Trajectory`: role/content messages (system, user,
assistant, tool), with assistant reasoning kept apart from content and tool calls as structures.
`head` leading messages are the task statement (system prompt and first request). Records are
cut at assistant-turn boundaries:

* source: an earlier stretch of the session (whole messages, after the head, ending before the
  recent turns), trimmed from the front at a message boundary when it exceeds MAX_SOURCE_CHARS;
* writer context: the task statement only;
* consumer context: the task statement and the most recent RECENT_TURNS assistant turns with their
  observations, verbatim;
* target: the next assistant message.

Nothing after the target is ever an input, and the writer never sees the recent turns or the
target. Background documents that the task supplies up front (protocols, policies, worked
examples) can ride along as extra sources; with them, a window may start at the first assistant
turn, so the writer reads only the background.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field

from .common import Reject
from .records import VERSION, exact_refs_from, license_, normalize_text, seal, source, strip_markup

RECENT_TURNS = 2
MIN_SOURCE_MESSAGES = 2
MAX_SOURCE_CHARS = 120_000
MAX_CONSUMER_CHARS = 120_000

_THINK_RE = re.compile(r"^\s*(?:<think>)?(?P<think>.*?)</think>\s*", re.S)


@dataclass
class Trajectory:
    id: str                      # record id prefix, e.g. "upstream:agenttrove:<run>"
    family: str
    messages: list
    head: int                    # leading messages that state the task
    noun: str                    # what the session is, for writer instructions ("coding session")
    lineage: dict                # project, store, store_version, row, upstream, upstream_id, upstream_revision, teacher, converter, notes
    license: dict
    split: str
    split_groups: list
    outcome: dict
    tools: list | None = None
    background: list = field(default_factory=list)   # extra source dicts (protocols, policies)


def split_think(text: str | None) -> tuple[str | None, str | None]:
    """Separate a leading `<think>…</think>` (or a bare `…</think>`) block from message content."""
    if not isinstance(text, str) or "</think>" not in text:
        return None, text
    m = _THINK_RE.match(text)
    if not m:
        return None, text
    return m.group("think").strip() or None, text[m.end():]


def message(role: str, content: str | None, *, reasoning: str | None = None, tool_calls: list | None = None,
            name: str | None = None, tool_call_id: str | None = None) -> dict:
    out: dict = {"role": role, "content": strip_markup(content) if isinstance(content, str) else None}
    if reasoning and reasoning.strip():
        out["reasoning"] = strip_markup(reasoning).strip()
    if tool_calls:
        out["tool_calls"] = tool_calls
    if name:
        out["name"] = name
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    return out


def tool_call(name: str, arguments, call_id: str | None = None) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    out = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id:
        out["id"] = call_id
    return out


def text_of(messages: list) -> str:
    parts = []
    for m in messages:
        if m.get("reasoning"):
            parts.append(m["reasoning"])
        if m.get("content"):
            parts.append(m["content"])
        for c in m.get("tool_calls") or []:
            parts.append(str((c.get("function") or {}).get("arguments") or ""))
    return "\n".join(parts)


def has_payload(m: dict) -> bool:
    return bool((m.get("content") or "").strip() or m.get("tool_calls") or (m.get("reasoning") or "").strip())


def windows(t: Trajectory, count: int, seed: str) -> list[tuple[int, int]]:
    """(recent_start, target) pairs at assistant-turn boundaries, chosen deterministically from the seed."""
    assistant = [i for i, m in enumerate(t.messages) if i >= t.head and m.get("role") == "assistant" and has_payload(m)]
    candidates = []
    # Short sessions keep fewer recent turns verbatim (down to none) rather than producing no window.
    for recent in range(RECENT_TURNS, -1, -1):
        for k in range(recent, len(assistant)):
            start = assistant[k - recent] if recent else assistant[k]
            if start - t.head >= MIN_SOURCE_MESSAGES:
                candidates.append((start, assistant[k]))
        if candidates:
            break
    if t.background and assistant:
        candidates.append((t.head, assistant[0]))  # the first step, informed by background only
    if not candidates:
        return []
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(candidates)), min(count, len(candidates))))
    return [candidates[p] for p in picks]


def _trim_front(messages: list, limit: int) -> tuple[list, int]:
    """Drop whole messages from the front until the stretch fits; return (kept, dropped)."""
    sizes = [len(text_of([m])) + 1 for m in messages]  # +1 for the joining newline
    total, start = sum(sizes), 0
    while start < len(messages) and total > limit:
        total -= sizes[start]
        start += 1
    return messages[start:], start


def record(t: Trajectory, recent_start: int, target_index: int) -> dict:
    head = t.messages[:t.head]
    earlier, dropped = _trim_front(t.messages[t.head:recent_start], MAX_SOURCE_CHARS)
    recent = t.messages[recent_start:target_index]
    target = t.messages[target_index]
    if target.get("role") != "assistant" or not has_payload(target):
        raise Reject("empty-target")
    sources = list(t.background)
    if earlier:
        stretch = text_of(earlier)
        sources.append(source("trajectory", messages=earlier, exact_refs=exact_refs_from(stretch),
                              meta={"span": [t.head + dropped, recent_start], "dropped_front_messages": dropped}))
    elif not t.background:
        raise Reject("empty-source")
    consumer = [*head, *recent]
    if len(text_of(consumer)) > MAX_CONSUMER_CHARS:
        raise Reject(f"oversize-consumer: {len(text_of(consumer))} chars")
    if earlier:
        instructions = f"Read this earlier part of the {t.noun} so that the task can be continued from the most recent steps."
    else:
        instructions = f"Read this background material so that the {t.noun} can be carried out."
    lineage = dict(t.lineage)
    lineage["sha256"] = ""
    consumer_block = {"context": consumer, "withheld": ["sources"]}
    if t.tools:
        consumer_block["tools"] = t.tools
    rec = {
        "version": VERSION,
        "id": f"{t.id}:{target_index}",
        "family": t.family,
        "task": "continue",
        "sources": sources,
        "writer": {"instructions": instructions,
                   "result_type": "Neuralese<SessionHistory>" if earlier else "Neuralese<Background>",
                   "context": head},
        "consumer": consumer_block,
        "target": {"kind": "message", "value": target, "alternatives": []},
        "outcome": t.outcome,
        "lineage": lineage,
        "license": t.license,
        "split": t.split,
        "split_groups": t.split_groups,
    }
    return seal(rec)


def records_for(t: Trajectory, count: int, sink, store: str, row) -> None:
    """Emit up to `count` windows of a trajectory into a Sink, rejecting per window."""
    picks = windows(t, count, t.id)
    if not picks:
        sink.reject(store, row, "too-short-trajectory")
        return
    for recent_start, target in picks:
        try:
            sink.accept(record(t, recent_start, target))
        except Reject as exc:
            sink.reject(store, f"{row}:{target}", str(exc))


def task_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:20]


def licence(spdx: str, noncommercial: bool = False, notes: str = "") -> dict:
    return license_(spdx, noncommercial, notes)
