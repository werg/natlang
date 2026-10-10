#!/usr/bin/env python3
"""Exact source-to-saved-child-request joins for provider campaign audits.

Request text, capture snapshots, and transport hashes are separate evidence. This
module never treats parent/sibling context or opaque hashes as source visibility.
For item-specific claims, callers must bind that item to explicit invocation IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "neuralese"))
from natlang_neuralese.common.hashing import sha256_hex  # noqa: E402


def _state_forms(state: str) -> tuple[str, ...]:
    forms = {state}
    for ensure_ascii in (False, True):
        forms.add(json.dumps(state, ensure_ascii=ensure_ascii))
        forms.add(json.dumps(state, ensure_ascii=ensure_ascii, separators=(",", ":")))
    return tuple(sorted((x for x in forms if x), key=lambda x: (-len(x), x)))


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _texts(value: Any) -> Iterable[str]:
    for _, node in _walk(value):
        if isinstance(node, str):
            yield node


def _bounded_json_fields(messages: Any) -> list[dict[str, Any]]:
    """Decode only saved function-argument JSON envelopes, exactly one level.

    These strings are structurally identified by the rendered chat schema. We do
    not try to parse arbitrary message text or recursively decode string fields.
    The saved messages remain rendered-request evidence, not original wire bytes.
    """
    found: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return found
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call_index, call in enumerate(calls):
            fn = call.get("function") if isinstance(call, Mapping) else None
            encoded = fn.get("arguments") if isinstance(fn, Mapping) else None
            if not isinstance(encoded, str):
                continue
            try:
                decoded = json.loads(encoded)
            except (json.JSONDecodeError, TypeError):
                continue
            for path, value in _walk(decoded, f"$.messages[{message_index}].tool_calls[{call_index}].function.arguments"):
                if isinstance(value, str):
                    found.append({"path": path, "depth": 1, "value": value})
    return found


def _snapshot_equalities(invocation: Mapping[str, Any], state: str) -> list[str]:
    """Read named snapshots only from this exact invocation ledger row."""
    paths: list[str] = []
    for root_name in ("runtime_capture_snapshots", "capture_snapshots"):
        for path, node in _walk(invocation.get(root_name), f"$.{root_name}"):
            if isinstance(node, str) and node == state:
                paths.append(path)
            elif isinstance(node, Mapping) and node.get("value") == state:
                paths.append(path + ".value")
    site = invocation.get("inline_instruction_site")
    if isinstance(site, Mapping):
        for root_name in ("runtime_capture_snapshots", "capture_snapshots"):
            for path, node in _walk(site.get(root_name), f"$.inline_instruction_site.{root_name}"):
                if isinstance(node, str) and node == state:
                    paths.append(path)
                elif isinstance(node, Mapping) and node.get("value") == state:
                    paths.append(path + ".value")
    return sorted(set(paths))


def _record_child_requests(record: Mapping[str, Any], record_index: int) -> list[dict[str, Any]]:
    trajectory = record.get("trajectory") or []
    ledger = (record.get("outcome") or {}).get("invocation_ledger") or []
    ledger_by_id = {x.get("invocation_id"): x for x in ledger if isinstance(x, Mapping)}
    ir = (record.get("task") or {}).get("program_ir") or {}
    program_id = ir.get("id") or record.get("program_id")
    out = []
    for trajectory_index, entry in enumerate(trajectory):
        if not isinstance(entry, Mapping):
            continue
        response = entry.get("model_response") or {}
        transport = response.get("transport_provenance") or {}
        messages = transport.get("rendered_messages")
        if not isinstance(messages, list):
            continue
        invocation_id = entry.get("invocation_id")
        invocation = ledger_by_id.get(invocation_id, {})
        if not (invocation.get("inline_instruction_site") or invocation.get("origin")):
            continue
        out.append({
            "event_id": f"record-{record_index}:trajectory-{trajectory_index}",
            "record_index": record_index,
            "trajectory_index": trajectory_index,
            "program_id": program_id,
            "invocation_id": invocation_id,
            "messages": messages,
            "request_text": "\n".join(_texts(messages)),
            "nested_json_fields": _bounded_json_fields(messages),
            "capture_snapshot_paths_by_state": None,
            "transport_hashes": {
                "raw_request_sha256": transport.get("raw_request_sha256"),
                "rendered_request_sha256": transport.get("rendered_request_sha256"),
                "response_sha256": entry.get("raw_response_sha256") or transport.get("raw_response_sha256"),
            },
            "rendered_messages_count": len(messages),
            "truncated_messages": bool(transport.get("rendered_messages_truncated") or transport.get("messages_truncated")),
        })
    return out


def analyze_item_visibility(
    items: Iterable[Mapping[str, Any]],
    result_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join source items to saved child requests without crossing item bindings.

    Each item needs `id`, `state`, and either:
      * `invocation_ids`: explicit invocation IDs bound to this source item; or
      * `program_id`: then matches are reported only as program-level evidence and
        are never promoted to an item-specific visibility class.

    For capture-only item evidence, provide invocation_ids; snapshots are read only
    from those exact invocation ledger records.
    """
    item_list = list(items)
    records = list(result_records)
    all_events: list[dict[str, Any]] = []
    program_events: dict[str, list[dict[str, Any]]] = {}
    invocation_rows: dict[str, list[Mapping[str, Any]]] = {}
    for record_index, record in enumerate(records):
        ledger = (record.get("outcome") or {}).get("invocation_ledger") or []
        for row in ledger:
            if isinstance(row, Mapping) and row.get("invocation_id"):
                invocation_rows.setdefault(row["invocation_id"], []).append(row)
        events = _record_child_requests(record, record_index)
        for event in events:
            all_events.append(event)
            if event.get("program_id"):
                program_events.setdefault(event["program_id"], []).append(event)
    by_invocation: dict[str, list[dict[str, Any]]] = {}
    for event in all_events:
        if event.get("invocation_id"):
            by_invocation.setdefault(event["invocation_id"], []).append(event)

    per_item = []
    for item in item_list:
        item_id, state, program_id = item.get("id"), item.get("state"), item.get("program_id")
        raw_ids = item.get("invocation_ids")
        explicitly_bound = isinstance(raw_ids, (list, tuple, set))
        allowed_ids = set(raw_ids or []) if explicitly_bound else set()
        if not isinstance(state, str) or (not explicitly_bound and not isinstance(program_id, str)):
            per_item.append({"id": item_id, "program_id": program_id, "class": "unbound-input"})
            continue
        events = []
        if explicitly_bound:
            for invocation_id in sorted(allowed_ids):
                events.extend(by_invocation.get(invocation_id, []))
        else:
            events = list(program_events.get(program_id, []))
        forms = _state_forms(state)
        request_matches = []
        for event in events:
            matching = next((form for form in forms if form in event["request_text"]), None)
            if matching is not None:
                request_matches.append({
                    "event_id": event["event_id"],
                    "invocation_id": event["invocation_id"],
                    "match_form": "literal" if matching == state else "json-escaped",
                    "match_source": "rendered-message-text",
                    **event["transport_hashes"],
                    "rendered_messages_count": event["rendered_messages_count"],
                    "truncated_messages": event["truncated_messages"],
                })
            nested_match = next(
                ((field, form) for field in event.get("nested_json_fields", [])
                 for form in forms if form in field["value"]),
                None,
            )
            if nested_match is not None:
                field, matching = nested_match
                request_matches.append({
                    "event_id": event["event_id"],
                    "invocation_id": event["invocation_id"],
                    "match_form": "literal" if matching == state else "json-escaped",
                    "match_source": "one-level-json-field",
                    "json_depth": field["depth"],
                    "json_path": field["path"],
                    **event["transport_hashes"],
                    "rendered_messages_count": event["rendered_messages_count"],
                    "truncated_messages": event["truncated_messages"],
                })
        snapshot_matches = []
        if explicitly_bound:
            seen_rows = set()
            for invocation_id in allowed_ids:
                for row in invocation_rows.get(invocation_id, []):
                    key = (invocation_id, id(row))
                    if key in seen_rows:
                        continue
                    seen_rows.add(key)
                    for path in _snapshot_equalities(row, state):
                        snapshot_matches.append({"invocation_id": invocation_id, "path": path})
        if not explicitly_bound:
            cls = "program-level-only-request-match" if request_matches else "program-level-no-request-match"
        elif request_matches and snapshot_matches:
            cls = "item-bound-rendered-request-and-capture-snapshot"
        elif request_matches:
            cls = "item-bound-rendered-request"
        elif snapshot_matches:
            cls = "item-bound-capture-snapshot-only"
        else:
            cls = "item-bound-no-exact-request-or-snapshot-evidence"
        per_item.append({
            "id": item_id,
            "program_id": program_id,
            "invocation_binding": "explicit-invocation-ids" if explicitly_bound else "program-only-unbound-to-item",
            "state_sha256": sha256_hex(state.encode("utf-8")),
            "state_chars": len(state),
            "class": cls,
            "request_matches": request_matches,
            "capture_snapshot_matches": snapshot_matches,
        })
    class_counts: dict[str, int] = {}
    for item in per_item:
        class_counts[item["class"]] = class_counts.get(item["class"], 0) + 1
    return {
        "items": per_item,
        "class_counts": class_counts,
        "child_request_turn_count": len(all_events),
        "child_request_turns_with_raw_request_hash": sum(bool(x["transport_hashes"].get("raw_request_sha256")) for x in all_events),
        "child_request_turns_with_rendered_request_hash": sum(bool(x["transport_hashes"].get("rendered_request_sha256")) for x in all_events),
        "child_request_turns_with_saved_rendered_messages": len(all_events),
        "transport_hashes_are_not_original_wire_bytes": True,
    }
