"""Fail-closed source receipts for provider-authored, student-forced native turns."""
from __future__ import annotations

import hashlib
import json


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def validate_provider_teacher_source(source, assistant, opening, tools, native_text_spans,
                                     *, source_opening=None, native_text_bindings=None):
    """Validate the Luna branch without weakening the pinned Qwen source contract.

    File bytes and row membership are checked by the source-loader before this
    transport receipt is created. Here we bind the exact provider identity,
    complete source artifacts, production prompt/target, and each host-authored
    typed text result to the forced replay request.
    """
    keys = {
        "schema", "source_path", "source_file_sha256", "source_row_id", "source_row_sha256",
        "source_turn_index", "source_turn_sha256", "task_sha256", "program_sha256",
        "prompt_sha256", "source_prompt_sha256", "target_sha256", "raw_target_sha256", "target_binding_sha256",
        "plan_path", "plan_sha256", "runtime_path",
        "runtime_receipt_sha256", "split", "source_groups", "trace_sha256", "request_sha256",
        "response_sha256", "transport_provenance_sha256", "actor", "typed_text_writes",
    }
    if not isinstance(source, dict) or source.get("schema") != "natlang.provider-teacher-turn-source/1" or set(source) != keys:
        raise ValueError("provider teacher source receipt needs the complete declared Luna source binding")
    actor = source.get("actor")
    if (not isinstance(actor, dict) or set(actor) != {"model", "provider", "role", "transport"}
            or actor != {"model": "gpt-6-luna", "provider": "openai-codex", "role": "teacher",
                         "transport": "pi-provider"}):
        raise ValueError("provider teacher source receipt does not identify the actual Luna teacher actor")
    if (source.get("split") != "train" or not isinstance(source.get("source_groups"), list)
            or not source["source_groups"] or any(not isinstance(group, str) or not group for group in source["source_groups"])
            or len(set(source["source_groups"])) != len(source["source_groups"])):
        raise ValueError("provider teacher source receipt needs its exact training split and source groups")
    for key in ("source_file_sha256", "source_row_sha256", "source_turn_sha256", "task_sha256",
                "program_sha256", "prompt_sha256", "source_prompt_sha256", "target_sha256", "raw_target_sha256",
                "target_binding_sha256", "plan_sha256",
                "runtime_receipt_sha256", "trace_sha256", "request_sha256", "response_sha256",
                "transport_provenance_sha256"):
        if not _valid_sha(source.get(key)):
            raise ValueError(f"provider teacher source receipt has no exact {key}")
    if (not isinstance(source.get("source_path"), str) or not source["source_path"]
            or not isinstance(source.get("source_row_id"), str) or not source["source_row_id"]
            or type(source.get("source_turn_index")) is not int or source["source_turn_index"] < 0
            or not isinstance(source.get("plan_path"), str) or not source["plan_path"]
            or not isinstance(source.get("runtime_path"), str) or not source["runtime_path"]):
        raise ValueError("provider teacher source receipt has incomplete source artifact locations")
    writes = source.get("typed_text_writes")
    spans = native_text_spans
    if not isinstance(writes, list) or not isinstance(spans, list):
        raise ValueError("provider teacher source receipt needs its typed text write inventory")
    if _sha(_canonical({"messages": opening, "tools": tools or []})) != source["prompt_sha256"]:
        raise ValueError("provider teacher source prompt differs from the exact production prompt")
    if source_opening is not None or native_text_bindings is not None:
        if not isinstance(source_opening, list) or not isinstance(native_text_bindings, list):
            raise ValueError("provider teacher native replay needs its exact original prompt and encoder bindings")
        if _sha(_canonical({"messages": source_opening, "tools": tools or []})) != source.get("source_prompt_sha256"):
            raise ValueError("provider teacher native replay changed its original provider context")
        transformed = json.loads(json.dumps(source_opening, ensure_ascii=False))
        used = set()
        for binding in native_text_bindings:
            required = {"teacher_block_id", "body_sha256", "native_block_id", "source_occurrence", "source_capture"}
            if not isinstance(binding, dict) or set(binding) != required:
                raise ValueError("provider teacher native text encoder binding is malformed")
            if (not isinstance(binding["teacher_block_id"], str) or not binding["teacher_block_id"].startswith("nzt1_")
                    or not isinstance(binding["native_block_id"], str) or not binding["native_block_id"].startswith("nzs1_")
                    or not isinstance(binding["source_occurrence"], str) or not binding["source_occurrence"]
                    or not _valid_sha(binding["body_sha256"]) or not isinstance(binding["source_capture"], dict)):
                raise ValueError("provider teacher native text encoder binding has invalid ids or body hash")
            matching_writes = [write for write in writes if write["block_id"] == binding["teacher_block_id"]
                               and write["body_sha256"] == binding["body_sha256"]]
            if len(matching_writes) != 1 or binding["teacher_block_id"] in used:
                raise ValueError("provider teacher native encoder does not match one authenticated host text write")
            used.add(binding["teacher_block_id"])
            capture = binding["source_capture"]
            if (not isinstance(capture.get("policy"), dict)
                    or not isinstance(capture.get("segments"), list)):
                raise ValueError("provider teacher encoder lacks an actual native source capture")
            positions = []
            def find_refs(value, path=()):
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        find_refs(item, path + (index,))
                elif isinstance(value, dict):
                    if value.get("type") == "teacher_text" and value.get("id") == binding["teacher_block_id"]:
                        positions.append(path)
                    else:
                        for key, item in value.items():
                            find_refs(item, path + (key,))
            find_refs(transformed)
            if len(positions) != matching_writes[0]["context_occurrences"]:
                raise ValueError("provider teacher text occurrence count differs from its authenticated read receipt")
            for path in positions:
                parent, last = transformed, path[-1]
                for part in path[:-1]: parent = parent[part]
                parent[last] = {"type": "native_state", "id": binding["native_block_id"],
                                "source_occurrence": binding["source_occurrence"]}
        if used != {write["block_id"] for write in writes}:
            raise ValueError("provider teacher native replay omitted an authenticated typed text write")
        if transformed != opening:
            raise ValueError("provider teacher native prompt differs beyond the exact authenticated text encoder rewrite")
    if not isinstance(assistant, dict) or _sha(_canonical(assistant)) != source["target_sha256"]:
        raise ValueError("provider teacher source target differs from the exact normalized assistant action")
    spans_by_path = {}
    for span in spans:
        if (not isinstance(span, dict) or not isinstance(span.get("path"), list)
                or not isinstance(span.get("text"), str) or not isinstance(span.get("marker"), str)):
            raise ValueError("provider teacher native text span is malformed")
        path = tuple(span["path"])
        if path in spans_by_path:
            raise ValueError("provider teacher native text span paths must be unique")
        spans_by_path[path] = span
    seen_paths = set()
    for write in writes:
        if (not isinstance(write, dict)
                or set(write) != {"block_id", "body_sha256", "source", "writer_call_id", "writer_node",
                                  "host_result_receipt_sha256", "raw_response_sha256",
                                  "context_receipt_sha256", "target_span_path", "context_occurrences"}
                or not isinstance(write.get("block_id"), str) or not write["block_id"].startswith("nzt1_")
                or write.get("source") not in {"return_result", "eval-return", "eval-finish"}
                or not isinstance(write.get("writer_call_id"), str) or not write["writer_call_id"]
                or not isinstance(write.get("writer_node"), str) or not write["writer_node"]
                or not _valid_sha(write.get("body_sha256"))
                or not _valid_sha(write.get("host_result_receipt_sha256"))
                or not _valid_sha(write.get("context_receipt_sha256"))
                or not _valid_sha(write.get("raw_response_sha256"))
                or type(write.get("context_occurrences")) is not int
                or write["context_occurrences"] < 1):
            raise ValueError("provider teacher typed text write is not bound to one exact host result")
        path = write.get("target_span_path")
        if path is not None:
            if (not isinstance(path, list) or not path or any(not isinstance(part, (str, int)) or isinstance(part, bool)
                                                               for part in path)
                    or tuple(path) in seen_paths):
                raise ValueError("provider teacher target span path is malformed or repeated")
            seen_paths.add(tuple(path))
            span = spans_by_path.get(tuple(path))
            if span is None or _sha(span["text"].encode("utf-8")) != write["body_sha256"]:
                raise ValueError("provider teacher typed text write body differs from its exact target span")
    if seen_paths != set(spans_by_path):
        raise ValueError("provider teacher source receipt does not cover every exact native text span")
    return True
