"""Canonical exact write-site extraction for native trajectory records.

The extractor is pure data handling: JSON write markers retain their exact
argument path and inline code writers are exposed only after validating the
compiler-attested sidecar against the original arguments string.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping
import hashlib
import json


def direct_result_event_name(block_id: str, trajectory_id: str, call_id: str, node: str) -> str:
    """Keep same-content writes distinct by their authenticated producer event."""
    event = hashlib.sha256(json.dumps([trajectory_id, call_id, node], ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    return f"soft-state:{block_id}@{event}"


@dataclass(frozen=True)
class WriteSite:
    tool: str
    before_arguments: Mapping[str, Any]
    argument: str
    value_path: tuple[str | int, ...]
    name: str
    descriptor: Mapping[str, Any]
    source_kind: str
    arguments: Mapping[str, Any]
    inline_prefix: str | None = None
    schema: Any = None
    source_span: tuple[int, int] | None = None
    sidecar_code_sha256: str | None = None

    @property
    def source(self) -> str:
        value = self.descriptor.get("source")
        return value if isinstance(value, str) else ""

    @property
    def value_type(self) -> str:
        value = self.descriptor.get("type", "Neuralese<string>")
        return value if isinstance(value, str) else "Neuralese<string>"

    @property
    def legacy_path(self) -> tuple[str | int, ...]:
        # Historically inline writers had no JSON value path; preserve that
        # contract for the existing trajectory renderer.
        return () if self.source_kind == "inline-code-sidecar" else self.value_path


def _write_markers(value: Any, path: tuple[str | int, ...] = ()):
    """Yield exact JSON paths and well-formed native write descriptors."""
    if isinstance(value, dict) and "$write" in value:
        site = value["$write"]
        if isinstance(site, dict) and isinstance(site.get("name"), str) and isinstance(site.get("source"), str):
            yield path, site
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _write_markers(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _write_markers(item, path + (index,))


def extract_write_sites_from_calls(calls: Any) -> tuple[WriteSite, ...]:
    """Return writers from a native tool-call sequence.

    For a JSON marker, ``value_path`` includes its top-level argument. An
    inline sidecar writer is attached to the `code` argument and carries the
    validated source body/prefix; its legacy JSON value path remains empty.
    Duplicate writer names are retained here so callers can report their own
    existing ambiguity error rather than silently selecting one.
    """
    sites: list[WriteSite] = []
    for call in calls or []:
        function = call["function"]
        tool_name = function.get("name")
        arguments_json = function["arguments"]
        arguments = json.loads(arguments_json)
        before: dict[str, Any] = {}
        for argument, value in arguments.items():
            for relative_path, descriptor in _write_markers(value):
                name = descriptor["name"]
                sites.append(WriteSite(
                    tool=tool_name,
                    before_arguments=dict(before),
                    argument=argument,
                    value_path=(argument,) + relative_path,
                    name=name,
                    descriptor=descriptor,
                    source_kind="json-write-marker",
                    arguments=arguments,
                ))
            before[argument] = value

        if "neuralese_code" in call:
            from .inline_instructions import validate_inline_instruction_code

            checked = validate_inline_instruction_code(arguments_json, call["neuralese_code"])
            if not checked.valid:
                raise ValueError("invalid inline instruction code sidecar: " + str(checked.reason))
            assert checked.value is not None
            code_before: dict[str, Any] = {}
            for key, value in arguments.items():
                if key == "code":
                    break
                code_before[key] = value
            descriptors = {
                item.get("$write", {}).get("name"): item["$write"]
                for item in checked.value.parts
                if isinstance(item.get("$write"), Mapping)
            }
            for write in checked.value.writes:
                descriptor = descriptors.get(write.name)
                if descriptor is None:
                    descriptor = {"name": write.name, "type": write.type, "source": write.source}
                    if write.code_source is not None:
                        descriptor["code_source"] = write.code_source
                pair = checked.value.prefix_and_body(write.name)
                sites.append(WriteSite(
                    tool=tool_name,
                    before_arguments=dict(code_before),
                    argument="code",
                    value_path=("code",),
                    name=write.name,
                    descriptor=descriptor,
                    source_kind="inline-code-sidecar",
                    arguments=arguments,
                    inline_prefix=pair[0] if pair is not None else None,
                    source_span=(write.start, write.end),
                    sidecar_code_sha256=checked.value.code_sha256,
                ))
    return tuple(sites)


def extract_write_sites(record: Mapping[str, Any]) -> tuple[WriteSite, ...]:
    """Return target writers from JSON markers and compiler-attested text sites."""
    target = record.get("target") or {}
    sites = [replace(site, schema=argument_schema(record, site.tool, site.argument, site.value_path))
             for site in extract_write_sites_from_calls(target.get("tool_calls") or [])]
    sidecar = target.get("assistant_text_result")
    if sidecar is not None:
        import hashlib
        import re
        source_ref = record.get("source_ref") or {}
        decision = record.get("decision") or {}
        assistant = decision.get("assistant") or {}
        receipts = [item for item in assistant.get("typed_result_writes", []) or []
                    if isinstance(item, Mapping) and item.get("source") == "assistant-text-return"
                    and item.get("source_kind") == "typed-text-result"]
        raw = target.get("content")
        if (not isinstance(sidecar, Mapping) or sidecar.get("schema") != "natlang.assistant-text-result-write/1"
                or target.get("role") != "assistant" or target.get("tool_calls")
                or not isinstance(raw, str) or not isinstance(sidecar.get("body_source"), str)
                or type(sidecar.get("body_start")) is not int or type(sidecar.get("body_end")) is not int
                or sidecar["body_start"] < 0 or sidecar["body_end"] < sidecar["body_start"]
                or raw[sidecar["body_start"] : sidecar["body_end"]] != sidecar["body_source"]
                or hashlib.sha256(raw.encode("utf-8")).hexdigest() != sidecar.get("raw_assistant_text_sha256")
                or hashlib.sha256(sidecar["body_source"].encode("utf-8")).hexdigest() != sidecar.get("body_sha256")
                or sidecar.get("source_row_sha256") != source_ref.get("source_row_sha256")
                or sidecar.get("writer_call_id") != source_ref.get("invocation_id")
                or not re.fullmatch(r"[0-9a-f]{64}", str(source_ref.get("source_row_sha256")))
                or not isinstance(source_ref.get("trajectory_id"), str)
                or not isinstance(sidecar.get("writer_node"), str)
                or not isinstance(sidecar.get("block_id"), str) or not sidecar["block_id"].startswith("nz1_")
                or not re.fullmatch(r"[0-9a-f]{64}", str(sidecar.get("request_sha256")))
                or not re.fullmatch(r"[0-9a-f]{64}", str(sidecar.get("raw_response_sha256")))):
            raise ValueError("invalid assistant-text-result writer sidecar")
        if (assistant.get("content") != raw or len(receipts) != 1
                or receipts[0].get("result_type") != "Neuralese<string>"
                or receipts[0].get("result_path") != ["return"]
                or receipts[0].get("invocation_id") != sidecar.get("writer_call_id")
                or receipts[0].get("writer_call_id") != sidecar.get("writer_call_id")
                or receipts[0].get("writer_node") != sidecar.get("writer_node")
                or receipts[0].get("block_id") != sidecar.get("block_id")
                or receipts[0].get("source_row_sha256") != sidecar.get("source_row_sha256")
                or receipts[0].get("body_source") != sidecar.get("body_source")
                or receipts[0].get("body_sha256") != sidecar.get("body_sha256")
                or receipts[0].get("body_start") != sidecar.get("body_start")
                or receipts[0].get("body_end") != sidecar.get("body_end")
                or receipts[0].get("raw_assistant_text_sha256") != sidecar.get("raw_assistant_text_sha256")
                or receipts[0].get("request_sha256") != sidecar.get("request_sha256")
                or receipts[0].get("raw_response_sha256") != sidecar.get("raw_response_sha256")
                or decision.get("source_raw_response_sha256") != sidecar.get("raw_response_sha256")):
            raise ValueError("assistant-text-result sidecar is not bound to its materializer receipt")
        # This is a real assistant result, not a synthesized tool invocation.
        # The empty argument path tells the trajectory renderer to retain its
        # original natural-language prompt and final assistant text.
        name = direct_result_event_name(sidecar["block_id"], source_ref["trajectory_id"],
                                        sidecar["writer_call_id"], sidecar["writer_node"])
        sites.append(WriteSite(
            tool="assistant-text-return", before_arguments={}, argument="content", value_path=(),
            name=name, descriptor={"name": name, "type": "Neuralese<string>",
                                   "source": sidecar["body_source"]},
            source_kind="assistant-text-sidecar", arguments={}, inline_prefix=None,
            source_span=(sidecar["body_start"], sidecar["body_end"])))
    return tuple(sites)


def argument_schema(record: Mapping[str, Any], tool_name: str | None, argument: str,
                    value_path: tuple[str | int, ...] | list[str | int] | None = None) -> Any:
    """Return the exact tool argument schema at a writer's validated path."""
    def strip_descriptions(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip_descriptions(child) for key, child in value.items() if key != "description"}
        if isinstance(value, list):
            return [strip_descriptions(child) for child in value]
        return value

    for tool in record.get("tools", []) or []:
        function = tool.get("function", {})
        if function.get("name") != tool_name:
            continue
        schema = (function.get("parameters") or {}).get("properties", {}).get(argument)
        for part in tuple(value_path or ())[1:]:
            if not isinstance(schema, dict):
                return None
            schema = (schema.get("properties", {}).get(part) if isinstance(part, str) else
                      schema.get("items") if type(part) is int else None)
        return strip_descriptions(schema)
    return None
