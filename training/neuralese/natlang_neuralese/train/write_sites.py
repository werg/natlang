"""Canonical exact write-site extraction for native trajectory records.

The extractor is pure data handling: JSON write markers retain their exact
argument path and inline code writers are exposed only after validating the
compiler-attested sidecar against the original arguments string.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping
import json


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
    """Return target writers from JSON marker values and validated inline sites."""
    target = record.get("target") or {}
    return tuple(replace(site, schema=argument_schema(record, site.tool, site.argument, site.value_path))
                 for site in extract_write_sites_from_calls(target.get("tool_calls") or []))


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
