"""Validation and rendering for compiler-attested inline instruction code sites.

The converter keeps the original eval arguments intact and attaches a `neuralese_code` sidecar. This module
validates the complete sidecar before exposing any writer or rendering any selected block. It does not change
trajectory admission or the conversation data model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping


INLINE_CODE_SCHEMA = "natlang.inline-instruction-code/1"
INLINE_WRITE_TYPE = "Neuralese<string>"


@dataclass(frozen=True)
class InlineInstructionWrite:
    """One model-authored, non-interpolated template body in an eval code string."""

    name: str
    type: str
    source: str
    start: int
    end: int
    code_source: str | None = None


@dataclass(frozen=True)
class InlineInstructionCode:
    """Fully validated sidecar and exact crisp code reconstruction."""

    code: str
    code_sha256: str
    writes: tuple[InlineInstructionWrite, ...]
    # The original parts are retained so callers can inspect source partitioning without reparsing.
    parts: tuple[Mapping[str, Any], ...]

    def prefix_and_body(self, name: str) -> tuple[str, str] | None:
        """Return crisp code before a writer body and that body's source, by action-scoped writer name."""
        write = next((item for item in self.writes if item.name == name), None)
        if write is None:
            return None
        return self.code[:write.start], write.source

    def render_code(self, block_ids: Mapping[str, str] | None = None) -> str | list[dict[str, str]]:
        """Render crisp source or engine text/Neuralese parts; unselected writes stay crisp text."""
        selected = block_ids or {}
        if not any(write.name in selected for write in self.writes):
            return self.code
        rendered: list[dict[str, str]] = []
        for part in self.parts:
            if part.get("type") == "text":
                _append_text(rendered, str(part["text"]))
                continue
            site = part["$write"]
            block_id = selected.get(str(site["name"]))
            if block_id is None:
                _append_text(rendered, str(site.get("code_source", site["source"])))
            else:
                if rendered and rendered[-1].get("type") == "neuralese":
                    # Preserve distinct block boundaries even for adjacent sites.
                    pass
                rendered.append({"type": "neuralese", "id": block_id, "value_type": "string"})
        return rendered


@dataclass(frozen=True)
class InlineInstructionValidation:
    valid: bool
    value: InlineInstructionCode | None = None
    reason: str | None = None


def validate_inline_instruction_code(arguments_json: Any, sidecar: Any) -> InlineInstructionValidation:
    """Validate the all-or-nothing inline code sidecar against the exact JSON tool arguments.

    `arguments_json` must be the original function.arguments JSON string. Invalid/missing data returns a reason and no
    decoded writes, so callers cannot accidentally render the valid prefix of a corrupt sidecar.
    """
    if not isinstance(arguments_json, str):
        return InlineInstructionValidation(False, reason="arguments-not-original-json-string")
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return InlineInstructionValidation(False, reason="arguments-invalid-json")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
        return InlineInstructionValidation(False, reason="code-argument-missing")
    if not isinstance(sidecar, Mapping) or sidecar.get("schema") != INLINE_CODE_SCHEMA:
        return InlineInstructionValidation(False, reason="sidecar-schema-invalid")
    code = arguments["code"]
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if sidecar.get("code_sha256") != digest:
        return InlineInstructionValidation(False, reason="code-hash-mismatch")
    raw_parts = sidecar.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        return InlineInstructionValidation(False, reason="parts-missing")

    parts: list[Mapping[str, Any]] = []
    writes: list[InlineInstructionWrite] = []
    plans_raw = sidecar.get("sites")
    plans = {str(site.get("name")): site.get("plan") for site in plans_raw if isinstance(site, Mapping) and isinstance(site.get("name"), str)} if isinstance(plans_raw, list) else {}
    seen_names: set[str] = set()
    pieces: list[str] = []
    cursor = 0
    for item in raw_parts:
        if not isinstance(item, Mapping):
            return InlineInstructionValidation(False, reason="part-invalid")
        if item.get("type") == "text":
            text = item.get("text")
            if not isinstance(text, str):
                return InlineInstructionValidation(False, reason="text-part-invalid")
            pieces.append(text)
            cursor += len(text)
            parts.append({"type": "text", "text": text})
            continue
        write = item.get("$write")
        if not isinstance(write, Mapping):
            return InlineInstructionValidation(False, reason="unsupported-part")
        name, value_type, source = write.get("name"), write.get("type"), write.get("source")
        code_source = write.get("code_source")
        if not isinstance(name, str) or not name or name in seen_names:
            return InlineInstructionValidation(False, reason="writer-name-missing-or-duplicate")
        if value_type != INLINE_WRITE_TYPE or not isinstance(source, str):
            return InlineInstructionValidation(False, reason="writer-contract-invalid")
        if code_source is not None:
            if not isinstance(code_source, str):
                return InlineInstructionValidation(False, reason="soft-body-code-source-invalid")
            if "\\" in source or "`" in source or "${" in source:
                return InlineInstructionValidation(False, reason="escaped-or-interpolated-body")
            plan = plans.get(name)
            if not _valid_capture_binding_plan(plan, code_source, source, code):
                return InlineInstructionValidation(False, reason="capture-binding-plan-invalid")
        # Stage 2 accepts only raw, non-interpolated template bodies: no escaped delimiters or holes.
        if "\\" in source or "`" in source or "${" in source:
            return InlineInstructionValidation(False, reason="escaped-or-interpolated-body")
        before = "".join(pieces)
        if code_source is None:
            if not _TEMPLATE_PREFIX.search(before):
                return InlineInstructionValidation(False, reason="writer-not-at-inline-template")
        else:
            prefix = _WITH_TEMPLATE_PREFIX.search(before)
            plan = plans.get(name)
            names = [capture.get("name") for capture in plan.get("capture_binding_plan", {}).get("captures", [])] if isinstance(plan, Mapping) else []
            if not prefix or not _capture_names_match(prefix.group(1), names):
                return InlineInstructionValidation(False, reason="explicit-with-prefix-mismatch")
        start = cursor
        reconstruction = code_source if isinstance(code_source, str) else source
        pieces.append(reconstruction)
        cursor += len(reconstruction)
        parts.append({"$write": {"name": name, "type": value_type, "source": source,
          **({"code_source": code_source} if isinstance(code_source, str) else {})}})
        writes.append(InlineInstructionWrite(name, value_type, source, start, cursor, code_source if isinstance(code_source, str) else None))
        seen_names.add(name)

    reconstructed = "".join(pieces)
    if not writes:
        return InlineInstructionValidation(False, reason="no-writer-parts")
    if reconstructed != code:
        return InlineInstructionValidation(False, reason="source-reconstruction-mismatch")
    # The body source must end before the exact closing template delimiter in the original code.
    for write in writes:
        if write.end >= len(code) or code[write.end] != "`":
            return InlineInstructionValidation(False, reason="template-close-mismatch")
    return InlineInstructionValidation(True, InlineInstructionCode(code, digest, tuple(writes), tuple(parts)))


def inline_instruction_prefix_body(validated: InlineInstructionValidation | InlineInstructionCode,
                                   writer_name: str) -> tuple[str, str] | None:
    """Return `(prefix_code, body_source)` for a validated action-scoped writer, or None if absent/invalid."""
    if isinstance(validated, InlineInstructionValidation) and not validated.valid:
        return None
    value = validated.value if isinstance(validated, InlineInstructionValidation) else validated
    if value is None:
        return None
    return value.prefix_and_body(writer_name)


def render_inline_instruction_code(validated: InlineInstructionValidation | InlineInstructionCode,
                                   block_ids: Mapping[str, str] | None = None) -> str | list[dict[str, str]] | None:
    """Render one validated eval `code` value. No selected blocks returns the exact original code string."""
    if isinstance(validated, InlineInstructionValidation) and not validated.valid:
        return None
    value = validated.value if isinstance(validated, InlineInstructionValidation) else validated
    return None if value is None else value.render_code(block_ids)


def render_inline_instruction_arguments(arguments_json: Any, sidecar: Any,
                                        block_ids: Mapping[str, str] | None = None) -> str | None:
    """Render tool arguments while preserving exact original JSON bytes on the crisp/no-block path.

    When any writer is selected, only `code` is replaced by the existing engine's text/Neuralese part list. The
    result is JSON text ready to assign to a call's `function.arguments`; invalid sidecars return None atomically.
    """
    checked = validate_inline_instruction_code(arguments_json, sidecar)
    if not checked.valid or checked.value is None:
        return None
    selected = block_ids or {}
    if not any(write.name in selected for write in checked.value.writes):
        return arguments_json
    try:
        arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):  # Already checked, retain fail-closed behavior if inputs mutate.
        return None
    arguments["code"] = checked.value.render_code(selected)
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


_TEMPLATE_PREFIX = re.compile(r"\bnl\s*(?:<[^`]*?>\s*)?`$")
_WITH_TEMPLATE_PREFIX = re.compile(r"\bnl\.with\s*(?:<[^`]*>\s*)?\(\{([^{}]*)\}\)\s*(?:<[^`]*>\s*)?`$")


def _capture_names_match(raw: str, expected: Any) -> bool:
    if not isinstance(expected, list) or not expected or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_$][\w$]*", name) for name in expected):
        return False
    names = [part.strip() for part in raw.split(",")]
    return names == expected


def _valid_capture_binding_plan(plan: Any, code_source: str, body_source: str, code: str) -> bool:
    if not isinstance(plan, Mapping) or not isinstance(plan.get("capture_binding_plan"), Mapping):
        return False
    binding = plan["capture_binding_plan"]
    block_id = re.fullmatch(r"\ue000(nz1_[a-z2-7]{20,})\ue001", code_source)
    schema = binding.get("schema")
    body_kind = binding.get("body_kind")
    block_id_value = binding.get("body_block_id")
    valid_block_id = (isinstance(block_id_value, str) and
                      re.fullmatch(r"nz1_[a-z2-7]{20,}", block_id_value) is not None)
    marker_body = "<|neuralese|>" + body_source + "<|/neuralese|>"
    valid_body = ((body_kind == "literal" and block_id_value is None and code_source == body_source) or
                  (body_kind == "neuralese_block" and valid_block_id and
                   ((block_id and block_id_value == block_id.group(1)) or code_source == marker_body)))
    if (binding.get("syntax") != "nl.with" or
            schema not in {"natlang.inline-capture-binding-plan/1", "natlang.inline-capture-binding-plan/2"} or
            binding.get("body_source_sha256") != hashlib.sha256(body_source.encode("utf-8")).hexdigest() or
            not isinstance(binding.get("parent_invocation_id"), str) or not binding.get("parent_invocation_id") or
            not _is_sha256(binding.get("parent_scope_sha256")) or not _is_sha256(binding.get("child_scope_sha256"))):
        return False
    if schema == "natlang.inline-capture-binding-plan/1":
        if (not block_id or binding.get("body_block_id") != block_id.group(1) or
                binding.get("body_kind") is not None or binding.get("creation") is not None):
            return False
    else:
        origin_code = code
        if body_kind == "neuralese_block" and code_source == marker_body:
            sentinel = "\ue000" + block_id_value + "\ue001"
            if code.count(marker_body) != 1:
                return False
            origin_code = code.replace(marker_body, sentinel, 1)
        if not valid_body or not _valid_capture_creation(binding, plan, origin_code):
            return False
    captures = binding.get("captures")
    if not isinstance(captures, list) or not captures:
        return False
    names: list[str] = []
    for capture in captures:
        if not isinstance(capture, Mapping):
            return False
        name, value_type, mode, value = (capture.get(key) for key in ("name", "type", "mode", "value"))
        value_matches = _primitive_matches(value_type, value)
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_$][\w$]*", name) or mode != "snapshot" or
                value_type not in {"string", "number", "boolean"} or not value_matches):
            return False
        if schema == "natlang.inline-capture-binding-plan/2":
            source, snapshot = capture.get("source"), capture.get("host_snapshot")
            if source not in {"input", "local", "block"} or not isinstance(snapshot, Mapping):
                return False
            if (snapshot.get("name") != name or snapshot.get("type") != value_type or
                    snapshot.get("source") != source or snapshot.get("mode") != "snapshot" or
                    not _primitive_matches(value_type, snapshot.get("value")) or
                    not _primitive_equal(value_type, value, snapshot.get("value")) or
                    snapshot.get("creation") != binding.get("creation")):
                return False
            declared_type = capture.get("declared_type")
            if snapshot.get("declared_type") != declared_type:
                return False
            if declared_type is not None:
                if declared_type not in {"unknown", "any"}:
                    return False
                descriptors = plan.get("captures")
                if not isinstance(descriptors, list):
                    return False
                matches = [item for item in descriptors if isinstance(item, Mapping) and item.get("name") == name]
                if len(matches) != 1:
                    return False
                descriptor_type = matches[0].get("type")
                if (not isinstance(descriptor_type, Mapping) or
                        (descriptor_type.get("natlang") or descriptor_type.get("text")) != declared_type or
                        matches[0].get("mode") != "snapshot" or matches[0].get("mutable") is not False or
                        matches[0].get("source") != source):
                    return False
            canonical_value = snapshot.get("value_canonical")
            if (not _valid_snapshot_fingerprint(canonical_value, snapshot.get("value_sha256"),
                                                value_type, value)):
                return False
        names.append(name)
    return len(set(names)) == len(names)


def _primitive_matches(value_type: Any, value: Any) -> bool:
    if value_type == "string":
        return isinstance(value, str)
    if value_type == "boolean":
        return type(value) is bool
    if value_type == "number":
        return (type(value) is int or
                (type(value) is float and math.isfinite(value) and
                 not (value == 0 and math.copysign(1.0, value) < 0)))
    return False


def _primitive_equal(value_type: str, left: Any, right: Any) -> bool:
    if not _primitive_matches(value_type, left) or not _primitive_matches(value_type, right):
        return False
    if value_type == "number":
        return left == right
    return type(left) is type(right) and left == right


def _valid_snapshot_fingerprint(canonical_value: Any, digest: Any, value_type: str, value: Any) -> bool:
    if not isinstance(canonical_value, str) or not _is_sha256(digest):
        return False
    expected_digest = hashlib.sha256(
        b"natlang.inline-capture-snapshot/v1\0" + canonical_value.encode("utf-8")).hexdigest()
    if digest != expected_digest:
        return False

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    def reject_constant(_: str) -> Any:
        raise ValueError("non-finite JSON number")

    def parse_integer(text: str) -> int:
        if text == "-0":
            raise ValueError("negative zero")
        return int(text)

    try:
        payload = json.loads(canonical_value, object_pairs_hook=unique_object,
                             parse_constant=reject_constant, parse_int=parse_integer)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return (isinstance(payload, dict) and set(payload) == {"type", "value"} and
            payload.get("type") == value_type and
            _primitive_matches(value_type, payload.get("value")) and
            _primitive_equal(value_type, value, payload.get("value")))


def _valid_capture_creation(binding: Mapping[str, Any], plan: Mapping[str, Any], code: str) -> bool:
    creation = binding.get("creation")
    if not isinstance(creation, Mapping):
        return False
    source_span, template_span, checked_template_span = (creation.get(key) for key in
        ("sourceSpan", "templateSpan", "checkedTemplateSpan"))
    def valid_span(value: Any, *, require_file: bool) -> bool:
        return (isinstance(value, Mapping) and
                (not require_file or (isinstance(value.get("file"), str) and bool(value.get("file")))) and
                type(value.get("start")) is int and type(value.get("end")) is int and
                value["start"] >= 0 and value["end"] > value["start"])
    if (creation.get("parentInvocationId") != binding.get("parent_invocation_id") or
            creation.get("definitionId") != plan.get("definition_id") or
            not isinstance(creation.get("toolCallId"), str) or not creation.get("toolCallId") or
            type(creation.get("actionOrdinal")) is not int or not 0 <= creation["actionOrdinal"] < 2**53 or
            creation.get("writtenCodeSha256") != hashlib.sha256(code.encode("utf-8")).hexdigest() or
            not _is_sha256(creation.get("checkedCodeSha256")) or
            not valid_span(source_span, require_file=True) or not valid_span(template_span, require_file=False) or
            not valid_span(checked_template_span, require_file=True)):
        return False
    return True


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _append_text(parts: list[dict[str, str]], text: str) -> None:
    if not text:
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1]["text"] += text
    else:
        parts.append({"type": "text", "text": text})
