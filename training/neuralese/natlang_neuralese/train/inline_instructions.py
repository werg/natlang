"""Validation and rendering for compiler-attested inline instruction code sites.

The converter keeps the original eval arguments intact and attaches a `neuralese_code` sidecar. This module
validates the complete sidecar before exposing any writer or rendering any selected block. It does not change
trajectory admission or the conversation data model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
                _append_text(rendered, str(site["source"]))
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
        if not isinstance(name, str) or not name or name in seen_names:
            return InlineInstructionValidation(False, reason="writer-name-missing-or-duplicate")
        if value_type != INLINE_WRITE_TYPE or not isinstance(source, str):
            return InlineInstructionValidation(False, reason="writer-contract-invalid")
        # Stage 2 accepts only raw, non-interpolated template bodies: no escaped delimiters or holes.
        if "\\" in source or "`" in source or "${" in source:
            return InlineInstructionValidation(False, reason="escaped-or-interpolated-body")
        before = "".join(pieces)
        if not _TEMPLATE_PREFIX.search(before):
            return InlineInstructionValidation(False, reason="writer-not-at-inline-template")
        start, end = cursor, cursor + len(source)
        pieces.append(source)
        cursor = end
        parts.append({"$write": {"name": name, "type": value_type, "source": source}})
        writes.append(InlineInstructionWrite(name, value_type, source, start, end))
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


def _append_text(parts: list[dict[str, str]], text: str) -> None:
    if not text:
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1]["text"] += text
    else:
        parts.append({"type": "text", "text": text})
