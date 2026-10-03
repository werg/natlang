"""Port records (`natlang.port-record/1`, S1 §2.1): loading and validation.

Records are model-neutral: text and structure only. This module checks the fields the
port trainer relies on; the full JSON schema is owned by S1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

VERSION = "natlang.port-record/1"
TASKS = {"consume", "reconstruct", "continue", "chain", "compare"}
IMITATION_LABELS = {"gold", "checked"}


class RecordError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    role: str
    text: str
    exact_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortRecord:
    id: str
    family: str
    task: str
    sources: tuple[Source, ...]
    writer_instructions: str
    result_type: str
    writer_context: tuple[dict, ...]
    consumer_context: tuple[dict, ...]
    withheld: tuple[str, ...]
    target_kind: str
    target: str
    alternatives: tuple[str, ...]
    outcome_label: str
    split: str
    split_groups: tuple[str, ...]
    license: dict = field(default_factory=dict)
    lineage: dict = field(default_factory=dict)
    contrasts: dict = field(default_factory=dict)

    @property
    def is_imitation_target(self) -> bool:
        return self.outcome_label in IMITATION_LABELS

    def source_text(self) -> str:
        return "\n\n".join(source.text for source in self.sources)


def _require(raw: dict, key: str, kind=None):
    if key not in raw:
        raise RecordError(f"missing field {key!r}")
    value = raw[key]
    if kind is not None and not isinstance(value, kind):
        raise RecordError(f"field {key!r} must be {kind.__name__}")
    return value


def _source_text(s: dict) -> str:
    """Text of a source; trajectory sources hold messages, rendered one per line as `role: content`."""
    if "text" in s:
        return _require(s, "text", str)
    messages = _require(s, "messages", list)
    return "\n".join(f"{m.get('role')}: {m.get('content') or ''}".rstrip() for m in messages)


def _ref_text(ref) -> str:
    """Exact references are `{text, kind}` objects in the S1 schema; bare strings are accepted too."""
    return ref["text"] if isinstance(ref, dict) else str(ref)


def _target_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_record(raw: dict) -> PortRecord:
    if raw.get("version") != VERSION:
        raise RecordError(f"unsupported version {raw.get('version')!r}")
    task = _require(raw, "task", str)
    if task not in TASKS:
        raise RecordError(f"unknown task {task!r}")
    sources = tuple(
        Source(role=_require(s, "role", str), text=_source_text(s), exact_refs=tuple(_ref_text(r) for r in s.get("exact_refs", ())))
        for s in _require(raw, "sources", list)
    )
    if not sources:
        raise RecordError("a record needs at least one source")
    writer = _require(raw, "writer", dict)
    consumer = _require(raw, "consumer", dict)
    target = _require(raw, "target", dict)
    outcome = _require(raw, "outcome", dict)
    result_type = _require(writer, "result_type", str)
    if not result_type.startswith("Neuralese<"):
        raise RecordError("writer.result_type must be a Neuralese type")
    for message in consumer.get("context", []):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise RecordError("consumer.context must hold role/content messages")
    return PortRecord(
        id=_require(raw, "id", str),
        family=_require(raw, "family", str),
        task=task,
        sources=sources,
        writer_instructions=_require(writer, "instructions", str),
        result_type=result_type,
        writer_context=tuple(writer.get("context", ())),
        consumer_context=tuple(consumer.get("context", ())),
        withheld=tuple(consumer.get("withheld", ())),
        target_kind=_require(target, "kind", str),
        target=_target_text(_require(target, "value")),
        alternatives=tuple(target.get("alternatives", ())),
        outcome_label=_require(outcome, "label", str),
        split=_require(raw, "split", str),
        split_groups=tuple(_require(raw, "split_groups", list)),
        license=raw.get("license", {}),
        lineage=raw.get("lineage", {}),
        contrasts=raw.get("contrasts", {}),
    )


def read_records(paths: Iterable[str | Path], split: str | None = None, imitation_only: bool = True,
                 strict: bool = False) -> Iterator[PortRecord]:
    """Records from JSONL files. Invalid rows raise when `strict`, else are skipped."""
    for path in paths:
        with open(path) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = parse_record(json.loads(line))
                except (RecordError, json.JSONDecodeError) as error:
                    if strict:
                        raise RecordError(f"{path}:{number}: {error}") from error
                    continue
                if split is not None and record.split != split:
                    continue
                if imitation_only and not record.is_imitation_target:
                    continue
                yield record
