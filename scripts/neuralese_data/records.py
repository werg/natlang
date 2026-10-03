"""Port records (`natlang.port-record/1`): construction, hashing, convention stripping and validation.

The JSON schema is spec/neuralese-port-record.schema.json. This module validates the same
contract with the standard library, so converters and checks run in any checkout Python;
`validate_with_schema` additionally applies the JSON schema when `jsonschema` is installed.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

VERSION = "natlang.port-record/1"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "spec" / "neuralese-port-record.schema.json"

TASKS = {"consume", "reconstruct", "continue", "chain", "compare"}
SOURCE_ROLES = {"document", "passage", "record", "schema", "tool_doc", "tool_output", "file", "trajectory", "block"}
REF_KINDS = {"path", "location", "identifier", "number", "quotation", "url", "other"}
TARGET_KINDS = {"text", "choice", "sql", "calls", "json", "message"}
LABELS = {"gold", "checked", "teacher", "failed", "repair"}
PROJECTS = {"bgkit", "schnitzeljagd", "natlang", "upstream"}
SPLITS = {"train", "validation", "test"}
ROLES = {"system", "user", "assistant", "tool"}
TOP_KEYS = {"version", "id", "family", "task", "sources", "writer", "consumer", "target", "contrasts",
            "outcome", "lineage", "license", "split", "split_groups"}
REQUIRED = TOP_KEYS - {"contrasts"}

# Model- and harness-specific markup that must never survive into a record (S1 §4).
_MARKUP = [
    "<|reserved_6|>", "<|im_start|>", "<|im_end|>", "<|startoftext|>", "<|endoftext|>",
    "<|tool_call_start|>", "<|tool_call_end|>", "<|tool_response_start|>", "<|tool_response_end|>",
    "<|tool_list_start|>", "<|tool_list_end|>", "<|neuralese|>", "<|/neuralese|>",
]
_MARKUP_RE = re.compile(r"<\|(?:reserved_\d+|im_start|im_end|startoftext|endoftext|pad|fim_\w+|tool_\w+|/?neuralese)\|>")
_EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\s*")
_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*", re.IGNORECASE)


def strip_markup(text: str) -> str:
    """Remove chat-template and sentinel tokens and the empty no-think block."""
    text = _MARKUP_RE.sub("", text)
    text = _EMPTY_THINK_RE.sub("", text)
    return text


def strip_answer_prefix(text: str) -> str:
    return _ANSWER_RE.sub("", text, count=1)


def residual_markup(text: str) -> list[str]:
    return sorted(set(_MARKUP_RE.findall(text)))


def normalize_text(text: str) -> str:
    """Normalisation used for exact-duplicate hashing and protected-set scans."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(record: dict) -> str:
    """SHA-256 over the record without lineage.sha256, split and split_groups (they change under closure)."""
    body = {k: v for k, v in record.items() if k not in ("split", "split_groups")}
    lineage = dict(body.get("lineage") or {})
    lineage.pop("sha256", None)
    body["lineage"] = lineage
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def seal(record: dict) -> dict:
    record["lineage"]["sha256"] = content_hash(record)
    return record


def license_(spdx: str, noncommercial: bool = False, notes: str = "") -> dict:
    out = {"spdx": spdx, "noncommercial": noncommercial}
    if notes:
        out["notes"] = notes
    return out


def source(role: str, text: str | None = None, *, messages: list | None = None, title: str | None = None,
           meta: dict | None = None, exact_refs: list | None = None, license: dict | None = None) -> dict:
    out: dict = {"role": role}
    if messages is not None:
        out["messages"] = messages
    else:
        out["text"] = text
    if title:
        out["title"] = title
    if meta:
        out["meta"] = meta
    out["exact_refs"] = exact_refs or []
    if license:
        out["license"] = license
    return out


def group_key(namespace: str, value: str) -> str:
    """A split-group key `namespace:value`, with value case-folded and whitespace-normalised."""
    value = re.sub(r"\s+", "_", unicodedata.normalize("NFKC", str(value)).strip()).casefold()
    return f"{namespace}:{value}"


# Exact references: paths with optional line numbers, path::symbol locations, URLs.
_URL_RE = re.compile(r"https?://[^\s)\"'<>\]]+")
_LOCATION_RE = re.compile(r"(?<![\w/.-])(/?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]{1,8})(?::(\d+)(?::\d+)?|::([A-Za-z_][\w.]*))")
_PATH_RE = re.compile(r"(?<![\w/.:-])(/?(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,8})(?![\w/])")


def exact_refs_from(text: str, limit: int = 64) -> list[dict]:
    """Identifiers that must stay exact: file locations, paths and URLs, in first-seen order."""
    seen: dict[str, str] = {}
    for match in _LOCATION_RE.finditer(text):
        seen.setdefault(match.group(0), "location")
    for match in _PATH_RE.finditer(text):
        seen.setdefault(match.group(1), "path")
    for match in _URL_RE.finditer(text):
        seen.setdefault(match.group(0).rstrip(".,;"), "url")
    return [{"text": t, "kind": k} for t, k in list(seen.items())[:limit]]


def _err(errors: list, where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def _check_message(m, where: str, errors: list) -> None:
    if not isinstance(m, dict):
        return _err(errors, where, "message must be an object")
    if m.get("role") not in ROLES:
        _err(errors, where, f"bad role {m.get('role')!r}")
    content = m.get("content")
    if content is not None and not isinstance(content, str):
        _err(errors, where, "content must be a string or null")
    if isinstance(content, str) and residual_markup(content):
        _err(errors, where, f"chat markup in content: {residual_markup(content)}")
    reasoning = m.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            _err(errors, where, "reasoning must be a string")
        elif residual_markup(reasoning):
            _err(errors, where, f"chat markup in reasoning: {residual_markup(reasoning)}")


def _check_license(lic, where: str, errors: list) -> None:
    if not isinstance(lic, dict) or not isinstance(lic.get("spdx"), str) or not lic.get("spdx"):
        return _err(errors, where, "license.spdx required")
    if not isinstance(lic.get("noncommercial"), bool):
        _err(errors, where, "license.noncommercial must be boolean")
    if set(lic) - {"spdx", "noncommercial", "notes"}:
        _err(errors, where, f"unknown license keys {sorted(set(lic) - {'spdx', 'noncommercial', 'notes'})}")


def _check_source(s, where: str, errors: list) -> None:
    if not isinstance(s, dict):
        return _err(errors, where, "source must be an object")
    if s.get("role") not in SOURCE_ROLES:
        _err(errors, where, f"bad source role {s.get('role')!r}")
    has_text, has_messages = "text" in s, "messages" in s
    if has_text == has_messages:
        _err(errors, where, "exactly one of text or messages required")
    if has_text:
        if not isinstance(s["text"], str) or not s["text"].strip():
            _err(errors, where, "text must be a non-empty string")
        elif residual_markup(s["text"]):
            _err(errors, where, f"chat markup in text: {residual_markup(s['text'])}")
    if has_messages:
        if not isinstance(s["messages"], list) or not s["messages"]:
            _err(errors, where, "messages must be a non-empty list")
        else:
            for i, m in enumerate(s["messages"]):
                _check_message(m, f"{where}.messages[{i}]", errors)
    refs = s.get("exact_refs")
    if not isinstance(refs, list):
        _err(errors, where, "exact_refs must be a list")
    else:
        for i, r in enumerate(refs):
            if not isinstance(r, dict) or not r.get("text") or r.get("kind") not in REF_KINDS:
                _err(errors, f"{where}.exact_refs[{i}]", "needs text and a known kind")
    if "license" in s:
        _check_license(s["license"], f"{where}.license", errors)
    extra = set(s) - {"role", "text", "messages", "title", "meta", "exact_refs", "license"}
    if extra:
        _err(errors, where, f"unknown keys {sorted(extra)}")


def validate(record) -> list[str]:
    """Return a list of contract violations (empty when valid)."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    missing = REQUIRED - set(record)
    if missing:
        _err(errors, "record", f"missing {sorted(missing)}")
    extra = set(record) - TOP_KEYS
    if extra:
        _err(errors, "record", f"unknown keys {sorted(extra)}")
    if record.get("version") != VERSION:
        _err(errors, "version", f"must be {VERSION}")
    if not isinstance(record.get("id"), str) or not re.match(r"^[a-z0-9_-]+:[^\s]+$", record.get("id") or ""):
        _err(errors, "id", "must look like project:...")
    if not isinstance(record.get("family"), str) or not re.match(r"^[a-z0-9_]+$", record.get("family") or ""):
        _err(errors, "family", "must be snake_case")
    if record.get("task") not in TASKS:
        _err(errors, "task", f"bad task {record.get('task')!r}")
    sources = record.get("sources")
    if not isinstance(sources, list) or not sources:
        _err(errors, "sources", "non-empty list required")
    else:
        for i, s in enumerate(sources):
            _check_source(s, f"sources[{i}]", errors)
    writer = record.get("writer")
    if not isinstance(writer, dict):
        _err(errors, "writer", "object required")
    else:
        if not isinstance(writer.get("instructions"), str) or not writer["instructions"].strip():
            _err(errors, "writer.instructions", "non-empty string required")
        if "instructions_general" in writer and not (isinstance(writer["instructions_general"], str) and writer["instructions_general"].strip()):
            _err(errors, "writer.instructions_general", "non-empty string when present")
        if not isinstance(writer.get("result_type"), str) or not writer.get("result_type"):
            _err(errors, "writer.result_type", "natlang type required")
        if not isinstance(writer.get("context"), list):
            _err(errors, "writer.context", "list required")
        else:
            for i, m in enumerate(writer["context"]):
                _check_message(m, f"writer.context[{i}]", errors)
        extra = set(writer) - {"instructions", "instructions_general", "result_type", "context"}
        if extra:
            _err(errors, "writer", f"unknown keys {sorted(extra)}")
    consumer = record.get("consumer")
    if not isinstance(consumer, dict):
        _err(errors, "consumer", "object required")
    else:
        if not isinstance(consumer.get("context"), list):
            _err(errors, "consumer.context", "list required")
        else:
            for i, m in enumerate(consumer["context"]):
                _check_message(m, f"consumer.context[{i}]", errors)
        if not isinstance(consumer.get("withheld"), list) or any(w != "sources" for w in consumer.get("withheld", [])):
            _err(errors, "consumer.withheld", "list of 'sources'")
        extra = set(consumer) - {"context", "tools", "withheld"}
        if extra:
            _err(errors, "consumer", f"unknown keys {sorted(extra)}")
    target = record.get("target")
    if not isinstance(target, dict) or target.get("kind") not in TARGET_KINDS or "value" not in target:
        _err(errors, "target", "kind and value required")
    elif target["value"] in (None, "", [], {}):
        _err(errors, "target.value", "empty target")
    elif isinstance(target["value"], str) and residual_markup(target["value"]):
        _err(errors, "target.value", f"chat markup: {residual_markup(target['value'])}")
    contrasts = record.get("contrasts")
    if contrasts is not None:
        if not isinstance(contrasts, dict) or set(contrasts) - {"purpose_pairs", "distractors"}:
            _err(errors, "contrasts", "only purpose_pairs and distractors")
        else:
            for i, d in enumerate(contrasts.get("distractors", [])):
                _check_source(d, f"contrasts.distractors[{i}]", errors)
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("label") not in LABELS or "checked" not in outcome:
        _err(errors, "outcome", "label and checked required")
    lineage = record.get("lineage")
    if not isinstance(lineage, dict):
        _err(errors, "lineage", "object required")
    else:
        if lineage.get("project") not in PROJECTS:
            _err(errors, "lineage.project", f"bad project {lineage.get('project')!r}")
        for key in ("store", "row", "upstream", "teacher", "sha256"):
            if key not in lineage:
                _err(errors, "lineage", f"missing {key}")
        teacher = lineage.get("teacher")
        if not (teacher is None or teacher == "unknown" or (isinstance(teacher, dict) and isinstance(teacher.get("model"), str))):
            _err(errors, "lineage.teacher", "null, 'unknown' or {model, provider}")
        if lineage.get("sha256") != content_hash(record):
            _err(errors, "lineage.sha256", "does not match content")
        extra = set(lineage) - {"project", "store", "store_version", "row", "upstream", "upstream_id",
                                "upstream_revision", "teacher", "converter", "sha256", "notes"}
        if extra:
            _err(errors, "lineage", f"unknown keys {sorted(extra)}")
    _check_license(record.get("license"), "license", errors)
    if record.get("split") not in SPLITS:
        _err(errors, "split", f"bad split {record.get('split')!r}")
    groups = record.get("split_groups")
    if not isinstance(groups, list) or not groups or not all(isinstance(g, str) and re.match(r"^[a-z0-9_-]+:.+$", g) for g in groups):
        _err(errors, "split_groups", "non-empty list of namespace:value keys")
    return errors


@lru_cache(maxsize=1)
def _schema_validator():
    """Load and construct the process-wide validator once; return it with the exact schema hash."""
    import jsonschema  # type: ignore
    raw_schema = SCHEMA_PATH.read_bytes()
    schema = json.loads(raw_schema)
    validator = jsonschema.Draft202012Validator(schema)
    return validator, hashlib.sha256(raw_schema).hexdigest()


def require_schema_validator():
    """Return the cached validator and its schema SHA-256, or raise if jsonschema is unavailable."""
    return _schema_validator()


def validate_with_schema(record) -> list[str]:
    """Validate against the JSON schema when jsonschema is available, plus the stdlib checks."""
    errors = validate(record)
    try:
        validator, _schema_sha256 = require_schema_validator()
    except ImportError:
        return errors
    errors.extend(f"schema {'/'.join(map(str, e.path))}: {e.message}" for e in validator.iter_errors(record))
    return errors


def leakage(record: dict) -> list[str]:
    """Writer inputs must not contain the target (S1 §6.5)."""
    value = record["target"]["value"]
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    consumer = "\n".join(m.get("content") or "" for m in record["consumer"]["context"])
    exact = {s.get("title") for s in record["sources"]} | {r["text"] for s in record["sources"] for r in s.get("exact_refs", [])}
    if len(text.strip()) < 12 or text.strip() in consumer or text.strip() in exact:
        return []  # short answers, answers the consumer's request names, or exact source references
    writer = record["writer"]
    haystack = "\n".join([writer["instructions"], writer.get("instructions_general", ""),
                          *(m.get("content") or "" for m in writer["context"])])
    return ["target text appears in writer inputs"] if text.strip() in haystack else []
