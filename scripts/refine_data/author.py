"""refine-author (plans/REFINEMENT_TYPES.md section 5.3): authoring examples generated statically from the repository.

Each example pairs a natural-language function and its `types.ts` whose instructions carry a guard sentence with the same
function whose return type states the property instead (`Is<T, "predicate">`). The anchors live in
applications/refine-data/authoring.json; the text, the types and the predicates all come from the repository:

- the function and `types.ts` are read from the working tree, or from git history for entries with a `source.rev` (guard
  sentences that the applications' rebuilds already removed, e.g. logs/assess.nl, wiki/reconcile.nl, build/choose.nl,
  migration/propose.nl);
- the predicate and base type of each slot come from the DECOMPOSITION.md tables (`predicates.harvest`);
- `after` is `before` with the guard text edited out and the slot's field (or the function's `returns:`) refined.

An entry whose find text, slot or field cannot be located is returned in `omissions` with its reason.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
from pathlib import Path

from .common import REPO, normalize_predicate, predicate_id, sha256_hex
from .predicates import Candidate, harvest

REGISTRY = "applications/refine-data/authoring.json"
SCHEMA = "natlang.refine-author/1"
TASK = ("The instructions of this natural-language function state properties of its result as guard sentences. "
        "Rewrite the function and its types.ts so that each property is stated by the type, as a refinement "
        "`Is<T, \"predicate\">`, and the instructions only describe the task. Reply with the two files.")


def _git_show(repo: Path, rev: str, path: str) -> str | None:
    result = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def split_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"(---\n.*?\n---\n)(.*)", text, re.S)
    if not match:
        raise ValueError("no frontmatter")
    return match[1], match[2]


def frontmatter_returns(front: str) -> str | None:
    match = re.search(r"^returns:[ \t]*(.+)$", front, re.M)
    return match[1].strip() if match else None


def _quoted(predicate: str) -> str:
    return json.dumps(predicate, ensure_ascii=False)


def is_type(base: str, predicate: str) -> str:
    return f"Is<{base}, {_quoted(predicate)}>"


def declaration_span(types_text: str, owner: str) -> tuple[int, int] | None:
    """Start and end offsets of `type Owner = ...` up to its closing brace (or the statement end)."""
    match = re.search(rf"\btype\s+{re.escape(owner)}\s*=", types_text)
    if not match:
        return None
    depth, index = 0, match.end()
    while index < len(types_text):
        char = types_text[index]
        if char in "{[(<":
            depth += 1
        elif char in "}])>":
            depth -= 1
        elif char == ";" and depth <= 0:
            return match.start(), index
        elif char == "\n" and depth <= 0 and types_text[match.end():index].strip():
            return match.start(), index
        index += 1
    return match.start(), len(types_text)


def refine_field(types_text: str, owner: str, field: str, base: str, predicate: str) -> str | None:
    span = declaration_span(types_text, owner)
    if span is None:
        return None
    block = types_text[span[0]:span[1]]
    pattern = re.compile(rf"(\b{re.escape(field)}\??:\s*){re.escape(base)}(?=\s*[,}};\n])")
    refined, count = pattern.subn(lambda m: m[1] + is_type(base, predicate), block, count=1)
    if count != 1:
        return None
    return types_text[:span[0]] + refined + types_text[span[1]:]


def alias_predicate(types_text: str, owner: str, field: str, base: str) -> str | None:
    """The predicate a field already carries through a named refined alias (`quantity?: Quantity`, with
    `type Quantity = Is<number, "...">`), following alias chains; None when the field is not declared with such an alias."""
    span = declaration_span(types_text, owner)
    if span is None:
        return None
    match = re.search(rf"\b{re.escape(field)}\??:\s*([A-Z][A-Za-z0-9]*)(?=\s*[,}};\n])", types_text[span[0]:span[1]])
    seen: set[str] = set()
    name = match[1] if match else None
    while name and name not in seen:
        seen.add(name)
        alias = declaration_span(types_text, name)
        if alias is None:
            return None
        body = types_text[alias[0]:alias[1]].split("=", 1)[1].strip()
        refined = re.fullmatch(rf"Is<\s*{re.escape(base)}\s*,\s*(\"(?:[^\"\\]|\\.)*\")\s*>", body, re.S)
        if refined:
            return json.loads(refined[1])
        name = body if re.fullmatch(r"[A-Z][A-Za-z0-9]*", body) else None
    return None


def closure(types_text: str, names: set[str]) -> set[str]:
    """Type names reachable from `names` through the declarations of types.ts."""
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        span = declaration_span(types_text, name)
        if span:
            todo += [word for word in re.findall(r"[A-Z][A-Za-z0-9]*", types_text[span[0]:span[1]]) if word not in seen]
    return seen


def _candidate(candidates: list[Candidate], app: str, slot: str, contains: str | None) -> Candidate | None:
    found = [c for c in candidates if c.app == app and slot in c.slots and (not contains or contains in c.predicate)]
    return found[0] if found else None


def build_example(entry: dict, candidates: list[Candidate], repo: Path = REPO) -> tuple[dict | None, str | None]:
    app = entry["app"]
    source = entry.get("source")
    if source:
        nl_path, types_path, rev = source["function"], source["types"], source["rev"]
        nl_text, types_text = _git_show(repo, rev, nl_path), _git_show(repo, rev, types_path)
        origin = {"kind": "history", "rev": rev, "function": nl_path, "types": types_path}
    else:
        nl_path = f"applications/{app}/{entry['function']}"
        types_path = f"applications/{app}/types.ts"
        rev = None
        nl_file, types_file = repo / nl_path, repo / types_path
        nl_text = nl_file.read_text(encoding="utf-8") if nl_file.is_file() else None
        types_text = types_file.read_text(encoding="utf-8") if types_file.is_file() else None
        origin = {"kind": "tree", "function": nl_path, "types": types_path}
    if nl_text is None:
        return None, f"function not found: {nl_path}" + (f" at {rev}" if rev else "")
    if types_text is None:
        return None, f"types.ts not found: {types_path}" + (f" at {rev}" if rev else "")
    front, body = split_frontmatter(nl_text)
    new_body = body
    for edit in entry["edits"]:
        if new_body.count(edit["find"]) != 1:
            return None, f"guard text found {new_body.count(edit['find'])} times (expected once): {edit['find'][:60]!r}"
        new_body = new_body.replace(edit["find"], edit["replace"], 1)
    new_front, new_types = front, types_text
    predicates, slots, refined_types = [], [], []
    returns = frontmatter_returns(front)
    returned = set(re.findall(r"[A-Z][A-Za-z0-9]*", returns or ""))
    reachable = closure(types_text, returned)
    for slot in entry["slots"]:
        candidate = _candidate(candidates, app, slot, entry.get("predicate_contains"))
        if candidate is None:
            return None, f"slot not in the DECOMPOSITION.md tables: {app} {slot}"
        predicate, base = candidate.predicate, candidate.base
        if entry.get("return_type"):
            new_returns = f"'{is_type(returns or base, predicate)}'"
            new_front = re.sub(r"^returns:.*$", lambda m: "returns: " + new_returns, new_front, count=1, flags=re.M)
        else:
            owner, _, field = slot.partition(".")
            owner = entry.get("owner_override", owner)
            if not field:
                return None, f"slot {slot} is not Owner.field"
            if owner not in reachable:
                return None, f"{owner} is not reachable from the function's return type {returns!r}"
            carried = alias_predicate(new_types, owner, field, base)
            if carried is not None:
                # The field is already a refined alias: the example teaches moving the guard into the type that exists,
                # and records the predicate the type states.
                predicate = carried
            else:
                refined = refine_field(new_types, owner, field, base, predicate)
                if refined is None:
                    return None, f"field {slot} of type {base} not found in {types_path}"
                new_types = refined
        predicates.append(predicate)
        slots.append(slot)
        refined_types.append(is_type(returns or base, predicate) if entry.get('return_type') else is_type(base, predicate))
    if entry.get("predicate"):
        predicate = entry["predicate"]
        base = returns or "unknown"
        new_front = re.sub(r"^returns:.*$", lambda m: "returns: '" + is_type(base, predicate) + "'", new_front, count=1, flags=re.M)
        predicates.append(predicate)
        slots.append("return")
        refined_types.append(is_type(base, predicate))
    if new_body == body and new_types == types_text and new_front == front:
        return None, "no change"
    after_nl = new_front + new_body
    example = {
        "schema": SCHEMA, "id": f"refine-author:{entry['id']}", "app": app, "slots": slots,
        "predicates": predicates, "refined_types": refined_types, "predicate_ids": [predicate_id(p) for p in predicates], "origin": origin,
        "predicate_from": entry.get("predicate_from", "DECOMPOSITION.md"),
        "guards": [edit["find"] for edit in entry["edits"]],
        "before": {"function": nl_text, "types": types_text},
        "after": {"function": after_nl, "types": new_types},
        "diff": {"function": _diff(nl_text, after_nl, nl_path), "types": _diff(types_text, new_types, types_path)},
    }
    example["messages"] = [
        {"role": "user", "content": f"{TASK}\n\n<<<function {nl_path}\n{nl_text}\nfunction>>>\n\n<<<types types.ts\n{types_text}\ntypes>>>"},
        {"role": "assistant", "content": f"<<<function {nl_path}\n{after_nl}\nfunction>>>\n\n<<<types types.ts\n{new_types}\ntypes>>>"},
    ]
    example["content_sha256"] = sha256_hex(json.dumps(example["messages"], ensure_ascii=False, sort_keys=True).encode())
    return example, None


def _diff(before: str, after: str, path: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{path}", f"b/{path}", n=1))


def generate(repo: Path = REPO) -> tuple[list[dict], list[dict], dict]:
    registry = json.loads((repo / REGISTRY).read_text(encoding="utf-8"))
    candidates, _ = harvest(repo)
    examples, omissions = [], []
    for entry in registry["entries"]:
        example, reason = build_example(entry, candidates, repo)
        if example:
            examples.append(example)
        else:
            omissions.append({"id": entry["id"], "app": entry["app"], "slots": entry["slots"], "reason": reason})
    # Predicates of the DECOMPOSITION.md tables that no entry covers are omissions too, so the gap is visible.
    covered = {pid for example in examples for pid in example["predicate_ids"]}
    uncovered = sorted({c.id for c in candidates} - covered)
    report = {"examples": len(examples), "entry_omissions": len(omissions), "candidates": len({c.id for c in candidates}),
              "candidates_without_example": len(uncovered), "uncovered_predicate_ids": uncovered}
    return examples, omissions, report
