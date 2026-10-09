"""Harvest the refinement candidates of applications/*/DECOMPOSITION.md (the "Refinements" tables).

Each row is `| slot(s) | `Is<T, "predicate">` |`. Rows whose type is not `Is<...>` (`Untrusted<...>`) are not predicates and
are reported in `skipped`, not dropped silently.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import REPO, normalize_predicate, predicate_id

SECTION = re.compile(r"^##\s+Refinements?(?:\s+candidates)?\s*$", re.I)
IS_TYPE = re.compile(r'^Is<(?P<base>.+?),\s*"(?P<predicate>.*)">$', re.S)
CODE_SPAN = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class Candidate:
    app: str
    slots: tuple[str, ...]
    base: str
    predicate: str
    note: str  # trailing remark of the row (for example "(`ledger.validate`)")
    source: str  # repo path of the DECOMPOSITION.md

    @property
    def id(self) -> str:
        return predicate_id(self.predicate)

    def to_json(self) -> dict:
        row = asdict(self)
        row["slots"] = list(self.slots)
        row["predicate_id"] = self.id
        return row


def _split_cells(line: str) -> list[str]:
    """Split a table row on unescaped pipes, ignoring pipes inside backtick spans."""
    cells, current, in_code, index = [], [], False, 0
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    while index < len(text):
        char = text[index]
        if char == "`":
            in_code = not in_code
        if char == "\\" and index + 1 < len(text) and text[index + 1] == "|":
            current.append("|")
            index += 2
            continue
        if char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    cells.append("".join(current).strip())
    return cells


def parse_table(text: str, app: str, source: str) -> tuple[list[Candidate], list[dict]]:
    found: list[Candidate] = []
    skipped: list[dict] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = bool(SECTION.match(line))
            continue
        if not inside or not line.lstrip().startswith("|"):
            continue
        cells = _split_cells(line)
        if len(cells) < 2 or set("".join(cells)) <= set("-: ") or cells[0].lower() == "slot":
            continue
        slots = tuple(CODE_SPAN.findall(cells[0])) or (cells[0],)
        spans = CODE_SPAN.findall(cells[1])
        if not spans:
            skipped.append({"app": app, "slots": list(slots), "reason": "no-type", "text": cells[1]})
            continue
        matches = [(span, IS_TYPE.match(span.strip())) for span in spans]
        matches = [(span, match) for span, match in matches if match]
        if not matches:
            skipped.append({"app": app, "slots": list(slots), "reason": "not-an-Is-type", "text": spans[0].strip()})
            continue
        note = CODE_SPAN.sub("", cells[1]).strip()
        for _, match in matches:
            found.append(Candidate(app=app, slots=slots, base=match["base"].strip(), predicate=normalize_predicate(match["predicate"]),
                                   note=note, source=source))
    return found, skipped


def harvest(repo: Path = REPO) -> tuple[list[Candidate], list[dict]]:
    """Every `Is<T, "predicate">` candidate of the applications' DECOMPOSITION.md files, in path order."""
    found: list[Candidate] = []
    skipped: list[dict] = []
    for path in sorted((repo / "applications").glob("*/DECOMPOSITION.md")):
        app = path.parent.name
        rows, rest = parse_table(path.read_text(encoding="utf-8"), app, path.relative_to(repo).as_posix())
        found += rows
        skipped += rest
    return found, skipped


def unique_predicates(candidates: list[Candidate]) -> list[dict]:
    """One entry per distinct predicate text, with every (app, slot, base) it was proposed for."""
    by_id: dict[str, dict] = {}
    for item in candidates:
        entry = by_id.setdefault(item.id, {"predicate_id": item.id, "predicate": item.predicate, "uses": []})
        entry["uses"].append({"app": item.app, "slots": list(item.slots), "base": item.base})
    return list(by_id.values())
