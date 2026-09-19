"""Exact search over documents explicitly supplied to a host programme."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    document: str
    line: int
    excerpt: str


class ExactSearch:
    def __init__(self, documents: dict[str, str]):
        self.documents = dict(documents)

    def lookup(self, name: str) -> str | None:
        return self.documents.get(name)

    def scan(self, needle: str, *, case_sensitive: bool = False, limit: int = 20) -> list[Hit]:
        if not needle or limit < 1:
            raise ValueError("search needs nonempty text and a positive limit")
        wanted = needle if case_sensitive else needle.casefold()
        hits = []
        for name, text in sorted(self.documents.items()):
            for number, line in enumerate(text.splitlines(), 1):
                candidate = line if case_sensitive else line.casefold()
                if wanted in candidate:
                    hits.append(Hit(name, number, line[:500]))
                    if len(hits) >= limit:
                        return hits
        return hits
