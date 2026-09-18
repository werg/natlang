"""Path syntax (SPEC 1.1)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .diag import reject

META = ("status", "note", "problems", "origin", "effects", "dist")


@dataclass
class Path:
    segs: list
    rng: Optional[tuple] = None  # inclusive (a, b)
    meta: Optional[str] = None
    text: str = ""

    @property
    def append(self) -> bool:
        return bool(self.segs) and self.segs[-1] == "+"


def parse_path(text: str) -> Path:
    raw = text
    meta = None
    if "@" in text:
        text, _, meta = text.rpartition("@")
        if meta not in META:
            raise reject(raw, "no-such-path", "a meta suffix: " + ", ".join(META))
    rng = None
    m = re.search(r"\[(\d+)\.\.(\d+)\]$", text)
    if m:
        rng = (int(m.group(1)), int(m.group(2)))
        text = text[: m.start()]
        if rng[0] > rng[1]:
            raise reject(raw, "bad-range")
    elif "[" in text or "]" in text:
        raise reject(raw, "bad-range", "PATH[a..b]")
    segs = [s for s in text.split("/") if s != ""]
    if not segs:
        raise reject(raw, "no-such-path")
    if any(s in ("..", ".") for s in segs) or text.startswith("/"):
        raise reject(raw, "out-of-scope")
    if "+" in segs[:-1]:
        raise reject(raw, "no-such-path", "`+` only as the last segment")
    return Path(segs, rng, meta, raw)
