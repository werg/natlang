"""Internal operation arguments shared by structured session tools."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Action:
    tool: str
    path: str = ""
    dst: str = ""
    paths: list = field(default_factory=list)
    body: str = ""
