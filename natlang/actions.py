"""Parsing of agent actions: a header line plus an optional body (SPEC 5)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .diag import reject

TOOLS = ("read", "edit", "set", "unset", "copy", "reduce", "reopen", "eval")
START, END = "<|tool_call_start|>", "<|tool_call_end|>"


@dataclass
class Action:
    tool: str
    path: str = ""
    dst: str = ""
    paths: list = field(default_factory=list)
    type_text: str = ""
    body: str = ""

    def header(self) -> str:
        if self.tool == "set":
            return f"set {self.path} : {self.type_text}"
        if self.tool == "copy":
            return f"copy {self.path} to {self.dst}"
        if self.tool == "reduce":
            return "reduce " + " ".join(self.paths)
        if self.tool == "eval":
            return "eval"
        return f"{self.tool} {self.path}"


def strip_tokens(text: str) -> str:
    text = text.strip("\n")
    if text.lstrip().startswith(START):
        text = text.lstrip()[len(START):]
    if text.rstrip().endswith(END):
        text = text.rstrip()[: -len(END)]
    return text.strip("\n")


def parse_action(text: str) -> Action:
    text = strip_tokens(text)
    header, _, body = text.partition("\n")
    header = header.strip()
    m = re.match(r"^(\w+)\b\s*(.*)$", header)
    if not m or m.group(1) not in TOOLS:
        raise reject("", "bad-action", "one of " + ", ".join(TOOLS), header[:40])
    tool, rest = m.group(1), m.group(2).strip()
    if tool == "set":
        hm = re.match(r"^(\S+)\s+:\s+(.+)$", rest)
        if not hm:
            raise reject("", "bad-action", "set PATH : TYPE", header[:60])
        return Action("set", path=hm.group(1), type_text=hm.group(2).strip(), body=body)
    if tool == "copy":
        hm = re.match(r"^(\S+)\s+to\s+(\S+)$", rest)
        if not hm:
            raise reject("", "bad-action", "copy SRC to DST", header[:60])
        return Action("copy", path=hm.group(1), dst=hm.group(2))
    if tool == "reduce":
        paths = rest.split()
        if not paths:
            raise reject("", "bad-action", "reduce PATH ...", header[:60])
        return Action("reduce", paths=paths)
    if tool == "eval":
        if rest:
            raise reject("", "bad-action", "eval with the code in the body", header[:60])
        return Action("eval", body=body)
    if not rest or " " in rest:
        raise reject("", "bad-action", f"{tool} PATH", header[:60])
    return Action(tool, path=rest, body=body)
