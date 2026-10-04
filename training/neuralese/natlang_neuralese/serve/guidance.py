"""Guided generation: the reply checks behind the `guidance` request field (both servers; the llama.cpp fork follows
this file).

Hard constraints only where errors are pure form, check-and-backtrack where the model should keep its own choices:

Guidance is off unless a request asks for it (`"guidance": true` or an object of settings).

- **Envelope.** With `require_call` (default: the request's `tool_choice` is "required") the reply is forced to open a tool call; the call's
  name must be one of the offered tools, checked when its `(` arrives.
- **Eval code, line by line.** Inside an `eval(code=…)` string (decoded from the call's quoted form), every line is
  checked when it completes:
  - repetition: the line (stripped, at least `MIN_REPEAT_CHARS` long) occurs `repeat` times among the code's lines;
  - redeclaration: the line declares (`const`/`let`/`var`) a name already declared at the same indentation, with no
    line of smaller indentation (a closed block) in between;
  - syntax: tree-sitter (TypeScript) finds an error or a missing token inside the completed text that does not touch
    its end (an unfinished construct at the end, `foo(` or an open `{`, is only incomplete, not wrong).
  When the code string closes, the whole code is checked the same way. The unfinished last line is checked for a run:
  its end is one chunk (`RUN_MIN`–`RUN_MAX` characters) repeated `run` times in a row (a runaway that never ends the
  line, `x||x||x||…`).

A rejection names the reply offset where the offending line (or call name) starts. The server rolls back to its
snapshot there, bans the token it chose first at that point and samples again, at most `retries` times per point;
then the line is accepted as it is. Masking every invalid token would force the model into tokens it finds unlikely;
rejection keeps its own distribution except where it is definitely wrong. Neuralese blocks inside code are read as
the identifier `__nz` (the engine renders them so in the reply it checks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_REPEAT_CHARS = 8
RUN_MIN, RUN_MAX = 3, 60
_DECLARATION = re.compile(r"^(\s*)(?:const|let|var)\s+([A-Za-z_$][\w$]*)")
_CALL_START = "<|tool_call_start|>"
_EVAL_CODE = re.compile(r"\beval\(\s*code\s*=\s*(['\"])")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "0": "\0"}

_parser = None


def _ts_parser():
    global _parser
    if _parser is None:
        import tree_sitter_typescript
        from tree_sitter import Language, Parser

        _parser = Parser(Language(tree_sitter_typescript.language_typescript()))
    return _parser


def syntax_errors(code: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of tree-sitter errors and missing tokens in `code`."""
    data = code.encode()
    found: list[tuple[int, int]] = []
    stack = [_ts_parser().parse(data).root_node]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            found.append((len(data[:node.start_byte].decode(errors="ignore")), len(data[:node.end_byte].decode(errors="ignore"))))
        elif node.has_error:
            stack.extend(node.children)
    return found


@dataclass
class Settings:
    require_call: bool = False
    tools: list[str] | None = None  # names a call may use; None: any
    repeat: int = 3  # occurrences of one line that reject it (0: off)
    run: int = 4  # repetitions of one chunk at the end of the unfinished line that reject it (0: off)
    syntax: bool = True
    retries: int = 4

    @staticmethod
    def of(body: dict | None, tools: list | None, tool_choice=None) -> "Settings | None":
        """The request's guidance (`{"require_call"?, "tools"?, "repeat"?, "syntax"?, "retries"?}`, or true), or None.
        `tools` narrows the call names allowed below the offered tools."""
        if not body:
            return None
        body = body if isinstance(body, dict) else {}
        names = body.get("tools") or [t.get("function", t).get("name") for t in tools or [] if isinstance(t, dict)]
        return Settings(require_call=bool(body.get("require_call", tool_choice == "required")),
                        tools=[n for n in names if n] or None, repeat=int(body.get("repeat", 3)), run=int(body.get("run", 4)),
                        syntax=bool(body.get("syntax", True)), retries=int(body.get("retries", 4)))


def ends_in_run(line: str, times: int) -> bool:
    """`line` ends with one chunk of RUN_MIN..RUN_MAX characters repeated `times` times in a row."""
    for size in range(RUN_MIN, min(RUN_MAX, len(line) // times) + 1):
        unit = line[-size:]
        if unit.strip() and line.endswith(unit * times):
            return True
    return False


@dataclass
class _Code:
    raw_start: int  # reply offset of the code's first character (after the quote)
    text: str  # decoded code
    raw_lines: list[int]  # reply offset where each decoded line starts
    closed: bool


def code_strings(reply: str) -> list[_Code]:
    """Every eval call's code in the reply, decoded from its quoted form, with where each line starts in the reply."""
    out = []
    for match in _EVAL_CODE.finditer(reply):
        quote, i = match.group(1), match.end()
        text, lines, closed = [], [i], False
        while i < len(reply):
            c = reply[i]
            if c == "\\" and i + 1 < len(reply):
                decoded = _ESCAPES.get(reply[i + 1], reply[i + 1])
                text.append(decoded)
                i += 2
                if decoded == "\n":
                    lines.append(i)
                continue
            if c == "\\":
                break  # an escape split across tokens: wait for the next one
            if c == quote:
                closed = True
                break
            text.append(c)
            i += 1
            if c == "\n":
                lines.append(i)
        out.append(_Code(match.end(), "".join(text), lines, closed))
    return out


@dataclass
class Guide:
    """The checks over one reply. `check(reply)` runs after every token."""
    settings: Settings
    checked: set = field(default_factory=set)  # (code start, line index) already accepted
    names_checked: set = field(default_factory=set)
    rejections: list = field(default_factory=list)

    def forced_prefix(self) -> str:
        return _CALL_START + "[" if self.settings.require_call else ""

    def check(self, reply: str) -> tuple[str, int] | None:
        """None, or (reason, reply offset to roll back to)."""
        verdict = self._names(reply)
        if verdict:
            return verdict
        for code in code_strings(reply):
            verdict = self._code(code)
            if verdict:
                return verdict
        return None

    def rewind(self, offset: int):
        """The reply was cut back to `offset`: what was checked from there on is checked again (accepted points stay)."""
        self.names_checked = {start for start in self.names_checked if start < offset}
        self.checked = {key for key in self.checked if key[0] == "at" or key[1] < offset}

    def accept(self, at: int):
        """Retries at `at` are exhausted: what is there stands."""
        self.checked.add(("at", at))

    def _names(self, reply: str):
        tools = self.settings.tools
        if not tools:
            return None
        for match in re.finditer(r"(?:\[|,\s*)([A-Za-z_][\w.]*)\(", reply):
            start = match.start(1)
            if start in self.names_checked:
                continue
            self.names_checked.add(start)
            if match.group(1) not in tools and ("at", start) not in self.checked:
                return ("unknown-tool", start)
        return None

    def _code(self, code: _Code):
        lines = code.text.split("\n")
        complete = len(lines) if code.closed else len(lines) - 1
        if not code.closed and self.settings.run and ("at", code.raw_lines[-1]) not in self.checked and \
                ends_in_run(lines[-1], self.settings.run):
            return ("repetition", code.raw_lines[-1])
        for index in range(complete):
            key = ("line", code.raw_lines[index], code.closed and index == complete - 1)
            if key in self.checked:
                continue
            self.checked.add(key)
            at = code.raw_lines[index]
            if ("at", at) in self.checked:
                continue
            line = lines[index].strip()
            if self.settings.repeat and len(line) >= MIN_REPEAT_CHARS and \
                    sum(1 for other in lines[:index + 1] if other.strip() == line) >= self.settings.repeat:
                return ("repetition", at)
            declared = _DECLARATION.match(lines[index])
            if declared:
                indent, name = declared.group(1), declared.group(2)
                for earlier in reversed(lines[:index]):
                    if earlier.strip() and len(earlier) - len(earlier.lstrip()) < len(indent):
                        break  # a block closed (or opened) at a smaller indentation
                    match = _DECLARATION.match(earlier)
                    if match and match.group(1) == indent and match.group(2) == name:
                        return ("redeclaration", at)
            if self.settings.syntax:
                final = code.closed and index == complete - 1
                prefix = "\n".join(lines[:index + 1]) + ("" if final else "\n")
                line_start = len("\n".join(lines[:index])) + (1 if index else 0)
                end = len(prefix.rstrip())
                for _, stop in syntax_errors(prefix):
                    # Errors ending before this line belong to lines already accepted; before the code closes, an
                    # error reaching the end is an unfinished construct, not a wrong one.
                    if stop >= line_start and (final or stop < end):
                        return ("syntax", at)
        return None
