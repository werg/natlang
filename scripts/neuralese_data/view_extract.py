"""Deterministic extraction over artifacts, for the `view`/`ask` corpus's code-computed targets.

Small, dependency-free engines whose answers are computed, never generated:

- `JsonDoc`: JSONPath-style paths (`$.a.b[0]['key with space']`) over parsed JSON or YAML values.
- `HtmlDoc`: a forgiving DOM from the standard library's HTML parser, with a small CSS selector subset
  (type, `.class`, `#id`, `[attr]`, `[attr=value]`, compound, descendant and `>` child combinators).
- table helpers: CSV rendering and typed column reads.
- `python_facts`: parameters, called names and the docstring span of one Python function (via `ast`).
- tool-output facts: exit code, failing test ids, the final exception and its innermost frame, grep hits.
"""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import textwrap
import warnings
from html.parser import HTMLParser

# ---------------------------------------------------------------------------------------------- JSON / YAML

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def path_join(parent: str, key) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    if _IDENT.match(key):
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=False)}]"


def walk(value, path: str = "$"):
    """(path, value) for every node, parents before children, in document order."""
    yield path, value
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk(v, path_join(path, str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, path_join(path, i))


def scalar_text(value) -> str:
    """How a scalar answer is written: strings as themselves, everything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def depth(path: str) -> int:
    return path.count(".") + path.count("[")


# ---------------------------------------------------------------------------------------------- HTML

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_WS = re.compile(r"\s+")


class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict, parent: "Node | None"):
        self.tag, self.attrs, self.children, self.parent = tag, attrs, [], parent

    def text(self) -> str:
        parts: list[str] = []

        def rec(n):
            for c in n.children:
                if isinstance(c, str):
                    parts.append(c)
                elif c.tag not in ("script", "style"):
                    rec(c)
        rec(self)
        return _WS.sub(" ", " ".join(parts)).strip()

    def elements(self):
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.elements()

    def classes(self) -> set:
        return set((self.attrs.get("class") or "").split())


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {}, None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k: (v if v is not None else "") for k, v in attrs}, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, {k: (v if v is not None else "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:  # close up to the matching open tag; ignore strays
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


_COMPOUND = re.compile(r"([a-zA-Z][a-zA-Z0-9-]*)?((?:[.#][A-Za-z0-9_-]+|\[[a-zA-Z_:][-a-zA-Z0-9_:.]*(?:=\"[^\"]*\")?\])*)$")
_PART = re.compile(r"([.#])([A-Za-z0-9_-]+)|\[([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:=\"([^\"]*)\")?\]")


def _parse_compound(text: str):
    m = _COMPOUND.match(text)
    if not m or not text:
        raise ValueError(f"unsupported selector part {text!r}")
    tag, rest = m.group(1), m.group(2) or ""
    conds = []
    for p in _PART.finditer(rest):
        if p.group(1) == ".":
            conds.append(("class", p.group(2)))
        elif p.group(1) == "#":
            conds.append(("id", p.group(2)))
        else:
            conds.append(("attr", p.group(3), p.group(4)))
    return tag.lower() if tag else None, conds


def _matches(node: Node, compound) -> bool:
    tag, conds = compound
    if tag and node.tag != tag:
        return False
    for c in conds:
        if c[0] == "class" and c[1] not in node.classes():
            return False
        if c[0] == "id" and node.attrs.get("id") != c[1]:
            return False
        if c[0] == "attr" and (c[1] not in node.attrs or (c[2] is not None and node.attrs[c[1]] != c[2])):
            return False
    return True


def parse_selector(selector: str):
    tokens = selector.replace(">", " > ").split()
    steps, combinator = [], " "
    for t in tokens:
        if t == ">":
            combinator = ">"
            continue
        steps.append((combinator, _parse_compound(t)))
        combinator = " "
    if not steps:
        raise ValueError("empty selector")
    return steps


class HtmlDoc:
    def __init__(self, html: str):
        b = _Builder()
        b.feed(html)
        b.close()
        self.root = b.root

    def select(self, selector: str) -> list[Node]:
        steps = parse_selector(selector)

        def ok(node: Node, i: int) -> bool:
            comb, compound = steps[i]
            if not _matches(node, compound):
                return False
            if i == 0:
                return True
            p = node.parent
            if comb == ">":
                return p is not None and p.tag != "#document" and ok(p, i - 1)
            while p is not None and p.tag != "#document":
                if ok(p, i - 1):
                    return True
                p = p.parent
            return False
        return [n for n in self.root.elements() if ok(n, len(steps) - 1)]

    def title(self) -> str:
        found = self.select("title")
        return found[0].text() if found else ""


_SCRIPT = re.compile(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TID = re.compile(r"\s+tid=\"?\d+\"?")
_BLANKS = re.compile(r"\n\s*\n+")


def clean_html(html: str, drop_attrs: tuple[str, ...] = ()) -> str:
    """Remove scripts, styles, inline SVG, comments and the named dataset-only attributes; collapse blank runs."""
    out = _SCRIPT.sub("", html)
    out = _COMMENT.sub("", out)
    if "tid" in drop_attrs:
        out = _TID.sub("", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return _BLANKS.sub("\n", out).strip()


# ---------------------------------------------------------------------------------------------- tables

def table_csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue().rstrip("\n")


_NUM = re.compile(r"^[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def number(cell: str):
    c = cell.strip().replace("−", "-")
    if not _NUM.match(c):
        return None
    v = float(c.replace(",", ""))
    return int(v) if v.is_integer() and "." not in c else v


# ---------------------------------------------------------------------------------------------- Python code

def _call_name(node) -> str | None:
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    elif parts:
        parts.append("…")  # call on an expression: keep the attribute chain only
    else:
        return None
    name = ".".join(reversed(parts))
    return name if not name.startswith("…") else name[2:]


def python_facts(code: str) -> dict | None:
    """Facts about the first function in `code`: name, parameters, called names, docstring and its line span."""
    src = textwrap.dedent(code)
    try:
        with warnings.catch_warnings():  # upstream code with invalid escape sequences still parses
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None:
        return None
    a = fn.args
    params = [x.arg for x in [*a.posonlyargs, *a.args]]
    if a.vararg:
        params.append("*" + a.vararg.arg)
    params += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        params.append("**" + a.kwarg.arg)
    # Order by where each callee expression ends in the source (first appearance), one entry per name.
    found = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            name = _call_name(n)
            if name:
                found.append(((n.func.end_lineno, n.func.end_col_offset), name))
    calls: list[str] = []
    for _, name in sorted(found):
        if name not in calls:
            calls.append(name)
    doc_node = fn.body[0] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
        and isinstance(fn.body[0].value.value, str) else None
    return {"name": fn.name, "params": params, "calls": calls, "source": src,
            "docstring": ast.get_docstring(fn) if doc_node else None,
            "doc_span": (doc_node.lineno, doc_node.end_lineno) if doc_node else None}


def without_docstring(facts: dict) -> str:
    """The function's source with its docstring statement removed (a `pass` keeps an emptied body valid)."""
    lines = facts["source"].split("\n")
    if not facts["doc_span"]:
        return facts["source"]
    start, end = facts["doc_span"]
    indent = re.match(r"\s*", lines[start - 1]).group(0)
    rest = lines[end:]
    body_left = any(l.strip() and not l.strip().startswith("#") for l in rest)
    out = lines[: start - 1] + ([] if body_left else [indent + "pass"]) + rest
    return "\n".join(out).rstrip() + "\n"


def summary_sentence(docstring: str) -> str | None:
    """The docstring's first paragraph, whitespace-joined; None when it is not a prose description."""
    para = docstring.strip().split("\n\n")[0]
    text = _WS.sub(" ", para).strip()
    if not text or text.startswith((":", "@", ">>>", "Args", "Parameters", "Returns", "..")):
        return None
    words = text.split()
    if len(words) < 3 or len(words) > 80:
        return None
    return text


# ---------------------------------------------------------------------------------------------- tool outputs

_EXIT = re.compile(r"\[Command finished with exit code (-?\d+)\]|exit code (-?\d+)")
_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+::\S+)", re.M)
_PYTEST_SUMMARY = re.compile(r"^=+ (.*?(?:passed|failed|error|errors|skipped).*?) in [\d.]+s(?: \([^)]*\))? =+$", re.M)
_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (\S+)', re.M)
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failure))(?::\s?(.*))?$")
_GREP = re.compile(r"^(\.?/?[\w./-]+\.[A-Za-z0-9]{1,8}):(\d+):", re.M)


def exit_code(text: str) -> int | None:
    found = _EXIT.findall(text)
    if not found:
        return None
    a, b = found[-1]
    return int(a or b)


def failing_tests(text: str) -> list[str]:
    out: list[str] = []
    for t in _FAILED.findall(text):
        if t not in out:
            out.append(t)
    return out


def pytest_summary(text: str) -> str | None:
    found = _PYTEST_SUMMARY.findall(text)
    return found[-1].strip() if found else None


def final_exception(text: str) -> dict | None:
    """The last traceback's exception line and its innermost frame."""
    at = text.rfind("Traceback (most recent call last):")
    if at < 0:
        return None
    tail = text[at:]
    frames = _FRAME.findall(tail)
    if not frames:
        return None
    for line in tail.split("\n")[1:]:
        m = _EXC.match(line.strip())
        if m and not line.startswith(" "):
            path, lineno, func = frames[-1]
            return {"type": m.group(1), "message": (m.group(2) or "").strip(), "path": path, "line": int(lineno),
                    "function": func}
    return None


def grep_hits(text: str) -> list[tuple[str, int]]:
    return [(p, int(n)) for p, n in _GREP.findall(text)]


# ---------------------------------------------------------------------------------------------- licences

# Ordered: more specific texts first (LGPL/AGPL before GPL; BSD-3 before BSD-2). Matching is on the licence file's
# text, so a reviewer should spot-check (VIEW_CORPUS.md §6); an unmatched file is "unknown", never guessed.
LICENSE_PATTERNS = [
    ("AGPL-3.0", re.compile(r"GNU AFFERO GENERAL PUBLIC LICENSE", re.I)),
    ("LGPL-3.0", re.compile(r"GNU LESSER GENERAL PUBLIC LICENSE\s+Version 3", re.I)),
    ("LGPL-2.1", re.compile(r"GNU LESSER GENERAL PUBLIC LICENSE|GNU LIBRARY GENERAL PUBLIC LICENSE", re.I)),
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", re.I)),
    ("GPL-2.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 2", re.I)),
    ("MPL-2.0", re.compile(r"Mozilla Public License,?\s+(?:Version|v\.?)\s*2\.0", re.I)),
    ("EPL-1.0", re.compile(r"Eclipse Public License", re.I)),
    ("Apache-2.0", re.compile(r"Apache License,?\s+Version 2\.0", re.I)),
    ("MIT", re.compile(r"Permission is hereby granted, free of charge", re.I)),
    ("BSD-3-Clause", re.compile(r"Redistribution and use in source and binary forms[\s\S]*(?:Neither the name|names of its\s+contributors)", re.I)),
    ("BSD-2-Clause", re.compile(r"Redistribution and use in source and binary forms", re.I)),
    ("ISC", re.compile(r"Permission to use, copy, modify, and/or distribute this software for any purpose", re.I)),
    ("Unlicense", re.compile(r"This is free and unencumbered software released into the public domain", re.I)),
    ("CC0-1.0", re.compile(r"CC0 1\.0 Universal|Creative Commons Zero", re.I)),
    ("Zlib", re.compile(r"This software is provided 'as-is', without any express or implied\s+warranty", re.I)),
    ("BSL-1.0", re.compile(r"Boost Software License", re.I)),
    ("WTFPL", re.compile(r"DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE", re.I)),
]

LICENSE_CLASSES = {
    "permissive": {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "CC0-1.0", "Zlib", "BSL-1.0",
                   "WTFPL", "0BSD", "LicenseRef-PublicDomain"},
    "attribution": {"CC-BY-4.0", "CC-BY-3.0"},
    "share-alike": {"CC-BY-SA-4.0", "CC-BY-SA-3.0"},
    "weak-copyleft": {"LGPL-2.1", "LGPL-3.0", "MPL-2.0", "EPL-1.0"},
    "copyleft": {"GPL-2.0", "GPL-3.0", "AGPL-3.0"},
}
# Ordered from least to most restrictive for a training-data licence review.
CLASS_ORDER = ("permissive", "attribution", "share-alike", "weak-copyleft", "copyleft", "unverified", "noncommercial")


def detect_license(text: str) -> str | None:
    for spdx, pattern in LICENSE_PATTERNS:
        if pattern.search(text):
            return spdx
    return None


def license_class(spdx: str | None) -> str:
    if not spdx:
        return "unverified"
    if "-NC" in spdx:
        return "noncommercial"
    for name, members in LICENSE_CLASSES.items():
        if spdx in members:
            return name
    return "unverified"
