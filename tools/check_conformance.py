#!/usr/bin/env python3
"""Static checks for the conformance suite against spec/SPEC.md v0.1.

- every YAML file parses and has the required keys
- every type expression parses under the SPEC section 2 grammar
- named types resolve
- every pending-node wrapper is well formed
- bound inputs are declared parameters
- `set` headers in canonical traces and harness scripts carry parseable types
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRIMS = {"Text", "Num", "Bool", "Null", "Blob"}
BUILTIN_NAMES = {"LoopVerdict"}
WRAPPERS = {"$lambda": "Lambda", "$map": "Map", "$fold": "Fold", "$iterate": "Iterate"}
PARTS = {
    "Map": {"over", "fn"},
    "Fold": {"over", "init", "step"},
    "Iterate": {"init", "step", "check", "max"},
}
META = {"type", "types", "effects", "status", "note", "effects_journal"}
HEADER_WORDS = {"read", "edit", "set", "unset", "copy", "reduce", "reopen", "eval"}


class TypeErr(Exception):
    pass


TOKEN = re.compile(r'\s*(?:("(?:[^"\\]|\\.)*")|([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)|(\[\])|([{}<>|,:?()]))')


def tokenize(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise TypeErr(f"bad character at {pos}: {s[pos:pos+10]!r}")
        out.append(m.group(1) or m.group(2) or m.group(3) or m.group(4))
        pos = m.end()
    return out


class Parser:
    def __init__(self, text):
        self.toks = tokenize(text)
        self.i = 0
        self.names = set()

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, tok=None):
        t = self.peek()
        if t is None or (tok is not None and t != tok):
            raise TypeErr(f"expected {tok!r}, got {t!r}")
        self.i += 1
        return t

    def parse(self):
        t = self.union()
        if self.peek() is not None:
            raise TypeErr(f"trailing input at {self.peek()!r}")
        return t

    def union(self):
        members = [self.postfix()]
        while self.peek() == "|":
            self.eat("|")
            members.append(self.postfix())
        return members[0] if len(members) == 1 else ("union", members)

    def postfix(self):
        t = self.atom()
        while self.peek() == "[]":
            self.eat("[]")
            t = ("list", t)
        return t

    def atom(self):
        t = self.peek()
        if t is None:
            raise TypeErr("unexpected end of type")
        if t == "(":
            self.eat("(")
            inner = self.union()
            self.eat(")")
            return inner
        if t == "{":
            return self.record()
        if t.startswith('"'):
            self.eat()
            return ("lit", t[1:-1])
        if re.match(r"-?\d", t):
            self.eat()
            return ("lit", float(t))
        if not re.match(r"[A-Za-z_]", t):
            raise TypeErr(f"unexpected {t!r}")
        self.eat()
        if t in PRIMS:
            return ("prim", t)
        if t == "Dict":
            return ("dict", self.args(1)[0])
        if t == "Lambda":
            p, r = self.args(2)
            if p[0] != "record":
                raise TypeErr("Lambda params must be a record type")
            return ("Lambda", p, r)
        if t in ("Map", "Fold"):
            a, b = self.args(2)
            return (t, a, b)
        if t == "Iterate":
            return ("Iterate", self.args(1)[0])
        self.names.add(t)
        return ("name", t)

    def args(self, n):
        self.eat("<")
        out = [self.union()]
        while self.peek() == ",":
            self.eat(",")
            out.append(self.union())
        self.eat(">")
        if len(out) != n:
            raise TypeErr(f"expected {n} type arguments, got {len(out)}")
        return out

    def record(self):
        self.eat("{")
        fields = {}
        while self.peek() != "}":
            name = self.eat()
            if not re.match(r"[A-Za-z_]", name):
                raise TypeErr(f"bad field name {name!r}")
            optional = False
            if self.peek() == "?":
                self.eat("?")
                optional = True
            self.eat(":")
            fields[name] = (self.union(), optional)
            if self.peek() == ",":
                self.eat(",")
        self.eat("}")
        return ("record", fields)


def parse_type(text):
    p = Parser(text)
    return p.parse(), p.names


class Checker:
    def __init__(self, path):
        self.path = path
        self.errors = []
        self.types_seen = 0

    def err(self, msg):
        self.errors.append(f"{self.path.relative_to(ROOT)}: {msg}")

    def check_type(self, text, scope, where):
        if "${" in text:  # template literal inside crisp code; not static
            return None
        try:
            t, names = parse_type(text)
        except TypeErr as e:
            self.err(f"{where}: type {text!r}: {e}")
            return None
        self.types_seen += 1
        for n in names - scope - BUILTIN_NAMES:
            self.err(f"{where}: unknown type name {n!r} in {text!r}")
        return t

    def walk(self, node, scope, where):
        if isinstance(node, list):
            for i, x in enumerate(node):
                self.walk(x, scope, f"{where}/{i}")
            return
        if not isinstance(node, dict):
            return
        wrappers = [k for k in node if k in WRAPPERS]
        if wrappers:
            if len(node) != 1:
                self.err(f"{where}: wrapper {wrappers[0]} must be the only key")
            self.pending(wrappers[0], node[wrappers[0]], scope, where)
            return
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("$"):
                self.err(f"{where}: reserved key {k!r}")
            self.walk(v, scope, f"{where}/{k}")

    def pending(self, wrapper, body, scope, where):
        kind = WRAPPERS[wrapper]
        if not isinstance(body, dict) or "type" not in body:
            self.err(f"{where}: {wrapper} needs a `type`")
            return
        scope = set(scope)
        for name, text in (body.get("types") or {}).items():
            scope.add(name)
        for name, text in (body.get("types") or {}).items():
            self.check_type(text, scope, f"{where}/types/{name}")
        t = self.check_type(body["type"], scope, where)
        if t is not None and t[0] != kind:
            self.err(f"{where}: {wrapper} has type {t[0]}")
            return
        if kind == "Lambda":
            has_i, has_c = "instructions" in body, "code" in body
            if has_i == has_c:
                self.err(f"{where}: exactly one of instructions/code required")
            extra = set(body) - META - {"instructions", "code", "args", "return"}
            if extra:
                self.err(f"{where}: unexpected keys {sorted(extra)}")
            if t is not None:
                params = t[1][1]
                for k in (body.get("args") or {}):
                    if k not in params:
                        self.err(f"{where}: args/{k} is not a declared parameter")
            self.walk(body.get("args"), scope, f"{where}/args")
            self.walk(body.get("return"), scope, f"{where}/return")
        else:
            extra = set(body) - META - PARTS[kind] - {"acc", "at", "state", "iteration"}
            if extra:
                self.err(f"{where}: unexpected keys {sorted(extra)}")
            for part in PARTS[kind]:
                if part in body:
                    self.walk(body[part], scope, f"{where}/{part}")
            if kind == "Iterate" and "max" not in body:
                self.err(f"{where}: Iterate requires max")
        return t

    def actions(self, text, scope, where):
        """Check `set PATH : TYPE` headers and header words in an action block."""
        for line in text.splitlines():
            m = re.match(r"\s*(?:>>>\s*)?(\w+)\b(.*)$", line)
            if not m:
                continue
            is_action_line = line.lstrip().startswith(">>>") or where.endswith("action")
            if not is_action_line:
                continue
            word, rest = m.group(1), m.group(2)
            if word not in HEADER_WORDS:
                self.err(f"{where}: unknown tool {word!r}")
            if word == "set":
                hm = re.match(r"\s*(\S+)\s+:\s+(.*)$", rest)
                if not hm:
                    self.err(f"{where}: malformed set header {line.strip()!r}")
                else:
                    self.check_type(hm.group(2), scope, f"{where}: set {hm.group(1)}")
            if where.endswith("action"):
                break  # only the first line of a scripted action is a header


def all_type_names(doc):
    names = set()

    def rec(n):
        if isinstance(n, dict):
            for k, v in n.items():
                if k == "types" and isinstance(v, dict):
                    names.update(v)
                rec(v)
        elif isinstance(n, list):
            for x in n:
                rec(x)

    rec(doc)
    return names


def check_file(path):
    c = Checker(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        c.err(f"YAML: {e}")
        return c
    for key in ("id", "title", "kind"):
        if key not in doc:
            c.err(f"missing key {key!r}")
    if doc.get("id") != path.stem:
        c.err(f"id {doc.get('id')!r} does not match file name")
    names = all_type_names(doc)
    if doc.get("kind") == "program":
        root = doc.get("program") or doc.get("start_state")
        if doc.get("program_file"):                     # a code base on disk (spec/CODEBASES.md)
            import os
            if not os.path.exists(os.path.join(os.path.dirname(path), doc["program_file"])):
                c.err(f"program_file not found: {doc['program_file']}")
        elif not root:
            c.err("program or start_state required")
        else:
            c.walk(root, set(), "program")
            wrapper = next(iter(root))
            body = root[wrapper]
            if "inputs" in doc and wrapper == "$lambda":
                try:
                    t, _ = parse_type(body["type"])
                    for k in doc["inputs"]:
                        if k not in t[1][1]:
                            c.err(f"inputs/{k} is not a declared parameter")
                    for k, (_, optional) in t[1][1].items():
                        if not optional and k not in doc["inputs"] and k not in (body.get("args") or {}):
                            c.err(f"required parameter {k!r} is not bound")
                except TypeErr:
                    pass
        if "expect" not in doc:
            c.err("expect required")
        if "lint" not in doc:
            c.err("lint required")
        for key in ("must", "must_not", "may"):
            for item in (doc.get("lint") or {}).get(key) or []:
                if not isinstance(item, str):
                    c.err(f"lint/{key}: entry is not a string (unquoted colon?): {item!r}")
        if "canonical_trace" in doc:
            c.actions(doc["canonical_trace"], names, "canonical_trace")
    elif doc.get("kind") == "harness":
        c.walk(doc.get("setup"), set(), "setup")
        for i, step in enumerate(doc.get("script") or []):
            if "action" not in step or "expect" not in step:
                c.err(f"script/{i}: action and expect required")
                continue
            c.actions(step["action"], names, f"script/{i}/action")
            for j, d in enumerate(step.get("during") or []):
                c.actions(d["action"], names, f"script/{i}/during/{j}/action")
    else:
        c.err(f"unknown kind {doc.get('kind')!r}")
    return c


def main():
    files = sorted((ROOT / "conformance").glob("*/*.yaml"))
    errors, types_seen = [], 0
    for f in files:
        c = check_file(f)
        errors += c.errors
        types_seen += c.types_seen
    # self-test of the type parser on cases that must fail
    for good in ['1 | 2 | 3', '"a" | "b"', '{ n: 1 | 2, s?: Text[] }[]', 'Lambda<{ k: 0 | 1 }, Dict<Num>>']:
        try:
            parse_type(good)
        except TypeErr as e:
            errors.append(f"type parser rejected valid type {good!r}: {e}")
    for bad in ["Lambda<Text, Bool>", "Map<Text>", "{ a Text }", "Text[", "Dict<>", '"a" |']:
        try:
            parse_type(bad)
            errors.append(f"type parser accepted invalid type {bad!r}")
        except TypeErr:
            pass
    print(f"{len(files)} files, {types_seen} type expressions checked")
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
