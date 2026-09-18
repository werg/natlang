"""A small GBNF recognizer, for testing generated grammars without an inference engine.

Supports the portable subset natlang emits: literals, character classes,
rule references, grouping, alternation, and the postfix operators * + ?.
"""
from __future__ import annotations

import re
from functools import lru_cache


class GbnfError(ValueError):
    pass


def parse(text: str) -> dict:
    rules, cur, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z0-9-]*)\s*::=\s*(.*)$", line)
        if m:
            if cur:
                rules[cur] = _Expr(" ".join(buf)).parse()
            cur, buf = m.group(1), [m.group(2)]
        elif cur and line.strip():
            buf.append(line.strip())
    if cur:
        rules[cur] = _Expr(" ".join(buf)).parse()
    for name, node in rules.items():
        for ref in _refs(node):
            if ref not in rules:
                raise GbnfError(f"rule {name!r} references undefined rule {ref!r}")
    return rules


def _refs(node):
    if node[0] == "ref":
        yield node[1]
    elif node[0] in ("seq", "alt"):
        for c in node[1]:
            yield from _refs(c)
    elif node[0] in ("star", "plus", "opt"):
        yield from _refs(node[1])


class _Expr:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def parse(self):
        node = self.alt()
        self.ws()
        if self.i != len(self.s):
            raise GbnfError(f"unexpected {self.s[self.i:self.i + 20]!r}")
        return node

    def alt(self):
        items = [self.seq()]
        self.ws()
        while self.i < len(self.s) and self.s[self.i] == "|":
            self.i += 1
            items.append(self.seq())
            self.ws()
        return items[0] if len(items) == 1 else ("alt", items)

    def seq(self):
        items = []
        while True:
            self.ws()
            if self.i >= len(self.s) or self.s[self.i] in "|)":
                break
            items.append(self.postfix())
        return ("seq", items)

    def postfix(self):
        node = self.atom()
        while self.i < len(self.s) and self.s[self.i] in "*+?":
            node = ({"*": "star", "+": "plus", "?": "opt"}[self.s[self.i]], node)
            self.i += 1
        return node

    def atom(self):
        c = self.s[self.i]
        if c == "(":
            self.i += 1
            node = self.alt()
            self.ws()
            if self.i >= len(self.s) or self.s[self.i] != ")":
                raise GbnfError("missing )")
            self.i += 1
            return node
        if c == '"':
            return ("lit", self.string())
        if c == "[":
            return self.charclass()
        m = re.match(r"[A-Za-z][A-Za-z0-9-]*", self.s[self.i:])
        if not m:
            raise GbnfError(f"unexpected {self.s[self.i:self.i + 20]!r}")
        self.i += m.end()
        return ("ref", m.group(0))

    def _esc(self):
        c = self.s[self.i]
        if c != "\\":
            self.i += 1
            return c
        n = self.s[self.i + 1]
        self.i += 2
        if n == "x":
            v = chr(int(self.s[self.i:self.i + 2], 16))
            self.i += 2
            return v
        return {"n": "\n", "t": "\t", "r": "\r"}.get(n, n)

    def string(self):
        self.i += 1
        out = []
        while self.s[self.i] != '"':
            out.append(self._esc())
        self.i += 1
        return "".join(out)

    def charclass(self):
        self.i += 1
        neg = self.s[self.i] == "^"
        if neg:
            self.i += 1
        ranges = []
        while self.s[self.i] != "]":
            a = self._esc()
            if self.s[self.i] == "-" and self.s[self.i + 1] != "]":
                self.i += 1
                ranges.append((a, self._esc()))
            else:
                ranges.append((a, a))
        self.i += 1
        return ("cls", neg, tuple(ranges))


def accepts(grammar: str, text: str, root: str = "root") -> bool:
    rules = parse(grammar)
    n = len(text)

    @lru_cache(maxsize=None)
    def rule_ends(name, pos):
        return ends(rules[name], pos)

    def ends(node, pos):
        kind = node[0]
        if kind == "lit":
            return frozenset({pos + len(node[1])}) if text.startswith(node[1], pos) else frozenset()
        if kind == "cls":
            if pos >= n:
                return frozenset()
            hit = any(a <= text[pos] <= z for a, z in node[2])
            return frozenset({pos + 1}) if hit != node[1] else frozenset()
        if kind == "ref":
            return rule_ends(node[1], pos)
        if kind == "alt":
            out = set()
            for c in node[1]:
                out |= ends(c, pos)
            return frozenset(out)
        if kind == "seq":
            cur = {pos}
            for c in node[1]:
                nxt = set()
                for p in cur:
                    nxt |= ends(c, p)
                cur = nxt
                if not cur:
                    break
            return frozenset(cur)
        if kind == "opt":
            return frozenset({pos}) | ends(node[1], pos)
        if kind in ("star", "plus"):
            seen, frontier = set(), {pos}
            while frontier:
                nxt = set()
                for p in frontier:
                    for e in ends(node[1], p):
                        if e not in seen and e != p:
                            seen.add(e)
                            nxt.add(e)
                frontier = nxt
            return frozenset(seen | ({pos} if kind == "star" else set()))
        raise GbnfError(kind)

    return n in rule_ends(root, 0)


def longest_prefix(grammar: str, text: str, root: str = "root") -> int:
    """Length of the longest prefix of `text` that some derivation consumes. For debugging."""
    lo, hi = 0, len(text)
    best = 0
    for k in range(len(text) + 1):
        # a prefix is viable if the grammar followed by "anything" accepts it
        g = grammar + f'\nprobe ::= {root} anyrest\nanyrest ::= ( [^\\x00] )*\n'
        if accepts(g, text[:k] + "", "probe"):
            best = k
    return best
