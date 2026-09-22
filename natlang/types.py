"""Type grammar, environments, and the fit relation (SPEC 2)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Union as U


class TypeSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Prim:
    name: str  # Text | Num | Bool | Null | Blob | Folder | FileHandle


@dataclass(frozen=True)
class Lit:
    value: U[str, int, float]


@dataclass(frozen=True)
class Record:
    fields: tuple  # ((name, Type, optional), ...)

    def get(self, name):
        for n, t, opt in self.fields:
            if n == name:
                return t, opt
        return None

    @property
    def names(self):
        return [n for n, _, _ in self.fields]


@dataclass(frozen=True)
class ListT:
    elem: "Type"


@dataclass(frozen=True)
class DictT:
    elem: "Type"


@dataclass(frozen=True)
class UnionT:
    members: tuple


@dataclass(frozen=True)
class Name:
    name: str


@dataclass(frozen=True)
class LambdaT:
    params: Record
    returns: "Type"


@dataclass(frozen=True)
class MapT:
    a: "Type"
    b: "Type"


@dataclass(frozen=True)
class FoldT:
    a: "Type"
    s: "Type"


@dataclass(frozen=True)
class IterateT:
    s: "Type"


Type = U[Prim, Lit, Record, ListT, DictT, UnionT, Name, LambdaT, MapT, FoldT, IterateT]
PENDING_TYPES = (LambdaT, MapT, FoldT, IterateT)

TEXT, NUM, BOOL, NULL, BLOB, FOLDER, FILE = (
    Prim(n) for n in ("Text", "Num", "Bool", "Null", "Blob", "Folder", "FileHandle"))
_PRIMS = {t.name: t for t in (TEXT, NUM, BOOL, NULL, BLOB, FOLDER, FILE)}

_TOKEN = re.compile(
    r'\s*(?:("(?:[^"\\]|\\.)*")|(-?\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\[\])|([{}<>|,;:?()]))'
)


def _tokenize(text: str):
    text = text.strip()
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise TypeSyntaxError(f"bad character at {pos}: {text[pos:pos + 12]!r}")
        if m.group(1) is not None:
            out.append(("str", m.group(1)[1:-1]))
        elif m.group(2) is not None:
            out.append(("num", m.group(2)))
        elif m.group(3) is not None:
            out.append(("id", m.group(3)))
        elif m.group(4) is not None:
            out.append(("p", "[]"))
        else:
            out.append(("p", m.group(5)))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, text):
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def eat(self, value=None):
        kind, v = self.peek()
        if kind is None or (value is not None and v != value):
            raise TypeSyntaxError(f"expected {value!r}, got {v!r}")
        self.i += 1
        return kind, v

    def parse(self):
        t = self.union()
        if self.peek()[0] is not None:
            raise TypeSyntaxError(f"trailing input at {self.peek()[1]!r}")
        return t

    def union(self):
        members = [self.postfix()]
        while self.peek() == ("p", "|"):
            self.eat("|")
            members.append(self.postfix())
        if len(members) == 1:
            return members[0]
        flat = []
        for m in members:
            flat.extend(m.members if isinstance(m, UnionT) else [m])
        return UnionT(tuple(flat))

    def postfix(self):
        t = self.atom()
        while self.peek() == ("p", "[]"):
            self.eat("[]")
            t = ListT(t)
        return t

    def atom(self):
        kind, v = self.peek()
        if kind is None:
            raise TypeSyntaxError("unexpected end of type")
        if (kind, v) == ("p", "("):
            self.eat("(")
            inner = self.union()
            self.eat(")")
            return inner
        if (kind, v) == ("p", "{"):
            return self.record()
        if kind == "str":
            self.eat()
            return Lit(v)
        if kind == "num":
            self.eat()
            f = float(v)
            return Lit(int(f) if f.is_integer() and "." not in v else f)
        if kind != "id":
            raise TypeSyntaxError(f"unexpected {v!r}")
        self.eat()
        if v in _PRIMS:
            return _PRIMS[v]
        if v == "Dict":
            return DictT(self.args(1)[0])
        if v == "Lambda":
            p, r = self.args(2)
            if not isinstance(p, Record):
                raise TypeSyntaxError("Lambda params must be a record type")
            return LambdaT(p, r)
        if v == "Map":
            return MapT(*self.args(2))
        if v == "Fold":
            return FoldT(*self.args(2))
        if v == "Iterate":
            return IterateT(self.args(1)[0])
        return Name(v)

    def args(self, n):
        self.eat("<")
        out = [self.union()]
        while self.peek() == ("p", ","):
            self.eat(",")
            out.append(self.union())
        self.eat(">")
        if len(out) != n:
            raise TypeSyntaxError(f"expected {n} type arguments, got {len(out)}")
        return out

    def record(self):
        self.eat("{")
        fields = []
        while self.peek() != ("p", "}"):
            kind, name = self.eat()
            if kind != "id":
                raise TypeSyntaxError(f"bad field name {name!r}")
            optional = False
            if self.peek() == ("p", "?"):
                self.eat("?")
                optional = True
            self.eat(":")
            fields.append((name, self.union(), optional))
            if self.peek() in (("p", ","), ("p", ";")):
                self.eat()
        self.eat("}")
        names = [f[0] for f in fields]
        if len(set(names)) != len(names):
            raise TypeSyntaxError("duplicate field name")
        return Record(tuple(fields))


def parse_type(text: str) -> Type:
    if not isinstance(text, str):
        raise TypeSyntaxError(f"type must be a string, got {type(text).__name__}")
    return _Parser(text).parse()


def format_type(t: Type) -> str:
    if isinstance(t, Prim):
        return t.name
    if isinstance(t, Lit):
        return f'"{t.value}"' if isinstance(t.value, str) else repr(t.value)
    if isinstance(t, Record):
        inner = ", ".join(f"{n}{'?' if o else ''}: {format_type(ft)}" for n, ft, o in t.fields)
        return "{ " + inner + " }" if inner else "{}"
    if isinstance(t, ListT):
        e = format_type(t.elem)
        return f"({e})[]" if isinstance(t.elem, UnionT) else f"{e}[]"
    if isinstance(t, DictT):
        return f"Dict<{format_type(t.elem)}>"
    if isinstance(t, UnionT):
        return " | ".join(format_type(m) for m in t.members)
    if isinstance(t, Name):
        return t.name
    if isinstance(t, LambdaT):
        return f"Lambda<{format_type(t.params)}, {format_type(t.returns)}>"
    if isinstance(t, MapT):
        return f"Map<{format_type(t.a)}, {format_type(t.b)}>"
    if isinstance(t, FoldT):
        return f"Fold<{format_type(t.a)}, {format_type(t.s)}>"
    if isinstance(t, IterateT):
        return f"Iterate<{format_type(t.s)}>"
    raise TypeError(t)


LOOP_VERDICT = parse_type('{ reason: Text, verdict: "continue" | "done" | "degenerate" }')


@dataclass
class TypeEnv:
    """Named types visible at a point in the tree. Inner scopes shadow outer ones."""

    names: dict = field(default_factory=dict)
    parent: Optional["TypeEnv"] = None

    def lookup(self, name: str) -> Optional[Type]:
        env = self
        while env is not None:
            if name in env.names:
                return env.names[name]
            env = env.parent
        if name == "LoopVerdict":
            return LOOP_VERDICT
        return None

    def child(self, names: dict) -> "TypeEnv":
        return TypeEnv(dict(names), self) if names else self

    def resolve(self, t: Type) -> Type:
        seen = set()
        while isinstance(t, Name):
            if t.name in seen:
                raise TypeSyntaxError(f"type {t.name} is defined only in terms of itself")
            seen.add(t.name)
            found = self.lookup(t.name)
            if found is None:
                raise TypeSyntaxError(f"unknown type name {t.name}")
            t = found
        return t

    def check_names(self, t: Type) -> None:
        """Raise if `t` mentions a name that is not declared."""
        for n in _names_in(t):
            if self.lookup(n) is None:
                raise TypeSyntaxError(f"unknown type name {n}")


def _names_in(t: Type):
    if isinstance(t, Name):
        yield t.name
    elif isinstance(t, Record):
        for _, ft, _ in t.fields:
            yield from _names_in(ft)
    elif isinstance(t, (ListT, DictT)):
        yield from _names_in(t.elem)
    elif isinstance(t, UnionT):
        for m in t.members:
            yield from _names_in(m)
    elif isinstance(t, LambdaT):
        yield from _names_in(t.params)
        yield from _names_in(t.returns)
    elif isinstance(t, (MapT, FoldT)):
        yield from _names_in(t.a)
        yield from _names_in(t.b if isinstance(t, MapT) else t.s)
    elif isinstance(t, IterateT):
        yield from _names_in(t.s)


def result_type(t: Type) -> Type:
    """The value type a pending type reduces to."""
    if isinstance(t, LambdaT):
        return t.returns
    if isinstance(t, MapT):
        return ListT(t.b)
    if isinstance(t, FoldT):
        return t.s
    if isinstance(t, IterateT):
        return t.s
    raise TypeError(f"not a pending type: {t}")


def is_pending_type(t: Type, env: TypeEnv) -> bool:
    return isinstance(env.resolve(t), PENDING_TYPES)


def fits(a: Type, b: Type, env: TypeEnv, _seen=None) -> bool:
    """Does type `a` fit a slot of type `b`? (SPEC 2.1)"""
    _seen = _seen if _seen is not None else set()
    key = (id(a), id(b)) if not (isinstance(a, Name) and isinstance(b, Name)) else (a.name, b.name)
    if isinstance(a, Name) and isinstance(b, Name):
        if a.name == b.name or key in _seen:
            return True
        _seen.add(key)
    a, b = env.resolve(a), env.resolve(b)
    if a == b:
        return True
    # promise rule
    if isinstance(a, PENDING_TYPES) and not isinstance(b, PENDING_TYPES):
        return fits(result_type(a), b, env, _seen)
    if isinstance(a, UnionT):
        return all(fits(m, b, env, _seen) for m in a.members)
    if isinstance(b, UnionT):
        return any(fits(a, m, env, _seen) for m in b.members)
    if isinstance(a, Lit) and isinstance(b, Prim):      # a literal is a value of its base type: "spam" fits Text
        return b == (TEXT if isinstance(a.value, str) else NUM)
    if isinstance(a, ListT) and isinstance(b, ListT):
        return fits(a.elem, b.elem, env, _seen)
    if isinstance(a, DictT) and isinstance(b, DictT):
        return fits(a.elem, b.elem, env, _seen)
    if isinstance(a, Record) and isinstance(b, Record):
        for n, bt, bopt in b.fields:
            got = a.get(n)
            if got is None:
                if not bopt:
                    return False
                continue
            if not fits(got[0], bt, env, _seen):
                return False
        return all(b.get(n) is not None for n in a.names)
    if isinstance(a, LambdaT) and isinstance(b, LambdaT):
        return fits(a.returns, b.returns, env, _seen) and fits(b.params, a.params, env, _seen)
    return False
