"""Code bases (spec/CODEBASES.md): immutable function definitions, loaded from `.nl` / `.ts` files with
frontmatter and companion folders, or from the inline `codebase:` key of a YAML program.

A definition is shared by reference: instantiating a function builds a fresh lambda that points at the same
FunctionDef and therefore at the same code base. Scope is lexical: a function sees its companion folder and
its `uses`, nothing else.
"""
from __future__ import annotations

import re
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .diag import reject

MAX_FUNCTIONS = 12                     # listing budget per code base
_FRONT = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)
_FRONT_TS = re.compile(r"\A\s*/\*---\n(.*?)\n---\*/\n?(.*)\Z", re.S)
_KEYS = {"description", "args", "returns", "types", "uses", "effects", "engine"}


@dataclass(eq=False)
class FunctionDef:
    name: str
    kind: str                          # "instructions" | "code"
    body: str
    args: dict                         # name -> type text, in signature order
    returns: str
    engine: str = "quickjs-isolated"
    types: dict = field(default_factory=dict)      # name -> type text; own and inherited (lexical)
    description: str = ""
    effects: list = field(default_factory=list)
    codebase: dict = field(default_factory=dict)   # name -> FunctionDef
    source: str = ""                   # where it was defined, for messages

    def __deepcopy__(self, memo):      # definitions are immutable and shared
        return self

    @property
    def type_text(self) -> str:
        fields = ", ".join(f"{n}: {t}" for n, t in self.args.items())
        return "Lambda<" + ("{ " + fields + " }" if fields else "{}") + f", {self.returns}>"

    @property
    def signature(self) -> str:
        return f"{self.name}(" + ", ".join(f"{n}: {t}" for n, t in self.args.items()) + f") -> {self.returns}"

    def required(self) -> list:
        return [n for n in self.args if not n.endswith("?")]

    def to_inline(self) -> dict:
        """The inline form (`codebase:` of a YAML program), complete with the nested code base."""
        doc = {"description": self.description, "args": dict(self.args), "returns": self.returns, self.kind: self.body}
        if self.types:
            doc["types"] = dict(self.types)
        if self.effects:
            doc["effects"] = list(self.effects)
        if self.kind == "code" and self.engine != "quickjs-isolated":
            doc["engine"] = self.engine
        if self.codebase:
            doc["codebase"] = {n: f.to_inline() for n, f in self.codebase.items()}
        return doc

    def to_lambda_doc(self) -> dict:
        """The `$lambda` body of a fresh instance."""
        doc = {"type": self.type_text, self.kind: self.body}
        if self.types:
            doc["types"] = dict(self.types)
        if self.effects:
            doc["effects"] = list(self.effects)
        if self.kind == "code" and self.engine != "quickjs-isolated":
            doc["engine"] = self.engine
        return doc


def _make(name: str, meta: dict, body: str, kind: str, inherited: dict, source: str) -> FunctionDef:
    meta = meta or {}
    extra = set(meta) - _KEYS
    if extra:
        raise reject(source, "unknown-field", "frontmatter keys: " + ", ".join(sorted(_KEYS)), sorted(extra)[0])
    if "returns" not in meta:
        raise reject(source, "type-mismatch", "frontmatter with `returns`")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise reject(source, "type-mismatch", "a function name that is an identifier", name)
    return FunctionDef(name=name, kind=kind, body=body.strip("\n") + "\n",
                       engine=str(meta.get("engine") or "quickjs-isolated"),
                       args={str(k): str(v) for k, v in (meta.get("args") or {}).items()},
                       returns=str(meta["returns"]),
                       types={**inherited, **{str(k): str(v) for k, v in (meta.get("types") or {}).items()}},
                       description=str(meta.get("description") or ""), effects=list(meta.get("effects") or []),
                       source=source)


# -- from disk ---------------------------------------------------------------------------------------------
def _find(base: Path) -> Path:
    for ext in (".nl", ".ts"):
        p = base.with_name(base.name + ext)
        if p.is_file():
            return p
    raise reject(str(base), "no-such-path", "a function file (.nl or .ts)")


def _folder_types(folder: Path) -> dict:
    f = folder / "types.ts"
    if not f.is_file():
        return {}
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"type\s+(\w+)\s*=\s*([^;]+);", f.read_text())}


def load_function(path, *, _inherited: Optional[dict] = None, _cache: Optional[dict] = None) -> FunctionDef:
    """Load `path` (with or without extension) and, recursively, its code base."""
    cache = {} if _cache is None else _cache
    p = Path(path)
    file = p if p.suffix in (".nl", ".ts") and p.is_file() else _find(p.with_suffix("") if p.suffix else p)
    file = file.resolve()
    if file in cache:                                  # a link to something already loading or loaded: share it
        return cache[file]
    text = file.read_text()
    m = (_FRONT_TS if file.suffix == ".ts" else _FRONT).match(text)
    if not m:
        raise reject(str(file), "type-mismatch", "frontmatter between --- lines")
    meta = yaml.safe_load(m.group(1)) or {}
    inherited = {**(_inherited or {}), **_folder_types(file.parent)}
    fn = _make(file.stem, meta, m.group(2), "code" if file.suffix == ".ts" else "instructions", inherited, str(file))
    cache[file] = fn
    folder = file.with_suffix("")
    if folder.is_dir():
        for child in sorted(folder.iterdir()):
            if child.suffix in (".nl", ".ts") and child.name != "types.ts":
                fn.codebase[child.stem] = load_function(child, _inherited=fn.types, _cache=cache)
    for name, rel in (meta.get("uses") or {}).items():
        fn.codebase[str(name)] = load_function((file.parent / str(rel)), _inherited=None, _cache=cache)
    if _cache is None:
        check(fn)
    return fn


# -- inline (YAML programs) --------------------------------------------------------------------------------
def from_inline(entries: dict, inherited: dict, source: str, base: Optional[Path] = None) -> dict:
    out = {}
    for name, doc in (entries or {}).items():
        if isinstance(doc, dict) and "$link" in doc:
            out[str(name)] = load_function((base or Path(".")) / str(doc["$link"]))
            continue
        if not isinstance(doc, dict):
            raise reject(f"{source}/codebase/{name}", "type-mismatch", "a function definition")
        kind = "code" if "code" in doc else "instructions"
        meta = {k: v for k, v in doc.items() if k not in ("instructions", "code", "codebase")}
        fn = _make(str(name), meta, str(doc.get(kind) or ""), kind, inherited, f"{source}/codebase/{name}")
        fn.codebase = from_inline(doc.get("codebase"), fn.types, fn.source, base)
        out[str(name)] = fn
    return out


@dataclass(frozen=True)
class CheckedGraph:
    """A checked snapshot of supplied definitions, independent of file access."""

    root: FunctionDef
    definitions: dict[str, FunctionDef]
    revision: str

    def get(self, name: str) -> FunctionDef:
        try:
            return self.definitions[name]
        except KeyError:
            raise reject(name, "no-such-path", "a function in the checked source graph") from None


def from_definitions(entries: dict[str, dict], root: str) -> CheckedGraph:
    """Link a flat, already supplied source graph. `uses` names other entries.

    The input is copied before validation, so edits to a notebook cell or a
    caller's dictionary cannot alter a definition held by an active run.
    """
    supplied = copy.deepcopy(entries)
    if root not in supplied:
        raise reject(root, "no-such-path", "a root definition")
    definitions = {}
    links = {}
    for name, doc in supplied.items():
        if not isinstance(doc, dict):
            raise reject(name, "type-mismatch", "a function definition")
        kind = "code" if "code" in doc else "instructions"
        meta = {k: v for k, v in doc.items() if k not in ("code", "instructions", "codebase")}
        uses = meta.pop("uses", {}) or {}
        if not isinstance(uses, dict):
            raise reject(f"memory:{name}/uses", "type-mismatch", "a mapping of aliases to supplied definitions")
        fn = _make(str(name), meta, str(doc.get(kind) or ""), kind, {}, f"memory:{name}")
        definitions[str(name)] = fn
        links[str(name)] = uses
    for name, fn in definitions.items():
        for alias, target in links[name].items():
            if target not in definitions:
                raise reject(f"memory:{name}/uses/{alias}", "no-such-path", "a supplied definition", target)
            fn.codebase[str(alias)] = definitions[target]
    for fn in definitions.values():
        check(fn, strict_names=True)
    digest = hashlib.sha256(json.dumps(supplied, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
    return CheckedGraph(definitions[root], definitions, digest)


# -- load-time checks --------------------------------------------------------------------------------------
def check(root: FunctionDef, *, strict_names: bool = False) -> None:
    """Listing budget, and no recursion: a function can never reach itself. Repetition is `call` with
    over / init / until, whose bounds the harness controls."""
    seen, stack = set(), []

    def visit(fn):
        if id(fn) in [id(s) for s in stack]:
            cycle = stack[[id(s) for s in stack].index(id(fn)):]
            raise reject(fn.source, "recursion", "a code base in which no function can reach itself "
                                                 "(repeat with `until`, or go over a list, instead)",
                         " -> ".join(x.name for x in cycle + [fn]))
        if id(fn) in seen:
            return
        seen.add(id(fn))
        from .types import TypeEnv, TypeSyntaxError, parse_type
        try:
            named = {name: parse_type(text) for name, text in fn.types.items()}
            env = TypeEnv(named)
            signature = parse_type(fn.type_text)
            if strict_names:
                env.check_names(signature)
                for declared in named.values():
                    env.check_names(declared)
        except TypeSyntaxError as exc:
            raise reject(fn.source, "type-mismatch", "a valid checked function signature", str(exc)) from exc
        if len(fn.codebase) > MAX_FUNCTIONS:
            raise reject(fn.source, "codebase-too-large", f"at most {MAX_FUNCTIONS} functions", str(len(fn.codebase)))
        stack.append(fn)
        for child in fn.codebase.values():
            visit(child)
        stack.pop()

    visit(root)


def listing(codebase: dict) -> str:
    if not codebase:
        return ""
    width = max(len(f.signature) for f in codebase.values())
    return "\n".join(f"  {f.signature:<{width}}   {f.description}".rstrip() for f in codebase.values())
