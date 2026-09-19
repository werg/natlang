"""The host side of the I/O boundary (SPEC 10): binding inputs, running, exporting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .nodes import Lambda, Pending
from .types import DictT, ListT, TEXT, TypeEnv
from .values import coerce, load_program


def import_path(path: Path, t, env: TypeEnv) -> Any:
    """Type-directed import of a file or directory."""
    rt = env.resolve(t)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if not p.name.startswith("."))
        if isinstance(rt, ListT):
            return [import_path(p, rt.elem, env) for p in files]
        if isinstance(rt, DictT):
            return {p.stem: import_path(p, rt.elem, env) for p in files}
        raise ValueError(f"{path} is a directory but the parameter type is not a list or Dict")
    text = path.read_text()
    if path.suffix in (".json", ".yaml", ".yml"):
        return yaml.safe_load(text)
    return text if rt == TEXT else yaml.safe_load(text)


def instantiate(fn) -> Lambda:
    """A fresh root lambda for a function of a code base (natlang/codebase.py)."""
    root = load_program({"$lambda": {**fn.to_lambda_doc(), "function": fn.name}})
    root.codebase = fn.codebase
    return root


def load_fold(step_file: Path, init, source) -> Pending:
    """A long-lived program: a root Fold whose step is a code-base function `f(acc, item) -> State` and whose
    list is open, fed by `source` (an iterable of events; "$close" or exhaustion ends the run)."""
    from .codebase import load_function
    from .runtime import OpenList
    fn = load_function(step_file)
    args = {n.rstrip("?"): t for n, t in fn.args.items()}
    if set(args) != {"acc", "item"} or args["acc"].strip() != fn.returns.strip():
        raise ValueError(f"a fold step is f(acc: S, item: A) -> S; got {fn.signature}")
    root = load_program({"$fold": {"type": f"Fold<{args['item']}, {args['acc']}>", "types": dict(fn.types), "init": init,
                                   "step": {"$lambda": {**fn.to_lambda_doc(), "function": fn.name}}}})
    root.step.codebase = fn.codebase
    root.over = OpenList(source)
    return root


def load(program_file: Path, inputs: dict, streams: dict | None = None) -> Pending:
    """Load a program and bind its inputs. `streams` maps a part of a root combinator (`over`) to an iterable
    of events: the part becomes an open list that the run pulls from until the source ends or yields "$close"
    (SPEC 4.4). A YAML program may carry its own `streams:` for tests."""
    program_file = Path(program_file)
    if program_file.suffix in (".nl", ".ts"):            # a code base on disk: main.nl + main/
        from .codebase import load_function
        root = instantiate(load_function(program_file))
    else:
        root = load_program(yaml.safe_load(program_file.read_text()).get("program")
                            or yaml.safe_load(program_file.read_text()))
    doc_streams = {}
    if program_file.suffix not in (".nl", ".ts"):
        doc_streams = (yaml.safe_load(program_file.read_text()) or {}).get("streams") or {}
    for part, source in {**doc_streams, **(streams or {})}.items():
        from .runtime import OpenList
        if not hasattr(root, part) or isinstance(root, Lambda):
            raise ValueError(f"a stream needs a root Map or Fold with a part `{part}`")
        setattr(root, part, OpenList(source))
    if isinstance(root, Lambda):
        env = root.env(TypeEnv())
        for name, src in inputs.items():
            ft = root.type.params.get(name)
            if ft is None:
                raise ValueError(f"{name} is not a parameter of the program")
            value = import_path(Path(src), ft[0], env) if _is_file(src) else src
            root.in_[name] = coerce(value, ft[0], env, yaml=False, path=f"args/{name}")
    return root


def export(value: Any, fmt: str = "yaml") -> str:
    from .values import dump
    data = dump(value)
    return json.dumps(data, indent=2, ensure_ascii=False) if fmt == "json" else \
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _is_file(src) -> bool:
    """An input may name a file to import. A long text is a value, not a path (and probing it raises ENAMETOOLONG)."""
    if isinstance(src, Path):
        return src.exists()
    return isinstance(src, str) and len(src) < 256 and "\n" not in src and Path(src).exists()
