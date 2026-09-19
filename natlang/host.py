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


def load(program_file: Path, inputs: dict) -> Pending:
    root = load_program(yaml.safe_load(program_file.read_text()).get("program")
                        or yaml.safe_load(program_file.read_text()))
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
