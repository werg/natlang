"""Enumerate addressable typed slots for the structured tool surface."""
from __future__ import annotations

from .nodes import FoldNode, IterateNode, Lambda, MapNode, is_pending
from .paths import parse_path
from .refs import resolve
from .types import DictT, ListT, Record, TypeEnv, UnionT

MAX_ENUM_PATHS = 8
MAX_DEPTH = 6

class Slot:
    __slots__ = ("path", "ref", "value")

    def __init__(self, path, ref):
        self.path, self.ref, self.value = path, ref, ref.get()


def enumerate_slots(lam: Lambda, outer_env: TypeEnv):
    """Every addressable slot under the acting lambda, as (path text, Ref).

    Long lists contribute a pattern path ending in `/#`, meaning "any index".
    """
    out = []

    def visit(path: str, depth: int):
        try:
            ref = resolve(lam, outer_env, parse_path(path), create=True)
        except Exception:
            return
        out.append(Slot(path, ref))
        if depth >= MAX_DEPTH:
            return
        v = ref.get()
        if is_pending(v):
            for part in _parts(v):
                visit(f"{path}/{part}", depth + 1)
            return
        rt = ref.env.resolve(ref.type) if ref.type is not None else None
        if isinstance(rt, UnionT):
            rt = next((ref.env.resolve(m) for m in rt.members
                       if isinstance(ref.env.resolve(m), (Record, ListT, DictT))), rt)
        if isinstance(rt, Record):
            for n in rt.names:
                visit(f"{path}/{n}", depth + 1)
        elif isinstance(rt, DictT) and isinstance(v, dict):
            for k in list(v)[:MAX_ENUM_PATHS]:
                visit(f"{path}/{k}", depth + 1)
        elif isinstance(rt, ListT):
            n = len(v) if isinstance(v, list) else 0
            for i in range(min(n, MAX_ENUM_PATHS)):
                visit(f"{path}/{i}", depth + 1)

    for part in _parts(lam):
        visit(part, 0)
    return out


def _parts(node):
    if isinstance(node, Lambda):
        return [node.kind, "args", "return"] + [f"let/{n}" for n in node.let_types]
    if isinstance(node, MapNode):
        slots = [str(i) for i in range(min(len(node.slots or []), MAX_ENUM_PATHS))]
        return ["over", "fn"] + slots
    if isinstance(node, FoldNode):
        return ["over", "init", "step", "acc"] + (["current"] if node.current is not None else [])
    if isinstance(node, IterateNode):
        return ["init", "step", "check", "max", "state"] + (["current"] if node.current is not None else [])
    return []


