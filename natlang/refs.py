"""Typed slot references and path resolution inside a lambda's subtree."""
from __future__ import annotations

from typing import Any, Callable, Optional

from .diag import reject
from .nodes import MISSING, RUNNING, FoldNode, IterateNode, Lambda, MapNode, Pending, is_pending
from .paths import Path
from .types import (NUM, TEXT, DictT, ListT, Record, Type, TypeEnv, UnionT)
from .values import part_type


class Ref:
    """A slot: somewhere a value or pending node can live, with its declared type."""

    def __init__(self, *, type: Optional[Type], env: TypeEnv, path: str, holder=None, attr=None,
                 parent: "Ref" = None, container=None, key=None, deny: str = ""):
        self.type, self.env, self.path = type, env, path
        self.holder, self.attr = holder, attr
        self.parent, self.container, self.key = parent, container, key
        self.deny = deny  # "" when writable, else a diagnostic code

    # -- storage
    def _container(self):
        if self.container is not None:
            return self.container
        return self.parent.get() if self.parent is not None else MISSING

    def get(self) -> Any:
        if self.holder is not None:
            return getattr(self.holder, self.attr)
        c = self._container()
        if isinstance(c, dict):
            return c.get(self.key, MISSING)
        if isinstance(c, list) and isinstance(self.key, int) and 0 <= self.key < len(c):
            return c[self.key]
        return MISSING

    def exists(self) -> bool:
        return self.get() is not MISSING

    def set(self, value: Any) -> None:
        if self.holder is not None:
            setattr(self.holder, self.attr, value)
            return
        c = self._container()
        if c is MISSING or c is None:
            rt = self.parent.env.resolve(self.parent.type) if self.parent.type is not None else None
            c = [] if isinstance(rt, ListT) else {}
            self.parent.set(c)
        if isinstance(c, list):
            if self.key == "+" or self.key == len(c):
                c.append(value)
            else:
                c[self.key] = value
        else:
            c[self.key] = value

    def delete(self) -> None:
        if self.holder is not None:
            setattr(self.holder, self.attr, MISSING)
            return
        c = self._container()
        if isinstance(c, dict):
            c.pop(self.key, None)
        elif isinstance(c, list) and isinstance(self.key, int) and 0 <= self.key < len(c):
            c.pop(self.key)

    def slot_key(self):
        if self.holder is not None:
            return (id(self.holder), self.attr)
        return (id(self._container()), self.key)

    # -- navigation into values
    def child(self, seg: str, *, create: bool = False) -> "Ref":
        path = f"{self.path}/{seg}"
        if self.type is None:
            raise reject(path, "no-such-path")
        rt = self.env.resolve(self.type)
        value = self.get()
        if isinstance(rt, UnionT):
            rt = _pick_member(rt, value, seg, self.env) or rt
        if isinstance(rt, Record):
            ft = rt.get(seg)
            if ft is None:
                raise reject(path, "unknown-field", "one of " + ", ".join(rt.names))
            return Ref(type=ft[0], env=self.env, path=path, parent=self, key=seg, deny=self.deny)
        if isinstance(rt, DictT):
            if seg.startswith("$"):
                raise reject(path, "reserved-key")
            return Ref(type=rt.elem, env=self.env, path=path, parent=self, key=seg, deny=self.deny)
        if isinstance(rt, ListT):
            n = len(value) if isinstance(value, list) else 0
            if seg == "+":
                return Ref(type=rt.elem, env=self.env, path=path, parent=self, key="+", deny=self.deny)
            if not seg.isdigit():
                raise reject(path, "no-such-path", "a list index")
            i = int(seg)
            if i > n or (i == n and not create):
                raise reject(path, "no-such-path", f"an index below {n}")
            return Ref(type=rt.elem, env=self.env, path=path, parent=self, key=i, deny=self.deny)
        raise reject(path, "no-such-path", "a container")


class LetRef(Ref):
    """`let`: the locals of a lambda. Each local has the type given by the write that created it."""

    def __init__(self, lam, env, deny):
        super().__init__(type=None, env=env, path="let", holder=lam, attr="let", deny=deny)

    def child(self, seg: str, *, create: bool = False) -> Ref:
        t = self.holder.let_types.get(seg)
        if t is None:
            raise reject(f"let/{seg}", "no-such-path", "an existing local (a write with a type creates one)")
        return Ref(type=t, env=self.env, path=f"let/{seg}", container=self.holder.let, key=seg, deny=self.deny)


def _pick_member(u: UnionT, value, seg, env):
    for m in u.members:
        rm = env.resolve(m)
        if isinstance(rm, Record) and rm.get(seg):
            return rm
        if isinstance(rm, (ListT, DictT)) and isinstance(value, (list, dict)):
            return rm
    return None


def _enter(node: Pending, seg: str, env: TypeEnv, path: str, deny: str, acting: Lambda,
           outer_elem: Optional[Type]) -> Ref:
    """Resolve a part of a pending node."""
    inner = node.env(env)
    p = f"{path}/{seg}" if path else seg
    frozen = deny or ("frozen" if node.status == RUNNING and node is not acting else "")
    if isinstance(node, Lambda):
        if seg in ("instructions", "code"):
            if seg != node.kind:
                raise reject(p, "no-such-path", node.kind)
            return Ref(type=TEXT, env=inner, path=p, holder=node, attr="body", deny=frozen)
        if seg == "args":
            d = deny or ("not-writable" if node is acting else frozen)
            return Ref(type=node.type.params, env=inner, path=p, holder=node, attr="in_", deny=d)
        if seg == "return":
            return Ref(type=node.type.returns, env=inner, path=p, holder=node, attr="ret", deny=frozen)
        if seg == "let" and node is acting:          # locals are private to the lambda that owns them
            return LetRef(node, inner, deny)
        raise reject(p, "no-such-path", "instructions, args, let, or return")
    if isinstance(node, MapNode):
        if seg in ("over", "fn"):
            return Ref(type=part_type(node, seg), env=inner, path=p, holder=node, attr=seg, deny=frozen)
        if seg.isdigit() and node.slots is not None and int(seg) < len(node.slots):
            return Ref(type=outer_elem or node.type.b, env=inner, path=p, container=node.slots,
                       key=int(seg), deny=frozen)
        raise reject(p, "no-such-path", "over, fn, or a slot index")
    if isinstance(node, FoldNode):
        if seg in ("over", "init", "step"):
            return Ref(type=part_type(node, seg), env=inner, path=p, holder=node, attr=seg, deny=frozen)
        if seg == "acc":
            return Ref(type=node.type.s, env=inner, path=p, holder=node, attr="acc", deny="not-writable")
        if seg == "at":
            return Ref(type=NUM, env=inner, path=p, holder=node, attr="at", deny="not-writable")
        if seg == "current" and node.current is not None:
            return Ref(type=part_type(node, "step"), env=inner, path=p, holder=node, attr="current",
                       deny=frozen)
        raise reject(p, "no-such-path", "over, init, step, acc, at, or current")
    if isinstance(node, IterateNode):
        if seg in ("init", "step", "check", "max"):
            return Ref(type=part_type(node, seg), env=inner, path=p, holder=node, attr=seg, deny=frozen)
        if seg == "state":
            return Ref(type=node.type.s, env=inner, path=p, holder=node, attr="state", deny="not-writable")
        if seg == "iteration":
            return Ref(type=NUM, env=inner, path=p, holder=node, attr="iteration", deny="not-writable")
        if seg == "current" and node.current is not None:
            return Ref(type=part_type(node, "step"), env=inner, path=p, holder=node, attr="current",
                       deny=frozen)
        raise reject(p, "no-such-path", "init, step, check, max, state, or iteration")
    raise reject(p, "no-such-path")


def resolve(acting: Lambda, outer_env: TypeEnv, path: Path, *, create: bool = False) -> Ref:
    """Resolve `path` relative to the acting lambda."""
    segs = list(path.segs)
    ref = _enter(acting, segs[0], outer_env, "", "", acting, None)
    for seg in segs[1:]:
        value = ref.get()
        if is_pending(value):
            rt = ref.env.resolve(ref.type) if ref.type is not None else None
            outer_elem = rt.elem if isinstance(rt, ListT) else None
            ref = _enter(value, seg, ref.env, ref.path, ref.deny, acting, outer_elem)
        else:
            ref = ref.child(seg, create=create)
    return ref


def pending_refs_under(ref: Ref):
    """Yield refs to pending nodes found in the value at `ref` (not descending into them)."""
    value = ref.get()
    if is_pending(value):
        yield ref
        return
    if isinstance(value, dict):
        keys = list(value.keys())
    elif isinstance(value, list):
        keys = [str(i) for i in range(len(value))]
    else:
        return
    for k in keys:
        try:
            yield from pending_refs_under(ref.child(str(k)))
        except Exception:
            continue
