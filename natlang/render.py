"""Rendering policy render/0.1 (SPEC 8): the text the model is shown."""
from __future__ import annotations

from typing import Any

from .nodes import MISSING, QUIESCED, FoldNode, IterateNode, Lambda, MapNode, is_pending
from .types import DictT, ListT, Record, TypeEnv, UnionT, format_type

VERSION = "render/0.1"
INLINE, PREVIEW_ITEMS, MAX_DEPTH, NOTE_CHARS = 80, 3, 3, 60


def numbered(text: str) -> str:
    lines = text.splitlines()
    w = len(str(len(lines)))
    return "\n".join(f"  {str(i).rjust(w)}| {l}" for i, l in enumerate(lines, 1))


def scalar(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, str):
        if "\n" in x.rstrip("\n"):
            first = x.splitlines()[0]
            return f'"{first[:INLINE]}" ({len(x.splitlines())} lines)'
        s = x.rstrip("\n")
        return f'"{s}"' if len(s) <= INLINE else f'"{s[:INLINE]}" … ({len(s)} chars)'
    return repr(x)


def pending_line(node) -> str:
    head = f"{format_type(node.type)}  {node.status}"
    if isinstance(node, MapNode) and node.slots is not None:
        done = sum(1 for s in node.slots if not is_pending(s))
        head += f"  {done} of {len(node.slots)} reduced"
    if isinstance(node, FoldNode) and node.acc is not MISSING:
        n = len(node.over) if isinstance(node.over, list) else "?"
        head += f"  at {node.at} of {n}"
    if isinstance(node, IterateNode) and node.state is not MISSING:
        head += f"  iteration {node.iteration} of max {node.max}"
    if node.status == QUIESCED and node.note:
        head += f'  "{node.note[:NOTE_CHARS]}"'
    return head


def render(value: Any, t, env: TypeEnv, indent: int = 0, depth: int = 0) -> list:
    """Lines describing `value` of declared type `t`."""
    pad = "  " * indent
    if value is MISSING:
        return [f"{pad}·"]
    if is_pending(value):
        return [f"{pad}{pending_line(value)}"]
    rt = env.resolve(t) if t is not None else None
    if isinstance(rt, UnionT):
        rt = next((env.resolve(m) for m in rt.members
                   if isinstance(env.resolve(m), (Record, ListT, DictT))), rt)
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return [f"{pad}…"]
        out = []
        names = rt.names if isinstance(rt, Record) else list(value.keys())
        for n in names:
            ft = rt.get(n)[0] if isinstance(rt, Record) else (rt.elem if isinstance(rt, DictT) else None)
            if n not in value:
                if isinstance(rt, Record) and not rt.get(n)[1]:
                    out.append(f"{pad}{n}  {format_type(ft)}  ·")
                continue
            out += _entry(n, value[n], ft, env, indent, depth)
        return out or [f"{pad}(empty)"]
    if isinstance(value, list):
        if depth >= MAX_DEPTH:
            return [f"{pad}…"]
        et = rt.elem if isinstance(rt, ListT) else None
        out = []
        for i, x in enumerate(value[:PREVIEW_ITEMS]):
            out += _entry(str(i), x, et, env, indent, depth)
        if len(value) > PREVIEW_ITEMS:
            out.append(f"{pad}… {len(value) - PREVIEW_ITEMS} more")
        return out or [f"{pad}(empty)"]
    return [f"{pad}{scalar(value)}"]


def _entry(name, x, t, env, indent, depth):
    pad = "  " * indent
    tt = format_type(t) if t is not None else ""
    if is_pending(x):
        return [f"{pad}{name}  {pending_line(x)}"]
    if isinstance(x, list):
        open_mark = "open, " if getattr(x, "is_open", False) else ""
        head = f"{pad}{name}  {tt}  {open_mark}{len(x)} {'so far' if open_mark else 'items'}"
        return [head] + render(x, t, env, indent + 1, depth + 1)
    if isinstance(x, dict):
        return [f"{pad}{name}  {tt}"] + render(x, t, env, indent + 1, depth + 1)
    return [f"{pad}{name}  {tt}  {scalar(x)}"]


def opening(lam: Lambda, outer_env: TypeEnv, *, events: str = "none", holes: int = 0,
            blocking: int = 0, cold: bool = False) -> str:
    env = lam.env(outer_env)
    lines = [f"events: {events}", f"problems: {blocking} blocking · {holes} holes"]
    if cold and lam.journal:
        lines.append("effects so far:")
        lines += [f"  {j.get('seq')}. {j.get('capability')} {j.get('args_preview', '')} {j.get('status', '')}"
                  for j in lam.journal]
    n = len(lam.body.splitlines())
    lines += ["", f"{lam.kind}  Text  {n} lines", numbered(lam.body) if lam.body else "  (empty)", "", "args"]
    lines += render(lam.in_, lam.type.params, env, 1, 0) if lam.in_ else ["  (nothing bound)"]
    lines += ["", f"return  {format_type(lam.type.returns)}"]
    lines += render(lam.ret, lam.type.returns, env, 1, 0)
    return "\n".join(lines)
