"""Type-directed construction and checking of values and pending nodes."""
from __future__ import annotations

import re
from typing import Any

from .diag import BLOCKS, HOLE, Diagnostic, Reject, reject
from .nodes import (MISSING, WRAPPERS, WRAPPER_OF, FoldNode, IterateNode, Lambda, MapNode,
                    Pending, is_pending)
from .types import (BOOL, NUM, TEXT, DictT, FoldT, IterateT, LambdaT, ListT, Lit, MapT, Name,
                    Prim, Record, TypeEnv, TypeSyntaxError, UnionT, fits, format_type, parse_type,
                    LOOP_VERDICT)

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _preview(x) -> str:
    if is_pending(x):
        return format_type(x.type)
    kind = {str: "Text", bool: "Bool", int: "Num", float: "Num", list: "list", dict: "record",
            type(None): "Null"}.get(type(x), type(x).__name__)
    s = repr(x) if not isinstance(x, str) else '"' + x + '"'
    return f"{kind} {s if len(s) <= 40 else s[:37] + '...'}"


def _wrapper_key(raw):
    if isinstance(raw, dict) and len(raw) >= 1:
        keys = [k for k in raw if k in WRAPPERS]
        if keys:
            if len(raw) != 1:
                return "invalid"
            return keys[0]
    return None


def coerce(raw: Any, t, env: TypeEnv, *, yaml: bool, path: str) -> Any:
    """Turn `raw` into a value of (draft) type `t`, or raise Reject.

    With yaml=True scalars arrive as strings (BaseLoader) and are read against
    the declared type. With yaml=False they are already typed Python values.
    Missing record fields are allowed here; see `problems` for commit checks.
    """
    want = env.resolve(t)
    if isinstance(want, LambdaT):
        # A higher-order slot. Body lambdas may declare extra, pre-bound params,
        # so this is looser than the generic `fits` on Lambda types.
        node = raw
        if not is_pending(node):
            if _wrapper_key(raw) != "$lambda":
                raise reject(path, "type-mismatch", format_type(t), _preview(raw))
            node = build_pending("$lambda", raw["$lambda"], env, yaml=yaml, path=path)
        if not isinstance(node, Lambda) or not body_lambda_fits(node.type, want, node.env(env)):
            raise reject(path, "type-does-not-fit-slot", format_type(t), format_type(node.type))
        return node
    if is_pending(raw):
        if not fits(raw.type, t, env):
            raise reject(path, "type-does-not-fit-slot", format_type(t), format_type(raw.type))
        return raw
    wk = _wrapper_key(raw)
    if wk == "invalid":
        raise reject(path, "reserved-key", "a single wrapper key")
    if wk:
        node = build_pending(wk, raw[wk], env, yaml=yaml, path=path)
        if not fits(node.type, t, env):
            raise reject(path, "type-does-not-fit-slot", format_type(t), format_type(node.type))
        return node

    rt = env.resolve(t)
    if isinstance(rt, UnionT):
        last = None
        for m in rt.members:
            try:
                return coerce(raw, m, env, yaml=yaml, path=path)
            except Reject as e:
                last = e
        raise reject(path, "type-mismatch", format_type(t), _preview(raw)) from last
    if isinstance(rt, Prim):
        return _coerce_prim(raw, rt, yaml, path)
    if isinstance(rt, Lit):
        if isinstance(rt.value, str):
            if isinstance(raw, str) and raw == rt.value:
                return raw
        else:
            v = _as_num(raw, yaml)
            if v is not None and v == rt.value:
                return rt.value
        raise reject(path, "type-mismatch", format_type(rt), _preview(raw))
    if isinstance(rt, ListT):
        if not isinstance(raw, list):
            raise reject(path, "type-mismatch", format_type(t), _preview(raw))
        return [coerce(x, rt.elem, env, yaml=yaml, path=f"{path}/{i}") for i, x in enumerate(raw)]
    if isinstance(rt, DictT):
        if not isinstance(raw, dict):
            raise reject(path, "type-mismatch", format_type(t), _preview(raw))
        out = {}
        for k, v in raw.items():
            k = str(k)
            if k.startswith("$"):
                raise reject(f"{path}/{k}", "reserved-key")
            out[k] = coerce(v, rt.elem, env, yaml=yaml, path=f"{path}/{k}")
        return out
    if isinstance(rt, Record):
        if not isinstance(raw, dict):
            raise reject(path, "type-mismatch", format_type(t), _preview(raw))
        out = {}
        for k, v in raw.items():
            k = str(k)
            if k.startswith("$"):
                raise reject(f"{path}/{k}", "reserved-key")
            ft = rt.get(k)
            if ft is None:
                raise reject(f"{path}/{k}", "unknown-field", format_type(rt))
            out[k] = coerce(v, ft[0], env, yaml=yaml, path=f"{path}/{k}")
        # keep declared field order
        return {n: out[n] for n in rt.names if n in out}
    if isinstance(rt, (LambdaT, MapT, FoldT, IterateT)):
        raise reject(path, "type-mismatch", format_type(t), _preview(raw))
    raise reject(path, "type-mismatch", format_type(t), _preview(raw))


def _as_num(raw, yaml):
    if yaml:
        if isinstance(raw, str) and _NUM_RE.match(raw.strip()):
            f = float(raw)
            return int(f) if f.is_integer() and not re.search(r"[.eE]", raw) else f
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if isinstance(raw, float) and raw.is_integer():
        return int(raw) if abs(raw) < 2**53 else raw
    return raw


def _coerce_prim(raw, rt: Prim, yaml: bool, path: str):
    if rt.name == "Text" or rt.name == "Blob":
        if isinstance(raw, str):
            return raw
    elif rt.name == "Num":
        v = _as_num(raw, yaml)
        if v is not None:
            return v
    elif rt.name == "Bool":
        if yaml and isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true"
        if not yaml and isinstance(raw, bool):
            return raw
    elif rt.name == "Null":
        if raw is None or (yaml and isinstance(raw, str) and raw.strip() in ("", "null", "~")):
            return None
    raise reject(path, "type-mismatch", rt.name, _preview(raw))


# --------------------------------------------------------------------------- pending nodes

_LAMBDA_KEYS = {"type", "types", "effects", "instructions", "code", "args", "return", "status", "note",
                "effects_journal"}


def build_pending(wrapper: str, body: Any, env: TypeEnv, *, yaml: bool, path: str,
                  header_type=None) -> Pending:
    if not isinstance(body, dict):
        raise reject(path, "type-mismatch", f"a mapping for {wrapper}", _preview(body))
    cls = WRAPPERS[wrapper]
    types, types_src = {}, {}
    for name, text in (body.get("types") or {}).items():
        try:
            types[str(name)] = parse_type(text)
        except TypeSyntaxError as e:
            raise reject(f"{path}/types/{name}", "type-mismatch", "a type", str(e))
        types_src[str(name)] = str(text)
    inner = env.child(types)
    if header_type is not None:
        t = header_type
    else:
        if "type" not in body:
            raise reject(path, "type-mismatch", f"{wrapper} with a `type`")
        try:
            t = parse_type(body["type"])
        except TypeSyntaxError as e:
            raise reject(f"{path}/type", "type-mismatch", "a type", str(e))
    try:
        inner.check_names(t)
        for tt in types.values():
            inner.check_names(tt)
    except TypeSyntaxError as e:
        raise reject(path, "type-mismatch", "declared type names", str(e))
    expected = {Lambda: LambdaT, MapNode: MapT, FoldNode: FoldT, IterateNode: IterateT}[cls]
    if not isinstance(t, expected):
        raise reject(path, "type-mismatch", f"a {expected.__name__[:-1]} type", format_type(t))

    common = dict(type=t, types=types, types_src=types_src)
    if "status" in body:
        common["status"] = str(body["status"])
    if "note" in body:
        common["note"] = str(body["note"])

    if cls is Lambda:
        extra = set(map(str, body)) - _LAMBDA_KEYS
        if extra:
            raise reject(f"{path}/{sorted(extra)[0]}", "unknown-field", "a Lambda part")
        has_i, has_c = "instructions" in body, "code" in body
        if has_i == has_c:
            raise reject(path, "type-mismatch", "exactly one of instructions / code")
        text = body["instructions"] if has_i else body["code"]
        if not isinstance(text, str):
            raise reject(path, "type-mismatch", "Text body", _preview(text))
        effects = body.get("effects") or []
        if not isinstance(effects, list):
            raise reject(f"{path}/effects", "type-mismatch", "a list of capabilities")
        node = Lambda(kind="instructions" if has_i else "code", body=_norm_text(text),
                      effects=[str(e) for e in effects], **common)
        raw_in = body.get("args") or {}
        if not isinstance(raw_in, dict):
            raise reject(f"{path}/args", "type-mismatch", format_type(t.params))
        for k, v in raw_in.items():
            ft = t.params.get(str(k))
            if ft is None:
                raise reject(f"{path}/args/{k}", "unknown-field", format_type(t.params))
            node.in_[str(k)] = coerce(v, ft[0], inner, yaml=yaml, path=f"{path}/args/{k}")
        if "return" in body:
            node.ret = coerce(body["return"], t.returns, inner, yaml=yaml, path=f"{path}/return")
        node.journal = list(body.get("effects_journal") or [])
        return node

    parts = {MapNode: ("over", "fn"), FoldNode: ("over", "init", "step"),
             IterateNode: ("init", "step", "check", "max")}[cls]
    state_keys = {"acc", "at", "state", "iteration"}
    extra = set(map(str, body)) - set(parts) - {"type", "types", "status", "note"} - state_keys
    if extra:
        raise reject(f"{path}/{sorted(extra)[0]}", "unknown-field", f"a {cls.__name__} part")
    node = cls(**common)
    for part in parts:
        if part in body:
            setattr(node, part, coerce(body[part], part_type(node, part), inner, yaml=yaml,
                                       path=f"{path}/{part}"))
    if isinstance(node, FoldNode):
        if "acc" in body:
            node.acc = coerce(body["acc"], t.s, inner, yaml=yaml, path=f"{path}/acc")
        if "at" in body:
            node.at = int(body["at"])
    if isinstance(node, IterateNode):
        if "state" in body:
            node.state = coerce(body["state"], t.s, inner, yaml=yaml, path=f"{path}/state")
        if "iteration" in body:
            node.iteration = int(body["iteration"])
    return node


def _norm_text(text: str) -> str:
    return text if text == "" or text.endswith("\n") else text + "\n"


def part_type(node: Pending, part: str):
    """Declared type of a combinator part."""
    t = node.type
    if isinstance(node, MapNode):
        return {"over": ListT(t.a),
                "fn": LambdaT(Record((("item", t.a, False),)), t.b)}[part]
    if isinstance(node, FoldNode):
        return {"over": ListT(t.a), "init": t.s, "acc": t.s,
                "step": LambdaT(Record((("acc", t.s, False), ("item", t.a, False))), t.s)}[part]
    if isinstance(node, IterateNode):
        return {"init": t.s, "state": t.s, "max": NUM,
                "step": LambdaT(Record((("state", t.s, False),)), t.s),
                "check": LambdaT(Record((("recent", ListT(t.s), False), ("iteration", NUM, False))),
                                 Name("LoopVerdict"))}[part]
    raise KeyError(part)


def body_lambda_fits(lam_t: LambdaT, want: LambdaT, env: TypeEnv) -> bool:
    """A body lambda may declare extra (pre-bound) params beyond the required ones."""
    if not fits(lam_t.returns, want.returns, env):
        return False
    for n, wt, _ in want.params.fields:
        got = lam_t.params.get(n)
        if got is None or not fits(wt, got[0], env):
            return False
    return True


# --------------------------------------------------------------------------- problems


def problems(value: Any, t, env: TypeEnv, path: str):
    """Return (holes, pending_paths) for a draft value of declared type `t`."""
    holes, pend = [], []
    _walk_problems(value, t, env, path, holes, pend)
    return holes, pend


def _walk_problems(value, t, env, path, holes, pend):
    if value is MISSING:
        holes.append(Diagnostic(path, "hole", HOLE, format_type(t)))
        return
    if is_pending(value):
        pend.append(path)
        return
    rt = env.resolve(t)
    if isinstance(rt, UnionT):
        for m in rt.members:
            try:
                coerce(value, m, env, yaml=False, path=path)
            except Reject:
                continue
            _walk_problems(value, m, env, path, holes, pend)
            return
        return
    if isinstance(rt, Record) and isinstance(value, dict):
        for n, ft, opt in rt.fields:
            if n not in value:
                if not opt:
                    holes.append(Diagnostic(f"{path}/{n}", "hole", HOLE, n))
            else:
                _walk_problems(value[n], ft, env, f"{path}/{n}", holes, pend)
    elif isinstance(rt, ListT) and isinstance(value, list):
        for i, x in enumerate(value):
            _walk_problems(x, rt.elem, env, f"{path}/{i}", holes, pend)
    elif isinstance(rt, DictT) and isinstance(value, dict):
        for k, x in value.items():
            _walk_problems(x, rt.elem, env, f"{path}/{k}", holes, pend)


def unbound_parts(node: Pending, env: TypeEnv, path: str):
    """Diagnostics for required parts / params that are not bound (SPEC 6.4.2)."""
    out = []
    inner = node.env(env)
    if isinstance(node, Lambda):
        for n, ft, opt in node.type.params.fields:
            if n not in node.in_:
                if not opt:
                    out.append(Diagnostic(f"{path}/args/{n}", "unbound-param", BLOCKS, format_type(ft)))
            else:
                h, _ = problems(node.in_[n], ft, inner, f"{path}/args/{n}")
                out += [Diagnostic(d.path, "unbound-param", BLOCKS, d.expected) for d in h]
        return out
    parts = {MapNode: ("over", "fn"), FoldNode: ("over", "init", "step"),
             IterateNode: ("init", "step", "check", "max")}[type(node)]
    for part in parts:
        if getattr(node, part) is MISSING:
            out.append(Diagnostic(f"{path}/{part}", "unbound-part", BLOCKS,
                                  format_type(part_type(node, part))))
    return out


# --------------------------------------------------------------------------- serialization


def dump(x: Any) -> Any:
    """Plain YAML-able form of a value or pending node (SPEC 11)."""
    if x is MISSING:
        return None
    if isinstance(x, list):
        return [dump(i) for i in x]
    if isinstance(x, dict):
        return {k: dump(v) for k, v in x.items()}
    if not is_pending(x):
        return x
    body = {"type": format_type(x.type)}
    if x.types:
        body["types"] = {n: x.types_src.get(n) or format_type(t) for n, t in x.types.items()}
    if x.status != "unreduced":
        body["status"] = x.status
    if x.note:
        body["note"] = x.note
    if isinstance(x, Lambda):
        if x.effects:
            body["effects"] = list(x.effects)
        body["instructions" if x.kind == "instructions" else "code"] = x.body
        if x.in_:
            body["args"] = {k: dump(v) for k, v in x.in_.items()}
        if x.ret is not MISSING:
            body["return"] = dump(x.ret)
        if x.journal:
            body["effects_journal"] = [dict(j) for j in x.journal]
    elif isinstance(x, MapNode):
        for part in ("over", "fn"):
            if getattr(x, part) is not MISSING:
                body[part] = dump(getattr(x, part))
        if x.slots is not None:
            body["slots"] = [dump(s) for s in x.slots]
    elif isinstance(x, FoldNode):
        for part in ("over", "init", "step", "acc"):
            if getattr(x, part) is not MISSING:
                body[part] = dump(getattr(x, part))
        if x.at:
            body["at"] = x.at
    elif isinstance(x, IterateNode):
        for part in ("init", "step", "check", "max", "state"):
            if getattr(x, part) is not MISSING:
                body[part] = dump(getattr(x, part))
        if x.iteration:
            body["iteration"] = x.iteration
    return {WRAPPER_OF[type(x)]: body}


def load_program(doc: Any, env: TypeEnv | None = None) -> Pending:
    """Build the root pending node from a parsed YAML document."""
    env = env or TypeEnv()
    wk = _wrapper_key(doc)
    if not wk or wk == "invalid":
        raise reject("", "type-mismatch", "a root pending node ($lambda, $map, $fold, $iterate)")
    return build_pending(wk, doc[wk], env, yaml=False, path="")
