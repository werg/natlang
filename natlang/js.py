"""QuickJS sandbox for `eval` and crisp lambdas (SPEC 9)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .diag import reject
from .nodes import MISSING, is_pending
from .types import format_type

try:  # optional dependency
    import quickjs
except ImportError:  # pragma: no cover
    quickjs = None

_PRELUDE = (Path(__file__).parent / "prelude.js").read_text()
TIME_LIMIT_S, MEMORY_LIMIT = 2, 64 * 1024 * 1024


class JsError(Exception):
    pass


def to_js(x: Any) -> Any:
    """JSON view of a value; pending nodes become opaque handles."""
    if x is MISSING:
        return None
    if is_pending(x):
        return {"$pending": format_type(x.type), "status": x.status}
    if isinstance(x, list):
        return [to_js(i) for i in x]
    if isinstance(x, dict):
        return {k: to_js(v) for k, v in x.items()}
    return x


def run(code: str, scope: dict, fx: Callable[[str, str, list], Any], *, body: bool, path: str,
        effectful: bool = False) -> Any:
    """Run `code` as an expression script (eval) or a function body (crisp lambda).

    Limitation: the QuickJS binding cannot call into Python while a time limit
    is set. Pure code therefore runs with the time limit and no bridge (any
    `fx` call is `effect-undeclared`); code in a lambda that declares effects
    runs with the bridge and without the time limit. A subprocess worker with
    a wall-clock kill should replace this.
    """
    if quickjs is None:
        raise JsError("quickjs is not installed: `uv pip install quickjs-ng`")
    ctx = quickjs.Context()
    ctx.set_memory_limit(MEMORY_LIMIT)

    if effectful:
        def bridge(cap, fn, args_json):
            try:
                return json.dumps({"value": fx(cap, fn, json.loads(args_json))})
            except EffectError as e:
                return json.dumps({"__error": e.code})

        ctx.add_callable("__fx", bridge)
    else:
        ctx.set_time_limit(TIME_LIMIT_S)
        ctx.eval('globalThis.__fx = () => JSON.stringify({ __error: "effect-undeclared" });')
    ctx.eval(_PRELUDE)
    ctx.eval("globalThis.self = __deepFreeze(" + json.dumps(scope) + "); globalThis.args = self.args; globalThis.locals = self.let || {};")
    runner = ctx.eval("__runBody" if body else "__runExpr")
    try:
        out = runner(code)
    except quickjs.JSException as e:
        msg = str(e).split("\n")[0]
        m = re.search(r"NATLANG:([\w-]+)", msg)
        if m:
            raise reject(path, m.group(1))
        if "read-only" in msg or "not extensible" in msg or "Cannot assign" in msg:
            raise reject(path, "eval-cannot-write", got=msg)
        raise JsError(msg)
    return json.loads(out)


class EffectError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
