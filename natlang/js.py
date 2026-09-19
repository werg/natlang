"""QuickJS sandbox for `eval` and crisp lambdas (SPEC 9)."""
from __future__ import annotations

import json
import re
import functools
import subprocess
import sys
import time
import selectors
import threading
import queue
from pathlib import Path
from typing import Any, Callable

from .diag import reject, Reject, Diagnostic
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


@functools.lru_cache(maxsize=2048)
def prepare(code: str) -> str:
    """Compile-check JS without executing it; strip erasable TS only when needed.

    Node is a parser here, never the executor of user code. QuickJS remains the sandbox.
    """
    if quickjs is None:
        raise JsError("quickjs is not installed: `uv pip install quickjs-ng`")
    ctx = quickjs.Context()
    ctx.set_memory_limit(MEMORY_LIMIT)
    ctx.set_time_limit(TIME_LIMIT_S)
    try:
        ctx.eval("new Function(" + json.dumps(code) + ")")
        return code
    except quickjs.JSException:
        pass
    parser = (
        "const fs = require('node:fs'); const {stripTypeScriptTypes} = require('node:module');"
        "try { const src = fs.readFileSync(0, 'utf8');"
        "const out = stripTypeScriptTypes('function __snippet(){\\n'+src+'\\n}', {mode:'strip'});"
        "process.stdout.write(out.slice(out.indexOf('{')+1, out.lastIndexOf('}')));"
        "} catch(e) { process.stderr.write(e.message); process.exit(1); }"
    )
    try:
        result = subprocess.run(["node", "--no-warnings", "-e", parser], input=code, text=True,
                                capture_output=True, timeout=5)
    except FileNotFoundError as e:
        raise JsError("TypeScript annotations require Node.js >=22.13; plain JavaScript needs only QuickJS") from e
    except subprocess.TimeoutExpired as e:
        raise JsError("TypeScript parsing timed out") from e
    if result.returncode:
        raise JsError("TypeScript syntax: " + result.stderr.strip()[:1000])
    return result.stdout


def _effect_worker(code, scope, fx, *, body, path):
    """A killable JS process; capabilities execute in the host, retaining its state.

    Host hooks are trusted Python. A timed-out hook may still complete, so its
    outcome is uncertain and must not be silently retried. Hosts must provide
    cancellation/idempotency for external operations that need those guarantees.
    """
    process = subprocess.Popen([sys.executable, "-u", "-m", "natlang.js_worker"],
                               cwd=Path(__file__).resolve().parent.parent,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True)
    deadline = time.monotonic() + TIME_LIMIT_S
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        process.stdin.write(json.dumps({"code": code, "scope": scope, "body": body, "path": path}) + "\n")
        process.stdin.flush()
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(left):
                raise JsError("effectful code wall-clock limit exceeded")
            line = process.stdout.readline()
            if not line:
                raise JsError("JavaScript worker exited without a result")
            msg = json.loads(line)
            if msg["kind"] == "result":
                return msg["value"]
            if msg["kind"] == "reject":
                raise Reject(*(Diagnostic(**d) for d in msg["diags"]))
            if msg["kind"] == "error":
                raise JsError(msg["message"])
            if msg["kind"] != "effect":
                raise JsError("invalid JavaScript worker message")
            replies = queue.Queue(maxsize=1)

            def invoke(request=msg, sink=replies):
                try:
                    sink.put({"value": fx(request["cap"], request["fn"], request["args"])})
                except EffectError as e:
                    sink.put({"__error": e.code})
                except Exception as e:
                    sink.put({"__error": "effect-error", "message": str(e)})

            threading.Thread(target=invoke, daemon=True).start()
            try:
                reply = replies.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty as e:
                raise JsError("host effect timed out; it may still complete, do not retry automatically") from e
            process.stdin.write(json.dumps(reply) + "\n")
            process.stdin.flush()
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


def run(code: str, scope: dict, fx: Callable[[str, str, list], Any], *, body: bool, path: str,
        effectful: bool = False) -> Any:
    code = prepare(code)
    if effectful:
        return _effect_worker(code, scope, fx, body=body, path=path)
    return _execute(code, scope, fx, body=body, path=path)


def _execute(code: str, scope: dict, fx: Callable[[str, str, list], Any], *, body: bool, path: str,
             effectful: bool = False) -> Any:
    """Execute in-process for pure code, or inside the killable worker for effectful code."""
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
    ctx.eval("globalThis.self = __deepFreeze(JSON.parse(" + json.dumps(json.dumps(scope)) + ")); globalThis.args = self.args; globalThis.locals = self.let || {};")
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
