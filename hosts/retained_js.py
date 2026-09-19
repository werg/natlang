"""Optional retained Node.js execution binding with direct JS-native host objects."""
from __future__ import annotations

import json
import selectors
import subprocess
from pathlib import Path

from natlang.execution import CrispRequest, ExecutionError, portable
from natlang.nodes import MISSING

WORKER = Path(__file__).with_name("retained_js_worker.js")


def _scope(value):
    if isinstance(value, dict):
        return {k: _scope(v) for k, v in value.items() if v is not MISSING}
    if isinstance(value, list):
        return [_scope(v) for v in value]
    return portable(value)


class RetainedJSExecutor:
    name = "node-retained"

    def __init__(self, timeout: float = 10):
        self.timeout = timeout
        self.process = subprocess.Popen(["node", str(WORKER)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.events = []

    def run(self, request: CrispRequest, effect):
        if self.process.poll() is not None:
            raise ExecutionError("retained JS environment is disposed")
        if request.effectful:
            raise ExecutionError("declared Python effects are unavailable in node-retained; use host APIs")
        payload = {"kind": "execute", "code": request.code, "scope": _scope(request.scope),
                   "body": request.body, "timeoutMs": int(self.timeout * 1000)}
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout + 1):
                self.close()
                raise ExecutionError("retained JS evaluation timed out; native effects may have occurred")
            line = self.process.stdout.readline()
        if not line:
            raise ExecutionError("retained JS worker exited")
        message = json.loads(line)
        self.events.extend(message.get("events") or [])
        if message.get("kind") == "error":
            raise ExecutionError(message.get("message", "retained JS error"))
        if message.get("kind") != "result":
            raise ExecutionError("unexpected retained JS response")
        return portable(message["value"])

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"kind":"dispose"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
