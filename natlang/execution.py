"""Crisp execution boundary shared by authored bodies and inline eval.

The compatibility binding is isolated QuickJS. Validation belongs here so a
replacement engine cannot return an arbitrary Python object into typed state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from . import js
from .nodes import MISSING, is_pending


class ExecutionError(Exception):
    pass


@dataclass(frozen=True)
class CrispRequest:
    code: str
    scope: dict
    body: bool
    path: str
    effectful: bool = False


class Executor(Protocol):
    def run(self, request: CrispRequest, effect: Callable[[str, str, list], Any]) -> Any: ...


def portable(value: Any, path: str = "value") -> Any:
    """An exact JSON value; no missing, pending, nonfinite or host-native objects."""
    if value is MISSING or is_pending(value):
        raise ExecutionError(f"{path}: missing or pending value has no portable representation")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ExecutionError(f"{path}: integer is not exactly representable in JavaScript")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionError(f"{path}: nonfinite number")
        return value
    if isinstance(value, list):
        return [portable(item, f"{path}/{i}") for i, item in enumerate(value)]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ExecutionError(f"{path}: record keys must be text")
        return {k: portable(v, f"{path}/{k}") for k, v in value.items()}
    raise ExecutionError(f"{path}: unsupported host value {type(value).__name__}")


class QuickJSExecutor:
    """Existing isolated JS/TS behavior, including its legacy input view."""

    name = "quickjs-isolated"

    def run(self, request: CrispRequest, effect: Callable[[str, str, list], Any]) -> Any:
        scope = {key: js.to_js(value) for key, value in request.scope.items()}
        try:
            result = js.run(request.code, scope, effect, body=request.body, path=request.path,
                            effectful=request.effectful)
        except js.JsError as exc:
            raise ExecutionError(str(exc)) from exc
        return portable(result)
