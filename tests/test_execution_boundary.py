import math

import pytest

from natlang.execution import CrispRequest, ExecutionError, portable
from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import load_program


class RecordingExecutor:
    def __init__(self, value):
        self.value, self.requests = value, []

    def run(self, request, effect):
        self.requests.append(request)
        return self.value


def test_crisp_body_uses_injected_executor_and_common_type_check():
    root = load_program({"$lambda": {"type": "Lambda<{ n: Num }, Num>",
                                      "code": "return args.n + 1;", "args": {"n": 3}}})
    executor = RecordingExecutor(4)
    out, value = Runtime(None, executor=executor).run_root(root)
    assert (out.kind, value) == ("done", 4)
    assert executor.requests == [CrispRequest("return args.n + 1;\n", {"args": {"n": 3}, "return": root.ret}, True, "")]

    bad = RecordingExecutor({"wrong": True})
    out, _ = Runtime(None, executor=bad).run_root(load_program({"$lambda": {
        "type": "Lambda<{}, Num>", "code": "return 1;"}}))
    assert out.kind == "quiesced" and "rejected" in out.detail


def test_inline_eval_uses_same_boundary_without_committing_state():
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Compute a number."}})
    executor = RecordingExecutor([1, 2])
    session = Session(Runtime(None, executor=executor), root, TypeEnv())
    result = session.apply("run_code", {"code": "[1, 2]"})
    assert result.kind == "ok" and result.value == [1, 2]
    assert executor.requests[0].body is False and root.ret is not result.value


@pytest.mark.parametrize("value", [math.inf, math.nan, 2**53, b"bytes", object(), {"a": math.inf}, [2**53]])
def test_portable_boundary_rejects_lossy_or_native_values(value):
    with pytest.raises(ExecutionError):
        portable(value)


def test_portable_preserves_null_empty_boolean_and_nested_values():
    value = {"null": None, "text": "", "yes": False, "list": [0, {"n": 2**53 - 1}]}
    assert portable(value) == value
