"""Structured tool calls cover the useful checks from the retired text-action fixtures."""

from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import load_program


def session(type_text, *, instructions="Fill in the record.", args=None, effects=None):
    root = load_program({"$lambda": {"type": type_text, "instructions": instructions,
                                     **({"args": args} if args is not None else {}),
                                     **({"effects": effects} if effects is not None else {})}})
    return Session(Runtime(None), root, TypeEnv())


def test_partial_draft_requires_all_fields_before_finish():
    s = session("Lambda<{}, { id: Num, label: Text, score: Num }>")
    assert s.apply("write", {"path": "return", "value": {"id": 7}}).kind == "ok"
    assert not s.finish()
    assert s.apply("read", {"path": "return@problems"}).kind == "ok"
    assert s.apply("write", {"path": "return/label", "value": "fine"}).kind == "ok"
    assert not s.finish()
    assert s.apply("write", {"path": "return/score", "value": 0.9}).kind == "ok"
    assert s.finish()
    assert s.lam.ret == {"id": 7, "label": "fine", "score": 0.9}


def test_typed_writes_reject_contradictions_and_preserve_text_scalars():
    s = session('Lambda<{}, { code: Text, flag: Text, n: Num, level: "low" | "high" }>')
    for path, value, code in [("return/n", "many", "type-mismatch"),
                              ("return/level", "medium", "type-mismatch"),
                              ("return/extra", "hello", "unknown-field")]:
        result = s.apply("write", {"path": path, "value": value})
        assert result.kind == "rejected" and code in result.codes
    assert s.apply("write", {"path": "return/code", "value": "0077"}).kind == "ok"
    assert s.apply("write", {"path": "return/flag", "value": "no"}).kind == "ok"
    assert s.lam.ret["code"] == "0077" and s.lam.ret["flag"] == "no"


def test_copy_keeps_types_ranges_and_read_only_inputs():
    s = session("Lambda<{ names: Text[], n: Num }, { first_two: Text[], total: Num, plan: Text }>",
                instructions="Line one.\nLine two.\nLine three.",
                args={"names": ["a", "b", "c"], "n": 5})
    assert s.apply("copy", {"from": "args/names[0..1]", "to": "return/first_two"}).kind == "ok"
    assert s.lam.ret["first_two"] == ["a", "b"]
    assert "type-does-not-fit-slot" in s.apply("copy", {"from": "args/names", "to": "return/total"}).codes
    assert s.apply("copy", {"from": "args/n", "to": "return/total"}).kind == "ok"
    assert s.apply("copy", {"from": "instructions[2..3]", "to": "return/plan"}).kind == "ok"
    assert s.lam.ret["plan"] == "Line two.\nLine three.\n"
    assert "not-writable" in s.apply("copy", {"from": "return/total", "to": "args/n"}).codes


def test_tool_budget_and_finished_session():
    s = session("Lambda<{}, Bool>", instructions="Decide.")
    assert s.apply("write", {"path": "return", "value": True}).kind == "ok"
    assert s.finish()
    assert s.apply("read", {"path": "return"}).kind == "error"


def test_scope_and_declared_effects_are_enforced():
    s = session("Lambda<{ x: Num }, Num>", args={"x": 1}, effects=["out.emit"])
    assert "out-of-scope" in s.apply("read", {"path": "../sibling"}).codes
    assert s.apply("run_code", {"code": "fx.out.emit({ x: args.x })"}).kind == "ok"
    assert len(s.lam.journal) == 1
    assert "effect-undeclared" in s.apply("run_code", {"code": 'fx.fs.write("/tmp/x", "1")'}).codes
    assert "eval-cannot-write" in s.apply("run_code", {"code": "self.return = 3"}).codes
    assert len(s.lam.journal) == 1
