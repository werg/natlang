from natlang.nodes import MISSING
from natlang.runtime import Runtime, Session
from natlang.scope_surface import ScopeEvalSurface
from natlang.types import TypeEnv
from natlang.values import load_program


def session():
    root = load_program({"$lambda": {
        "type": "Lambda<{ flags: Bool[] }, Num>",
        "instructions": "function total(flags) -> Num\n  Count the true flags.\n",
        "args": {"flags": [True, False, True]},
        "codebase": {"count_true": {"args": {"flags": "Bool[]"}, "returns": "Num",
                                      "code": "return args.flags.filter(Boolean).length;"},
                     "as_num": {"args": {"flag": "Bool"}, "returns": "Num",
                                "code": "return args.flag ? 1 : 0;"}},
        "function": "total",
    }})
    return root, Session(Runtime(lambda lam: None), root, TypeEnv())


def test_scope_eval_persists_pure_declarations_and_reads_selections():
    root, active = session()
    surface = ScopeEvalSurface()
    result = surface.apply(active, "eval", {"code": "const record = { count: flags.length, first: flags[0] };\nrecord"})
    assert result.kind == "ok" and result.value == {"count": 3, "first": True}
    assert root.let["record"] == result.value
    assert surface.apply(active, "read_value", {"expression": "record.count"}).value == 3
    assert surface.apply(active, "read_value", {"expression": "flags[1]"}).value is False


def test_scope_eval_calls_imports_positionally_and_stages_named_result():
    root, active = session()
    surface = ScopeEvalSurface()
    called = surface.apply(active, "eval", {"code": "const count = await count_true(flags);\ncount"})
    assert called.kind == "ok" and called.value == 2
    assert root.let["count"] == 2 and root.let_types["count"] is not None
    assert surface.apply(active, "return_value", {"variable": "count"}).kind == "ok"
    assert root.ret == 2


def test_scope_eval_lowers_promise_all_map_to_checked_child_calls():
    root, active = session()
    result = active.apply("eval", {"code":
        "const counts = await Promise.all(flags.map(flag => as_num(flag)));\ncounts"})
    assert result.kind == "ok" and result.value == [1, 0, 1]
    assert root.let["counts"] == [1, 0, 1]


def test_scope_eval_sequences_multiple_imported_calls_in_one_snippet():
    root, active = session()
    result = active.apply("eval", {"code":
        "const counts: Num[] = flags.map(flag => as_num(flag));\n"
        "const result: Num = await count_true(flags);\nresult"})
    assert result.kind == "ok" and result.value == 2
    assert root.let["counts"] == [1, 0, 1] and root.let["result"] == 2


def test_scope_surface_is_stable_and_completion_ignores_blank_lines():
    root, active = session()
    tools = ScopeEvalSurface().tools(active)
    assert [entry["function"]["name"] for entry in tools] == [
        "eval", "read_value", "write_value", "return_value", "mark_lines",
        "report_blocker", "report_error"]
    assert active.apply("write_value", {"name": "answer", "value": 2}).kind == "ok"
    assert active.apply("return_value", {"variable": "answer"}).kind == "ok"
    assert active.apply("mark_lines", {"start": 2}).kind == "ok"
    assert active.finish()
    assert root.ret == 2 and root.body == ""
    assert ScopeEvalSurface().tools(active) == []


def test_scope_eval_can_bind_and_return_an_explicit_null():
    root = load_program({"$lambda": {"type": "Lambda<{}, Null>",
                                     "instructions": "Return null."}})
    active = Session(Runtime(lambda lam: None), root, TypeEnv())
    result = active.apply("eval", {"code": "const result: Null = null; result"})
    assert result.kind == "ok" and "result" in root.let and root.let["result"] is None
    assert active.apply("return_value", {"variable": "result"}).kind == "ok"


def test_scope_writes_require_types_only_for_ambiguous_values():
    root, active = session()
    assert active.apply("write_value", {"name": "label", "value": "ok"}).kind == "ok"
    assert active.apply("write_value", {"name": "empty", "value": []}).kind == "rejected"
    assert active.apply("write_value", {"name": "empty", "value": [], "as_type": "Text[]"}).kind == "ok"
    assert root.ret is MISSING
