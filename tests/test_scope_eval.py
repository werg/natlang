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
    assert surface.apply(active, "read_value", {"expression": "flags", "start": 1, "end": 3}).value == [False, True]


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


def test_scope_eval_lowers_ordinary_accumulation_and_bounded_repeat():
    root = load_program({"$lambda": {"type": "Lambda<{ values: Num[] }, Num>",
        "instructions": "Accumulate and advance.", "args": {"values": [2, 3]},
        "codebase": {
            "add": {"args": {"acc": "Num", "item": "Num"}, "returns": "Num",
                    "code": "return args.acc + args.item;"},
            "step": {"args": {"state": "Num"}, "returns": "Num", "code": "return args.state + 1;"},
            "finished": {"args": {"state": "Num"}, "returns": "Bool", "code": "return args.state >= 3;"},
        }}})
    active = Session(Runtime(lambda lam: None), root, TypeEnv())
    folded = active.apply("eval", {"code":
        "let total: Num = 1; for (const item of values) { total = await add(total, item); } total"})
    assert folded.kind == "done" and folded.value == 6
    repeated = active.apply("eval", {"code":
        "let current: Num = 0; for (let attempt = 0; attempt < 8; attempt++) { "
        "if (await finished(current)) break; current = await step(current); } current"})
    assert repeated.kind == "done" and repeated.value == 3


def test_scope_surface_is_stable_and_completion_ignores_blank_lines():
    root, active = session()
    tools = ScopeEvalSurface().tools(active)
    assert [entry["function"]["name"] for entry in tools] == [
        "eval", "read_value", "write_value", "return_value", "list_files", "search_files",
        "read_file", "write_file", "edit_file", "diff_files", "mark_lines",
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


def test_codebase_existing_source_is_live_editable_but_file_set_is_fixed():
    root = load_program({"$lambda": {"type": "Lambda<{}, Text>", "instructions": "Call label.",
        "codebase": {"label": {"args": {}, "returns": "Text", "code": 'return "old";'}}}})
    active = Session(Runtime(lambda lam: None), root, TypeEnv())
    surface = ScopeEvalSurface()
    source = surface.apply(active, "read_file", {"path": "codebase/label.ts"}).value
    assert 'return "old";' in source
    assert surface.apply(active, "edit_file", {"path": "codebase/label.ts", "find": 'return "old";',
                                                 "replace_with": 'return "new";'}).kind == "ok"
    called = surface.apply(active, "eval", {"code": "const answer = await label(); answer"})
    assert called.kind == "ok" and called.value == "new"
    rejected = surface.apply(active, "write_file", {"path": "codebase/extra.nl", "content": source})
    assert rejected.kind == "rejected"


def test_codebase_folder_exposes_and_live_edits_nested_lexical_sources():
    root = load_program({"$lambda": {"type": "Lambda<{}, Text>", "instructions": "Inspect helpers.",
        "codebase": {"outer": {"args": {}, "returns": "Text", "instructions": "Call inner.",
            "codebase": {"inner": {"args": {}, "returns": "Text", "code": 'return "old";'}}}}}})
    active = Session(Runtime(lambda lam: None), root, TypeEnv())
    listed = active.apply("list_files", {}).value
    assert [item["path"] for item in listed] == ["codebase/outer.nl", "codebase/outer/inner.ts"]
    assert 'import { inner } from "./outer/inner.ts";' in active.apply(
        "read_file", {"path": "codebase/outer.nl"}).value
    edited = active.apply("edit_file", {"path": "codebase/outer/inner.ts", "find": 'return "old";',
                                         "replace_with": 'return "new";'})
    assert edited.kind == "ok"
    assert root.codebase["outer"].codebase["inner"].body == 'return "new";\n'
    source = active.apply("read_file", {"path": "codebase/outer.nl"}).value
    relinked = source.replace('import { inner }', 'import { inner as renamed }')
    assert active.apply("write_file", {"path": "codebase/outer.nl", "content": relinked}).kind == "ok"
    assert list(root.codebase["outer"].codebase) == ["renamed"]
    assert active.apply("edit_file", {"path": "codebase/outer/inner.ts", "find": 'return "new";',
                                      "replace_with": 'return "newer";'}).kind == "ok"
    assert root.codebase["outer"].codebase["renamed"].body == 'return "newer";\n'
    invalid = relinked.replace('./outer/inner.ts', './missing.ts')
    assert active.apply("write_file", {"path": "codebase/outer.nl", "content": invalid}).kind == "rejected"
    assert list(root.codebase["outer"].codebase) == ["renamed"]


def test_injected_fs_and_file_tools_share_the_live_codebase_overlay():
    root = load_program({"$lambda": {"type": "Lambda<{}, Text>", "instructions": "Inspect label.",
        "codebase": {"label": {"args": {}, "returns": "Text", "code": 'return "old";'}}}})
    active = Session(Runtime(lambda lam: None), root, TypeEnv())
    read = active.apply("eval", {"code": 'const source: Text = await fs.readText("codebase/label.ts"); source'})
    assert read.kind == "ok" and 'return "old";' in read.value
    edited = active.apply("eval", {"code":
        'const receipt = await fs.editText("codebase/label.ts", { find: "old", replaceWith: "new" }); receipt'})
    assert edited.kind == "ok"
    assert "new" in active.apply("read_file", {"path": "codebase/label.ts"}).value
