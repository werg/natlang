import json
from pathlib import Path

from natlang.gen.codebases import ref_key
from natlang.invocation import RunOptions, SeedPolicy
from scripts.collect_scenario_teacher import collect
from scripts.audit_trajectory_admission import audit_scope_turns
from scripts.materialize_teacher_trajectory_ir import materialize
from scripts.migrate_teacher_scope_ir import migrate
from scripts.project_teacher_trajectory_ir import project
from scripts.teacher_probe_trajectory_ir import convert_probe
from scripts.teacher_trajectory_ir import convert
from natlang.scope_surface import ScopeEvalSurface
from natlang.decoder import ChatTurn


def test_teacher_turns_preserve_reasoning_order_and_rejected_actions():
    args = {"purpose": "sample"}
    key = ref_key("page_content", args)
    audit = {"key": key, "function": "page_content", "args": args,
             "status": "quiesced", "detail": "validation failed", "value": None,
             "accepted": False, "admitted": False, "checks": [],
             "teacher_turns": [
                 {"function": "page_content", "messages_before": [{"role": "user", "content": "Write the page"}],
                  "response": {"choices": [{"message": {"content": None,
                               "reasoning_content": "The destination is required.",
                               "tool_calls": [{"id": "c1", "function": {"name": "call_function",
                                               "arguments": '{"function":"make_page"}'}}]}}]},
                  "calls": [["call", {"function": "make_page"}]], "text": "",
                  "reviews": [], "executions": [{"call_index": 0, "name": "call",
                      "args": {"function": "make_page"}, "kind": "rejected", "text": "bad type"}]},
             ]}
    row = convert(audit, audit_path=Path("audit.jsonl"), line_number=1,
                  links={key: ["program-1"]}, program_ir_hash="abc")
    assert row["version"] == "natlang.teacher_trajectory/1"
    assert row["task"]["source_program_ids"] == ["program-1"]
    turn = row["trajectory"][0]
    assert turn["assistant"]["reasoning"] == "The destination is required."
    assert turn["assistant"]["calls"][0] == {"tool": "call", "source_tool": "call",
                                                   "arguments": {"function": "make_page"}, "call_id": "c1"}
    assert turn["executions"][0]["kind"] == "rejected"
    assert row["capture_limits"] == []


def test_legacy_audit_keeps_done_and_marks_missing_reasoning():
    args = {"purpose": "sample"}
    key = ref_key("page_content", args)
    audit = {"key": key, "function": "page_content", "args": args,
             "status": "done", "detail": "", "value": "page", "accepted": True,
             "log": [{"action": "done {}", "kind": "completed"}],
             "transcript": [{"role": "assistant", "content": "", "tool_calls": [
                 {"id": "d1", "function": {"name": "done", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "d1", "content": "completed"}]}
    row = convert(audit, audit_path=Path("old.jsonl"), line_number=2,
                  links={key: ["program-1"]}, program_ir_hash="abc")
    assert row["trajectory"][0]["assistant"]["calls"][0]["tool"] == "end_turn"
    assert row["trajectory"][0]["executions"][0]["text"] == "completed"
    assert row["legacy_action_log"] == audit["log"]
    assert row["legacy_action_sequence"][0]["call"]["tool"] == "end_turn"
    assert "reasoning_not_recorded" in row["capture_limits"]
    view = project(row, tool_map={"end_turn": None}, accepted_only=True,
                   empty_success_reply=True)
    assert view["trajectory"][0]["assistant"]["calls"] == []
    assert view["trajectory"][0]["assistant"]["content"] == ""
    assert view["trajectory"][0]["terminal_migrated"] is True
    assert row["trajectory"][0]["assistant"]["calls"][0]["tool"] == "end_turn"


def test_legacy_harness_opening_is_context_not_teacher_choice():
    args = {"purpose": "sample"}
    key = ref_key("page_content", args)
    audit = {"key": key, "function": "page_content", "args": args,
             "transcript": [{"role": "assistant", "content": "", "tool_calls": [
                 {"id": "call_0", "function": {"name": "read", "arguments": '{"path":"."}'}}]},
                 {"role": "tool", "tool_call_id": "call_0", "content": "workspace"},
                 {"role": "assistant", "content": "Finished."}]}
    row = convert(audit, audit_path=Path("old.jsonl"), line_number=1,
                  links={key: ["program-1"]}, program_ir_hash="abc")
    assert len(row["harness_opening_context"]) == 2
    assert len(row["trajectory"]) == 1
    assert row["trajectory"][0]["assistant"]["content"] == "Finished."


def test_unlinked_audit_can_be_preserved_with_explicit_flag():
    args = {"purpose": "unseen"}
    key = ref_key("page_content", args)
    audit = {"key": key, "function": "page_content", "args": args, "transcript": []}
    row = convert(audit, audit_path=Path("old.jsonl"), line_number=1,
                  links={}, program_ir_hash="abc", allow_unlinked=True)
    assert row["task"]["source_program_ids"] == []
    assert "unlinked_program" in row["capture_limits"]


def test_projection_restores_missing_legacy_final_turn_without_changing_source():
    row = {"version": "natlang.teacher_trajectory/1", "task": {"function": "say"},
           "outcome": {"accepted": True, "status": "done"},
           "trajectory": [{"assistant": {"content": "", "reasoning": None,
                                          "calls": [{"tool": "write", "arguments": {"value": "ok"}}]},
                           "reviews": [], "executions": []}]}
    view = project(row, tool_map={}, empty_success_reply=True)
    assert len(view["trajectory"]) == 2
    assert view["trajectory"][1]["synthesized_end_turn"] is True
    assert view["trajectory"][1]["assistant"]["calls"] == []
    assert len(row["trajectory"]) == 1


def test_accepted_teacher_choices_replay_into_structured_turns():
    row = {"version": "natlang.teacher_trajectory/1", "id": "teacher-leaf:test",
           "task": {"kind": "generative_leaf", "leaf_program": {"$lambda": {"type": "Lambda<{}, Num>",
                     "instructions": "Return 7.", "args": {}}},
                    "source_program_ids": ["program-1"], "reference_key": "test"},
           "provenance": {"model": "teacher"},
           "outcome": {"accepted": True, "status": "done", "value": 7},
           "trajectory": [
               {"assistant": {"content": "", "reasoning": "write the requested number",
                              "calls": [{"tool": "write", "arguments": {"path": "return",
                                         "type": "Num", "value": 7}}]},
                "executions": [{"name": "write", "kind": "ok"}]},
               {"assistant": {"content": "", "reasoning": None, "calls": []},
                "executions": []}]}
    samples = materialize(row, system_prompt="Follow the instructions.")
    assert [sample["skill"] for sample in samples] == ["write", "reply"]
    assert samples[0]["target"]["tool_calls"][0]["function"]["name"] == "write"
    assert samples[1]["target"]["content"] == ""
    assert samples[0]["teacher_reasoning"] == "write the requested number"
    assert samples[0]["trace_admission"]["admitted"] is True


def test_behavior_probe_choices_convert_without_leaf_reference_key():
    doc = {"model": "teacher", "system_prompt": "execute"}
    row = {"case": "fold_calls", "group": 0,
           "program": {"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Return 7."}},
           "status": "done", "pass": True, "value": 7,
           "teacher_turns": [{"function": "root", "messages_before": [],
                              "response": {"choices": [{"message": {"content": "Finished.",
                                                            "reasoning_content": "The value is ready."}}]},
                              "calls": [], "executions": [], "reviews": []}]}
    ir = convert_probe(doc, row, path=Path("probe.json"), index=1)
    assert ir["task"]["kind"] == "behavior_probe"
    assert ir["trajectory"][0]["assistant"]["reasoning"] == "The value is ready."


def test_scope_eval_trajectory_replays_and_preserves_model_choices():
    record = {"version": "natlang.program/1", "id": "fixture:scope-leaf",
              "kind": "lambda_source", "source": "fixture", "split": "train",
              "source_ids": ["fixture:scope-leaf"], "source_groups": ["fixture:scope-leaf"],
              "license": "project-generated",
              "semantics": {"operation": "leaf", "inputs": {"a": 7}, "expected": 7,
                            "root": {"$lambda": {"type": "Lambda<{ a: Num }, Num>",
                                "function": "root", "instructions": "Return the supplied number."}}}}

    class Teacher:
        def __init__(self):
            self.seen = []

        def chat(self, _messages, _tools, **_kwargs):
            calls = {
                1: [("eval", {"code": "const answer: Num = args.a; answer"})],
                2: [("mark_lines", {"start": 1})],
                3: [("return_value", {"variable": "answer"})],
                4: [],
            }[len(getattr(self, "seen", [])) + 1]
            self.seen.append(calls)
            return ChatTurn(calls=calls, text="", completion_tokens=1,
                            raw_response={"choices": [{"message": {"content": ""}}]})

    row, _ = collect(record, Teacher(), model_id="scope-teacher",
                     options=RunOptions(seed=SeedPolicy("derived", 17)),
                     system_prompt="Execute a typed scope program.", surface=ScopeEvalSurface())
    assert row["provenance"]["tool_schema"] == "scope-eval-v1"
    assert row["outcome"]["accepted"] is True
    assert [call["tool"] for call in row["trajectory"][0]["assistant"]["calls"]] == ["eval"]
    assert row["trajectory"][0]["assistant"]["calls"][0]["arguments"]["code"].startswith("const answer")

    samples = materialize(row, system_prompt="Execute a typed scope program.")
    assert [sample["skill"] for sample in samples] == ["eval", "mark_lines", "return_value", "reply"]
    assert samples[0]["target"]["tool_calls"][0]["function"]["name"] == "eval"
    assert json.loads(samples[0]["target"]["tool_calls"][0]["function"]["arguments"])["code"].startswith("const answer")
    assert "eval" in {tool["function"]["name"] for tool in samples[0]["tools"]}
    assert samples[-1]["target"]["content"] == ""
    assert audit_scope_turns(samples)


def test_unambiguous_legacy_leaf_migrates_to_scope_eval_and_replays():
    row = {"version": "natlang.teacher_trajectory/1", "id": "teacher-leaf:legacy",
           "task": {"kind": "generative_leaf", "reference_key": "legacy",
                    "source_program_ids": ["program-1"],
                    "leaf_program": {"$lambda": {"type": "Lambda<{}, Num>",
                        "instructions": "Return seven.\n\nKeep the numeric type.", "args": {}}}},
           "provenance": {"model": "teacher", "tool_schema": "tools-v4"},
           "outcome": {"accepted": True, "status": "done", "value": 7},
           "trajectory": [{"index": 0, "function": "root",
               "assistant": {"content": "", "reasoning": "Seven is requested.",
                             "calls": [{"tool": "write", "source_tool": "write",
                                        "arguments": {"path": "return", "type": "Num", "value": 7}}]},
               "reviews": [], "executions": [{"call_index": 0, "kind": "done"}]}]}
    migrated, reason = migrate(row)
    assert reason == "migrated"
    assert migrated["provenance"]["tool_schema"] == "scope-eval-v1"
    assert migrated["source_trajectory"] == row["trajectory"]
    assert [t["assistant"]["calls"][0]["tool"] for t in migrated["trajectory"][:-1]] == [
        "write_value", "mark_lines", "mark_lines", "return_value"]
    samples = materialize(migrated, system_prompt="Execute the scope program.")
    assert samples[0]["teacher_reasoning"] == "Seven is requested."
    assert samples[-1]["skill"] == "reply"


def test_legacy_scope_migration_rejects_corrections_and_keeps_source_untouched():
    row = {"version": "natlang.teacher_trajectory/1", "id": "teacher-leaf:legacy",
           "task": {"kind": "generative_leaf", "reference_key": "legacy",
                    "source_program_ids": ["program-1"],
                    "leaf_program": {"$lambda": {"type": "Lambda<{}, Text>",
                        "instructions": "Return text.", "args": {}}}},
           "provenance": {"tool_schema": "tools-v3"},
           "outcome": {"accepted": True, "status": "done", "value": "right"},
           "trajectory": [{"assistant": {"calls": [
               {"tool": "write", "arguments": {"path": "return", "type": "Text", "value": "wrong"}},
               {"tool": "write", "arguments": {"path": "return", "type": "Text", "value": "right"}}]},
               "executions": []}]}
    before = json.loads(json.dumps(row))
    migrated, reason = migrate(row)
    assert migrated is None and reason == "not-one-write"
    assert row == before
