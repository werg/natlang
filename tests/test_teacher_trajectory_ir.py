from pathlib import Path

from natlang.gen.codebases import ref_key
from scripts.materialize_teacher_trajectory_ir import materialize
from scripts.project_teacher_trajectory_ir import project
from scripts.teacher_trajectory_ir import convert


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


def test_accepted_teacher_choices_replay_into_structured_turns():
    row = {"version": "natlang.teacher_trajectory/1", "id": "teacher-leaf:test",
           "task": {"leaf_program": {"$lambda": {"type": "Lambda<{}, Num>",
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
