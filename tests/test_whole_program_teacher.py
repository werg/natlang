from natlang.decoder import ChatTurn
from natlang.invocation import RunOptions, SeedPolicy
from scripts.collect_scenario_teacher import collect
from scripts.materialize_teacher_trajectory_ir import materialize
from scripts.build_scenario_ir import scenario_record
from scripts.generate_failures import matched_cases


class Teacher:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools, *, temperature, seed, max_tokens):
        turn = self.turns.pop(0)
        return ChatTurn(calls=turn[0], text=turn[1], completion_tokens=1)


def program_ir():
    return {"version": "natlang.program/1", "id": "fixture:teacher-map", "kind": "lambda_source",
            "source": "fixture", "split": "train", "source_ids": ["fixture-map"],
            "source_groups": ["fixture-map"], "license": "project-generated",
            "semantics": {"operation": "map", "root": {"$lambda": {
                "type": "Lambda<{ tickets: Text[], rubric: Text }, Text[]>",
                "function": "root", "instructions": "Classify each ticket.",
                "codebase": {"classify": {"args": {"item": "Text", "rubric": "Text"},
                                          "returns": "Text", "instructions": "Classify the ticket."}}}},
                "inputs": {"tickets": ["invoice", "advertisement"], "rubric": "billing or spam"},
                "expected": ["billing", "spam"], "leaf_oracles": [
                    {"input": {"item": "invoice", "rubric": "billing or spam"}, "output": "billing"},
                    {"input": {"item": "advertisement", "rubric": "billing or spam"}, "output": "spam"}]}}


def test_nested_teacher_run_round_trips_through_shared_ir():
    decoder = Teacher([
        ([("call", {"function": "classify", "to": "return", "over": "args/tickets",
                     "inputs": {"rubric": "args/rubric"}})], ""),
        ([("write", {"path": "return", "type": "Text", "value": "billing"})], ""),
        ([], "billing ticket"),
        ([("write", {"path": "return", "type": "Text", "value": "spam"})], ""),
        ([], "advertising"),
        ([], "classified"),
    ])
    row, recorder = collect(program_ir(), decoder, model_id="recorded-fixture",
                            options=RunOptions(seed=SeedPolicy("derived", 43)),
                            system_prompt="Follow the instructions.")
    assert row["outcome"]["accepted"] is True
    assert row["outcome"]["value"] == ["billing", "spam"]
    assert len(row["trajectory"]) == 6
    assert {turn["call_id"] for turn in row["trajectory"]} >= {"$root@1", "return/0@1", "return/1@1"}
    assert [event["phase"] for event in recorder.events if event["kind"] == "proposal"].count("generated") == 6
    samples = materialize(row, system_prompt="Follow the instructions.")
    assert [sample["skill"] for sample in samples] == ["call", "write", "reply", "write", "reply", "reply"]
    assert all(sample["trace_admission"]["admitted"] for sample in samples)
    assert {sample["program_id"] for sample in samples} == {"fixture-map"}


def test_correct_blocker_is_admitted_as_whole_program_gold():
    case = next(c for c in matched_cases(81, 0) if c["name"] == "evidence_missing")
    record = scenario_record(case, "failures", 81, 0, ["fixture"])
    required = record["semantics"]["contract"]["required_actions"]
    assert required[-1]["tool"] == "report_blocker"
    decoder = Teacher([([(required[-1]["tool"], required[-1]["arguments"])], "")])
    row, _ = collect(record, decoder, model_id="recorded-fixture",
                     system_prompt="Follow the instructions.")
    assert row["outcome"]["status"] == "quiesced" and row["outcome"]["accepted"]
    samples = materialize(row, system_prompt="Follow the instructions.")
    assert len(samples) == 1 and samples[0]["skill"] == "report_blocker"
