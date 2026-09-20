"""A long invocation can reopen from state without carrying its old chat transcript."""
from natlang.decoder import ChatTurn
from natlang.gen.policy import ReferenceAgent
from natlang.gen.programs import Plan
from natlang.runtime import Runtime, Session
from natlang.surface import ToolSurface
from natlang.tool_agent import TOOLS_PROMPT, ToolAgent
from natlang.types import TypeEnv
from natlang.terminal import reply_only_sample
from natlang.values import dump, load_program
from scripts.materialize_teacher_trajectory_ir import ReplayDecoder
from scripts.materialize_teacher_trajectory_ir import materialize
from scripts.collect_scenario_teacher import collect
from scripts.teacher_trajectory_ir import new_turns


class Decoder:
    def __init__(self):
        self.requests = []

    def chat(self, messages, tools, **_kwargs):
        self.requests.append((messages, tools))
        n = len(self.requests)
        if n == 3:
            assert tools == []
            return ChatTurn(text="The final value still needs writing.", completion_tokens=5,
                            raw_response={"choices": [{"message": {"content": "The final value still needs writing."}}]})
        if n == 4:
            opening = messages[3]["content"]
            assert "Earlier working note" in opening
            assert "let/a (Num): 1" in opening and "let/b (Num): 2" in opening
            assert len(messages) == 4  # fresh system, request and opening read; no old tool exchanges
        actions = {
            1: [("write", {"path": "let/a", "type": "Num", "value": 1})],
            2: [("write", {"path": "let/b", "type": "Num", "value": 2})],
            4: [("write", {"path": "return", "type": "Num", "value": 3})],
            5: [],
        }
        return ChatTurn(calls=actions[n], completion_tokens=5,
                        raw_response={"choices": [{"message": {"content": "", "tool_calls": []}}]})


def test_checkpoint_reopens_without_old_messages_and_persists_note():
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>",
                                     "instructions": "Compute one plus two and write it to return."}})
    decoder, captured = Decoder(), []
    outcome, value = Runtime(lambda lam: ToolAgent(decoder, segment_turns=2,
                                                  teacher_turns=captured)).run_root(root)
    assert (outcome.kind, value) == ("done", 3)
    assert [turn.get("phase") for turn in captured] == [None, None, "checkpoint", None, None]
    assert load_program(dump(root)).continuation_note == "The final value still needs writing."
    replay = ReplayDecoder(new_turns({"teacher_turns": captured}))
    second = load_program({"$lambda": {"type": "Lambda<{}, Num>",
                                        "instructions": "Compute one plus two and write it to return."}})
    replay_outcome, replay_value = Runtime(lambda lam: ToolAgent(replay, segment_turns=2)).run_root(second)
    assert (replay_outcome.kind, replay_value) == ("done", 3)
    assert [sample["skill"] for sample in replay.samples] == ["write", "write", "checkpoint", "write", "reply"]
    assert len(replay.samples[3]["messages"]) == 4
    assert reply_only_sample(replay.samples[2])["target"]["content"] == "The final value still needs writing."


def test_message_count_triggers_teacher_note_and_ir_records_its_author():
    class Scripted:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools, **_kwargs):
            self.n += 1
            if self.n == 1:
                return ChatTurn(calls=[("write", {"path": "let/a", "type": "Num", "value": 1})],
                                raw_response={"choices": [{"message": {"content": ""}}]})
            if self.n == 2:
                return ChatTurn(calls=[("write", {"path": "let/b", "type": "Num", "value": 2})],
                                raw_response={"choices": [{"message": {"content": ""}}]})
            if self.n == 3:
                assert tools == []
                return ChatTurn(text="Write the final value.",
                                raw_response={"choices": [{"message": {"content": "Write the final value."}}]})
            if self.n == 4:
                assert len(messages) == 4
                return ChatTurn(calls=[("write", {"path": "return", "type": "Num", "value": 3})],
                                raw_response={"choices": [{"message": {"content": ""}}]})
            return ChatTurn(text="", raw_response={"choices": [{"message": {"content": ""}}]})

    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Return one."}})
    captured = []
    outcome, value = Runtime(lambda lam: ToolAgent(Scripted(), teacher_turns=captured,
                                                   segment_turns=None, segment_messages=6)).run_root(root)
    assert (outcome.kind, value) == ("done", 3)
    turns = new_turns({"teacher_turns": captured})
    assert turns[2]["note"] == {"text": "Write the final value.", "author": "teacher"}
    assert turns[2]["segment_messages"] == 6
    assert turns[0]["note"] is None


def test_rejection_stays_in_context_until_a_successful_action():
    class RetryDecoder:
        def __init__(self):
            self.requests = []

        def chat(self, messages, tools, **_kwargs):
            self.requests.append((messages, tools))
            n = len(self.requests)
            if n == 1:
                return ChatTurn(calls=[("write", {"path": "return", "type": "Bool", "value": "wrong"})])
            if n == 2:
                assert "rejected" in messages[-1]["content"]
                return ChatTurn(calls=[("write", {"path": "let/answer", "type": "Bool", "value": True})])
            if n == 3:
                assert tools == []
                return ChatTurn(text="Copy the local to return.")
            if n == 4:
                assert len(messages) == 4
                return ChatTurn(calls=[("write", {"path": "return", "type": "Bool", "source": "let/answer"})])
            assert n == 5
            return ChatTurn(text="")

    root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
    decoder = RetryDecoder()
    outcome, value = Runtime(lambda lam: ToolAgent(decoder, segment_turns=1,
                                                  validation_feedback="local")).run_root(root)
    assert (outcome.kind, value) == ("done", True)


def test_synthetic_reference_actions_use_saved_state_after_boundary():
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>",
                                     "instructions": "Compute 1 + 2 + 3 and return the result."}})
    steps = [("write", {"path": f"let/{name}", "type": "Num", "value": value})
             for name, value in (("a", 1), ("b", 2), ("c", 3))]
    steps.append(("write", {"path": "return", "type": "Num", "value": 6}))
    samples = []
    outcome, value = Runtime(lambda lam: ReferenceAgent(Plan("calls", steps=steps), samples,
                                                        segment_turns=2)).run_root(root)
    assert (outcome.kind, value) == ("done", 6)
    assert len(samples[2]["messages"]) == 4
    assert "let/a (Num): 1" in samples[2]["messages"][3]["content"]
    assert "let/b (Num): 2" in samples[2]["messages"][3]["content"]


def test_continuation_shows_recent_effects_and_can_read_full_journal():
    root = load_program({"$lambda": {"type": "Lambda<{}, Bool>", "instructions": "Return true."}})
    root.journal = [{"seq": n, "capability": "out.emit", "status": "done"} for n in range(1, 7)]
    session = Session(Runtime(None), root, TypeEnv())
    surface = ToolSurface()
    opening = surface.opening_read(session)[2]
    assert "Effects already attempted: 6" in opening
    assert "last 4 shown" in opening and '"seq": 1' not in opening
    assert "args@effects" in str(surface.tools(session))
    full = session.apply("read", {"path": "args@effects"})
    assert full.kind == "ok" and "seq: 1" in full.text and "seq: 6" in full.text


def test_transient_code_result_is_kept_until_written():
    root = load_program({"$lambda": {"type": "Lambda<{}, Num>", "instructions": "Return 1 + 2."}})
    samples = []
    outcome, value = Runtime(lambda lam: ReferenceAgent(Plan("crisp", code="1 + 2"), samples,
                                                        segment_turns=1)).run_root(root)
    assert (outcome.kind, value) == ("done", 3)
    assert len(samples[1]["messages"]) > 2
    assert "3" in samples[1]["messages"][-1]["content"]


def test_whole_program_capture_replays_checkpoint_as_training_turn():
    record = {"version": "natlang.program/1", "id": "checkpoint:record", "kind": "lambda_source",
              "source": "test", "split": "train", "source_ids": ["checkpoint:record"],
              "source_groups": ["checkpoint:record"], "license": "project-generated",
              "semantics": {"operation": "leaf", "inputs": {}, "expected": {"a": 1, "b": 2},
                            "root": {"$lambda": {"type": "Lambda<{}, { a: Num, b: Num }>",
                                                 "instructions": "Write 1 to a, then 2 to b."}}}}

    class Scripted:
        def __init__(self):
            self.n = 0

        def chat(self, _messages, _tools, **_kwargs):
            self.n += 1
            calls = {1: [("write", {"path": "return/a", "type": "Num", "value": 1})],
                     3: [("write", {"path": "return/b", "type": "Num", "value": 2})]}.get(self.n, [])
            content = "Fill b next." if self.n == 2 else ""
            return ChatTurn(calls=calls, text=content, completion_tokens=1,
                            raw_response={"choices": [{"message": {"content": content}}]})

    row, _ = collect(record, Scripted(), model_id="scripted", segment_turns=1)
    assert row["outcome"]["accepted"]
    assert [turn["phase"] for turn in row["trajectory"]] == ["action", "checkpoint", "action", "action"]
    assert row["trajectory"][1]["segment_turns"] == 1
    assert row["trajectory"][1]["note"] == {"text": "Fill b next.", "author": "teacher"}
    samples = materialize(row, system_prompt=TOOLS_PROMPT)
    assert [sample["skill"] for sample in samples] == ["write", "checkpoint", "write", "reply"]
    assert len(samples[2]["messages"]) == 4
