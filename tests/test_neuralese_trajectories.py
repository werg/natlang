import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from neuralese_data import agents, bgkit, finalize, schnitzel_turns, trajectory  # noqa: E402
from neuralese_data.common import Sink  # noqa: E402
from neuralese_data.records import validate  # noqa: E402
from test_neuralese_port_records import _rec  # noqa: E402


def _traj(n_steps=6, background=None):
    msgs = [trajectory.message("system", "You are an agent."), trajectory.message("user", "Task: fix the bug.")]
    for i in range(n_steps):
        msgs.append(trajectory.message("assistant", f"step {i}", reasoning=f"think {i}"))
        msgs.append(trajectory.message("tool", f"observation {i}"))
    return trajectory.Trajectory(
        id="upstream:test:t1", family="trajectory_continuation_test", messages=msgs, head=2, noun="test session",
        lineage={"project": "upstream", "store": "test", "store_version": None, "row": 0, "upstream": "test",
                 "upstream_id": "t1", "upstream_revision": None, "teacher": "unknown", "converter": "test"},
        license=trajectory.licence("MIT"), split="train", split_groups=["task:t1"],
        outcome={"label": "teacher", "checked": None}, background=background or [])


def test_continuation_is_causal_and_valid():
    t = _traj()
    picks = trajectory.windows(t, 10, "seed")
    assert picks
    for recent_start, target in picks:
        rec = trajectory.record(t, recent_start, target)
        assert validate(rec) == []
        later = {m["content"] for m in t.messages[target:]}
        writer_text = json.dumps([rec["writer"], rec["sources"]])
        consumer_text = json.dumps(rec["consumer"])
        # Nothing at or after the target is an input; the writer never sees the recent turns.
        for content in later:
            assert content not in writer_text and content not in consumer_text
        for m in t.messages[recent_start:target]:
            assert m in rec["consumer"]["context"]
            assert m not in rec["sources"][0]["messages"]
        assert rec["target"]["value"] == t.messages[target]
        assert rec["writer"]["context"] == t.messages[:2]


def test_short_sessions_fall_back_and_background_windows():
    t = _traj(n_steps=2)
    picks = trajectory.windows(t, 10, "s")
    assert picks  # fewer recent turns rather than no window
    bg = [trajectory.source("tool_doc", "protocol: act one step at a time")]
    t = _traj(n_steps=1, background=bg)
    picks = trajectory.windows(t, 10, "s")
    assert (2, 2) in picks
    rec = trajectory.record(t, 2, 2)
    assert validate(rec) == [] and rec["writer"]["result_type"] == "Neuralese<Background>"


def test_front_trimming_keeps_whole_messages():
    t = _traj(n_steps=8)
    old = trajectory.MAX_SOURCE_CHARS
    try:
        trajectory.MAX_SOURCE_CHARS = 40
        recent_start, target = trajectory.windows(t, 1, "x")[-1]
        rec = trajectory.record(t, recent_start, target)
        meta = rec["sources"][0]["meta"]
        assert meta["dropped_front_messages"] > 0
        assert len(trajectory.text_of(rec["sources"][0]["messages"])) <= 40
    finally:
        trajectory.MAX_SOURCE_CHARS = old


def test_split_think():
    assert trajectory.split_think("<think>plan</think>\nanswer") == ("plan", "answer")
    assert trajectory.split_think("plan</think>answer") == ("plan", "answer")
    assert trajectory.split_think("no thinking") == (None, "no thinking")


def test_harmony_messages():
    raw = [
        {"role": "system", "content": [{"tools": {"browser": {"tools": [{"name": "search", "description": "Search.",
                                                                          "parameters": {"properties": {"query": {"type": "string"}}}}]}}}]},
        {"role": "developer", "content": [{"text": "You are a research agent."}]},
        {"role": "user", "content": [{"text": "Question: who?"}]},
        {"role": "assistant", "channel": "analysis", "content": [{"text": "I should search."}]},
        {"role": "assistant", "channel": "analysis", "recipient": "browser.search", "content": [{"text": "{\"query\": \"who\"}"}]},
        {"role": "tool", "name": "browser.search", "content": [{"text": "results"}]},
        {"role": "assistant", "channel": "analysis", "content": [{"text": "Found it."}]},
        {"role": "assistant", "channel": "final", "content": [{"text": "Exact Answer: X"}]},
    ]
    msgs, tools = agents.harmony_messages(raw)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[2]["reasoning"] == "I should search." and msgs[2]["tool_calls"][0]["function"]["name"] == "browser.search"
    assert msgs[4] == {"role": "assistant", "content": "Exact Answer: X", "reasoning": "Found it."}
    assert tools[0]["function"]["name"] == "browser.search"


def test_webshaper_harmony_text():
    conv = [{"role": "assistant", "content": "<|channel|>analysis<|message|>Think first.<|end|><|start|>assistant"
                                               "<|channel|>commentary to=browser.search code<|message|>{\"query\": \"q\"}"},
            {"role": "tool", "content": "[1] results"},
            {"role": "assistant", "content": "<|channel|>analysis<|message|>Done.<|end|><|start|>assistant<|channel|>final<|message|>Answer: Y"}]
    msgs = agents._webshaper_messages(conv, "who?")
    assert msgs[0] == {"role": "user", "content": "Question: who?"}
    assert msgs[1]["reasoning"] == "Think first." and msgs[1]["tool_calls"][0]["function"]["name"] == "browser.search"
    assert msgs[3]["content"] == "Answer: Y" and msgs[3]["reasoning"] == "Done."
    assert "<|" not in json.dumps(msgs)


def test_schnitzel_turns_episode():
    episode = {
        "episode_id": "alfworld-x1", "environment": "alfworld-train",
        "query": "Use the stored protocol, know-how and worked examples.\nTask: put a mug in the sink.",
        "supports": [{"record_id": "p", "kind": "protocol", "text": "alfworld protocol: one command per turn."}],
        "turns": [{"role": "assistant", "text": "Action: look"}, {"role": "environment", "text": "Observation: a mug."},
                  {"role": "assistant", "text": "Action: take mug"}, {"role": "environment", "text": "Observation: done."},
                  {"role": "assistant", "text": "Action: go to sink"}],
        "verify": {"type": "trajectory", "env": "alfworld", "game": "json_2.1.1/train/pick-Mug/trial_1/game.tw-pddl"},
        "provenance": {"dataset": "alfworld", "domain": "alfworld", "split": "train"},
    }
    t = schnitzel_turns.trajectory("tasks-alfworld-20260927", "train", 0, episode)
    assert t.messages[0] == {"role": "user", "content": "Task: put a mug in the sink."}
    assert [m["role"] for m in t.messages[1:]] == ["assistant", "user", "assistant", "user", "assistant"]
    assert t.split_groups == ["alfworld-game:json_2.1.1/train/pick-mug"]
    for recent_start, target in trajectory.windows(t, 5, t.id):
        assert validate(trajectory.record(t, recent_start, target)) == []


def test_bgkit_slot_qa_drops_sentinels():
    row = {"context": json.dumps(["[session on 2022/01/01]\nA: I like tea.", "[session on 2022/02/01]\nA: Now coffee."]),
           "prompt": json.dumps(["What do I drink?"] * 2),
           "instruction": json.dumps([{"role": "system", "content": "Answer from the memory."},
                                      {"role": "user", "content": "Memory:\n<|reserved_6|>\n<|reserved_6|>\n\nQuestion: What do I drink?"}]),
           "tool_name": "", "tool_args": "", "target": "Coffee", "split": "train",
           "meta": json.dumps({"source": "chronicles", "episode": 7, "qtype": "knowledge_update"})}
    rec = bgkit.convert_slot_qa("memory_qa_v2_user_assistant", 3, row)
    assert validate(rec) == []
    assert len(rec["sources"]) == 2 and "reserved" not in json.dumps(rec)
    assert rec["consumer"]["context"][-1] == {"role": "user", "content": "Question: What do I drink?"}
    assert rec["split_groups"] == ["memory-episode:chronicles:7"]


def test_streaming_finalize(tmp_path):
    inp = tmp_path / "in"
    sink = Sink(inp, "t")
    shared = "shared document " + " ".join(f"tok{j}" for j in range(60))
    recs = [_rec(1, "A", text=shared), _rec(2, "B", text=shared), _rec(3, "C", split="eval"), _rec(4, "C"), _rec(5, "E")]
    for r in recs:
        sink.accept(r)
    sink.close()
    protected = {"question_hashes": {}, "ids": {"5": "bench"}}
    report = finalize.build([inp], tmp_path / "out", protected, log=lambda *_: None)
    assert report["records_in"] == 5 and report["dropped_exact_duplicates"] == 1
    assert report["closure_violations"] == 0 and not report["invalid"]
    out = [json.loads(line) for line in (tmp_path / "out" / "qa_extractive.port-records.jsonl").read_text().splitlines()]
    splits = {r["id"]: r["split"] for r in out}
    assert splits["bgkit:qa_squad:4"] == "test"   # closed with its eval group mate
    assert splits["bgkit:qa_squad:5"] == "test"   # protected id
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["records_out"] == 4
