"""The recurrence of calling a function, retrieving its value and splicing it into the caller's trajectory, trained
across one run's chunks (train.trajectories, child results): a caller's loss reaches the child's write and, one level
further, the grandchild's."""

import json

import pytest

from natlang_neuralese.train.trajectories import main, write_site

GRAND = "The fee on line 2 matches clause 4.2 of the supply contract."
CHILD = "Line 2 is a contractual fee, not a disputed charge."


def opening(name, text):
    return {"role": "user", "content": [{"type": "text", "text": f"You are inside this call: {name}(): string\n\nInstructions:\n"},
                                        {"type": "soft", "name": f"instructions@{name}"},
                                        {"type": "text", "text": "\n\nIn eval you can use nothing."}]}


def result(name, value):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"r-{name}", "type": "function", "function": {
        "name": "return_result", "arguments": json.dumps({"status": "success", "value": {"$write": {
            "name": f"result:{name}", "type": "Neuralese<string>", "source": value}}})}}]}


def output(call_id, name, value):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "console.log(await nl`Look it up.`())"})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": [{"type": "text", "text": "console:\n"},
                                                                  {"type": "read", "name": f"result:{name}", "source": value}]}]


RECORDS = [
    # Grandchild and child are producers only (held out); the caller is the one training record.
    {"id": "grand", "split": "test", "messages": [{"role": "system", "content": "You run one call."}, opening("grand", "")],
     "target": result("grand", GRAND)},
    {"id": "child", "split": "test", "messages": [{"role": "system", "content": "You run one call."}, opening("child", ""),
                                                   *output("e1", "grand", GRAND)], "target": result("child", CHILD)},
    {"id": "caller", "split": "train", "messages": [{"role": "system", "content": "You run one call."}, opening("caller", ""),
                                                    *output("e2", "child", CHILD)],
     "target": {"role": "assistant", "content": "The charge stands: it is a contractual fee."}},
]
PIECES = [{"name": f"instructions@{n}", "kind": "function-body", "text": t} for n, t in
          [("grand", "Find the clause that covers line 2."), ("child", "Classify line 2."), ("caller", "Decide whether the charge stands.")]]


def test_write_site_is_the_template_readout_cut():
    assert write_site(RECORDS[1]) == ("return_result", {"status": "success"}, "value", "result:child")


@pytest.mark.parametrize("depth,grand_moves", [(1, False), (2, True)])
def test_a_callers_loss_trains_its_childs_write_and_nested_writes_to_the_depth(loaded, tmp_path, depth, grand_moves):
    records, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    out = tmp_path / "out"
    assert main(["--records", str(records), "--pieces", str(pieces), "--out", str(out), "--device", "cpu", "--steps", "1",
                 "--batch", "1", "--eval", "0", "--handover", "written", "--write-depth", str(depth), "--tokens-per-vector", "8",
                 "--distill", "0", "--lr", "1e-2"]) == 0
    moved = json.loads((out / "summary.json").read_text())["relative_change"]
    assert moved["instructions@caller"] > 0, "the caller's own soft instructions train"
    assert moved["instructions@child"] > 0, "the caller's loss reaches the child's write context"
    assert (moved["instructions@grand"] > 0) == grand_moves, "nested writes train to --write-depth levels"
