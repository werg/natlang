"""The recurrence of calling a function, retrieving its value and splicing it into the caller's trajectory, trained
across one run's chunks (train.trajectories, child results): a caller's loss reaches the child's write and, one level
further, the grandchild's."""

import json

import pytest
import torch

from natlang_neuralese.train.trajectories import main, write_site

GRAND = "The fee on line 2 matches clause 4.2 of the supply contract."
CHILD = "Line 2 is a contractual fee, not a disputed charge."
RETURN_TOOLS = [{"type": "function", "function": {"name": "return_result", "parameters": {
    "type": "object", "properties": {"status": {"type": "string"}, "value": {"type": "string"}},
    "required": ["status", "value"]}}}]


def opening(name, text, body_name=None):
    return {"role": "user", "content": [{"type": "text", "text": f"You are inside this call: {name}(): string\n\nInstructions:\n"},
                                        {"type": "soft", "name": f"instructions@{body_name or name}"},
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
    {"id": "grand", "split": "test", "source_groups": ["grand-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("grand", "")],
     "tools": RETURN_TOOLS, "target": result("grand", GRAND)},
    {"id": "child", "split": "test", "source_groups": ["grand-facts", "child-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("child", ""),
                  *output("e1", "grand", GRAND)], "tools": RETURN_TOOLS,
     "target": result("child", CHILD)},
    {"id": "caller", "split": "train", "source_groups": ["grand-facts", "child-facts", "caller-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("caller", ""),
                  *output("e2", "child", CHILD)],
     "target": {"role": "assistant", "content": "The charge stands: it is a contractual fee."}},
]
PIECES = [{"name": f"instructions@{n}", "kind": "function-body", "text": t} for n, t in
          [("grand", "Find the clause that covers line 2."), ("child", "Classify the line item."), ("caller", "Decide whether the charge stands.")]]


@pytest.fixture
def latent_sketch_heads(tmp_path, loaded):
    """Unqualified test fixture for raw/top-state gradient mechanics, not a foundation certificate."""
    from natlang_neuralese.model.heads import PortHeads

    backbone = loaded[2]
    heads = PortHeads(backbone, cutoff=2, max_length=64, profile="latent-sketch-v2").eval()
    path = tmp_path / "latent-sketch-heads.pt"
    torch.save({
        "port_config": {"cutoff": heads.cutoff, "max_length": heads.max_length,
                        **heads.port_config()},
        "heads": heads.state_dict(),
        "control_rows": backbone.control_rows.detach().cpu(),
    }, path)
    return path


def test_write_site_is_the_template_readout_cut():
    assert write_site(RECORDS[1]) == ("return_result", {"status": "success"}, "value", "result:child")


@pytest.mark.parametrize("depth,grand_moves", [(1, False), (2, True)])
def test_a_callers_loss_trains_its_childs_write_and_nested_writes_to_the_depth(loaded, latent_sketch_heads, tmp_path, depth, grand_moves, device):
    records, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    out = tmp_path / "out"
    assert main(["--records", str(records), "--pieces", str(pieces), "--out", str(out), "--heads", str(latent_sketch_heads), "--device", device, "--steps", "1",
                 "--batch", "1", "--eval", "0", "--handover", "written", "--write-depth", str(depth), "--tokens-per-vector", "1",
                 "--writer-supervision", "native-value", "--writer-length-policy", "native-value",
                 "--stop-supervision", "generated-length", "--writer-text-weight", "0",
                 "--distill", "0", "--lr", "1e-2"]) == 0
    moved = json.loads((out / "summary.json").read_text())["relative_change"]
    assert moved["instructions@caller"] > 0, "the caller's own soft instructions train"
    assert moved["instructions@child"] > 0, "the caller's loss reaches the child's write context"
    assert (moved["instructions@grand"] > 0) == grand_moves, "nested writes train to --write-depth levels"


def test_held_out_readers_compare_their_written_values_with_another_readers(loaded, latent_sketch_heads, tmp_path, device):
    other = "Line 7 is a late fee under clause 9."
    records = [
        {**RECORDS[1], "split": "train"},
        {"id": "child2", "split": "train", "source_groups": ["child2-facts"],
         "messages": [{"role": "system", "content": "You run one call."}, opening("child2", "", body_name="child")],
         "tools": RETURN_TOOLS, "target": result("child2", other)},
        {**RECORDS[2], "split": "test"},
        {"id": "caller2", "split": "test", "source_groups": ["child2-facts", "caller2-facts"],
         "messages": [{"role": "system", "content": "You run one call."},
                      opening("caller", ""), *output("e3", "child2", other)],
         "target": {"role": "assistant", "content": "The late fee stands under clause 9."}},
    ]
    path, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    out = tmp_path / "out"
    assert main(["--records", str(path), "--pieces", str(pieces), "--out", str(out), "--heads", str(latent_sketch_heads), "--device", device, "--steps", "1",
                 "--batch", "1", "--eval", "4", "--handover", "written", "--write-depth", "1", "--tokens-per-vector", "1",
                 "--writer-supervision", "native-value", "--writer-length-policy", "native-value",
                 "--stop-supervision", "generated-length", "--writer-text-weight", "0",
                 "--distill", "0"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    for label in ("written-init", "written-trained"):
        got = summary[label]
        assert got["n"] == 2 and got["written"] > 0 and got["shuffled"] > 0 and 0 <= got["written_better"] <= 1
        assert got["written"] != got["shuffled"]
