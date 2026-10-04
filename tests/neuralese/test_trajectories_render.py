import json

from natlang_neuralese.train.trajectories import crisp_messages, handover_notes, reads, render, target_write

NOTE = "Line 2 is a fee; check prior invoices."
WRITE = {"$write": {"name": "handover:abc", "type": "Neuralese<HandoverNote>", "source": NOTE}}
RECORD = {"messages": [
    {"role": "system", "content": [{"type": "soft", "name": "prompt:interpreter"}]},
    {"role": "user", "content": [{"type": "soft", "name": "prompt:handover/open"}, {"type": "read", "name": "handover:abc"},
                                 {"type": "soft", "name": "prompt:handover/close"}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "compact_history", "arguments": json.dumps({"note": WRITE})}}]},
], "target": {"role": "assistant", "content": "true"}}
TEXTS = {"prompt:interpreter": "You run one call.", "prompt:handover/open": "Your note: ", "prompt:handover/close": "\n\nContinue."}


def test_crisp_rendering_restores_the_original_texts():
    crisp = crisp_messages(RECORD["messages"], TEXTS, handover_notes(RECORD))
    assert crisp[0]["content"] == "You run one call."
    assert crisp[1]["content"] == "Your note: " + NOTE + "\n\nContinue."
    assert json.loads(crisp[2]["tool_calls"][0]["function"]["arguments"]) == {"note": NOTE}
    assert reads(RECORD) == {"handover:abc"} and target_write(RECORD) is None


def test_written_notes_are_block_parts_in_reads_and_inside_the_quoted_argument():
    block = "nz1_" + "b" * 52
    out = render(RECORD["messages"], lambda name: {"type": "neuralese", "id": "nz1_" + "c" * 52}, handover_notes(RECORD),
                 {"handover:abc": block})
    assert {"type": "neuralese", "id": block} in out[1]["content"]
    parts = out[2]["tool_calls"][0]["function"]["arguments"]
    text = "".join(p["text"] if p["type"] == "text" else "BLOCK" for p in parts)
    assert json.loads(text) == {"note": "BLOCK"}
