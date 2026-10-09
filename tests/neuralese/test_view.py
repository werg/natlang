import json
from pathlib import Path

import pytest

from natlang_neuralese import view
from natlang_neuralese.view import (INSTRUCTIONS, TEMPLATE, TOOLS, combine_site, listing_instructions, plan,
                                    reject_digest_part, view_note, view_site)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "view-site.json").read_text())


def test_write_site_matches_the_fixture_the_runtime_and_the_fork_are_pinned_to():
    site, listing = FIXTURE["site"], FIXTURE["listing"]
    assert listing_instructions(listing["name"], listing["type"], listing["instructions"]) == site["instructions"]
    assert view_site(INSTRUCTIONS, site["value"], site["instructions"]) == FIXTURE["messages"]
    assert view_site(INSTRUCTIONS, site["value"]) == FIXTURE["faithful"]
    assert TOOLS == FIXTURE["tools"] and TEMPLATE == FIXTURE["template"] and view_note("state") == FIXTURE["note"]


def test_the_site_system_text_is_the_body_of_builtin_view():
    source = (ROOT / "ts-host" / "src" / "builtin" / "view.nl").read_text()
    body = source.split("\n---\n", 1)[1].strip("\n")
    assert INSTRUCTIONS == body
    assert "generic:\n  R: string | Neuralese<string>\nreturns: R" in source


def test_faithful_and_instructed_sites_differ_only_by_the_purpose():
    faithful = view_site("sys", "x = 1")[1]["content"]
    instructed = view_site("sys", "x = 1", "Find x.")[1]["content"]
    assert faithful == "value:\nx = 1" and instructed == "instructions:\nFind x.\n\nvalue:\nx = 1"


def test_plans_chunk_long_values_and_combine_part_views():
    class Words:  # a tokenizer of whitespace-separated words
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    value = " ".join(f"w{i}" for i in range(25))
    assert plan(Words(), value, 100).chunks == [value]
    chunks = plan(Words(), value, 10).chunks
    assert len(chunks) == 3 and " ".join(chunks) == value
    site = combine_site("sys", "Decide.", ["nz1_" + "a" * 52, "nz1_" + "b" * 52])
    assert [p["id"] for p in site[1]["content"] if p["type"] == "neuralese"] == ["nz1_" + "a" * 52, "nz1_" + "b" * 52]
    writes = []
    block, parts = view.write_view(lambda messages: writes.append(messages) or f"b{len(writes)}", "sys", value, None,
                                   Words(), 10)
    assert parts == 3 and block == "b4" and len(writes) == 4


def test_the_forced_reply_is_the_template_readout_cut():
    from natlang_neuralese.serve.chat import write_reply

    def apply(messages, generation):  # a chat template rendering tool calls as JSON
        out = "".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages)
        for message in messages:
            for call in message.get("tool_calls") or []:
                out += json.dumps({"name": call["function"]["name"], "arguments": call["function"]["arguments"]})
        return out + ("<assistant>" if generation else "")

    prefix = view.reply_prefix(apply)
    assert prefix == write_reply(apply, "return_result", {"status": "success"}, "value", "string")[0]
    assert prefix.endswith('"value": "') and '"status": "success"' in prefix


def test_digest_parts_are_rejected_with_their_conversion():
    with pytest.raises(ValueError, match="digest_to_view.py"):
        reject_digest_part({"type": "digest", "name": "digest:abc"})
    reject_digest_part({"type": "view", "name": "view:abc"})


def test_harness_bench_records_carry_view_parts_for_long_tool_outputs():
    from natlang_neuralese.harness_bench.openhands import Normalized
    from natlang_neuralese.harness_bench.records import native_messages

    long = "x" * 50
    transcript = Normalized(id="t", instance_id="i", repo="r", resolved=True, mapping="pi", goal="g", cwd="/", messages=[
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": [{"type": "text", "text": "Look."},
                                          {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}]},
        {"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": long}]}])
    messages, _, views = native_messages(transcript, "prompt:pi", view_chars=10, preview_chars=100)
    part = messages[-1]["content"][0]
    assert views == 1 and part["type"] == "view" and part["name"].startswith("view:") and part["source"] == long
    assert part["note"] == '  // view of the output; recall("c1") returns all of it'
    assert "bash" in part["instructions"]


def test_the_registered_conversion_rewrites_digest_parts_and_the_piece(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("digest_to_view", ROOT / "scripts" / "neuralese_data" / "digest_to_view.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source = tmp_path / "old"
    source.mkdir()
    record = {"id": "r", "provenance": {"digest_chars": 2000}, "digests_in_trajectory": 1,
              "neuralese_conversion": {"version": "natlang.harness-bench-conversion/1", "sites": {"digest": {"converted": 1}}},
              "messages": [{"role": "tool", "content": [{"type": "digest", "name": "digest:abc", "holder": 'recall("c1")',
                                                         "value_type": "string", "source": "s", "preview": "p",
                                                         "note": '  // digest of the output; recall("c1") returns all of it'}]}]}
    (source / "records.jsonl").write_text(json.dumps(record) + "\n")
    (source / "pieces.jsonl").write_text(json.dumps({"name": "prompt:digest", "kind": "system-prompt", "text": "old"}) + "\n")
    (source / "summary.json").write_text("{}\n")
    assert script.main(["--source", str(source), "--out", str(tmp_path / "new"), "--copy", "summary.json"]) == 0
    converted = json.loads((tmp_path / "new" / "records.jsonl").read_text())
    part = converted["messages"][0]["content"][0]
    assert part["type"] == "view" and part["name"] == "view:abc" and part["note"].startswith("  // view of the output")
    reject_digest_part(part)
    assert converted["neuralese_conversion"] == {"version": "natlang.harness-bench-conversion/2", "sites": {"view": {"converted": 1}}}
    assert converted["views_in_trajectory"] == 1 and converted["provenance"] == {"view_chars": 2000}
    assert converted["view_rename"]["from_version"] == "natlang.harness-bench-conversion/1"
    piece = json.loads((tmp_path / "new" / "pieces.jsonl").read_text())
    assert piece == {"name": "prompt:view", "kind": "system-prompt", "text": INSTRUCTIONS}
    receipt = json.loads((tmp_path / "new" / "conversion.json").read_text())
    assert receipt["counts"]["digest_parts"] == 1 and set(receipt["output_sha256"]) == {"records.jsonl", "pieces.jsonl", "summary.json"}
