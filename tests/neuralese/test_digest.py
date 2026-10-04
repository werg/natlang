import json
from pathlib import Path

from natlang_neuralese.digest import PREFIX, digest_note, digest_site

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "digest-site.json").read_text())


def test_write_site_matches_the_runtime():
    site = FIXTURE["site"]
    system = FIXTURE["messages"][0]["content"]
    assert digest_site(system, site["name"], site["type"], site["value"], site["instructions"]) == FIXTURE["messages"]
    assert PREFIX == FIXTURE["prefix"] and digest_note("state") == FIXTURE["note"]


def test_instructions_match_the_runtime_piece_and_plans_chunk_long_values():
    from natlang_neuralese.digest import INSTRUCTIONS, combine_site, plan

    assert INSTRUCTIONS == FIXTURE["messages"][0]["content"]

    class Words:  # a tokenizer of whitespace-separated words
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    value = " ".join(f"w{i}" for i in range(25))
    assert plan(Words(), value, 100).chunks == [value]
    chunks = plan(Words(), value, 10).chunks
    assert len(chunks) == 3 and " ".join(chunks) == value
    site = combine_site("sys", "state", "unknown", "Decide.", ["nz1_" + "a" * 52, "nz1_" + "b" * 52])
    assert [p["id"] for p in site[1]["content"] if p["type"] == "neuralese"] == ["nz1_" + "a" * 52, "nz1_" + "b" * 52]
