"""refine-judge: predicate harvest, exact labels, split, near-miss plumbing, miner, and parity with the TS judge prompt."""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from refine_data import exact, judge, miner, predicates  # noqa: E402
from refine_data.common import (DECISION_SCHEMA, canonical_json_str, judge_messages, judge_row, predicate_bucket,  # noqa: E402
                                predicate_id, split_of, value_sha256)


def test_harvest_reads_every_decomposition_table():
    found, skipped = predicates.harvest()
    apps = {item.app for item in found}
    assert apps == {"build", "games", "logs", "migration", "scheduling", "wiki", "workflow"}
    assert len(found) >= 110
    by_slot = {slot: item for item in found for slot in item.slots}
    assert by_slot["Outgoing.subject"].predicate == "one line of at most 60 characters, without a trailing period"
    assert by_slot["Outgoing.subject"].base == "string"
    # An escaped pipe inside a union type survives; the Untrusted rows are reported, not silently dropped.
    assert by_slot["LinkStatus.resolves_to"].base == "string | null"
    assert all(item["reason"] == "not-an-Is-type" for item in skipped) and len(skipped) >= 3
    # The runbook row carries an Untrusted type and an Is type; the Is type is harvested.
    assert any(item.predicate.startswith("a relative path under the files folder") for item in found)
    assert len({item.id for item in found}) == len(found), "predicate texts are unique across the tables"


def test_parse_table_handles_pipes_and_notes():
    text = "## Refinements\n\n| Slot | Proposed type |\n|---|---|\n| `A.x`, `A.y` | `Is<\"a\" \\| \"b\", \"one of a or b\">` (the verifier) |\n| `C` | `Untrusted<string>` |\n"
    found, skipped = predicates.parse_table(text, "demo", "applications/demo/DECOMPOSITION.md")
    assert [(c.slots, c.base, c.predicate, c.note) for c in found] == [(("A.x", "A.y"), '"a" | "b"', "one of a or b", "(the verifier)")]
    assert skipped[0]["slots"] == ["C"]


def test_exact_labels_are_the_checkers_verdicts_and_balanced():
    rows, report = exact.generate(per_predicate=8)
    assert report["dropped_intent_mismatch"] == 0
    assert report["predicates"] >= 90 and len(rows) >= 600
    for row in rows:
        assert row["schema"] == DECISION_SCHEMA and row["options"] == ["true", "false"] and row["kind"] == "noul"
        assert row["gold"] in ([1.0, 0.0], [0.0, 1.0]) and row["teachers"] == {}
        assert row["messages"][0]["role"] == "system" and "<<<value" in row["messages"][1]["content"]
        assert row["source"] == "exact" and row["label_source"].startswith("crisp:")
    positives = sum(row["gold"][0] for row in rows)
    assert 0.4 < positives / len(rows) < 0.6
    assert len({row["id"] for row in rows}) == len(rows)


def test_exact_generation_is_deterministic():
    first, _ = exact.generate(per_predicate=4)
    second, _ = exact.generate(per_predicate=4)
    assert first == second


@pytest.mark.parametrize("spec", exact.all_specs(), ids=lambda spec: spec.predicate[:50])
def test_each_spec_has_both_kinds_of_value(spec):
    import random
    rng = random.Random(1)
    assert any(spec.check(spec.pos(rng)) for _ in range(20)), "the satisfying generator produces satisfying values"
    assert any(not spec.check(spec.neg(rng)) for _ in range(20)), "the violating generator produces violating values"


def test_split_is_by_predicate_and_by_family():
    rows, _ = exact.generate(per_predicate=6)
    by_predicate = {}
    for row in rows:
        by_predicate.setdefault(row["predicate_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_predicate.values()), "a predicate never straddles splits"
    splits = {row["split"] for row in rows}
    assert splits == {"train", "heldout-predicate", "heldout-family"}
    for row in rows:
        assert row["role"] == ("train" if row["split"] == "train" else "heldout")
        if row["split"] == "heldout-family":
            assert row["family"].split("/")[-1] in exact.HELDOUT_FAMILIES
        if row["split"] == "heldout-predicate":
            assert predicate_bucket(row["predicate"]) < 2 and row["family"].split("/")[-1] not in exact.HELDOUT_FAMILIES
    # The same rule serves the other sources: a predicate text has one split whichever source asks.
    assert split_of("ends with a period") == split_of("  ends   with a period ")


def test_judge_row_prompt_and_distribution():
    row = judge_row(row_id="r", family="f", split="train", value=["b", "a"], predicate="  a   list ", gold_true=0.25, teachers={"t": [0.3, 0.7]})
    user = row["messages"][1]["content"]
    assert user.startswith('Value, between the markers:\n<<<value\n["b","a"]\nvalue>>>')
    assert "Property: the value is a list\n" in user
    assert row["gold"] == [0.25, 0.75] and row["group"] == predicate_id("a list")
    assert row["value_sha256"] == value_sha256(["b", "a"])


@pytest.mark.skipif(not (REPO / "ts-host/dist/native/refinement.js").exists() or shutil.which("node") is None, reason="ts-host not built")
def test_prompt_parity_with_the_typescript_judge(tmp_path):
    cases = [{"id": "s", "value": "line one\nline two", "predicate": "  one   line "}, {"id": "o", "value": {"b": [1, {"z": "é☃", "a": None}], "a": True}, "predicate": "an object"},
             {"id": "n", "value": 42, "predicate": "a whole number from 0 to 100"}, {"id": "l", "value": ["x", "y"], "predicate": "a list"}]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    out = subprocess.run(["node", str(REPO / "ts-host/scripts/refine-data/cli.mjs"), "messages", "--in", str(pairs)], capture_output=True, text=True, check=True,
                         env={**os.environ, "PATH": str(Path.home() / ".local" / "bin") + os.pathsep + os.environ["PATH"]})
    typescript = {row["id"]: row["messages"] for row in map(json.loads, out.stdout.splitlines())}
    for case in cases:
        assert typescript[case["id"]] == judge_messages(case["value"], case["predicate"]), case["id"]


# ---- near-miss plumbing -----------------------------------------------------------------------------------------

CANDIDATES = [{"predicate_id": predicate_id("one line of at most 20 characters"), "predicate": "one line of at most 20 characters",
               "uses": [{"app": "demo", "slots": ["Outgoing.subject"], "base": "string"}]}]


def test_near_miss_pipeline_accepts_only_supported_pairs():
    requests = judge.exemplify_requests(CANDIDATES, count=3)
    assert requests[0]["args"] == {"predicate": "one line of at most 20 characters", "base": "string", "slot": "demo:Outgoing.subject", "count": 3}
    results = [{"id": requests[0]["id"], "ok": True, "value": {"values": ["Order shipped", "Payment received", "Order shipped", "", "Delayed"]}}]
    satisfying = judge.satisfying_from_exemplify(requests, results)
    assert [row["value"] for row in satisfying].count("Order shipped") == 1 and len(satisfying) == 3
    nm_requests = judge.near_miss_requests(satisfying)
    ids = {row["value"]: row["id"] for row in satisfying}
    edits = {"Order shipped": "Order shipped to the address you gave us", "Payment received": "Payment received", "Delayed": 7}
    nm_results = [{"id": "nm:" + ids[v], "ok": True, "value": {"edited": e, "edit": "lengthened"}} for v, e in edits.items()]
    vf_requests = judge.verify_requests(satisfying, nm_results)
    assert len(vf_requests) == 3
    good = {"original_holds": True, "edited_holds": False, "minimal": True, "reason": "length"}
    vf_results = [{"id": "vf:" + ids["Order shipped"], "ok": True, "value": good},
                  {"id": "vf:" + ids["Payment received"], "ok": True, "value": good},
                  {"id": "vf:" + ids["Delayed"], "ok": True, "value": good}]
    accepted, stats = judge.accept_near_misses(satisfying, nm_results, vf_results)
    assert [pair["original"] for pair in accepted] == ["Order shipped"]
    assert stats["unchanged"] == 1 and stats["type-changed"] == 1 and stats["accepted"] == 1
    bad = {**good, "edited_holds": True}
    vf_results[0]["value"] = bad
    assert judge.accept_near_misses(satisfying, nm_results, vf_results)[0] == []


def test_assemble_keeps_agreeing_pairs_and_lists_disagreements():
    accepted = [{"id": "sat:a", "predicate": "short", "original": "ok", "edited": "much too long", "edit": "lengthened", "verifier_reason": "r"},
                {"id": "sat:b", "predicate": "short", "original": "fine", "edited": "also long", "edit": "e", "verifier_reason": "r"}]
    mined = [{"id": "mined:x", "predicate": "short", "value": "tiny", "observed": [{"outcome": "pass"}], "shadow_disagreement": True}]
    labels = [{"id": "sat:a:orig", "teacher": "T", "p_true": 0.9}, {"id": "sat:a:edit", "teacher": "T", "p_true": 0.1},
              {"id": "sat:b:orig", "teacher": "T", "p_true": 0.8}, {"id": "sat:b:edit", "teacher": "T", "p_true": 0.7},
              {"id": "mined:x", "teacher": "T", "p_true": 0.65}, {"id": "bad", "teacher": "T", "error": "x"}]
    rows, disagreements, report = judge.assemble(accepted, mined, labels, "T")
    assert report["near_miss_disagreements"] == 1 and [d["id"] for d in disagreements] == ["sat:b"]
    near = [row for row in rows if row["source"] == "near-miss"]
    assert sorted(row["gold"][0] for row in near) == [0.0, 1.0]
    assert near[0]["teachers"]["T"][0] in (0.9, 0.1)
    (mined_row,) = [row for row in rows if row["source"] == "mined"]
    assert mined_row["gold"] == [0.65, 0.35] and mined_row["shadow_disagreement"] is True
    assert len({row["split"] for row in rows}) == 1, "all rows of one predicate share a split"


# ---- miner ------------------------------------------------------------------------------------------------------

def _check(value, predicate="polite", outcome="pass", **extra):
    return {"kind": "refinement_check", "call_id": "c1", "phase": "return", "path": "return", "predicate": predicate, "value": value,
            "value_sha256": extra.pop("sha", None) or value_sha256(value), "outcome": outcome, "probability": 0.9, "judge": "m", "source": "judge", **extra}


def test_miner_reads_trace_files_and_dedupes(tmp_path):
    trace = tmp_path / "t.jsonl"
    events = [_check("Thank you."), _check("Thank you."), _check("Your fault.", outcome="fail"),
              {"kind": "refinement_shadow", "call_id": "c2", "path": "return", "predicate": "short", "value": "abc", "crisp": True, "nl": False, "agree": False, "probability": 0.2, "judge": "m"},
              {"kind": "other"}]
    trace.write_text(json.dumps({"events": events[:3]}) + "\n" + json.dumps(events[3]) + "\n" + json.dumps(events[4]) + "\n")
    pairs, stats = miner.mine(list(miner.events_of_trace_file(trace)))
    assert stats["events"] == 4 and stats["duplicates"] == 1 and stats["shadow_disagreements"] == 1
    assert sorted(pair["value"] for pair in pairs) == ["Thank you.", "Your fault.", "abc"]
    shadow = next(pair for pair in pairs if pair["value"] == "abc")
    assert shadow["shadow_disagreement"] is True and shadow["predicate"] == "short"
    assert all("label" not in pair and "gold" not in pair for pair in pairs), "the miner emits unlabeled pairs"


def _store(tmp_path, call_id, output, events):
    root = tmp_path / "store"
    (root / "blobs").mkdir(parents=True)

    def blob(text):
        digest = __import__("hashlib").sha256(text.encode()).hexdigest()
        path = root / "blobs" / digest[:2] / digest[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return digest
    out_hash = blob(json.dumps(output))
    record_hash = blob(json.dumps({"output": {"complete": True, "hash": out_hash, "bytes": 1}}))
    events_hash = blob("\n".join(json.dumps(event) for event in events))
    db = sqlite3.connect(root / "calls.sqlite")
    db.execute("CREATE TABLE calls (call_id TEXT PRIMARY KEY, record_hash TEXT, events_hash TEXT, started_at TEXT)")
    db.execute("INSERT INTO calls VALUES (?, ?, ?, '2026-10-09')", (call_id, record_hash, events_hash))
    db.commit()
    db.close()
    return root


def test_miner_completes_truncated_previews_from_the_store(tmp_path):
    long_text = "word " * 200
    sha = canonical_json_sha256_hex_of(long_text)
    preview = long_text[:400] + f" … ({len(long_text)} chars)"
    event = _check(preview, sha=sha, call_id="c1")
    unrecoverable = _check("y" * 400 + " … (900 chars)", sha="11" * 32, call_id="c1")
    root = _store(tmp_path, "c1", long_text, [event, unrecoverable])
    reader = miner.StoreReader(root)
    try:
        sources = list(miner.events_of_store(reader))
        pairs, stats = miner.mine(sources, [reader])
    finally:
        reader.close()
    assert stats["truncated_recovered"] == 1 and stats["truncated_unrecoverable"] == 1
    (pair,) = pairs
    assert pair["value"] == long_text and pair["value_rendered_only"] is False


def canonical_json_sha256_hex_of(value):
    from refine_data.common import canonical_json_sha256_hex
    return canonical_json_sha256_hex(value)
