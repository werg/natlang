"""refine-author: static before/after authoring examples from the repository."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from refine_data import author  # noqa: E402


@pytest.fixture(scope="module")
def generated():
    return author.generate()


def test_registry_entries_all_resolve(generated):
    examples, omissions, report = generated
    assert omissions == [], omissions
    registry = json.loads((REPO / author.REGISTRY).read_text())["entries"]
    assert len(examples) == len(registry) >= 25
    assert len({example["id"] for example in examples}) == len(examples)


def test_guard_moves_from_instructions_to_the_type(generated):
    examples = {example["id"]: example for example in generated[0]}
    subject = examples["refine-author:workflow-compose-subject"]
    guard = "one line of at most 60 characters, without a trailing period"
    assert guard in subject["before"]["function"] and guard not in subject["after"]["function"]
    assert guard not in subject["before"]["types"]
    assert f'subject: Is<string, "{guard}">' in subject["after"]["types"]
    # Nothing else changed: the after files differ from the before files only at the edited spans.
    assert subject["before"]["function"].replace(subject["guards"][0], "") != subject["after"]["function"]
    changed = [line for line in subject["diff"]["types"].splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    assert len(changed) == 2
    assert [m["role"] for m in subject["messages"]] == ["user", "assistant"]
    assert subject["after"]["function"] in subject["messages"][1]["content"]
    assert subject["origin"] == {"kind": "tree", "function": "applications/workflow/inform/compose.nl", "types": "applications/workflow/types.ts"}


def test_return_type_entries_refine_the_frontmatter(generated):
    examples = {example["id"]: example for example in generated[0]}
    pick = examples["refine-author:build-pick-return"]
    assert "returns: 'Is<string, \"the id of one of the supplied ready tasks\">'" in pick["after"]["function"]
    assert "returns: string" in pick["before"]["function"]
    assert pick["before"]["types"] == pick["after"]["types"]


def test_history_entries_come_from_git(generated):
    history = [example for example in generated[0] if example["origin"]["kind"] == "history"]
    assert {example["id"].split(":")[1] for example in history} == {
        "history-logs-assess", "history-wiki-reconcile-accounted", "history-build-choose-return", "history-migration-propose-patches"}
    for example in history:
        assert example["origin"]["rev"].endswith("^") and example["after"]["function"] != example["before"]["function"]
    reconcile = next(e for e in history if "reconcile" in e["id"])
    assert "Account for every update ID exactly once in accounted." in reconcile["before"]["function"]
    assert "Account for every update ID" not in reconcile["after"]["function"]
    assert 'accounted: Is<string[], "every update ID of the block once, sorted">' in reconcile["after"]["types"]
    propose = next(e for e in history if "propose" in e["id"])
    assert "Patch.old" in propose["slots"] and "Patch.path" in propose["slots"]
    assert propose["after"]["types"].count("Is<") == 2


def test_generation_is_deterministic_and_reports_uncovered_predicates(generated):
    again = author.generate()
    assert [e["content_sha256"] for e in again[0]] == [e["content_sha256"] for e in generated[0]]
    report = generated[2]
    harvested = {c.id for c in author.harvest()[0]}
    covered = harvested & {pid for e in generated[0] for pid in e["predicate_ids"]}
    assert report["candidates"] == report["candidates_without_example"] + len(covered)
    assert report["candidates_without_example"] > 0, "the report states what is not covered"


def test_missing_guard_and_slot_are_omissions_not_silence():
    candidates = author.harvest()[0]
    entry = {"id": "x", "app": "workflow", "function": "inform/compose.nl", "slots": ["Outgoing.subject"], "edits": [{"find": "no such text", "replace": ""}]}
    assert author.build_example(entry, candidates) == (None, "guard text found 0 times (expected once): 'no such text'")
    entry = {"id": "x", "app": "workflow", "function": "inform/compose.nl", "slots": ["Outgoing.nonexistent"],
             "edits": [{"find": "Write the message for the customer", "replace": "Write the message"}]}
    example, reason = author.build_example(entry, candidates)
    assert example is None and "not in the DECOMPOSITION.md tables" in reason


@pytest.mark.skipif(not (REPO / "ts-host/dist/native/types.js").exists() or shutil.which("node") is None, reason="ts-host not built")
def test_refined_types_parse_with_the_runtime_parser(tmp_path, generated):
    path = tmp_path / "examples.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in generated[0]))
    out = subprocess.run(["node", str(REPO / "ts-host/scripts/refine-data/check-author.mjs"), str(path)], capture_output=True, text=True,
                         env={**os.environ, "PATH": "/home/werg/.local/bin:" + os.environ["PATH"]})
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(out.stdout)["checked"] >= 28
