import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from neuralese_data import bgkit, dedup, inventory, schnitzel, splits, swe_compaction  # noqa: E402
from neuralese_data.common import Reject  # noqa: E402
from neuralese_data.records import content_hash, exact_refs_from, strip_markup, validate  # noqa: E402

SCRIPT = ROOT / "scripts/neuralese_port_records.py"


def squad_row(**over):
    row = {
        "family": "qa_squad", "framing": "chat_context",
        "context": "Architecturally, the school has a Catholic character. <|reserved_6|>",
        "prompt": "Compress the passage to answer: To whom did the Virgin Mary appear?\n",
        "instruction": "To whom did the Virgin Mary appear?", "tool_name": "", "tool_args": "",
        "target": "ANSWER: Saint Bernadette Soubirous", "split": "train",
        "meta": json.dumps({"id": "5733be", "title": "University_of_Notre_Dame",
                            "answers": ["Saint Bernadette Soubirous", "Bernadette"]}),
    }
    row.update(over)
    return row


def test_bgkit_qa_strips_harness_and_validates():
    record = bgkit.convert_qa("qa_squad", 7, squad_row())
    assert validate(record) == []
    assert record["id"] == "bgkit:qa_squad:7"
    assert "<|reserved_6|>" not in record["sources"][0]["text"]
    assert record["target"]["value"] == "Saint Bernadette Soubirous"
    assert record["target"]["alternatives"] == ["Bernadette"]
    assert record["writer"]["instructions"].startswith("Read the passage so that this question can be answered")
    assert "Compress" not in json.dumps(record)
    assert record["split_groups"] == ["wiki:university_of_notre_dame"]
    assert record["consumer"]["withheld"] == ["sources"]


def test_bgkit_eval_split_becomes_test_and_bad_rows_reject():
    assert bgkit.convert_qa("qa_squad", 0, squad_row(split="eval"))["split"] == "test"
    for bad in (squad_row(target=""), squad_row(prompt="Summarise this"), squad_row(meta="{}")):
        try:
            bgkit.convert_qa("qa_squad", 0, bad)
        except Reject:
            continue
        raise AssertionError("expected rejection")


def test_bgkit_tool_digest_groups_by_workspace_repo():
    row = {"context": "Traceback in /workspace/Unidata__MetPy__1.5/src/metpy/calc/thermo.py:2850",
           "prompt": "Compress the bash result to: Find the failing call.\n", "instruction": "x",
           "tool_name": "bash", "tool_args": json.dumps({"cmd": "python t.py"}), "target": "thermo.py line 2850 fails.",
           "split": "train", "meta": json.dumps({"key": "abc"})}
    record = bgkit.convert_tool_digest("tool_digest_ext_v3", 3, row, {})
    assert validate(record) == []
    assert record["split_groups"] == ["repo:unidata/metpy"]
    assert record["outcome"]["label"] == "teacher"
    assert record["writer"]["instructions"] == "Read this bash result for: Find the failing call"
    assert {"text": "/workspace/Unidata__MetPy__1.5/src/metpy/calc/thermo.py:2850", "kind": "location"} in record["sources"][0]["exact_refs"]


def test_bgkit_repo_heldout_forces_test():
    row = {"context": "# file: scripts/x.py\nimport string\n", "prompt": "Compress the file to answer: Which modules import `string`?",
           "instruction": "Which modules import `string`?", "target": "scripts/x.py", "split": "train",
           "meta": json.dumps({"repo": "a/b", "task_id": "repo_qa:1"}), "tool_name": "", "tool_args": ""}
    record = bgkit.convert_repo("repo_qa_file_v3", 0, row, {"repo_qa:1": {"repo": "a/b", "commit": "c0", "verifier": "set"}}, {"a/b"})
    assert validate(record) == []
    assert record["split"] == "test"
    assert record["sources"][0]["title"] == "scripts/x.py"
    assert record["lineage"]["upstream_revision"] == "c0"


def sql_episode():
    return {"episode_id": "spider-train-0", "query": "Use the stored notes on database dm. Write one SQLite query.\nQuestion: How many heads?",
            "answer": "SELECT count(*) FROM head", "supports": [{"record_id": "r1", "text": "Database dm, table head (schema): CREATE TABLE head (id int)", "kind": "schema"}],
            "required_ids": ["r1"], "neutral": ["n1"], "support_annotation": "verified",
            "verify": {"type": "sql", "db": "/archive/raw/x.sqlite", "gold": "SELECT count(*) FROM head"},
            "provenance": {"dataset": "spider", "domain": "spider", "split": "train", "db_id": "dm"}}


def test_schnitzel_sql_episode():
    record = schnitzel.convert_episode("tasks-spider-20260927", "train", 0, sql_episode(), {"n1": "Distractor table text."})
    assert validate(record) == []
    assert record["id"] == "sdkb:spider:spider-train-0"
    assert record["consumer"]["context"][0]["content"].startswith("Write one SQLite query.")
    assert "Use the stored" not in record["writer"]["instructions"]
    assert record["target"] == {"kind": "sql", "value": "SELECT count(*) FROM head", "alternatives": []}
    assert record["outcome"]["details"]["db"].startswith("/mnt/external/sdkb-archive/")
    assert record["split_groups"] == ["sql-db:spider:dm"]
    assert record["lineage"]["notes"]["required_sources"] == [0]
    assert len(record["contrasts"]["distractors"]) == 1


def test_schnitzel_multi_turn_rejected():
    episode = sql_episode() | {"verify": {"type": "trajectory"}, "turns": [{"role": "assistant", "text": "x"}]}
    try:
        schnitzel.convert_episode("tasks-spider-20260927", "train", 0, episode, {})
    except Reject as exc:
        assert "multi-turn" in str(exc)
    else:
        raise AssertionError("expected rejection")


def trajectory(n_turns=8):
    msgs = [{"role": "system", "content": "You are OpenHands."}, {"role": "user", "content": "Fix the bug in pkg/a.py"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"step {i}", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"cat pkg/a.py\"}"}}]})
        msgs.append({"role": "tool", "content": f"output {i} of pkg/a.py:{i}", "tool_call_id": f"c{i}", "name": "bash"})
    return msgs


def test_swe_windows_at_assistant_boundaries():
    traj = trajectory()
    picks = swe_compaction.windows_for(traj, 2, "seed")
    assert picks and all(traj[r]["role"] == "assistant" and traj[t]["role"] == "assistant" for r, t in picks)
    row = {"trajectory_id": "t1", "instance_id": "o__r-1", "repo": "o/r", "trajectory": traj, "resolved": 1,
           "tools": [{"function": {"name": "bash", "description": "run", "parameters": None}}], "exit_status": "submit"}
    recent, target = picks[0]
    record = swe_compaction.convert_window(0, row, recent, target, set())
    assert validate(record) == []
    assert record["task"] == "continue" and record["outcome"]["label"] == "checked"
    assert record["consumer"]["context"][0]["content"] == "Fix the bug in pkg/a.py"
    assert all(m["role"] != "system" for m in record["sources"][0]["messages"])
    assert record["consumer"]["tools"] == [{"function": {"name": "bash", "description": "run"}}]
    protected = swe_compaction.convert_window(0, row, recent, target, {"o__r-1"})
    assert protected["split"] == "test"


def test_validate_catches_markup_and_tampering():
    record = bgkit.convert_qa("qa_squad", 0, squad_row())
    record["target"]["value"] = "x <|im_end|>"
    errors = validate(record)
    assert any("markup" in e for e in errors) and any("sha256" in e for e in errors)
    assert strip_markup("a<|im_start|>b<think>\n</think>c") == "abc"


def test_exact_refs():
    refs = exact_refs_from("see src/a/b.py:12 and pkg/mod.py::func plus https://x.org/y.")
    kinds = {r["text"]: r["kind"] for r in refs}
    assert kinds["src/a/b.py:12"] == "location"
    assert kinds["pkg/mod.py::func"] == "location"
    assert kinds["https://x.org/y"] == "url"


def _rec(i, group, split="train", text=None, question="q?", target="answer text"):
    row = squad_row(context=text or f"unique document number {i} " + " ".join(f"w{i}x{j}" for j in range(40)),
                    instruction=question, target=target, split=split,
                    meta=json.dumps({"id": str(i), "title": group}))
    return bgkit.convert_qa("qa_squad", i, row)


def test_dedup_exact_and_near_and_background():
    shared = "shared document " + " ".join(f"tok{j}" for j in range(60))
    a, b = _rec(1, "A", text=shared), _rec(2, "B", text=shared)
    c = _rec(3, "C", text=shared + " extra")
    result = dedup.analyse([a, b, c])
    assert result["drop"] == ["bgkit:qa_squad:2"] or result["drop"] == ["bgkit:qa_squad:1"]
    reasons = {r for _, _, r in result["links"]}
    assert {"exact-example", "near-duplicate-source"} <= reasons
    rules = "Rule text " + " ".join(f"r{j}" for j in range(60))
    many = [_rec(10 + i, f"G{i}", text=rules, question=f"question {i}", target=f"t{i}") for i in range(4)]
    result = dedup.analyse(many)
    assert result["report"]["background_source_texts"] == 1
    assert not [l for l in result["links"] if l[2] in ("same-source", "near-duplicate-source")]


def test_closure_pulls_component_to_most_protected_split():
    a = _rec(1, "A")
    b = _rec(2, "A", split="eval")
    c = _rec(3, "C")
    d = _rec(4, "D")
    report = splits.close([a, b, c, d], links=[(c["id"], d["id"])],
                          protected={"question_hashes": {}, "ids": {"4": "bench"}})
    assert a["split"] == "test" and b["split"] == "test"
    assert c["split"] == "test" and d["split"] == "test"
    assert report["moved"] == {"train->test": 3}
    assert splits.check_closed([a, b, c, d]) == []
    assert content_hash(a) == a["lineage"]["sha256"]  # split changes do not invalidate the content hash


def test_store_family_and_version_parsing():
    assert inventory._base_and_version("tool_slots_synth_v3b_mapped.parquet") == ("tool_slots_synth_mapped", 3, "b")
    assert inventory._base_and_version("repo_tree_v6_heldout.parquet") == ("repo_tree", 6, "")
    assert inventory._base_and_version("qa_squad.parquet") == ("qa_squad", 0, "")


def test_cli_validate_reports_errors(tmp_path):
    good = bgkit.convert_qa("qa_squad", 0, squad_row())
    bad = dict(good, task="summarise")
    path = tmp_path / "x.port-records.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    result = subprocess.run([sys.executable, str(SCRIPT), "validate", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["records"] == 2 and report["duplicate_ids"] == 1 and "task" in report["errors"]
