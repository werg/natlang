"""The external-data adapters must produce verifiable interpreter trajectories."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_external import case_report_program, generate, leaf_program, map_report_program, read_tasks, select_tasks  # noqa: E402
from generate import run_program  # noqa: E402
from prepare_recent_tasks import sales_tasks, sql_tasks  # noqa: E402


def task(i, state, gold, *, split="train", license="CC0-1.0"):
    return {"id": f"sample-{i}", "source": "fixture", "group_id": f"group-{i}",
            "split": split, "state": state, "instruction": "Is this concerning?",
            "kind": "boolean", "labels": ["false", "true"], "gold": gold,
            "gold_source": "programmatic", "license": license}


def test_leaf_and_map_report_use_real_harness(tmp_path):
    rows = [task(1, "Customer asks a question", "false"),
            task(2, "Customer says payment is blocked", "true")]
    leaf, _ = run_program(leaf_program(rows[0]))
    assert any(s["skill"] == "write_value" for s in leaf)
    mapped, episodes = run_program(map_report_program(rows))
    assert episodes == 3
    assert any(s["skill"] in ("run_function", "for_each") for s in mapped)
    assert any(s["skill"] == "run_code" for s in mapped)
    out = tmp_path / "traces.jsonl"
    counts = generate(rows, out)
    assert counts["programs"] == 3
    assert all(json.loads(line)["split"] == "train" for line in out.read_text().splitlines())


def test_split_and_license_gate():
    rows = [task(1, "a", "true"), task(2, "b", "false", split="test"),
            task(3, "c", "true", license="Unknown")]
    selected, skipped = select_tasks(rows, False, {"train"})
    assert [t["id"] for t in selected] == ["sample-1"]
    assert skipped == {"split": 1, "license": 1}


def test_source_group_cannot_cross_splits(tmp_path):
    first = task(1, "a", "true")
    second = {**task(2, "b", "false", split="test"), "group_id": first["group_id"]}
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (first, second)) + "\n")
    try:
        read_tasks([path])
    except ValueError as exc:
        assert "crosses splits" in str(exc)
    else:
        raise AssertionError("source group crossed splits")


def test_jev_retry_supersedes_error_row(tmp_path):
    base = task(1, "a", "true")
    failed = {**base, "jev": {"status": "error", "input_hash": "same"}}
    accepted = {**base, "jev": {"status": "accepted", "input_hash": "same"}}
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (failed, accepted)) + "\n")
    tasks, _ = read_tasks([path])
    assert len(tasks) == 1
    assert tasks[0]["jev"]["status"] == "accepted"


def test_case_report_calls_independent_questions():
    state = "The customer asks for a demo and mentions price."
    concern = {**task(1, state, "true"), "id": "sales/c1/4/concern",
               "source_meta": {"question_key": "concern"}}
    demo = {**task(2, state, "true"), "id": "sales/c1/4/demo",
            "source_meta": {"question_key": "demo"}}
    samples, episodes = run_program(case_report_program([concern, demo]))
    assert episodes == 3
    assert sum(sample["skill"] == "run_function" for sample in samples) == 2
    assert samples[-1]["skill"] == "reply"


def test_sql_adapter_drops_empty_conflict_and_duplicate(tmp_path):
    path = tmp_path / "sql.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Query", "Label"])
        writer.writerows([["", "1"], ["same", "0"], ["same", "1"],
                          ["safe text", "0"], ["safe text", "0"], ["' OR 1=1 --", "1"]])
    rows = list(sql_tasks(path))
    assert len(rows) == 2
    assert {t["gold"] for t in rows} == {"false", "true"}
    assert all(t["license"] == "Unknown" for t in rows)


def test_sales_prefix_hides_future_and_annotations(tmp_path):
    path = tmp_path / "sales.csv"
    messages = [{"speaker": "customer", "message": "Is this expensive?"},
                {"speaker": "sales_rep", "message": "It costs five."},
                {"speaker": "customer", "message": "Can we try a demo?"}]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["conversation_id", "company_id", "conversation",
                                                     "outcome", "probability_trajectory"])
        writer.writeheader()
        writer.writerow({"conversation_id": "c1", "company_id": "co1",
                         "conversation": json.dumps(messages), "outcome": "1",
                         "probability_trajectory": '{"0":0.1}'})
    rows = list(sales_tasks(path, max_prefixes=2))
    assert len(rows) == 4
    first = rows[0]
    assert "Can we try a demo?" not in first["state"]
    assert "outcome" not in json.dumps(first)
    assert "probability_trajectory" not in json.dumps(first)
    assert all("gold" not in row for row in rows)
