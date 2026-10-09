"""The refine_data_cli commands end to end on temporary directories (no model, no network)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLI = str(REPO / "scripts/refine_data_cli.py")


def run(*args):
    result = subprocess.run([sys.executable, CLI, *map(str, args)], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    text = result.stdout.strip()
    try:
        return json.loads(text)
    except ValueError:
        return json.loads(text.splitlines()[-1])


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_harvest_exact_and_author_corpora(tmp_path):
    seeds = tmp_path / "seeds.jsonl"
    assert run("harvest", "--out", seeds)["predicates"] >= 110
    exact_dir = tmp_path / "exact"
    summary = run("exact", "--out-dir", exact_dir, "--id", "refine-judge-exact-test", "--per-predicate", 8)
    manifest = json.loads((exact_dir / "manifest.json").read_text())
    assert manifest["training_admission"] is False and manifest["admission"] == "held-pending-review" and manifest["model_calls"] == 0
    assert "refine-judge-split/1" in manifest["split_rule"] and manifest["rows"] == summary["rows"] == len(lines(exact_dir / "judge-rows.jsonl"))
    assert set(manifest["rows_by_split"]) == {"train", "heldout-predicate", "heldout-family"}
    author_dir = tmp_path / "author"
    assert run("author", "--out-dir", author_dir, "--id", "refine-author-test")["examples"] >= 25
    assert (author_dir / "omissions.jsonl").read_text() == ""
    assert json.loads((author_dir / "manifest.json").read_text())["training_admission"] is False


def test_near_miss_flow_through_the_cli(tmp_path):
    ex_requests = tmp_path / "ex-req.jsonl"
    run("exemplify-requests", "--out", ex_requests, "--count", 2)
    requests = lines(ex_requests)
    assert len(requests) >= 110 and requests[0]["args"]["count"] == 2
    ex_results = tmp_path / "ex-res.jsonl"
    ex_results.write_text("".join(json.dumps({"id": r["id"], "ok": True, "value": {"values": [f"value {i} for {r['id']}"]}}) + "\n" for i, r in enumerate(requests[:5])))
    satisfying = tmp_path / "sat.jsonl"
    assert run("ingest-exemplify", "--requests", ex_requests, "--results", ex_results, "--out", satisfying)["satisfying"] == 5
    nm_req = tmp_path / "nm-req.jsonl"
    run("near-miss-requests", "--satisfying", satisfying, "--out", nm_req)
    nm_res = tmp_path / "nm-res.jsonl"
    nm_res.write_text("".join(json.dumps({"id": r["id"], "ok": True, "value": {"edited": r["args"]["value"] + " changed", "edit": "appended"}}) + "\n" for r in lines(nm_req)))
    vf_req = tmp_path / "vf-req.jsonl"
    run("verify-requests", "--satisfying", satisfying, "--near-miss-results", nm_res, "--out", vf_req)
    vf_res = tmp_path / "vf-res.jsonl"
    vf_res.write_text("".join(json.dumps({"id": r["id"], "ok": True, "value": {"original_holds": True, "edited_holds": False, "minimal": True, "reason": "x"}}) + "\n" for r in lines(vf_req)))
    accepted = tmp_path / "accepted.jsonl"
    assert run("accept", "--satisfying", satisfying, "--near-miss-results", nm_res, "--verify-results", vf_res, "--out", accepted)["accepted"] == 5
    label_req = tmp_path / "label-req.jsonl"
    assert run("label-requests", "--accepted", accepted, "--out", label_req)["pairs"] == 10
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"id": r["id"], "teacher": "T", "p_true": 0.95 if r["id"].endswith(":orig") else 0.05}) + "\n" for r in lines(label_req)))
    out = tmp_path / "assembled"
    report = run("assemble", "--out-dir", out, "--id", "refine-judge-nm-test", "--teacher", "T", "--accepted", accepted, "--labels", labels)
    assert report["near_miss_rows"] == 10 and report["near_miss_disagreements"] == 0


def test_calibration_cli_queue_status_export(tmp_path):
    exact_dir = tmp_path / "exact"
    run("exact", "--out-dir", exact_dir, "--id", "x", "--per-predicate", 4)
    queue = tmp_path / "queue.jsonl"
    assert run("calibration", "--action", "queue", "--rows", exact_dir / "judge-rows.jsonl", "--per-predicate", 2, "--out", queue)["items"] > 100
    items = lines(queue)
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text("".join(json.dumps({"schema": "natlang.refine-calibration/1", "id": i["id"], "predicate": i["predicate"], "predicate_id": i["predicate_id"],
                                            "family": i["family"], "value": i["value"], "value_sha256": i["value_sha256"], "verdict": True, "reviewer": "r",
                                            "reviewed_at": "t", "note": "", "review_round": 1}) + "\n" for i in items[:3]))
    status = run("calibration", "--action", "status", "--queue", queue, "--verdicts", verdicts)
    assert status["reviewed"] == 3
    exported = tmp_path / "rows.jsonl"
    assert run("calibration", "--action", "export", "--verdicts", verdicts, "--out", exported)["rows"] == 3
    assert {row["split"] for row in lines(exported)} == {"calibration"}
