import json

from scripts.audit_teacher_coverage import audit


def test_coverage_audit_rejects_new_unregistered_target_and_low_counts(tmp_path):
    root = tmp_path
    (root / "codebases/new").mkdir(parents=True)
    (root / "codebases/new/main.nl").write_text("return ok")
    (root / "applications").mkdir()
    config = root / "coverage.json"
    config.write_text(json.dumps({"targets": [], "excluded": [],
        "minimums": {"studio": {"train": 1, "eval": 0}}}))
    studio = root / "studio.jsonl"; studio.write_text("")
    programs = root / "programs.jsonl"; programs.write_text("")
    report = audit(root, config, studio, programs)
    assert report["unknown"] == ["codebase:new"] and not report["ready"]

    config.write_text(json.dumps({"targets": [{"id": "studio:new", "provider": "studio",
        "owns": ["codebase:new"]}], "excluded": [],
        "minimums": {"studio": {"train": 2, "eval": 0}}}))
    studio.write_text(json.dumps({"target": "studio:new", "split": "train"}) + "\n")
    report = audit(root, config, studio, programs)
    assert report["unknown"] == []
    assert report["below_minimum"][0]["found"] == 1
