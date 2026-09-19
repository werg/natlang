import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_classifier_output", Path(__file__).resolve().parents[1] / "scripts" / "audit_classifier_output.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_compact_audit_checks_batch_result(tmp_path):
    labels = tmp_path / "labels.jsonl"
    batches = labels.with_suffix(".jsonl.batches.jsonl")
    response = {"results": [{"label": "yes", "scores": {"yes": .9, "no": .1}}]}
    batch = {"batch_id": "abc", "ids": ["id-1"],
             "request": {"inputs": ["text"]}, "response": response}
    row = {"id": "id-1", "state": "text", "instruction": "Question?", "split": "train",
           "group_id": "group", "gold": "yes", "gold_source": "jev",
           "jev": {"batch_id": "abc", "response_index": 0,
                   "result": response["results"][0], "status": "accepted",
                   "model": "jev-1", "label": "yes"}}
    batches.write_text(json.dumps(batch) + "\n")
    labels.write_text(json.dumps(row) + "\n")
    assert module.audit(labels)["status"] == {"accepted": 1}
    row["jev"]["result"] = {"label": "no"}
    labels.write_text(json.dumps(row) + "\n")
    try:
        module.audit(labels)
    except ValueError as exc:
        assert "batch result mismatch" in str(exc)
    else:
        raise AssertionError("corrupt item result went unnoticed")
