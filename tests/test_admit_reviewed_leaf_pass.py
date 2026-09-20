import hashlib
import json

import pytest

from natlang.gen.codebases import ref_key
from scripts.admit_reviewed_leaf_pass import admit


def test_admission_requires_reviewed_digest_and_is_idempotent(tmp_path):
    args = {"action": {"code": "counter_offer", "good": "rope", "qty": 1, "price": 10}}
    row = {"key": ref_key("say", args), "function": "say", "args": args,
           "value": "The rope is ten coins.", "accepted": False, "status": "done"}
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps(row) + "\n")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"source_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
                                  "approved": {"counter_offer": [0]},
                                  "notes": {"0": "Valid counter-offer; no sale claimed."}}))
    bank = tmp_path / "bank.jsonl"
    bank.write_text("")

    assert admit(audit, review, bank) == {"reviewed": 1, "appended": 1, "already_present": 0}
    assert admit(audit, review, bank) == {"reviewed": 1, "appended": 0, "already_present": 1}
    assert json.loads(bank.read_text())["value"] == row["value"]

    audit.write_text(audit.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from reviewed snapshot"):
        admit(audit, review, bank)


def test_judge_override_requires_a_note(tmp_path):
    args = {"action": {"code": "counter_offer", "good": "rope", "qty": 1, "price": 10}}
    row = {"key": ref_key("say", args), "function": "say", "args": args,
           "value": "The rope is ten coins.", "accepted": False, "status": "done"}
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps(row) + "\n")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"source_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
                                  "approved": {"counter_offer": [0]}}))
    bank = tmp_path / "bank.jsonl"
    bank.write_text("")
    with pytest.raises(ValueError, match="override note"):
        admit(audit, review, bank)
    assert bank.read_text() == ""
