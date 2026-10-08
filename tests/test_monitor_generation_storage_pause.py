import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import monitor_generation as monitor


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return path


def test_closed_luna_claim_clears_prior_storage_pause_to_idle(tmp_path):
    campaign = tmp_path / "campaign-v1"
    key = "reviewed-campaign-case-000007"
    queue = _write_json(campaign / "slot-01/claims" / key / "queue.jsonl", {
        "key": key, "index": 7, "count": 1,
    })
    journal = campaign / "slot-01/journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"event": "storage_pause", "key": key, "time": 123.0}) + "\n")
    ledger = campaign / "dispatch/claims.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("".join(json.dumps(row) + "\n" for row in [
        {"event": "campaign_open", "identity": {"campaign_id": "reviewed-campaign"}},
        {"event": "claim", "claim_id": key, "index": 7},
        {"event": "abandoned", "claim_id": key, "index": 7},
        {"event": "campaign_finished", "status": "finished", "pending_indices": []},
    ]))
    state_path = _write_json(tmp_path / "luna-state.json", {
        "status": "storage_pause_needs_agent_review",
        "workers": [{"queue": str(queue), "journal": str(journal),
                     "status": "storage_pause_needs_agent_review"}],
    })
    authority_path = _write_json(tmp_path / "authority.json", {
        "luna_workers": [{"pid": 987654321, "queue": str(queue), "journal": str(journal),
                          "campaign": "reviewed-campaign", "status": "storage_pause_needs_agent_review"}],
        "luna_state": str(state_path),
        "luna_status": "storage_pause_needs_agent_review",
        "additional_teachers": {"luna": {"workers": []}},
    })
    authority = json.loads(authority_path.read_text())

    assert monitor.unattended_storage_pauses(authority) == []
    assert monitor.mark_storage_pause_status(authority_path, [], []) == [
        {"kind": "luna", "status": "idle", "reason": "no active workers or unresolved queue pauses"}
    ]
    refreshed = json.loads(authority_path.read_text())
    assert refreshed["luna_status"] == "idle"
    assert refreshed["luna_active_campaigns"] == []
    assert refreshed["additional_teachers"]["luna"]["status"] == "idle"
    assert json.loads(state_path.read_text())["status"] == "idle"


def test_unclosed_luna_claim_keeps_storage_pause_reviewable(tmp_path):
    campaign = tmp_path / "campaign-v1"
    key = "reviewed-campaign-case-000002"
    queue = _write_json(campaign / "slot-01/claims" / key / "queue.jsonl", {
        "key": key, "index": 2, "count": 1,
    })
    journal = campaign / "slot-01/journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"event": "storage_pause", "key": key, "time": 123.0}) + "\n")
    ledger = campaign / "dispatch/claims.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("".join(json.dumps(row) + "\n" for row in [
        {"event": "campaign_open", "identity": {"campaign_id": "reviewed-campaign"}},
        {"event": "claim", "claim_id": key, "index": 2},
    ]))
    authority = {
        "luna_workers": [{"pid": 987654322, "queue": str(queue), "journal": str(journal),
                          "campaign": "reviewed-campaign", "status": "not_running"}],
    }

    paused = monitor.unattended_storage_pauses(authority)
    assert len(paused) == 1
    assert paused[0]["queue"] == str(queue.resolve())
    assert paused[0]["event"]["key"] == key
