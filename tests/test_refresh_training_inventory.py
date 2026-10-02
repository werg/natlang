import hashlib
import json
import sys
from pathlib import Path

from scripts import refresh_training_inventory as refresh


def test_inventory_refresh_persists_blocked_readiness(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    (repo / "training/data_sources.json").write_text("{}\n")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text('{"stages":[]}\n')
    catalog_report = tmp_path / "catalog-report.json"
    payload = json.dumps({
        "version": "natlang.training_data_inventory/1",
        "missing_required_default_inputs": ["a held input"],
        "included_quality_blockers": [],
    }, sort_keys=True).encode() + b"\n"
    catalog_report.write_bytes(payload)
    monkeypatch.setattr(refresh, "catalog", lambda _repo, _config: (catalog_report, json.loads(payload)))
    report_out, ready_out = tmp_path / "run/report.json", tmp_path / "run/ready.json"
    report_out.parent.mkdir(parents=True)
    report_out.write_text("old report\n")
    monkeypatch.setattr(sys, "argv", ["refresh", "--repo", str(repo), "--pipeline", str(pipeline),
                                       "--report-out", str(report_out), "--ready-out", str(ready_out)])
    assert refresh.main() == 75
    assert report_out.read_bytes() == payload
    ready = json.loads(ready_out.read_text())
    assert ready["ready"] is False
    assert ready["missing_required_default_inputs"] == ["a held input"]
    assert ready["sha256"] == hashlib.sha256(payload).hexdigest()
    assert list(report_out.parent.glob("*.pending")) == []

