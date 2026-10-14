"""The Luna replenisher must distinguish a long resumable case from a stall."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replenish_luna_held_shapes as luna  # noqa: E402


def test_durable_turns_reset_relaunch_streak(tmp_path, monkeypatch):
    campaign = tmp_path / 'provider-luna-held-shapes-s52046-v1'
    worker = campaign / 'worker-01'
    worker.mkdir(parents=True)
    (campaign / 'campaign-manifest.json').write_text(json.dumps({'seed': 52046, 'workers': 1}))
    (worker / 'queue.jsonl').write_text(json.dumps({'key': 'case-one'}) + '\n')
    (worker / 'journal.jsonl').write_text(json.dumps({
        'event': 'activity', 'key': 'case-one', 'saved_turns': 245}) + '\n')
    (worker / 'relaunch-state.json').write_text(json.dumps({'failures': 3, 'finished': 0}) + '\n')
    launched = []
    monkeypatch.setattr(luna, 'active', lambda unit: False)
    monkeypatch.setattr(luna, 'launch_worker', lambda *args, **kwargs: launched.append(args) or True)
    monkeypatch.setattr(luna.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))

    assert luna.complete(campaign) is False
    assert len(launched) == 1
    assert json.loads((worker / 'relaunch-state.json').read_text()) == {
        'failures': 0, 'finished': 0, 'saved_turns': 245}


def test_large_partial_gets_resume_memory_budget(tmp_path):
    worker = tmp_path / 'worker-01'
    jobs = worker / 'case-000' / 'jobs'
    jobs.mkdir(parents=True)
    assert luna.launch_budget_gb(worker) == '0.9'
    partial = jobs / 'case.partial.json'
    with partial.open('wb') as stream:
        stream.truncate(8_000_000)
    assert luna.launch_budget_gb(worker) == '2.0'
    retry = worker / 'retry-000-01'
    retry.mkdir()
    assert luna.launch_budget_gb(retry) == '1.5'
