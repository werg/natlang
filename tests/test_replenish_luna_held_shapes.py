"""The Luna replenisher must distinguish a long resumable case from a stall."""

import json
import hashlib
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


def test_worker_and_retry_keep_campaign_runtime_after_default_changes(tmp_path, monkeypatch):
    campaign = tmp_path / 'provider-luna-held-shapes-s52116-v1'
    retry = campaign / 'worker-01' / 'retry-000-01'
    retry.mkdir(parents=True)
    (campaign / 'campaign-manifest.json').write_text(json.dumps({'seed': 52116}))
    pinned = tmp_path / 'pinned-runtime'
    pinned.mkdir()
    fingerprint_file = pinned / 'frozen-runtime.json'
    fingerprint_file.write_text('{}\n')
    (campaign / 'runtime-pin.json').write_text(json.dumps({
        'schema': 'natlang.generation-runtime-pin/1', 'runtime': str(pinned),
        'runtime_sha256': hashlib.sha256(fingerprint_file.read_bytes()).hexdigest()}))
    monkeypatch.setattr(luna, 'RUNTIME', tmp_path / 'new-default')

    assert luna.runtime_for_queue(campaign / 'worker-01') == pinned
    assert luna.runtime_for_queue(retry) == pinned
    fingerprint_file.write_text('{"changed":true}\n')
    try:
        luna.runtime_for_queue(retry)
    except ValueError as exc:
        assert 'fingerprint changed' in str(exc)
    else:
        raise AssertionError('changed runtime must not resume a pinned queue')


def test_repair_queue_uses_manifest_runtime(tmp_path, monkeypatch):
    campaign = tmp_path / 'repair' / 'teacher'
    worker = campaign / 'worker-01'
    worker.mkdir(parents=True)
    pinned = tmp_path / 'pinned-repair-runtime'
    pinned.mkdir()
    fingerprint_file = pinned / 'frozen-runtime.json'
    fingerprint_file.write_text('{}\n')
    (campaign / 'campaign-manifest.json').write_text(json.dumps({
        'seed': 52114, 'repair_cycle': True, 'runtime': str(pinned),
        'runtime_sha256': hashlib.sha256(fingerprint_file.read_bytes()).hexdigest()}))
    monkeypatch.setattr(luna, 'RUNTIME', tmp_path / 'new-default')

    assert luna.runtime_for_queue(worker) == pinned
