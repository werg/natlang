from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))

import vibestudio_campaign


def test_failed_infrastructure_does_not_use_a_teacher_attempt(tmp_path):
    cases = {
        'reset': {'workspace': 'system', 'family': 'settings'},
        'android': {'workspace': 'system', 'family': 'device', 'requiresAndroidEmulator': True},
        'other': {'workspace': 'system', 'family': 'settings'},
        'recovered': {'workspace': 'system', 'family': 'settings'},
    }
    receipt = {
        'reset': {'status': 'test-failed', 'reason': '[eval.get] DO RPC EvalDO.getRun ECONNRESET'},
        'android': {'status': 'test-failed', 'reason': 'Connect exactly one ready disposable Android emulator'},
        'other': {'status': 'test-failed', 'reason': 'The requested update was not saved'},
        'recovered': {'status': 'export-rejected'},
    }
    rows = vibestudio_campaign.publish_status(
        tmp_path, cases, {'recovered': 'manifest.json'}, [receipt], None, 1,
        {'android': 'requires one booted disposable Android emulator'},
    )
    assert rows['reset']['attempts'] == 0
    assert rows['android']['attempts'] == 0
    assert rows['android']['status'] == 'prerequisite-unavailable'
    assert rows['other']['status'] == 'test-failed'
    assert rows['recovered']['status'] == 'verified'
    assert vibestudio_campaign.next_wave(cases, rows, 6) == ['reset']
    summary = vibestudio_campaign.read(tmp_path / 'status.json')
    assert summary['attempted_goals'] == summary['attempts'] == 4
    assert summary['infrastructure_failures'] == 2


def test_only_cases_with_unavailable_prerequisites_are_held(monkeypatch, tmp_path):
    cases = {
        'android': {'requiresAndroidEmulator': True},
        'deployment': {'requiresDeploymentPackage': True},
        'desktop': {'devices': ['desktop']},
    }
    monkeypatch.setattr(vibestudio_campaign.shutil, 'which', lambda name: name)

    def probes(command, **_kwargs):
        if command == ['adb', 'devices']:
            return subprocess.CompletedProcess(command, 0, stdout='emulator-5554\tdevice\nphysical\tdevice\n')
        if command[:2] == ['adb', '-s']:
            assert command[2] == 'emulator-5554'
            return subprocess.CompletedProcess(command, 0, stdout='1\n')
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', probes)
    assert set(vibestudio_campaign.prerequisite_holds(cases, tmp_path)) == {'deployment'}


def test_failed_accepted_eval_reconciliation_is_an_infrastructure_failure():
    outcome = {'status': 'test-failed', 'reason': 'eval: status reconciliation failed for accepted run example'}
    assert vibestudio_campaign.infrastructure_failure(outcome, {})


def test_release_epochs_reset_failures_without_losing_verified_history(tmp_path):
    cases = {name: {'workspace': 'system', 'family': 'settings'} for name in ['retry', 'done', 'reset']}
    failed = {'retry': {'status': 'test-failed', 'reason': 'wrong outcome'},
              'reset': {'status': 'test-failed', 'reason': '[eval.get] DO RPC EvalDO.getRun ECONNRESET'}}
    historical = {'retry': {'status': 'export-rejected'}, 'done': {'status': 'verified'}}
    receipts = [failed, historical, failed]
    rows = vibestudio_campaign.publish_status(tmp_path, cases, {}, receipts, None, 2, {},
                                             attempt_epoch='new-release', receipt_epochs=[None, 'old-release', 'new-release'])
    assert rows['done']['status'] == 'verified'
    assert rows['retry']['status'] == 'pending'
    assert rows['retry']['attempts'] == 1
    assert rows['retry']['cumulative_attempts'] == 3
    assert rows['reset']['attempts'] == 0
    assert rows['reset']['epoch_infrastructure_failures'] == 1
    assert vibestudio_campaign.next_wave(cases, rows, 6) == ['reset']
    summary = vibestudio_campaign.read(tmp_path / 'status.json')
    assert summary['attempts'] == 6
    assert summary['epoch_attempts'] == 2
    assert summary['epoch_teacher_attempts'] == 1
    exhausted = vibestudio_campaign.publish_status(tmp_path, cases, {}, receipts + [failed], None, 2, {},
        attempt_epoch='new-release', receipt_epochs=[None, 'old-release', 'new-release', 'new-release'])
    assert exhausted['retry']['status'] == 'test-failed'
    fresh = vibestudio_campaign.publish_status(tmp_path, cases, {}, receipts + [failed], None, 2, {},
        attempt_epoch='next-release', receipt_epochs=[None, 'old-release', 'new-release', 'new-release'])
    assert fresh['retry']['status'] == 'pending'
    assert fresh['retry']['attempts'] == 0
    assert fresh['done']['status'] == 'verified'


def test_receipt_history_uses_immutable_original_wave_epochs(tmp_path):
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    old = tmp_path / '0000-result.json'
    current = tmp_path / '0001-result.json'
    vibestudio_campaign.publish_immutable(old, {'goal': {'status': 'test-failed'}})
    vibestudio_campaign.publish_immutable(current, {'goal': {'status': 'verified'}})
    vibestudio_campaign.publish_immutable(jobs / '0000.json', {'cases': ['goal']})
    vibestudio_campaign.publish_immutable(jobs / '0001.json', {'cases': ['goal'], 'attempt_epoch': 'release-sha'})
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in [old, current]}
    receipts, epochs = vibestudio_campaign.receipt_history(tmp_path, jobs)
    assert epochs == [None, 'release-sha']
    assert receipts == [vibestudio_campaign.read(old), vibestudio_campaign.read(current)]
    vibestudio_campaign.publish_immutable(old, vibestudio_campaign.read(old))
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    import pytest
    with pytest.raises(ValueError, match='immutable campaign record differs'):
        vibestudio_campaign.publish_immutable(old, {})


def test_default_attempt_epoch_is_verified_release_sha_with_exact_provenance(monkeypatch, tmp_path):
    import hashlib
    from types import SimpleNamespace
    release = tmp_path / 'release' / 'userland-builds' / 'release.json'
    release.parent.mkdir(parents=True)
    release.write_text('{}\n')
    digest = hashlib.sha256(release.read_bytes()).hexdigest()
    pins = tmp_path / 'pins.json'
    catalog = tmp_path / 'catalog.jsonl'
    catalog.write_text('{"goal":"example"}\n')
    vibestudio_campaign.publish(pins, {'release': {'manifest': str(release), 'sha256': digest}})
    code = {'schema': 'natlang.code-provenance/1', 'git_commit': 'driver-sha', 'dirty': False}
    monkeypatch.setattr(vibestudio_campaign, 'code_provenance', lambda *_: code)
    args = SimpleNamespace(release_pins=pins, attempt_epoch=None, host=tmp_path)
    epoch, provenance, env = vibestudio_campaign.attempt_context(args, catalog)
    assert epoch == digest
    assert provenance == {'release': vibestudio_campaign.read(pins)['release'],
        'release_pins_sha256': hashlib.sha256(pins.read_bytes()).hexdigest(),
        'catalog_sha256': hashlib.sha256(catalog.read_bytes()).hexdigest(), 'model': vibestudio_campaign.LUNA, 'driver': code, 'host': code}
    assert env['VIBESTUDIO_WORKSPACE_RELEASE_ROOT'] == str(release.parent.parent)
    args.attempt_epoch = 'intentional-rerun'
    assert vibestudio_campaign.attempt_context(args, catalog)[0] == 'intentional-rerun'


def test_drive_retries_historical_exhaustion_in_new_immutable_wave(monkeypatch, tmp_path):
    from types import SimpleNamespace
    cases = {name: {'taskId': name, 'workspace': 'system', 'family': 'settings'} for name in ['retry', 'done']}
    waves = tmp_path / 'waves'
    jobs = waves / 'jobs'
    jobs.mkdir(parents=True)
    historical = {
        jobs / '0000.json': {'cases': ['retry', 'done']},
        waves / '0000-result.json': {'retry': {'status': 'test-failed'}, 'done': {'status': 'verified'}},
        jobs / '0001.json': {'cases': ['retry']},
        waves / '0001-result.json': {'retry': {'status': 'export-rejected'}},
        jobs / '0002.json': {'cases': ['retry']},
    }
    for path, value in historical.items():
        vibestudio_campaign.publish_immutable(path, value)
    originals = {path: path.read_bytes() for path in historical}
    provenance = {'release': {'sha256': 'new-release'}, 'driver': {'git_commit': 'new-driver'}}
    monkeypatch.setattr(vibestudio_campaign, 'campaign_scope', lambda *_: (None, {}, cases))
    monkeypatch.setattr(vibestudio_campaign, 'attempt_context', lambda *_: ('new-release', provenance, {'PINNED': 'yes'}))
    monkeypatch.setattr(vibestudio_campaign, 'prerequisite_holds', lambda *_: {})
    monkeypatch.setattr(vibestudio_campaign, 'sync_index', lambda *_: None)
    monkeypatch.setattr(vibestudio_campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    archived = []
    monkeypatch.setattr(vibestudio_campaign, 'remote_archive', lambda *args, **kwargs: archived.append(kwargs['wave']))
    monkeypatch.setattr(vibestudio_campaign, 'outcomes', lambda _capture, names, *_:
        {name: {'status': 'test-failed', 'reason': 'genuine unsuccessful goal'} for name in names})
    def collect(command, **kwargs):
        assert command[:4] == ['pnpm', '--silent', 'natlang:teacher', '--model']
        assert kwargs['env'] == {'PINNED': 'yes'}
        assert command.count('--case') == 1 and command[-1] == 'retry'
        Path(command[command.index('--out') + 1]).mkdir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', collect)
    args = SimpleNamespace(root=tmp_path, held=tmp_path / 'held', seed=[], host=tmp_path, archive=tmp_path / 'archive',
        review_model=None, attempts=2, max_waves=1, wave_size=6, concurrency=3, reserve_disk_gib=8)
    vibestudio_campaign.drive(args)
    plan = vibestudio_campaign.read(jobs / '0003.json')
    assert plan == {'cases': ['retry'], 'attempt_epoch': 'new-release', 'provenance': provenance}
    execution = vibestudio_campaign.read(waves / '0003-execution.json')
    assert archived == [{'plan': plan, 'execution': execution}]
    status = vibestudio_campaign.read(tmp_path / 'status.json')
    assert status['cases']['retry']['attempts'] == 1
    assert status['cases']['retry']['cumulative_attempts'] == 3
    assert status['cases']['done']['status'] == 'verified'
    assert {path: path.read_bytes() for path in historical} == originals


def test_unfinished_historical_capture_archives_but_never_restarts_on_new_release(tmp_path):
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    plan = jobs / '0000.json'
    vibestudio_campaign.publish_immutable(plan, {'cases': ['goal']})
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'new-release', {}) is None
    (tmp_path / '0000-capture').mkdir()
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'new-release', {}) == plan


def test_unfinished_plan_requires_exact_release_provenance_and_preserves_admitted_cases(tmp_path):
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    plan = jobs / '0000.json'
    provenance = {'release': {'sha256': 'release-sha'}, 'driver': {'git_commit': 'driver'}}
    vibestudio_campaign.publish_immutable(plan, {'cases': ['goal', 'done'], 'attempt_epoch': 'release-sha',
                                               'provenance': provenance})
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'release-sha', provenance) == plan
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'another-release', provenance) is None
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'release-sha', {'release': 'different'}) is None
    execution = tmp_path / '0000-execution.json'
    vibestudio_campaign.publish_immutable(execution, {'cases': ['goal', 'done']})
    before = execution.read_bytes()
    rows = {'goal': {'status': 'pending'}, 'done': {'status': 'verified'}}
    assert vibestudio_campaign.unfinished_wave([plan], tmp_path, 'release-sha', provenance, rows) is None
    assert execution.read_bytes() == before


def test_receipt_epoch_correspondence_cannot_silently_drop_attempts(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='original wave epoch'):
        vibestudio_campaign.publish_status(tmp_path, {}, {}, [{}], None, 2, {},
                                           attempt_epoch='current-release', receipt_epochs=[])
