from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))

import vibestudio_campaign


@pytest.mark.parametrize('sync_catalog', [True, False])
def test_drive_publishes_current_catalog_before_archival_work(tmp_path, monkeypatch, sync_catalog):
    catalog = tmp_path / 'catalog.jsonl'
    catalog.write_text('{"requiresReview":true}\n')
    monkeypatch.setattr(vibestudio_campaign, 'campaign_scope', lambda *_: (None, {}, {}))
    monkeypatch.setattr(vibestudio_campaign, 'attempt_context', lambda *_: ('epoch', {}, {}))
    monkeypatch.setattr(vibestudio_campaign, 'prerequisite_holds', lambda *_: {})
    monkeypatch.setenv('NATLANG_CAMPAIGN_SYNC_CATALOG', '1' if sync_catalog else '0')
    published = []
    def sync(_args, paths, *, checkpoint):
        published.append(list(paths))
        if paths == [catalog]:
            assert not (tmp_path / 'reused.json').exists()
            assert catalog.read_text() == '{"requiresReview":true}\n'
    monkeypatch.setattr(vibestudio_campaign, 'sync_index', sync)
    vibestudio_campaign.drive(SimpleNamespace(root=tmp_path, held=tmp_path/'held', seed=[], host=tmp_path,
        review_model=None, attempts=2, max_waves=0))
    assert ([catalog] in published) is sync_catalog
    assert published[-1] == [tmp_path/'status.json', tmp_path/'reused.json']


def test_new_release_fresh_coverage_precedes_historical_genuine_failures():
    cases = {'old': {'workspace': 'system', 'family': 'old'},
             'new-a': {'workspace': 'system', 'family': 'fresh-a'},
             'new-b': {'workspace': 'system', 'family': 'fresh-b'}}
    rows = {'old': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 2},
            'new-a': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 0},
            'new-b': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 0}}
    assert vibestudio_campaign.next_wave(cases, rows, 64) == ['new-a', 'new-b', 'old']
    assert vibestudio_campaign.next_wave(cases, rows, 64, strict_priority_waves=True) == ['new-a', 'new-b']
    assert vibestudio_campaign.next_wave(cases, rows, 64, epoch_first_retries=True) == ['old', 'new-a', 'new-b']


def test_partial_and_interrupted_training_do_not_complete_or_consume_cancelled_goal(tmp_path):
    cases = {'goal': {'workspace': 'system', 'family': 'draft'}}
    receipt = {'goal': {'status': 'partial', 'manifest': 'verified/manifest.json', 'interrupted': True}}
    rows = vibestudio_campaign.publish_status(tmp_path, cases, {}, [receipt], None, 2, {},
        attempt_epoch='release', receipt_epochs=['release'])
    assert rows['goal']['status'] == 'pending'
    assert rows['goal']['attempts'] == rows['goal']['cumulative_teacher_attempts'] == 0
    assert rows['goal']['interruptions'] == 1
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, cases, 1) is None
    assert vibestudio_campaign.has_training_manifest(receipt['goal'])


def test_partial_manifest_is_preserved_as_training_without_completing_goal(tmp_path, monkeypatch):
    manifest = tmp_path/'verified/manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{}')
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', lambda *_: {
        'task_id': 'goal-id', 'turns': [1], 'trajectory_ir': {'outcome': {'status': 'partial', 'task_completed': False}}})
    result = vibestudio_campaign.outcomes(tmp_path, ['goal'], None, {}, {'goal': {'taskId': 'goal-id'}})
    assert result['goal']['status'] == 'partial'
    assert result['goal']['manifest'] == 'verified/manifest.json'
    assert vibestudio_campaign.has_training_manifest(result['goal'])


def test_completion_reassessment_does_not_rewrite_immutable_original_receipt(tmp_path):
    import hashlib
    waves = tmp_path/'waves'
    jobs = waves/'jobs'
    jobs.mkdir(parents=True)
    receipt = waves/'0001-result.json'
    vibestudio_campaign.publish_immutable(receipt, {'goal': {'status': 'verified', 'manifest': 'verified/manifest.json'}})
    original = receipt.read_bytes()
    vibestudio_campaign.publish(tmp_path/'completion-reassessment.json', {
        'schema': 'natlang.campaign-completion-reassessment/1', 'errors': [], 'summary': {'unresolved': 0},
        'receipts': {receipt.name: {'receipt_sha256': hashlib.sha256(original).hexdigest(),
                                  'cases': {'goal': {'task_completed': False}}}}})
    receipts, _ = vibestudio_campaign.receipt_history(waves, jobs)
    assert receipts[0]['goal']['status'] == 'partial'
    assert receipt.read_bytes() == original
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='original receipt changed'):
        vibestudio_campaign.receipt_history(waves, jobs)


def test_collection_lifetime_and_nested_hub_budget_are_independent_of_wave_size():
    from types import SimpleNamespace
    args = SimpleNamespace(concurrency=8, wave_size=64)
    assert vibestudio_campaign.collection_limits(args, [{}] * 64) == {
        'cases_per_instance': 12, 'concurrency': 8}
    assert vibestudio_campaign.collection_limits(args, [{}, {'requiresDeploymentPackage': True}]) == {
        'cases_per_instance': 12, 'concurrency': 2}
    args.concurrency = 1
    assert vibestudio_campaign.collection_limits(args, [{'requiresDeploymentPackage': True}])['concurrency'] == 1
    args.cases_per_instance = 0
    args.deployment_concurrency = 8
    args.concurrency = 8
    assert vibestudio_campaign.collection_limits(args, [{'requiresDeploymentPackage': True}]) == {
        'cases_per_instance': 0, 'concurrency': 8}


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
    assert vibestudio_campaign.next_wave(cases, rows, 6) == ['reset', 'retry']
    assert vibestudio_campaign.next_wave(cases, rows, 6, strict_priority_waves=True) == ['reset']
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


@pytest.mark.parametrize("teacher_timeout", [3_600_000, 600_000])
@pytest.mark.parametrize("finish_archive_wave", [False, True])
def test_drive_retries_historical_exhaustion_in_new_immutable_wave(monkeypatch, tmp_path, teacher_timeout, finish_archive_wave):
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
    monkeypatch.setattr(vibestudio_campaign, 'sync_index', lambda *_, **__: None)
    monkeypatch.setattr(vibestudio_campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    archived = []
    monkeypatch.setattr(vibestudio_campaign, 'remote_archive', lambda *args, **kwargs: archived.append(kwargs['wave']))
    monkeypatch.setattr(vibestudio_campaign, 'outcomes', lambda _capture, names, *_:
        {name: {'status': 'test-failed', 'reason': 'genuine unsuccessful goal'} for name in names})
    def collect(_owner, command, **kwargs):
        assert command[:4] == ['node', '--import', 'tsx', 'scripts/collect-live-workflow-teacher.mjs']
        assert kwargs['env'] == {'PINNED': 'yes'}
        assert command[command.index('--checkpoint-request') + 1] == str(kwargs['request_file'])
        assert ('--drain-request' in command) is not finish_archive_wave
        if not finish_archive_wave:
            assert command[command.index('--drain-request') + 1] == str(tmp_path / 'stop-after-wave')
        assert command.count('--case') == 1 and command[-1] == 'retry'
        assert command[command.index('--test-timeout-ms') + 1] == str(teacher_timeout)
        assert command[command.index('--cases-per-instance') + 1] == '12'
        assert command[command.index('--concurrency') + 1] == '3'
        Path(command[command.index('--out') + 1]).mkdir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(vibestudio_campaign.CollectorCheckpoint, 'run', collect)
    args = SimpleNamespace(root=tmp_path, held=tmp_path / 'held', seed=[], host=tmp_path, archive=tmp_path / 'archive',
        review_model=None, attempts=2, max_waves=1, wave_size=6, concurrency=3, reserve_disk_gib=8,
        teacher_timeout_ms=teacher_timeout, finish_archive_wave=finish_archive_wave)
    vibestudio_campaign.drive(args)
    plan = vibestudio_campaign.read(jobs / '0003.json')
    assert plan == {'cases': ['retry'], 'attempt_epoch': 'new-release', 'provenance': provenance}
    execution = vibestudio_campaign.read(waves / '0003-execution.json')
    assert execution['teacher_timeout_ms'] == teacher_timeout
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


def test_concurrent_exports_register_and_publish_a_cohort_once(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    registering, release, second_started = Event(), Event(), Event()
    writers, registrations = [], []

    class Writer:
        def __init__(self, path):
            self.rows = 0
            writers.append(path)

        def write(self, row):
            self.rows += 1

        def close(self, **kwargs):
            pass

    def register(*args, **kwargs):
        registrations.append(args)
        registering.set()
        assert release.wait(3)
        return {'id': 'one-cohort'}

    def export_again():
        second_started.set()
        vibestudio_campaign.export_trajectories(tmp_path, receipt, set(), set())

    monkeypatch.setattr(vibestudio_campaign, 'ZstdWriter', Writer)
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', lambda *args: {'trajectory_ir': {'outcome': {'status': 'passed', 'task_completed': True}}, 'turns': [1]})
    monkeypatch.setattr(vibestudio_campaign.data_registry, 'record_dataset', register)
    receipt = {'goal': {'status': 'verified', 'manifest': 'goal/manifest.json'}}
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(vibestudio_campaign.export_trajectories, tmp_path, receipt, set(), set())
        try:
            assert registering.wait(3)
            second = pool.submit(export_again)
            assert second_started.wait(3)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert len(writers) == len(registrations) == 1
    assert vibestudio_campaign.read(tmp_path / 'exported.json') == {
        'verified': 1, 'dataset': 'one-cohort', 'completed_goals': 1,
        'partial_trajectories': 0, 'supervised_decisions': 1, 'evidence_only': 0}


def test_package_bootstrap_failures_do_not_use_teacher_budget_after_no_call_capture(tmp_path):
    reason = '[build.getBuild] No export ./a-new-helper found for @workspace-skills/a-domain'
    before = {'status': 'test-failed', 'reason': reason, 'teacher_started': False}
    after = {**before, 'teacher_started': True}
    unknown = {'status': 'test-failed', 'reason': reason}
    assert vibestudio_campaign.infrastructure_failure(before, {})
    assert not vibestudio_campaign.infrastructure_failure(after, {})
    assert not vibestudio_campaign.infrastructure_failure(unknown, {})
    cases = {'before': {'workspace': 'system', 'family': 'bootstrap'}, 'after': {'workspace': 'system', 'family': 'bootstrap'}}
    rows = vibestudio_campaign.publish_status(tmp_path, cases, {}, [{'before': before, 'after': after}], None, 2, {},
        attempt_epoch='release', receipt_epochs=['release'])
    assert rows['before']['attempts'] == 0
    assert rows['after']['attempts'] == 1
    assert rows['before']['epoch_infrastructure_failures'] == 1


def test_raw_capture_records_teacher_start_evidence_without_rewriting_it(tmp_path):
    import gzip
    import json
    for roots in [[], ['real-root']]:
        capture = tmp_path / ('before' if not roots else 'after')
        packet_dir = capture / 'st_example'
        packet_dir.mkdir(parents=True)
        raw = packet_dir / 'trajectory-example-full.json.gz'
        with gzip.open(raw, 'wt') as stream:
            json.dump({'result': {'passed': False}, 'execution': {'error': '[build.getBuild] Could not resolve a package',
                'diagnostics': {'natlangTeacher': {'rootCallIds': roots, 'calls': []}}}}, stream)
        original = raw.read_bytes()
        outcome = vibestudio_campaign.outcomes(capture, ['example'], None, {}, {'example': {}})['example']
        assert outcome['teacher_started'] is (True if roots else None)
        assert not vibestudio_campaign.infrastructure_failure(outcome, {})
        assert raw.read_bytes() == original


@pytest.mark.parametrize('partial_receipt_status', ['verified', 'partial'])
@pytest.mark.parametrize('verification_status', ['passed', 'partial'])
def test_export_counts_separate_completed_goals_partial_decisions_and_evidence_only(monkeypatch, tmp_path, partial_receipt_status, verification_status):
    episodes = [
        {'trajectory_ir': {'outcome': {'status': 'passed', 'task_completed': True}}, 'turns': [1, 2]},
        {'trajectory_ir': {'outcome': {'status': verification_status, 'task_completed': False}}, 'turns': [1]},
        {'trajectory_ir': {'outcome': {'status': 'passed', 'task_completed': True}}, 'turns': []},
    ]
    def episode(path, *args):
        return episodes[int(path.parent.name)]
    registered = []
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', episode)
    monkeypatch.setattr(vibestudio_campaign.data_registry, 'record_dataset',
                        lambda *args, **kwargs: registered.append(kwargs) or {'id': 'mixed'})
    receipt = {str(index): {'status': 'verified', 'manifest': f'{index}/manifest.json'} for index in range(3)}
    receipt['1']['status'] = partial_receipt_status
    vibestudio_campaign.export_trajectories(tmp_path, receipt, set(), set())
    result = vibestudio_campaign.read(tmp_path / 'exported.json')
    assert result == {'verified': 3, 'dataset': 'mixed', 'completed_goals': 2, 'partial_trajectories': 1,
                      'supervised_decisions': 3, 'evidence_only': 1}
    assert vibestudio_campaign.read(tmp_path / 'graph-metrics.json')['categories']['partial']['captures'] == 1
    assert registered[0]['rows'] == 3
    assert registered[0]['extra']['trajectory_counts'] == {key: result[key] for key in
        ('completed_goals', 'partial_trajectories', 'supervised_decisions', 'evidence_only')}


def test_partial_training_capture_cannot_seed_a_completed_campaign_goal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    partial = tmp_path / 'partial.json'
    vibestudio_campaign.publish(tmp_path / 'reused.json', {'goal': str(partial)})
    monkeypatch.setattr(vibestudio_campaign, 'campaign_scope', lambda *a: (set(), set(), {'goal': {'taskId': 'goal'}}))
    monkeypatch.setattr(vibestudio_campaign, 'attempt_context', lambda *a: ('epoch', {}, {}))
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', lambda *a: {
        'task_id': 'goal', 'trajectory_ir': {'outcome': {'status': 'partial', 'task_completed': False}}})
    with pytest.raises(ValueError, match='partial training evidence cannot complete'):
        vibestudio_campaign.drive(SimpleNamespace(root=tmp_path, held=tmp_path/'held', seed=[]))
    assert not (tmp_path / 'status.json').exists()


def test_inventory_partial_does_not_reserve_the_later_completed_goal(tmp_path, monkeypatch):
    import json
    import inventory_live_teacher as inventory
    paths = [tmp_path / name / 'manifest.json' for name in ('a-partial', 'b-completed')]
    for path in paths:
        path.parent.mkdir()
        path.write_text(json.dumps({'schema': 'natlang.live-teacher-task/1', 'id': 'goal', 'family': 'domain', 'split': 'train'}))
    task = {'id': 'goal', 'source_ids': ['goal'], 'source_groups': ['group'], 'interaction': {'goal': {'request': 'save'}}}
    monkeypatch.setattr(inventory, 'load_episode', lambda path: {
        'task': task, 'teacher': {'model_id': 'luna'}, 'turns': [1], 'call_tree': [],
        'trajectory_ir': {'outcome': {'task_completed': path == paths[1]}}})
    monkeypatch.setattr(inventory.data_registry, 'load', lambda: type('Registry', (), {
        'find': lambda self, path: {'id': 'dataset', 'admission': 'train'}})())
    monkeypatch.setattr(inventory, 'held_hit', lambda *a: False)
    monkeypatch.setattr(inventory, 'require_registered_open_goal', lambda *a: None)
    result = inventory.inventory([tmp_path], set(), {}, 'luna')
    assert [row['status'] for row in result['manifests']] == ['ready-for-partial-extraction', 'ready-for-extraction']
    assert [row['task_completed'] for row in result['manifests']] == [False, True]


def test_graph_telemetry_retains_failed_and_missing_captures_separately(tmp_path):
    import gzip
    import json
    import hashlib
    raw = tmp_path / 'capture' / 'failed.json.gz'
    raw.parent.mkdir()
    records = [{'record': {'call_id': 'root', 'parent_call_id': None, 'program_root': 'skills/browser',
                           'definition': {'name': 'inspect', 'source': 'inspect.nl'}}},
               {'record': {'call_id': 'child', 'parent_call_id': 'root', 'program_root': 'skills/editor',
                           'definition': {'name': 'change', 'source': 'change.nl'}}}]
    with gzip.open(raw, 'wt') as stream:
        json.dump({'execution': {'diagnostics': {'natlangTeacher': {'calls': records}}}}, stream)
    summary = vibestudio_campaign.write_graph_telemetry(tmp_path, {
        'failed': {'status': 'test-failed', 'raw': 'failed.json.gz'}, 'absent': {'status': 'no-capture'}})
    assert summary['categories']['preserved_failure']['captures'] == 2
    assert summary['categories']['preserved_failure']['invocations'] == 2
    assert summary['categories']['preserved_failure']['captures_without_graph'] == 1
    assert summary['categories']['completed']['captures'] == 0
    saved = vibestudio_campaign.read(tmp_path / 'graph-metrics.json')
    assert saved['captures'][0]['metrics']['max_depth'] == 1
    assert saved['captures'][0]['source']['sha256'] == hashlib.sha256(raw.read_bytes()).hexdigest()


def test_mixed_wave_holds_only_repeated_typed_infrastructure_failures():
    receipt = {'done': {'status': 'verified'}, 'missing-a': {'status': 'no-capture'},
               'missing-b': {'status': 'no-capture'}, 'wrong': {'status': 'test-failed', 'reason': 'wrong contents'}}
    hold = vibestudio_campaign.wave_infrastructure_hold(receipt, {})
    assert hold == {'infrastructure_failures': ['missing-a', 'missing-b'], 'infrastructure_hold_threshold': 2,
                    'structural_failures': ['missing-a', 'missing-b'], 'provider_transport_failures': [],
                    'provider_transport_failure_fraction': 0, 'provider_transport_hold_fraction': 0.5,
                    'noncancelled_attempts': 4, 'workerd_resets': [], 'all_captures_missing': False}
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}, 0) is None
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}, 3) is None
    assert vibestudio_campaign.wave_infrastructure_hold({'wrong-a': receipt['wrong'], 'wrong-b': receipt['wrong']}, {}) is None
    assert vibestudio_campaign.wave_infrastructure_hold({}, {}) is None


def test_storage_capture_errors_are_retained_and_classified_as_infrastructure(tmp_path):
    import gzip
    import json
    directory = tmp_path / 'st_owned'; directory.mkdir()
    errors = [{'callId': 'root', 'message': 'ENOSPC writing model exchange'}]
    with gzip.open(directory / 'trajectory-owned-full.json.gz', 'wt') as stream:
        json.dump({'result': {'passed': False}, 'execution': {'diagnostics': {'natlangTeacher': {
            'rootCallIds': ['root'], 'calls': [], 'captureErrors': errors}}}}, stream)
    result = vibestudio_campaign.outcomes(tmp_path, ['owned'], None, {}, {'owned': {}})['owned']
    assert result['capture_errors'] == errors
    assert vibestudio_campaign.infrastructure_failure(result, {})
    assert vibestudio_campaign.wave_infrastructure_hold({'owned': result}, {}, 1)['infrastructure_failures'] == ['owned']
    assert vibestudio_campaign.wave_infrastructure_hold({'owned': result}, {}, 0) is None


def test_recorded_teacher_failure_is_not_capture_corruption(tmp_path):
    import gzip
    import json
    directory = tmp_path / 'st_owned'; directory.mkdir()
    errors = ['Assigned authored root did not complete successfully']
    with gzip.open(directory / 'trajectory-owned-full.json.gz', 'wt') as stream:
        json.dump({'result': {'passed': False}, 'execution': {'diagnostics': {'natlangTeacher': {
            'rootCallIds': ['root'], 'calls': [], 'captureErrors': errors,
            'executionError': {'message': 'quiesced: requested operation could not be completed'}}}}}, stream)
    result = vibestudio_campaign.outcomes(tmp_path, ['owned'], None, {}, {'owned': {}})['owned']
    assert result['capture_errors'] == errors
    assert result['teacher_started'] is True
    assert result['status'] == 'test-failed'
    assert not vibestudio_campaign.infrastructure_failure(result, {})
    assert vibestudio_campaign.wave_infrastructure_hold({'owned': result}, {}, 1) is None
    result['capture_errors'].append({'message': 'ENOSPC writing model exchange'})
    assert vibestudio_campaign.infrastructure_failure(result, {})
    assert vibestudio_campaign.wave_infrastructure_hold({'owned': result}, {}, 1)


def test_missing_teacher_wrapper_means_unknown_start_not_unexecuted(tmp_path):
    import gzip
    import json
    capture = tmp_path / 'st_case'
    capture.mkdir()
    raw = capture / 'trajectory-example-full.json.gz'
    with gzip.open(raw, 'wt') as stream:
        json.dump({'result': {'passed': False, 'reason': 'Owned operation cancelled at the diagnostic deadline'},
                   'execution': {'diagnostics': {}}}, stream)
    original = raw.read_bytes()
    outcome = vibestudio_campaign.outcomes(tmp_path, ['example'], None, {}, {'example': {}})['example']
    assert outcome['teacher_started'] is None
    assert outcome['status'] == 'test-failed'
    assert raw.read_bytes() == original


def test_cohort_exports_are_resumable_for_ledger_backfill(monkeypatch, tmp_path):
    from types import SimpleNamespace
    (tmp_path / 'waves').mkdir()
    args = SimpleNamespace(root=tmp_path, archive_host='dgx', archive_repo=Path('/repo'),
        archive=Path('/archive'), archive_held=Path('/held.json'), defer_export=False)
    monkeypatch.setattr(vibestudio_campaign, 'transfer', lambda *_, **__: None)
    commands, probes = [], []
    def run(command, **_):
        commands.append(command)
        if command[0] == 'ssh' and command[-1].startswith('test -f '):
            probes.append(command)
            return SimpleNamespace(returncode=1 if len(probes) == 1 else 0)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    vibestudio_campaign.remote_archive(args, Path('/archive/0001'), {'goal': {'status': 'verified'}}, tmp_path/'capture')
    launched = next(command[-1] for command in commands if command[0] == 'ssh' and 'memory_ledger.py' in command[-1])
    assert '--resumable' in launched
    assert '--no-resumable' not in launched
    assert '--budget-gb 1.5' in launched and '--reserve-gb 15' in launched
    assert 'NATLANG_REPO=/repo' in launched
    assert '/repo/scripts/neuralese_data/vibestudio_campaign.py' in launched
    assert '/archive/vibestudio_campaign.py' not in launched


@pytest.mark.parametrize('cause', [
    'fetch failed',
    'upstream HTTP 502',
    'upstream HTTP 429',
    'Circuit breaker is open; last failure: POST https://chatgpt.com/backend-api/codex/responses: upstream HTTP 503',
    'Circuit breaker is open; last failure: POST https://chatgpt.com/backend-api/codex/responses: fetch failed',
    'Circuit breaker is open; last failure: GET https://chatgpt.com/backend-api/codex/responses: Client network socket disconnected before secure TLS connection was established',
    'connect ECONNRESET',
    'getaddrinfo EAI_AGAIN chatgpt.com',
    'Circuit breaker is open; last failure: POST https://chatgpt.com/backend-api/codex/responses: getaddrinfo EAI_AGAIN chatgpt.com',
])
def test_provider_transport_errors_do_not_spend_goal_retries(tmp_path, cause):
    outcome = {'status': 'test-failed', 'teacher_model_sampled': True,
               'reason': 'natlang chooseScope: failed: ' + cause + ' (record real-call/2)'}
    assert vibestudio_campaign.infrastructure_failure(outcome, {})
    cases = {'goal': {'workspace': 'system', 'family': 'scope'}}
    rows = vibestudio_campaign.publish_status(tmp_path, cases, {}, [{'goal': outcome}], None, 2, {},
        attempt_epoch='release', receipt_epochs=['release'])
    assert rows['goal']['attempts'] == 0
    assert rows['goal']['cumulative_teacher_attempts'] == 1
    assert rows['goal']['epoch_infrastructure_failures'] == 1
    assert rows['goal']['status'] == 'pending'
    assert vibestudio_campaign.wave_infrastructure_hold({'goal': outcome}, cases, 1)


@pytest.mark.parametrize('reason', [
    'fetch failed',
    'natlang describeIncident: quiesced: fetch failed (record reply/2)',
    'natlang describeIncident: failed: The document says fetch failed',
    'natlang chooseScope: failed: schema validation failed: fetch failed',
    'natlang chooseScope: failed: unauthorized',
    'natlang describeIncident: failed: The document says upstream HTTP 502',
    'natlang chooseScope: failed: schema validation failed: upstream HTTP 502',
    'natlang chooseScope: failed: upstream HTTP 400',
    'natlang chooseScope: failed: Circuit breaker is open',
    'natlang chooseScope: failed: The document says getaddrinfo EAI_AGAIN chatgpt.com',
    'natlang chooseScope: failed: schema validation failed: getaddrinfo EAI_AGAIN chatgpt.com',
])
def test_described_network_failures_are_not_operational_transport_errors(reason):
    assert not vibestudio_campaign.infrastructure_failure({'status': 'test-failed', 'reason': reason}, {})


def test_provider_transport_accounting_opt_out(monkeypatch):
    monkeypatch.setenv('NATLANG_CAMPAIGN_TRANSPORT_FAILURES_AS_GOAL_FAILURES', '1')
    assert not vibestudio_campaign.infrastructure_failure(
        {'status': 'test-failed', 'reason': 'natlang chooseScope: failed: fetch failed'}, {})


@pytest.mark.parametrize('command,code', [(['ssh', 'host', 'mkdir'], 255), (['rsync', 'source', 'destination'], 12)])
def test_archive_network_retries_do_not_repeat_teacher_work(monkeypatch, command, code):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(code, argv)
        return subprocess.CompletedProcess(argv, 0, stdout='verified')
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    monkeypatch.setattr(vibestudio_campaign.time, 'sleep', lambda _: None)
    assert vibestudio_campaign.archive_transport(command, capture_output=True).stdout == 'verified'
    assert calls == [command, command]


@pytest.mark.parametrize('code', [11, 23, 24])
def test_archive_integrity_and_file_errors_are_not_retried(monkeypatch, code):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(code, argv)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.archive_transport(['rsync'])
    assert len(calls) == 1


def test_archive_transport_retries_are_bounded_and_can_be_disabled(monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        raise subprocess.CalledProcessError(255, argv)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    monkeypatch.setattr(vibestudio_campaign.time, 'sleep', lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.archive_transport(['ssh'])
    assert len(calls) == 4
    monkeypatch.setenv('NATLANG_ARCHIVE_TRANSPORT_ATTEMPTS', '1')
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.archive_transport(['ssh'])
    assert len(calls) == 5


def test_archive_checksum_mismatch_keeps_original_capture(monkeypatch, tmp_path):
    source = tmp_path/'capture'
    source.mkdir()
    original = source/'original.json'
    original.write_text('{"kept":true}')
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='>fcsT...... original.json' if '--dry-run' in argv else '')
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    monkeypatch.setattr(vibestudio_campaign, 'prepare_retained_state_packs', lambda _: [])
    with pytest.raises(RuntimeError, match='archive differs after transfer'):
        vibestudio_campaign.transfer(source, 'dgx', Path('/archive/capture'))
    assert len(commands) == 3
    assert original.read_text() == '{"kept":true}'


def transport_wave(failures, attempts):
    return {**{f'network-{i}': {'status': 'test-failed',
                'reason': 'natlang decide: failed: fetch failed (record actual/2)'}
               for i in range(failures)},
            **{f'other-{i}': {'status': 'test-failed', 'reason': 'wrong user outcome'}
               for i in range(attempts - failures)}}


def test_sparse_provider_failures_keep_large_cohort_running():
    receipt = transport_wave(9, 64)
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}) is None
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}, provider_transport_fraction=0)
    assert all(vibestudio_campaign.infrastructure_failure(value, {})
               for name, value in receipt.items() if name.startswith('network-'))


def test_provider_wave_fraction_boundary_and_minimum_count():
    assert vibestudio_campaign.wave_infrastructure_hold(transport_wave(2, 5), {}) is None
    hold = vibestudio_campaign.wave_infrastructure_hold(transport_wave(2, 4), {})
    assert hold['provider_transport_failure_fraction'] == 0.5
    assert hold['noncancelled_attempts'] == 4
    assert hold['structural_failures'] == []
    assert vibestudio_campaign.wave_infrastructure_hold(transport_wave(1, 1), {}) is None
    assert vibestudio_campaign.wave_infrastructure_hold(transport_wave(2, 2), {}, 0) is None


def test_cancelled_attempts_neither_trigger_nor_dilute_provider_hold():
    receipt = transport_wave(2, 4)
    receipt['cancelled'] = {'status': 'interrupted'}
    receipt['cancelled-raw'] = {'status': 'no-capture', 'interrupted': True}
    hold = vibestudio_campaign.wave_infrastructure_hold(receipt, {})
    assert hold['noncancelled_attempts'] == 4
    assert len(hold['infrastructure_failures']) == 2
    receipt['network-0']['interrupted'] = True
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}) is None


def test_structural_damage_remains_strict_even_with_transport_cause():
    receipt = transport_wave(2, 64)
    receipt['network-0']['capture_errors'] = ['missing call record']
    receipt['network-1']['capture_errors'] = ['missing call record']
    hold = vibestudio_campaign.wave_infrastructure_hold(receipt, {})
    assert hold['structural_failures'] == ['network-0', 'network-1']
    assert hold['provider_transport_failures'] == []


def test_mixed_single_structural_and_sparse_transport_only_hold_in_legacy_mode():
    receipt = transport_wave(1, 64)
    receipt['missing'] = {'status': 'no-capture'}
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}) is None
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {}, provider_transport_fraction=0)
    receipt['second-missing'] = {'status': 'no-capture'}
    assert vibestudio_campaign.wave_infrastructure_hold(receipt, {})


@pytest.mark.parametrize('fraction', [-0.1, 1.1, float('nan')])
def test_invalid_provider_hold_fraction_rejected(fraction):
    with pytest.raises(ValueError, match='fraction'):
        vibestudio_campaign.wave_infrastructure_hold({}, {}, provider_transport_fraction=fraction)


def wave_candidates(spec):
    cases, rows = {}, {}
    for workspace, review, exposures in spec:
        for index, exposure in enumerate(exposures):
            name = f'{workspace}-{review}-{index}'
            cases[name] = {'workspace': workspace, 'requiresReview': review, 'family': f'family-{index % 3}'}
            rows[name] = {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': exposure}
    return cases, rows


def test_full_fresh_group_precedes_tiny_fresh_group_tail():
    cases, rows = wave_candidates([('spectrolite', False, [0]*2), ('system', False, [0]*64)])
    result = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)
    assert len(result) == 64
    assert {cases[name]['workspace'] for name in result} == {'system'}
    assert len(vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8, strict_priority_waves=True)) == 2


def test_same_group_tail_uses_next_ranks_without_displacing_fresh_tasks():
    cases, rows = wave_candidates([('system', False, [2, 0, 1, 0, 3, 1])])
    result = vibestudio_campaign.next_wave(cases, rows, 5, concurrency=4)
    assert [rows[name]['cumulative_teacher_attempts'] for name in result] == [0, 0, 1, 1]


def test_retried_full_group_cannot_displace_any_fresh_group():
    cases, rows = wave_candidates([('system', False, [1]*64), ('spectrolite', True, [0]*2)])
    result = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)
    assert len(result) == 2
    assert {cases[name]['workspace'] for name in result} == {'spectrolite'}


def test_full_groups_preserve_workspace_exposure_fairness_and_homogeneity():
    cases, rows = wave_candidates([('system', False, [0]*8), ('personal', True, [0]*8)])
    cases['done'] = {'workspace':'system','family':'other'}
    rows['done'] = {'status':'verified','attempts':1,'cumulative_teacher_attempts':4}
    result = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)
    assert {(cases[name]['workspace'], cases[name].get('requiresReview',False)) for name in result} == {('personal',True)}
    assert 'done' not in result
    for name in result:rows[name]['status']='verified'
    assert len(vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)) == 8


def test_workers_are_actual_fill_target_and_smallest_tails_do_not_win():
    cases, rows = wave_candidates([('a', False, [0]*2), ('b', False, [0]*7), ('c', False, [0]*8)])
    assert {cases[name]['workspace'] for name in vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)} == {'c'}
    for name in list(rows):
        if cases[name]['workspace']=='c':rows[name]['status']='verified'
    assert len(vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)) == 7
    assert {cases[name]['workspace'] for name in vibestudio_campaign.next_wave(cases, rows, 64, concurrency=2)} == {'a'}


def test_same_workspace_review_groups_stay_separate_and_families_rotate():
    cases, rows = wave_candidates([('system', False, [0]*3), ('system', True, [0]*9)])
    result = vibestudio_campaign.next_wave(cases, rows, 8, concurrency=8)
    assert all(cases[name]['requiresReview'] for name in result)
    assert [cases[name]['family'] for name in result[:3]] == ['family-0','family-1','family-2']
    assert len(set(result)) == len(result) == 8


def test_fresh_group_fill_can_cross_epoch_attempt_bands_without_preferring_retries():
    cases, rows = wave_candidates([('a', False, [0]*2), ('b', False, [0]*8)])
    for name, row in rows.items():
        if cases[name]['workspace']=='b':row['attempts']=1
    result = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8)
    assert {cases[name]['workspace'] for name in result} == {'b'}


def test_tail_fallback_only_fills_workers_without_padding_with_retries():
    cases, rows = wave_candidates([('a',False,[0]*2+[1]*62),('b',False,[0]*1)])
    result = vibestudio_campaign.next_wave(cases,rows,64,concurrency=8)
    assert len(result)==8
    assert [rows[name]['cumulative_teacher_attempts'] for name in result]==[0,0]+[1]*6
    cases, rows = wave_candidates([('a',False,[0]*9+[1]*55)])
    result = vibestudio_campaign.next_wave(cases,rows,64,concurrency=8)
    assert len(result)==9
    assert all(rows[name]['cumulative_teacher_attempts']==0 for name in result)


def test_one_deployment_case_does_not_throttle_ordinary_wave():
    cases, rows = wave_candidates([('system', False, [0] * 10)])
    nested = list(cases)[-1]
    cases[nested]['requiresDeploymentPackage'] = True
    selected = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4)
    assert len(selected) == 9
    assert nested not in selected
    limits = SimpleNamespace(concurrency=4, deployment_concurrency=2, cases_per_instance=8)
    assert vibestudio_campaign.collection_limits(limits, [cases[name] for name in selected])['concurrency'] == 4
    for name in selected:
        rows[name]['status'] = 'verified'
    assert vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4) == [nested]
    assert vibestudio_campaign.collection_limits(limits, [cases[nested]])['concurrency'] == 2


def test_fresh_deployment_goals_precede_full_ordinary_retry_group():
    cases, rows = wave_candidates([('system', False, [1] * 8 + [0] * 2)])
    for name in list(cases)[-2:]:
        cases[name]['requiresDeploymentPackage'] = True
    selected = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4)
    assert len(selected) == 2
    assert all(cases[name].get('requiresDeploymentPackage') for name in selected)


def test_deployment_tail_only_fills_its_effective_worker_limit():
    cases, rows = wave_candidates([('system', False, [0] + [1] * 8)])
    for case in cases.values():
        case['requiresDeploymentPackage'] = True
    selected = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8, deployment_concurrency=2)
    assert len(selected) == 2
    assert [rows[name]['cumulative_teacher_attempts'] for name in selected] == [0, 1]
    assert len(vibestudio_campaign.next_wave(cases, rows, 64, concurrency=8, deployment_concurrency=3)) == 3


def test_resource_selection_uses_actual_slots_and_preserves_all_goals():
    cases, rows = wave_candidates([('system', False, [0] * 5)])
    for name in list(cases)[-2:]:
        cases[name]['requiresDeploymentPackage'] = True
    first = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4)
    assert len(first) == 3  # Three ordinary workers beat a full two-worker deployment group.
    assert not any(cases[name].get('requiresDeploymentPackage') for name in first)
    for name in first:
        rows[name]['status'] = 'verified'
    second = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4)
    assert len(second) == 2
    assert set(first + second) == set(cases)


def test_mixed_deployment_optout_restores_whole_wave_limit():
    cases, rows = wave_candidates([('system', False, [0] * 9)])
    cases[list(cases)[-1]]['requiresDeploymentPackage'] = True
    selected = vibestudio_campaign.next_wave(cases, rows, 64, concurrency=4, mixed_deployment_waves=True)
    assert len(selected) == 9
    limits = SimpleNamespace(concurrency=4, deployment_concurrency=2)
    assert vibestudio_campaign.collection_limits(limits, [cases[name] for name in selected])['concurrency'] == 2


def captured_transport_tree(kind='generation'):
    return {
        'rootCallIds': ['root'], 'rootOutcome': 'error', 'captureErrors': [],
        'executionError': {'name': 'NatlangCallError', 'message': 'natlang outer: quiesced: dependent helper unavailable'},
        'calls': [
            {'record': {'call_id': 'root', 'parent_call_id': None, 'outcome': 'error',
                        'definition': {'key': 'outer'}}},
            {'record': {'call_id': 'child', 'parent_call_id': 'root', 'outcome': 'failed',
                        'definition': {'key': 'read'}, 'model_turns': [
                            {'kind': kind, 'error': {'name': 'Error', 'message': 'fetch failed'}}]}},
        ],
    }


@pytest.mark.parametrize('kind', ['generation', 'score'])
def test_recorded_provider_error_survives_quiesced_caller(kind):
    teacher = captured_transport_tree(kind)
    evidence = vibestudio_campaign.captured_provider_transport_errors(teacher)
    assert len(evidence) == 1 and evidence[0]['call_id'] == 'child' and evidence[0]['kind'] == kind
    assert len(evidence[0]['error_sha256']) == 64
    outcome = {'status': 'test-failed', 'reason': teacher['executionError']['message'],
               'captured_provider_transport_errors': evidence}
    assert vibestudio_campaign.infrastructure_failure(outcome, {})
    assert vibestudio_campaign.provider_transport_failure({**outcome, 'status': 'partial'})
    assert not vibestudio_campaign.provider_transport_failure({**outcome, 'status': 'verified'})


@pytest.mark.parametrize('change', ['returned-text', 'recovered-exchange', 'unrelated-root', 'completed-root',
                                  'contradictory-root', 'invalid-capture'])
def test_only_unrecovered_native_failed_descendants_supply_provider_evidence(change):
    teacher = captured_transport_tree()
    child = teacher['calls'][1]['record']
    if change == 'returned-text':
        child['model_turns'] = [{'kind': 'generation', 'reply': {'text': 'fetch failed'}}]
        child['detail'] = 'fetch failed'
    elif change == 'recovered-exchange':
        child['model_turns'].append({'kind': 'generation', 'reply': {'text': 'actual answer'}})
    elif change == 'unrelated-root':
        child['parent_call_id'] = None
    elif change == 'completed-root':
        teacher['rootOutcome'] = 'done'
        teacher['calls'][0]['record']['outcome'] = 'done'
    elif change == 'contradictory-root':
        teacher['calls'][0]['record']['outcome'] = 'done'
    else:
        teacher['captureErrors'] = ['Missing actual ancestor']
    assert vibestudio_campaign.captured_provider_transport_errors(teacher) == []


def test_successful_recorded_retry_excludes_recovered_provider_fault():
    teacher = captured_transport_tree()
    teacher['calls'][1]['annotations'] = [{'kind': 'retried', 'source': 'runtime', 'value': {'by': 'retry'}}]
    teacher['calls'].append({'record': {'call_id': 'retry', 'parent_call_id': 'root', 'outcome': 'done',
                                      'definition': {'key': 'read'}}})
    assert vibestudio_campaign.captured_provider_transport_errors(teacher) == []
    teacher['calls'][2]['record']['definition']['key'] = 'unrelated'
    assert len(vibestudio_campaign.captured_provider_transport_errors(teacher)) == 1


def test_checked_capture_does_not_fall_back_to_model_shaped_failure_text():
    outcome = {'status': 'test-failed', 'reason': 'natlang caller: error: fetch failed',
               'captured_provider_transport_errors': []}
    assert not vibestudio_campaign.provider_transport_failure(outcome)
    legacy = {key: value for key, value in outcome.items() if key != 'captured_provider_transport_errors'}
    assert vibestudio_campaign.provider_transport_failure(legacy)


def test_captured_provider_accounting_preserves_failed_goal_and_real_exposure(tmp_path):
    teacher = captured_transport_tree()
    teacher['calls'][1]['record']['cost'] = {'model_requests': 1}
    teacher['calls'][1]['record']['model_turns'].insert(0, {'kind': 'generation', 'reply': {'text': 'prior real sample'}})
    packet = {'result': {'passed': False, 'reason': 'Requested explanation missing'},
              'execution': {'diagnostics': {'natlangTeacher': teacher}}}
    raw = tmp_path / 'st_actual' / 'trajectory-goal-full.json.gz'
    raw.parent.mkdir()
    with gzip.open(raw, 'wt') as stream:
        json.dump(packet, stream)
    before = raw.read_bytes()
    outcome = vibestudio_campaign.outcomes(tmp_path, ['goal'], None, {}, {'goal': {}})['goal']
    assert outcome['status'] == 'test-failed'
    assert outcome['teacher_model_sampled'] is True
    assert outcome['reason'] == teacher['executionError']['message']
    assert vibestudio_campaign.infrastructure_failure(outcome, {})
    assert raw.read_bytes() == before


@pytest.mark.parametrize('command,code', [(['ssh', 'archive-host'], 255), (['rsync', 'owned-source'], 12)])
def test_standing_archive_outage_retries_past_bounded_window_without_teacher(monkeypatch, tmp_path, command, code):
    from types import SimpleNamespace
    calls = []
    def owned_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, code if len(calls) <= 6 else 0, stdout='verified')
    owner = SimpleNamespace(requested=False, run=owned_run)
    monkeypatch.setattr(vibestudio_campaign.time, 'sleep', lambda _: None)
    original = tmp_path/'retained-capture.json'; original.write_bytes(b'actual completed capture')
    health = tmp_path/'archive-health.json'
    result = vibestudio_campaign.archive_transport(command, checkpoint=owner, health_path=health,
                                                   capture_output=True, input='bound inventory')
    assert result.stdout == 'verified' and len(calls) == 7
    assert all(kwargs['input'] == 'bound inventory' for _, kwargs in calls)
    assert vibestudio_campaign.read(health)['phase'] == 'archive-connected'
    assert original.read_bytes() == b'actual completed capture'


def test_owned_archive_checkpoint_preserves_capture_without_failed_verdict(monkeypatch, tmp_path):
    from types import SimpleNamespace
    owner = SimpleNamespace(requested=False)
    calls = []
    def owned_run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 4:
            owner.requested = True
        return subprocess.CompletedProcess(argv, 255)
    owner.run = owned_run
    monkeypatch.setattr(vibestudio_campaign.time, 'sleep', lambda _: None)
    retained = tmp_path/'raw.json'; retained.write_bytes(b'original')
    with pytest.raises(vibestudio_campaign.CheckpointRequested):
        vibestudio_campaign.archive_transport(['ssh'], checkpoint=owner)
    assert len(calls) == 4 and retained.read_bytes() == b'original'
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('code', [11, 23, 24])
def test_owned_durable_archive_does_not_retry_integrity_failure(monkeypatch, code):
    from types import SimpleNamespace
    calls = []
    def owned_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, code)
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.archive_transport(['rsync'], checkpoint=SimpleNamespace(requested=False, run=owned_run))
    assert len(calls) == 1


def test_standing_durable_archive_optout_is_bounded(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('NATLANG_CAMPAIGN_DURABLE_ARCHIVE', '0')
    monkeypatch.setattr(vibestudio_campaign.time, 'sleep', lambda _: None)
    calls = []
    def owned_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 255)
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.archive_transport(['ssh'], checkpoint=SimpleNamespace(requested=False, run=owned_run))
    assert len(calls) == 4


def test_ambiguous_remote_job_submission_is_never_retried(monkeypatch, tmp_path):
    from types import SimpleNamespace
    (tmp_path/'waves').mkdir()
    args = SimpleNamespace(root=tmp_path, archive_host='dgx', archive_repo=Path('/repo'),
        archive=Path('/archive'), archive_held=Path('/held'), defer_export=False)
    monkeypatch.setattr(vibestudio_campaign, 'transfer', lambda *_, **__: None)
    launched = []
    def submit(argv, **kwargs):
        if 'memory_ledger.py' in argv[-1]:
            launched.append(argv)
            raise subprocess.CalledProcessError(255, argv)
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', submit)
    owner = SimpleNamespace(requested=False, run=lambda argv, **_: subprocess.CompletedProcess(argv, 0))
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.remote_archive(args, Path('/archive/0001'), {'goal': {'status': 'verified'}},
                                           tmp_path/'capture', checkpoint=owner)
    assert len(launched) == 1


def test_archive_reconnect_clears_previous_owner_network_wait(tmp_path):
    from types import SimpleNamespace
    health = tmp_path/'archive-health.json'
    vibestudio_campaign.publish(health, {'phase': 'archive-network-wait', 'transport_failures': 9})
    owner = SimpleNamespace(requested=False, run=lambda argv, **_: subprocess.CompletedProcess(argv, 0))
    vibestudio_campaign.archive_transport(['ssh'], checkpoint=owner, health_path=health)
    assert vibestudio_campaign.read(health)['phase'] == 'archive-connected'


def test_one_invalid_capture_preserves_other_wave_results_and_raw_artifacts(tmp_path, monkeypatch):
    for name in ['bad', 'good']:
        path = tmp_path/'cases'/name/'verified/manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*.json')}
    def load(path):
        if path.parent.parent.name == 'bad':
            raise ValueError('observed workspace call graph differs from recorded child inventory')
        return {'task_id': 'good-id', 'task': {'id': 'good-id', 'interaction': {'goal': 'owned'},
                'source_ids': [], 'source_groups': []}, 'teacher': {'model_id': vibestudio_campaign.LUNA},
                'turns': [1], 'trajectory_ir': {'outcome': {'task_completed': True}}}
    monkeypatch.setattr(vibestudio_campaign, 'load_episode', load)
    monkeypatch.setattr(vibestudio_campaign, 'held_hit', lambda *_: False)
    monkeypatch.setattr(vibestudio_campaign, 'require_registered_open_goal', lambda *_: None)
    cases = {'bad': {'taskId': 'bad-id'}, 'good': {'taskId': 'good-id'}}
    rows = vibestudio_campaign.outcomes(tmp_path, ['bad', 'good'], None, {}, cases)
    assert rows['bad']['status'] == 'export-rejected'
    assert rows['bad']['rejected_manifest'] == 'cases/bad/verified/manifest.json'
    assert vibestudio_campaign.infrastructure_failure(rows['bad'], {})
    assert not vibestudio_campaign.has_training_manifest(rows['bad'])
    assert rows['good']['status'] == 'verified'
    assert all(Path(path).read_bytes() == content for path, content in before.items())


def test_capture_isolation_does_not_hide_held_policy_or_task_identity_errors(tmp_path, monkeypatch):
    path = tmp_path/'verified/manifest.json'
    path.parent.mkdir(); path.write_text('{}')
    episode = {'task_id': 'actual', 'task': {'id': 'actual', 'interaction': {'goal': 'owned'},
        'source_ids': [], 'source_groups': []}, 'teacher': {'model_id': vibestudio_campaign.LUNA}}
    monkeypatch.setattr(vibestudio_campaign, 'load_episode', lambda *_: episode)
    monkeypatch.setattr(vibestudio_campaign, 'held_hit', lambda *_: True)
    with pytest.raises(ValueError, match='held closure'):
        vibestudio_campaign.outcomes(tmp_path, ['goal'], None, {}, {'goal': {'taskId': 'other'}})
    monkeypatch.setattr(vibestudio_campaign, 'held_hit', lambda *_: False)
    monkeypatch.setattr(vibestudio_campaign, 'require_registered_open_goal', lambda *_: None)
    with pytest.raises(ValueError, match='another task'):
        vibestudio_campaign.outcomes(tmp_path, ['goal'], None, {}, {'goal': {'taskId': 'other'}})
    episode['teacher']['model_id'] = 'disallowed-other-teacher'
    with pytest.raises(ValueError, match='actual Luna'):
        vibestudio_campaign.outcomes(tmp_path, ['goal'], None, {}, {'goal': {'taskId': 'actual'}})


def test_unreadable_raw_capture_is_preserved_and_does_not_erase_other_results(tmp_path):
    directory = tmp_path/'cases'/'bad'/'st_example'
    directory.mkdir(parents=True)
    raw = directory/'trajectory-bad-full.json.gz'
    raw.write_bytes(b'original incomplete gzip')
    rows = vibestudio_campaign.outcomes(tmp_path, ['bad', 'unstarted'], None, {}, {'bad': {}, 'unstarted': {}})
    assert rows['bad']['status'] == 'export-rejected'
    assert rows['bad']['capture_errors']
    assert rows['unstarted']['status'] == 'no-capture'
    assert raw.read_bytes() == b'original incomplete gzip'
    vibestudio_campaign.publish(tmp_path/'collection-interruption.json', {'interruptedCases': ['bad', 'unstarted']})
    rows = vibestudio_campaign.outcomes(tmp_path, ['bad', 'unstarted'], None, {}, {'bad': {}, 'unstarted': {}})
    assert rows['bad']['interrupted']
    assert vibestudio_campaign.wave_infrastructure_hold(rows, {}, 1) is None


def lifecycle_packet(started):
    return {'result': {'passed': False}, 'execution': {
        'error': 'Native fixture policy exceeded its finite deadline',
        'failure': {'phase': 'orchestration', 'error': {'name': 'Error', 'message': 'Native fixture failed',
            'errorData': {'authoredLifecycle': {
                'schema': 'vibestudio.authored-lifecycle/1',
                'runId': 'system-test-runner:teacher:00000000-0000-0000-0000-000000000001',
                'runtimeId': 'do:vibestudio/internal:EvalDO:' + 'a' * 40,
                'boundary': 'assigned-program-load', 'teacherStarted': started}}}}}}


@pytest.mark.parametrize('started', [False, True, None])
def test_authoritative_lifecycle_preserves_no_capture_uncertainty_and_attempt_budget(tmp_path, started):
    packet = lifecycle_packet(started)
    if started is None:
        del packet['execution']['failure']['error']['errorData']
    raw = tmp_path / 'st_original' / 'trajectory-example-full.json.gz'
    raw.parent.mkdir()
    with gzip.open(raw, 'wt') as stream:
        json.dump(packet, stream)
    original = raw.read_bytes()
    outcome = vibestudio_campaign.outcomes(tmp_path, ['example'], None, {}, {'example': {}})['example']
    assert outcome['teacher_started'] is started
    assert outcome['teacher_model_sampled'] is (False if started is False else None)
    assert bool(vibestudio_campaign.infrastructure_failure(outcome, {})) is (started is False)
    rows = vibestudio_campaign.publish_status(tmp_path, {'example': {'workspace': 'examples', 'family': 'native'}},
        {}, [{'example': outcome}], None, 2, {}, attempt_epoch='release', receipt_epochs=['release'])
    assert rows['example']['attempts'] == (0 if started is False else 1)
    assert rows['example']['cumulative_teacher_attempts'] == (0 if started is False else 1)
    assert raw.read_bytes() == original


@pytest.mark.parametrize('damage', ['nested-guest', 'unknown-version', 'wrong-boundary', 'foreign-runtime',
                                    'unbound-run', 'truthy-string', 'malformed-error'])
def test_guest_or_malformed_lifecycle_does_not_prove_pre_teacher(damage):
    packet = lifecycle_packet(False)
    error = packet['execution']['failure']['error']
    evidence = error['errorData']['authoredLifecycle']
    if damage == 'nested-guest':
        packet['execution']['failure']['error'] = {'message':'outer', 'cause': error}
    elif damage == 'unknown-version': evidence['schema'] = 'vibestudio.authored-lifecycle/999'
    elif damage == 'wrong-boundary': evidence['boundary'] = 'after-root'
    elif damage == 'foreign-runtime': evidence['runtimeId'] = 'some-other-runtime'
    elif damage == 'unbound-run': evidence['runId'] = 'unrelated'
    elif damage == 'truthy-string': evidence['teacherStarted'] = 'false'
    else: packet['execution']['failure']['error'] = []
    assert vibestudio_campaign.authored_lifecycle(packet) is None


def test_retained_teacher_call_contradicts_no_start_marker_as_integrity_failure(tmp_path):
    packet = lifecycle_packet(False)
    packet['execution']['diagnostics'] = {'natlangTeacher': {'rootCallIds':['actual'], 'calls':[]}}
    raw = tmp_path / 'st_original' / 'trajectory-example-full.json.gz'
    raw.parent.mkdir()
    with gzip.open(raw, 'wt') as stream: json.dump(packet, stream)
    outcome = vibestudio_campaign.outcomes(tmp_path, ['example'], None, {}, {'example': {}})['example']
    assert outcome['teacher_started'] is True
    assert 'authored_lifecycle' not in outcome
    assert outcome['capture_errors'] == ['Authored lifecycle contradicts retained teacher calls']


def dispatch_lifecycle_packet(case_id='example'):
    packet = lifecycle_packet(False)
    packet['execution']['failure']['error']['errorData']['authoredLifecycle'] = {
        'schema':'vibestudio.authored-lifecycle/2','boundary':'teacher-worker-dispatch',
        'caseId':case_id,'sourceContextId':'TEST-owned-source-context','teacherStarted':False}
    return packet


def test_owned_predispatch_failure_is_bound_to_the_case_and_preserves_original_raw(tmp_path):
    packet = dispatch_lifecycle_packet()
    raw = tmp_path/'st_original'/'trajectory-open-goal-example-full.json.gz'
    raw.parent.mkdir()
    with gzip.open(raw,'wt') as stream: json.dump(packet,stream)
    original=raw.read_bytes()
    outcome=vibestudio_campaign.outcomes(tmp_path,['open-goal-example'],None,{}, {'open-goal-example':{}})['open-goal-example']
    assert outcome['teacher_started'] is False
    assert outcome['teacher_model_sampled'] is False
    assert vibestudio_campaign.infrastructure_failure(outcome,{})
    assert raw.read_bytes()==original


@pytest.mark.parametrize('damage',['wrong-case','no-case-binding','already-dispatched','wrong-boundary','empty-context','fake-eval-id','nested-guest'])
def test_predispatch_witness_cannot_claim_another_case_or_a_started_teacher(damage):
    packet=dispatch_lifecycle_packet()
    error=packet['execution']['failure']['error']
    witness=error['errorData']['authoredLifecycle']
    expected='example'
    if damage=='wrong-case': witness['caseId']='foreign'
    elif damage=='no-case-binding': expected=None
    elif damage=='already-dispatched': witness['teacherStarted']=True
    elif damage=='wrong-boundary': witness['boundary']='assigned-program-load'
    elif damage=='empty-context': witness['sourceContextId']=''
    elif damage=='fake-eval-id': witness['runtimeId']='invented-eval'
    else: packet['execution']['failure']['error']={'message':'outer transport error','cause':error}
    assert vibestudio_campaign.authored_lifecycle(packet,expected) is None
@pytest.mark.parametrize('outcome,attempts,stop,expected', [
    ('verified',2,False,'drain-native'),('test-failed',2,False,'continue-paid'),
    ('test-failed',1,False,'drain-native'),('verified',2,True,'continue-paid'),
])
def test_native_continuation_uses_actual_retry_queue_without_publishing_goal_success(tmp_path,monkeypatch,outcome,attempts,stop,expected):
    import hashlib
    capture=tmp_path/'capture';capture.mkdir();host='a'*40
    repositories={'vibestudio':{'git_commit':host,'dirty':False}}
    source=json.dumps({'repositories':repositories}).encode();(capture/'source.json').write_bytes(source)
    ready=tmp_path/'ready.json';ready.write_text(json.dumps({'schema':'vibestudio.native-reference-continuation/1','nonce':'TEST-only','capture':str(capture),'sourceSha256':hashlib.sha256(source).hexdigest(),'repositories':repositories}))
    monkeypatch.setattr(vibestudio_campaign,'outcomes',lambda *args:{'case':{'status':outcome}})
    result=vibestudio_campaign.native_completion_decision(ready,capture,['case'],cases={'case':{}},held=set(),training={},seeds={},receipts=[],receipt_epochs=[],attempt_epoch='TEST-epoch',review_model=None,attempts=attempts,holds={},host_commit=host,stop_requested=stop)
    assert result['decision']==expected and result['additionalTeacherCalls']==0
    assert not (tmp_path/'status.json').exists()
    assert result['requestSha256']==hashlib.sha256(ready.read_bytes()).hexdigest()
    for key in ['capture','sourceSha256','repositories','nonce']:
        original=json.loads(ready.read_text());mutated=dict(original);mutated[key]=None;ready.write_text(json.dumps(mutated))
        with pytest.raises(ValueError,match='owned capture'):
            vibestudio_campaign.native_completion_decision(ready,capture,['case'],cases={'case':{}},held=set(),training={},seeds={},receipts=[],receipt_epochs=[],attempt_epoch='TEST-epoch',review_model=None,attempts=attempts,holds={},host_commit=host)
        ready.write_text(json.dumps(original))


def test_native_reference_packets_are_excluded_from_training_snapshots(tmp_path,monkeypatch):
    registrations=[]
    episode={'trajectory_ir':{'outcome':{'status':'passed','task_completed':True}},'turns':[1]}
    monkeypatch.setattr(vibestudio_campaign,'verified_episode',lambda *_:episode)
    monkeypatch.setattr(vibestudio_campaign.data_registry,'record_dataset',lambda *args,**kwargs:registrations.append(kwargs) or {'id':'TEST-training-only'})
    reference=tmp_path/'capture/native-reference-only/held-reference.json';reference.parent.mkdir(parents=True);reference.write_text('TEST-only held native reference original')
    vibestudio_campaign.export_trajectories(tmp_path,{'teacher':{'status':'verified','manifest':'manifest.json'}},set(),set())
    assert 'capture/native-reference-only/**' in registrations[0]['exclude']
    assert reference.read_text()=='TEST-only held native reference original'


@pytest.mark.parametrize("previous_release", ["old-release", "current-release", None, 17])
@pytest.mark.parametrize("legacy_hold", [False, True])
def test_archival_recovery_only_holds_current_or_unidentified_release(tmp_path, monkeypatch, previous_release, legacy_hold):
    from types import SimpleNamespace
    names=['one','two']
    cases={name:{'taskId':name,'workspace':'system','family':'settings'} for name in names}
    waves=tmp_path/'waves';jobs=waves/'jobs';jobs.mkdir(parents=True)
    plan={'cases':names,'attempt_epoch':'old-epoch','provenance':{'release':{'sha256':previous_release}}}
    vibestudio_campaign.publish_immutable(jobs/'0000.json',plan)
    (waves/'0000-capture').mkdir()
    monkeypatch.setattr(vibestudio_campaign,'campaign_scope',lambda *_:(None,{},cases))
    monkeypatch.setattr(vibestudio_campaign,'attempt_context',lambda *_:('new-epoch',{'release':{'sha256':'current-release'}},{}))
    monkeypatch.setattr(vibestudio_campaign,'native_reference_capable',lambda *_:False)
    monkeypatch.setattr(vibestudio_campaign,'prerequisite_holds',lambda *_:{})
    monkeypatch.setattr(vibestudio_campaign,'sync_index',lambda *_,**__:None)
    archived=[]
    monkeypatch.setattr(vibestudio_campaign,'remote_archive',lambda *args,**kw:archived.append(kw['wave']))
    monkeypatch.setattr(vibestudio_campaign,'outcomes',lambda *_:{name:{'status':'no-capture','teacher_started':False,'reason':'Original harness produced no capture'} for name in names})
    def no_teacher(*_,**__):raise AssertionError('Historical capture must never regenerate teacher calls')
    monkeypatch.setattr(vibestudio_campaign.CollectorCheckpoint,'run',no_teacher)
    args=SimpleNamespace(root=tmp_path,held=tmp_path/'held',seed=[],host=tmp_path,archive=tmp_path/'archive',review_model=None,attempts=2,max_waves=1,wave_size=64,concurrency=8,hold_historical_infrastructure=legacy_hold)
    vibestudio_campaign.drive(args)
    expected_hold=legacy_hold or previous_release!='old-release'
    assert (tmp_path/'runtime-hold.json').is_file() is expected_hold
    assert (tmp_path/'stop-after-wave').is_file() is expected_hold
    assert (waves/'0000-historical-infrastructure-hold.json').is_file() is not expected_hold
    assert len(archived)==1 and archived[0]['plan']==plan
    assert vibestudio_campaign.read(waves/'0000-result.json')['one']['status']=='no-capture'


def test_review_transport_failure_preserves_teacher_exposure_without_new_teacher_selection():
    cases={'goal':{'taskId':'goal-id','requiresReview':True}}
    receipt={'goal':{'status':'pending-review','teacher_started':True,'teacher_model_sampled':True}}
    rows,summary=vibestudio_campaign.campaign_status(cases,{},[receipt],'test-only:reviewer',2,{},attempt_epoch='e',receipt_epochs=['e'])
    assert rows['goal']['status']=='pending-review' and rows['goal']['attempts']==0
    assert rows['goal']['cumulative_teacher_attempts']==1 and summary['epoch_teacher_attempts']==1
    assert vibestudio_campaign.next_wave(cases,rows,8)==[]
    rows,_=vibestudio_campaign.campaign_status(cases,{'goal':'accepted-manifest'},[receipt],'test-only:reviewer',2,{})
    assert rows['goal']['status']=='verified'


def test_completed_seed_feedback_validates_before_append_and_preserves_ordinal_identity(tmp_path,monkeypatch):
    source=tmp_path/'candidate/recovered-campaign-seed-paths.txt';source.parent.mkdir()
    old=tmp_path/'old-manifest.json';new=tmp_path/'new-manifest.json';old.write_text('old');new.write_text('new')
    source.write_text(str(old)+'\n'+str(new)+'\n')
    seeds={'old':str(old.resolve())};calls=[]
    def episode(path,*_):
        return {'task_id':'new-id','trajectory_ir':{'outcome':{'task_completed':True}}}
    monkeypatch.setattr(vibestudio_campaign,'verified_episode',episode)
    monkeypatch.setattr(vibestudio_campaign,'remote_archive',lambda args,destination,receipt,capture,**kw:calls.append((destination,receipt,capture)))
    monkeypatch.setattr(vibestudio_campaign,'sync_index',lambda *args,**kw:None)
    args=SimpleNamespace(root=tmp_path,archive=Path('/durable/archive'))
    cases={'old':{'taskId':'old-id'},'new':{'taskId':'new-id'}}
    vibestudio_campaign.refresh_completed_seeds(args,None,seeds,cases,None,{})
    assert list(seeds)==['old','new'] and seeds['old']==str(old.resolve())
    assert calls==[(Path('/durable/archive/seed-0001'),{'new':{'status':'verified','manifest':'manifest.json'}},new.parent)]
    receipt=json.loads((tmp_path/'seed-feedback.json').read_text())
    assert receipt['sourceSha256']==__import__('hashlib').sha256(source.read_bytes()).hexdigest()
    vibestudio_campaign.refresh_completed_seeds(args,None,seeds,cases,None,{})
    assert len(calls)==1


def test_partial_completed_seed_feedback_fails_before_archival_or_reused_mutation(tmp_path,monkeypatch):
    source=tmp_path/'candidate/recovered-campaign-seed-paths.txt';source.parent.mkdir();source.write_text(str(tmp_path/'partial.json')+'\n')
    monkeypatch.setattr(vibestudio_campaign,'verified_episode',lambda *_:{'task_id':'goal-id','trajectory_ir':{'outcome':{'task_completed':False}}})
    monkeypatch.setattr(vibestudio_campaign,'remote_archive',lambda *a,**k:pytest.fail('must validate before publication'))
    with pytest.raises(ValueError,match='partial training evidence'):
        vibestudio_campaign.refresh_completed_seeds(SimpleNamespace(root=tmp_path),None,{}, {'goal':{'taskId':'goal-id'}},None,{})
    assert not (tmp_path/'reused.json').exists()


def test_native_passed_packet_with_bound_review_transport_waits_for_review(tmp_path):
    import hashlib
    run=tmp_path/'st_owned';run.mkdir()
    raw=run/'trajectory-goal-full.json.gz'
    packet={'result':{'passed':True},'execution':{'diagnostics':{'natlangTeacher':{'rootCallIds':['call'],'calls':[{'record':{'cost':{'model_requests':1}}}]}}}}
    with gzip.open(raw,'wt') as f:json.dump(packet,f)
    pins=tmp_path/'source.json';pins.write_text(json.dumps({'teacher':'test-only:teacher'}))
    capture={'schema':'natlang.live-workflow-review-failure/1','model':'test-only:reviewer','teacher':'test-only:teacher',
             'input':{'actual':{'goal':'the goal'}},'complete':False,'failure':{'kind':'provider-transport','phase':'provider-turn','error':{'name':'Error','message':'fetch failed'}}}
    sidecar=tmp_path/'failed-review.json';sidecar.write_text(json.dumps(capture))
    diagnostic={'schema':'vibestudio.teacher-export-rejection/1','scenario':'goal','runId':'st_owned',
                'trajectory':{'path':raw.name,'sha256':hashlib.sha256(raw.read_bytes()).hexdigest()},'pinsSha256':hashlib.sha256(pins.read_bytes()).hexdigest(),
                'error':{'message':'Error: fetch failed'},'reviewFailure':{k:capture[k] for k in ['schema','model','teacher','failure']}}
    diagnostic['reviewFailure']['capture']={'path':sidecar.name,'sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    (run/'export-rejections.jsonl').write_text(json.dumps(diagnostic)+'\n')
    outcomes=vibestudio_campaign.outcomes(tmp_path,['goal'],None,{}, {'goal':{'taskId':'goal-id'}})
    assert outcomes['goal']['status']=='pending-review' and outcomes['goal']['teacher_model_sampled'] is True
    sidecar.write_text('{}')
    assert vibestudio_campaign.outcomes(tmp_path,['goal'],None,{}, {'goal':{'taskId':'goal-id'}})['goal']['status']=='export-rejected'
