from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))

import vibestudio_campaign


def test_new_release_fresh_coverage_precedes_historical_genuine_failures():
    cases = {'old': {'workspace': 'system', 'family': 'old'},
             'new-a': {'workspace': 'system', 'family': 'fresh-a'},
             'new-b': {'workspace': 'system', 'family': 'fresh-b'}}
    rows = {'old': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 2},
            'new-a': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 0},
            'new-b': {'status': 'pending', 'attempts': 0, 'cumulative_teacher_attempts': 0}}
    assert vibestudio_campaign.next_wave(cases, rows, 64) == ['new-a', 'new-b']
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


@pytest.mark.parametrize("teacher_timeout", [3_600_000, 600_000])
def test_drive_retries_historical_exhaustion_in_new_immutable_wave(monkeypatch, tmp_path, teacher_timeout):
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
    def collect(_owner, command, **kwargs):
        assert command[:4] == ['node', '--import', 'tsx', 'scripts/collect-live-workflow-teacher.mjs']
        assert kwargs['env'] == {'PINNED': 'yes'}
        assert command[command.index('--checkpoint-request') + 1] == str(kwargs['request_file'])
        assert command.count('--case') == 1 and command[-1] == 'retry'
        assert command[command.index('--test-timeout-ms') + 1] == str(teacher_timeout)
        assert command[command.index('--cases-per-instance') + 1] == '12'
        assert command[command.index('--concurrency') + 1] == '3'
        Path(command[command.index('--out') + 1]).mkdir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(vibestudio_campaign.CollectorCheckpoint, 'run', collect)
    args = SimpleNamespace(root=tmp_path, held=tmp_path / 'held', seed=[], host=tmp_path, archive=tmp_path / 'archive',
        review_model=None, attempts=2, max_waves=1, wave_size=6, concurrency=3, reserve_disk_gib=8,
        teacher_timeout_ms=teacher_timeout)
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
                    'workerd_resets': [], 'all_captures_missing': False}
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
    monkeypatch.setattr(vibestudio_campaign, 'transfer', lambda *_: None)
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
