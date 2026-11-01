import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
import run_reviewed_luna_dispatcher as dispatcher
from freeze_training_runtime import tree_identity


@pytest.fixture
def reviewed_plan(tmp_path):
    campaign_root = ROOT / 'runs' / f'.dispatch-test-{uuid.uuid4().hex}'
    source = tmp_path / 'source.cases.jsonl'
    source.write_text(''.join(json.dumps({'id': f'case-{i}'}) + '\n' for i in range(3)))
    runtime = tmp_path / 'runtime'
    for directory in ('dist/native', 'scripts', 'src'):
        (runtime / directory).mkdir(parents=True, exist_ok=True)
    (runtime / 'dist/native/runtime.js').write_text('// frozen test runtime\n')
    (runtime / 'scripts/worker.js').write_text('// frozen script\n')
    (runtime / 'src/source.ts').write_text('// source identity\n')
    receipt = runtime / 'frozen-runtime.json'
    receipt.write_text(json.dumps({'files': tree_identity(runtime)}))
    authority = tmp_path / 'authority.json'
    authority.write_text(json.dumps({'luna_workers': [], 'additional_teachers': {'luna': {'workers': []}}}))
    plan = {
        'schema': 'natlang.reviewed_luna_dispatch_plan/1',
        'root_approved': True,
        'campaign_id': 'dispatch-test-v1',
        'journal_spool_root': str(dispatcher.machine_resolve('data_nvme', 'natlang-teacher-journal-spool', 'dispatch-test-v1')),
        'cwd': str(ROOT),
        'supervisor': str(dispatcher.RUNNER),
        'dispatcher_sha256': dispatcher.digest(dispatcher.DISPATCHER),
        'supervisor_sha256': dispatcher.digest(dispatcher.RUNNER),
        'source': str(source),
        'source_sha256': dispatcher.digest(source),
        'runtime': str(runtime),
        'runtime_receipt_sha256': dispatcher.digest(receipt),
        'artifact_hashes': {
            str(dispatcher.DISPATCHER): dispatcher.digest(dispatcher.DISPATCHER),
            str(dispatcher.RUNNER): dispatcher.digest(dispatcher.RUNNER),
            str(ROOT / 'scripts/rl_tasks/execution_adapter.py'):
                dispatcher.digest(ROOT / 'scripts/rl_tasks/execution_adapter.py'),
            str(source): dispatcher.digest(source),
            str(receipt): dispatcher.digest(receipt),
        },
        'campaign_root': str(campaign_root),
        'claim_ledger': str(campaign_root / 'dispatch/claims.jsonl'),
        'launch_record': str(campaign_root / 'dispatch/launch.json'),
        'authority': str(authority),
        'provider': 'openai-codex',
        'model_id': 'gpt-6-luna',
        'model_concurrency': 1,
        'slots': 2,
        'resource_policy': {
            'schema': 'natlang.teacher_execution_resource_policy/1',
            'resumable': False, 'backfill': False,
            'partial_execution_is_resumable': False,
            'partial_execution_disposition': 'durable evidence only; no graph or simulator-state continuation',
            'campaign_resume': 'explicit only for unclaimed cases; uncertain claims are terminalized',
        },
        'execution_plans': True,
        'reasoning_effort': 'low',
        'case_seconds': 120,
        'min_free_mib': 1,
        'max_transport_retries': 0,
        'cases': [{'index': i, 'seed': 1000 + i} for i in range(3)],
    }
    plan_path = tmp_path / 'reviewed-plan.json'
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    yield plan_path, dispatcher.digest(plan_path), campaign_root, plan
    shutil.rmtree(campaign_root, ignore_errors=True)


def _mock_controls(monkeypatch):
    monkeypatch.setattr(dispatcher, '_authority_preflight', lambda identity: None)
    monkeypatch.setattr(dispatcher, '_authority_update', lambda identity, binding=None: None)
    monkeypatch.setattr(dispatcher, '_require_nonresumable_ledger_claim', lambda *_args: None)
    monkeypatch.setattr(dispatcher.time, 'sleep', lambda seconds: None)


def _entry_for_queue(command):
    queue = Path(command[2])
    return queue, json.loads(queue.read_text().splitlines()[0]), command[3]


def test_reviewed_plan_validates_runtime_tree_and_all_identity_pins(reviewed_plan):
    plan_path, plan_sha, _, _ = reviewed_plan
    plan, identity = dispatcher.validate_plan(plan_path, plan_sha)
    assert plan['campaign_id'] == identity['campaign_id']
    assert identity['runtime_receipt_sha256'] == plan['runtime_receipt_sha256']
    assert identity['max_transport_retries'] == 0
    assert identity['journal_spool_root'] == str(dispatcher.machine_resolve(
        'data_nvme', 'natlang-teacher-journal-spool', plan['campaign_id']).resolve())
    assert identity['journal_spool_policy']['per_case_bytes'] == 256 * 1024 * 1024
    assert identity['journal_spool_policy']['active_case_credits'] == 6
    assert [row['index'] for row in identity['cases']] == [0, 1, 2]
    entry, *_ = dispatcher._claim_queue(plan, identity, 1, identity['cases'][0])
    assert entry['prompt_rendering'] == 'full'
    assert entry['campaign_id'] == plan['campaign_id']


@pytest.mark.parametrize('policy', [None, {},
    {'schema': 'natlang.teacher_execution_resource_policy/1', 'resumable': True,
     'backfill': False, 'partial_execution_is_resumable': False},
    {'schema': 'natlang.teacher_execution_resource_policy/1', 'resumable': False,
     'backfill': True, 'partial_execution_is_resumable': False},
    {'schema': 'natlang.teacher_execution_resource_policy/1', 'resumable': False,
     'backfill': False, 'partial_execution_is_resumable': True}])
def test_plan_requires_nonresumable_execution_policy(reviewed_plan, policy):
    path, _, _, plan = reviewed_plan
    plan['resource_policy'] = policy
    path.write_text(json.dumps(plan, sort_keys=True))
    with pytest.raises(ValueError, match='non-resumable execution'):
        dispatcher.validate_plan(path, dispatcher.digest(path))


def test_dispatch_claim_must_explicitly_disable_resumability(tmp_path, monkeypatch):
    state_path = tmp_path / 'ledger.json'
    unit = 'campaign-test.service'
    monkeypatch.setenv('NATLANG_MEMORY_LEDGER', str(state_path))
    monkeypatch.setenv('NATLANG_LEDGER_UNIT', unit)
    import memory_ledger
    monkeypatch.setattr(memory_ledger, 'STATE', str(state_path))
    monkeypatch.setattr(dispatcher, 'memory_ledger', memory_ledger)
    with pytest.raises(ValueError, match='no memory-ledger claim'):
        dispatcher._require_nonresumable_ledger_claim('campaign-test')
    state_path.write_text(json.dumps({'schema': 'natlang.memory-ledger/1', 'claims': {
        unit: {'resumable': True, 'spec': {'resumable': True}}}, 'events': []}))
    with pytest.raises(ValueError, match='--no-resumable'):
        dispatcher._require_nonresumable_ledger_claim('campaign-test')
    state_path.write_text(json.dumps({'schema': 'natlang.memory-ledger/1', 'claims': {
        unit: {'resumable': False, 'spec': {'resumable': False}}}, 'events': []}))
    with pytest.raises(ValueError, match='reviewed campaign unit'):
        dispatcher._require_nonresumable_ledger_claim('other-campaign')
    dispatcher._require_nonresumable_ledger_claim('campaign-test')


def test_campaign_preflight_allows_distinct_campaigns_under_six_worker_ceiling(reviewed_plan, monkeypatch):
    _, _, _, plan = reviewed_plan
    authority = Path(plan['authority'])
    authority.write_text(json.dumps({'luna_workers': [], 'additional_teachers': {'luna': {
        'workers': [{'campaign': 'older-campaign', 'status': 'running'}]}}}))
    monkeypatch.setattr(dispatcher, 'reconcile_luna_authority', lambda state: {
        'actual_live_workers': 1, 'active_campaigns': ['older-campaign']})
    _, identity = dispatcher.validate_plan(*[Path(reviewed_plan[0]), reviewed_plan[1]])

    dispatcher._authority_preflight(identity)

    state = json.loads(authority.read_text())
    assert state['additional_teachers']['luna']['workers'][0]['campaign'] == 'older-campaign'
    assert state['luna_active_campaigns'] == ['older-campaign']


def test_worker_slot_reservations_atomically_enforce_global_six_cap(tmp_path, monkeypatch):
    authority = tmp_path / 'authority.json'
    authority.write_text(json.dumps({'luna_workers': [], 'additional_teachers': {'luna': {'workers': []}}}))
    monkeypatch.setattr(dispatcher, 'reconcile_luna_authority', lambda state: {
        'actual_live_workers': 0, 'active_campaigns': []})
    monkeypatch.setattr(dispatcher, '_reservation_is_live', lambda row: True)
    def identity(campaign):
        return {'authority': str(authority), 'campaign_id': campaign, 'plan_path': f'/plans/{campaign}.json',
                'plan_sha256': 'a' * 64, 'runtime': '/runtime'}

    assert all(dispatcher._reserve_worker_slot(identity('campaign-a'), index=i, slot=i + 1,
               queue=f'/a/{i}/queue.jsonl', journal=f'/a/{i}/journal.jsonl') for i in range(4))
    assert all(dispatcher._reserve_worker_slot(identity('campaign-b'), index=i, slot=i + 1,
               queue=f'/b/{i}/queue.jsonl', journal=f'/b/{i}/journal.jsonl') for i in range(2))
    assert not dispatcher._reserve_worker_slot(identity('campaign-c'), index=0, slot=1,
               queue='/c/0/queue.jsonl', journal='/c/0/journal.jsonl')
    state = json.loads(authority.read_text())
    assert len(state['luna_worker_reservations']) == 6
    assert {row['campaign'] for row in state['luna_worker_reservations']} == {'campaign-a', 'campaign-b'}


def test_reservation_recognizes_relative_dispatcher_argv(tmp_path, monkeypatch):
    plan_path = str(tmp_path / 'plan.json')
    plan_sha = 'b' * 64
    monkeypatch.setattr(dispatcher, '_proc_argv', lambda pid: [
        '.venv-neuralese/bin/python', 'scripts/run_reviewed_luna_dispatcher.py',
        plan_path, '--sha256', plan_sha])
    row = {'dispatcher_pid': os.getpid(), 'plan_path': plan_path, 'plan_sha256': plan_sha,
           'queue': '/missing/queue.jsonl', 'journal': '/missing/journal.jsonl', 'runtime': '/runtime'}

    assert dispatcher._reservation_is_live(row)


def test_reviewed_runner_binds_spool_root(reviewed_plan, tmp_path):
    plan_path, plan_sha, campaign_root, raw = reviewed_plan
    plan, identity = dispatcher.validate_plan(plan_path, plan_sha)
    entry, queue, _, journal, _, _ = dispatcher._claim_queue(plan, identity, 1, identity['cases'][0])
    command = dispatcher._runner_command(plan, entry, queue, journal)
    assert command[command.index('--journal-spool-root') + 1] == identity['journal_spool_root']


def test_reviewed_plan_rejects_spool_outside_configured_root(reviewed_plan):
    plan_path, _, _, plan = reviewed_plan
    plan['journal_spool_root'] = str(ROOT / 'runs' / 'unbounded-spool')
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    with pytest.raises(ValueError, match='campaign-specific directory'):
        dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))


def test_external_campaign_root_is_allowlisted_and_bound_to_its_filesystem(reviewed_plan):
    plan_path, _, _, plan = reviewed_plan
    approved_root = dispatcher._external_campaign_root()
    external_root = approved_root / f'.dispatch-test-{uuid.uuid4().hex}'
    plan['campaign_root'] = str(external_root)
    plan['claim_ledger'] = str(external_root / 'dispatch/claims.jsonl')
    plan['launch_record'] = str(external_root / 'dispatch/launch.json')
    plan['min_free_mib'] = 40960
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    validated, identity = dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))
    assert identity['campaign_root'] == str(external_root.resolve())
    assert identity['storage_root'] == str(approved_root)
    assert validated['min_free_mib'] == 40960
    assert dispatcher._free_space_mib(identity['storage_root']) > 40960


def test_external_campaign_root_uses_configured_data_hdd_mount(reviewed_plan, tmp_path, monkeypatch):
    data_hdd = tmp_path / 'configured-volume'
    data_hdd.mkdir()
    expected = (data_hdd / 'natlang-luna-campaigns').resolve()
    monkeypatch.setattr(dispatcher, 'machine_resolve', lambda name, *parts: data_hdd.joinpath(*parts))
    mounted_paths = []
    monkeypatch.setattr(dispatcher.os.path, 'ismount', lambda path: mounted_paths.append(Path(path)) or True)

    actual = dispatcher._campaign_storage_root(expected / 'campaign-a')

    assert actual == expected
    assert mounted_paths == [data_hdd.resolve()]


def test_external_campaign_root_requires_configured_data_hdd_mount(reviewed_plan, tmp_path, monkeypatch):
    data_hdd = tmp_path / 'configured-volume'
    data_hdd.mkdir()
    monkeypatch.setattr(dispatcher, 'machine_resolve', lambda name, *parts: data_hdd.joinpath(*parts))
    monkeypatch.setattr(dispatcher.os.path, 'ismount', lambda path: False)

    with pytest.raises(ValueError, match='configured data_hdd root to be mounted'):
        dispatcher._campaign_storage_root(data_hdd / 'natlang-luna-campaigns' / 'campaign-a')


def test_campaign_root_rejects_non_allowlisted_external_paths():
    with pytest.raises(ValueError, match='allowlisted external campaign root'):
        dispatcher._campaign_storage_root(Path('/tmp/unreviewed-campaign'))


def test_text_neuralese_capability_reads_isolated_typescript_seal_and_fails_closed(tmp_path, monkeypatch):
    runtime = tmp_path / 'isolated-runtime'
    (runtime / 'dist/teacher').mkdir(parents=True)
    (runtime / 'dist/teacher/cli.js').write_text("// CLI exposes --text-neuralese-emulation\n")
    (runtime / 'dist/teacher/collector.js').write_text('// collector transport implementation\n')
    source_inputs = {'src/teacher/collector.ts': 'a' * 64}
    source_digest = hashlib.sha256(json.dumps(
        source_inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    build = {'mode': 'isolated_typescript', 'source_inputs': source_inputs,
             'source_inputs_sha256': source_digest}
    manifest = {'version': 'natlang.frozen_runtime/1', 'files': tree_identity(runtime), 'build': build}
    (runtime / 'frozen-runtime.json').write_text(json.dumps(manifest, sort_keys=True))

    def cli_help(command, **kwargs):
        assert command[1] == str(runtime / 'dist/teacher/cli.js')
        return SimpleNamespace(returncode=0, stdout='--text-neuralese-emulation\n', stderr='')

    monkeypatch.setattr(dispatcher.subprocess, 'run', cli_help)
    capability = dispatcher._validate_text_neuralese_emulation(runtime, manifest)
    assert capability['build_mode'] == 'isolated_typescript'
    assert capability['cli_sha256'] == dispatcher.digest(runtime / 'dist/teacher/cli.js')
    assert capability['collector_sha256'] == dispatcher.digest(runtime / 'dist/teacher/collector.js')

    (runtime / 'dist/teacher/collector.js').write_text('// changed after freeze\n')
    with pytest.raises(ValueError, match='does not bind'):
        dispatcher._validate_text_neuralese_emulation(runtime, manifest)


def test_prompt_rendering_is_forwarded_into_each_queue(reviewed_plan):
    plan_path, _, _, plan = reviewed_plan
    plan['prompt_rendering'] = 'program-header'
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    validated, identity = dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))
    entry, *_ = dispatcher._claim_queue(validated, identity, 1, identity['cases'][0])
    assert entry['prompt_rendering'] == 'program-header'


def test_bound_system_file_is_forwarded_into_each_queue(reviewed_plan, tmp_path):
    plan_path, _, _, plan = reviewed_plan
    system_file = tmp_path / 'system-with-attempt-feedback.md'
    system_file.write_text('Use the bound previous-attempt feedback as diagnostic evidence only.')
    plan['system_file'] = str(system_file)
    plan['artifact_hashes'][str(system_file)] = dispatcher.digest(system_file)
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    validated, identity = dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))
    entry, _, _, _, _, payload = dispatcher._claim_queue(validated, identity, 1, identity['cases'][0])
    assert entry['system_file'] == str(system_file.resolve())
    assert json.loads(payload)['system_file'] == str(system_file.resolve())

    system_file.write_text('tampered feedback')
    with pytest.raises(ValueError, match='Reviewed artifact changed'):
        dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))


def test_judged_workflows_require_and_preserve_independent_judge_binding(reviewed_plan):
    plan_path, _, _, plan = reviewed_plan
    source = Path(plan['source'])
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[1]['semantics'] = {'oracle': {'level': 'judged', 'rubric': 'Check the actual outcome.'}}
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan['source_sha256'] = plan['artifact_hashes'][str(source)] = dispatcher.digest(source)
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    with pytest.raises(ValueError, match='independent judge_model'):
        dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))
    plan['judge_model'] = {'model_id': 'actual-independent-judge', 'provider': 'openai-codex'}
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    validated, identity = dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))
    entry, _, _, _, _, payload = dispatcher._claim_queue(validated, identity, 1, identity['cases'][1])
    assert entry['judge_model'] == plan['judge_model']
    assert json.loads(payload)['judge_model'] == plan['judge_model']
    plan['judge_model']['model_id'] = 'gpt-6-luna'
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    with pytest.raises(ValueError, match='distinct actual model_id'):
        dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))


def test_reviewed_plan_requires_explicit_nonnegative_retry_budget(reviewed_plan):
    plan_path, _, _, plan = reviewed_plan
    for value in (None, True, -1, '0'):
        plan['max_transport_retries'] = value
        plan_path.write_text(json.dumps(plan, sort_keys=True))
        with pytest.raises(ValueError, match='max_transport_retries'):
            dispatcher.validate_plan(plan_path, dispatcher.digest(plan_path))


def test_source_indices_match_queue_runner_physical_jsonl_lines(tmp_path):
    source = tmp_path / 'blank-lines.jsonl'
    source.write_text('{"id":"first"}\n\n{"id":"third"}\n')
    assert dispatcher._source_rows(source) == [(0, {'id': 'first'}), (2, {'id': 'third'})]


def test_free_slot_pulls_next_case_and_reuses_failure_cooldown_journal(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, _ = reviewed_plan
    made = []
    by_index = {}
    active_count = 0
    max_active = 0

    class FakeRunner:
        def __init__(self, command, **kwargs):
            nonlocal active_count, max_active
            self.pid = 50000 + len(made)
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            self.index = self.entry['index']
            self.slot = int(self.queue.parts[-4].split('-')[1])
            self.polls = 0
            self.done = False
            active_count += 1
            max_active = max(max_active, active_count)
            if self.index == 2:
                # Index 2 reuses slot 1's journal after index 0 failed.
                previous = [json.loads(line) for line in Path(self.journal).read_text().splitlines()]
                assert any(row.get('event') == 'finish' and row.get('status') == 'incomplete'
                           and row.get('next_allowed_at') for row in previous)
            self.finish_after = 1 if self.index in (0, 2) else 5
            self.terminate_calls = 0
            made.append(self)
            by_index[self.index] = self

        def poll(self):
            nonlocal active_count
            if self.done:
                return 0
            self.polls += 1
            if self.polls < self.finish_after:
                return None
            status = 'incomplete' if self.index == 0 else 'complete'
            row = {'event': 'finish', 'key': self.entry['key'], 'status': status,
                   'exit_code': 0, 'failure_streak': 1 if status == 'incomplete' else 0,
                   'next_allowed_at': time.time() + 60 if status == 'incomplete' else 0,
                   'output_accounting': {'complete': True}}
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            self.done = True
            active_count -= 1
            return 0

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, timeout=None):
            return self.poll()

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha) == 0
    assert all(runner.entry['transport_retries'] == 0 for runner in made)
    events = [json.loads(line) for line in (campaign_root / 'dispatch/claims.jsonl').read_text().splitlines()]
    claims = [row for row in events if row.get('event') == 'claim']
    assert [(row['index'], row['slot']) for row in claims] == [(0, 1), (1, 2), (2, 1)]
    assert len({row['index'] for row in claims}) == 3
    assert max_active == 2
    assert by_index[1].polls > by_index[0].polls
    assert Path(by_index[0].journal) == Path(by_index[2].journal)


@pytest.mark.parametrize('blocked_role', ['campaign-output', 'hot-journal-spool'])
def test_storage_floor_pauses_before_claim_and_resumes_pending_case(reviewed_plan, tmp_path, monkeypatch, blocked_role):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    plan['slots'] = 1
    plan['cases'] = plan['cases'][:1]
    plan['min_free_mib'] = 10

    # Use two independent filesystem roots and bind the reviewed external
    # campaign plus internal hot spool to them. The check must gate on either.
    external = tmp_path / 'data-hdd'
    external_root = external / 'natlang-luna-campaigns'
    nvme = tmp_path / 'data-nvme'
    external_root.mkdir(parents=True)
    nvme.mkdir()
    roots = {'data_hdd': external, 'data_nvme': nvme}
    monkeypatch.setattr(dispatcher, 'machine_resolve', lambda name, *parts: roots[name].joinpath(*parts))
    monkeypatch.setattr(dispatcher.os.path, 'ismount', lambda path: Path(path) == external.resolve())
    campaign_root = external_root / 'campaign'
    plan['campaign_root'] = str(campaign_root)
    plan['claim_ledger'] = str(campaign_root / 'dispatch/claims.jsonl')
    plan['launch_record'] = str(campaign_root / 'dispatch/launch.json')
    plan['journal_spool_root'] = str(nvme / 'natlang-teacher-journal-spool' / plan['campaign_id'])
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_sha = dispatcher.digest(plan_path)

    first_block = True
    def free_bytes(path):
        nonlocal first_block
        if Path(path) == external_root:
            if blocked_role == 'campaign-output' and first_block:
                first_block = False
                return 0
            return 20 * dispatcher.MIB
        if blocked_role == 'hot-journal-spool' and first_block:
            first_block = False
            return dispatcher.JOURNAL_SPOOL_FREE_FLOOR_BYTES + dispatcher.JOURNAL_SPOOL_CASE_BYTES - 1
        return dispatcher.JOURNAL_SPOOL_FREE_FLOOR_BYTES + dispatcher.JOURNAL_SPOOL_CASE_BYTES + 1
    monkeypatch.setattr(dispatcher, '_free_space_bytes', free_bytes)
    monkeypatch.setattr(dispatcher, '_spool_counts', lambda _root: (0, 0))
    made = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            self.pid = 51000 + len(made)
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            made.append(self.entry['index'])
        def poll(self):
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps({'event': 'finish', 'key': self.entry['key'], 'status': 'complete',
                                         'exit_code': 0, 'output_accounting': {'complete': True}}) + '\n')
            return 0
        def terminate(self): pass
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha) == 0
    events = [json.loads(line) for line in (campaign_root / 'dispatch/claims.jsonl').read_text().splitlines()]
    kinds = [row.get('event') for row in events]
    pause = kinds.index('storage_paused')
    resume = kinds.index('storage_resumed')
    claims = [(i, row['index']) for i, row in enumerate(events) if row.get('event') == 'claim']
    assert pause < resume < claims[0][0]
    assert [index for _, index in claims] == [0]
    assert events[pause]['next_case_index'] == 0
    assert events[pause]['disposition'] == 'no case claimed; pending case list preserved until all bound storage floors are met'
    checks = {check['role']: check for check in events[pause]['storage_checks']}
    assert set(checks) == {'campaign-output', 'hot-journal-spool'}
    assert checks[blocked_role]['ok'] is False
    assert checks['campaign-output']['path'] == str(external_root)
    assert checks['hot-journal-spool']['path'] == plan['journal_spool_root']
    assert made == [0]
    assert not any(row.get('event') == 'abandoned' for row in events)
    _, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    dispatcher._verify_ledger_events(events, identity)


def test_spool_preflight_ignores_only_dead_pid_leases(tmp_path):
    spool = tmp_path / 'spool'
    leases = spool / '.leases'
    leases.mkdir(parents=True)
    (leases / 'dead.json').write_text(json.dumps({'pid': 2**30}), encoding='utf-8')
    (leases / 'malformed.json').write_text('{', encoding='utf-8')
    (leases / 'live.json').write_text(json.dumps({'pid': os.getpid()}), encoding='utf-8')
    assert dispatcher._spool_counts(spool) == (2, 0)


def test_resume_holds_uncertain_claim_and_dispatches_only_unclaimed_cases(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    _, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    case0 = identity['cases'][0]
    entry, queue, queue_sha, journal, _, payload = dispatcher._claim_queue(plan, identity, 1, case0)
    dispatcher._write_claim_queue(queue, payload)
    ledger = Path(identity['ledger'])
    dispatcher._append_event(ledger, {'event': 'campaign_open', 'identity': identity, 'opened_at': 'test'})
    claim = {'claim_id': entry['key'], 'index': 0, 'slot': 1, 'key': entry['key'],
             'queue': queue, 'queue_sha256': queue_sha, 'journal': journal,
             'source': identity['source'], 'runtime': identity['runtime'], 'campaign': identity['campaign_id']}
    dispatcher._append_event(ledger, {'event': 'claim', 'identity_sha256': identity['identity_sha256'], **claim})
    dispatcher._append_event(ledger, {'event': 'runner_started', 'identity_sha256': identity['identity_sha256'],
                                      'claim_id': entry['key'], 'runner_pid': 99999999})
    Path(identity['launch_record']).parent.mkdir(parents=True, exist_ok=True)
    Path(identity['launch_record']).write_text(json.dumps({'status': 'operator_stopped'}))
    monkeypatch.setattr(dispatcher, '_is_our_runner', lambda *args, **kwargs: False)
    monkeypatch.setattr(dispatcher, '_find_runner', lambda *args, **kwargs: None)
    started_indices = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            self.pid = 60000 + len(started_indices)
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            started_indices.append(self.entry['index'])
            self.done = False
        def poll(self):
            if self.done:
                return 0
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps({'event': 'finish', 'key': self.entry['key'], 'status': 'complete',
                                         'exit_code': 0, 'output_accounting': {'complete': True}}) + '\n')
            self.done = True
            return 0
        def terminate(self): pass
        def wait(self, timeout=None): return self.poll()

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha, resume=True) == 0
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert started_indices == [1, 2]
    assert sum(row.get('event') == 'claim' and row.get('index') == 0 for row in events) == 1
    assert any(row.get('event') == 'abandoned' and row.get('index') == 0 for row in events)


def test_resume_tracks_live_orphan_without_serially_blocking_free_slot(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, _, plan = reviewed_plan
    _, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    entry, queue, queue_sha, journal, _, payload = dispatcher._claim_queue(plan, identity, 1, identity['cases'][0])
    assert entry['transport_retries'] == 0
    dispatcher._write_claim_queue(queue, payload)
    dispatcher._append_event(identity['ledger'], {'event': 'campaign_open', 'identity': identity, 'opened_at': 'test'})
    claim = {'claim_id': entry['key'], 'index': 0, 'slot': 1, 'key': entry['key'], 'queue': queue,
             'queue_sha256': queue_sha, 'journal': journal, 'source': identity['source'],
             'runtime': identity['runtime'], 'campaign': identity['campaign_id']}
    dispatcher._append_event(identity['ledger'], {'event': 'claim', 'identity_sha256': identity['identity_sha256'], **claim})
    dispatcher._append_event(identity['ledger'], {'event': 'runner_started', 'identity_sha256': identity['identity_sha256'],
                                                    'claim_id': entry['key'], 'runner_pid': 77777})
    Path(identity['launch_record']).parent.mkdir(parents=True, exist_ok=True)
    Path(identity['launch_record']).write_text(json.dumps({'status': 'operator_stopped'}))
    orphan_live = True
    checks = {'count': 0}
    def is_runner(pid, *args, **kwargs):
        if pid == 77777:
            checks['count'] += 1
            return orphan_live
        return False
    monkeypatch.setattr(dispatcher, '_is_our_runner', is_runner)
    monkeypatch.setattr(dispatcher, '_find_runner', lambda *args, **kwargs: 77777)
    started_indices = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            nonlocal orphan_live
            self.pid = 70000 + len(started_indices)
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            self.done = False
            started_indices.append(self.entry['index'])
            if self.entry['index'] == 1:
                assert orphan_live, 'free slot should be filled while the earlier claim is still running'
        def poll(self):
            nonlocal orphan_live
            if not self.done:
                with Path(self.journal).open('a') as stream:
                    stream.write(json.dumps({'event': 'finish', 'key': self.entry['key'], 'status': 'complete',
                                             'exit_code': 0, 'output_accounting': {'complete': True}}) + '\n')
                self.done = True
                if self.entry['index'] == 1:
                    orphan_live = False
            return 0
        def terminate(self): pass
        def wait(self, timeout=None): return self.poll()

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha, resume=True) == 0
    assert started_indices == [1, 2]
    assert checks['count'] >= 2


def test_graceful_stop_signals_active_supervisor_once_and_holds_claim(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    plan['slots'] = 1
    plan['cases'] = plan['cases'][:2]
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_sha = dispatcher.digest(plan_path)
    made = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            self.pid = 81818
            self.polls = 0
            self.terminated = 0
            made.append(self)
        def poll(self):
            self.polls += 1
            if self.terminated:
                return 1
            if self.polls == 1:
                dispatcher._stop_requested = True
            return None
        def terminate(self): self.terminated += 1
        def wait(self, timeout=None): return 1

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha) == 130
    assert len(made) == 1 and made[0].terminated == 1
    events = [json.loads(line) for line in (campaign_root / 'dispatch/claims.jsonl').read_text().splitlines()]
    assert [row['index'] for row in events if row.get('event') == 'claim'] == [0]
    assert any(row.get('event') == 'abandoned' and row.get('index') == 0 for row in events)


def test_drain_marker_preserves_active_runner_and_leaves_later_cases_unclaimed(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    plan['slots'] = 1
    plan['cases'] = plan['cases'][:2]
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_sha = dispatcher.digest(plan_path)
    made = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            self.pid = 82818
            self.polls = 0
            made.append(self)
        def poll(self):
            self.polls += 1
            if self.polls == 1:
                Path(self.queue).parents[3].joinpath('drain.requested').touch()
                return None
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps({'event': 'finish', 'key': self.entry['key'], 'status': 'complete',
                                         'exit_code': 0, 'output_accounting': {'complete': True}}) + '\n')
            return 0
        def terminate(self):
            raise AssertionError('drain must not terminate an active runner')
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    assert dispatcher.run_dispatcher(plan_path, plan_sha) == 0
    assert len(made) == 1 and made[0].polls >= 2
    launch = json.loads((campaign_root / 'dispatch/launch.json').read_text())
    assert launch['status'] == 'drained'
    assert launch['pending_indices'] == [1]
    events = [json.loads(line) for line in (campaign_root / 'dispatch/claims.jsonl').read_text().splitlines()]
    assert [row['index'] for row in events if row.get('event') == 'claim'] == [0]
    assert any(row.get('event') == 'campaign_draining' and row.get('active_indices') == [0]
               for row in events)
    assert any(row.get('event') == 'campaign_drained' and row.get('pending_indices') == [1]
               for row in events)
    _, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    dispatcher._verify_ledger_events(events, identity)


def test_spawned_runner_is_owned_before_runner_started_ledger_write(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    plan['slots'] = 1
    plan['cases'] = plan['cases'][:1]
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_sha = dispatcher.digest(plan_path)
    made = []

    class FakeRunner:
        def __init__(self, command, **kwargs):
            self.queue, self.entry, self.journal = _entry_for_queue(command)
            self.pid = 82828
            self.terminated = False
            made.append(self)
        def poll(self): return 1 if self.terminated else None
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 1

    monkeypatch.setattr(dispatcher.subprocess, 'Popen', FakeRunner)
    append = dispatcher._append_event
    def fail_after_spawn(path, event):
        if event.get('event') == 'runner_started':
            raise OSError('simulated ledger write failure')
        return append(path, event)
    monkeypatch.setattr(dispatcher, '_append_event', fail_after_spawn)
    with pytest.raises(OSError, match='simulated ledger write failure'):
        dispatcher.run_dispatcher(plan_path, plan_sha)
    assert len(made) == 1 and made[0].terminated
    events = [json.loads(line) for line in (campaign_root / 'dispatch/claims.jsonl').read_text().splitlines()]
    assert any(row.get('event') == 'abandoned' and row.get('index') == 0 for row in events)


@pytest.mark.parametrize('slots,accepted', [(6, True), (7, False)])
def test_owner_six_worker_ceiling(reviewed_plan, slots, accepted):
    path, _, _, plan = reviewed_plan
    plan['slots'] = slots
    path.write_text(json.dumps(plan, sort_keys=True))
    if accepted:
        _, identity = dispatcher.validate_plan(path, dispatcher.digest(path))
        assert identity['slots'] == 6
    else:
        with pytest.raises(ValueError, match='one through six'):
            dispatcher.validate_plan(path, dispatcher.digest(path))


def test_source_inventory_preserves_every_related_world_group(reviewed_plan):
    from luna_source_inventory import inventory
    path, _, _, plan = reviewed_plan
    source = Path(plan['source'])
    rows = [{'id':f'case-{i}','split':'train','source_groups':[f'world-{i}-a',f'world-{i}-b'],
             'curriculum':{'reference':{'root':[['main',{'code':'return 1;'}]]}},
             'semantics':{'files':{}}}for i in range(3)]
    source.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    witness=source.with_suffix('.inventory.json')
    witness.write_text(json.dumps(inventory(source,plan['cases'])))
    plan.update(schema='natlang.reviewed_luna_dispatch_plan/2',source_sha256=dispatcher.digest(source),
                source_inventory={'path':str(witness),'sha256':dispatcher.digest(witness)})
    plan['artifact_hashes'][str(source)]=dispatcher.digest(source)
    plan['artifact_hashes'][str(witness)]=dispatcher.digest(witness)
    path.write_text(json.dumps(plan,sort_keys=True))
    _,identity=dispatcher.validate_plan(path,dispatcher.digest(path))
    assert identity['cases'][0]['source_groups']==['world-0-a','world-0-b']
    entry,*_=dispatcher._claim_queue(plan,identity,1,identity['cases'][0])
    assert entry['source_groups']==['world-0-a','world-0-b']


def test_offline_plan_validation_uses_bound_original_workspace(reviewed_plan, monkeypatch, tmp_path):
    path,digest,_,_=reviewed_plan
    monkeypatch.setattr(dispatcher,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='canonical checkout'):
        dispatcher.validate_plan(path,digest)
    _,identity=dispatcher.validate_plan(path,digest,verification_root=ROOT)
    assert identity['plan_sha256']==digest
