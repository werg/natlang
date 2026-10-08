import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

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
    assert [row['index'] for row in identity['cases']] == [0, 1, 2]


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


def test_storage_floor_pauses_before_claim_and_resumes_pending_case(reviewed_plan, monkeypatch):
    _mock_controls(monkeypatch)
    plan_path, plan_sha, campaign_root, plan = reviewed_plan
    plan['slots'] = 1
    plan['cases'] = plan['cases'][:2]
    plan['min_free_mib'] = 10
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    plan_sha = dispatcher.digest(plan_path)

    free = iter([10, 0, 10, 10, 10])
    monkeypatch.setattr(dispatcher, '_free_space_mib', lambda _path: next(free, 10))
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
    assert claims[0][0] < pause < resume < claims[1][0]
    assert [index for _, index in claims] == [0, 1]
    assert events[pause]['next_case_index'] == 1
    assert events[pause]['disposition'] == 'no case claimed; pending case list preserved'
    assert made == [0, 1]
    assert not any(row.get('event') == 'abandoned' for row in events)
    _, identity = dispatcher.validate_plan(plan_path, plan_sha)
    identity['identity_sha256'] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    dispatcher._verify_ledger_events(events, identity)


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
