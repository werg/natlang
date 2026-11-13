import json
import hashlib
import io
import time
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import pytest
import scripts.run_bonsai_queue as queue_module
from scripts.run_bonsai_queue import run_queue, local_decode_progress as read_local_decode_progress


def test_journal_spool_root_uses_authenticated_campaign_id_not_storage_basename():
    from scripts.run_bonsai_queue import _approved_journal_spool_root
    campaign = 'luna-campaign-a'
    storage_name = 'campaign-a'
    jobs = str(Path(queue_module.resolve_machine_path('data_hdd')) /
               'natlang-luna-campaigns' / storage_name / 'slot-01' / 'jobs')
    approved = _approved_journal_spool_root(
        queue_module.resolve_machine_path('data_nvme', 'natlang-teacher-journal-spool', campaign), jobs, campaign)
    assert approved.endswith('/natlang-teacher-journal-spool/' + campaign)
    with pytest.raises(ValueError, match='configured data_nvme'):
        _approved_journal_spool_root('/tmp/unbound-spool', jobs, campaign)
    with pytest.raises(ValueError, match='configured data_nvme'):
        _approved_journal_spool_root(
            queue_module.resolve_machine_path('data_nvme', 'natlang-teacher-journal-spool', storage_name),
            jobs, campaign)

ROOT = Path(__file__).resolve().parents[1]
TS_HOST = ROOT / 'ts-host'


@pytest.fixture(autouse=True)
def own_collector_process(monkeypatch):
    # These tests fake the collector child process; the shared node server is covered by its own test.
    monkeypatch.setenv('NATLANG_COLLECTOR_SHARED', '0')


@pytest.fixture(autouse=True)
def no_live_server_queries(monkeypatch):
    monkeypatch.setattr(queue_module, 'local_decode_progress', lambda *args: None)
    original = queue_module.urllib.request.urlopen
    class Healthy:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
    def fixture_urlopen(url, *args, **kwargs):
        if str(url).endswith('/health'):
            return Healthy()
        return original(url, *args, **kwargs)
    monkeypatch.setattr(queue_module.urllib.request, 'urlopen', fixture_urlopen)


def source_rows(path, count=1):
    rows = []
    for index in range(count):
        root = f'case-{index}.nl'
        rows.append({
            'version': 'natlang.program/2', 'id': f'case-{index}', 'kind': 'lambda_source',
            'source_layout': {'root': root, 'subfunctions': {}},
            'semantics': {'root': root, 'files': {root: f'export default ({index})'}},
        })
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return rows


def queue_entry(tmp_path, *, key='case', index=0, count=1, source=None, output=None, max_turns=20):
    source = Path(source or (tmp_path / 'source.ir.jsonl'))
    if not source.exists():
        source_rows(source, max(1, index + count))
    return dict(key=key, source=str(source), jobs=str(tmp_path / 'jobs'),
                output=str(output or (tmp_path / 'teacher.results.jsonl')), index=index,
                count=count, seed=1, log=str(tmp_path / f'{key}.log'), max_turns=max_turns)


def write_exact_export(command):
    """Emit fixture artifacts matching current collector identity/accounting checks."""
    source, jobs_dir, output = map(Path, command[3:6])
    start = int(command[command.index('--start') + 1])
    count = int(command[command.index('--limit') + 1])
    entry = {'source': str(source), 'jobs': str(jobs_dir), 'output': str(output),
             'index': start, 'count': count}
    jobs = queue_module.resolve_entry_jobs(entry, TS_HOST)
    source_data = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    jobs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        result = {
            'task': {'program_ir': source_data[job['index']]},
            'provenance': {'program_ir_sha256': job['digest']},
            'outcome': {'status': 'done', 'accepted': True},
            'trajectory': [],
        }
        (jobs_dir / f"{job['key']}.result.json").write_text(json.dumps(result))
        rows.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = {
        'version': 'natlang.teacher_batch.native/1',
        'range': {'start': start, 'count': count}, 'completed': count, 'missing': [],
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'source': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    Path(str(output) + '.manifest.json').write_text(json.dumps(manifest))


def test_storage_floor_pauses_without_attempting_or_negatively_labelling_case(tmp_path,monkeypatch):
    from types import SimpleNamespace
    entry=queue_entry(tmp_path)
    queue=tmp_path/'queue.jsonl';raw=json.dumps(entry)+'\n';queue.write_text(raw)
    journal=tmp_path/'journal.jsonl'
    monkeypatch.setattr(queue_module.shutil,'disk_usage',lambda _:SimpleNamespace(free=1024))
    original_popen=queue_module.subprocess.Popen
    def forbidden_launch(*args,**kwargs):
        if '--input-type=module' in args[0]:
            return original_popen(*args,**kwargs)  # exact source digest, no model
        raise AssertionError('storage-paused case must not launch a collector')
    monkeypatch.setattr(queue_module.subprocess,'Popen',forbidden_launch)
    assert run_queue(queue,journal,TS_HOST,min_free_mib=20480)==75
    rows=[json.loads(line)for line in journal.read_text().splitlines()]
    assert [row['event']for row in rows]==['storage_pause']
    assert rows[0]['key']==entry['key'] and 'restart same queue' in rows[0]['disposition']
    assert queue.read_text()==raw and not Path(entry['output']).exists()


def test_output_accounting_resolves_manifest_source_path_but_keeps_hash_checks(tmp_path):
    source = tmp_path / 'source.ir.jsonl'
    source_rows(source, 1)
    output = tmp_path / 'teacher.results.jsonl'
    jobs = tmp_path / 'jobs'
    write_exact_export(['node', 'collector', 'unused', str(source), str(jobs), str(output),
                        '--start', '0', '--limit', '1'])
    entry = dict(source=os.path.relpath(source, ROOT), jobs=str(jobs), output=str(output), index=0, count=1)

    manifest_path = Path(str(output) + '.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['source'] = str(source.resolve())
    manifest_path.write_text(json.dumps(manifest))
    accounting = queue_module.output_accounting(entry, TS_HOST)
    assert accounting['complete'] is True
    assert accounting['manifest_valid'] is True
    assert accounting['manifest_errors'] == []

    manifest['source_sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(manifest))
    accounting = queue_module.output_accounting(entry, TS_HOST)
    assert accounting['complete'] is False
    assert 'manifest_source_sha256_mismatch' in accounting['manifest_errors']


def test_provider_observations_ignore_known_cleanup_and_count_unknown_phases(tmp_path):
    from scripts.run_bonsai_queue import ProviderObservations
    path = tmp_path / 'provider.log'
    path.write_text('')
    observer = ProviderObservations(path)
    started = datetime.now(timezone.utc) - timedelta(seconds=1)
    observed = datetime.now(timezone.utc)
    iso = lambda value: value.isoformat().replace('+00:00', 'Z')
    events = [
        {'event': 'provider_request_phase', 'phase': 'provider_close', 'status': 'started',
         'provider': 'openrouter'},
        {'event': 'provider_request_phase', 'phase': 'provider_close', 'status': 'completed',
         'provider': 'openrouter'},
        {'event': 'provider_request_phase', 'phase': 'provider_action_cycle', 'status': 'started',
         'role': 'teacher', 'provider': 'openrouter', 'request_ordinal': None},
        {'event': 'provider_request_phase', 'phase': 'provider_turn', 'status': 'started',
         'role': 'teacher', 'provider': 'openrouter', 'request_ordinal': 1},
        {'event': 'provider_stream_progress', 'status': 'progress', 'role': 'teacher',
         'provider': 'openrouter', 'request_ordinal': 1, 'deltaEvents': 1, 'deltaBytes': 12,
         'startedAt': iso(started), 'observedAt': iso(observed)},
        {'event': 'provider_request_phase', 'phase': 'future_unknown_phase', 'status': 'started',
         'role': 'teacher', 'provider': 'openrouter', 'request_ordinal': 1},
    ]
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))
    progressed, snapshot = observer.poll()
    assert progressed is True
    assert snapshot['invalid_events'] == 1, 'only the unknown phase is invalid; cleanup and action-cycle records are known'
    assert snapshot['pending_requests'] == [{'role': 'teacher', 'provider': 'openrouter',
                                             'request_ordinal': 1, 'phase': 'provider_turn'}]
    assert snapshot['latest_delta_at'] == observed.timestamp()


def install_child(monkeypatch, child_class):
    # A faked collector child is the invocation's own process: the shared node server (default) is not faked.
    monkeypatch.setenv('NATLANG_COLLECTOR_SHARED', '0')
    real_subprocess = queue_module.subprocess
    class SubprocessProxy:
        Popen = child_class
        TimeoutExpired = real_subprocess.TimeoutExpired
        STDOUT = real_subprocess.STDOUT
        check_output = real_subprocess.check_output
    monkeypatch.setattr(queue_module, 'subprocess', SubprocessProxy)


def successful_child(monkeypatch, return_codes=None, timeout_first=False):
    """Fake process that writes the exact current collector result/export contract on success."""
    created = []
    codes = list(return_codes or [])
    class Child:
        def __init__(self, command, **kwargs):
            self.command = command
            self.index = len(created)
            self.terminated = self.killed = False
            self.wait_calls = 0
            created.append(self)
        def wait(self, timeout=None):
            self.wait_calls += 1
            if timeout_first and self.index == 0:
                if self.wait_calls == 1 or (self.terminated and not self.killed):
                    raise subprocess.TimeoutExpired(self.command, timeout)
            if self.terminated or self.killed:
                return 0
            code = codes[self.index] if self.index < len(codes) else 0
            if code == 0:
                self._write_exact_export()
            return code
        def terminate(self): self.terminated = True
        def kill(self): self.killed = True
        def _write_exact_export(self):
            write_exact_export(self.command)
    install_child(monkeypatch, Child)
    return created


def test_timeout_kills_stuck_child_advances_and_resume_skips_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_module, 'failure_cooldown', lambda streak: 0)
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    source = tmp_path / 'source.ir.jsonl'
    source_rows(source, 2)
    entries = [queue_entry(tmp_path, key=str(i), index=i, source=source,
                           output=tmp_path / 'out.jsonl') for i in range(2)]
    queue.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    processes = successful_child(monkeypatch, timeout_first=True)
    run_queue(queue, journal, TS_HOST, seconds=1)
    assert processes[0].terminated and processes[0].killed
    assert len(processes) == 2
    finished = [json.loads(row) for row in journal.read_text().splitlines() if json.loads(row)['event'] == 'finish']
    assert [row['status'] for row in finished] == ['timeout', 'complete']
    run_queue(queue, journal, TS_HOST, seconds=1)
    assert len(processes) == 2


def test_a_stop_that_reached_the_child_first_records_no_attempt(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    source = tmp_path / 'source.ir.jsonl'
    source_rows(source, 2)
    entries = [queue_entry(tmp_path, key=str(i), index=i, source=source, output=tmp_path / 'out.jsonl') for i in range(2)]
    queue.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    processes = successful_child(monkeypatch, return_codes=[143])   # the child died of the same SIGTERM
    monkeypatch.setattr(queue_module, 'STOP_REQUESTED', [15])        # the handler ran, wait() returned normally
    with pytest.raises(KeyboardInterrupt):
        run_queue(queue, journal, TS_HOST, seconds=1)
    rows = [json.loads(row) for row in journal.read_text().splitlines()]
    assert [row['event'] for row in rows] == ['start'] and len(processes) == 1   # pending again on the next start


def test_entry_turn_budget_is_passed_and_recorded(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path, key='world', max_turns=48)
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1)
    command = commands[0].command
    assert command[command.index('--max-turns') + 1] == '48'
    assert json.loads(journal.read_text().splitlines()[0])['max_turns'] == 48
    for value in [True, 0, -1, '48']:
        entry['max_turns'] = value
        queue.write_text(json.dumps(entry) + '\n')
        with pytest.raises(ValueError, match='max_turns'):
            run_queue(queue, tmp_path / 'fresh.jsonl', TS_HOST, seconds=1)


def test_independent_judge_reaches_collector_and_invalid_later_binding_starts_nothing(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path)
    entry['judge_model'] = {'model_id': 'actual-judge', 'provider': 'openai-codex'}
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1, model_id='actual-teacher')
    command = commands[0].command
    assert command[command.index('--judge-model-id') + 1] == 'actual-judge'
    assert command[command.index('--judge-provider') + 1] == 'openai-codex'
    for judge in ({'model_id': 'actual-teacher', 'provider': 'openai-codex'},
                  {'model_id': 'actual-judge'},
                  {'model_id': 'actual-judge', 'provider': 'openai-codex', 'server': 'http://localhost'},
                  {'model_id': 'actual-judge', 'provider': ''}):
        queue.write_text(json.dumps(entry) + '\n' + json.dumps({**entry, 'key': 'later', 'judge_model': judge}) + '\n')
        before = len(commands)
        with pytest.raises(ValueError, match='judge_model'):
            run_queue(queue, tmp_path / 'unstarted.jsonl', TS_HOST, seconds=1, model_id='actual-teacher')
        assert len(commands) == before


def test_entry_action_output_cap_is_passed_and_validated(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path, key='action-cap')
    entry['action_max_tokens'] = 4096
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1)
    assert commands[0].command[commands[0].command.index('--action-max-tokens') + 1] == '4096'
    for value in [True, 0, -1, '4096']:
        entry['action_max_tokens'] = value
        queue.write_text(json.dumps(entry) + '\n')
        with pytest.raises(ValueError, match='action_max_tokens'):
            run_queue(queue, tmp_path / 'fresh.jsonl', TS_HOST, seconds=1)


def test_entry_system_file_is_passed_and_validated(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    system = tmp_path / 'system.md'
    system.write_text('You are running one call.\n\n<natlang_program_guidance>\nCheck every field.\n</natlang_program_guidance>\n')
    entry = queue_entry(tmp_path, key='system-file')
    entry['system_file'] = str(system)
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1)
    assert commands[0].command[commands[0].command.index('--system-file') + 1] == str(system)
    for value in [str(tmp_path / 'missing.md'), 3]:
        entry['system_file'] = value
        queue.write_text(json.dumps(entry) + '\n')
        with pytest.raises(ValueError, match='system_file'):
            run_queue(queue, tmp_path / 'fresh.jsonl', TS_HOST, seconds=1)


def test_entry_transport_retry_budget_is_passed_and_validated(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path, key='retry-budget')
    entry['transport_retries'] = 0
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1)
    assert commands[0].command[commands[0].command.index('--transport-retries') + 1] == '0'
    for value in [True, -1, '0']:
        entry['transport_retries'] = value
        queue.write_text(json.dumps(entry) + '\n')
        with pytest.raises(ValueError, match='transport_retries'):
            run_queue(queue, tmp_path / 'fresh.jsonl', TS_HOST, seconds=1)


def test_repeated_action_shapes_do_not_imply_repeated_requests(tmp_path):
    from scripts.run_bonsai_queue import partial_metrics
    entry = queue_entry(tmp_path, key='fixture')
    job = queue_module.resolve_entry_jobs(entry, TS_HOST)[0]
    turns = [dict(request_sha256=digest, response=dict(raw_response={"provider": "fixture"},
             calls=[["return_result", {"status": "success", "value": False}]], completion_tokens=2))
             for digest in ["input-a", "input-b"]]
    path = tmp_path / 'jobs' / f"{job['key']}.partial.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(program_id=job['program_id'],
                                    provenance={'program_ir_sha256': job['digest']}, turns=turns)))
    metrics = partial_metrics(entry)
    assert metrics['repeated_action_sets'] == 1
    assert metrics['repeated_request_hashes'] == 0
    assert metrics['unique_request_hashes'] == 2
    turns.append(turns[0])
    path.write_text(json.dumps(dict(program_id=job['program_id'],
                                    provenance={'program_ir_sha256': job['digest']}, turns=turns)))
    assert partial_metrics(entry)['repeated_request_hashes'] == 1


def test_active_spool_partial_metrics_require_exact_case_lease(tmp_path):
    from scripts.run_bonsai_queue import partial_metrics
    entry = queue_entry(tmp_path, key='spooled')
    job = queue_module.resolve_entry_jobs(entry, TS_HOST)[0]
    spool = tmp_path / 'nvme-spool'
    identity = hashlib.sha256(f"{Path(entry['jobs']).resolve()}:{job['key']}".encode()).hexdigest()
    lease_dir = spool / '.leases'; case_dir = spool / 'cases' / identity
    lease_dir.mkdir(parents=True); case_dir.mkdir(parents=True)
    lease_path = lease_dir / f'{identity}.json'
    lease = {'schema':'natlang.teacher_journal_spool_lease/1','key':identity,
             'case_identity_sha256':identity,'pid':os.getpid(),'per_case_limit_bytes':268435456}
    lease_path.write_text(json.dumps(lease))
    partial = {'program_id':job['program_id'],
               'provenance':{'program_ir_sha256':job['digest']},
               'turns':[{'request_sha256':'actual-request','response':{
                   'raw_response':{'sampled':True},'completion_tokens':13,'calls':[]}}]}
    (case_dir / f"{job['key']}.partial.json").write_text(json.dumps(partial))
    metrics = partial_metrics(entry, spool)
    assert metrics['saved_turns'] == 1
    assert metrics['fresh_model_replies'] == 1
    assert metrics['completion_tokens'] == 13

    lease['key'] = '0' * 64
    lease_path.write_text(json.dumps(lease))
    assert partial_metrics(entry, spool)['saved_turns'] == 0


def test_local_pair_shares_one_collector_and_one_global_two_request_cap(tmp_path, monkeypatch):
    from scripts.run_bonsai_queue import partial_metrics
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    source = tmp_path / 'source.ir.jsonl'; source_rows(source, 2)
    entry = queue_entry(tmp_path, key='pair', count=2, source=source,
                        output=tmp_path / 'out.jsonl')
    entry['members'] = [{'key': 'one'}, {'key': 'two'}]
    queue.write_text(json.dumps(entry) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1, model_concurrency=2)
    command = commands[0].command
    assert command[command.index('--workers') + 1] == '2'
    assert command[command.index('--limit') + 1] == '2'
    assert command[command.index('--model-concurrency') + 1] == '2'
    assert [json.loads(line)['key'] for line in journal.read_text().splitlines()
            if json.loads(line)['event'] == 'finish'] == ['pair', 'one', 'two']
    jobs = queue_module.resolve_entry_jobs(entry, TS_HOST)
    for i, job in enumerate(jobs):
        (tmp_path / 'jobs' / f"{job['key']}.partial.json").write_text(json.dumps({
            'program_id': job['program_id'], 'provenance': {'program_ir_sha256': job['digest']},
            'turns': [{'request_sha256': str(i), 'response': {'raw_response': {'fixture': True}, 'calls': [], 'completion_tokens': 3}}]}))
    assert partial_metrics(entry)['fresh_model_replies'] == 2
    with pytest.raises(ValueError, match='single-case'):
        run_queue(queue, tmp_path / 'provider.jsonl', TS_HOST, provider='openai-codex', model_id='gpt-6-luna', model_concurrency=1)


@pytest.mark.parametrize('count', [4, 5])
def test_larger_batches_use_one_global_cap(tmp_path, monkeypatch, count):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    source = tmp_path / 'source.ir.jsonl'; source_rows(source, count)
    queue.write_text(json.dumps(queue_entry(tmp_path, key='batch', source=source,
        output=tmp_path / 'out.jsonl', count=count)) + '\n')
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1, model_concurrency=4)
    assert len(commands) == 1
    command = commands[0].command
    assert command[command.index('--workers') + 1] == str(count)
    assert command[command.index('--model-concurrency') + 1] == '4'
    with pytest.raises(ValueError, match='single-case'):
        run_queue(queue, tmp_path / 'provider.jsonl', TS_HOST, provider='openai-codex', model_concurrency=1)


def test_batch_metrics_include_completed_and_running_roots(tmp_path):
    from scripts.run_bonsai_queue import partial_metrics
    source = tmp_path / 'source.ir.jsonl'; source_rows(source, 2)
    entry = queue_entry(tmp_path, key='batch', count=2, source=source)
    jobs = queue_module.resolve_entry_jobs(entry, TS_HOST)
    jobs_dir = Path(entry['jobs']); jobs_dir.mkdir(parents=True, exist_ok=True)
    (jobs_dir / f"{jobs[0]['key']}.result.json").write_text(json.dumps({
        'task': {'program_ir': json.loads(source.read_text().splitlines()[0])},
        'provenance': {'program_ir_sha256': jobs[0]['digest']},
        'trajectory': [{'request_sha256': 'done', 'raw_response_sha256': 'hash',
                        'model_response': {'calls': [], 'completion_tokens': 7}}]}))
    (jobs_dir / f"{jobs[1]['key']}.partial.json").write_text(json.dumps({
        'program_id': jobs[1]['program_id'], 'provenance': {'program_ir_sha256': jobs[1]['digest']},
        'turns': [{'request_sha256': 'running', 'response': {'raw_response': {'fixture': True},
                  'calls': [], 'completion_tokens': 3}}]}))
    metrics = partial_metrics(entry)
    assert metrics['fresh_model_replies'] == 2
    assert metrics['completion_tokens'] == 10


def test_local_decode_metrics_are_finite_and_parse_failures_are_ignored(monkeypatch):
    import io
    for value, expected in [('42', 42), ('NaN', None), ('-1', None), ('bad', None)]:
        monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: io.BytesIO(
            ('# metric\nllamacpp:n_decode_total ' + value + '\n').encode()))
        assert read_local_decode_progress() == expected


@pytest.mark.parametrize('decoding', [True, False])
@pytest.mark.parametrize('native_server', [True, False])
def test_long_reply_decoding_is_activity_but_a_motionless_request_times_out(tmp_path, monkeypatch, decoding, native_server):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path, key='long')
    queue.write_text(json.dumps(entry) + '\n')
    clock = [0]
    monkeypatch.setattr('scripts.run_bonsai_queue.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('scripts.run_bonsai_queue.local_decode_progress', lambda *args: clock[0] if decoding else None)
    class Child:
        def __init__(self, command, **kwargs): self.terminated = False; self.calls = 0; self.command = command
        def wait(self, timeout=None):
            if self.terminated: return 0
            clock[0] += 30; self.calls += 1
            if self.calls <= 12: raise subprocess.TimeoutExpired(self.command, timeout)
            # The successful test path must provide the exact native result/output proof.
            write_exact_export(self.command)
            return 0
        def terminate(self): self.terminated = True
    install_child(monkeypatch, Child)
    run_queue(queue, journal, TS_HOST, seconds=600, no_observation_seconds=360,
              **({'server': 'http://127.0.0.1:43217', 'neuralese_server': True} if native_server else {}))
    finish = json.loads(journal.read_text().splitlines()[-1])
    assert finish['status'] == ('complete' if decoding else 'no_observation_limit')
    assert finish['resource_limit_reason'] == (None if decoding else 'no_observation_limit')


def test_provider_failure_cooldown_doubles_caps_and_does_not_delay_rejections(tmp_path, monkeypatch):
    import scripts.run_bonsai_queue as module
    monkeypatch.setattr(module.random, 'random', lambda: 0.5)
    assert [module.failure_cooldown(i) for i in range(1, 7)] == [30, 60, 120, 240, 300, 300]
    now = [1000.0]
    monkeypatch.setattr(module.time, 'time', lambda: now[0])
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(module.time, 'sleep', sleep)
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    source = tmp_path / 'source.ir.jsonl'; source_rows(source, 4)
    entries = [queue_entry(tmp_path, key=str(i), index=i, source=source,
                           output=tmp_path / 'out.jsonl') for i in range(4)]
    queue.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    commands = []
    commands = successful_child(monkeypatch, return_codes=[2, 1, 0, 0])
    run_queue(queue, journal, TS_HOST, provider='fixture')
    assert sum(waits) == 90
    assert len(commands) == 4
    assert commands[0].command[commands[0].command.index('--retry-delay-ms') + 1] == '15000'
    finishes = [json.loads(line) for line in journal.read_text().splitlines() if json.loads(line)['event'] == 'finish']
    assert [row['failure_streak'] for row in finishes] == [1, 2, 0, 0]
    # Restart still observes the original deadline, rather than resetting the failure streak.
    journal.write_text(json.dumps(finishes[0]) + '\n')
    now[0] = 1010
    commands.clear()
    entries = entries[:2]
    queue.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    waits.clear()
    commands = successful_child(monkeypatch, return_codes=[2])
    run_queue(queue, journal, TS_HOST, provider='fixture')
    assert sum(waits) == 20
    assert json.loads(journal.read_text().splitlines()[-1])['failure_streak'] == 2


def test_provider_retry_deadline_is_not_a_stall(tmp_path, monkeypatch):
    from scripts.run_bonsai_queue import retry_waiting
    entry = queue_entry(tmp_path)
    job = queue_module.resolve_entry_jobs(entry, TS_HOST)[0]
    path = Path(entry['jobs']) / f"{job['key']}.retry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr('scripts.run_bonsai_queue.time.time', lambda: 1000)
    path.write_text(json.dumps(dict(until=1001000, program_id=job['program_id'])))
    assert retry_waiting(entry)
    path.write_text(json.dumps(dict(until=999000, program_id=job['program_id'])))
    assert not retry_waiting(entry)
    path.write_text('broken')
    assert not retry_waiting(entry)


def test_exhausted_provider_retry_keeps_minimum_delay(tmp_path):
    from scripts.run_bonsai_queue import retry_deadline
    entry = queue_entry(tmp_path)
    job = queue_module.resolve_entry_jobs(entry, TS_HOST)[0]
    error_path = Path(entry['jobs']) / f"{job['index']:06d}.error.json"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(json.dumps(dict(retry_not_before=1234567890, program_id=job['program_id'])))
    assert retry_deadline(entry) == 1234567.89


def test_fresh_queue_creates_collector_parents_before_launch(tmp_path, monkeypatch):
    entry = queue_entry(tmp_path, key='fresh', output=tmp_path / 'outputs' / 'nested' / 'result.jsonl')
    entry['jobs'] = str(tmp_path / 'jobs-parent' / 'nested' / 'jobs')
    entry['log'] = str(tmp_path / 'logs' / 'nested' / 'collector.log')
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    queue.write_text(json.dumps(entry) + '\n')

    class Child:
        def __init__(self, command, **kwargs):
            self.command = command
            assert Path(entry['jobs']).is_dir()
            assert Path(entry['output']).parent.is_dir()
            assert Path(entry['log']).parent.is_dir()
        def wait(self, timeout=None):
            write_exact_export(self.command)
            return 0

    install_child(monkeypatch, Child)
    run_queue(queue, journal, TS_HOST, seconds=1, provider='openai-codex')
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert events[-1]['event'] == 'finish'
    assert events[-1]['status'] == 'complete'
    assert Path(entry['log']).is_file()


def test_provider_request_controls_are_forwarded_by_canonical_runner(tmp_path, monkeypatch):
    entry = queue_entry(tmp_path)
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    queue.write_text(json.dumps(entry) + '\n')
    config = tmp_path / 'provider-request-config.json'
    config.write_text(json.dumps({'piPayload': {'tool_choice': 'auto'}, 'omitPayloadKeys': ['seed']}))
    commands = successful_child(monkeypatch)
    run_queue(queue, journal, TS_HOST, seconds=1, provider='openrouter',
              provider_request_config=config)
    command = commands[0].command
    assert command[command.index('--provider-request-config') + 1] == str(config.resolve())
    assert json.loads(journal.read_text().splitlines()[-1])['status'] == 'complete'

    help_result = subprocess.run([__import__('sys').executable, str(ROOT / 'scripts/run_bonsai_queue.py'), '--help'],
                                 text=True, capture_output=True)
    assert help_result.returncode == 0
    assert '--provider-request-config' in help_result.stdout


def test_invalid_provider_controls_fail_before_attempt_start(tmp_path, monkeypatch):
    entry = queue_entry(tmp_path)
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    queue.write_text(json.dumps(entry) + '\n')
    config = tmp_path / 'provider-request-config.json'
    config.write_text('[]')
    commands = successful_child(monkeypatch)
    with pytest.raises(ValueError, match='explicit provider'):
        run_queue(queue, journal, TS_HOST, provider_request_config=config)
    with pytest.raises(ValueError, match='JSON object'):
        run_queue(queue, journal, TS_HOST, provider='openrouter', provider_request_config=config)
    config.write_text('{invalid')
    with pytest.raises(json.JSONDecodeError):
        run_queue(queue, journal, TS_HOST, provider='openrouter', provider_request_config=config)
    assert not commands
    assert not journal.exists()


def test_exact_terminated_transport_error_is_sha_bound_in_output_accounting(tmp_path, monkeypatch):
    entry = queue_entry(tmp_path)
    jobs = Path(entry['jobs']); jobs.mkdir()
    job = {'key': '000000-fixture', 'index': 0, 'program_id': 'case-0', 'digest': 'a' * 64}
    monkeypatch.setattr(queue_module, 'resolve_entry_jobs', lambda *args: [job])
    monkeypatch.setattr(queue_module, 'frozen_record_digests', lambda *args: [])
    error = jobs / '000000.error.json'
    error.write_text(json.dumps({'index': 0, 'program_id': 'case-0', 'program_ir_sha256':job['digest'],
                                 'source_sha256':hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest(),
                                 'error': 'TypeError: terminated'}))
    accounting = queue_module.output_accounting(entry)
    state = accounting['job_states'][0]
    assert state['state'] == 'transport_failed'
    assert state['terminal_error'] == {'error': 'TypeError: terminated', 'path': str(error),
                                       'sha256': hashlib.sha256(error.read_bytes()).hexdigest(),
        'program_ir_sha256':job['digest'],'source_sha256':hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest(),
        'record_binding_verified':False}
    # The same cut through a natlang call's wrapper (an application-controller root, ts-host 9f13cce63).
    error.write_text(json.dumps({'index': 0, 'program_id': 'case-0', 'program_ir_sha256':job['digest'],
                                 'source_sha256':hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest(),
                                 'error': 'NatlangCallError: nl@eval:1: failed: terminated'}))
    assert queue_module.output_accounting(entry)['job_states'][0]['state'] == 'transport_failed'
    # An unrelated model/runtime failure is not declared infrastructure by this exact transport case.
    error.write_text(json.dumps({'index': 0, 'program_id': 'case-0', 'program_ir_sha256':job['digest'],
                                 'source_sha256':hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest(),
                                 'error': 'unknown action terminated_work'}))
    assert queue_module.output_accounting(entry)['job_states'][0]['state'] == 'failed'


@pytest.mark.parametrize('binding',['correct','wrong_record','wrong_source','legacy'])
def test_native_terminal_accounting_requires_exact_record_source_before_scoring(tmp_path,monkeypatch,binding):
    from natlang_neuralese.eval.task_packet import classify_case
    entry=queue_entry(tmp_path);jobs=Path(entry['jobs']);jobs.mkdir()
    job={'key':'000000-fixture','index':0,'program_id':'case-0','digest':'a'*64}
    monkeypatch.setattr(queue_module,'resolve_entry_jobs',lambda *args:[job])
    monkeypatch.setattr(queue_module,'frozen_record_digests',lambda *args:[])
    source_sha=hashlib.sha256(Path(entry['source']).read_bytes()).hexdigest()
    output=Path(entry['output']);output.write_text('')
    Path(str(output)+'.manifest.json').write_text(json.dumps({
        'version':'natlang.teacher_batch.native/1','range':{'start':entry['index'],'count':1},
        'completed':0,'missing':[0],'output_sha256':hashlib.sha256(b'').hexdigest(),
        'source':entry['source'],'source_sha256':source_sha}))
    error={'index':0,'program_id':'case-0','program_ir_sha256':job['digest'],'source_sha256':source_sha,
        'error':'Error: model stream error: '+json.dumps({'code':'neuralese-native-state-open',
            'message':'neuralese-native-state-open: generation stopped inside a produce span'})}
    if binding=='wrong_record':error['program_ir_sha256']='b'*64
    elif binding=='wrong_source':error['source_sha256']='b'*64
    elif binding=='legacy':error.pop('program_ir_sha256');error.pop('source_sha256')
    path=jobs/'000000.error.json';path.write_text(json.dumps(error))
    accounting=queue_module.output_accounting(entry)
    assert accounting['manifest_valid'] and not accounting['complete']
    state=accounting['job_states'][0]
    finish={'status':'incomplete','output_accounting':accounting}
    if binding=='correct':
        assert state['terminal_error']['record_binding_verified']
        assert classify_case(None,finish)=='model_incomplete'
        # Real terminal transport errors retain their existing precedence.
        error['error']='TypeError: terminated';path.write_text(json.dumps(error))
        assert classify_case(None,{'status':'incomplete','output_accounting':queue_module.output_accounting(entry)})=='infrastructure_error'
        error['error']='unknown runtime failure';path.write_text(json.dumps(error))
        assert classify_case(None,{'status':'incomplete','output_accounting':queue_module.output_accounting(entry)})=='unresolved'
    else:
        assert state['state']=='unverified_terminal_error' and 'terminal_error' not in state
        assert classify_case(None,finish)=='unresolved'


def test_same_id_index_changed_source_cannot_reuse_old_native_terminal_error(tmp_path):
    from natlang_neuralese.eval.task_packet import classify_case
    entry=queue_entry(tmp_path);original=queue_module.resolve_entry_jobs(entry,TS_HOST)[0]
    source=Path(entry['source']);old_source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    value=json.loads(source.read_text());value['semantics']['files']['case-0.nl']='export default (42)'
    source.write_text(json.dumps(value)+'\n');entry.pop('_resolved_jobs')
    current=queue_module.resolve_entry_jobs(entry,TS_HOST)[0]
    assert current['program_id']==original['program_id'] and current['index']==original['index']
    assert current['digest']!=original['digest']
    jobs=Path(entry['jobs']);jobs.mkdir()
    (jobs/'000000.error.json').write_text(json.dumps({'index':0,'program_id':original['program_id'],
        'program_ir_sha256':original['digest'],'source_sha256':old_source_sha,
        'error':'Error: model stream error: '+json.dumps({'code':'neuralese-native-state-open',
            'message':'neuralese-native-state-open: generation stopped inside a produce span'})}))
    output=Path(entry['output']);output.write_text('')
    Path(str(output)+'.manifest.json').write_text(json.dumps({
        'version':'natlang.teacher_batch.native/1','range':{'start':0,'count':1},
        'completed':0,'missing':[0],'output_sha256':hashlib.sha256(b'').hexdigest(),
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
    accounting=queue_module.output_accounting(entry,TS_HOST)
    assert accounting['manifest_valid'] and accounting['job_states'][0]['state']=='unverified_terminal_error'
    assert classify_case(None,{'status':'incomplete','output_accounting':accounting})=='unresolved'


def test_cooperative_drain_before_case_does_not_start_or_relabel_unstarted(tmp_path,monkeypatch):
    queue,journal,drain=tmp_path/'queue.jsonl',tmp_path/'journal.jsonl',tmp_path/'DRAIN'
    entry=queue_entry(tmp_path);queue.write_text(json.dumps(entry)+'\n');drain.touch()
    monkeypatch.setattr(queue_module.subprocess,'Popen',lambda *a,**k:pytest.fail('request after drain'))
    assert run_queue(queue,journal,TS_HOST,drain_file=drain)==75
    rows=[json.loads(x) for x in journal.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['event']=='operator_drain'
    assert rows[0]['unstarted_keys']==['case'] and rows[0]['completed_keys']==[]
    assert rows[0]['queue_sha256']==hashlib.sha256(queue.read_bytes()).hexdigest()


def test_cooperative_drain_finishes_active_case_then_resumes_only_remaining(tmp_path,monkeypatch):
    queue,journal,drain=tmp_path/'queue.jsonl',tmp_path/'journal.jsonl',tmp_path/'DRAIN'
    source=tmp_path/'source.ir.jsonl';source_rows(source,2)
    entries=[queue_entry(tmp_path,key=f'case-{i}',index=i,source=source,output=tmp_path/f'out-{i}.jsonl') for i in range(2)]
    queue.write_text(''.join(json.dumps(row)+'\n' for row in entries));calls=[]
    class Child:
        def __init__(self,command,**kwargs):self.command=command;calls.append(command)
        def wait(self,timeout=None):
            write_exact_export(self.command)
            if len(calls)==1:drain.touch()
            return 0
        def terminate(self):pytest.fail('drain terminated an active case')
        def kill(self):pytest.fail('drain killed an active case')
    install_child(monkeypatch,Child)
    assert run_queue(queue,journal,TS_HOST,drain_file=drain)==75
    rows=[json.loads(x) for x in journal.read_text().splitlines()]
    assert [r['key'] for r in rows if r['event']=='finish']==['case-0']
    event=rows[-1];assert event['event']=='operator_drain' and event['completed_keys']==['case-0'] and event['unstarted_keys']==['case-1']
    drain.unlink();assert run_queue(queue,journal,TS_HOST,drain_file=drain)==0
    assert len(calls)==2 and [c[c.index('--start')+1] for c in calls]==['0','1']


def test_native_decode_observer_uses_actual_loopback_origin_only(monkeypatch):
    seen = []
    def urlopen(url, **kwargs):
        seen.append(url)
        return io.BytesIO(b'natlang_native_decode_tokens_total 73\n')
    monkeypatch.setattr(queue_module.urllib.request, 'urlopen', urlopen)
    assert read_local_decode_progress('http://127.0.0.1:43217/v1/chat/completions') == 73
    assert seen == ['http://127.0.0.1:43217/metrics']
    assert read_local_decode_progress('https://external.example/v1') is None
    assert len(seen) == 1


def test_repetition_guard_wakes_on_atomic_partial_progress_and_skips_unchanged_file(tmp_path, monkeypatch):
    entry = queue_entry(tmp_path, key='rapid-repeat', max_turns=100)
    queue_path, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    queue_path.write_text(json.dumps(entry) + '\n')
    job = queue_module.resolve_entry_jobs(entry, TS_HOST)[0]
    jobs = Path(entry['jobs']); jobs.mkdir(parents=True)
    partial_path = jobs / f"{job['key']}.partial.json"

    def repeated_turn(index):
        return {'request_sha256': f'{index:064x}', 'invocation_id': 'actual-child',
                'response': {'calls': [['eval', {'code': '2'}]], 'raw_response': {'sampled': True}},
                'last_tool_observation': {'content_sha256': 'a' * 64}}

    def publish(turns):
        temp = partial_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'program_id': job['program_id'],
            'provenance': {'program_ir_sha256': job['digest']}, 'turns': turns}))
        os.replace(temp, partial_path)

    publish([repeated_turn(0), repeated_turn(1)])
    waits, checks, children = [], [], []
    original_guard = queue_module.identical_turn_streak
    def counted_guard(*args, **kwargs):
        checks.append(1)
        return original_guard(*args, **kwargs)
    monkeypatch.setattr(queue_module, 'identical_turn_streak', counted_guard)

    class Child:
        def __init__(self, command, **kwargs):
            self.command = command; self.terminated = False; self.reaped = False
            self.wait_count = 0; children.append(self)
        def wait(self, timeout=None):
            self.wait_count += 1; waits.append(timeout)
            if self.terminated:
                self.reaped = True
                return -15
            assert timeout <= queue_module.REPETITION_GUARD_POLL_SECONDS
            if self.wait_count == 1:
                # Attach once to the already-present durable partial.
                raise queue_module.subprocess.TimeoutExpired(self.command, timeout)
            if self.wait_count == 2:
                # The supervisor must not reread or re-evaluate an unchanged partial.
                raise queue_module.subprocess.TimeoutExpired(self.command, timeout)
            if self.wait_count == 3:
                publish([repeated_turn(0), repeated_turn(1), repeated_turn(2)])
                raise queue_module.subprocess.TimeoutExpired(self.command, timeout)
            pytest.fail('the exact three-repeat guard should terminate before a fourth poll')
        def terminate(self): self.terminated = True
        def kill(self): pytest.fail('the owned child should terminate cooperatively')

    install_child(monkeypatch, Child)
    class Healthy:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(queue_module.urllib.request, 'urlopen', lambda *a, **k: Healthy())
    assert run_queue(queue_path, journal, TS_HOST, seconds=60, identical_turn_limit=3) == 2
    assert checks == [1, 1] and waits[:3] == [1.0, 1.0, 1.0]
    assert children[0].terminated and children[0].reaped
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    detected = next(row for row in events if row['event'] == 'repetition_detected')
    finished = next(row for row in events if row['event'] == 'finish')
    assert detected['limit'] == 3 and detected['saved_turns'] == 3
    assert finished['status'] == 'repetition'
    assert finished['resource_limit_reason'] == 'identical_turn_limit'


def test_saved_authored_actions_are_not_counted_as_model_replies(tmp_path):
    entry = queue_entry(tmp_path, count=2)
    jobs = queue_module.resolve_entry_jobs(entry, TS_HOST)
    directory = Path(entry['jobs']); directory.mkdir()
    rows = [json.loads(line) for line in Path(entry['source']).read_text().splitlines()]
    authored = {'kind': 'authored_reference_root_eval', 'sampled': False}
    for index, job in enumerate(jobs):
        common = {'request_sha256': 'authored', 'action_provenance': authored}
        real = {'request_sha256': 'actual'}
        if index == 0:
            value = {'task': {'program_ir': rows[index]}, 'provenance': {'program_ir_sha256': job['digest']},
                'trajectory': [{**common, 'raw_response_sha256': 'reference',
                    'model_response': {'completion_tokens': 0}},
                    {**real, 'raw_response_sha256': 'real', 'model_response': {'completion_tokens': 7}}]}
            suffix = '.result.json'
        else:
            value = {'program_id': job['program_id'], 'provenance': {'program_ir_sha256': job['digest']},
                'turns': [{'request_sha256': 'authored', 'response': {
                    'raw_response': {'natlang_action_provenance': authored}}},
                    {**real, 'response': {'raw_response': {'sampled': True}, 'completion_tokens': 3}}]}
            suffix = '.partial.json'
        (directory / (job['key'] + suffix)).write_text(json.dumps(value))
    metrics = queue_module.partial_metrics(entry)
    assert metrics['saved_turns'] == 4
    assert metrics['fresh_model_replies'] == 2 and metrics['completion_tokens'] == 10


def test_shared_collector_runs_invocations_in_one_node_process(tmp_path, monkeypatch):
    cli = TS_HOST / 'dist/teacher/cli.js'
    if not queue_module.SharedChild.supported(cli):
        pytest.skip('ts-host dist is not built with the server mode')
    import tempfile, shutil
    short = tempfile.mkdtemp(dir='/tmp', prefix='nlsc')
    monkeypatch.setattr(queue_module.SharedChild, 'ROOT', short)
    monkeypatch.setenv('INVOCATION_ID', 'test-shared-collector')
    log = io.StringIO()
    first = queue_module.SharedChild(['node', '--x', str(cli), '--capabilities', 'json'], log, 2)
    second = queue_module.SharedChild(['node', '--x', str(cli), '--capabilities', 'json'], log, 2)
    assert first.wait(60) == 0 and second.wait(60) == 0
    assert first.server_pid == second.server_pid
    failing = queue_module.SharedChild(['node', '--x', str(cli), '--no-such-run'], log, 2)
    assert failing.wait(60) == 1
    os.kill(first.server_pid, 15)
    shutil.rmtree(short, ignore_errors=True)


def test_orphan_temporaries_of_dead_writers_are_swept(tmp_path):
    dead = subprocess.Popen(['true']); dead.wait()
    orphan = tmp_path / f'results.jsonl.tmp-{dead.pid}-01234567-89ab-cdef-0123-456789abcdef'
    live = tmp_path / f'results.jsonl.tmp-{os.getpid()}-01234567-89ab-cdef-0123-456789abcdef'
    for path in (orphan, live, tmp_path / 'results.jsonl'):
        path.write_text('x')
    assert queue_module.sweep_orphan_temporaries(str(tmp_path), str(tmp_path / 'missing')) == 1
    assert not orphan.exists() and live.exists() and (tmp_path / 'results.jsonl').exists()


def test_shared_collector_heap_stays_bounded_and_recycles(tmp_path, monkeypatch):
    """Many invocations through shared servers against a chat model that never finishes a case: the heap after a
    collection stays flat (no per-invocation retention), and a server stops admitting after --recycle-after (64)."""
    import http.server, tempfile, threading, shutil
    cli = TS_HOST / 'dist/teacher/cli.js'
    if not queue_module.SharedChild.supported(cli) or 'recycle-after' not in cli.read_text():
        pytest.skip('ts-host dist is not built with the recycling server mode')
    requests = []

    class Chat(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers['content-length'] or 0))
            requests.append(1)
            data = json.dumps({'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'fake',
                               'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                                   'role': 'assistant', 'content': 'Looking at the inputs first. ' * 50}}],
                               'usage': {'prompt_tokens': 1000, 'completion_tokens': 50}}).encode()
            self.send_response(200)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Chat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    short = tempfile.mkdtemp(dir='/tmp', prefix='nlsh')
    monkeypatch.setattr(queue_module.SharedChild, 'ROOT', short)
    monkeypatch.setenv('INVOCATION_ID', 'test-shared-collector-heap')
    source = tmp_path / 'source.ir.jsonl'
    root = 'solve.nl'   # a natural-language root: every invocation calls the model
    source.write_text(''.join(json.dumps({
        'version': 'natlang.program/2', 'id': f'nl-case-{index}', 'kind': 'lambda_source',
        'source_layout': {'root': root, 'subfunctions': {}},
        'semantics': {'root': root, 'files': {root: '---\nargs:\n  problem: "string"\nreturns: "number"\n---\n'
                                                     'Work out the answer and return it as a plain number.\n'},
                      'inputs': {'problem': f'What is {index} plus {index}?'}, 'expected': 2 * index,
                      'oracle': 'normalized'}}) + '\n' for index in range(4)))
    total, capacity, log = 72, 4, io.StringIO()

    def command(index):
        jobs = tmp_path / f'jobs-{index}'
        jobs.mkdir()
        return ['node', '--x', str(cli), str(source), str(jobs), str(tmp_path / f'out-{index}.jsonl'), '--start',
                str(index % 4), '--limit', '1', '--model-id', 'fake', '--root-seed', str(index), '--workers', '1',
                '--max-turns', '20', '--model-concurrency', '1', '--max-model-requests', '2', '--transport-retries', '0',
                '--temperature', '1.0', '--server', url, '--context-tokens', '32768', '--kv-tokens', '0',
                '--chat-completions-url', url + '/v1/chat/completions']
    try:
        running, started = [], 0
        while started < total or running:
            while started < total and len(running) < capacity:
                running.append(queue_module.SharedChild(command(started), log, capacity))
                started += 1
            running = [child for child in running if child.poll() is None]
            time.sleep(0.05)
        ends = {}
        for path in Path(short).glob('*/*/s*.log'):
            for line in path.read_text().splitlines():
                if '"invocation_end"' in line:
                    event = json.loads(line)
                    ends.setdefault(path.name, []).append(event)
        assert requests, 'the invocations reached the model'
        assert sum(len(events) for events in ends.values()) == total
        pids = {json.loads(line)['pid'] for line in log.getvalue().splitlines() if '"shared_collector"' in line}
        assert len(pids) >= 2, 'a server stops admitting after 64 invocations and a new one serves the rest'
        heaps = [event['heap_used_mb'] for events in ends.values() for event in events[8:]]
        first = [event['heap_used_mb'] for events in ends.values() for event in events[4:8]]
        assert max(heaps) <= max(first) + 32, (first, heaps)
    finally:
        server.shutdown()
        shutil.rmtree(short, ignore_errors=True)
