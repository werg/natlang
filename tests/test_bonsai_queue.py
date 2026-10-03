import json
import hashlib
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import pytest
import scripts.run_bonsai_queue as queue_module
from scripts.run_bonsai_queue import run_queue, local_decode_progress as read_local_decode_progress

ROOT = Path(__file__).resolve().parents[1]
TS_HOST = ROOT / 'ts-host'


@pytest.fixture(autouse=True)
def no_live_server_queries(monkeypatch):
    monkeypatch.setattr(queue_module, 'local_decode_progress', lambda: None)
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
def test_long_reply_decoding_is_activity_but_a_motionless_request_times_out(tmp_path, monkeypatch, decoding):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = queue_entry(tmp_path, key='long')
    queue.write_text(json.dumps(entry) + '\n')
    clock = [0]
    monkeypatch.setattr('scripts.run_bonsai_queue.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('scripts.run_bonsai_queue.local_decode_progress', lambda: clock[0] if decoding else None)
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
    run_queue(queue, journal, TS_HOST, seconds=600, no_observation_seconds=360)
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
