import json
import subprocess
import pytest
from scripts.run_bonsai_queue import run_queue, local_decode_progress as read_local_decode_progress


@pytest.fixture(autouse=True)
def no_live_server_queries(monkeypatch):
    monkeypatch.setattr('scripts.run_bonsai_queue.local_decode_progress', lambda: None)


def test_timeout_kills_stuck_child_advances_and_resume_skips_finished(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entries = [dict(key=str(i), source='input', jobs='jobs', output='output', index=i,
                    seed=1, log=str(tmp_path / f'{i}.log')) for i in range(2)]
    queue.write_text(''.join(json.dumps(row) + '\n' for row in entries))
    processes = []
    class Child:
        def __init__(self, command, **kwargs):
            self.index = len(processes)
            self.killed = self.terminated = False
            processes.append(self)
        def wait(self, timeout=None):
            if self.index == 0 and not self.killed:
                raise subprocess.TimeoutExpired('fixture', timeout)
            return 0
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.killed = True
    monkeypatch.setattr(subprocess, 'Popen', Child)
    run_queue(queue, journal, tmp_path, seconds=1)
    assert processes[0].terminated and processes[0].killed
    assert len(processes) == 2
    finished = [json.loads(row) for row in journal.read_text().splitlines() if json.loads(row)['event'] == 'finish']
    assert [row['status'] for row in finished] == ['timeout', 'complete']
    run_queue(queue, journal, tmp_path, seconds=1)
    assert len(processes) == 2


def test_entry_turn_budget_is_passed_and_recorded(tmp_path, monkeypatch):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = dict(key='world', source='input', jobs=str(tmp_path / 'jobs'), output='output', index=0,
                 seed=1, log=str(tmp_path / 'case.log'), max_turns=48)
    queue.write_text(json.dumps(entry) + '\n')
    commands = []
    class Child:
        def __init__(self, command, **kwargs):
            commands.append(command)
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(subprocess, 'Popen', Child)
    run_queue(queue, journal, tmp_path, seconds=1)
    assert commands[0][commands[0].index('--max-turns') + 1] == '48'
    assert json.loads(journal.read_text().splitlines()[0])['max_turns'] == 48
    for value in [True, 0, -1, '48']:
        entry['max_turns'] = value
        queue.write_text(json.dumps(entry) + '\n')
        with pytest.raises(ValueError, match='max_turns'):
            run_queue(queue, tmp_path / 'fresh.jsonl', tmp_path, seconds=1)


def test_repeated_action_shapes_do_not_imply_repeated_requests(tmp_path):
    from scripts.run_bonsai_queue import partial_metrics
    turns = [dict(request_sha256=digest, response=dict(raw_response={"provider": "fixture"},
             calls=[["return_result", {"status": "success", "value": False}]], completion_tokens=2))
             for digest in ["input-a", "input-b"]]
    path = tmp_path / '000000-fixture.partial.json'
    path.write_text(json.dumps(dict(turns=turns)))
    entry = dict(index=0, jobs=str(tmp_path))
    metrics = partial_metrics(entry)
    assert metrics['repeated_action_sets'] == 1
    assert metrics['repeated_request_hashes'] == 0
    assert metrics['unique_request_hashes'] == 2
    turns.append(turns[0])
    path.write_text(json.dumps(dict(turns=turns)))
    assert partial_metrics(entry)['repeated_request_hashes'] == 1


def test_local_pair_shares_one_collector_and_one_global_two_request_cap(tmp_path, monkeypatch):
    from scripts.run_bonsai_queue import partial_metrics
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    entry = dict(key='pair', source='input', jobs=str(tmp_path), output='output', index=0, count=2,
                 members=[{'key': 'one'}, {'key': 'two'}], seed=1, log=str(tmp_path / 'case.log'))
    queue.write_text(json.dumps(entry) + '\n')
    commands = []
    class Child:
        def __init__(self, command, **kwargs): commands.append(command)
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(subprocess, 'Popen', Child)
    run_queue(queue, journal, tmp_path, seconds=1, model_concurrency=2)
    command = commands[0]
    assert command[command.index('--workers') + 1] == '2'
    assert command[command.index('--limit') + 1] == '2'
    assert command[command.index('--model-concurrency') + 1] == '2'
    assert [json.loads(line)['key'] for line in journal.read_text().splitlines()
            if json.loads(line)['event'] == 'finish'] == ['pair', 'one', 'two']
    for i in range(2):
        (tmp_path / f'{i:06d}-fixture.partial.json').write_text(json.dumps({'turns': [
            {'request_sha256': str(i), 'response': {'raw_response': {'fixture': True}, 'calls': [], 'completion_tokens': 3}}]}))
    assert partial_metrics(entry)['fresh_model_replies'] == 2
    with pytest.raises(ValueError, match='single-case'):
        run_queue(queue, tmp_path / 'provider.jsonl', tmp_path, provider='openai-codex', model_id='gpt-6-luna', model_concurrency=1)


@pytest.mark.parametrize('count', [4, 5])
def test_larger_batches_use_one_global_cap(tmp_path, monkeypatch, count):
    queue, journal = tmp_path / 'queue.jsonl', tmp_path / 'journal.jsonl'
    queue.write_text(json.dumps(dict(key='batch', source='input', jobs=str(tmp_path), output='output',
        index=0, count=count, seed=1, log=str(tmp_path / 'case.log'))) + '\n')
    commands = []
    class Child:
        def __init__(self, command, **kwargs): commands.append(command)
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(subprocess, 'Popen', Child)
    run_queue(queue, journal, tmp_path, seconds=1, model_concurrency=4)
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index('--workers') + 1] == str(count)
    assert command[command.index('--model-concurrency') + 1] == '4'
    with pytest.raises(ValueError, match='single-case'):
        run_queue(queue, tmp_path / 'provider.jsonl', tmp_path, provider='openai-codex', model_concurrency=1)


def test_batch_metrics_include_completed_and_running_roots(tmp_path):
    from scripts.run_bonsai_queue import partial_metrics
    (tmp_path / '000000-fixture.result.json').write_text(json.dumps({'trajectory': [
        {'request_sha256': 'done', 'raw_response_sha256': 'hash', 'model_response': {'calls': [], 'completion_tokens': 7}}]}))
    (tmp_path / '000001-fixture.partial.json').write_text(json.dumps({'turns': [
        {'request_sha256': 'running', 'response': {'raw_response': {'fixture': True}, 'calls': [], 'completion_tokens': 3}}]}))
    metrics = partial_metrics(dict(index=0, count=2, jobs=str(tmp_path)))
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
    queue.write_text(json.dumps(dict(key='long', source='input', jobs=str(tmp_path), output='output',
        index=0, seed=1, log=str(tmp_path / 'case.log'))) + '\n')
    clock = [0]
    monkeypatch.setattr('scripts.run_bonsai_queue.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('scripts.run_bonsai_queue.local_decode_progress', lambda: clock[0] if decoding else None)
    class Child:
        def __init__(self, *args, **kwargs): self.terminated = False; self.calls = 0
        def wait(self, timeout=None):
            if self.terminated: return 0
            clock[0] += 30; self.calls += 1
            if self.calls <= 12: raise subprocess.TimeoutExpired('fixture', timeout)
            return 0
        def terminate(self): self.terminated = True
    monkeypatch.setattr(subprocess, 'Popen', Child)
    run_queue(queue, journal, tmp_path, seconds=600)
    finish = json.loads(journal.read_text().splitlines()[-1])
    assert finish['status'] == ('complete' if decoding else 'inactivity_timeout')
