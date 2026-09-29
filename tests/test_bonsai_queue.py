import json
import subprocess
import pytest
from scripts.run_bonsai_queue import run_queue


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
