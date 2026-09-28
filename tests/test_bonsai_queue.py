import json
import subprocess
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
