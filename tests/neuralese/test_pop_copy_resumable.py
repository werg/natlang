import subprocess
import sys

sys.path.insert(0, 'scripts/neuralese_data')
import pop_rl_actor


def test_copy_retries_a_stalled_rsync(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(30, argv)   # rsync's timeout exit

    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', run)
    monkeypatch.setattr(pop_rl_actor.time, 'sleep', lambda s: None)
    pop_rl_actor.copy_resumable('a', 'pop:b', attempts=4)
    assert len(calls) == 3 and all('--partial' in c and '--timeout=120' in c for c in calls)
