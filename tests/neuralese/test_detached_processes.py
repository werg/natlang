"""Restart in place: workers and the actor outlive the controller and the relaunched controller re-attaches."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from natlang_neuralese.rl import detached_processes as dp


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clear_restart():
    dp.RESTART.clear()
    yield
    dp.RESTART.clear()


def sleeper(seconds):
    return subprocess.Popen([sys.executable, '-c', f'import time; time.sleep({seconds})'], start_new_session=True)


def test_adopted_process_follows_a_process_it_did_not_start(tmp_path):
    child = sleeper(30)
    try:
        record = dp.record_process(tmp_path / 'actor.pid.json', child.pid, policy_id='p')
        adopted = dp.AdoptedProcess(dp.load_record(tmp_path / 'actor.pid.json'))
        assert record['starttime'] and adopted.poll() is None and dp.alive(record)
        with pytest.raises(subprocess.TimeoutExpired):
            adopted.wait(timeout=.3)
        adopted.terminate()
        child.wait(timeout=10)
        assert adopted.poll() == 0 and not dp.alive(record) and adopted.wait(timeout=1) == 0
        # a recycled pid is not the recorded process: the start time differs
        assert not dp.alive({**record, 'pid': os.getpid()})
    finally:
        if child.poll() is None:
            child.kill()
