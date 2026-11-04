"""In-place driver handover inside one sandbox (scripts/coreweave/driver_ownership.sh): the entrypoint's driver never
ends the sandbox while a newer driver owns the lineage; a second driver cannot start beside a running owner."""
import subprocess
import time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / 'scripts' / 'coreweave' / 'driver_ownership.sh'


def driver(tmp_path, name, body, hold=0):
    script = tmp_path / f'{name}.sh'
    script.write_text(f'''set -uo pipefail
OWNER={tmp_path}/owner
NATLANG_CW_HOLD_ON_FAILURE={hold}
NATLANG_CW_OWNER_POLL=0.2
. {LIB}
driver_take run
trap 'driver_exit $?; echo "{name} exited" >> {tmp_path}/events' EXIT
{body}
''')
    return subprocess.Popen(['bash', str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def events(tmp_path):
    path = tmp_path / 'events'
    return path.read_text().split('\n')[:-1] if path.exists() else []


def test_second_driver_refused_while_owner_runs(tmp_path):
    first = driver(tmp_path, 'first', 'sleep 3')
    time.sleep(0.5)
    second = driver(tmp_path, 'second', 'exit 0')
    assert second.wait(timeout=10) == 75
    assert first.wait(timeout=10) == 0


def test_failed_owner_hands_over_and_waits_for_newer_driver(tmp_path):
    first = driver(tmp_path, 'first', 'exit 2', hold=60)  # fails at once, enters its failure hold
    time.sleep(1)
    assert (tmp_path / 'owner').read_text().split()[2] == 'hold'
    second = driver(tmp_path, 'second', 'sleep 2')  # in-place handover during the hold
    time.sleep(0.5)
    assert first.poll() is None  # the entrypoint's child stays alive while the newer driver trains
    assert second.wait(timeout=20) == 0
    first.wait(timeout=20)  # ends after the newer driver, long before its 60 s hold would
    assert events(tmp_path) == ['second exited', 'first exited']


def test_owner_without_successor_holds_then_exits(tmp_path):
    started = time.monotonic()
    first = driver(tmp_path, 'first', 'exit 2', hold=5)
    first.wait(timeout=30)
    assert time.monotonic() - started >= 4.5 and events(tmp_path) == ['first exited']


def test_owner_registers_with_the_bootstrap_control_dir(tmp_path):
    control = tmp_path / 'cw'
    script = tmp_path / 'reg.sh'
    script.write_text(f'''OWNER={tmp_path}/owner
NATLANG_CW_CONTROL_DIR={control}
. {LIB}
driver_take run
driver_take run
''')
    assert subprocess.run(['bash', str(script)]).returncode == 0
    assert (control / 'driver-owners').read_text() == f'{tmp_path}/owner\n'   # registered once
