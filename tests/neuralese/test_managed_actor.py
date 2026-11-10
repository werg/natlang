"""A dead managed RL actor is relaunched mid-collection; the SGLang wrapper exits when its engine dies (2026-11-10)."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from natlang_neuralese.rl import production
from natlang_neuralese.rl.managed_actor import ManagedActor

ROOT = Path(__file__).resolve().parents[2]


class FakeProcess:
    def __init__(self, pid):
        self.pid, self.code, self.terminated = pid, None, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated, self.code = True, -15

    def wait(self, timeout=None):
        return self.code


def harness(*, ready_seconds=900., backoff=60.):
    clock = {'now': 0.}
    launched, events = [], []
    serving = {'on': True}

    def launch():
        launched.append(FakeProcess(100 + len(launched) + 1))
        return launched[-1]
    first = launch()
    actor = ManagedActor(first, launch, lambda: serving['on'], ready_seconds=ready_seconds, backoff_seconds=backoff,
                         clock=lambda: clock['now'], log=lambda text, **kw: events.append(json.loads(text)))
    return actor, launched, events, clock, serving


def test_a_serving_actor_is_ready_and_never_relaunched():
    actor, launched, events, clock, _ = harness()
    clock['now'] = 500
    assert actor.ready() and len(launched) == 1 and not events


def test_a_dead_actor_is_relaunched_once_and_its_slot_waits_until_it_attests():
    actor, launched, events, clock, serving = harness()
    launched[0].code = 70   # the wrapper's exit when the SGLang engine died
    clock['now'] = 1000
    serving['on'] = False
    assert actor.ready() is False and len(launched) == 2 and actor.pid == 102
    assert events == [{'event': 'relaunching the managed RL actor', 'reason': 'exited 70', 'previous_pid': 101, 'relaunch': 1}]
    clock['now'] = 1100
    assert actor.ready() is False and len(launched) == 2   # still loading: no second launch
    serving['on'] = True
    clock['now'] = 1200
    assert actor.ready() is True and len(launched) == 2


def test_a_crash_loop_is_spaced_by_the_backoff():
    actor, launched, events, clock, _ = harness()
    launched[0].code = 70
    clock['now'] = 1000
    actor.ready()
    launched[1].code = 70
    clock['now'] = 1010
    assert actor.ready() is False and len(launched) == 2   # inside the 60 s backoff
    clock['now'] = 1061
    assert actor.ready() is False and len(launched) == 3


def test_a_launch_that_never_attests_is_replaced_after_ready_seconds():
    actor, launched, events, clock, serving = harness(ready_seconds=300)
    serving['on'] = False
    clock['now'] = 299
    assert actor.ready() is False and len(launched) == 1
    clock['now'] = 301
    assert actor.ready() is False and launched[0].terminated and len(launched) == 2
    assert events[0]['reason'].startswith('did not attest')


def test_the_controller_relaunches_a_managed_actor_that_dies_during_collection(tmp_path, monkeypatch):
    from natlang_neuralese.rl import actor_client
    import time
    started = []

    class Process(FakeProcess):
        def __init__(self, command, **kw):
            super().__init__(2**22 + 10 + len(started))
            started.append(command)

    class Server:
        def __init__(self, url, **kw):
            pass

        def get_bytes(self, path):
            return json.dumps({'rl_policy_binding': {'checkpoint_sha256': 'cur'}}).encode()
    monkeypatch.setattr(production.subprocess, 'Popen', Process)
    monkeypatch.setattr(actor_client, 'NativeActorClient', Server)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    ticks = iter(range(0, 10**6, 100))
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    seen = {}

    def collect(**kw):
        ready = kw['local_actor_ready']
        assert ready() is True and len(started) == 1
        production_actor = ready.__self__
        production_actor.process.code = 70   # the engine died under the live wrapper
        assert ready() is False and len(started) == 2   # same command: same store dir, same policy
        assert started[0] == started[1] == ['actor']
        seen['record'] = json.loads((tmp_path / 'actor.pid.json').read_text())
        assert ready() is True
        return {'collection_complete': True}
    monkeypatch.setattr(production, 'collect', collect)
    config = {'actor_command': ['actor']}
    report = production.collect_with_actor(config, {'out': str(tmp_path), 'policy_id': 'cur'}, out=tmp_path,
                                           endpoints=['http://dgx'], policy_id='cur', prompt_rendering='full',
                                           neuralese_server=False)
    assert report == {'collection_complete': True}
    assert seen['record']['pid'] == 2**22 + 11 and seen['record']['policy_id'] == 'cur'


def test_the_sglang_wrapper_exits_nonzero_when_its_engine_dies(tmp_path):
    """Real process, fake engine (a short sleep): the wrapper logs the reason and exits ENGINE_DIED_EXIT."""
    script = tmp_path / 'wrapper.py'
    script.write_text(textwrap.dedent(f'''
        import importlib.util, subprocess, sys, time
        spec = importlib.util.spec_from_file_location('actor_cli', {str(ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py')!r})
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        engine = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'], start_new_session=True)
        cli.supervise_sglang(engine)
        time.sleep(60)   # serve_forever
    '''))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 70
    event = json.loads(next(line for line in run.stderr.splitlines() if line.startswith('{')))
    assert event['event'] == 'sglang_engine_exited' and event['returncode'] == 0


def test_the_sglang_wrapper_stops_its_engine_on_sigterm_without_reporting_a_crash(tmp_path):
    script = tmp_path / 'wrapper.py'
    script.write_text(textwrap.dedent(f'''
        import importlib.util, os, signal, subprocess, sys, time
        spec = importlib.util.spec_from_file_location('actor_cli', {str(ROOT / 'scripts/neuralese_data/serve_native_checkpoint.py')!r})
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        engine = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        cli.supervise_sglang(engine)
        print(engine.pid, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
    '''))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 143 and 'sglang_engine_exited' not in run.stderr
    import os
    try:
        os.kill(int(run.stdout.split()[0]), 0)
        raise AssertionError('engine still running')
    except ProcessLookupError:
        pass
