"""A live Colab master must survive service redeployment without a second login."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location('colab_tunnel', Path(__file__).resolve().parents[2] / 'scripts/colab/tunnel.py')
tunnel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tunnel)


def test_reuses_live_master_and_recovers_only_after_it_disappears():
    calls = []
    checks = iter([0, 0, 1])

    def run(command, **kwargs):
        calls.append(command)
        if command[:3] == ['ssh', '-O', 'check']:
            return SimpleNamespace(returncode=next(checks))
        if command[0] == 'ss':
            return SimpleNamespace(stdout='LISTEN 0 128 127.0.0.1:19731 0.0.0.0:*\n')
        return SimpleNamespace(returncode=0)

    def sleep(interval):
        assert not any('-M' in call for call in calls)

    assert tunnel.maintain('colab-g4', ['19731:9570'], run=run, sleep=sleep) == 0
    assert sum('-M' in call for call in calls) == 1
    assert calls[-1][-3:] == ['-L', '19731:127.0.0.1:9570', 'colab-g4']


def test_missing_forward_is_added_without_opening_another_connection():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if 'forward' in command else 0, stdout='')

    assert tunnel.maintain('colab-g4b', ['19732:8571'], run=run) == 1
    assert calls[-1] == ['ssh', '-O', 'forward', '-L', '19732:127.0.0.1:8571', 'colab-g4b']
    assert not any('-M' in call for call in calls)


def test_reuse_can_be_disabled_explicitly():
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    assert tunnel.maintain('colab-g4', [], reuse=False, run=run) == 0
    assert len(calls) == 1 and '-M' in calls[0]
