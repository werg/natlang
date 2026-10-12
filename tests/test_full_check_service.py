"""Timer exit must not kill queued checks or certify interrupted attempts."""
import importlib.util
from pathlib import Path
import subprocess


def service(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/full_check_service.py'
    spec = importlib.util.spec_from_file_location('full_check_service_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'BASE', tmp_path)
    monkeypatch.setattr(module, 'STATE', tmp_path / 'state.json')
    monkeypatch.setattr(module, 'git', lambda *args, **kwargs: 'commit')
    return module


def test_admission_waiter_has_its_own_managed_service(monkeypatch, tmp_path):
    module = service(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs:
                        calls.append((args, kwargs)) or subprocess.CompletedProcess(args, 0))
    state = {}
    module.launch(state, 'a' * 40)
    args, kwargs = calls[0]
    assert args[:3] == ['systemd-run', '--user', '--collect']
    assert args[args.index('--unit') + 1] == 'natlang-full-check-aaaaaaaaaa-admission'
    assert '--property=Type=exec' in args
    assert '--wait' in args and '86400' in args
    assert kwargs['check'] is True
    assert state['running']['unit'] == 'natlang-full-check-aaaaaaaaaa'


def test_waiting_admission_service_counts_as_live(monkeypatch, tmp_path):
    module = service(monkeypatch, tmp_path)
    def run(args, **kwargs):
        assert args[0] == 'systemctl'
        return subprocess.CompletedProcess(args, 0 if args[-1].endswith('-admission.service') else 3)
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.unit_live('natlang-full-check-aaaaaaaaaa')


def test_interrupted_check_does_not_answer_requests_or_advance_last_sha(monkeypatch, tmp_path):
    module = service(monkeypatch, tmp_path)
    log = tmp_path / 'partial.log'
    log.write_text('FAILED tests/test_something.py::test_partial\n')
    messages = []
    monkeypatch.setattr(module, 'coord', lambda *args: messages.append(args) or '')
    def unexpected(*args):
        raise AssertionError('an incomplete check must not certify coverage or blame test owners')
    monkeypatch.setattr(module, 'open_full_check_requests', unexpected)
    monkeypatch.setattr(module, 'owner_of', unexpected)
    state = {'last_sha': 'previous', 'running': {'sha': 'a' * 40, 'unit': 'unit',
             'log': str(log), 'worktree': str(tmp_path / 'worktree'), 'command': 'test'}}
    module.finish(state)
    assert state['last_sha'] == 'previous'
    assert state['history'][-1]['complete'] is False
    assert state['history'][-1]['answered'] == []
    assert 'running' not in state
    assert 'incomplete' in messages[0][4]


def test_completed_failure_is_a_real_verdict(monkeypatch, tmp_path):
    module = service(monkeypatch, tmp_path)
    log = tmp_path / 'complete.log'
    log.write_text('=== summary\npython-tests FAILED (1) 3s\nexit=1\n')
    monkeypatch.setattr(module, 'coord', lambda *args: '')
    monkeypatch.setattr(module, 'open_full_check_requests', lambda: [])
    state = {'running': {'sha': 'a' * 40, 'unit': 'unit', 'log': str(log),
                        'worktree': str(tmp_path / 'worktree'), 'command': 'test'}}
    module.finish(state)
    assert state['last_sha'] == 'a' * 40
    assert state['history'][-1]['complete'] is True
    assert state['history'][-1]['exit'] == 1
