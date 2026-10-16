"""Timer exit must not kill queued checks or certify interrupted attempts."""
import importlib.util
from pathlib import Path
import subprocess
import pytest


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
    args, kwargs = next(call for call in calls if call[0][0] == 'systemd-run')
    assert args[:3] == ['systemd-run', '--user', '--collect']
    assert args[args.index('--unit') + 1] == 'natlang-full-check-aaaaaaaaaa-launch'
    assert '--property=Type=exec' in args
    assert '--wait' in args and '86400' in args
    assert kwargs['check'] is True
    assert state['running']['unit'] == 'natlang-full-check-aaaaaaaaaa'
    temporary=Path(state['running']['temporary_directory'])
    assert f'TMPDIR={temporary}' in args
    assert (temporary/'.full-check-owner').read_text()=='a'*40


def test_temporary_storage_is_explicit_and_only_owned_terminal_fixtures_are_removed(monkeypatch,tmp_path):
    module=service(monkeypatch,tmp_path)
    monkeypatch.setenv('NATLANG_FULL_CHECK_TMPDIR',str(tmp_path/'external'))
    monkeypatch.setattr(module.subprocess,'run',lambda args,**kwargs:subprocess.CompletedProcess(args,0))
    state={};module.launch(state,'b'*40)
    run=state['running'];temporary=Path(run['temporary_directory'])
    assert temporary.parent==tmp_path/'external'
    (temporary/'fixture').write_text('generated test fixture')
    (temporary/'.full-check-owner').write_text('different owner')
    with pytest.raises(ValueError,match='unowned'):
        module.clean_temporary(run)
    assert (temporary/'fixture').exists()
    (temporary/'.full-check-owner').write_text('b'*40)
    module.clean_temporary(run)
    assert not temporary.exists()


def test_waiting_admission_service_counts_as_live(monkeypatch, tmp_path):
    module = service(monkeypatch, tmp_path)
    def run(args, **kwargs):
        assert args[0] == 'systemctl'
        return subprocess.CompletedProcess(args, 0 if args[-1].endswith('-launch.service') else 3)
    monkeypatch.setattr(module.subprocess, 'run', run)
    assert module.unit_live('natlang-full-check-aaaaaaaaaa')


def test_cleanup_removes_frozen_fixtures_without_following_symlinks(monkeypatch,tmp_path):
    module=service(monkeypatch,tmp_path)
    root=tmp_path/('c'*10);root.mkdir()
    (root/'.full-check-owner').write_text('c'*40)
    frozen=root/'frozen';frozen.mkdir()
    program=frozen/'program';program.mkdir();(program/'source').write_text('fixture')
    outside=tmp_path/'outside';outside.mkdir();(outside/'kept').write_text('live')
    (frozen/'linked').symlink_to(outside,target_is_directory=True)
    program.chmod(0o500);frozen.chmod(0o500);outside.chmod(0o500)
    try:
        module.clean_temporary({'sha':'c'*40,'temporary_directory':str(root)})
        assert not root.exists()
        assert (outside/'kept').read_text()=='live' and outside.stat().st_mode&0o777==0o500
    finally:outside.chmod(0o700)


def test_cleanup_failure_retains_owner_witness(monkeypatch,tmp_path):
    module=service(monkeypatch,tmp_path)
    root=tmp_path/('d'*10);root.mkdir();marker=root/'.full-check-owner'
    marker.write_text('d'*40);(root/'fixture').mkdir()
    monkeypatch.setattr(module.shutil,'rmtree',lambda path:(_ for _ in ()).throw(PermissionError('busy fixture')))
    with pytest.raises(PermissionError,match='busy fixture'):
        module.clean_temporary({'sha':'d'*40,'temporary_directory':str(root)})
    assert marker.read_text()=='d'*40


@pytest.mark.parametrize('contents', ['FAILED tests/test_something.py::test_partial\n', 'exit=137\n'])
def test_interrupted_check_does_not_answer_requests_or_advance_last_sha(monkeypatch, tmp_path, contents):
    module = service(monkeypatch, tmp_path)
    log = tmp_path / 'partial.log'
    log.write_text(contents)
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
    assert state['history'][-1]['incomplete'] is True
    assert state['retry']['attempts'] == 1
    assert 'running' not in state
    assert messages == []


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
    assert not state['history'][-1].get('incomplete')
    assert state['history'][-1]['exit'] == 1


@pytest.mark.parametrize('status', ['failed', 'timeout', 'killed (out of memory?)', 'signal SIGTERM'])
def test_isolated_test_runner_terminal_file_is_routed(monkeypatch, tmp_path, status):
    module = service(monkeypatch, tmp_path)
    text = (f'{status}   test/neuralese-learning.test.mjs (14 failed, 5s)\n'
            '=== summary\nts-tests FAILED (1) 580s\nexit=1\n')
    result = module.failures(text)
    assert result['failed_tests'] == ['test/neuralese-learning.test.mjs']
    assert result['failed_steps'] == ['ts-tests'] and result['exit'] == 1
