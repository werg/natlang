"""Independent regressions for actual actor participant binding and owned-child durability."""
import json

import pytest

from natlang_neuralese.rl.production import validate_actor_receipts


def test_actor_receipt_must_match_requested_participant_and_identity():
    identity = {'checkpoint_sha256': 'a'*64, 'tokenizer_fingerprint': '1'*64,
                'embedding_weight_sha256': '2'*64, 'architecture': 'lfm2'}
    expected = {'native_identity': identity, 'native_participant': 'student'}
    receipt = {'schema': 'natlang.rl-policy-binding/2', 'communication': 'native-state',
        'checkpoint_sha256': 'a'*64, **expected}
    validate_actor_receipts({'endpoint': receipt}, policy_id='a'*64, neuralese_server=True, expected_native=expected)
    for change in ({'native_participant': 'other'}, {'native_identity': {**identity, 'architecture': 'mellum'}}):
        with pytest.raises(ValueError, match='requested native participant'):
            validate_actor_receipts({'endpoint': {**receipt, **change}}, policy_id='a'*64, neuralese_server=True, expected_native=expected)


def test_authenticated_history_profile_must_match_actual_actor_renderer():
    identity = {'checkpoint_sha256':'a'*64, 'tokenizer_fingerprint':'b'*64,
        'embedding_weight_sha256':'c'*64, 'architecture':'lfm2'}
    profile = {'policy':'keep','backbone':'lfm2','template_kwargs':{},'probe':'keep'}
    expected = {'native_identity':identity,'native_participant':'lfm','history_reasoning':profile}
    receipt = {'schema':'natlang.rl-policy-binding/2','communication':'native-state',
        'checkpoint_sha256':'a'*64,'native_identity':identity,'native_participant':'lfm',
        'history_reasoning':profile}
    validate_actor_receipts({'endpoint':receipt},policy_id='a'*64,neuralese_server=True,
        expected_native=expected)
    for wrong in ({**profile,'policy':'last_turn_only'},
                  {**profile,'probe':'drops-history'},
                  {**profile,'template_kwargs':{'preserve_thinking':False}}):
        with pytest.raises(ValueError,match='requested native participant'):
            validate_actor_receipts({'endpoint':{**receipt,'history_reasoning':wrong}},
                policy_id='a'*64,neuralese_server=True,expected_native=expected)


def test_child_launch_retries_transient_proc_identity_before_persisting(tmp_path, monkeypatch):
    import sys
    from natlang_neuralese.train import native_readiness as controller
    actual = controller._owned_process_identity
    reads = []
    def transient(pid, launch, **kwargs):
        reads.append(pid)
        return None if len(reads) <= 2 else actual(pid, launch, **kwargs)
    monkeypatch.setattr(controller, '_owned_process_identity', transient)
    argv = [sys.executable, '-c', 'import time; time.sleep(300)']
    child, launch = controller._launch_child(argv, 'provider-worker', {}, tmp_path, 'execution', None)
    try:
        assert len(reads) >= 3
        assert launch['identity']['pid'] == child.pid
        assert json.loads((tmp_path/'child-launch.json').read_text())['identity'] == launch['identity']
        assert controller._recover_child(launch).pid == child.pid
    finally:
        child.terminate(); child.wait(timeout=5)


def test_child_launch_identity_miss_does_not_block_supervision(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_readiness as controller
    class Child:
        pid = 2147483647
        def poll(self): return None
    child = Child()
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: child)
    monkeypatch.setattr(controller, '_owned_process_identity', lambda *args, **kwargs: None)
    monkeypatch.setattr(controller.time, 'sleep', lambda _: pytest.fail('absent process must not be retried'))
    observed, launch = controller._launch_child(['missing-fixture'], 'provider-worker', {}, tmp_path, 'execution', None)
    assert observed is child and launch['identity'] is None
    assert launch['status'] == 'running' and launch['token']


def test_wrapper_argv_identity_miss_is_bounded_and_owned_tree_recoverable(tmp_path):
    import sys
    from natlang_neuralese.train import native_readiness as controller
    script = tmp_path/'worker'
    script.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(300)\n')
    script.chmod(0o700)
    child, launch = controller._launch_child([str(script)], 'provider-worker', {}, tmp_path, 'execution', None)
    try:
        assert child.poll() is None and launch['identity'] is None
        recovered = controller._recover_child(launch)
        assert recovered is not None
        assert any(entry['pid'] == child.pid for entry in controller._owned_tree(launch))
    finally:
        child.terminate(); child.wait(timeout=5)


def test_restart_recovery_binds_exact_live_child_identity(tmp_path):
    import sys
    from natlang_neuralese.train import native_readiness as controller
    out = tmp_path/'out'; out.mkdir()
    command = [sys.executable, '-c', 'import time; time.sleep(300)']
    child, launch = controller._launch_child(command, 'provider-worker', {}, out, 'bound-execution', None)
    try:
        assert launch['identity']['pid'] == child.pid
        assert controller._recover_child(launch).pid == child.pid
        # A PID match is insufficient: boot/start identity and argv are bound.
        with pytest.raises(RuntimeError, match='identity changed'):
            controller._recover_child({**launch, 'identity': {**launch['identity'], 'start_ticks': '0'}})
        assert controller._owned_process_identity(child.pid, {**launch, 'argv': [sys.executable, '-c', 'wrong']}) is None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_orphaned_descendant_is_drained_before_retry(tmp_path):
    import sys
    import time
    from natlang_neuralese.train import native_readiness as controller
    out = tmp_path/'out'; out.mkdir()
    # The command exits while its subprocess retains exact launch ownership.
    script = ('import subprocess,sys; '
              'subprocess.Popen([sys.executable,"-c","import time; time.sleep(300)"])')
    child, launch = controller._launch_child([sys.executable, '-c', script], 'native-rl', {}, out, 'execution', 'rl')
    child.wait(timeout=5)
    recovered = controller._recover_child(launch)
    assert recovered is not None and recovered.identity is None
    try:
        recovered.terminate()
        deadline = time.monotonic() + 5
        while recovered.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert recovered.returncode != 0  # no inferred successful training
        assert controller._owned_tree(launch) == []
    finally:
        recovered.terminate()
