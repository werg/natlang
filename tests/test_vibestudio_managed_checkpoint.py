import importlib.util
import json
from pathlib import Path
import subprocess
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch


MODULE = Path(__file__).resolve().parents[1] / 'scripts/vibestudio/managed_checkpoint.py'
spec = importlib.util.spec_from_file_location('managed_checkpoint', MODULE)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def test_authenticated_cancel_exports_before_child_retirement(tmp_path):
    run = tmp_path / ('st_' + 'a' * 32)
    run.mkdir()
    (run / 'run.json').write_text(json.dumps({'runId': run.name}))
    packet = run / 'failure-evidence/trajectory-completed-full.json.gz'
    events = []
    def cancel(argv, **kwargs):
        assert argv == ['node', 'launcher', '--instance', 'owned', 'cancel', run.name, '--json']
        assert kwargs['timeout'] <= 50
        events.append('authenticated-cancel')
        packet.parent.mkdir()
        packet.write_bytes(b'original completed packet bytes')
        return SimpleNamespace(returncode=0)
    child = Mock()
    child.poll.return_value = None
    child.wait.side_effect = lambda **kwargs: events.append('join-child')
    with patch.object(checkpoint.subprocess, 'run', side_effect=cancel):
        checkpoint.cancel_and_join(child, ['node', 'launcher', '--instance', 'owned'], tmp_path, cwd=tmp_path, env={})
    assert events == ['authenticated-cancel', 'join-child']
    child.send_signal.assert_not_called()
    child.terminate.assert_not_called()
    assert packet.read_bytes() == b'original completed packet bytes'
    assert json.loads((tmp_path/'checkpoint.json').read_text())['runs'][0]['fullPackets'] == 1


def test_cancel_timeout_still_joins_owner_without_deleting_evidence(tmp_path):
    run = tmp_path / ('st_' + 'b' * 32)
    run.mkdir()
    (run / 'run.json').write_text(json.dumps({'runId': run.name}))
    child = Mock()
    child.poll.return_value = None
    with patch.object(checkpoint.subprocess, 'run', side_effect=subprocess.TimeoutExpired('cancel', 50)):
        checkpoint.cancel_and_join(child, ['node', 'owned'], tmp_path, cwd=tmp_path, env={})
    child.wait.assert_called_once_with(timeout=10)
    assert json.loads((tmp_path/'checkpoint.json').read_text())['runs'][0]['timedOut'] is True
    assert (run / 'run.json').exists()


def test_unstarted_owner_does_not_invent_a_run_or_outcome(tmp_path):
    with patch.object(checkpoint.subprocess, 'run') as rpc:
        record = checkpoint.checkpoint_managed_run(['node', 'owned'], tmp_path, cwd=tmp_path, env={})
    rpc.assert_not_called()
    assert record['runs'] == []
    assert list(tmp_path.glob('st_*')) == []


def test_collector_forwards_one_main_signal_and_child_finishes_checkpoint(tmp_path):
    retained = tmp_path/'retained.txt'
    code = '''import os,signal,sys
from pathlib import Path
def stop(sig,frame):
 Path(sys.argv[1]).write_text("actual child checkpoint")
 raise SystemExit(130)
signal.signal(signal.SIGTERM,stop)
os.kill(os.getppid(),signal.SIGTERM)
signal.pause()
'''
    with checkpoint.CollectorCheckpoint() as owner:
        result = owner.run([sys.executable, '-c', code, str(retained)], request_file=tmp_path/'request.json')
        assert owner.requested
        assert result.returncode == 130
        assert json.loads((tmp_path/'request.json').read_text())['signal'] == 'SIGTERM'
    assert retained.read_text() == 'actual child checkpoint'


def test_checkpoint_before_admission_starts_no_collector(tmp_path):
    with checkpoint.CollectorCheckpoint() as owner, patch.object(checkpoint.subprocess, 'Popen') as launch:
        owner._request(signal.SIGTERM, None)
        result = owner.run(['must-not-start'], request_file=tmp_path/'request.json')
    launch.assert_not_called()
    assert result.returncode == 130


def test_generic_owned_io_communicates_without_managed_request_file(tmp_path):
    with checkpoint.CollectorCheckpoint() as owner:
        result = owner.run([sys.executable, '-c', 'import sys;print(sys.stdin.read());print("diagnostic",file=sys.stderr)'],
                           input='immutable inventory', capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == 'immutable inventory\n' and result.stderr == 'diagnostic\n'
    assert owner.child is None and list(tmp_path.iterdir()) == []


def test_generic_owned_io_forwards_checkpoint_and_joins(tmp_path):
    retained = tmp_path/'retained'
    code = """import os,signal,sys
from pathlib import Path
def stop(sig,frame):
 Path(sys.argv[1]).write_text('retained original IO')
 raise SystemExit(130)
signal.signal(signal.SIGTERM,stop)
os.kill(os.getppid(),signal.SIGTERM)
signal.pause()
"""
    with checkpoint.CollectorCheckpoint() as owner:
        result = owner.run([sys.executable, '-c', code, str(retained)], capture_output=True, text=True)
        assert owner.requested and result.returncode == 130
    assert owner.child is None and retained.read_text() == 'retained original IO'


def test_same_owner_can_decide_post_collection_continuation_without_losing_io(tmp_path):
    ready, decision = tmp_path/'ready', tmp_path/'decision'
    code = '''import sys,time
from pathlib import Path
value=sys.stdin.read()
Path(sys.argv[1]).write_text("actual paid export settled")
while not Path(sys.argv[2]).exists():time.sleep(.01)
print(value+":"+Path(sys.argv[2]).read_text())
'''
    calls=[]
    def tick():
        if ready.exists() and not decision.exists():
            calls.append(ready.read_text())
            decision.write_text('drain-native')
    with checkpoint.CollectorCheckpoint() as owner:
        result=owner.run([sys.executable,'-c',code,str(ready),str(decision)],
                         input='preserved input',capture_output=True,text=True,on_tick=tick)
    assert result.returncode==0 and result.stdout=='preserved input:drain-native\n'
    assert calls==['actual paid export settled']
    assert owner.child is None and not owner.requested
