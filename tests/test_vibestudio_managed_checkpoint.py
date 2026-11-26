import pytest
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


@pytest.fixture(autouse=True)
def isolated_collector_ledger_owner(monkeypatch):
    monkeypatch.delenv('NATLANG_LEDGER_UNIT', raising=False)


def test_normal_completion_and_explicit_owner_signal_withdraw_continuation(monkeypatch,tmp_path):
    monkeypatch.setenv('NATLANG_LEDGER_UNIT','test-owner.service')
    state=tmp_path/'ledger.json';state.write_text(json.dumps({'waiting':{}}));monkeypatch.setenv('NATLANG_MEMORY_LEDGER',str(state))
    calls=[]
    def cancel(argv,**kwargs):
        if argv[0]=='systemctl':return SimpleNamespace(returncode=0,stdout=str(__import__('os').getpid()))
        calls.append(argv);return SimpleNamespace(returncode=1,stdout=json.dumps({'unit':'test-owner.service','cancelled':False}))
    monkeypatch.setattr(checkpoint.subprocess,'run',cancel)
    with checkpoint.CollectorCheckpoint():pass
    with checkpoint.CollectorCheckpoint()as owner:owner._request(signal.SIGTERM,None)
    cancellations=[c for c in calls if 'cancel' in c]
    assert len(cancellations)==2 and all(c[-3:]==['cancel','--unit','test-owner.service']for c in cancellations)
    assert all(c==['systemd-run','--version'] for c in calls if c not in cancellations)


def test_guard_owned_pressure_checkpoint_keeps_queue_despite_clean_process_exit(monkeypatch,tmp_path):
    monkeypatch.setenv('NATLANG_LEDGER_UNIT','test-owner.service')
    state=tmp_path/'ledger.json';state.write_text(json.dumps({'waiting':{'test-owner.service':{'requeued':True}}}));monkeypatch.setenv('NATLANG_MEMORY_LEDGER',str(state))
    monkeypatch.setattr(checkpoint.subprocess,'run',lambda argv,**k:SimpleNamespace(returncode=0,stdout=str(__import__('os').getpid())) if argv[0]=='systemctl' else pytest.fail('must preserve guard-owned continuation'))
    with checkpoint.CollectorCheckpoint()as owner:owner._request(signal.SIGTERM,None)
    assert owner.requested and not owner.continuation_cancelled
    assert json.loads(state.read_text())['waiting']['test-owner.service']['requeued'] is True


def test_explicit_stop_overrides_a_guard_owned_queue(monkeypatch,tmp_path):
    monkeypatch.setenv('NATLANG_LEDGER_UNIT','test-owner.service')
    state=tmp_path/'ledger.json';state.write_text(json.dumps({'waiting':{'test-owner.service':{'requeued':True}}}));monkeypatch.setenv('NATLANG_MEMORY_LEDGER',str(state))
    calls=[]
    monkeypatch.setattr(checkpoint.subprocess,'run',lambda argv,**kw:SimpleNamespace(returncode=0,stdout=str(__import__('os').getpid())) if argv[0]=='systemctl' else calls.append(argv)or SimpleNamespace(returncode=0,stdout=json.dumps({'unit':'test-owner.service','cancelled':True})))
    with checkpoint.CollectorCheckpoint()as owner:
        owner._request(signal.SIGTERM,None)
        owner.cancel_continuation()  # canonical stop-after-wave/runtime hold
    assert len(calls)==1 and owner.continuation_cancelled


def test_actual_child_guard_checkpoint_exits_zero_without_dropping_continuation(monkeypatch,tmp_path):
    import os
    state=tmp_path/'ledger.json';state.write_text(json.dumps({'waiting':{'test-owner.service':{'requeued':True}}}))
    env={**os.environ,'NATLANG_LEDGER_UNIT':'test-owner.service','NATLANG_MEMORY_LEDGER':str(state)}
    code='''import os,signal,json
from scripts.vibestudio.managed_checkpoint import CollectorCheckpoint
CollectorCheckpoint.ledger_owner_unit=lambda self:'test-owner.service'  # TEST-only owner observer
with CollectorCheckpoint()as owner:
 os.kill(os.getpid(),signal.SIGTERM)
 assert owner.requested
print(json.dumps({'checkpointed':owner.requested,'cancelled':owner.continuation_cancelled}))
'''
    result=subprocess.run([sys.executable,'-c',code],env=env,cwd=MODULE.parents[2],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)=={'checkpointed':True,'cancelled':False}
    assert json.loads(state.read_text())['waiting']['test-owner.service']['requeued'] is True


def test_inherited_unit_does_not_grant_parent_continuation_authority(monkeypatch):
    monkeypatch.setenv('NATLANG_LEDGER_UNIT','parent-owner.service')
    calls=[]
    def observe(argv,**kwargs):
        calls.append(argv)
        assert argv[0]=='systemctl', 'child must never issue cancellation'
        return SimpleNamespace(returncode=0,stdout=str(__import__('os').getpid()+1))
    monkeypatch.setattr(checkpoint.subprocess,'run',observe)
    with checkpoint.CollectorCheckpoint()as child:pass
    assert not child.continuation_cancelled and calls
