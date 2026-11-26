"""Actual same-connection cancellation ownership; no model or training data."""
import io,json,os,socket,subprocess,threading
from pathlib import Path
import pytest
import run_bonsai_queue as queue
from natlang_neuralese.rl import episode_files as ef


def child_pair(tmp_path):
    ours,theirs=socket.socketpair()
    child=queue.SharedChild.__new__(queue.SharedChild)
    child.connection=ours;child.returncode=None;child._buffer=b'';child.log=io.StringIO()
    child.request_id='a'*32;child.server_pid=os.getpid();child.cancel_requested=False
    child.jobs=tmp_path/'sample/jobs';child.jobs.mkdir(parents=True)
    return child,theirs


def test_cancel_joins_finalizer_before_terminal_or_compaction(tmp_path):
    child,peer=child_pair(tmp_path);gate=threading.Event();received=threading.Event();file=child.jobs/'one.partial.json'
    def serve():
        request=json.loads(peer.recv(4096));assert request=={'cancel':True,'request_id':child.request_id}
        received.set();gate.wait(5);file.write_text('finalized native evidence')
        peer.sendall((json.dumps({'exit':2,'request_id':child.request_id})+'\n').encode());peer.close()
    thread=threading.Thread(target=serve);thread.start()
    try:
        child.terminate();child.terminate();assert received.wait(5)
        assert child.returncode is None
        with pytest.raises(subprocess.TimeoutExpired):child.wait(.01)
        assert not file.exists();gate.set();assert child.wait(5)==2;assert file.read_text()=='finalized native evidence'
    finally:gate.set();thread.join(5);child.connection.close();peer.close()


def test_foreign_terminal_cannot_certify_cleanup(tmp_path):
    child,peer=child_pair(tmp_path)
    try:
        peer.sendall((json.dumps({'exit':0,'request_id':'b'*32})+'\n').encode())
        with pytest.raises(queue.SharedCollectorUnjoined):child.wait(1)
        assert ef.unjoined_writer(child.jobs.parent)
    finally:peer.close();child.connection.close()


def test_unjoined_cancel_preserves_both_snapshots_and_refuses_compaction(tmp_path):
    child,peer=child_pair(tmp_path);raw=child.jobs/'one.partial.json';raw.write_bytes(b'new'*4096)
    packed=raw.with_name(raw.name+'.zst');packed.write_bytes(ef._compress(b'old'*4096))
    try:
        child.terminate()
        with pytest.raises(queue.SharedCollectorUnjoined):child.kill()
        assert not ef.quiet_sample(child.jobs.parent,10**12,0,set())
        assert ef.compact_sample(child.jobs.parent)=={}
        assert raw.read_bytes()==b'new'*4096 and ef.read_bytes(packed)==b'old'*4096
    finally:peer.close();child.connection.close()


def test_protocol3_is_not_reused_as_a_joinable_server(tmp_path):
    cli=tmp_path/'cli.js';cli.write_text('natlang-collector-serve/3')
    assert not queue.SharedChild.supported(cli)
    cli.write_text('natlang-collector-serve/4');assert queue.SharedChild.supported(cli)


def test_peer_close_while_cancelled_stays_unjoined(tmp_path):
    child,peer=child_pair(tmp_path);child.terminate();peer.close()
    with pytest.raises(queue.SharedCollectorUnjoined):child.wait(1)
    assert child.returncode is None
    assert ef.unjoined_writer(child.jobs.parent)


def test_actual_node_server_cancel_finalizes_exact_attempt_and_preserves_sibling(tmp_path, monkeypatch):
    import http.server,time,tempfile,shutil,signal
    repo=Path(__file__).resolve().parents[1];cli=repo/'ts-host/dist/teacher/cli.js'
    assert queue.SharedChild.supported(cli),'build this exact client/server source before the integration control'
    arrived=threading.Event();release=threading.Event()
    class Chat(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('content-length','0')));arrived.set();release.wait(20)
            try:self.send_response(503);self.end_headers()
            except OSError:pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Chat);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    short=tempfile.mkdtemp(prefix='nlcancel-',dir='/tmp');monkeypatch.setattr(queue.SharedChild,'ROOT',short);monkeypatch.setenv('INVOCATION_ID','TEST-only-cancel-join-'+tmp_path.name)
    source=tmp_path/'source.ir.jsonl';source.write_text(json.dumps({'version':'natlang.program/2','id':'TEST-only-cancelled-call','kind':'lambda_source','source_layout':{'root':'solve.nl','subfunctions':{}},'semantics':{'root':'solve.nl','files':{'solve.nl':'---\nargs:\n  problem: string\nreturns: number\n---\nAnswer the question.\n'},'inputs':{'problem':'A TEST-only cancellation control'},'expected':4,'oracle':'normalized'}})+'\n')
    jobs=tmp_path/'jobs';jobs.mkdir();url='http://127.0.0.1:'+str(server.server_address[1]);command=['node','--x',str(cli),str(source),str(jobs),str(tmp_path/'out.jsonl'),'--start','0','--limit','1','--model-id','TEST-only-local-no-model','--root-seed','0','--workers','1','--max-turns','2','--model-concurrency','1','--max-model-requests','2','--transport-retries','0','--server',url,'--chat-completions-url',url+'/v1/chat/completions','--context-tokens','32768','--kv-tokens','0']
    child=None;sibling=None
    try:
        child=queue.SharedChild(command,io.StringIO(),2);assert arrived.wait(30)
        child.connection.sendall((json.dumps({'cancel':True,'request_id':'b'*32})+'\n').encode())
        with pytest.raises(subprocess.TimeoutExpired):child.wait(.02)
        sibling=queue.SharedChild(['node','--x',str(cli),'--capabilities','json'],io.StringIO(),2)
        assert sibling.server_pid==child.server_pid;assert sibling.wait(10)==0
        child.terminate();code=child.wait(30);assert isinstance(code,int);assert not child.lost
        snapshots={str(p):p.read_bytes()for p in jobs.iterdir()if p.is_file()};assert any('.partial.json' in p for p in snapshots)
        release.set();time.sleep(.05)
        assert snapshots=={str(p):p.read_bytes()for p in jobs.iterdir()if p.is_file()}
        assert not list(jobs.glob('.shared-cancel-unjoined-*'))
    finally:
        release.set();server.shutdown();server.server_close();thread.join(5)
        if child is not None:
            try:os.kill(child.server_pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:os.waitpid(child.server_pid,0)
            except ChildProcessError:pass
        shutil.rmtree(short,ignore_errors=True)


def test_a_killed_unreaped_server_is_not_a_live_writer():
    import sys
    child=subprocess.Popen([sys.executable,'-c','pass'])
    try:
        os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
        assert not ef._pid_alive(child.pid)
    finally:child.wait()
