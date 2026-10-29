import json
from pathlib import Path
import pytest
from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal


def test_durable_chain_reopens_and_rejects_gap_or_payload_mutation(tmp_path):
    payload = tmp_path/'source';payload.write_bytes(b'payload')
    root=tmp_path/'replica';j=ReplicaJournal(root,run_identity='run',baseline_sha256='a'*64)
    initial=j.head
    first=j.append({'kind':'accepted-update'},payload,expected_sequence=1,expected_previous=initial)
    assert j.append({'kind':'accepted-update'},payload,expected_sequence=1,expected_previous=initial)==first
    with pytest.raises(ValueError,match='duplicate differs'):j.append({'kind':'other'},payload,expected_sequence=1,expected_previous=initial)
    with pytest.raises(ValueError):j.append({},payload,expected_sequence=3,expected_previous=first['head'])
    with pytest.raises(ValueError,match='already has a writer'):ReplicaJournal(root,run_identity='run',baseline_sha256='a'*64)
    j.close()
    with pytest.raises(ValueError):ReplicaJournal(root,run_identity='other',baseline_sha256='a'*64)
    again=ReplicaJournal(root,run_identity='run',baseline_sha256='a'*64);assert again.verify()==first
    with pytest.raises(ValueError,match='committed head'):again.verify(expected_sequence=2,expected_head='0'*64)
    (root/'00000000000000000001'/'payload').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload mismatch'):again.verify()


def test_streamed_replica_ack_is_durable_and_retry_safe_across_process(tmp_path):
    import multiprocessing as mp
    import socket
    from natlang_neuralese.train.diloco.replica_transport import serve, transmit
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);root=tmp_path/'server'
    def child():
        j=ReplicaJournal(root,run_identity='run',baseline_sha256='a'*64)
        ready[1].send(j.head)
        serve(j,'127.0.0.1',port,'s'*32,max_bytes=4<<20,stop=stop,allow_unmaterialized_for_test=True)
        j.close()
    proc=ctx.Process(target=child);proc.start();initial=ready[0].recv()
    payload=tmp_path/'payload';payload.write_bytes(b'x'*(3<<20));url=f'http://127.0.0.1:{port}'
    import time
    for _ in range(100):
        try:
            ack=transmit(url,'s'*32,sequence=1,previous=initial,event={'kind':'delta'},payload=payload,allow_http=True)
            break
        except OSError:time.sleep(.02)
    else:pytest.fail('replica server did not start')
    assert transmit(url,'s'*32,sequence=1,previous=initial,event={'kind':'delta'},payload=payload,allow_http=True)==ack
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    local=ReplicaJournal(tmp_path/'local',run_identity='run',baseline_sha256='a'*64)
    assert local.append({'kind':'delta'},payload,expected_sequence=1,expected_previous=initial)==ack
    writer=ReplicatedExecutor(local,url,'s'*32,allow_http=True)
    result=writer.submit({'kind':'second'},payload,lambda:{'result':'committed'},seal=True)
    assert result=={'result':'committed'} and local.sequence==3
    final={'sequence':local.sequence,'head':local.head};writer.close();local.close()
    stop.set();proc.join(5);assert proc.exitcode==0
    recovered=ReplicaJournal(root,run_identity='run',baseline_sha256='a'*64)
    assert recovered.verify(expected_sequence=3,expected_head=final['head'])==final
    recovered.close()


def test_executor_never_applies_when_replica_ack_fails(tmp_path):
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    journal=ReplicaJournal(tmp_path/'primary',run_identity='run',baseline_sha256='a'*64)
    payload=tmp_path/'payload';payload.write_bytes(b'event')
    e=ReplicatedExecutor(journal,'http://127.0.0.1:1','s'*32,allow_http=True,timeout=.2)
    applied=[]
    with pytest.raises(RuntimeError,match='failed closed'):e.submit({'kind':'event'},payload,lambda:applied.append(1))
    with pytest.raises(RuntimeError,match='failed closed'):e.submit({'kind':'later'},payload,lambda:applied.append(2))
    assert applied==[];e.close();journal.close()


def test_storage_budget_fails_closed_without_discarding_acknowledged_events(tmp_path):
    payload=tmp_path/'source';payload.write_bytes(b'1234')
    j=ReplicaJournal(tmp_path/'journal',run_identity='run',baseline_sha256='a'*64,max_payload_bytes=6)
    first=j.append({'kind':'first'},payload,expected_sequence=1,expected_previous=j.head)
    with pytest.raises(ValueError,match='budget exhausted'):j.append({'kind':'second'},payload,expected_sequence=2,expected_previous=j.head)
    assert j.verify(expected_sequence=1,expected_head=first['head'])==first
    j.close()


def test_live_peer_credential_rotation_rejects_old_token_and_survives_restart(tmp_path):
    import json,multiprocessing as mp,socket,time
    from natlang_neuralese.train.diloco.replica_transport import serve,witness_head,rotate_replica_credential
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.worker import RemoteError
    old='o'*32;new='n'*32;tokenpath=tmp_path/'token';tokenpath.write_text(old)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);root=tmp_path/'wal'
    def peer():
        journal=ReplicaJournal(root,run_identity='r',baseline_sha256='a'*64)
        ready[1].send(True);serve(journal,'127.0.0.1',port,tokenpath.read_text(),max_bytes=1024,stop=stop,machine=object(),token_path=tokenpath);journal.close()
    p=ctx.Process(target=peer);p.start();ready[0].recv();url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:head=witness_head(url,old,allow_http=True);break
            except OSError:time.sleep(.02)
        rotate_replica_credential(url,old,new,sequence=head['sequence'],head=head['head'],nonce='z'*48,allow_http=True)
        assert tokenpath.read_text()==new and tokenpath.stat().st_mode & 0o077==0
        history=json.loads((root.parent/'credential-history'/('z'*48+'.json')).read_text())
        assert history['state']=='committed' and history['old_token_sha256']==__import__('hashlib').sha256(old.encode()).hexdigest() and history['new_token_sha256']==__import__('hashlib').sha256(new.encode()).hexdigest()
        with pytest.raises(RemoteError):witness_head(url,old,allow_http=True)
        assert witness_head(url,new,allow_http=True)==head
        stop.set();p.join(8);assert p.exitcode==0;stop.clear()
        import json
        receipt=root.parent/'credential-rotation.json';state=json.loads(receipt.read_text());state['state']='prepared';receipt.write_text(json.dumps(state))
        p=ctx.Process(target=peer);p.start();ready[0].recv()
        for _ in range(100):
            try:assert witness_head(url,new,allow_http=True)==head;break
            except OSError:time.sleep(.02)
        assert json.loads(receipt.read_text())['state']=='committed'
    finally:stop.set();p.join(8)
    assert p.exitcode==0


def _peer(tmp_path, *, slow_seal=False, diverge=False):
    import multiprocessing as mp,socket,time
    from natlang_neuralese.train.diloco.replica_transport import serve
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False)
    def child():
        j=ReplicaJournal(tmp_path/'peer',run_identity='run',baseline_sha256='a'*64)
        if diverge:
            other=tmp_path/'other';other.write_bytes(b'peer-only')
            j.append({'kind':'peer-only'},other,expected_sequence=1,expected_previous=j.head)
        if slow_seal:
            original=ReplicaJournal.append;state={'slow':True}
            def append(self,event,payload,**kw):
                result=original(self,event,payload,**kw)
                if event.get('kind')=='seal' and state['slow']:state['slow']=False;time.sleep(1.5)  # durable, ack delayed
                return result
            ReplicaJournal.append=append
        ready[1].send(True)
        serve(j,'127.0.0.1',port,'s'*32,max_bytes=4<<20,stop=stop,allow_unmaterialized_for_test=True)
        j.close()
    proc=ctx.Process(target=child);proc.start();ready[0].recv();time.sleep(.2)
    return f'http://127.0.0.1:{port}',stop,proc


def test_slow_seal_ack_is_retried_identically_without_fencing(tmp_path):
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    url,stop,proc=_peer(tmp_path,slow_seal=True)
    local=ReplicaJournal(tmp_path/'local',run_identity='run',baseline_sha256='a'*64)
    payload=tmp_path/'payload';payload.write_bytes(b'operation')
    writer=ReplicatedExecutor(local,url,'s'*32,allow_http=True,timeout=.5,seal_timeout=.5,retry_backoff=(1.5,1.5))
    try:
        assert writer.submit({'kind':'op'},payload,lambda:{'state':1},seal=True)=={'state':1}
        assert writer.failure is None and [r['sequence'] for r in writer.retries]==[2]
        assert writer.submit({'kind':'next'},payload,lambda:{'state':2},seal=True)=={'state':2} and local.sequence==4
        final={'sequence':local.sequence,'head':local.head}
    finally:writer.close();local.close();stop.set();proc.join(5)
    peer=ReplicaJournal(tmp_path/'peer',run_identity='run',baseline_sha256='a'*64)
    assert peer.verify(expected_sequence=4,expected_head=final['head'])==final;peer.close()


def test_divergent_peer_record_still_fences_without_retry(tmp_path, capfd):
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    url,stop,proc=_peer(tmp_path,diverge=True)
    local=ReplicaJournal(tmp_path/'local',run_identity='run',baseline_sha256='a'*64)
    payload=tmp_path/'payload';payload.write_bytes(b'operation');applied=[]
    writer=ReplicatedExecutor(local,url,'s'*32,allow_http=True,timeout=2,retry_backoff=(.1,.1))
    try:
        with pytest.raises(RuntimeError,match='failed closed'):writer.submit({'kind':'op'},payload,lambda:applied.append(1))
        assert writer.retries==[] and applied==[] and 'HTTP 400' in str(writer.failure)
        with pytest.raises(RuntimeError,match='failed closed'):writer.submit({'kind':'later'},payload,lambda:applied.append(2))
        assert 'replicated writer fenced at sequence 1' in capfd.readouterr().err
    finally:writer.close();local.close();stop.set();proc.join(5)
