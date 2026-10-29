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
        serve(j,'127.0.0.1',port,'s'*32,max_bytes=4<<20,stop=stop)
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
