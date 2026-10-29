import shutil
import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.protocol import load_tensors,save_tensors
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator


def test_replay_exact_pending_merge_and_progress(tmp_path):
    root=tmp_path/'base';initialize_coordinator(root,run_id='r',masters={'weight':torch.arange(8,dtype=torch.float32)},identity={'parent':'p'},workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,outer_momentum=.5,ema={'decay':.88,'warmup':0})
    a=tmp_path/'a';b=tmp_path/'b';shutil.copytree(root,a);shutil.copytree(root,b);left=ReplayCoordinator(a,quorum_since={});right=ReplayCoordinator(b,quorum_since={});delta=tmp_path/'delta';save_tensors(delta,{'delta':torch.ones(8)})
    sessions=['00000000-0000-0000-0000-000000000001','00000000-0000-0000-0000-000000000002']
    actions=[('lease',{'worker':i,'session':sessions[i]}) for i in range(2)]+[('submit',{'index':0,'worker':i,'session':sessions[i],'sequence':1,'base_version':0,'steps':1,'tokens':3}) for i in range(2)]
    for n,(kind,args) in enumerate(actions):
        event={'kind':kind,'args':args,'wall':100+n,'monotonic':20+n};assert left.apply(event,delta)==right.apply(event,delta)
    for k,v in load_tensors(a/'fragments/00000.safetensors').items():assert torch.equal(v,load_tensors(b/'fragments/00000.safetensors')[k])


def test_journal_replay_rejects_unsealed_or_wrong_state(tmp_path):
    import json
    import pytest
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_state import replay
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024,grace_seconds=0)
    primary=tmp_path/'primary';target=tmp_path/'target';shutil.copytree(base,primary);shutil.copytree(base,target);machine=ReplayCoordinator(primary,quorum_since={})
    payload=tmp_path/'empty';payload.write_bytes(b'');j=ReplicaJournal(tmp_path/'wal',run_identity='r',baseline_sha256='a'*64)
    event={'kind':'lease','args':{'worker':0,'session':'00000000-0000-0000-0000-000000000001'},'wall':100,'monotonic':10};j.append(event,payload,expected_sequence=1,expected_previous=j.head)
    with pytest.raises(ValueError,match='unsealed'):replay(j,target,expected_sequence=1,expected_head=j.head,quorum_since={})
    state=machine.apply(event,payload);j.append({'kind':'seal','operation_sequence':1,'state':state},payload,expected_sequence=2,expected_previous=j.head)
    assert replay(j,target,expected_sequence=2,expected_head=j.head,quorum_since={}).coordinator.progress==machine.coordinator.progress
    j.close()


def test_kill_primary_replay_pending_and_continue_exactly(tmp_path):
    import multiprocessing as mp
    import os,signal,socket,time
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_transport import serve
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_state import replay
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.arange(8,dtype=torch.float32)},identity={'parent':'p'},workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,outer_momentum=.5,ema={'decay':.88,'warmup':0})
    primary=tmp_path/'primary';restored=tmp_path/'restored';expected=tmp_path/'expected'
    for target in (primary,restored,expected):shutil.copytree(base,target)
    delta=tmp_path/'delta';save_tensors(delta,{'delta':torch.ones(8)});empty=tmp_path/'empty';empty.write_bytes(b'')
    sessions=['00000000-0000-0000-0000-000000000001','00000000-0000-0000-0000-000000000002']
    events=[{'kind':'lease','args':{'worker':i,'session':sessions[i]},'wall':100+i,'monotonic':20+i} for i in range(2)]
    events += [{'kind':'submit','args':{'index':0,'worker':0,'session':sessions[0],'sequence':1,'base_version':0,'steps':1,'tokens':3},'wall':102,'monotonic':22}]
    final={'kind':'submit','args':{'index':0,'worker':1,'session':sessions[1],'sequence':1,'base_version':0,'steps':1,'tokens':3},'wall':103,'monotonic':23}
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);result=ctx.Pipe(duplex=False);replica=tmp_path/'replica'
    def receiver():
        journal=ReplicaJournal(replica,run_identity='r',baseline_sha256='a'*64);ready[1].send(journal.head);serve(journal,'127.0.0.1',port,'s'*32,max_bytes=1<<20,stop=stop);journal.close()
    receiver_proc=ctx.Process(target=receiver);receiver_proc.start();ready[0].recv();url=f'http://127.0.0.1:{port}'
    def leader():
        journal=ReplicaJournal(tmp_path/'leader-wal',run_identity='r',baseline_sha256='a'*64);machine=ReplayCoordinator(primary,quorum_since={});writer=ReplicatedExecutor(journal,url,'s'*32,allow_http=True,timeout=5)
        for event in events:
            source=delta if event['kind']=='submit' else empty
            writer.submit(event,source,lambda event=event,source=source:machine.apply(event,source),seal=True)
        result[1].send({'sequence':journal.sequence,'head':journal.head});time.sleep(30)
    time.sleep(.1);leader_proc=ctx.Process(target=leader);leader_proc.start();assert result[0].poll(20);ack=result[0].recv();os.kill(leader_proc.pid,signal.SIGKILL);leader_proc.join(5);assert leader_proc.exitcode==-signal.SIGKILL
    stop.set();receiver_proc.join(5);assert receiver_proc.exitcode==0
    journal=ReplicaJournal(replica,run_identity='r',baseline_sha256='a'*64);recovered=replay(journal,restored,expected_sequence=ack['sequence'],expected_head=ack['head'],quorum_since={});truth=ReplayCoordinator(expected,quorum_since={})
    for event in events:truth.apply(event,delta if event['kind']=='submit' else empty)
    assert recovered.apply(final,delta)==truth.apply(final,delta)
    for key,value in load_tensors(restored/'fragments/00000.safetensors').items():assert torch.equal(value,load_tensors(expected/'fragments/00000.safetensors')[key])
    journal.close()


def test_replicated_http_primary_kill_recovers_pending_then_merges(tmp_path):
    import multiprocessing as mp
    import os,signal,socket,time
    from natlang_neuralese.train.diloco.coordinator import serve_coordinator
    from natlang_neuralese.train.diloco.worker import Transport
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_transport import serve
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco.replica_state import replay
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.arange(8,dtype=torch.float32)},identity={'parent':'p'},workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,outer_momentum=.5,ema={'decay':.88,'warmup':0})
    primary=tmp_path/'primary';restored=tmp_path/'restored';shutil.copytree(base,primary)
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_promotion import promote
    baseline_machine=ReplayCoordinator(base,quorum_since={})
    class CaptureDispatch:
        machine=baseline_machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';baseline_receipt=capture(CaptureDispatch(),baseline);baseline_sha=baseline_receipt['sha256']
    delta=tmp_path/'delta';save_tensors(delta,{'delta':torch.ones(8)});empty=tmp_path/'empty';empty.write_bytes(b'')
    def freeport():
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
    peerport=freeport();port=freeport();ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);commands=ctx.Pipe();replica=tmp_path/'replica';token='s'*32
    def receiver():
        j=ReplicaJournal(replica,run_identity='r',baseline_sha256=baseline_sha);ready[1].send(True);serve(j,'127.0.0.1',peerport,token,max_bytes=1<<20,stop=stop);j.close()
    remote=ctx.Process(target=receiver);remote.start();ready[0].recv();time.sleep(.1)
    def leader():
        import threading
        j=ReplicaJournal(tmp_path/'primary-wal',run_identity='r',baseline_sha256=baseline_sha);machine=ReplayCoordinator(primary,quorum_since={});writer=ReplicatedExecutor(j,f'http://127.0.0.1:{peerport}',token,allow_http=True,timeout=5);dispatch=ReplicatedDispatch(machine,writer,empty)
        thread=threading.Thread(target=serve_coordinator,args=(primary,'127.0.0.1',port,token),kwargs={'replication':dispatch},daemon=True);thread.start();commands[1].send('started');commands[1].recv();commands[1].send(writer.read(lambda:{'sequence':j.sequence,'head':j.head}));time.sleep(30)
    leader_proc=ctx.Process(target=leader);leader_proc.start();assert commands[0].recv()=='started';client=Transport(f'http://127.0.0.1:{port}',token,5,allow_http=True)
    for _ in range(100):
        try:client.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    sessions=['00000000-0000-0000-0000-000000000001','00000000-0000-0000-0000-000000000002']
    for i in range(2):client.json('POST','/v1/lease',{'worker':i,'session':sessions[i]})
    from natlang_neuralese.train.diloco.worker import RemoteError
    import pytest
    with pytest.raises(RemoteError):
        client.request('POST','/v1/updates/0',file=delta,headers={'X-Worker':'0','X-Session':sessions[0],'X-Sequence':'1','X-Base-Version':'99','X-Steps':'1','X-Tokens':'3'})
    status,value,_=client.request('POST','/v1/updates/0',file=delta,headers={'X-Worker':'0','X-Session':sessions[0],'X-Sequence':'1','X-Base-Version':'0','X-Steps':'1','X-Tokens':'3'});assert status==200 and value['status']=='queued'
    commands[0].send('head');assert commands[0].poll(20);ack=commands[0].recv();os.kill(leader_proc.pid,signal.SIGKILL);leader_proc.join(5);assert leader_proc.exitcode==-signal.SIGKILL
    stop.set();remote.join(5);assert remote.exitcode==0
    j=ReplicaJournal(replica,run_identity='r',baseline_sha256=baseline_sha)
    def verify_fence(_):
        assert leader_proc.exitcode==-signal.SIGKILL
        return {'primary_fenced':True,'all_workers_quiesced':True,'primary_boot_id':'fixture-boot','run_id':'r','proof_sha256':'f'*64}
    machine,promotion=promote(baseline=baseline,baseline_sha256=baseline_sha,journal=j,sequence=ack['sequence'],head=ack['head'],destination=restored,supervisor_proof={},verify_supervisor_fence=verify_fence)
    assert promotion['fence']['primary_fenced']
    assert (restored/'pending/00000/000.safetensors').is_file()
    truth=ReplayCoordinator(primary,quorum_since={})
    event={'kind':'submit','args':{'index':0,'worker':1,'session':sessions[1],'sequence':1,'base_version':0,'steps':1,'tokens':3},'wall':time.time(),'monotonic':time.monotonic()}
    state=machine.apply(event,delta);assert state==truth.apply(event,delta) and state['versions']==[1]
    recovered=load_tensors(restored/'fragments/00000.safetensors');expected=load_tensors(primary/'fragments/00000.safetensors')
    assert set(recovered)=={'theta','momentum','ema'} and all(torch.equal(recovered[k],expected[k]) for k in recovered);j.close()


def test_baseline_pins_all_authoritative_files_and_rejects_tamper(tmp_path):
    import json
    import pytest
    from natlang_neuralese.train.diloco.replica_baseline import capture,verify
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(root,quorum_since={})
    class Dispatch:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    pinned=tmp_path/'baseline';receipt=capture(Dispatch(),pinned);value=verify(pinned,receipt['sha256']);assert value['quorum_since']=={}
    progress=root/'progress.json';old=(pinned/'progress.json').read_bytes()
    # Atomic replacement cannot alter the pinned inode.
    replacement=root/'new-progress';replacement.write_text('{}');replacement.replace(progress)
    assert (pinned/'progress.json').read_bytes()==old
    (pinned/'fragments/00000.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='baseline file differs'):verify(pinned,receipt['sha256'])


def test_promotion_requires_external_fence_and_preserves_baseline(tmp_path):
    import pytest
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_promotion import promote
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(root,quorum_since={})
    class Dispatch:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'base';receipt=capture(Dispatch(),baseline);j=ReplicaJournal(tmp_path/'wal',run_identity='r',baseline_sha256=receipt['sha256'])
    with pytest.raises(ValueError,match='fence'):
        promote(baseline=baseline,baseline_sha256=receipt['sha256'],journal=j,sequence=0,head=j.head,destination=tmp_path/'no',supervisor_proof={},verify_supervisor_fence=None)
    evidence={'primary_fenced':True,'all_workers_quiesced':True,'primary_boot_id':'boot','run_id':'r','proof_sha256':'a'*64}
    result,proof=promote(baseline=baseline,baseline_sha256=receipt['sha256'],journal=j,sequence=0,head=j.head,destination=tmp_path/'yes',supervisor_proof={},verify_supervisor_fence=lambda _:evidence)
    assert proof['fence']==evidence and result.coordinator.manifest['run_id']=='r'
    assert (baseline/'fragments/00000.safetensors').is_file();j.close()


def test_manifest_stays_responsive_while_slow_replica_blocks_mutable_status(tmp_path,monkeypatch):
    import threading,time,socket
    from natlang_neuralese.train.diloco.coordinator import serve_coordinator
    from natlang_neuralese.train.diloco.worker import Transport
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco import replica_executor
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(root,quorum_since={});journal=ReplicaJournal(tmp_path/'journal',run_identity='r',baseline_sha256='a'*64)
    blocked=threading.Event();release=threading.Event()
    def delayed(*args,**kwargs):
        blocked.set();assert release.wait(10);return {'sequence':journal.sequence,'head':journal.head}
    monkeypatch.setattr(replica_executor,'transmit',delayed);writer=ReplicatedExecutor(journal,'http://unused','s'*32,allow_http=True);empty=tmp_path/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(machine,writer,empty)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,'s'*32),kwargs={'stop':stop,'replication':dispatch},daemon=True);thread.start();client=Transport(f'http://127.0.0.1:{port}','s'*32,5,allow_http=True)
    for _ in range(100):
        try:client.json('GET','/v1/manifest');break
        except OSError:time.sleep(.02)
    lease_result=[];status_result=[];fragment_result=[]
    lease=threading.Thread(target=lambda:lease_result.append(client.json('POST','/v1/lease',{'worker':0,'session':'00000000-0000-0000-0000-000000000001'})));lease.start();assert blocked.wait(5)
    start=time.monotonic();manifest=client.json('GET','/v1/manifest');assert manifest['run_id']=='r' and time.monotonic()-start<1
    status=threading.Thread(target=lambda:status_result.append(client.json('GET','/v1/status')));status.start()
    fragment=threading.Thread(target=lambda:fragment_result.append(client.request('GET','/v1/fragments/0')));fragment.start();time.sleep(.1);assert not status_result and not fragment_result
    release.set();lease.join(10);status.join(10);fragment.join(10);assert lease_result and status_result and fragment_result
    stop.set();thread.join(5);writer.close();journal.close()
