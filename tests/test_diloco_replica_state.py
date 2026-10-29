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
    pending=replay(j,target,expected_sequence=1,expected_head=j.head,quorum_since={},receiver_restart=True)
    assert pending.coordinator.progress['leases']=={}
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
        journal=ReplicaJournal(replica,run_identity='r',baseline_sha256='a'*64);ready[1].send(journal.head);serve(journal,'127.0.0.1',port,'s'*32,max_bytes=1<<20,stop=stop,allow_unmaterialized_for_test=True);journal.close()
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
    peer_state=tmp_path/'peer-state';shutil.copytree(base,peer_state)
    def receiver():
        j=ReplicaJournal(replica,run_identity='r',baseline_sha256=baseline_sha);peer_machine=ReplayCoordinator(peer_state,quorum_since={});ready[1].send(True);serve(j,'127.0.0.1',peerport,token,max_bytes=1<<20,stop=stop,machine=peer_machine);j.close()
    remote=ctx.Process(target=receiver);remote.start();ready[0].recv();time.sleep(.1)
    def leader():
        import threading
        from natlang_neuralese.train.diloco import replica_executor
        original_transmit=replica_executor.transmit
        def duplicated(*args,**kwargs):
            ack=original_transmit(*args,**kwargs)
            if kwargs['event'].get('kind')=='seal':assert original_transmit(*args,**kwargs)==ack
            return ack
        replica_executor.transmit=duplicated
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
    primary_identity={'boot_id':'fixture-boot','pid':leader_proc.pid,'start_ticks':'fixture-ticks'}
    def verify_fence(_):
        assert leader_proc.exitcode==-signal.SIGKILL
        return {'primary_fenced':True,'all_workers_quiesced':True,'primary':primary_identity,'quiesced_workers':[0,1],'run_id':'r','proof_sha256':'f'*64}
    machine,promotion=promote(baseline=baseline,baseline_sha256=baseline_sha,journal=j,sequence=ack['sequence'],head=ack['head'],destination=restored,supervisor_proof={},verify_supervisor_fence=verify_fence,expected_primary=primary_identity,expected_workers=[0,1])
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
        promote(baseline=baseline,baseline_sha256=receipt['sha256'],journal=j,sequence=0,head=j.head,destination=tmp_path/'no',supervisor_proof={},verify_supervisor_fence=None,expected_primary={'boot_id':'boot','pid':1,'start_ticks':'1'},expected_workers=[0])
    primary_identity={'boot_id':'boot','pid':1,'start_ticks':'1'}
    evidence={'primary_fenced':True,'all_workers_quiesced':True,'primary':primary_identity,'quiesced_workers':[0],'run_id':'r','proof_sha256':'a'*64}
    result,proof=promote(baseline=baseline,baseline_sha256=receipt['sha256'],journal=j,sequence=0,head=j.head,destination=tmp_path/'yes',supervisor_proof={},verify_supervisor_fence=lambda _:evidence,expected_primary=primary_identity,expected_workers=[0])
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


def test_replay_preserves_reanchor_source_transition_completion_and_reopen(tmp_path):
    from natlang_neuralese.train.native_diloco import source_identity_transition
    original={'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':'a'*64,'trainer_argv':['--native-graphs','/old','--epochs','1'],'diloco':{'workers':1}}
    current={**original,'trainer_argv':['--native-graphs','/new','--source-correction','/proof','--epochs','1']}
    root=tmp_path/'base';manifest=initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity=original,workers=1,min_quorum=1,fragment_bytes=1024)
    a=tmp_path/'a';b=tmp_path/'b';shutil.copytree(root,a);shutil.copytree(root,b);left=ReplayCoordinator(a,quorum_since={});right=ReplayCoordinator(b,quorum_since={});empty=tmp_path/'empty';empty.write_bytes(b'');session='00000000-0000-0000-0000-000000000001'
    proof={'schema':'natlang.diloco-source-identity-transition/1','correction_verified':True,'checkpoint_sha256':'b'*64,'correction_sha256':'c'*64,'previous_source_hash':'d'*64,'source_hash':'e'*64}
    transition={**source_identity_transition(original,current,proof),'worker':0,'fingerprint':manifest['fingerprint']}
    metrics={'reanchor_session':session,'reanchor_fingerprint':manifest['fingerprint'],'reanchor_checkpoint_sha256':'f'*64,'reanchor_source_hash':'d'*64,'reanchor_held_sha256':'a'*64}
    actions=[('lease',{'worker':0,'session':session}),('heartbeat',{'worker':0,'session':session,'steps':1,'tokens':1,'metrics':metrics}),('identity_transition',{'worker':0,'session':session,'transition':transition}),('completion',{'worker':0,'session':session,'drained':False,'generation':'d'*64}),('completion',{'worker':0,'session':session,'drained':True,'sequences':[0],'generation':'d'*64}),('reopen',{'worker':0,'session':session,'previous':'d'*64,'generation':'e'*64,'sequences':[0]})]
    for i,(kind,args) in enumerate(actions):
        event={'kind':kind,'args':args,'wall':100+i,'monotonic':20+i};state=left.apply(event,empty);assert state==right.apply(event,empty) and state['error'] is None
    progress=left.coordinator.progress;assert progress['reanchor_release']['source_hash']=='d'*64
    assert progress['identity_transitions']['0'][0]['source_hash']=='e'*64 and progress['reopened']['0']['generation']=='e'*64
    assert progress==right.coordinator.progress


def test_generation_switch_matches_peer_without_transferring_weights(tmp_path):
    from natlang_neuralese.train.diloco.replica_generation import rotate,prepare_peer_generation
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024);peer=tmp_path/'peer';shutil.copytree(root,peer)
    left=ReplayCoordinator(root,quorum_since={});right=ReplayCoordinator(peer,quorum_since={});leader=ReplicaJournal(tmp_path/'leader-wal',run_identity='r',baseline_sha256='a'*64);remote=ReplicaJournal(tmp_path/'peer-wal',run_identity='r',baseline_sha256='a'*64);writer=ReplicatedExecutor(leader,'http://unused','s'*32,allow_http=True);empty=tmp_path/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(left,writer,empty);staged={}
    def prepare(previous,clock,wall):
        baseline,journal=prepare_peer_generation(right,remote,tmp_path/'peer-base-next',tmp_path/'peer-wal-next',previous=previous,captured_monotonic=clock,captured_wall=wall);staged['journal']=journal
        return {'baseline_sha256':baseline['sha256'],'new_head':journal.head}
    def commit(intent):return {'baseline_sha256':intent['baseline_sha256'],'new_head':staged['journal'].head}
    receipt=rotate(dispatch,local_baseline_dir=tmp_path/'leader-base-next',local_journal_dir=tmp_path/'leader-wal-next',receipt_path=tmp_path/'generation.json',peer_prepare=prepare,peer_commit=commit)
    assert receipt['state']=='committed' and writer.journal.identity['previous_generation']==receipt['previous']
    assert (tmp_path/'leader-wal').is_dir() and (tmp_path/'peer-wal').is_dir()
    writer.close();writer.journal.close();staged['journal'].close();remote.close()


def test_real_peer_generation_switch_preserves_materialized_state(tmp_path):
    import json
    import multiprocessing as mp
    import socket,time
    from natlang_neuralese.train.diloco.replica_generation import rotate
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco.replica_transport import serve,generation_request
    from natlang_neuralese.train.diloco.replica_promotion import recover_receiver
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    primary=tmp_path/'primary';peer=tmp_path/'peer';shutil.copytree(base,primary);shutil.copytree(base,peer);ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);token='s'*32
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    store=tmp_path/'peer-store';store.mkdir()
    def receiver():
        j=ReplicaJournal(store/'journal0',run_identity='r',baseline_sha256='a'*64,max_transactions=2);machine=ReplayCoordinator(peer,quorum_since={});ready[1].send(True);serve(j,'127.0.0.1',port,token,max_bytes=1<<20,stop=stop,machine=machine)
    proc=ctx.Process(target=receiver);proc.start();ready[0].recv();time.sleep(.1);url=f'http://127.0.0.1:{port}';j=ReplicaJournal(tmp_path/'leader-wal0',run_identity='r',baseline_sha256='a'*64,max_transactions=2);machine=ReplayCoordinator(primary,quorum_since={});writer=ReplicatedExecutor(j,url,token,allow_http=True);empty=tmp_path/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(machine,writer,empty);session='00000000-0000-0000-0000-000000000001'
    dispatch.call('lease',{'worker':0,'session':session});generation='b'*32
    def prepare(previous,clock,wall):
        body={'id':generation,'previous':previous,'captured_monotonic':clock,'captured_wall':wall}
        result=generation_request(url,token,'prepare',body,allow_http=True)
        assert generation_request(url,token,'prepare',body,allow_http=True)==result
        return result
    def commit(intent):
        body={'id':generation,'previous':intent['previous'],'baseline_sha256':intent['baseline_sha256'],'new_head':intent['new_head']}
        result=generation_request(url,token,'commit',body,allow_http=True)
        assert generation_request(url,token,'commit',body,allow_http=True)==result
        return result
    writer.rollover_callback=lambda:rotate(dispatch,local_baseline_dir=tmp_path/'leader-base1',local_journal_dir=tmp_path/'leader-wal1',receipt_path=tmp_path/'generation.json',peer_prepare=prepare,peer_commit=commit,within_writer=True)
    dispatch.call('heartbeat',{'worker':0,'session':session,'steps':1,'tokens':3,'metrics':None});assert writer.journal.sequence==2
    receipt=json.loads((tmp_path/'generation.json').read_text());assert receipt['state']=='committed'
    writer.close();writer.journal.close();stop.set();proc.join(5);assert proc.exitcode==0
    pointer=json.loads((store/'replica-active.json').read_text());assert pointer['baseline_sha256']==receipt['baseline_sha256']
    from natlang_neuralese.train.diloco.replica_generation import load_active
    j2,recovered,prepared=load_active(store,run_identity='r',destination=tmp_path/'recovered');assert prepared is None and recovered.coordinator.progress==machine.coordinator.progress;j2.close()


def test_service_bootstrap_requires_fence_and_serves_with_peer(tmp_path):
    import multiprocessing as mp,socket,time
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_service import serve_peer,serve_new_primary
    from natlang_neuralese.train.diloco.worker import Transport
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(base,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Capture(),baseline);identity={'boot_id':'boot','pid':1,'start_ticks':'1'};evidence={'primary_fenced':True,'all_workers_quiesced':True,'primary':identity,'quiesced_workers':[0],'run_id':'r','proof_sha256':'f'*64}
    def freeport():
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
    peerport=freeport();port=freeport();ctx=mp.get_context('fork');peerstop=ctx.Event();leaderstop=ctx.Event();token='s'*32
    def peer():serve_peer(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'peerstore',work_root=tmp_path/'peerwork',host='127.0.0.1',port=peerport,token=token,max_transaction_bytes=1<<20,stop=peerstop)
    peerproc=ctx.Process(target=peer);peerproc.start();time.sleep(.2)
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(tmp_path/'peerstore'/'replica-files-frozen.json',{'schema':'natlang.diloco-replica-file-freeze/1','nonce':'f'*48})
    def leader():
        from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
        journal=ReplicaJournal(tmp_path/'oldjournal',run_identity='r',baseline_sha256=receipt['sha256'])
        serve_new_primary(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'leaderstore',root=tmp_path/'leaderroot',host='127.0.0.1',port=port,token=token,peer_url=f'http://127.0.0.1:{peerport}',supervisor_proof={},verify_supervisor_fence=lambda _:evidence,expected_primary=identity,expected_workers=[0],stop=leaderstop,journal=journal,sequence=journal.sequence,head=journal.head,session_transition={'previous_url':'http://old','new_url':'http://new','previous_token_sha256':__import__('hashlib').sha256(b'old-secret').hexdigest(),'new_token_sha256':__import__('hashlib').sha256(token.encode()).hexdigest(),'generation_nonce':'n'*48},previous_token='old-secret',file_custody_store=tmp_path/'peerstore',file_freeze_nonce='f'*48,allow_http=True)
    process=ctx.Process(target=leader);process.start();client=Transport(f'http://127.0.0.1:{port}',token,5,allow_http=True)
    for _ in range(150):
        try:client.json('GET','/v1/manifest');break
        except OSError:time.sleep(.02)
    ready=client.json('GET','/v1/replica-promotion');assert ready['ready'] and ready['run_id']=='r' and ready['sequence']==0
    assert client.json('POST','/v1/lease',{'worker':0,'session':'00000000-0000-0000-0000-000000000001'})['run_id']=='r'
    delta=tmp_path/'restart-delta';save_tensors(delta,{'delta':torch.ones(4)})
    status,answer,_=client.request('POST','/v1/updates/0',file=delta,headers={'X-Worker':'0','X-Session':'00000000-0000-0000-0000-000000000001','X-Sequence':'1','X-Base-Version':'0','X-Steps':'1','X-Tokens':'3'})
    assert status==200 and client.json('GET','/v1/status')['versions']==[1]
    import os,signal
    os.kill(process.pid,signal.SIGKILL);process.join(10);assert process.exitcode==-signal.SIGKILL
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    check=ReplicaJournal(tmp_path/'oldjournal',run_identity='r',baseline_sha256=receipt['sha256']);latest={'sequence':check.sequence,'head':check.head};check.close()
    assert latest['sequence']>2
    restartstop=ctx.Event()
    def restart(sequence,head,root):
        assert process.exitcode==-signal.SIGKILL
        journal=ReplicaJournal(tmp_path/'oldjournal',run_identity='r',baseline_sha256=receipt['sha256'])
        serve_new_primary(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'leaderstore',root=root,host='127.0.0.1',port=port,token=token,peer_url=f'http://127.0.0.1:{peerport}',supervisor_proof={},verify_supervisor_fence=lambda _:evidence,expected_primary=identity,expected_workers=[0],stop=restartstop,journal=journal,sequence=sequence,head=head,session_transition={'previous_url':'http://old','new_url':'http://new','previous_token_sha256':__import__('hashlib').sha256(b'old-secret').hexdigest(),'new_token_sha256':__import__('hashlib').sha256(token.encode()).hexdigest(),'generation_nonce':'n'*48},previous_token='old-secret',restart=True,allow_http=True)
    stale=ctx.Process(target=restart,args=(0,ready['head'],tmp_path/'staleroot'));stale.start();stale.join(10);assert stale.exitcode!=0 and not (tmp_path/'staleroot').exists()
    revived=ctx.Process(target=restart,args=(latest['sequence'],latest['head'],tmp_path/'restartroot'));revived.start()
    for _ in range(150):
        try:
            current=client.json('GET','/v1/replica-promotion');break
        except OSError:time.sleep(.02)
    assert current['sequence']==latest['sequence'] and current['head']==latest['head'] and current['sealed_rebase']['sequence']>latest['sequence']
    assert client.json('GET','/v1/status')['versions']==[1]
    restartstop.set();revived.join(10);assert revived.exitcode==0;peerstop.set();peerproc.join(10);assert peerproc.exitcode==0


def test_retention_three_real_rollovers_recover_after_primary_kill(tmp_path):
    import json,multiprocessing as mp,os,signal,socket,time
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.diloco.replica_generation import rotate,load_active
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco.replica_transport import serve,generation_request
    from natlang_neuralese.train.diloco.replica_retention import Retention
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    primary=tmp_path/'primary';peer=tmp_path/'peer';shutil.copytree(base,primary);shutil.copytree(base,peer)
    ctx=mp.get_context('fork');stop=ctx.Event();ready=ctx.Pipe(duplex=False);coord=ctx.Pipe();token='s'*32
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    store=tmp_path/'peer-store';store.mkdir()
    def receiver():
        j=ReplicaJournal(store/'journal0',run_identity='r',baseline_sha256='a'*64,max_transactions=2)
        m=ReplayCoordinator(peer,quorum_since={});ready[1].send(True);serve(j,'127.0.0.1',port,token,max_bytes=1<<20,stop=stop,machine=m,store_root=store)
    proc=ctx.Process(target=receiver);proc.start();ready[0].recv();url=f'http://127.0.0.1:{port}'
    def primary_process():
        leader_store=tmp_path/'leader-store';leader_store.mkdir();j=ReplicaJournal(leader_store/'journal0',run_identity='r',baseline_sha256='a'*64,max_transactions=2)
        machine=ReplayCoordinator(primary,quorum_since={});writer=ReplicatedExecutor(j,url,token,allow_http=True);empty=leader_store/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(machine,writer,empty)
        session='00000000-0000-0000-0000-000000000001';dispatch.call('lease',{'worker':0,'session':session})
        for n in range(1,4):
            ident=str(n)*32;directory=leader_store/('generation-'+ident)
            def prepare(previous,clock,wall):return generation_request(url,token,'prepare',{'id':ident,'previous':previous,'captured_monotonic':clock,'captured_wall':wall},allow_http=True)
            def commit(intent):return generation_request(url,token,'commit',{'id':ident,'previous':intent['previous'],'baseline_sha256':intent['baseline_sha256'],'new_head':intent['new_head']},allow_http=True)
            receipt=rotate(dispatch,local_baseline_dir=directory/'baseline',local_journal_dir=directory/'journal',receipt_path=directory/'generation.json',peer_prepare=prepare,peer_commit=commit)
            pointer=leader_store/'primary-active.json';write_json_durable(pointer,{'id':ident,'baseline_sha256':receipt['baseline_sha256'],'head':receipt['new_head'],'previous':receipt['previous']})
            confirmation=generation_request(url,token,'retain',{'id':ident,'previous':receipt['previous'],'baseline_sha256':receipt['baseline_sha256'],'new_head':receipt['new_head']},allow_http=True)
            retention=Retention(leader_store);retention.register(generation=ident,baseline_sha256=receipt['baseline_sha256'],head=receipt['new_head'],previous=retention.latest(),replica_confirmation=confirmation,active_pointer=pointer);retention.collect()
        coord[1].send({'progress':machine.coordinator.progress});time.sleep(30)
    leader=ctx.Process(target=primary_process);leader.start();assert coord[0].poll(30);saved=coord[0].recv();os.kill(leader.pid,signal.SIGKILL);leader.join(5);assert leader.exitcode==-signal.SIGKILL
    stop.set();proc.join(5);assert proc.exitcode==0
    for root in (store,tmp_path/'leader-store'):
        assert not (root/('generation-'+'1'*32)).exists()
        assert all((root/('generation-'+str(n)*32)).exists() for n in (2,3))
    recovered,machine,_=load_active(store,run_identity='r',destination=tmp_path/'recovered')
    assert machine.coordinator.progress==saved['progress'];recovered.close()


def test_cross_host_clock_rebase_sealed_and_pending_grace_exact(tmp_path):
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_state import replay
    root=tmp_path/'base';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=2,min_quorum=1,fragment_bytes=1024,grace_seconds=10)
    live=tmp_path/'live';restored=tmp_path/'restored';shutil.copytree(root,live);shutil.copytree(root,restored)
    machine=ReplayCoordinator(live,quorum_since={});journal=ReplicaJournal(tmp_path/'wal',run_identity='r',baseline_sha256='a'*64)
    empty=tmp_path/'empty';empty.write_bytes(b'');delta=tmp_path/'delta';save_tensors(delta,{'delta':torch.ones(4)})
    session='00000000-0000-0000-0000-000000000001'
    def sealed(kind,args,wall,mono,payload=empty):
        event={'kind':kind,'args':args,'wall':wall,'monotonic':mono};seq=journal.sequence+1
        journal.append(event,payload,expected_sequence=seq,expected_previous=journal.head)
        state=machine.apply(event,payload);assert state['error'] is None
        journal.append({'kind':'seal','operation_sequence':seq,'state':state},empty,expected_sequence=seq+1,expected_previous=journal.head)
        return state
    sealed('lease',{'worker':0,'session':session},99,999999)
    sealed('submit',{'index':0,'worker':0,'session':session,'sequence':1,'base_version':0,'steps':1,'tokens':3},100,1000000,delta)
    sealed('heartbeat',{'worker':0,'session':session,'steps':1,'tokens':3,'metrics':None},102,1000002)
    old={str(k):v for k,v in machine.coordinator.quorum_since.items()};assert old=={'0':1000000}
    change=sealed('clock_rebase',{'old_wall':102,'old_monotonic':1000002,'old_timers':old},105,1)
    assert change['result']['elapsed_wall']==3 and change['quorum_since']=={'0':-4}
    assert sealed('tick',{},109.9,5.9)['versions']==[0]
    assert sealed('tick',{},110.1,6.1)['versions']==[1]
    recovered=replay(journal,restored,expected_sequence=journal.sequence,expected_head=journal.head,quorum_since={})
    assert recovered.coordinator.progress==machine.coordinator.progress
    assert all(torch.equal(a,b) for a,b in zip(load_tensors(live/'fragments/00000.safetensors').values(),load_tensors(restored/'fragments/00000.safetensors').values()))
    journal.close()


def test_initial_peer_external_baseline_restarts_only_with_exact_pin(tmp_path):
    import multiprocessing as mp,socket,time
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_service import serve_peer
    from natlang_neuralese.train.diloco.replica_transport import witness_head
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(base,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'external-baseline';receipt=capture(Capture(),baseline);store=tmp_path/'store';token='s'*32
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event()
    def run(root,sha):serve_peer(baseline=baseline,baseline_sha256=sha,store=store,work_root=root,host='127.0.0.1',port=port,token=token,max_transaction_bytes=1<<20,stop=stop)
    first=ctx.Process(target=run,args=(tmp_path/'work1',receipt['sha256']));first.start()
    for _ in range(100):
        try:first_head=witness_head(f'http://127.0.0.1:{port}',token,allow_http=True);break
        except OSError:time.sleep(.02)
    stop.set();first.join(8);assert first.exitcode==0;stop.clear()
    bad=ctx.Process(target=run,args=(tmp_path/'bad','0'*64));bad.start();bad.join(8);assert bad.exitcode!=0
    again=ctx.Process(target=run,args=(tmp_path/'work2',receipt['sha256']));again.start()
    for _ in range(100):
        try:assert witness_head(f'http://127.0.0.1:{port}',token,allow_http=True)==first_head;break
        except OSError:time.sleep(.02)
    stop.set();again.join(8);assert again.exitcode==0


def test_pending_hash_cache_revalidates_exact_file_identity(tmp_path,monkeypatch):
    import hashlib,os
    root=tmp_path/'base';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(root,quorum_since={});path=tmp_path/'payload';path.write_bytes(b'first')
    digest=machine._immutable_hash(path);assert digest==hashlib.sha256(b'first').hexdigest()
    assert machine._immutable_hash(path)==digest and len(machine.hash_cache)==1
    before=path.stat();path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert machine._immutable_hash(path)==hashlib.sha256(b'other').hexdigest()
    replacement=tmp_path/'replacement';replacement.write_bytes(b'third');os.replace(replacement,path)
    assert machine._immutable_hash(path)==hashlib.sha256(b'third').hexdigest()


def test_first_outer_bootstrap_has_no_failed_primary_fence(tmp_path):
    import hashlib,multiprocessing as mp,socket,time
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_service import serve_peer,serve_new_primary
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.worker import Transport
    base=tmp_path/'base';initialize_coordinator(base,run_id='fresh',masters={'weight':torch.ones(4)},identity={'parent':'earned'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(base,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Capture(),baseline);token='s'*32;ctx=mp.get_context('fork');stop=ctx.Event();peerstop=ctx.Event()
    def port():
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
    pp=port();lp=port()
    def peer(root):serve_peer(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'peerstore',work_root=root,host='127.0.0.1',port=pp,token=token,max_transaction_bytes=1<<20,stop=peerstop)
    p=ctx.Process(target=peer,args=(tmp_path/'peerroot',));p.start();time.sleep(.2)
    def first():
        j=ReplicaJournal(tmp_path/'wal',run_identity='fresh',baseline_sha256=receipt['sha256'])
        serve_new_primary(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'store',root=tmp_path/'root',host='127.0.0.1',port=lp,token=token,peer_url=f'http://127.0.0.1:{pp}',supervisor_proof={},verify_supervisor_fence=lambda _:{'run_id':'fresh','source_verified':True},expected_primary=None,expected_workers=None,stop=stop,journal=j,sequence=0,head=j.head,session_transition={'new_url':'http://new','new_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'generation_nonce':'n'*48},previous_token=None,bootstrap=True,allow_http=True)
    leader=ctx.Process(target=first);leader.start();client=Transport(f'http://127.0.0.1:{lp}',token,3,allow_http=True)
    try:
        for _ in range(150):
            try:ready=client.json('GET','/v1/replica-promotion');break
            except OSError:time.sleep(.02)
        assert ready['run_id']=='fresh' and ready['sealed_rebase']['sequence']==2
        assert client.json('POST','/v1/lease',{'worker':0,'session':'00000000-0000-0000-0000-000000000001'})['run_id']=='fresh'
        payload=tmp_path/'source';payload.write_bytes(b'actual-source-custody')
        assert client.upload_file(payload,'source.bin',chunk_bytes=8)['status'] in ('present','committed')
        assert (tmp_path/'peerstore'/'replica-uploads'/'source.bin').read_bytes()==payload.read_bytes()
        peerstop.set();p.join(8);assert p.exitcode==0;peerstop.clear()
        import pytest
        from natlang_neuralese.train.diloco.worker import RemoteError
        later=tmp_path/'later';later.write_bytes(b'pending-after-peer-loss')
        with pytest.raises((RemoteError,OSError)):client.upload_file(later,'later.bin',chunk_bytes=100,attempts=1)
        assert (tmp_path/'root'/'uploads'/'.replication-pending'/'later.bin').exists()
        p=ctx.Process(target=peer,args=(tmp_path/'peerroot2',));p.start()
        from natlang_neuralese.train.diloco.replica_transport import witness_head
        for _ in range(150):
            try:witness_head(f'http://127.0.0.1:{pp}',token,allow_http=True);break
            except OSError:time.sleep(.02)
        assert client.upload_file(later,'later.bin',chunk_bytes=100,attempts=1)['status']=='present'
        assert not (tmp_path/'root'/'uploads'/'.replication-pending'/'later.bin').exists()
        assert (tmp_path/'peerstore'/'replica-uploads'/'later.bin').read_bytes()==later.read_bytes()
        from natlang_neuralese.train.diloco.replica_transport import freeze_replica_files
        from natlang_neuralese.train.diloco.replica_seed import restore_replica_uploads
        freeze_replica_files(f'http://127.0.0.1:{pp}',token,'f'*48,allow_http=True)
        restored=tmp_path/'restored-files';restored.mkdir()
        assert 'source.bin' in restore_replica_uploads(tmp_path/'peerstore',restored,freeze_nonce='f'*48)
        assert (restored/'uploads'/'source.bin').read_bytes()==payload.read_bytes()
    finally:stop.set();leader.join(8);peerstop.set();p.join(8)
    assert leader.exitcode==0 and p.exitcode==0


def test_reconcile_crash_after_peer_commit_rejects_old_pointer(tmp_path):
    import json,pytest
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_generation import reconcile_primary_pointer
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    m=ReplayCoordinator(base,quorum_since={})
    class Capture:
        def __init__(self):self.machine=m
        def read(self,fn):return fn()
    store=tmp_path/'store';store.mkdir();directory=store/('generation-'+'a'*32);directory.mkdir();baseline=directory/'baseline';receipt=capture(Capture(),baseline)
    j=ReplicaJournal(directory/'journal',run_identity='r',baseline_sha256=receipt['sha256']);head=j.head;j.close()
    previous={'sequence':2,'head':'b'*64}
    write_json_durable(directory/'generation.json',{'state':'prepared','baseline_sha256':receipt['sha256'],'baseline':str(baseline),'journal':str(directory/'journal'),'new_head':head,'previous':previous})
    pointer=store/'primary-active.json';write_json_durable(pointer,{'id':'initial','baseline_sha256':'f'*64,'journal':str(store/'old'),'baseline':str(tmp_path/'old'),'work_root':str(tmp_path/'old-root'),'head':'c'*64})
    witness={'run_id':'r','baseline_sha256':receipt['sha256'],'sequence':0,'head':head}
    with pytest.raises(ValueError):reconcile_primary_pointer(store,dict(witness,head='d'*64))
    assert json.loads(pointer.read_text())['id']=='initial'
    selected=reconcile_primary_pointer(store,witness)
    assert selected['id']=='a'*32 and json.loads(pointer.read_text())==selected


def test_replica_loss_after_seal_ack_loss_preserves_locally_sealed_state(tmp_path,monkeypatch):
    import pytest,time
    from natlang_neuralese.train.diloco import replica_executor
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
    from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
    from natlang_neuralese.train.diloco.replica_generation import load_active
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
    from natlang_neuralese.train.diloco.replica_promotion import recover_after_replica_loss
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(root,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Capture(),baseline)
    local=ReplicaJournal(tmp_path/'local',run_identity='r',baseline_sha256=receipt['sha256']);remote=ReplicaJournal(tmp_path/'remote',run_identity='r',baseline_sha256=receipt['sha256'])
    def lost_ack(_url,_token,*,sequence,previous,event,payload,**_):
        result=remote.append(event,payload,expected_sequence=sequence,expected_previous=previous)
        if event['kind']=='seal':raise OSError('peer disappeared after durable seal')
        return result
    monkeypatch.setattr(replica_executor,'transmit',lost_ack)
    empty=tmp_path/'empty';empty.write_bytes(b'');writer=ReplicatedExecutor(local,'http://unused','s'*32,allow_http=True)
    dispatch=ReplicatedDispatch(machine,writer,empty)
    with pytest.raises(RuntimeError):dispatch.call('lease',{'worker':0,'session':'00000000-0000-0000-0000-000000000001'})
    assert local.sequence==2 and remote.head==local.head
    with pytest.raises(RuntimeError):dispatch.call('tick',{})
    writer.close();remote.close()
    evidence={'run_id':'r','primary_process_absent':True,'replica_fenced':True,'all_workers_quiesced':True,'proof_sha256':'f'*64}
    recovered=recover_after_replica_loss(baseline=baseline,baseline_sha256=receipt['sha256'],journal=local,sequence=local.sequence,head=local.head,destination=tmp_path/'new-seed',verified_owner_evidence=evidence,source_work_root=root,freeze_nonce='f'*48)
    assert recovered['sealed_predecessor']=={'sequence':2,'head':local.head} and not recovered['unsealed_tail_retained']
    new,replayed,_=load_active(tmp_path/'new-seed',run_identity='r',destination=tmp_path/'peer')
    assert replayed.coordinator.progress==machine.coordinator.progress and new.identity['previous_generation']['head']==local.head
    new.close();sealed_head=local.head
    event={'kind':'tick','args':{},'wall':time.time(),'monotonic':time.monotonic()}
    local.append(event,empty,expected_sequence=3,expected_previous=local.head)
    second=recover_after_replica_loss(baseline=baseline,baseline_sha256=receipt['sha256'],journal=local,sequence=local.sequence,head=local.head,destination=tmp_path/'unsealed-seed',verified_owner_evidence=evidence,source_work_root=root,freeze_nonce='f'*48)
    assert second['unsealed_tail_retained'] and second['sealed_predecessor']=={'sequence':2,'head':sealed_head}
    assert (local.root/'00000000000000000003'/'record.json').is_file()
    local.close()
