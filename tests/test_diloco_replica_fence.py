import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.coreweave.diloco_replica_fence import owner_reference, verify_provider_fence


def test_provider_fence_exact_top_state_and_failures(tmp_path):
    primary={'sandbox_id':'old','boot_id':'boot','owner_id':'owner','pid':12,'start_ticks':'42'}
    worker={'sandbox_id':'peer','boot_id':'b','owner_id':'o','pid':13,'start_ticks':'43'}
    path=tmp_path/'owner.json'; path.write_text(json.dumps({'run_id':'run','coordinator':primary,'workers':{'0':worker}}))
    sha=hashlib.sha256(path.read_bytes()).hexdigest(); proof={'journal':{'path':str(path),'sha256':sha},'expected_run_id':'run'}
    def sandbox(state=3, sandbox_id='old'): return SimpleNamespace(sandbox_id=sandbox_id,status=SimpleNamespace(state=state,containers=[SimpleNamespace(state=2)]))
    def probe(identity,nonce): return {'identity':identity,'nonce':nonce,'process_absent':True,'owner_quiesced':True}
    def check(provider, worker_probe=probe): return verify_provider_fence(proof,expected_primary=primary,expected_workers=[0],get_sandbox=provider,probe_worker=worker_probe)
    assert check(lambda _:sandbox())['primary_fenced']
    for provider in (lambda _:sandbox(2),lambda _:sandbox(3,'different'),lambda _: (_ for _ in ()).throw(OSError('unavailable'))):
        with pytest.raises((ValueError,OSError)): check(provider)
    for changes in ({'nonce':'stale'},{'process_absent':False},{'owner_quiesced':False}):
        def bad(i,n): return dict(probe(i,n),**changes)
        with pytest.raises(ValueError): check(lambda _:sandbox(),bad)
    with pytest.raises(ValueError): owner_reference(path,sha,dict(primary,boot_id='old-boot'))
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError): check(lambda _:sandbox())


def test_actual_historical_owner_journal():
    path=Path('/home/werg/data/lfm3-sft-20261028/production-v3-live/four-rejoin-2512-ownership.json')
    if not path.is_file(): pytest.skip('host owner journal unavailable')
    journal=json.loads(path.read_text()); old=journal['coordinator']
    primary={key:old[key] for key in ('sandbox_id','boot_id','owner_id','pid','start_ticks')}
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert owner_reference(path,sha,primary)['run_id']==journal['run_id']
    with pytest.raises(ValueError): owner_reference(path,sha,dict(primary,owner_id='wrong'))


def test_local_role_observation_checks_actual_process(tmp_path):
    import os
    from scripts.coreweave.diloco_replica_fence import observe_local_worker
    pid=os.getpid()
    ticks=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identity={'sandbox_id':'peer','boot_id':boot,'owner_id':'owner','pid':999999999,'start_ticks':'1'}
    state={'schema':'natlang.diloco-role-supervisor/1','boot_id':boot,'owner_id':'owner',
           'supervisor_pid':pid,'supervisor_start_ticks':ticks,'roles':{'worker':{'pid':999999999,'alive':False}}}
    (tmp_path/'state.json').write_text(json.dumps(state))
    cmdsha=hashlib.sha256(Path(f'/proc/{pid}/cmdline').read_bytes()).hexdigest()
    (tmp_path/'promotion-fence.json').write_text(json.dumps({'schema':'natlang.diloco-role-promotion-fence/1','owner_id':'owner','boot_id':boot,'nonce':'x'*48}))
    assert observe_local_worker(identity,'nonce',tmp_path,'x'*48,cmdsha)['process_absent']
    with pytest.raises(ValueError): observe_local_worker(dict(identity,pid=pid,start_ticks=ticks),'nonce',tmp_path,'x'*48,cmdsha)
    state['roles']['worker']={'pid':pid,'alive':True}; (tmp_path/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError): observe_local_worker(identity,'nonce',tmp_path,'x'*48,cmdsha)
    state['roles']['worker']={'pid':999999999,'alive':False}; state['pending_start']={'role':'worker'}
    (tmp_path/'state.json').write_text(json.dumps(state))
    with pytest.raises(ValueError): observe_local_worker(identity,'nonce',tmp_path,'x'*48,cmdsha)


def test_completed_cohost_worker_never_executes_into_dead_sandbox(tmp_path):
    primary={'sandbox_id':'old','boot_id':'boot','owner_id':'owner','pid':12,'start_ticks':'42'}
    worker=dict(primary,pid=13,start_ticks='43')
    path=tmp_path/'owner.json'; path.write_text(json.dumps({'run_id':'run','coordinator':primary,'workers':{'0':worker}}))
    proof={'journal':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'expected_run_id':'run'}
    provider=lambda _:SimpleNamespace(sandbox_id='old',status=SimpleNamespace(state=3))
    probe=lambda *_: (_ for _ in ()).throw(AssertionError('completed cohost must not be execed'))
    result=verify_provider_fence(proof,expected_primary=primary,expected_workers=[0],get_sandbox=provider,probe_worker=probe)
    assert result['workers'][0]['method']=='provider-completed-cohost'
    worker['boot_id']='other-boot'; path.write_text(json.dumps({'run_id':'run','coordinator':primary,'workers':{'0':worker}}))
    proof['journal']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): verify_provider_fence(proof,expected_primary=primary,expected_workers=[0],get_sandbox=provider,probe_worker=probe)


def test_running_sandbox_restart_requires_owner_death_and_all_workers(tmp_path):
    from scripts.coreweave.diloco_replica_fence import verify_owned_restart_fence
    primary={'sandbox_id':'s','boot_id':'b','owner_id':'o','pid':8,'start_ticks':'9'}
    worker={'sandbox_id':'w','boot_id':'c','owner_id':'p','pid':10,'start_ticks':'11'}
    path=tmp_path/'owner.json';path.write_text(json.dumps({'run_id':'r','coordinator':primary,'workers':{'0':worker}}))
    proof={'journal':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'expected_run_id':'r'}
    stopped=lambda identity,nonce:{'identity':identity,'nonce':nonce,'process_absent':True,'port_closed':True}
    peer=lambda identity,nonce:{'identity':identity,'nonce':nonce,'process_absent':True,'owner_quiesced':True}
    result=verify_owned_restart_fence(proof,expected_primary=primary,expected_workers=[0],nonce='n'*48,observe_primary=stopped,probe_worker=peer)
    assert result['primary_fenced'] and result['schema']=='natlang.diloco-owner-restart-fence/1'
    with pytest.raises(ValueError):verify_owned_restart_fence(proof,expected_primary=primary,expected_workers=[0],nonce='n'*48,observe_primary=lambda i,n:dict(stopped(i,n),port_closed=False),probe_worker=peer)
    with pytest.raises(ValueError):verify_owned_restart_fence(proof,expected_primary=primary,expected_workers=[0],nonce='n'*48,observe_primary=stopped,probe_worker=lambda i,n:dict(peer(i,n),owner_quiesced=False))


def test_real_owned_stopped_coordinator_and_closed_port(tmp_path):
    import os,socket,subprocess,sys,time
    from scripts.coreweave.diloco_nested_owner import prepare
    from scripts.coreweave.diloco_promotion_authority import fence
    from scripts.coreweave.diloco_replica_fence import observe_coordinator_stop
    script=Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'
    spec=prepare(source=script,expected_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),python=sys.executable,control_directory=str(tmp_path/'control'),runtime=str(Path(__file__).parents[1]))
    proc=subprocess.Popen(spec['argv'])
    try:
        statepath=tmp_path/'control'/'state.json';end=time.monotonic()+8
        while not statepath.exists() and time.monotonic()<end:time.sleep(.02)
        state=json.loads(statepath.read_text());identity={'sandbox_id':'local','boot_id':state['boot_id'],'owner_id':state['owner_id'],'pid':999999999,'start_ticks':'1'}
        nonce='z'*48;fence(statepath.parent,expected_owner=identity['owner_id'],expected_boot=identity['boot_id'],nonce=nonce)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        result=observe_coordinator_stop(identity,nonce,statepath.parent,port,spec['supervisor_cmdline_sha256'],spec['entrypoint_sha256'])
        assert result['process_absent'] and result['port_closed']
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port));sock.listen()
            with pytest.raises(ValueError):observe_coordinator_stop(identity,nonce,statepath.parent,port,spec['supervisor_cmdline_sha256'],spec['entrypoint_sha256'])
    finally:proc.terminate();proc.wait(timeout=8)
