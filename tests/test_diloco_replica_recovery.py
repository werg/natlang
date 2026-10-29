import hashlib,json,shutil
import torch
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.diloco.coordinator import FileStore,initialize_coordinator
from natlang_neuralese.train.diloco.protocol import save_tensors
from natlang_neuralese.train.diloco.replica_baseline import capture
from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
from natlang_neuralese.train.diloco.replica_recovery import capture_recovery,restore_recovery,finish_recovery,prepare_restored_peer_seed
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator


def test_recovery_closure_replays_and_continues_four_worker_state(tmp_path,monkeypatch):
    from natlang_neuralese.train.diloco import replica_executor
    root=tmp_path/'root';initialize_coordinator(root,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=4,min_quorum=4,fragment_bytes=1024,grace_seconds=30)
    machine=ReplayCoordinator(root,quorum_since={})
    class Pin:
        def __init__(self,m):self.machine=m
        def read(self,fn):return fn()
    store=tmp_path/'store';store.mkdir();baseline=store/'baseline';baseline_receipt=capture(Pin(machine),baseline)
    local=ReplicaJournal(store/'journal',run_identity='r',baseline_sha256=baseline_receipt['sha256']);remote=ReplicaJournal(tmp_path/'remote',run_identity='r',baseline_sha256=baseline_receipt['sha256'])
    monkeypatch.setattr(replica_executor,'transmit',lambda _u,_t,**kw:remote.append(kw['event'],kw['payload'],expected_sequence=kw['sequence'],expected_previous=kw['previous']))
    empty=tmp_path/'empty';empty.write_bytes(b'');writer=ReplicatedExecutor(local,'http://unused','s'*32,allow_http=True);dispatch=ReplicatedDispatch(machine,writer,empty)
    sessions=['00000000-0000-0000-0000-%012d'%(i+1) for i in range(4)]
    for i,session in enumerate(sessions):dispatch.call('lease',{'worker':i,'session':session})
    delta=tmp_path/'delta';save_tensors(delta,{'delta':torch.ones(4)})
    dispatch.call('submit',{'index':0,'worker':0,'session':sessions[0],'sequence':1,'base_version':0,'steps':1,'tokens':3},delta)
    write_json_durable(store/'primary-active.json',{'baseline':str(baseline),'baseline_sha256':baseline_receipt['sha256'],'journal':str(local.root)})
    upload=root/'uploads';upload.mkdir();(upload/'source').write_bytes(b'full-source');(upload/'source.sha256').write_text(hashlib.sha256(b'full-source').hexdigest()+'\n')
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    fingerprint=machine.coordinator.manifest['fingerprint'];workers={};worker_paths={};components={};owner={'phase':'complete','run_id':'r','workers':{}}
    for i in range(4):
        seq=[1 if i==0 else 0];versions=[0];dl={'worker':i,'fingerprint':fingerprint,'versions':versions,'sequences':seq}
        path=tmp_path/f'worker-{i}.pt';torch.save({'schema':CHECKPOINT_SCHEMA,'training':{'schema':TRAINING_SCHEMA,'optimizer':{'step':i},'rng':{'seed':i+1},'exposure_counts':{'source':i+1},'source_receipts':[{'sha256':'s'}],'source_hash':'source','options':{'recipe_binding':{},'ce_supervision_policy_binding':{}},'global_step':i,'diloco':dl}},path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest();workers[str(i)]={'id':'fence1','worker':i,'checkpoint':f'/worker-{i}/pin.pt','fingerprint':fingerprint,'versions':versions,'sequences':seq,'deadline':__import__('time').time()+60,'global_step':i};worker_paths[str(i)]=path;owner['workers'][str(i)]={'sandbox_id':f's{i}','pid':i+1,'boot_id':'boot','owner_id':'owner','start_ticks':1}
        components[str(i)]={'source':workers[str(i)]['checkpoint'],'fingerprint':fingerprint,'versions':versions,'sequences':seq,'sha256':sha,'global_step':i,'source_hash':'source','source_receipt_sha256':['s'],'runtime_code_sha256':baseline_receipt['value']['runtime_code_sha256'],'recipe_binding':{},'ce_supervision_policy_binding':{}}
    owner_path=tmp_path/'owner.json';owner_path.write_text(json.dumps(owner));owner_sha=hashlib.sha256(owner_path.read_bytes()).hexdigest()
    closure=capture_recovery(dispatch,FileStore(root),store,tmp_path/'closure',fence_id='fence1',workers=workers,owner_journal_sha256=owner_sha)
    # A dead pinning process can finish from its durable immutable pin record.
    (tmp_path/'closure'/'recovery-closure.json').unlink()
    (tmp_path/'closure'/'current'/'replica-baseline.json').unlink()
    closure=finish_recovery(tmp_path/'closure')
    from scripts.coreweave.diloco_recovery_pin import archive_replicated_closure
    from scripts.coreweave.diloco_stage_recovery import _extract
    archive=tmp_path/'closure.tar'
    exported=archive_replicated_closure(tmp_path/'closure',archive,closure['sha256'])
    assert archive_replicated_closure(tmp_path/'closure',archive,closure['sha256'])==exported
    _extract(archive,tmp_path/'download',prefix='diloco-replica-recovery')
    restored=restore_recovery(tmp_path/'download'/'diloco-replica-recovery',tmp_path/'restore',receipt_sha256=closure['sha256'],workers=workers,worker_components=components,owner_journal_sha256=owner_sha,owner_journal_path=owner_path,worker_paths=worker_paths)
    restored['journal'].close()
    seed=prepare_restored_peer_seed(closure=tmp_path/'download'/'diloco-replica-recovery',restored=tmp_path/'restore',destination=tmp_path/'seed-store',seed=tmp_path/'seed',receipt_sha256=closure['sha256'],freeze_nonce='a'*32)
    from natlang_neuralese.train.diloco.replica_seed import install_seed
    installed=install_seed(tmp_path/'seed',tmp_path/'seed-peer',expected_receipt_sha256=seed['receipt_sha256'])
    assert installed['head']==seed['head'] and (tmp_path/'seed-peer'/'replica-uploads'/'source').read_bytes()==b'full-source'
    restored['journal']=ReplicaJournal(tmp_path/'restore'/'journal',run_identity='r',baseline_sha256=closure['value']['current_baseline_sha256'],previous_generation={'sequence':closure['value']['lineage']['sequence'],'head':closure['value']['lineage']['head']})
    assert (restored['coordinator']/'uploads'/'source').read_bytes()==b'full-source'
    assert (restored['coordinator']/'pending'/'00000'/'000.safetensors').is_file()
    for i in range(4):assert torch.load(worker_paths[str(i)],weights_only=False)['training']['optimizer']['step']==i
    assert restored['journal'].identity['previous_generation']['head']==local.head
    current=closure['value'];from natlang_neuralese.train.diloco.replica_baseline import verify
    current_baseline=verify(tmp_path/'closure'/'current',current['current_baseline_sha256'])
    resumed=ReplayCoordinator(restored['coordinator'],quorum_since={int(k):v for k,v in current_baseline['quorum_since'].items()},clock_anchor={'wall':current_baseline['captured_wall'],'monotonic':current_baseline['captured_monotonic']})
    new_peer=ReplicaJournal(tmp_path/'new-peer',run_identity='r',baseline_sha256=current['current_baseline_sha256'],previous_generation={'sequence':current['lineage']['sequence'],'head':current['lineage']['head']})
    monkeypatch.setattr(replica_executor,'transmit',lambda _u,_t,**kw:new_peer.append(kw['event'],kw['payload'],expected_sequence=kw['sequence'],expected_previous=kw['previous']))
    new_writer=ReplicatedExecutor(restored['journal'],'http://unused','s'*32,allow_http=True);new_dispatch=ReplicatedDispatch(resumed,new_writer,empty)
    new_dispatch.call('clock_rebase',{'old_wall':current_baseline['captured_wall'],'old_monotonic':current_baseline['captured_monotonic'],'old_timers':current_baseline['quorum_since']})
    for i in range(1,4):new_dispatch.call('submit',{'index':0,'worker':i,'session':sessions[i],'sequence':1,'base_version':0,'steps':1,'tokens':3},delta)
    assert resumed.coordinator._version(0)==1
    new_writer.close();new_peer.close();restored['journal'].close();writer.close();local.close();remote.close()
