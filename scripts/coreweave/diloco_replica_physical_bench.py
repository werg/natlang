"""CPU-only isolated replica qualification; never opens a learner's writable state."""
import argparse,json,os,resource,shutil,time,uuid
from pathlib import Path
import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.protocol import load_tensors,save_tensors
from natlang_neuralese.train.diloco.replica_baseline import capture,reconstruct,verify
from natlang_neuralese.train.diloco.replica_generation import rotate
from natlang_neuralese.train.diloco.replica_retention import Retention
from natlang_neuralese.train.diloco.replica_transport import generation_request
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.diloco.replica_dispatch import ReplicatedDispatch
from natlang_neuralese.train.diloco.replica_executor import ReplicatedExecutor
from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator,replay
from natlang_neuralese.train.native_diloco import _checkpoint_surface


def sample():
    root=Path('/sys/fs/cgroup');result={}
    for name in ('memory.current','memory.peak','memory.max','cpu.stat','io.stat'):
        try:result[name]=(root/name).read_text().strip()
        except OSError:result[name]=None
    usage=resource.getrusage(resource.RUSAGE_SELF);result.update(cpu_user=usage.ru_utime,cpu_system=usage.ru_stime,maxrss=usage.ru_maxrss)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','measure'));p.add_argument('--out',required=True)
    p.add_argument('--checkpoint');p.add_argument('--base');p.add_argument('--full',action='store_true');p.add_argument('--minimum-free-bytes',type=int,default=40<<30)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--quorum',type=int,default=1);p.add_argument('--all-fragments',action='store_true');p.add_argument('--heartbeat-seconds',type=float,default=90)
    p.add_argument('--baseline');p.add_argument('--baseline-sha256');p.add_argument('--peer-url');p.add_argument('--token-file');p.add_argument('--gateway-file');p.add_argument('--allow-http',action='store_true')
    a=p.parse_args();
    if not 1<=a.quorum<=a.workers or not 0<a.heartbeat_seconds<=120:raise ValueError('invalid worker quorum or heartbeat interval')
    torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);before=sample();started=time.monotonic()
    if a.mode=='prepare':
        if not a.checkpoint or not a.base:raise ValueError('actual checkpoint and exact base are required')
        if shutil.disk_usage(out).free<a.minimum_free_bytes:raise ValueError('qualification scratch reserve is insufficient')
        surface=_checkpoint_surface(a.checkpoint,a.base)
        if a.full:masters=surface
        else:
            name,tensor=next((name,tensor) for name,tensor in surface.items() if tensor.numel()>=16_777_216)
            masters={name:tensor.reshape(-1)[:16_777_216]}
        root=out/'coordinator';manifest=initialize_coordinator(root,run_id='physical-qualification',masters=masters,identity={'purpose':'isolated-physical-qualification'},workers=a.workers,min_quorum=a.quorum,fragment_bytes=64<<20,grace_seconds=30,ema={'decay':.88,'warmup':0})
        machine=ReplayCoordinator(root,quorum_since={})
        class Pin:
            def __init__(self):self.machine=machine
            def read(self,fn):return fn()
        receipt=capture(Pin(),out/'baseline');result={'mode':'prepare','full':a.full,'workers':a.workers,'quorum':a.quorum,'fragments':len(manifest['fragments']),'fingerprint':manifest['fingerprint'],'baseline_sha256':receipt['sha256']}
    else:
        if not all((a.baseline,a.baseline_sha256,a.peer_url,a.token_file)):raise ValueError('baseline, exact digest, peer and token required')
        active_baseline=Path(a.baseline);active_sha=a.baseline_sha256
        value=reconstruct(active_baseline,out/'coordinator',expected_sha256=active_sha)
        machine=ReplayCoordinator(out/'coordinator',quorum_since={int(k):v for k,v in value['quorum_since'].items()},clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
        if machine.coordinator.manifest['workers']!=a.workers or machine.coordinator.manifest['min_quorum']!=a.quorum:
            raise ValueError('prepared baseline worker/quorum differs')
        store=out/'store';store.mkdir();j=ReplicaJournal(store/'journal0',run_identity=value['run_id'],baseline_sha256=active_sha)
        token=Path(a.token_file).read_text().strip();gateway=Path(a.gateway_file).read_text().strip() if a.gateway_file else None
        writer=ReplicatedExecutor(j,a.peer_url,token,allow_http=a.allow_http,gateway=gateway,timeout=300);empty=out/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(machine,writer,empty)
        rotations=[]
        def rollover():
            nonlocal active_baseline,active_sha
            ident=uuid.uuid4().hex;directory=store/('generation-'+ident)
            def prepare(previous,clock,wall):return generation_request(a.peer_url,token,'prepare',{'id':ident,'previous':previous,'captured_monotonic':clock,'captured_wall':wall},allow_http=a.allow_http,gateway=gateway)
            def commit(intent):return generation_request(a.peer_url,token,'commit',{'id':ident,'previous':intent['previous'],'baseline_sha256':intent['baseline_sha256'],'new_head':intent['new_head']},allow_http=a.allow_http,gateway=gateway)
            receipt=rotate(dispatch,local_baseline_dir=directory/'baseline',local_journal_dir=directory/'journal',receipt_path=directory/'generation.json',peer_prepare=prepare,peer_commit=commit,within_writer=True)
            pointer=store/'primary-active.json';write_json_durable(pointer,{'id':ident,'baseline_sha256':receipt['baseline_sha256'],'head':receipt['new_head'],'previous':receipt['previous']})
            confirmation=generation_request(a.peer_url,token,'retain',{'id':ident,'previous':receipt['previous'],'baseline_sha256':receipt['baseline_sha256'],'new_head':receipt['new_head']},allow_http=a.allow_http,gateway=gateway)
            retention=Retention(store);retention.register(generation=ident,baseline_sha256=receipt['baseline_sha256'],head=receipt['new_head'],previous=retention.latest(),replica_confirmation=confirmation,active_pointer=pointer);retention.collect()
            active_baseline=directory/'baseline';active_sha=receipt['baseline_sha256'];rotations.append(receipt['new_head'])
        writer.rollover_callback=rollover
        sessions=[str(uuid.UUID(int=i+1)) for i in range(a.workers)];lease_seconds=[];heartbeats=[];last_heartbeat={}
        for worker,session in enumerate(sessions):
            t=time.monotonic();dispatch.call('lease',{'worker':worker,'session':session});lease_seconds.append(time.monotonic()-t);last_heartbeat[worker]=time.monotonic()
        count=len(machine.coordinator.fragments) if a.all_fragments else 1;acks=[];proposal_states=[];completed=0
        def renew_due():
            for worker,session in enumerate(sessions):
                if time.monotonic()-last_heartbeat[worker]>=a.heartbeat_seconds:
                    t=time.monotonic();dispatch.call('heartbeat',{'worker':worker,'session':session,'steps':1,'tokens':1,'metrics':None})
                    elapsed=time.monotonic()-t
                    if elapsed>300:raise ValueError('heartbeat exceeded safe serialized operation bound')
                    last_heartbeat[worker]=time.monotonic();heartbeats.append({'worker':worker,'elapsed_seconds':elapsed})
        for index in range(count):
            if shutil.disk_usage(out).free<a.minimum_free_bytes:raise ValueError('qualification stopped at scratch reserve')
            theta=load_tensors(out/'coordinator'/'fragments'/f'{index:05d}.safetensors')['theta'];delta=out/'delta.safetensors';save_tensors(delta,{'delta':torch.zeros_like(theta)})
            for worker,session in enumerate(sessions):
                renew_due();t=time.monotonic();dispatch.call('submit',{'index':index,'worker':worker,'session':session,'sequence':1,'base_version':0,'steps':1,'tokens':1},delta);elapsed=time.monotonic()-t;acks.append(elapsed)
                if elapsed>300:raise ValueError('proposal exceeded safe serialized operation bound')
                proposal_states.append({'fragment':index,'worker':worker,'pending':len(list(machine.coordinator._pending_dir(index).glob('*.safetensors'))),'version':machine.coordinator._version(index)})
            if proposal_states[-1]['version']!=1 or proposal_states[-1]['pending']!=0:raise ValueError('fragment did not complete its quorum merge')
            completed+=1
        writer.close();j=writer.journal;t=time.monotonic();value=verify(active_baseline,active_sha);reconstruct(active_baseline,out/'replay',expected_sha256=active_sha)
        recovered=replay(j,out/'replay',expected_sequence=j.sequence,expected_head=j.head,quorum_since={int(k):v for k,v in value['quorum_since'].items()},clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
        if recovered.coordinator.progress!=machine.coordinator.progress:raise ValueError('replayed progress differs')
        replay_seconds=time.monotonic()-t;result={'mode':'measure','workers':a.workers,'quorum':a.quorum,'all_fragments_requested':a.all_fragments,'fragments_completed':completed,'delta_bytes_last':delta.stat().st_size,'lease_ack_seconds':lease_seconds,'update_ack_seconds':acks,'proposal_states':proposal_states,'heartbeats':heartbeats,'replay_seconds':replay_seconds,'rollovers':rotations,'sequence':j.sequence,'head':j.head};j.close()
    result.update(wall_seconds=time.monotonic()-started,before=before,after=sample());(out/'qualification.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
