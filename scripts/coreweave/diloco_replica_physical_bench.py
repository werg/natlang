"""CPU-only isolated replica qualification; never opens a learner's writable state."""
import argparse,json,os,resource,shutil,time
from pathlib import Path
import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.protocol import load_tensors,save_tensors
from natlang_neuralese.train.diloco.replica_baseline import capture,reconstruct
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
    p.add_argument('--baseline');p.add_argument('--baseline-sha256');p.add_argument('--peer-url');p.add_argument('--token-file');p.add_argument('--gateway-file');p.add_argument('--allow-http',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);before=sample();started=time.monotonic()
    if a.mode=='prepare':
        if not a.checkpoint or not a.base:raise ValueError('actual checkpoint and exact base are required')
        if shutil.disk_usage(out).free<a.minimum_free_bytes:raise ValueError('qualification scratch reserve is insufficient')
        surface=_checkpoint_surface(a.checkpoint,a.base)
        if a.full:masters=surface
        else:
            name,tensor=next((name,tensor) for name,tensor in surface.items() if tensor.numel()>=16_777_216)
            masters={name:tensor.reshape(-1)[:16_777_216]}
        root=out/'coordinator';manifest=initialize_coordinator(root,run_id='physical-qualification',masters=masters,identity={'purpose':'isolated-physical-qualification'},workers=1,min_quorum=1,fragment_bytes=64<<20,ema={'decay':.88,'warmup':0})
        machine=ReplayCoordinator(root,quorum_since={})
        class Pin:
            def __init__(self):self.machine=machine
            def read(self,fn):return fn()
        receipt=capture(Pin(),out/'baseline');result={'mode':'prepare','full':a.full,'fragments':len(manifest['fragments']),'fingerprint':manifest['fingerprint'],'baseline_sha256':receipt['sha256']}
    else:
        if not all((a.baseline,a.baseline_sha256,a.peer_url,a.token_file)):raise ValueError('baseline, exact digest, peer and token required')
        value=reconstruct(a.baseline,out/'coordinator',expected_sha256=a.baseline_sha256)
        j=ReplicaJournal(out/'journal',run_identity=value['run_id'],baseline_sha256=a.baseline_sha256)
        machine=ReplayCoordinator(out/'coordinator',quorum_since={int(k):v for k,v in value['quorum_since'].items()},clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
        token=Path(a.token_file).read_text().strip();gateway=Path(a.gateway_file).read_text().strip() if a.gateway_file else None
        writer=ReplicatedExecutor(j,a.peer_url,token,allow_http=a.allow_http,gateway=gateway,timeout=300);empty=out/'empty';empty.write_bytes(b'');dispatch=ReplicatedDispatch(machine,writer,empty)
        session='00000000-0000-0000-0000-000000000001';t=time.monotonic();dispatch.call('lease',{'worker':0,'session':session});lease_seconds=time.monotonic()-t
        theta=load_tensors(out/'coordinator'/'fragments'/'00000.safetensors')['theta'];delta=out/'delta.safetensors';save_tensors(delta,{'delta':torch.zeros_like(theta)})
        t=time.monotonic();result=dispatch.call('submit',{'index':0,'worker':0,'session':session,'sequence':1,'base_version':0,'steps':1,'tokens':1},delta);ack_seconds=time.monotonic()-t
        writer.close();t=time.monotonic();reconstruct(a.baseline,out/'replay',expected_sha256=a.baseline_sha256)
        recovered=replay(j,out/'replay',expected_sequence=j.sequence,expected_head=j.head,quorum_since={int(k):v for k,v in value['quorum_since'].items()},clock_anchor={'wall':value['captured_wall'],'monotonic':value['captured_monotonic']})
        if recovered.coordinator.progress!=machine.coordinator.progress:raise ValueError('replayed progress differs')
        replay_seconds=time.monotonic()-t;result={'mode':'measure','delta_bytes':delta.stat().st_size,'lease_ack_seconds':lease_seconds,'update_ack_seconds':ack_seconds,'replay_seconds':replay_seconds,'sequence':j.sequence,'head':j.head,'result':result};j.close()
    result.update(wall_seconds=time.monotonic()-started,before=before,after=sample());(out/'qualification.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
