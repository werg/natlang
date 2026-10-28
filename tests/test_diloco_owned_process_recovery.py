"""Exercise real process death and coordinator restart with a tiny DiLoCo surface."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from multiprocessing.connection import Listener, Client

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/coreweave"))
import diloco_role_control as control
import multiprocessing as mp
import socket
import time

import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator, _metadata
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport

TOKEN='test-token-'+'x'*32


def _worker(conn, worker, url, directory, restored):
    torch.set_num_threads(1)
    torch.manual_seed(100+worker)
    model=torch.nn.Linear(256,2,bias=False)
    with torch.no_grad():model.weight.fill_(1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    settings=DilocoSettings(run_id='process-recovery',workers=4,worker_id=worker,coordinator_url=url,
                            local_steps=128,wire_dtype='fp32',heartbeat_seconds=.5,timeout_seconds=20)
    learner=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity={'parent':'fixture'},output=directory,
                          transport=Transport(url,TOKEN,20))
    exposure=0
    if restored:
        saved=torch.load(restored,weights_only=True)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);exposure=saved['exposure']
        learner.resume(saved['diloco'])
        state=next(iter(optimizer.state.values())); prior=next(iter(saved['optimizer']['state'].values()))
        exact=torch.equal(torch.get_rng_state(),saved['rng']) and all(torch.equal(state[k],prior[k]) for k in state)
        conn.send(('restored', exposure, learner.local_steps, int(state['step']),exact))
    else:
        learner.start();conn.send(('started',))
    try:
        while True:
            command=conn.recv()
            if command=='step':
                optimizer.zero_grad();(model.weight * torch.rand_like(model.weight)).sum().backward();optimizer.step()
                exposure+=1;learner.after_step(tokens=3);conn.send(('step',exposure,learner.local_steps))
            elif command in ('queue0','queue1'):
                learner._queue(int(command[-1]));assert learner.quiesce_for_checkpoint(timeout_seconds=25);conn.send(('queued',))
            elif command=='settle':
                assert learner.quiesce_for_checkpoint(timeout_seconds=25)
                assert learner._collect_uploads(learner.local_steps)==0 and not learner.pending
                conn.send(('settled',))
            elif command=='checkpoint':
                assert learner.quiesce_for_checkpoint(timeout_seconds=25)
                saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
                       'exposure':exposure,'diloco':learner.state_dict()}
                path=directory/'checkpoint.pt';torch.save(saved,path)
                conn.send(('checkpoint',str(path)))
            elif command=='close':
                learner.close();conn.send(('closed',));return
    finally:conn.close()


def _call(conn, command):
    conn.send(command)
    assert conn.poll(40), command
    return conn.recv()


def _wait_status(transport, predicate):
    for _ in range(200):
        try:
            status=transport.json('GET','/v1/status')
            if predicate(status):return status
        except (OSError,TimeoutError):pass
        time.sleep(.05)
    raise AssertionError('coordinator did not reach expected status')


def test_four_real_processes_recover_pending_worker_and_restart_coordinator(tmp_path):
    torch.set_num_threads(1)
    root=tmp_path/'coordinator';model=torch.nn.Linear(256,2,bias=False)
    with torch.no_grad():model.weight.fill_(1)
    manifest=initialize_coordinator(root,run_id='process-recovery',masters=TensorSurface.from_module(model).masters,
        identity={'parent':'fixture'},workers=4,min_quorum=3,fragment_bytes=1024,grace_seconds=0)
    assert len(manifest['fragments'])==2
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';transport=Transport(url,TOKEN,3);ctx=mp.get_context('fork')
    def coordinator():
        p=ctx.Process(target=serve_coordinator,args=(root,'127.0.0.1',port,TOKEN));p.start()
        _wait_status(transport,lambda s:True);return p
    server=coordinator();children=[];owners={}
    class Owned:
        def __init__(self,pid):self.pid=pid;self.exitcode=None
        def is_alive(self):return Path(f'/proc/{self.pid}').exists()
        def kill(self):os.kill(self.pid,signal.SIGKILL)
        def join(self,timeout=5):
            deadline=time.monotonic()+timeout
            while self.is_alive() and time.monotonic()<deadline:time.sleep(.05)
            if not self.is_alive():self.exitcode=-signal.SIGKILL
    def start(worker,restored=None):
        directory=tmp_path/f'worker-{worker}'
        if worker not in owners:
            control_dir=tmp_path/f'owner-{worker}'
            owner=subprocess.Popen([sys.executable,str(Path(__file__).parents[1]/'scripts/coreweave/diloco_role_supervisor.py'),str(control_dir)])
            deadline=time.monotonic()+10
            while not (control_dir/'state.json').exists() and time.monotonic()<deadline:time.sleep(.05)
            owners[worker]=(owner,control_dir)
        owner,control_dir=owners[worker];state=json.loads((control_dir/'state.json').read_text())
        listener=Listener(('127.0.0.1',0));listener._listener._socket.settimeout(20)
        argv=[sys.executable,str(Path(__file__)),'--worker-child',str(worker),url,str(directory),str(restored or '-'),str(listener.address[1])]
        request={'schema':'natlang.diloco-role-request/1','id':f'worker-{worker}-{state["generation"]}',
                 'action':'start','role':'worker','owner_id':state['owner_id'],'expected_generation':state['generation'],'argv':argv}
        receipt=control.submit(control_dir,request,timeout=10);p=Owned(receipt['pid'])
        try:parent=listener.accept()
        finally:listener.close()
        assert parent.poll(20);result=parent.recv();children.append((p,parent));return p,parent,result
    try:
        for i in range(4):assert start(i)[2]==('started',)
        for _,conn in children:assert _call(conn,'step')[1:]==(1,1)
        assert _call(children[0][1],'queue0')==('queued',)
        path=_call(children[0][1],'checkpoint')[1]
        saved=torch.load(path,weights_only=True)
        assert saved['diloco']['pending']['0']['sequence']==1
        assert saved['diloco']['steps_since'][1]==1 and saved['exposure']==1
        victim,conn=children[0];victim.kill();victim.join(5);assert victim.exitcode is not None;conn.close()
        for _,peer in children[1:]:
            assert _call(peer,'step')[1:]==(2,2);_call(peer,'queue0')
        _wait_status(transport,lambda s:s['versions'][0]>=1)
        ledger=json.loads(_metadata(root/'fragments'/'00000.safetensors')['last_seq'])
        assert ledger['0']==1 and len(ledger)>=3
        # Restart the actual coordinator process while the remaining learners live.
        server.kill();server.join(5);assert all(p.is_alive() for p,_ in children[1:])
        server=coordinator()
        assert json.loads(_metadata(root/'fragments'/'00000.safetensors')['last_seq'])==ledger
        p,peer,result=start(0,path)
        assert result==('restored',1,1,1,True)
        assert _call(peer,'step')[1:]==(2,2)
        assert _call(peer,'settle')==('settled',)
        after=_wait_status(transport,lambda s:len(s['worker_progress'])==4 and
                           all(w['tokens']==6 for w in s['worker_progress'].values()))
        assert json.loads(_metadata(root/'fragments'/'00000.safetensors')['last_seq'])['0']==1
        assert after['global_tokens']==24
        _call(peer,'queue1')
        for _,other in children[1:3]:_call(other,'queue1')
        _wait_status(transport,lambda s:s['versions'][1]>=1)
        second=json.loads(_metadata(root/'fragments'/'00001.safetensors')['last_seq'])
        assert second['0']==1 and len(second)>=3
    finally:
        for p,conn in children:
            if p.is_alive():
                try:_call(conn,'close')
                except (EOFError,OSError,AssertionError):p.kill()
            p.join(5)
            if p.is_alive():p.kill();p.join()
        if server.is_alive():server.kill();server.join()
        for owner,_ in owners.values():
            owner.terminate();owner.wait(timeout=10)


if __name__=='__main__' and len(sys.argv)==7 and sys.argv[1]=='--worker-child':
    _,_,worker,url,directory,restored,port=sys.argv
    _worker(Client(('127.0.0.1',int(port))),int(worker),url,Path(directory),None if restored=='-' else restored)
