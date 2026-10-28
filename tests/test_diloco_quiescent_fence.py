import socket,threading,time
import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoSettings,DilocoLearner,Transport


def test_four_unequal_workers_quiesce_88_fragments_without_quorum_or_lost_state(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(88*256))
        def forward(self):return self.weight.sum()
    seed=Model();root=tmp_path/'coordinator';token='x'*32
    manifest=initialize_coordinator(root,run_id='fence',masters=TensorSurface.from_module(seed).masters,
        identity={'parent':'fence'},workers=4,min_quorum=3,fragment_bytes=1024,grace_seconds=0)
    assert len(manifest['fragments'])==88
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';stop=threading.Event()
    server=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);server.start()
    transport=Transport(url,token,10)
    for _ in range(100):
        try:transport.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    learners=[]
    try:
        for worker in range(4):
            model=Model();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            settings=DilocoSettings(run_id='fence',workers=4,worker_id=worker,coordinator_url=url,local_steps=128,wire_dtype='fp32',heartbeat_seconds=.05)
            learner=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity={'parent':'fence'},output=tmp_path/f'w{worker}',transport=transport)
            learner.start();learners.append((model,optimizer,learner))
        for worker,(model,optimizer,learner) in enumerate(learners):
            for _ in range(worker+1):
                optimizer.zero_grad();model().backward();optimizer.step();learner.after_step(tokens=worker+1)
        # One submitted window cannot reach quorum; the other 87 remain unsent.
        learners[0][2]._queue(0)
        for _,_,learner in learners:assert learner.quiesce_for_checkpoint(timeout_seconds=5)
        snapshots=[x[2].state_dict() for x in learners]
        assert snapshots[0]['pending']['0']['steps']==1 and snapshots[0]['pending_payloads']['0'].numel()==256
        assert snapshots[0]['steps_since'][1:]==[1]*87
        for i,s in enumerate(snapshots[1:],1):
            assert s['local_steps']==i+1 and s['steps_since']==[i+1]*88
        assert all(v==0 for v in transport.json('GET','/v1/status')['versions'])
        # The pending payload and optimizer/RNG/exposure remain serializable for replay.
        path=tmp_path/'checkpoint.pt';rng=torch.get_rng_state().clone();optimizer=learners[0][1]
        torch.save({'diloco':snapshots[0],'model':learners[0][0].state_dict(),'optimizer':optimizer.state_dict(),'rng':rng,'exposure':{'row':1}},path)
        saved=torch.load(path,weights_only=True);assert saved['exposure']=={'row':1} and torch.equal(saved['rng'],rng)
        assert saved['diloco']['pending_payloads']['0'].equal(snapshots[0]['pending_payloads']['0'])
        learner=learners[0][2];learner.close()
        model=Model();model.load_state_dict(saved['model']);restored_optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        restored_optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
        restored=DilocoLearner(learner.settings,surface=TensorSurface.from_module(model),identity={'parent':'fence'},output=tmp_path/'restored',transport=transport)
        restored.resume(saved['diloco']);learners[0]=(model,restored_optimizer,restored)
        restored_optimizer.zero_grad();model().backward();restored_optimizer.step();restored.after_step(tokens=1)
        assert restored.local_steps==2 and restored.local_tokens==2 and restored_optimizer.state
    finally:
        for _,_,learner in learners:learner.close()
        stop.set();server.join(timeout=3)
