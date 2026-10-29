import json,multiprocessing as mp,os,socket,subprocess,sys,time
from pathlib import Path
import torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.replica_baseline import capture
from natlang_neuralese.train.diloco.replica_service import serve_peer
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator
from natlang_neuralese.train.diloco.replica_transport import witness_head


def test_cpu_runner_uses_four_real_proposals_and_replay(tmp_path):
    root=tmp_path/'root';initialize_coordinator(root,run_id='physical-qualification',masters={'w':torch.ones(16)},identity={'purpose':'isolated-physical-qualification'},workers=4,min_quorum=3,fragment_bytes=1024,grace_seconds=30,ema={'decay':.88,'warmup':0})
    machine=ReplayCoordinator(root,quorum_since={})
    class Pin:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Pin(),baseline);token='s'*32;tokenfile=tmp_path/'token';tokenfile.write_text(token)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event()
    def peer():serve_peer(baseline=baseline,baseline_sha256=receipt['sha256'],store=tmp_path/'peer',work_root=tmp_path/'peer-work',host='127.0.0.1',port=port,token=token,max_transaction_bytes=1<<20,stop=stop)
    proc=ctx.Process(target=peer);proc.start();url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:witness_head(url,token,allow_http=True);break
            except OSError:time.sleep(.03)
        script=Path(__file__).parents[1]/'scripts/coreweave/diloco_replica_physical_bench.py'
        result=subprocess.run([sys.executable,str(script),'measure','--out',str(tmp_path/'measure'),'--baseline',str(baseline),'--baseline-sha256',receipt['sha256'],'--peer-url',url,'--token-file',str(tokenfile),'--allow-http','--workers','4','--quorum','3','--minimum-free-bytes','0'],capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        summary=json.loads((tmp_path/'measure'/'qualification.json').read_text())
        assert summary['workers']==4 and summary['quorum']==3 and summary['fragments_completed']==1 and len(summary['update_ack_seconds'])==4
    finally:stop.set();proc.join(8)
    assert proc.exitcode==0
