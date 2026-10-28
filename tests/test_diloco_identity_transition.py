import copy,json,socket,threading,time
import pytest,torch
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoSettings,DilocoLearner,Transport
from natlang_neuralese.train.native_diloco import source_identity_transition


def test_authenticated_source_identity_handoff_keeps_original_surface(tmp_path):
    model=torch.nn.Linear(2,2);root=tmp_path/'coord';surface=TensorSurface.from_module(model)
    original={'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':'a'*64,'trainer_argv':['--native-graphs','/old','--epochs','1'],'diloco':{'workers':1}}
    current={**original,'trainer_argv':['--native-graphs','/new','--source-correction','/proof','--epochs','1']}
    proof={'schema':'natlang.diloco-source-identity-transition/1','correction_verified':True,'checkpoint_sha256':'b'*64,
           'correction_sha256':'c'*64,'previous_source_hash':'d'*64,'source_hash':'e'*64}
    transition=source_identity_transition(original,current,proof)
    bad=copy.deepcopy(current);bad['trainer_argv'][-1]='2'
    with pytest.raises(ValueError,match='fixed trainer'):source_identity_transition(original,bad,proof)
    with pytest.raises(ValueError):source_identity_transition(original,current,{**proof,'correction_verified':False})
    manifest=initialize_coordinator(root,run_id='identity',masters=surface.masters,identity=original,workers=1,min_quorum=1,fragment_bytes=1024,grace_seconds=0)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    token='x'*32;url=f'http://127.0.0.1:{port}';stop=threading.Event();server=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);server.start()
    transport=Transport(url,token,5)
    for _ in range(100):
        try:transport.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    settings=DilocoSettings(run_id='identity',workers=1,worker_id=0,coordinator_url=url,heartbeat_seconds=.05)
    learner=DilocoLearner(settings,surface=surface,identity=original,output=tmp_path/'worker',transport=transport)
    try:
        learner.start();learner.record_identity_transition(transition);learner.record_identity_transition(transition)
        progress=json.loads((root/'progress.json').read_text());record=progress['identity_transitions']['0']
        assert len(record)==1 and record[0]['current_identity']==current and record[0]['checkpoint_sha256']=='b'*64
        assert learner.manifest['identity']==original and learner.manifest['fingerprint']==manifest['fingerprint']
    finally:learner.close();stop.set();server.join(timeout=3)
