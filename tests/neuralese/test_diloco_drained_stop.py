"""An operator drain request stops a DiLoCo learner at a delivered boundary, so a
relaunch can register an exact supervised source append with a real coordinator."""
import json,socket,threading,time,uuid
import pytest,torch
from test_native_model import tiny_source
from test_native_state_sft import graph
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train import native_resident as resident,native_state_sft as trainer
from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint
from natlang_neuralese.train.diloco.worker import DilocoSettings,Transport
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator,TOKEN_ENV
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.native_diloco import identity


def rows(path,ids,fp,split='train'):
    path.write_text(''.join(json.dumps(dict(graph(i),split=split,tokenizer_fingerprint=fp))+'\n' for i in ids));return path


def test_drain_request_yields_registrable_source_append(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'));monkeypatch.setenv(TOKEN_ENV,'x'*32)
    base=tmp_path/'base';tiny_source(base);_,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4);fp=config['tokenizer_fingerprint']
    leaf=rows(tmp_path/'old.jsonl',['a','b','c','d','e','f'],fp);extra=rows(tmp_path/'extra.jsonl',['g'],fp);held=rows(tmp_path/'held.jsonl',['h'],fp,'test')
    old=tmp_path/'old.json';write_json_durable(old,{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**resident.binding(leaf)}]})
    new=tmp_path/'new.json';write_json_durable(new,{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**resident.binding(leaf)},{'kind':'native-graphs',**resident.binding(extra)}]})
    content=tmp_path/'content.json';write_json_durable(content,{'schema':'natlang.big2-multi-content-audit/1','base':{'sha256':resident.sha(old)},'candidates':[{'sha256':resident.sha(extra)}],'duplicates':[]})
    protect=tmp_path/'protection.json';write_json_durable(protect,{'schema':'natlang.fresh-restart-protection-union/1','sources':{'base':{'source':resident.binding(old),'conflicts':[]},'candidate-001':{'source':resident.binding(extra),'conflicts':[]}}})
    recipe=tmp_path/'recipe.json';write_json_durable(recipe,{'schema':'natlang.neuralese-training-recipe/1','id':'drain'})
    binding={**resident.binding(recipe),'file_sha256':resident.sha(recipe),'lineage_sha256':'1'*64,
             'external_provenances':[{'schema':'natlang.external-source-publication-provenance/1','closure':resident.binding(content),'content_audit':resident.binding(content),'clean_closure_audit':resident.binding(protect)}]}
    common=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--held-native-graphs',str(held),'--epochs','1','--batch-graphs','1','--accumulate-batches','1','--eval-every','100','--stage-id','sft','--recipe-binding',json.dumps(binding)]
    seedout=tmp_path/'seed';trainer.main(common+['--native-graphs',str(old),'--out',str(seedout),'--initialize-only'])
    seed=seedout/'checkpoint.pt';model,*_=load_native_checkpoint(seed,base=base,device='cpu')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    url=f'http://127.0.0.1:{port}';settings=DilocoSettings(run_id='drain',workers=1,worker_id=0,coordinator_url=url,allow_private_http=True,local_steps=4,wire_dtype='fp32',heartbeat_seconds=.1)
    out=tmp_path/'worker';argv=common+['--native-graphs',str(old),'--out',str(out),'--continue-from',str(seed),'--diloco',json.dumps(settings.__dict__)]
    role={'kind':'sft','identity':identity(seed,argv,settings),'dependencies':[resident.binding(old),resident.binding(leaf),resident.binding(held),resident.binding(recipe)]}
    authority={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':resident.sha(seed),'recipe_lineage_sha256':'1'*64,'diloco':settings.identity(),'roles':{'0':[role]}}
    root=tmp_path/'outer';initialize_coordinator(root,run_id='drain',masters=TensorSurface(dict(trainer.native_trainable_parameters(model))).masters,identity=authority,workers=1,min_quorum=1,grace_seconds=0,fragment_bytes=1<<28,seed_source={'sha256':resident.sha(seed)})
    stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,'x'*32),kwargs={'stop':stop},daemon=True);thread.start();transport=Transport(url,'x'*32,10,allow_http=True)
    for _ in range(100):
        try:manifest=transport.json('GET','/v1/manifest');break
        except OSError:time.sleep(.02)
    try:
        out.mkdir();request={'schema':trainer.DILOCO_DRAIN_SCHEMA,'id':uuid.uuid4().hex,'worker':0,'fingerprint':'0'*64}
        write_json_durable(out/trainer.DILOCO_DRAIN_REQUEST,request)
        with pytest.raises(ValueError,match='exact worker'):trainer.main(argv)
        request['fingerprint']=manifest['fingerprint'];write_json_durable(out/trainer.DILOCO_DRAIN_REQUEST,request)
        assert trainer.main(argv)==0
        state=torch.load(out/'checkpoint.pt',weights_only=False)['training'];saved=state['diloco']
        assert 0<state['global_step']<6 and sum(state['exposure_counts'].values())<6
        assert not saved['pending'] and not saved['awaiting_global'] and not any(saved['steps_since']) and not any(saved['tokens_since'])
        receipt=json.loads((out/f"diloco-drained-stop-{request['id']}.json").read_text())
        assert receipt['checkpoint']['sha256']==resident.sha(out/'checkpoint.pt') and receipt['global_step']==state['global_step']
        assert not (out/trainer.DILOCO_DRAIN_REQUEST).exists() and (out/f"diloco-drain-request-{request['id']}.consumed.json").exists()
        relaunch=common+['--native-graphs',str(new),'--continued-native-graphs',str(old),'--out',str(out),'--continue-from',str(seed),'--diloco',json.dumps(settings.__dict__)]
        assert trainer.main(relaunch)==0
        final=torch.load(out/'checkpoint.pt',weights_only=False)['training']
        assert final['exposure_counts']==dict.fromkeys('abcdefg',1) and final['global_step']>state['global_step']
        assert final['diloco']['contract_history'][0]['from']==saved['contract_sha256']
        progress=json.loads((root/'progress.json').read_text());registered=progress['role_contracts']['0']
        assert len(registered)==1 and registered[0]['source_transition']['checkpoint_sha256']==receipt['checkpoint']['sha256']
        assert registered[0]['source_transition']['prior_source_sha256']==resident.sha(old)
    finally:stop.set();thread.join(timeout=5)
