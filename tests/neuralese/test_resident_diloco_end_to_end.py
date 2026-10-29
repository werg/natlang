"""Exercise real trainer, resident transaction and HTTP outer recovery together."""
import copy,json,socket,threading,time
import pytest,torch
from pathlib import Path
from test_native_model import tiny_source
from test_native_state_sft import graph
from test_native_resident import setup
from natlang_neuralese.train import native_resident as resident,native_state_sft as trainer
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint
from natlang_neuralese.train.diloco.worker import DilocoSettings,Transport
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator,TOKEN_ENV
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.native_diloco import identity


def test_native_resident_append_crash_with_real_outer(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    monkeypatch.setenv(TOKEN_ENV,'x'*32)
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    ident={'model_config':config,'batch_graphs':1,'batch_tokens':0,'checkpoint_layers':True,'chunk_tokens':1024,'checkpoint_input_offload':False,'cotangent_accumulator_backend':'plain','share_causal_prefixes':True,'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16,'input_gradient_precision':None},'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':'cpu'}}
    paths,directory,manager,newrecipe=setup(tmp_path,config['tokenizer_fingerprint'],roles.ids,ident,manifest_train=True)
    # The appended manifest retains the exact first leaf, plus one new graph.
    extra=tmp_path/'extra.jsonl';row=graph('c');row.update(split='train',tokenizer_fingerprint=config['tokenizer_fingerprint']);extra.write_text(json.dumps(row)+'\n')
    successor=tmp_path/'successor.json';write_json_durable(successor,{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**resident.binding(paths['old'])},{'kind':'native-graphs',**resident.binding(extra)}]})
    content=tmp_path/'content.json';write_json_durable(content,{'schema':'natlang.big2-multi-content-audit/1','base':{'sha256':resident.sha(paths['old'])},'candidates':[{'sha256':resident.sha(extra)}],'duplicates':[]})
    protect=tmp_path/'protection.json';write_json_durable(protect,{'schema':'natlang.fresh-restart-protection-union/1','sources':{'base':{'source':resident.binding(paths['old']),'conflicts':[]},'candidate-001':{'source':resident.binding(extra),'conflicts':[]}}})
    new=json.loads(newrecipe.read_text());new['input_bindings']['train']=resident.binding(successor);new['external_provenances']=[{'schema':'natlang.external-source-publication-provenance/1','closure':resident.binding(content),'content_audit':resident.binding(content),'clean_closure_audit':resident.binding(protect)}];write_json_durable(newrecipe,new)
    seedout=tmp_path/'seed';common=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(paths['old']),'--held-native-graphs',str(paths['held']),'--epochs','1','--batch-graphs','1','--accumulate-batches','1','--eval-every','1','--checkpoint-every','1']
    trainer.main(common+['--out',str(seedout),'--initialize-only'])
    seed=seedout/'checkpoint.pt';model,*_=load_native_checkpoint(seed,base=base,device='cpu')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    url=f'http://127.0.0.1:{port}';settings=DilocoSettings(run_id='resident-e2e',workers=1,worker_id=0,coordinator_url=url,allow_private_http=True,local_steps=1,wire_dtype='fp32',heartbeat_seconds=.1)
    out=directory.parent/'sft';binding=manager.recipe_binding
    argv=common+['--out',str(out),'--continue-from',str(seed),'--resident-data-directory',str(directory),'--resident-geometry-qualification',str(tmp_path/'geometry-capacity.json'),'--recipe-binding',json.dumps(binding),'--stage-id','sft','--diloco',json.dumps(settings.__dict__)]
    role={'kind':'sft','identity':identity(seed,argv,settings),'dependencies':[resident.binding(paths['old']),resident.binding(paths['held']),resident.binding(Path(binding['path']))]}
    authority={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':resident.sha(seed),'recipe_lineage_sha256':binding['lineage_sha256'],'diloco':settings.identity(),'roles':{'0':[role]}}
    root=tmp_path/'outer';named=trainer.native_trainable_parameters(model);initialize_coordinator(root,run_id='resident-e2e',masters=TensorSurface(dict(named)).masters,identity=authority,workers=1,min_quorum=1,grace_seconds=0,fragment_bytes=1<<28,seed_source={'sha256':resident.sha(seed)})
    stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,'x'*32),kwargs={'stop':stop},daemon=True);thread.start();transport=Transport(url,'x'*32,10,allow_http=True)
    for _ in range(100):
        try:transport.json('GET','/v1/manifest');break
        except OSError:time.sleep(.02)
    original=trainer.CheckpointPolicy.due;prepared=[]
    def due(policy,step,**kwargs):
        if step==1 and not prepared:
            prepared.append(resident.prepare(directory,newrecipe,'sft',tokenizer_fingerprint=config['tokenizer_fingerprint'],role_ids=roles.ids))
        return original(policy,step,**kwargs)
    original_poll=resident.ResidentAdoption.poll
    class Crash(BaseException):pass
    def crash(self,step,**kwargs):
        result=original_poll(self,step,**kwargs)
        if result is not None:raise Crash()
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(trainer.CheckpointPolicy,'due',due);patch.setattr(resident.ResidentAdoption,'poll',crash)
            with pytest.raises(Crash):trainer.main(argv)
        assert resident.current(directory)
        assert trainer.main(argv)==0
        final=torch.load(out/'checkpoint.pt',weights_only=False)
        assert final['training']['exposure_counts']=={'a':1,'b':1,'c':1}
        assert final['training']['diloco']['contract_history']
        assert transport.json('GET','/v1/status')['versions'][0]>=1
    finally:stop.set();thread.join(timeout=5)
