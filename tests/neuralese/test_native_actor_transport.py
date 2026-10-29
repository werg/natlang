import copy,json
from pathlib import Path
import pytest,torch
from natlang_neuralese.model.native_bundle import prepare_actor_transport,load_transported_actor_view,_file_binding
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.model.shared_state import Invocation,compile_graph
from test_native_actor_view import source


def test_transported_view_matches_actual_full_actor_without_source(source,tmp_path,monkeypatch):
    engine,path,state,_=source;binding=_file_binding(path)
    recovery=tmp_path/'recovery.json';verified={**binding,'readback_location':'independent-test-receiver'}
    receipt={'private':True,'upload_commit_notified':True,'upload':{'status_code':200,**binding},'identity':binding,'independent_verification':verified}
    recovery.write_text(json.dumps(receipt))
    manifest=prepare_actor_transport(path,tmp_path/'views',recovery);m=json.loads(Path(manifest['path']).read_text())
    folder=Path(manifest['path']).parent
    view=folder/m['view']['path']
    proof=next(folder.glob('*.json'))
    if proof.name.endswith('.transport.json'):proof=next(p for p in folder.glob('*.json') if not p.name.endswith('.transport.json'))
    refs={'transport':manifest,'view_receipt':_file_binding(proof),'recovery_receipt':_file_binding(recovery),'source_sha256':binding['sha256'],'scope':'independent-recovery'}
    original=load_native_engine(str(path),device='cpu')
    path.unlink()
    transported=load_native_engine(str(view),device='cpu',transported_actor=refs)
    try:
        assert transported.native_identity['checkpoint_sha256']==binding['sha256']
        assert transported.native_inference_only and transported.native_training_state=={}
        assert transported.native_actor_view['view']['sha256']==m['view']['sha256']
        plan=compile_graph([Invocation('x',(4,5,6,7))])
        with torch.no_grad():
            a=original.shared_model.logits(original.shared_model(plan)['h_final'])
            b=transported.shared_model.logits(transported.shared_model(plan)['h_final'])
        assert torch.equal(a,b)
        def greedy(engine):
            cache=None;tokens=[4,5,6]
            with torch.no_grad():
                for token in tokens:out=engine._native_forward([token],cache);cache=out['cache']
                generated=[]
                for _ in range(4):
                    token=int(engine.shared_model.logits(out['h_final'])[:,-1].argmax(-1).item())
                    generated.append(token);out=engine._native_forward([token],cache);cache=out['cache']
            return generated
        assert greedy(original)==greedy(transported)
        import importlib.util,sys
        from types import SimpleNamespace
        cli_path=Path(__file__).parents[2]/'scripts/neuralese_data/serve_native_checkpoint.py'
        spec=importlib.util.spec_from_file_location('transport_actor_cli',cli_path)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        seen=[]
        def serve(actual,host,port):
            seen.append(actual.native_identity['checkpoint_sha256'])
            return SimpleNamespace(server_address=(host,port),serve_forever=lambda:None,server_close=lambda:None)
        monkeypatch.setattr(cli,'serve',serve)
        monkeypatch.setattr(sys,'argv',['serve','--checkpoint',str(view),'--checkpoint-sha256',m['view']['sha256'],
            '--transport-receipt',manifest['path'],'--transport-receipt-sha256',manifest['sha256'],
            '--transport-view-receipt',str(proof),'--transport-view-receipt-sha256',refs['view_receipt']['sha256'],
            '--transport-recovery-receipt',str(recovery),'--transport-recovery-receipt-sha256',refs['recovery_receipt']['sha256'],
            '--training-source-sha256',binding['sha256'],'--transport-scope','independent-recovery','--device','cpu','--threads','1'])
        cli.main();assert seen==[binding['sha256']]
        with pytest.raises(ValueError):load_transported_actor_view(view,**{**refs,'source_sha256':'b'*64})
        changed=copy.deepcopy(m);changed['model_config']['marker_ids']=[1,2,3,4]
        forged=tmp_path/'forged.json';forged.write_text(json.dumps(changed))
        with pytest.raises(ValueError):load_transported_actor_view(view,**{**refs,'transport':_file_binding(forged)})
        changed=copy.deepcopy(m);changed['python_runtime']['sha256']='0'*64
        forged.write_text(json.dumps(changed))
        with pytest.raises(ValueError):load_transported_actor_view(view,**{**refs,'transport':_file_binding(forged)})
        recovery.write_text(json.dumps({**receipt,'private':False}))
        with pytest.raises(ValueError):load_transported_actor_view(view,**refs)
        recovery.write_text(json.dumps(receipt))
        view.write_bytes(b'tampered')
        with pytest.raises(ValueError):load_transported_actor_view(view,**refs)
    finally:
        original.store.close();transported.store.close()


def test_diagnostic_scope_has_no_recovery_claim_and_cannot_be_rl_policy(source,tmp_path,monkeypatch):
    from natlang_neuralese.rl.production import validate_actor_receipts
    _,path,_,_=source;binding=_file_binding(path)
    manifest=prepare_actor_transport(path,tmp_path/'diagnostic',scope='diagnostic-local-pin')
    value=json.loads(Path(manifest['path']).read_text());folder=Path(manifest['path']).parent
    proof=next(p for p in folder.glob('*.json') if not p.name.endswith('.transport.json'))
    view=folder/value['view']['path'];path.unlink()
    refs={'transport':manifest,'view_receipt':_file_binding(proof),'recovery_receipt':None,
          'source_sha256':binding['sha256'],'scope':'diagnostic-local-pin'}
    state=load_transported_actor_view(view,**refs)
    assert state['actor_view']['scope']=='diagnostic-local-pin' and value['recovery_receipt'] is None
    import importlib.util,sys
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('diagnostic_actor_cli',Path(__file__).parents[2]/'scripts/neuralese_data/serve_native_checkpoint.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli);seen=[]
    def serve(actual,host,port):
        seen.append(actual.native_actor_view['scope'])
        return SimpleNamespace(server_address=(host,port),serve_forever=lambda:None,server_close=lambda:None)
    monkeypatch.setattr(cli,'serve',serve)
    monkeypatch.setattr(sys,'argv',['serve','--checkpoint',str(view),'--checkpoint-sha256',value['view']['sha256'],
        '--transport-receipt',manifest['path'],'--transport-receipt-sha256',manifest['sha256'],
        '--transport-view-receipt',str(proof),'--transport-view-receipt-sha256',refs['view_receipt']['sha256'],
        '--training-source-sha256',binding['sha256'],'--transport-scope','diagnostic-local-pin','--device','cpu','--threads','1'])
    cli.main();assert seen==['diagnostic-local-pin']
    with pytest.raises(ValueError,match='scope'):
        load_transported_actor_view(view,**{**refs,'scope':'independent-recovery'})
    with pytest.raises(ValueError,match='diagnostic-only'):
        validate_actor_receipts({'actor':{'checkpoint_sha256':binding['sha256'],'actor_view':state['actor_view']}},
                                policy_id=binding['sha256'],neuralese_server=True)
