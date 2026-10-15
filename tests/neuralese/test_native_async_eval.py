"""Native restores and queue provenance; fixture scoring is not task lift."""
import json
from pathlib import Path
import pytest
import torch
from natlang_neuralese.eval.native_async import seal_request,file_binding,poll_latest
from natlang_neuralese.eval import task_cadence
from natlang_neuralese.model.native_model import load_native_model,model_state,CHECKPOINT_SCHEMA
from natlang_neuralese.model.shared_state import Invocation,compile_graph
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from test_native_serve import _tiny_source as tiny_source


def fixture(tmp_path,step=7):
    base=tmp_path/'base';tiny_source(base)
    model,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    with torch.no_grad():model.input_rows.add_(.03)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'step':step,'model_config':config,**model_state(model)},checkpoint)
    root=tmp_path/'runtime';runtime=root/'ts-host';(runtime/'dist/teacher').mkdir(parents=True)
    for name in ('dist/teacher/cli.js','prelude.js','package.json'):(runtime/name).write_text('{}')
    for name in ('package.json','package-lock.json','training/source-reviews/holds.jsonl',
                 'training/source-reviews/family-holds.json','training/source-reviews/datasets.json'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    packet=tmp_path/'cases.jsonl';packet.write_text(json.dumps({'id':'held','split':'test'})+'\n')
    spec={'packet':{'path':str(packet),'sha256':sha256_file_hex_cached(packet)},'cases':[0],
          'runtime':str(runtime),'noise_repeats':1,'workers':1,'temperature':.7}
    identity=file_binding(checkpoint)
    export={'last_verified_step':step,'receipt':{'role':'lfm','private':True,'identity':identity,
        'independent_verification':identity,'run':'fixture/run','name':'fixture.pt'}}
    return base,model,checkpoint,spec,export


def test_real_native_async_restore_exact_binding_and_landed_result(tmp_path,monkeypatch):
    base,model,checkpoint,spec,export=fixture(tmp_path)
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    request_dir=Path(active['request']);request=json.loads((request_dir/'request.json').read_text())
    engine=task_cadence.restore_engine({**request,'dir':str(request_dir)})
    plan=compile_graph([Invocation('probe',(1,4,5,2))])
    with torch.no_grad():torch.testing.assert_close(engine.shared_model.logits(engine.shared_model(plan)['h_final']),
        model.logits(model(plan)['h_final']),rtol=0,atol=0)
    assert engine.native_identity['checkpoint_sha256']==export['receipt']['identity']['sha256']
    def execute(self,run,repeat=0):
        run.mkdir(parents=True,exist_ok=True)
        assert self.engine.heads is None and self.engine.shared_model is not None
        return {'task_successes':0,'task_success_rate':0,'complete':True,'cases':[],
                'cases_expected':1,'outcome_counts':{'task_success':0,'semantic_mismatch':1,'infrastructure_error':0,'unresolved':0}},[0]
    monkeypatch.setattr(task_cadence.TaskEval,'_execute',execute)
    result=task_cadence.run_request(request_dir)
    assert result['eligible'] and result['historical'] and not result['qualifies_current_weights']
    provenance=result['evaluation_provenance']
    assert provenance['native_identity']==engine.native_identity
    assert provenance['sampler']['temperature']==.7 and provenance['runtime']['inventory']
    assert not request_dir.exists() and checkpoint.exists()
    logged=json.loads((Path(active['stage_out'])/'task-eval.jsonl').read_text())
    assert logged['evaluation_provenance']==provenance


def test_native_runtime_change_is_unscored_and_never_falls_back_to_heads(tmp_path):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    (Path(spec['runtime'])/'prelude.js').write_text('changed runtime')
    result=task_cadence.run_request(Path(active['request']))
    assert result['eligible'] is False and result['task_successes'] is None
    assert result['model_outcome']=='unscored' and 'closure changed' in result['error']


def test_latest_export_coalesces_without_changing_running_request(tmp_path,monkeypatch):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    export_path=tmp_path/'export-state.json';export_path.write_text(json.dumps(export))
    options=dict(out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',fetch_command=[],
                 local_checkpoint=checkpoint,device='cpu')
    first=poll_latest(export_path,**options)
    frozen=(Path(first['request'])/'request.json').read_bytes()
    # A newer verified export is observed only after this exact request drains.
    export['last_verified_step']=8;export['receipt']['identity']={**export['receipt']['identity'],'sha256':'a'*64}
    export['receipt']['independent_verification']=export['receipt']['identity']
    export_path.write_text(json.dumps(export))
    second=poll_latest(export_path,**options)
    assert second['event']=='native_eval_pending' and second['active_step']==7
    assert (Path(first['request'])/'request.json').read_bytes()==frozen


def test_export_schema_step_and_identity_are_exact(tmp_path):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    options=dict(identity=export['receipt']['identity'],spec=spec,out=tmp_path/'eval',base=base,queue=tmp_path/'queue')
    with pytest.raises(ValueError,match='step differs'):seal_request(checkpoint,step=8,**options)
    with pytest.raises(ValueError,match='differs from independently'):
        seal_request(checkpoint,step=7,**{**options,'identity':{'sha256':'0'*64,'bytes':1}})


def test_sampler_tamper_is_unscored(tmp_path):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    path=Path(active['request'])/'request.json';request=json.loads(path.read_text())
    request['spec']['temperature']=.1;path.write_text(json.dumps(request))
    result=task_cadence.run_request(Path(active['request']))
    assert result['task_successes'] is None and result['model_outcome']=='unscored'
    assert 'specification changed' in result['error']


def test_native_protocol_mode_requires_boolean():
    with pytest.raises(ValueError,match='neuralese_server'):
        task_cadence.parse_task_eval({'packet':{'path':'x','sha256':'a'*64},'neuralese_server':'false'})


def test_verified_existing_download_cannot_be_replaced_by_stale_pending(tmp_path,monkeypatch):
    import importlib.util
    import sys
    import types
    path=Path(__file__).resolve().parents[2]/'scripts/coreweave/fetch_native_eval_export.py'
    monkeypatch.setitem(sys.modules,'cwctl',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace())
    spec=importlib.util.spec_from_file_location('native_fetch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    destination=tmp_path/'checkpoint.pt';destination.write_bytes(b'verified')
    destination.with_suffix('.pt.pending').write_bytes(b'leftover rejected download')
    identity=file_binding(destination)
    result=module.fetch({'private':True,'role':'lfm','identity':identity,
                        'independent_verification':identity},destination)
    assert result['sha256']==identity['sha256'] and destination.read_bytes()==b'verified'


def test_native_collector_infrastructure_is_unscored(tmp_path,monkeypatch):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    def infrastructure(self,run,repeat=0):
        run.mkdir(parents=True,exist_ok=True)
        return {'task_successes':0,'task_success_rate':0,'complete':False,'cases':[],
                'cases_expected':1,'outcome_counts':{'task_success':0,'infrastructure_error':1,'unresolved':0}},[1]
    monkeypatch.setattr(task_cadence.TaskEval,'_execute',infrastructure)
    result=task_cadence.run_request(Path(active['request']))
    assert result['task_successes'] is None and result['model_outcome']=='unscored'
    assert result['observed_task_successes']==0 and not result['eligible']


def test_drained_historical_result_admits_latest_export_and_retains_report(tmp_path,monkeypatch):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    export_path=tmp_path/'export-state.json';export_path.write_text(json.dumps(export))
    options=dict(out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',fetch_command=[],
                 local_checkpoint=checkpoint,device='cpu')
    first=poll_latest(export_path,**options)
    task_cadence.TaskEval(first['spec'],Path(first['stage_out']),None,prompt_rendering='full').record_failure(7,'fixture collector transport')
    task_cadence.drop_request(Path(first['request']))
    state=torch.load(checkpoint,weights_only=True);state['step']=8;torch.save(state,checkpoint)
    export['last_verified_step']=8;export['receipt']['identity']=file_binding(checkpoint)
    export['receipt']['independent_verification']=export['receipt']['identity']
    export_path.write_text(json.dumps(export))
    second=poll_latest(export_path,**options)
    assert second['event']=='native_eval_enqueued' and second['step']==8
    assert second['checkpoint_sha256']!=first['checkpoint_sha256']
    report=json.loads((tmp_path/'eval/latest-result.json').read_text())
    assert report['step']==7 and report['historical'] and not report['qualifies_current_weights']
    assert checkpoint.exists() and Path(second['request']).exists()


def test_exhausted_dispatch_is_native_unscored(tmp_path,monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'scripts/task_eval_dispatch.py'
    spec_module=importlib.util.spec_from_file_location('native_dispatch_test',path)
    dispatcher=importlib.util.module_from_spec(spec_module);spec_module.loader.exec_module(dispatcher)
    base,_,checkpoint,spec,export=fixture(tmp_path)
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    request=Path(active['request']);(request/'dispatch.json').write_text(json.dumps({'attempts':3}))
    assert dispatcher.budget_gb(request)==16.
    request_path=request/'request.json';binding=json.loads(request_path.read_text())
    binding['native_state']=False;request_path.write_text(json.dumps(binding))
    assert dispatcher.budget_gb(request)<16.
    binding['native_state']=True;request_path.write_text(json.dumps(binding))
    monkeypatch.setattr(dispatcher,'unit_state',lambda unit:'none')
    events=dispatcher.dispatch(tmp_path/'queue')
    assert events[0]['event']=='failed' and not request.exists()
    result=json.loads((Path(active['stage_out'])/'task-eval.jsonl').read_text())
    assert result['task_successes'] is None and result['model_outcome']=='unscored'
    assert result['evaluation_provenance']['source_checkpoint_sha256']==export['receipt']['identity']['sha256']
