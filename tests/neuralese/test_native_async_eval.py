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
          'runtime':str(runtime),'noise_repeats':1,'workers':1,'temperature':.7,'neuralese_server':True}
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
    (Path(active['spec']['runtime'])/'prelude.js').write_text('changed runtime')
    result=task_cadence.run_request(Path(active['request']))
    assert result['eligible'] is False and result['task_successes'] is None
    assert result['model_outcome']=='unscored' and 'closure changed' in result['error']


def test_live_rebuild_does_not_mutate_sealed_async_runtime(tmp_path):
    from natlang_neuralese.eval.native_async import verify_request
    base,_,checkpoint,spec,export=fixture(tmp_path)
    source=Path(spec['runtime'])
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu')
    request=json.loads((Path(active['request'])/'request.json').read_text())
    prepared=Path(request['spec']['runtime'])
    assert prepared!=source and (prepared/'prelude.js').stat().st_ino!=(source/'prelude.js').stat().st_ino
    (source/'prelude.js').write_text('live rebuild')
    (source.parent/'training/source-reviews/holds.jsonl').write_text('new live hold')
    verify_request(request)
    assert (prepared/'prelude.js').read_text()=='{}'


@pytest.mark.parametrize('consume_export',[False,True])
def test_sealed_weights_have_no_mutable_source_alias(tmp_path,consume_export):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    original=checkpoint.stat()
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',device='cpu',consume_export=consume_export)
    request=json.loads((Path(active['request'])/'request.json').read_text())
    weights=Path(request['weights'])
    assert weights.stat().st_nlink==1
    if consume_export:
        assert not checkpoint.exists() and weights.stat().st_ino==original.st_ino
    else:
        assert weights.stat().st_ino!=original.st_ino
    # Reusing a former download name or rewriting an actual local rolling
    # checkpoint cannot rewrite the sealed model's bytes.
    checkpoint.write_bytes(b'new mutable checkpoint')
    assert file_binding(weights)['sha256']==export['receipt']['identity']['sha256']


def test_consumed_external_payload_keeps_logical_path_as_request_symlink(tmp_path):
    base,_,checkpoint,spec,export=fixture(tmp_path/'protocol')
    identity=export['receipt']['identity'];payload_root=tmp_path/'external'
    payload_root.mkdir()
    payload=payload_root/identity['sha256']/('checkpoint-'+identity['sha256']+'.pt');payload.parent.mkdir();payload.write_bytes(checkpoint.read_bytes())
    incoming=tmp_path/'eval'/'incoming';incoming.mkdir(parents=True)
    logical=incoming/('checkpoint-'+identity['sha256']+'.pt')
    logical.symlink_to(payload)
    active=seal_request(logical,identity=identity,step=7,spec=spec,out=tmp_path/'eval',
        base=base,queue=tmp_path/'queue',device='cpu',consume_export=True)
    request=json.loads((Path(active['request'])/'request.json').read_text())
    weights=Path(request['weights'])
    assert not logical.exists() and not logical.is_symlink()
    assert weights.is_symlink() and weights.resolve()==payload.resolve()
    assert file_binding(weights)['sha256']==identity['sha256']
    assert payload.is_file() and payload.stat().st_size==identity['bytes']


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


@pytest.mark.parametrize('role',['lfm','mellum'])
def test_interrupted_real_transfer_resumes_original_receipt_before_newer_export(tmp_path,monkeypatch,role):
    import copy,subprocess
    from natlang_neuralese.eval import native_async
    base,_,checkpoint,spec,export=fixture(tmp_path)
    export['receipt']['role']=role
    old_bytes=checkpoint.read_bytes();old_identity=export['receipt']['identity']
    export_path=tmp_path/'export-state.json';export_path.write_text(json.dumps(export))
    newer=copy.deepcopy(export);newer['last_verified_step']=8
    newer['receipt']['identity']={'sha256':'a'*64,'bytes':len(old_bytes)}
    newer['receipt']['independent_verification']=newer['receipt']['identity']
    calls=[]
    def transfer(argv,**kwargs):
        receipt=json.loads(Path(argv[1]).read_text());destination=Path(argv[2])
        assert receipt==export['receipt']
        pending=destination.with_name(destination.name+'.pending')
        calls.append(receipt['identity']['sha256'])
        if len(calls)==1:
            pending.write_bytes(old_bytes[:len(old_bytes)//2])
            export_path.write_text(json.dumps(newer))
            raise subprocess.CalledProcessError(1,argv)
        assert pending.read_bytes()==old_bytes[:len(old_bytes)//2]
        destination.write_bytes(old_bytes);pending.unlink()
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(native_async.subprocess,'run',transfer)
    options=dict(out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',
                 fetch_command=['fetch','{receipt}','{out}'],device='cpu')
    with pytest.raises(subprocess.CalledProcessError):poll_latest(export_path,**options)
    state_path=tmp_path/'eval/native-async-state.json'
    interrupted=json.loads(state_path.read_text())
    assert interrupted['fetching_export']['export']=={'last_verified_step':7,'receipt':export['receipt']}
    assert 'active'not in interrupted
    # The exact verified transfer survives even an unavailable latest-export file.
    export_path.unlink()
    resumed=poll_latest(export_path,**options)
    assert resumed['step']==7 and resumed['checkpoint_sha256']==old_identity['sha256']
    request=Path(resumed['request'])/'request.json';frozen=request.read_bytes()
    assert json.loads(frozen)['evaluation_provenance']['source_export']==export['receipt']
    assert 'fetching_export'not in json.loads(state_path.read_text())
    export_path.write_text(json.dumps(newer))
    assert poll_latest(export_path,**options)['active_step']==7
    assert request.read_bytes()==frozen and calls==[old_identity['sha256']]*2


def test_interrupted_transfer_binding_tamper_rejected_before_latest_read(tmp_path):
    from natlang_neuralese.eval.native_async import selected_ingest_export
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    selected={'last_verified_step':7,'receipt':{'role':'lfm','private':True,
        'identity':{'sha256':'a'*64,'bytes':10},'independent_verification':{'sha256':'a'*64,'bytes':10}}}
    pending={'schema':'natlang.native-eval-transfer/1','export':selected,
             'sha256':canonical_json_sha256_hex(selected)}
    selected['last_verified_step']=8
    with pytest.raises(ValueError,match='transfer binding changed'):
        selected_ingest_export(tmp_path/'unavailable-export',{'fetching_export':pending})


def test_null_interrupted_transfer_rejected_before_latest_read(tmp_path):
    from natlang_neuralese.eval.native_async import selected_ingest_export
    with pytest.raises(ValueError,match='transfer binding changed'):
        selected_ingest_export(tmp_path/'unavailable-export',{'fetching_export':None})


def test_task_eval_selects_only_verified_current_owner_generation(tmp_path):
    from natlang_neuralese.eval.native_async import selected_ingest_export
    from natlang_neuralese.common.jsonio import write_json_durable
    role='lfm';generation='c'*64
    index_path=tmp_path/'owner-follow'/'owned-publication-owner-index.json'
    generation_dir=index_path.parent/'owner-generations'/generation
    generation_dir.mkdir(parents=True)
    manifest={'path':'/owner/manifest.json','sha256':'a'*64,'bytes':128}
    current={'schema':'natlang.owned-publication-owner-selection/1','role':role,
        'generation_sha256':generation,'manifest':manifest}
    write_json_durable(generation_dir/'owner-selection.json',current)
    identity={'sha256':'b'*64,'bytes':4096}
    state={'last_verified_step':17,'receipt':{'role':role,'private':True,
        'identity':identity,'independent_verification':dict(identity),'run':'fixture/run',
        'name':'fixture.pt'}}
    write_json_durable(generation_dir/'coreweave-native-lfm-periodic-export-state-20261015.json',state)
    write_json_durable(index_path,{'schema':'natlang.owned-publication-owner-index/1','role':role,
        'generations':[current],'current':current})
    selected=selected_ingest_export(index_path,{})
    assert selected['last_verified_step']==17
    assert selected['receipt']==state['receipt']
    assert selected['owner_generation_sha256']==generation


@pytest.mark.parametrize('tamper', ['wrong-role', 'wrong-generation', 'receipt-role'])
def test_task_eval_rejects_owner_generation_journal_mismatch(tmp_path,tamper):
    from natlang_neuralese.eval.native_async import selected_ingest_export
    from natlang_neuralese.common.jsonio import write_json_durable
    role='lfm';generation='d'*64
    index_path=tmp_path/'owner-follow'/'owned-publication-owner-index.json'
    generation_dir=index_path.parent/'owner-generations'/generation
    generation_dir.mkdir(parents=True)
    manifest={'path':'/owner/manifest.json','sha256':'a'*64,'bytes':128}
    current={'schema':'natlang.owned-publication-owner-selection/1','role':role,
        'generation_sha256':generation,'manifest':manifest}
    selection=dict(current)
    if tamper=='wrong-role':selection['role']='mellum'
    if tamper=='wrong-generation':selection['generation_sha256']='e'*64
    write_json_durable(generation_dir/'owner-selection.json',selection)
    identity={'sha256':'b'*64,'bytes':4096}
    receipt_role='mellum' if tamper=='receipt-role' else role
    state={'last_verified_step':17,'receipt':{'role':receipt_role,'private':True,
        'identity':identity,'independent_verification':dict(identity)}}
    write_json_durable(generation_dir/'coreweave-native-lfm-periodic-export-state-20261015.json',state)
    write_json_durable(index_path,{'schema':'natlang.owned-publication-owner-index/1','role':role,
        'generations':[current],'current':current})
    with pytest.raises(ValueError):selected_ingest_export(index_path,{})


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


@pytest.mark.parametrize('mode',[False,None])
def test_native_async_rejects_plain_transport_before_creating_a_request(tmp_path,mode):
    base,_,checkpoint,spec,export=fixture(tmp_path)
    if mode is None:spec.pop('neuralese_server')
    else:spec['neuralese_server']=mode
    with pytest.raises(ValueError,match='native task evaluation requires'):
        seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
                     out=tmp_path/'eval',base=base,queue=tmp_path/'queue')
    assert not (tmp_path/'queue').exists()


def test_live_native_evaluator_rejects_plain_client_but_legacy_parser_keeps_it(tmp_path):
    from types import SimpleNamespace
    spec=task_cadence.parse_task_eval({'packet':{'path':'x','sha256':'a'*64}})
    assert spec['neuralese_server'] is False
    with pytest.raises(ValueError,match='native task evaluation requires'):
        task_cadence.TaskEval(spec,tmp_path,SimpleNamespace(native_state_enabled=True),prompt_rendering='full')


@pytest.mark.parametrize('role',['lfm','mellum'])
def test_verified_existing_download_cannot_be_replaced_by_stale_pending(tmp_path,monkeypatch,role):
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
    result=module.fetch({'private':True,'role':role,'identity':identity,
                        'independent_verification':identity},destination)
    assert result['sha256']==identity['sha256'] and destination.read_bytes()==b'verified'


def test_external_fetch_publishes_only_verified_content_addressed_symlink(tmp_path,monkeypatch):
    import importlib.util,sys,types
    path=Path(__file__).resolve().parents[2]/'scripts/coreweave/fetch_native_eval_export.py'
    monkeypatch.setitem(sys.modules,'cwctl',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace())
    spec=importlib.util.spec_from_file_location('external_native_fetch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload_root=tmp_path/'external';payload_root.mkdir()
    seed=payload_root/'seed.pt';seed.write_bytes(b'private-view')
    identity=file_binding(seed)
    payload=payload_root/identity['sha256']/('checkpoint-'+identity['sha256']+'.pt');payload.parent.mkdir();seed.replace(payload)
    destination=tmp_path/'eval'/'incoming'/'checkpoint.pt'
    receipt={'private':True,'role':'mellum','identity':identity,'independent_verification':dict(identity)}
    result=module.fetch(receipt,destination,payload_root=payload_root,require_separate_filesystem=False)
    assert result['path']==str(destination) and result['payload_path']==str(payload_root/identity['sha256']/('checkpoint-'+identity['sha256']+'.pt'))
    assert destination.is_symlink() and destination.resolve()==Path(result['payload_path'])
    assert file_binding(destination)['sha256']==identity['sha256']
    destination.unlink();payload.write_bytes(b'tampered-view')
    with pytest.raises(ValueError,match='differs from private receipt'):
        module.publish_payload_link(payload,destination,identity)


def test_external_fetch_enforces_floor_before_provider_access(tmp_path,monkeypatch):
    import importlib.util,sys,types
    from types import SimpleNamespace
    path=Path(__file__).resolve().parents[2]/'scripts/coreweave/fetch_native_eval_export.py'
    monkeypatch.setitem(sys.modules,'cwctl',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace())
    spec=importlib.util.spec_from_file_location('external_native_floor_fetch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload_root=tmp_path/'external';payload_root.mkdir()
    receipt={'private':True,'role':'lfm','identity':{'sha256':'a'*64,'bytes':100},
             'independent_verification':{'sha256':'a'*64,'bytes':100}}
    class NeverRequests:
        def get(self,*args,**kwargs):pytest.fail('provider access before external floor admission')
    floor=20<<30
    with pytest.raises(OSError,match='free-space floor'):
        module.fetch(receipt,tmp_path/'logical.pt',payload_root=payload_root,
                     minimum_external_free_bytes=floor,requests_client=NeverRequests(),
                     wandb_api=object(),available=lambda _:SimpleNamespace(free=floor+99),
                     require_separate_filesystem=False)


def test_external_fetch_rejects_root_on_same_filesystem(tmp_path,monkeypatch):
    import importlib.util,sys,types
    path=Path(__file__).resolve().parents[2]/'scripts/coreweave/fetch_native_eval_export.py'
    monkeypatch.setitem(sys.modules,'cwctl',types.SimpleNamespace())
    monkeypatch.setitem(sys.modules,'wandb',types.SimpleNamespace())
    spec=importlib.util.spec_from_file_location('external_native_same_device_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload_root=tmp_path/'external';payload_root.mkdir()
    receipt={'private':True,'role':'lfm','identity':{'sha256':'b'*64,'bytes':100},
             'independent_verification':{'sha256':'b'*64,'bytes':100}}
    with pytest.raises(ValueError,match='shares the logical path filesystem'):
        module.fetch(receipt,tmp_path/'logical.pt',payload_root=payload_root)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_native_inference_view_seals_restores_and_keeps_distinct_parent(tmp_path,family):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.train.checkpoint_offload import load_state,pruned_copy
    from natlang_neuralese.eval.native_async import verified_export
    _,_,_,spec,_=fixture(tmp_path/'protocol')
    (tmp_path/'trained').mkdir()
    original=engine_fixture(tmp_path/'trained',family)
    state=load_state(original.native_checkpoint)
    state.update(step=5000,training={'optimizer':{'moment':torch.arange(100000)},'rng':{'cpu':torch.get_rng_state()}})
    full=tmp_path/'full.pt';torch.save(state,full)
    source_sha=sha256_file_hex_cached(full)
    view=tmp_path/'inference.pt';pruned_copy(full,view)
    identity=file_binding(view)
    exported={'last_verified_step':5000,'receipt':{'role':family,'private':True,'identity':identity,
        'independent_verification':dict(identity)}}
    assert verified_export(exported)==exported['receipt']
    active=seal_request(view,identity=identity,step=5000,spec=spec,out=tmp_path/'eval',
        base=Path(original.model_config['base']),queue=tmp_path/'queue',source_export=exported['receipt'],device='cpu')
    request_dir=Path(active['request']);request=json.loads((request_dir/'request.json').read_text())
    restored=task_cadence.restore_engine({**request,'dir':str(request_dir)})
    try:
        provenance=request['evaluation_provenance']
        assert provenance['source_checkpoint_sha256']==identity['sha256']!=source_sha
        assert provenance['source_training_checkpoint_sha256']==source_sha
        assert restored.native_identity['checkpoint_sha256']==identity['sha256']
        assert restored.rl_policy_binding['inference_source']['training_checkpoint_sha256']==source_sha
        for name,parameter in original.shared_model.named_parameters():
            assert torch.equal(parameter,dict(restored.shared_model.named_parameters())[name])
    finally:original.store.close();restored.store.close()


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


def test_native_dispatch_memory_families_are_separate_and_live_units_stay_owned(tmp_path,monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'scripts/task_eval_dispatch.py'
    module=importlib.util.spec_from_file_location('native_dispatch_family_test',path)
    dispatcher=importlib.util.module_from_spec(module);module.loader.exec_module(dispatcher)
    args={'stage_out':str(tmp_path/'stage'),'step':7,'native_state':True,
          'evaluation_provenance':{'model_config':{'architecture':'lfm2'}}}
    lfm=dispatcher.unit_name(tmp_path,args)
    args['evaluation_provenance']['model_config']['architecture']='mellum'
    mellum=dispatcher.unit_name(tmp_path,args)
    from scripts.memory_ledger import family
    assert family(lfm)=='taskeval-native-lfm2'
    assert family(mellum)=='taskeval-native-mellum'
    args['native_state']=False
    legacy=dispatcher.unit_name(tmp_path,args)
    assert family(legacy)=='taskeval'
    request=tmp_path/'request';request.mkdir();queue=tmp_path/'queue';queue.mkdir()
    args['native_state']=True
    (request/'request.json').write_text(json.dumps(args))
    (request/'dispatch.json').write_text(json.dumps({'unit':legacy,'attempts':1}))
    (queue/'pointer.json').write_text(json.dumps({'request':str(request)}))
    observed=[]
    def state(unit):
        observed.append(unit)
        return 'active' if unit==legacy else 'none'
    monkeypatch.setattr(dispatcher,'unit_state',state)
    assert dispatcher.dispatch(queue)==[]
    assert observed==[legacy]


@pytest.mark.parametrize('role',['lfm','mellum'])
@pytest.mark.parametrize('boundary',['preparation','weights','request','journal','published'])
def test_real_seal_crash_reuses_weights_request_and_single_queue_pointer(tmp_path,monkeypatch,role,boundary):
    import subprocess
    from natlang_neuralese.eval import native_async
    base,_,checkpoint,spec,export=fixture(tmp_path)
    export['receipt']['role']=role
    source_bytes=checkpoint.read_bytes();calls=[]
    export_path=tmp_path/'export.json';export_path.write_text(json.dumps(export))
    def fetch(argv,**kwargs):
        calls.append(argv);Path(argv[2]).write_bytes(source_bytes)
        return subprocess.CompletedProcess(argv,0)
    monkeypatch.setattr(native_async.subprocess,'run',fetch)
    real_write=native_async.write_json_durable;real_publish=native_async._publish_request
    class Crash(RuntimeError):pass
    def interrupt_write(path,value):
        path=Path(path)
        if boundary=='preparation'and path.name=='native-async-state.json'and 'sealing_request'in value:raise Crash()
        if boundary=='weights'and path.name=='request.json':raise Crash()
        if boundary=='request'and path.name=='native-async-state.json'and value.get('active',{}).get('publication_pending'):raise Crash()
        if boundary=='published'and path.name=='native-async-state.json'and 'active'in value and not value['active'].get('publication_pending'):raise Crash()
        return real_write(path,value)
    def interrupt_publish(active,queue):
        if boundary=='journal':raise Crash()
        return real_publish(active,queue)
    monkeypatch.setattr(native_async,'write_json_durable',interrupt_write)
    monkeypatch.setattr(native_async,'_publish_request',interrupt_publish)
    options=dict(out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',
                 fetch_command=['fetch','{receipt}','{out}'],device='cpu')
    with pytest.raises(Crash):poll_latest(export_path,**options)
    request=tmp_path/'eval/checkpoints'/export['receipt']['identity']['sha256']/'task-eval/async/step-0000007/request.json'
    frozen=request.read_bytes()if request.exists()else None
    weights=request.with_name('weights.pt');assert weights.read_bytes()==source_bytes
    state_path=tmp_path/'eval/native-async-state.json'
    assert json.loads(state_path.read_text())['fetching_export']['export']['last_verified_step']==7
    export_path.unlink()  # Recovery must use durable admission, not latest discovery.
    monkeypatch.setattr(native_async,'write_json_durable',real_write)
    monkeypatch.setattr(native_async,'_publish_request',real_publish)
    resumed=poll_latest(export_path,**options)
    assert resumed.get('step',resumed.get('active_step'))==7
    assert len(calls)==1 and weights.read_bytes()==source_bytes
    if frozen is not None:assert request.read_bytes()==frozen
    pointers=list((tmp_path/'queue').glob('*.json'));assert len(pointers)==1
    assert json.loads(pointers[0].read_text())=={'request':str(request.parent)}
    final=json.loads(state_path.read_text());assert 'fetching_export'not in final
    assert 'sealing_request'not in final
    assert not final['active'].get('publication_pending')
    assert poll_latest(export_path,**options)['active_step']==7
    assert len(calls)==1


@pytest.mark.parametrize('field',['source_export','spec','weights','model_config','valid_other_runtime'])
def test_recovered_sealed_request_cannot_change_admission_or_saved_model(tmp_path,monkeypatch,field):
    from natlang_neuralese.eval import native_async
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    base,_,checkpoint,spec,export=fixture(tmp_path)
    def prepare(directory,request):
        native_async.write_json_durable(tmp_path/'eval/native-async-state.json',dict(sealing_request=dict(
            schema='natlang.native-eval-seal/1',directory=str(directory),request=request,
            sha256=canonical_json_sha256_hex(request))))
    active=seal_request(checkpoint,identity=export['receipt']['identity'],step=7,spec=spec,
        out=tmp_path/'eval',base=base,queue=tmp_path/'queue',source_export=export['receipt'],device='cpu',on_prepared=prepare)
    request_path=Path(active['request'])/'request.json';request=json.loads(request_path.read_text())
    if field=='source_export':request['evaluation_provenance']['source_export']['run']='another/run'
    elif field=='spec':request['spec']['temperature']=.99
    elif field=='weights':Path(request['weights']).write_bytes(b'changed model')
    elif field=='model_config':request['evaluation_provenance']['model_config']['rank']=999
    else:
        from natlang_neuralese.common.runtime_inventory import snapshot_task_runtime,task_runtime_inventory
        (Path(spec['runtime'])/'prelude.js').write_text('another valid compiled runtime')
        other=snapshot_task_runtime(Path(spec['runtime']),tmp_path/'other-runtimes')
        inventory=task_runtime_inventory(other)
        request['spec']['runtime']=str(other)
        request['evaluation_provenance']['runtime']=dict(path=str(other),inventory=inventory,sha256=canonical_json_sha256_hex(inventory))
        request['evaluation_provenance']['evaluation_spec_sha256']=canonical_json_sha256_hex(request['spec'])
        native_async.verify_request(request)  # Internally valid; prior seal must still reject it.
    request_path.write_text(json.dumps(request))
    export_path=tmp_path/'export.json';export_path.write_text(json.dumps(export))
    monkeypatch.setattr(native_async.subprocess,'run',lambda *a,**kw:pytest.fail('must not fetch a sealed request'))
    with pytest.raises(ValueError,match='sealed request differs'):
        poll_latest(export_path,out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',fetch_command=[],device='cpu')


def test_publication_crash_after_dispatch_drain_keeps_result_and_clears_pin(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_async
    base,_,checkpoint,spec,export=fixture(tmp_path)
    export_path=tmp_path/'export.json';export_path.write_text(json.dumps(export))
    real_write=native_async.write_json_durable
    def interrupt(path,value):
        if Path(path).name=='native-async-state.json'and 'active'in value and not value['active'].get('publication_pending'):
            raise RuntimeError('crash after publishing')
        return real_write(path,value)
    monkeypatch.setattr(native_async,'write_json_durable',interrupt)
    options=dict(out=tmp_path/'eval',spec=spec,base=base,queue=tmp_path/'queue',fetch_command=[],local_checkpoint=checkpoint,device='cpu')
    with pytest.raises(RuntimeError,match='after publishing'):poll_latest(export_path,**options)
    state_path=tmp_path/'eval/native-async-state.json';active=json.loads(state_path.read_text())['active']
    def execute(self,run,repeat=0):
        run.mkdir(parents=True,exist_ok=True)
        return {'task_successes':1,'task_success_rate':1.,'complete':True,'cases':[],
                'cases_expected':1,'outcome_counts':{'task_success':1,'infrastructure_error':0,'unresolved':0}},[0]
    monkeypatch.setattr(task_cadence.TaskEval,'_execute',execute)
    assert task_cadence.run_request(Path(active['request']))['eligible']
    assert not Path(active['request']).exists()
    monkeypatch.setattr(native_async,'write_json_durable',real_write)
    assert poll_latest(export_path,**options)['event']=='native_eval_up_to_date'
    state=json.loads(state_path.read_text());assert 'active'not in state and 'fetching_export'not in state
    assert state['last_completed_sha256']==export['receipt']['identity']['sha256']
    assert json.loads((tmp_path/'eval/latest-result.json').read_text())['result']['task_successes']==1
