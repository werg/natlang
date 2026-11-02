"""Transported async custody plumbing; synthetic fixtures make no evaluation claim."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from natlang_neuralese.eval import native_async,task_cadence
from natlang_neuralese.model import native_bundle,native_model


def setup(tmp_path,monkeypatch):
    view=tmp_path/'view.pt';view.write_bytes(b'actual compact actor')
    transport=tmp_path/'transport.json';transport.write_text('{}')
    receipt=tmp_path/'view-receipt.json';receipt.write_text('{}')
    packet=tmp_path/'cases.jsonl';packet.write_text('{}\n')
    runtime=tmp_path/'runtime';runtime.mkdir();base=tmp_path/'base';base.mkdir()
    actor={'transport':native_async.file_binding(transport),'view_receipt':native_async.file_binding(receipt),
           'recovery_receipt':None,'source_sha256':'a'*64,'scope':'diagnostic-local-pin'}
    state={'schema':native_model.CHECKPOINT_SCHEMA,'step':7,'optimizer_pruned':True,
        'model_config':{'architecture':'lfm3','base_inventory':{'x':1}},
        'inference_source':{'schema':'natlang.native-inference-source/1','training_checkpoint_sha256':'a'*64,
                            'training_checkpoint_bytes':1000,'step':7}}
    monkeypatch.setattr(native_bundle,'load_transported_actor_view',lambda path,**kw: state if kw==actor else pytest.fail('wrong actor'))
    monkeypatch.setattr(native_model,'base_inventory',lambda path:{'x':1})
    monkeypatch.setattr(native_async,'snapshot_task_runtime',lambda source,out:runtime)
    monkeypatch.setattr(native_async,'task_runtime_inventory',lambda path:{'runtime':'fixture'})
    class Task:
        def __init__(self,spec,out,engine,**kw):self.root=Path(out)/'task-eval';self.repo=tmp_path;self.spec=spec
        def landed(self,step):return None
    monkeypatch.setattr(task_cadence,'TaskEval',Task)
    spec={'packet':{'path':str(packet),'sha256':native_async.file_binding(packet)['sha256']},
          'cases':[0],'runtime':str(runtime),'neuralese_server':True,'prompt_rendering':'full'}
    return view,actor,spec,base


def test_transported_request_seals_source_and_bounded_pending_queue(tmp_path,monkeypatch):
    view,actor,spec,base=setup(tmp_path,monkeypatch)
    args={'step':7,'spec':spec,'out':tmp_path/'out','base':base,'queue':tmp_path/'queue',
          'transported_actor':actor,'device':'cpu'}
    first=native_async.seal_transport_request(view,**args)
    assert first['event']=='native_eval_enqueued' and first['source_checkpoint_sha256']=='a'*64
    request=json.loads((Path(first['request'])/'request.json').read_text())
    assert request['evaluation_provenance']['source_training_checkpoint_sha256']=='a'*64
    assert request['evaluation_provenance']['historical'] and not request['evaluation_provenance']['qualifies_current_weights']
    assert native_async.seal_transport_request(view,**args)['event']=='native_eval_pending'
    assert native_async.file_binding(request['weights'])['sha256']==native_async.file_binding(view)['sha256']
    Path(actor['transport']['path']).write_text('{"changed":true}')
    with pytest.raises(ValueError,match='receipt changed'):
        native_async.seal_transport_request(view,**args)


def test_transport_restore_uses_source_identity_not_compact_hash(tmp_path,monkeypatch):
    import natlang_neuralese.serve as serve
    view,actor,spec,base=setup(tmp_path,monkeypatch)
    active=native_async.seal_transport_request(view,step=7,spec=spec,out=tmp_path/'out',base=base,
        queue=tmp_path/'queue',transported_actor=actor,device='cpu')
    request=json.loads((Path(active['request'])/'request.json').read_text())
    saved=request['evaluation_provenance']['model_config'];calls=[]
    def load(path,**kwargs):
        calls.append(kwargs);return SimpleNamespace(native_identity={'checkpoint_sha256':'a'*64},model_config=saved)
    monkeypatch.setattr(serve,'load_native_engine',load)
    engine=task_cadence.restore_engine({**request,'dir':active['request']})
    assert engine.native_identity['checkpoint_sha256']=='a'*64 and calls[0]['transported_actor']==actor
    assert request['evaluation_provenance']['checkpoint']['sha256']!=engine.native_identity['checkpoint_sha256']


def test_interrupted_transport_seal_recovers_original_request_and_queue(tmp_path,monkeypatch):
    view,actor,spec,base=setup(tmp_path,monkeypatch)
    args={'step':7,'spec':spec,'out':tmp_path/'out','base':base,'queue':tmp_path/'queue',
          'transported_actor':actor,'device':'cpu'}
    publish=native_async._publish_request;calls=[]
    def interrupted(active,queue):
        if not calls:calls.append(True);raise OSError('interrupted publication')
        return publish(active,queue)
    monkeypatch.setattr(native_async,'_publish_request',interrupted)
    with pytest.raises(OSError,match='interrupted'):native_async.seal_transport_request(view,**args)
    result=native_async.seal_transport_request(view,**args)
    assert result['event']=='native_eval_enqueued'
    pointer=tmp_path/'queue'/('native-'+native_async.file_binding(view)['sha256']+'.json')
    assert json.loads(pointer.read_text())['request']==result['request']
    pointer.unlink()
    assert native_async.seal_transport_request(view,**args)['event']=='native_eval_pending'
    assert pointer.exists()


def test_paired_development_checks_all_cases_and_runtime_geometry(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_execution_transition as tx
    packet=tmp_path/'packet.jsonl';rows=[{'id':f'case{i}','family':f'family{i}'} for i in range(16)]
    packet.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    monkeypatch.setattr(tx,'DEV_PACKET_SHA256',native_async.file_binding(packet)['sha256'])
    monkeypatch.setattr(native_async,'verify_request',lambda request:None)
    indices=list(range(16));refs=[]
    for label,source,success in [('parent','a'*64,10),('candidate','b'*64,11)]:
        common={'source_training_checkpoint_sha256':source,'checkpoint':{'sha256':label},
            'packet':native_async.file_binding(packet),'case_indices':indices,'evaluation_spec_sha256':'c'*64,
            'runtime':{'sha256':'d'*64},'transported_actor':{'source_sha256':source}}
        request={'evaluation_provenance':common,'spec':{'runtime':'/frozen','packet':{'path':str(packet)},'cases':indices}}
        rp=tmp_path/(label+'-request.json');rp.write_text(json.dumps(request));rr=native_async.file_binding(rp)
        detail=[{**row,'outcome':'task_success' if i<success else 'semantic_mismatch'} for i,row in enumerate(rows)]
        identity={'checkpoint_sha256':source}
        score={'historical':True,'qualifies_current_weights':False,'eligible':True,'cases':16,
            'packet_sha256':tx.DEV_PACKET_SHA256,'case_indices':indices,'cases_detail':detail,'task_successes':success,
            'outcome_counts':{'task_success':success,'semantic_mismatch':16-success,'infrastructure_error':0,'unresolved':0},
            'evaluation_provenance':{**common,'request_sha256':rr['sha256'],'native_identity':identity,
                'actor_policy_binding':{'native_identity':identity,'python_runtime':{'sha256':'e'*64}},
                'serving_config':{'geometry':'same'},'bf16_reduction':{'declared':'fp32-accumulate','reduced_enabled':False}}}
        sp=tmp_path/(label+'-score.json');sp.write_text(json.dumps(score));refs.append((native_async.file_binding(sp),rr))
    result=native_async.compare_transported_development(*refs[0],*refs[1],parent_source_sha256='a'*64,candidate_source_sha256='b'*64)
    assert result['parent_successes']==10 and result['candidate_successes']==11 and len(result['case_outcome_changes'])==1
    candidate=Path(refs[1][0]['path']);score=json.loads(candidate.read_text());score['cases_detail'][-1]['id']='foreign';candidate.write_text(json.dumps(score))
    with pytest.raises(ValueError,match='16 declared'):
        native_async.compare_transported_development(*refs[0],native_async.file_binding(candidate),refs[1][1],parent_source_sha256='a'*64,candidate_source_sha256='b'*64)


def test_completed_request_keeps_exact_small_receipt_after_snapshot_cleanup(tmp_path,monkeypatch):
    view,actor,spec,base=setup(tmp_path,monkeypatch)
    active=native_async.seal_transport_request(view,step=7,spec=spec,out=tmp_path/'out',base=base,
        queue=tmp_path/'queue',transported_actor=actor,device='cpu')
    request_dir=Path(active['request']);raw=(request_dir/'request.json').read_bytes()
    from hashlib import sha256
    digest=sha256(raw).hexdigest()
    class Completed:
        def __init__(self,*a,**kw):pass
        def landed(self,step):return {'evaluation_provenance':{'source_checkpoint_sha256':active['checkpoint_sha256'],
                                                               'request_sha256':digest}}
    monkeypatch.setattr(task_cadence,'TaskEval',Completed)
    task_cadence.run_request(request_dir)
    history=Path(active['stage_out'])/'task-eval/sealed-requests'/f'{digest}.json'
    assert history.read_bytes()==raw and not request_dir.exists() and view.read_bytes()==b'actual compact actor'


def test_prepared_sft_window_retains_existing_final_gate_and_marks_monitor_pending(tmp_path,monkeypatch):
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'scripts/neuralese_data/prepare_pop_async_sft_controls.py'
    module_spec=importlib.util.spec_from_file_location('pop_sft_controls',path)
    controls=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(controls)
    packet=tmp_path/'packet';packet.write_text(''.join(json.dumps({'id':f'case{i}','family':f'family{i}'})+'\n' for i in range(16)))
    monkeypatch.setattr(controls,'DEV_PACKET_SHA256',native_async.file_binding(packet)['sha256'])
    monitor={'packet':{'path':str(packet),'sha256':controls.DEV_PACKET_SHA256},'cases':list(range(16)),
             'prompt_rendering':'full','neuralese_server':True}
    spec=tmp_path/'monitor.json';spec.write_text(json.dumps(monitor))
    config=tmp_path/'config.json';config.write_text(json.dumps({'native_state':True,
        'trainer_parameters':{'task_eval':'existing-final-proof','epochs':3,'refresh_after_updates':1}}))
    receipt=controls.prepare(config,spec,'a'*64,tmp_path/'prepared')
    actual=json.loads(Path(receipt['controller_config']['path']).read_text())['trainer_parameters']
    assert actual=={'task_eval':'existing-final-proof','epochs':3,'refresh_after_updates':16,
                    'refresh_at_checkpoint':True,'task_eval_every':0}
    assert receipt['monitor_status']=='pending' and receipt['qualifies_current_weights'] is False
    bad=tmp_path/'bad.json';bad.write_text(json.dumps({'native_state':True,'trainer_parameters':{}}))
    with pytest.raises(ValueError):controls.prepare(bad,spec,'a'*64,tmp_path/'bad-out')
