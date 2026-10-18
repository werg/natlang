import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from natlang_neuralese.eval import native_async_service as service
from scripts.task_eval_dispatch import unit_name


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def dependency(tmp_path):
    request_dir=tmp_path/'request';request_dir.mkdir();weights=request_dir/'weights.pt';weights.write_bytes(b'exact-view')
    spec=dict(architecture='mellum')
    request=dict(native_state=True,step=16364,stage_out=str(tmp_path/'stage'),weights=str(weights),
        evaluation_provenance=dict(model_config=spec,checkpoint={**service.binding(weights),'sha256':'a'*64},source_checkpoint_sha256='a'*64))
    active=dict(step=16364,stage_out=request['stage_out'],request=str(request_dir),checkpoint_sha256='a'*64)
    state=dump(tmp_path/'state.json',dict(active=active));dump(request_dir/'request.json',request)
    dump(request_dir/'dispatch.json',dict(unit=unit_name(request_dir,request),placement='dgx-ledger'))
    return state,request_dir,request


def processes(unit,directory):return [dict(pid=123,start_ticks='456',boot_id='boot',argv=['python','-m','natlang_neuralese.eval.task_cadence','--async-request',str(directory)])]


def test_gate_waits_for_real_process_then_persists_exact_history(tmp_path):
    state,directory,request=dependency(tmp_path);requirement=dict(state=str(state),minimum_step=16364,architecture='mellum');status=tmp_path/'status';status.mkdir()
    def observe(*args,**kwargs):return service.dispatch_evidence(*args,**kwargs,verify=lambda r:None,processes=lambda *_:[])
    assert not service.gate(status,requirement,observe=observe)
    def observe(*args,**kwargs):return service.dispatch_evidence(*args,**kwargs,verify=lambda r:None,processes=processes)
    assert service.gate(status,requirement,observe=observe)
    proof=json.loads((status/'after-dispatch-satisfied.json').read_text());assert proof['evidence']['checkpoint_sha256']=='a'*64
    assert proof['evidence']['request']==service.binding(directory/'request.json')
    state.unlink();assert service.gate(status,requirement,observe=lambda *_:pytest.fail('historical gate must not require live request'))
    with pytest.raises(ValueError,match='historical'):service.gate(status,{**requirement,'minimum_step':16365})


@pytest.mark.parametrize('change',['sha','step','architecture','unit','weights'])
def test_mismatched_dependency_rejected(tmp_path,change):
    state,directory,request=dependency(tmp_path)
    if change=='sha':request['evaluation_provenance']['source_checkpoint_sha256']='b'*64
    if change=='step':request['step']+=1
    if change=='architecture':request['evaluation_provenance']['model_config']['architecture']='lfm2'
    if change=='weights':request['weights']=str(tmp_path/'other.pt')
    if change=='unit':dump(directory/'dispatch.json',dict(unit='some-other-active-unit',placement='dgx-ledger'))
    else:dump(directory/'request.json',request)
    with pytest.raises(ValueError):service.dispatch_evidence(state,minimum_step=16364,architecture='mellum',verify=lambda r:None,processes=processes)


def test_undispatched_or_older_state_waits(tmp_path):
    assert service.dispatch_evidence(tmp_path/'missing')is None
    state,directory,request=dependency(tmp_path)
    assert service.dispatch_evidence(state,minimum_step=16365)is None
    (directory/'dispatch.json').unlink();assert service.dispatch_evidence(state)is None


def test_observation_rechecks_source_bytes(tmp_path):
    state,directory,request=dependency(tmp_path)
    def changed(unit,path):
        dump(state,dict(active=None));return processes(unit,path)
    with pytest.raises(ValueError,match='changed'):service.dispatch_evidence(state,verify=lambda r:None,processes=changed)


def config(tmp_path):
    receipt=dict(role='lfm',private=True,identity=dict(sha256='f'*64,bytes=100),independent_verification=dict(sha256='f'*64,bytes=100))
    state=dump(tmp_path/'export.json',dict(last_verified_step=10922,receipt=receipt))
    spec=dump(tmp_path/'spec.json',dict(cases=list(range(0,48,3)),neuralese_server=True,noise_repeats=1,
        packet=dict(path='held.jsonl',sha256='5'*64),prompt_rendering='full',temperature=0,case_seconds=7200,max_model_requests=128,max_turns=160))
    return dict(status=str(tmp_path/'status'),out=str(tmp_path/'out'),export_state=str(state),spec=str(spec),minimum_step=10922,role='lfm',disk_reserve_bytes=200,
        base='actual-base',queue='same-queue',device='cuda',fetch_command=['sdk-python','fetch_native_eval_export.py','--receipt','{receipt}','--out','{out}'],after_dispatch=None)


def test_priority_hold_never_reads_export_or_fetches(tmp_path):
    c=config(tmp_path);c['after_dispatch']=dict(state='not-dispatched',minimum_step=16364,architecture='mellum');Path(c['export_state']).unlink()
    assert service.poll(c,observe=lambda *a,**k:None,ingest=lambda *a,**k:pytest.fail('must not ingest'))['event']=='native_eval_waiting_for_prior_dispatch'


def test_original_fetch_spec_and_active_work_unchanged(tmp_path):
    c=config(tmp_path);out=Path(c['out']);out.mkdir();state=dump(out/'native-async-state.json',dict(active=dict(step=10659,spec=dict(noise_repeats=1))));before=state.read_bytes();called=[]
    def ingest(export,**kwargs):called.append((export,kwargs));return dict(event='native_eval_pending',active_step=10659)
    result=service.poll(c,ingest=ingest,available=lambda _:pytest.fail('active request requires no new transfer capacity'))
    assert result['active_step']==10659 and state.read_bytes()==before
    kwargs=called[0][1];assert kwargs['fetch_command']==c['fetch_command'] and kwargs['queue']=='same-queue'and kwargs['base']=='actual-base'
    assert kwargs['spec']['cases']==list(range(0,48,3))and kwargs['spec']['case_seconds']==7200 and kwargs['spec']['max_model_requests']==128 and kwargs['spec']['max_turns']==160 and kwargs['spec']['noise_repeats']==1


def test_disk_guard_preserves_partial_fetch(tmp_path):
    c=config(tmp_path)
    assert service.poll(c,ingest=lambda *a,**k:pytest.fail('must wait'),available=lambda _:SimpleNamespace(free=299))['event']=='native_eval_disk_wait'
    out=Path(c['out']);pending=out/'incoming'/('checkpoint-'+'f'*64+'.pt.pending');pending.parent.mkdir();pending.write_bytes(b'x'*50)
    assert service.poll(c,ingest=lambda *a,**k:dict(event='native_eval_enqueued'),available=lambda _:SimpleNamespace(free=250))['event']=='native_eval_enqueued'
    assert pending.read_bytes()==b'x'*50


def test_disk_guard_uses_pinned_transfer_instead_of_newer_export(tmp_path):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    c=config(tmp_path);out=Path(c['out']);out.mkdir()
    selected=json.loads(Path(c['export_state']).read_text())
    dump(out/'native-async-state.json',dict(fetching_export=dict(
        schema='natlang.native-eval-transfer/1',export=selected,
        sha256=canonical_json_sha256_hex(selected))))
    incoming=out/'incoming';incoming.mkdir()
    (incoming/('checkpoint-'+'f'*64+'.pt.pending')).write_bytes(b'x'*50)
    newer=json.loads(Path(c['export_state']).read_text());newer['last_verified_step']+=1
    newer['receipt']['identity']={'sha256':'e'*64,'bytes':100000}
    newer['receipt']['independent_verification']=newer['receipt']['identity']
    dump(Path(c['export_state']),newer)
    result=service.poll(c,ingest=lambda *a,**kw:dict(event='native_eval_enqueued'),
                        available=lambda _:SimpleNamespace(free=250))
    assert result['event']=='native_eval_enqueued'


def test_first_transfer_rechecks_larger_export_advanced_during_disk_guard(tmp_path,monkeypatch):
    from natlang_neuralese.eval import native_async
    c=config(tmp_path);calls=[]
    def available(_):
        calls.append(True)
        if len(calls)==1:
            newer=json.loads(Path(c['export_state']).read_text());newer['last_verified_step']+=1
            newer['receipt']['identity']={'sha256':'e'*64,'bytes':100000}
            newer['receipt']['independent_verification']=newer['receipt']['identity']
            dump(Path(c['export_state']),newer)
        return SimpleNamespace(free=300)
    monkeypatch.setattr(native_async.subprocess,'run',lambda *a,**k:pytest.fail('unadmitted transfer'))
    result=service.poll(c,available=available)
    assert result==dict(event='native_eval_disk_wait',available_bytes=300,needed_bytes=100200)
    assert len(calls)==2
    assert not(Path(c['out'])/'native-async-state.json').exists()


@pytest.mark.parametrize('state,request_arg,active,expected',[
    ('S','correct',True,True),('R','foreign',True,False),('Z','correct',True,False),('S','correct',False,False)])
def test_actual_unit_process_reader_rejects_foreign_or_dead_process(tmp_path,state,request_arg,active,expected):
    proc=tmp_path/'proc';groups=tmp_path/'groups';group=groups/'actual-unit';group.mkdir(parents=True)
    (group/'cgroup.procs').write_text('123\n');(proc/'sys/kernel/random').mkdir(parents=True);(proc/'sys/kernel/random/boot_id').write_text('exact-boot')
    directory=proc/'123';directory.mkdir();tail=[state,'1']+['0']*17+['456']+['0']*5
    (directory/'stat').write_text('123 (python) '+' '.join(tail))
    target=tmp_path/'request';arg=target if request_arg=='correct'else tmp_path/'other'
    argv=['python','-m','natlang_neuralese.eval.task_cadence','--async-request',str(arg)]
    (directory/'cmdline').write_bytes(b'\0'.join(x.encode()for x in argv)+b'\0')
    systemctl=lambda *a,**k:SimpleNamespace(stdout='ActiveState='+('active'if active else'inactive')+'\nControlGroup=/actual-unit\n')
    observed=service.running_request_processes('actual-unit',target,proc=proc,cgroups=groups,systemctl=systemctl)
    assert bool(observed)is expected
    if expected:assert observed[0]['start_ticks']=='456'and observed[0]['boot_id']=='exact-boot'and observed[0]['argv']==argv


@pytest.mark.parametrize('changed',['unit','architecture','state_path','process_request'])
def test_historical_gate_cannot_relabel_observed_other_work(tmp_path,changed):
    state,directory,_=dependency(tmp_path);requirement=dict(state=str(state),minimum_step=16364,architecture='mellum');status=tmp_path/'status';status.mkdir()
    observe=lambda *a,**k:service.dispatch_evidence(*a,**k,verify=lambda r:None,processes=processes)
    assert service.gate(status,requirement,observe=observe)
    path=status/'after-dispatch-satisfied.json';proof=json.loads(path.read_text())
    if changed=='unit':proof['evidence']['unit']='a-stale-active-unit'
    if changed=='architecture':proof['evidence']['architecture']='lfm2'
    if changed=='state_path':proof['evidence']['state']['path']=str(tmp_path/'another-state')
    if changed=='process_request':proof['evidence']['processes'][0]['argv'][-1]=str(tmp_path/'another-request')
    dump(path,proof)
    with pytest.raises(ValueError,match='historical'):service.gate(status,requirement)


def test_shell_wrapper_arguments_are_not_an_executing_python_evaluator(tmp_path):
    directory=tmp_path/'request'
    assert service._executes_request(['python','-m','natlang_neuralese.eval.task_cadence','--async-request',str(directory)],directory)
    assert not service._executes_request(['sh','-c','exec "$@"','python','-m','natlang_neuralese.eval.task_cadence','--async-request',str(directory)],directory)


def test_sealed_owned_weights_need_only_existing_reserve_before_recovery(tmp_path):
    c=config(tmp_path)
    owned=Path(c['out'])/'checkpoints'/('f'*64)/'task-eval/async/step-0010922/weights.pt'
    owned.parent.mkdir(parents=True);owned.write_bytes(b'x'*100)
    assert service.poll(c,ingest=lambda *a,**kw:dict(event='native_eval_enqueued'),
                        available=lambda _:SimpleNamespace(free=200))['event']=='native_eval_enqueued'


def test_service_publishes_prior_landed_result_before_blocking_transfer(tmp_path,monkeypatch):
    c=config(tmp_path);result={'step':20513,'historical':True,'result':{'task_successes':12}}
    dump(Path(c['out'])/'latest-result.json',result)
    def blocking_poll(config):
        assert json.loads((Path(config['status'])/'latest-result.json').read_text())==result
        return dict(event='native_eval_up_to_date')
    monkeypatch.setattr(service,'poll',blocking_poll)
    service.run(c,max_polls=1)


def test_current_request_progress_does_not_depend_on_latest_export_availability(tmp_path):
    c=config(tmp_path)
    dump(Path(c['out'])/'native-async-state.json',dict(active=dict(step=10922)))
    Path(c['export_state']).unlink()
    assert service.poll(c,ingest=lambda *a,**kw:dict(event='native_eval_pending',active_step=10922),
        available=lambda _:pytest.fail('no new transfer is requested'))['active_step']==10922
