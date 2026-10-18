"""Idle ingestion never owns transfer memory; interrupted owners rejoin one worker."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from natlang_neuralese.eval import native_async_service as service
from test_native_async_service import config, dump


def ledger_config(tmp_path):
    c=config(tmp_path)
    c['transfer_ledger']=dict(budget_gib=28,reserve_gib=1)
    return c


@pytest.mark.parametrize('landed,exists,publishing,waiting',[
    (False,True,False,True),(True,True,False,True),
    (True,False,False,False),(False,True,True,False)])
def test_pending_watch_preserves_exact_draining_conditions(tmp_path,monkeypatch,landed,exists,publishing,waiting):
    from natlang_neuralese.eval import task_cadence
    c=ledger_config(tmp_path);request=tmp_path/'request'
    if exists:request.mkdir()
    active=dict(step=7,spec=dict(prompt_rendering='full'),stage_out=str(tmp_path/'stage'),request=str(request),publication_pending=publishing)
    state=dump(Path(c['out'])/'native-async-state.json',dict(active=active));before=state.read_bytes()
    monkeypatch.setattr(task_cadence,'TaskEval',lambda *a,**kw:SimpleNamespace(landed=lambda step: {'success':True}if landed else None))
    event=service.pending_event(c)
    assert (event is not None)==waiting
    assert state.read_bytes()==before


def test_completed_unchanged_export_does_not_reserve_transfer(tmp_path):
    c=ledger_config(tmp_path)
    dump(Path(c['out'])/'native-async-state.json',dict(last_completed_sha256='f'*64))
    assert service.pending_event(c)==dict(event='native_eval_up_to_date',step=10922)
    document=json.loads(Path(c['export_state']).read_text());document['receipt']['identity']['sha256']='e'*64
    document['receipt']['independent_verification']['sha256']='e'*64
    dump(Path(c['export_state']),document)
    assert service.pending_event(c)is None


def test_small_service_wait_does_not_launch_or_poll_heavy_path(tmp_path,monkeypatch):
    c=ledger_config(tmp_path)
    monkeypatch.setattr(service,'pending_event',lambda _:dict(event='native_eval_pending',active_step=7))
    monkeypatch.setattr(service,'ledgered_poll',lambda _:pytest.fail('idle wait must not reserve transfer memory'))
    monkeypatch.setattr(service,'poll',lambda _:pytest.fail('idle wait must not decode checkpoints'))
    service.run(c,max_polls=1)
    assert json.loads((Path(c['status'])/'status.json').read_text())['active_step']==7


def test_worker_runs_original_config_and_binds_result(tmp_path,monkeypatch):
    c=ledger_config(tmp_path);path=tmp_path/'worker.json';result=tmp_path/'result.json';Path(c['status']).mkdir()
    document=dict(configuration={k:v for k,v in c.items()if k!='transfer_ledger'},result=str(result),token='one',mechanism=service.binding(service.__file__))
    dump(path,document);seen=[]
    monkeypatch.setattr(service,'poll',lambda value:seen.append(value)or dict(event='native_eval_enqueued',step=7))
    service.worker_poll(path,service.binding(path)['sha256'])
    assert seen==[document['configuration']]
    assert json.loads(result.read_text())==dict(token='one',event=dict(event='native_eval_enqueued',step=7),
        configuration=service.binding(path),mechanism=document['mechanism'])
    path.write_text('{}')
    with pytest.raises(ValueError,match='hash differs'):service.worker_poll(path,'0'*64)


@pytest.mark.parametrize('when',['before','during'])
def test_worker_rejects_service_code_changed_after_admission(tmp_path,monkeypatch,when):
    c=ledger_config(tmp_path);Path(c['status']).mkdir();result=tmp_path/'result.json';path=tmp_path/'worker.json'
    mechanism=service.binding(service.__file__)
    dump(path,dict(configuration=c,result=str(result),token='one',mechanism=mechanism))
    calls=[];actual_binding=service.binding
    def binding(target):
        value=actual_binding(target)
        if str(target)==service.__file__ and(when=='before'or calls):value={**value,'sha256':'0'*64}
        return value
    monkeypatch.setattr(service,'binding',binding)
    monkeypatch.setattr(service,'poll',lambda _:calls.append(True)or dict(event='native_eval_enqueued'))
    with pytest.raises(ValueError,match='mechanism changed'):service.worker_poll(path,actual_binding(path)['sha256'])
    assert len(calls)==(when=='during')
    assert not result.exists()


def write_worker_result(c,event):
    status=Path(c['status']);document=json.loads((status/'ledger-poll-config.json').read_text())
    dump(Path(document['result']),dict(token=document['token'],event=event,
        configuration=service.binding(status/'ledger-poll-config.json'),mechanism=document['mechanism']))


def test_actual_admission_command_and_release_after_result(tmp_path):
    c=ledger_config(tmp_path);calls=[];event=dict(event='native_eval_enqueued',step=7)
    states=iter([dict(ActiveState='inactive'),dict(ActiveState='active'),dict(ActiveState='inactive',Result='success',ExecMainStatus='0')])
    def execute(argv,**kwargs):
        calls.append(argv);write_worker_result(c,event)
    assert service.ledgered_poll(c,execute=execute,inspect=lambda _:next(states),sleep=lambda _:None)==event
    argv=calls[0]
    assert argv[argv.index('--budget-gb')+1]=='28'
    assert '--resumable'in argv and argv[argv.index('--priority')+1]=='low'
    assert argv[argv.index('-m')+1]=='natlang_neuralese.eval.native_async_service'
    assert '--fetch-command'not in argv  # original exact argv remains in bound config
    assert not(Path(c['status'])/'ledger-poll.json').exists()


class Interrupted(BaseException):pass


def journal_intent(c):
    def interrupted(*args,**kwargs):raise Interrupted()
    with pytest.raises(Interrupted):service.ledgered_poll(c,execute=interrupted,inspect=lambda _:dict(ActiveState='inactive'))
    return json.loads((Path(c['status'])/'ledger-poll.json').read_text())


def test_supervisor_restart_rejoins_running_worker_without_second_launch(tmp_path):
    c=ledger_config(tmp_path);intent=journal_intent(c)
    states=iter([dict(ActiveState='active'),dict(ActiveState='inactive',Result='success',ExecMainStatus='0')])
    event=dict(event='native_eval_pending',active_step=7)
    def sleep(_):write_worker_result(c,event)
    assert service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate worker'),inspect=lambda _:next(states),sleep=sleep)==event
    assert intent['token']==json.loads((Path(c['status'])/'ledger-poll-result.json').read_text())['token']


def test_restart_after_finished_result_does_not_launch_again(tmp_path):
    c=ledger_config(tmp_path);journal_intent(c);event=dict(event='native_eval_enqueued')
    write_worker_result(c,event)
    assert service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate worker'),inspect=lambda _:dict(ActiveState='inactive',Result='success',ExecMainStatus='0'))==event


def test_completed_quality_is_published_while_next_transfer_still_runs(tmp_path):
    c=ledger_config(tmp_path);journal_intent(c)
    historical=dict(step=20515,checkpoint_sha256='a'*64,historical=True,qualifies_current_weights=False,
        result=dict(task_successes=10,complete=False))
    dump(Path(c['out'])/'latest-result.json',historical)
    dump(Path(c['status'])/'latest-result.json',dict(step=20513))
    event=dict(event='native_eval_enqueued',step=21477)
    states=iter([dict(ActiveState='active'),dict(ActiveState='active'),
        dict(ActiveState='inactive',Result='success',ExecMainStatus='0')])
    calls=[]
    def sleep(_):
        assert json.loads((Path(c['status'])/'latest-result.json').read_text())==historical
        calls.append(True)
        if len(calls)==1:
            historical['step']=20516
            dump(Path(c['out'])/'latest-result.json',historical)
        else:write_worker_result(c,event)
    assert service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate transfer'),
        inspect=lambda _:next(states),sleep=sleep)==event
    assert len(calls)==2
    assert json.loads((Path(c['status'])/'latest-result.json').read_text())==historical


@pytest.mark.parametrize('corrupt',[b'{"step":',b'\xff'])
def test_valid_durable_quality_repairs_corrupt_status_mirror(tmp_path,corrupt):
    c=ledger_config(tmp_path);Path(c['status']).mkdir()
    target=Path(c['status'])/'latest-result.json';target.write_bytes(corrupt)
    value=dict(step=20515,historical=True,qualifies_current_weights=False)
    dump(Path(c['out'])/'latest-result.json',value)
    service._publish_latest_result(c)
    assert json.loads(target.read_text())==value


def test_corrupt_canonical_quality_never_overwrites_good_status(tmp_path):
    c=ledger_config(tmp_path);Path(c['out']).mkdir()
    (Path(c['out'])/'latest-result.json').write_text('{')
    target=dump(Path(c['status'])/'latest-result.json',dict(step=20513));before=target.read_bytes()
    with pytest.raises(json.JSONDecodeError):service._publish_latest_result(c)
    assert target.read_bytes()==before


def test_rejoins_preempted_worker_and_waits_through_relaunch_gap(tmp_path):
    c=ledger_config(tmp_path);journal_intent(c);event=dict(event='native_eval_pending',active_step=7)
    # Queue custody precedes termination; claim custody precedes relaunch.
    states=iter([dict(ActiveState='failed',queued=True),dict(ActiveState='inactive',launch_pending=True),
        dict(ActiveState='active'),dict(ActiveState='inactive',Result='success',ExecMainStatus='0')])
    sleeps=[]
    def sleep(_):sleeps.append(True);write_worker_result(c,event)
    assert service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate queued worker'),
        inspect=lambda _:next(states),sleep=sleep)==event
    assert len(sleeps)==3


@pytest.mark.parametrize('key',['queued_argv','claimed_argv'])
def test_rejects_foreign_queued_or_launching_owner(tmp_path,key):
    c=ledger_config(tmp_path);journal_intent(c)
    with pytest.raises(ValueError,match='argv differs'):
        service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('foreign worker'),
            inspect=lambda _:dict(ActiveState='inactive',**{key:['python','foreign.py']}))


def test_run_consumes_finished_journal_before_pending_shortcut(tmp_path,monkeypatch):
    c=ledger_config(tmp_path);journal_intent(c);event=dict(event='native_eval_enqueued',step=7)
    write_worker_result(c,event)
    original=service.ledgered_poll
    monkeypatch.setattr(service,'pending_event',lambda _:pytest.fail('must reconcile prior worker first'))
    monkeypatch.setattr(service,'ledgered_poll',lambda c:original(c,
        execute=lambda *a,**kw:pytest.fail('duplicate worker'),inspect=lambda _:dict(ActiveState='inactive',Result='success',ExecMainStatus='0')))
    service.run(c,max_polls=1)
    assert not(Path(c['status'])/'ledger-poll.json').exists()


def test_finished_worker_reconciles_after_service_code_upgrade(tmp_path,monkeypatch):
    c=ledger_config(tmp_path);journal_intent(c);event=dict(event='native_eval_enqueued',step=7)
    write_worker_result(c,event);actual_binding=service.binding
    monkeypatch.setattr(service,'binding',lambda p:{**actual_binding(p),'sha256':'0'*64} if str(p)==service.__file__ else actual_binding(p))
    assert service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate worker'),
        inspect=lambda _:dict(ActiveState='inactive',Result='success',ExecMainStatus='0'))==event


@pytest.mark.parametrize('change',['config','options','argv'])
def test_recovery_rejects_changed_owner_or_configuration(tmp_path,change):
    c=ledger_config(tmp_path);journal_intent(c)
    if change=='config':(Path(c['status'])/'ledger-poll-config.json').write_text('{}')
    if change=='options':c['transfer_ledger']['budget_gib']=1
    state=dict(ActiveState='active',argv=['python','unrelated.py'])if change=='argv'else dict(ActiveState='inactive')
    with pytest.raises(ValueError,match='configuration changed|argv differs'):
        service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('unauthenticated launch'),inspect=lambda _:state)


def test_worker_failure_or_stale_result_does_not_report_success(tmp_path):
    c=ledger_config(tmp_path);journal_intent(c)
    dump(Path(c['status'])/'ledger-poll-result.json',dict(token='old',event=dict(event='native_eval_enqueued')))
    states=iter([dict(ActiveState='active'),dict(ActiveState='failed',Result='oom-kill',ExecMainStatus='9')])
    with pytest.raises(RuntimeError,match='worker failed'):
        service.ledgered_poll(c,execute=lambda *a,**kw:pytest.fail('duplicate launch'),inspect=lambda _:next(states),sleep=lambda _:None)
    assert not(Path(c['status'])/'ledger-poll.json').exists()
