"""Earned owned quanta return to the existing scheduler with exact crash retry."""
import copy
import json
from pathlib import Path

import pytest
from test_native_ordinary_runtime_handoff import ordinary
from test_native_bare_runtime_handoff import bare
from test_native_later_runtime_handoff import later
from natlang_neuralese.rl import native_raw_rounds as raw
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def restore_observer(tmp_path,monkeypatch):
    # A pinned harmless source fixture exercises the verifier's exact-byte gate;
    # production accepts only the separately reviewed observer SHA.
    observer=tmp_path/'observer.py';observer.write_text('# fixed read-only observer fixture\n')
    monkeypatch.setattr(raw,'_OWNED_RESTORE_OBSERVER_SHA256',resident.binding(observer)['sha256'])
    parent_path=tmp_path/'parent.pt';parent_path.write_bytes(b'actual immutable parent')
    parent=resident.binding(parent_path)
    command=['python','-u','-m','natlang_neuralese.train.native_state_sft',
             '--continue-from',parent['path'],'--out',str(tmp_path/'learner'),
             '--stage-id','sft','--recipe-binding','{}']
    original=tmp_path/'original-argv.json';write(original,command)
    argv=['python','-u',str(observer),'--argv-binding',json.dumps(resident.binding(original)),
          '--parent',json.dumps({**parent,'bytes':parent_path.stat().st_size}),
          '--runtime-root',str(tmp_path/'runtime'),'--out',str(tmp_path/'observations')]
    return argv,parent,command,observer,original


def test_exact_read_only_restore_observer_resolves_bound_trainer_command(restore_observer):
    argv,parent,command,observer,original=restore_observer
    result,evidence=raw._owned_training_invocation(argv,parent)
    assert result==command
    assert evidence['observer']==resident.binding(observer)
    assert evidence['original_argv']==resident.binding(original)
    direct=['python','-u','-m','natlang_neuralese.train.native_state_sft','--out','literal']
    assert raw._owned_training_invocation(direct,parent)==(direct,None)


@pytest.mark.parametrize('tamper',['source','original','parent','size','duplicate','reordered','module','restore-path'])
def test_restore_observer_cannot_borrow_another_invocation(restore_observer,tamper):
    argv,parent,command,observer,original=restore_observer
    if tamper=='source':observer.write_text('# altered observer\n')
    elif tamper=='original':original.write_text('[]')
    elif tamper in('parent','size'):
        value=json.loads(argv[6]);value['sha256'if tamper=='parent'else'bytes']='0'*64 if tamper=='parent'else 1;argv[6]=json.dumps(value)
    elif tamper=='duplicate':argv+=['--argv-binding',argv[4]]
    elif tamper=='reordered':argv[5],argv[7]=argv[7],argv[5]
    else:
        if tamper=='module':command[3]='untrusted.module'
        else:command[command.index('--continue-from')+1]=str(original)
        write(original,command);argv[4]=json.dumps(resident.binding(original))
    with pytest.raises((ValueError,AssertionError)):
        raw._owned_training_invocation(argv,parent)


@pytest.fixture
def owned(ordinary,tmp_path):
    x=ordinary;continuation=json.loads(x['owner_continuation'].read_text())
    state_path=x['out']/'state.json';state=json.loads(state_path.read_text())
    state.update(checkpoint=continuation['parent']['path'],phase='collect');write(state_path,state)
    cursor_path=x['root']/'cycle.json';cursor=json.loads(cursor_path.read_text())
    cursor.update(cycle=1,phase='rl',next_stage=0,readiness={'stale':True});write(cursor_path,cursor)
    receipt=x['slot'].parent/'native-training-receipt.json'
    write(receipt,{'schema':'natlang.native-training-readiness/1',
        'scope':'CPU protocol fixture; no production GPU qualification claim',
        'model_id':json.loads(x['config'].read_text())['raw_model_id'],'stage_id':'sft',
        'recipe':x['saved']['training']['options']['recipe_binding'],
        'checkpoint':resident.binding(x['slot']),
        'native_identity':{'checkpoint_sha256':resident.binding(x['slot'])['sha256']},
        'continuation':{'checkpoint_sha256':continuation['parent']['sha256']},
        'continuation_boundary':{'new_optimizer_updates':3,'checkpoint_sha256':resident.binding(x['slot'])['sha256']},
        'completed':False,'foundation_ready':False,'task_eval':{'success':0}})
    x.update(operations=[{'handoff':x['handoff'],'receipt':resident.binding(receipt)}],
             reconciliation=tmp_path/'owned-reconciliation.json',operation_receipt=receipt)
    return x


def invoke(x):
    return raw.reconcile_owned_operations(x['config'],x['operations'],x['bare']['recipe'],
        x['reconciliation'],bare_direct_owner=x['bare'])


def test_earned_quantum_publishes_current_optimizer_without_marking_stage_complete(owned):
    x=owned;original_config=x['config'].read_bytes();pin=x['pin'].read_bytes()
    original=json.loads((x['out']/'state.json').read_text())
    result=invoke(x);value=json.loads(Path(result['path']).read_text())
    state=json.loads((x['out']/'state.json').read_text());cursor=json.loads((x['root']/'cycle.json').read_text())
    assert state['checkpoint']==str(x['pin']) and state['actor_policy_pending_load']==resident.binding(x['pin'])['sha256']
    assert all(state[key]==original[key]for key in('round','trained_rounds','selection'))
    assert cursor['cycle']==1 and cursor['phase']=='sft'and cursor['next_stage']==1
    assert 'pending_rl_round'not in cursor and'readiness'not in cursor
    assert cursor['unknown_cursor']=={'debt':[4,5]}
    assert cursor['owned_operation_reconciliation']['runtime_configuration_required']is True
    assert value['whole_recipe_completed']is False
    assert x['config'].read_bytes()==original_config and x['pin'].read_bytes()==pin
    # A completed retry is historical success, never a rewind of later progress.
    state['later_earned_progress']={'retained':True};write(x['out']/'state.json',state)
    assert invoke(x)==result
    assert json.loads((x['out']/'state.json').read_text())['later_earned_progress']=={'retained':True}


def test_crash_after_state_publication_retries_only_its_exact_cursor_successor(owned,monkeypatch):
    x=owned;real=raw.write_bytes_durable;cursor=x['root']/'cycle.json'
    def fail_cursor(path,data):
        if Path(path)==cursor:raise OSError('simulated crash before cursor publication')
        return real(path,data)
    with monkeypatch.context()as m:
        m.setattr(raw,'write_bytes_durable',fail_cursor)
        with pytest.raises(OSError):invoke(x)
    assert not x['reconciliation'].exists()
    assert json.loads((x['out']/'state.json').read_text())['checkpoint']==str(x['pin'])
    result=invoke(x)
    assert json.loads(Path(result['path']).read_text())['acknowledged_finite_stages']==['sft']


@pytest.mark.parametrize('change',['parent','zero-updates','receipt-model','unearned-policy','active-operation','wrong-recipe','lost-stage'])
def test_unknown_or_unearned_progress_cannot_be_reconciled(owned,change,tmp_path):
    x=owned;state_path=x['out']/'state.json';before=state_path.read_bytes();cursor=x['root']/'cycle.json'
    if change in('unearned-policy','active-operation'):
        state=json.loads(state_path.read_text())
        if change=='unearned-policy':state['round']=0
        else:state['phase']='learner_pending'
        write(state_path,state)
    elif change=='wrong-recipe':x['bare']['recipe']=resident.binding(x['config'])
    elif change=='lost-stage':
        value=json.loads(cursor.read_text());value['next_stage']=99;write(cursor,value)
    else:
        value=json.loads(x['operation_receipt'].read_text())
        if change=='parent':value['continuation']['checkpoint_sha256']='0'*64
        elif change=='zero-updates':value['continuation_boundary']['new_optimizer_updates']=0
        else:value['model_id']='another-model'
        write(x['operation_receipt'],value);x['operations'][0]['receipt']=resident.binding(x['operation_receipt'])
    protected={p:p.read_bytes()for p in(state_path,cursor,x['config'],x['pin'])}
    with pytest.raises((ValueError,KeyError)):invoke(x)
    assert all(p.read_bytes()==data for p,data in protected.items())


def test_existing_scheduler_inventory_and_acknowledgement_map_stage_ids_not_old_position():
    foundation={'id':'foundation','kind':'native_state_sft','parameters':{'epochs':1}}
    preference={'id':'dpo','kind':'native_offline_preference','parameters':{'epochs':1}}
    # Use the actual mandatory kind from the production registry.
    from natlang_neuralese.rl.native_supervised import MANDATORY_STAGE_KINDS
    assert foundation['kind']in MANDATORY_STAGE_KINDS
    stages=raw._supervised_stage_inventory({'stages':[foundation,preference,
        {'id':'future','kind':'native_offline_preference','admitted':False}]},'foundation')
    assert [stage['id']for stage in stages]==['dpo','foundation']
    chain=[{'stage_id':stage['id'],'stage':stage}for stage in (foundation,preference,foundation,foundation)]
    assert raw._acknowledged_owned_stage_ids(chain,stages)==['dpo','foundation']
    for incomplete in([chain[0]],chain[:2],[chain[0],chain[2]]):
        with pytest.raises(ValueError,match='unearned'):raw._acknowledged_owned_stage_ids(incomplete,stages)
    changed=copy.deepcopy(stages);changed[0]['parameters']['epochs']=2
    with pytest.raises(ValueError,match='contract'):raw._acknowledged_owned_stage_ids(chain,changed)


def test_partial_commit_rejects_unknown_cursor_without_overwriting_it(owned,monkeypatch):
    x=owned;real=raw.write_bytes_durable;cursor=x['root']/'cycle.json'
    def fail_cursor(path,data):
        if Path(path)==cursor:raise OSError('simulated crash')
        return real(path,data)
    with monkeypatch.context()as m:
        m.setattr(raw,'write_bytes_durable',fail_cursor)
        with pytest.raises(OSError):invoke(x)
    value=json.loads(cursor.read_text());value['unrecognized_new_owner']=True;write(cursor,value)
    before={p:p.read_bytes()for p in(cursor,x['out']/'state.json')}
    with pytest.raises(ValueError,match='unknown successor'):invoke(x)
    assert all(p.read_bytes()==data for p,data in before.items())


def test_executable_owned_handoff_reuses_one_decode_and_binds_publication(owned,tmp_path,monkeypatch,capsys):
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    x=owned;manifest=tmp_path/'operations.json'
    write(manifest,{'schema':'natlang.native-owned-training-operations/1','operations':x['operations']})
    output=tmp_path/'owned-runtime.json';real=runtime.verify_handoff;calls=[]
    def counted(reference):
        calls.append(reference);return real(reference)
    monkeypatch.setattr(runtime,'verify_handoff',counted)
    assert runtime.main(['--bootstrap',x['bootstrap']['path'],'--handoff',x['handoff']['path'],
        '--bare-recipe',x['bare']['recipe']['path'],'--package',x['bare']['package'],
        '--ordinary-capacity',x['bare']['ordinary_capacity']['path'],'--config',str(x['config']),
        '--owned-operations',str(manifest),'--out',str(output)])==0
    assert calls==[x['handoff']]
    proof=json.loads(output.read_text());publication=json.loads(Path(proof['owned_operations']['path']).read_text())
    assert publication['published_checkpoint']==proof['checkpoint']
    assert publication['whole_recipe_completed']is False
    runtime.verify_runtime_handoff(resident.binding(output),bootstrap=x['bootstrap'])


def test_scheduler_cannot_dispatch_old_recipe_after_owned_publication(owned,monkeypatch):
    from types import SimpleNamespace
    x=owned;invoke(x)
    monkeypatch.setattr(raw,'configure_model',lambda *args:x['config'])
    with pytest.raises(raw.StudentConfigurationError,match='runtime configuration'):
        raw.student_cycle(x['plan'],x['declaration'],x['bootstrap'],x['root'],stop=SimpleNamespace(requested=False))


def test_owned_cycle_boundary_installs_next_actor_and_collector_before_fastpath(owned,tmp_path):
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    x=owned;config=json.loads(x['config'].read_text());config['actor_command']=['old-actor'];write(x['config'],config)
    published=invoke(x)
    proof=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],None,x['config'],
        tmp_path/'next-unsealed-runtime.json',bare_direct_owner=x['bare'],owned_operations=published)
    collector=tmp_path/'new-collector';(collector/'dist/teacher').mkdir(parents=True);(collector/'dist/teacher/cli.js').write_text('/* fixture */')
    declaration=copy.deepcopy(x['declaration']);declaration.update(runtime_handoff=proof,runtime=str(collector))
    declaration['commands']['actor_command']=['new-qualified-actor']
    write(x['plan'],{'models':[declaration]})
    raw.configure_model(x['plan'],declaration,x['bootstrap'],x['root'])
    actual=json.loads(x['config'].read_text())
    assert actual['actor_command']==['new-qualified-actor']and actual['runtime']==str(collector)
    assert actual['runtime_binding']['plan_sha256']==resident.binding(x['plan'])['sha256']
    raw.configure_model(x['plan'],declaration,x['bootstrap'],x['root'])
    assert json.loads(x['config'].read_text())['actor_command']==['new-qualified-actor']


@pytest.mark.parametrize('unfinished',['sft','round-zero','round-one','learner','unacknowledged','stale-receipt','pending-checkpoint'])
def test_incomplete_owned_work_retains_sealed_collection_protection(tmp_path,unfinished):
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    p=tmp_path/'publication.json';write(p,{'acknowledged_finite_stages':['sft'],'request_sha256':'request'})
    proof={'owned_operations':resident.binding(p)}
    cursor={'phase':'sft','next_stage':1,'owned_operation_reconciliation':{
        'receipt':str(p),'request_sha256':'request','runtime_configuration_required':True}}
    state={'phase':'collect'};recipe={'stages':[{'id':'sft','kind':'native_state_sft'}]}
    assert runtime._owned_cycle_boundary(proof,cursor,state,recipe,'sft')
    if unfinished=='sft':cursor['next_stage']=0
    elif unfinished.startswith('round-'):cursor['pending_rl_round']=0 if unfinished=='round-zero'else 1
    elif unfinished=='learner':state['phase']='learner_pending'
    elif unfinished=='pending-checkpoint':state['pending_native_checkpoint']='sealed.pt'
    elif unfinished=='stale-receipt':cursor['owned_operation_reconciliation']['request_sha256']='other'
    else:cursor['owned_operation_reconciliation']['runtime_configuration_required']=False
    assert not runtime._owned_cycle_boundary(proof,cursor,state,recipe,'sft')


def test_cli_rejects_missing_operation_controls_before_checkpoint_decode(owned,tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_runtime_handoff as runtime
    x=owned;manifest=tmp_path/'bad-operations.json'
    missing=copy.deepcopy(x['operations']);missing[0]['receipt']['path']=str(tmp_path/'absent-receipt.json')
    write(manifest,{'schema':'natlang.native-owned-training-operations/1','operations':missing})
    def forbidden(*args):raise AssertionError('must reject static controls before decoding a full parent')
    monkeypatch.setattr(runtime,'verify_handoff',forbidden)
    with pytest.raises(FileNotFoundError):
        runtime.main(['--bootstrap',x['bootstrap']['path'],'--handoff',x['handoff']['path'],
            '--bare-recipe',x['bare']['recipe']['path'],'--package',x['bare']['package'],
            '--ordinary-capacity',x['bare']['ordinary_capacity']['path'],'--config',str(x['config']),
            '--owned-operations',str(manifest),'--out',str(tmp_path/'not-written.json')])
